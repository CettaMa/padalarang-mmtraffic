import json
import logging
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from cctv_scraper.config import RuntimeConfig, ensure_dir
from cctv_scraper.storage_scan import trim_stderr


class DailyArchiver(threading.Thread):
    """Put each completed dataset day into an uncompressed 7-Zip archive.

    Store mode is intentional: raw transport-stream segments are handed straight
    to 7z with no intermediate re-encoding step -- the archiver's job is to
    package a day for transfer, not to shrink it. Archives are written to a
    temporary name, tested, and atomically renamed.
    """

    def __init__(self, config: RuntimeConfig, stop_event: threading.Event):
        super().__init__(name="daily-archiver", daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.logger = logging.getLogger("daily-archive")

    @property
    def archive_dir(self) -> Path:
        return self.config.storage.output_root / "archives"

    def run(self) -> None:
        if not self.config.archive.daily_archive_enabled:
            self.logger.info("Daily 7-Zip archiver disabled.")
            return

        self.logger.info("Daily 7-Zip archiver started.")
        while not self.stop_event.is_set():
            try:
                self.archive_completed_days()
            except Exception as exc:
                self.logger.exception("Daily 7-Zip archiver error: %s", exc)
            self.stop_event.wait(self.config.archive.daily_archive_scan_seconds)
        self.logger.info("Daily 7-Zip archiver stopped.")

    def archive_completed_days(self) -> None:
        root = self.config.storage.output_root
        today = datetime.now().date()
        if not root.exists():
            return
        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir() or date_dir.name == "archives":
                continue
            try:
                day = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day >= today:
                continue
            files = [path for path in date_dir.rglob("*") if path.is_file()]
            if not files:
                continue
            # The recorder may still be closing the last segment around midnight.
            # Wait until every source file is older than the normal safety window.
            try:
                newest_mtime = max(path.stat().st_mtime for path in files)
            except OSError:
                continue
            if time.time() < newest_mtime + self.config.archive.safe_age_seconds:
                continue
            self.archive_day(date_dir)

    def archive_day(self, date_dir: Path) -> None:
        ensure_dir(self.archive_dir)
        output = self.archive_dir / f"{date_dir.name}.7z"
        failure_marker = output.with_name(f".{output.name}.failure.json")

        if output.exists():
            if self.test_archive(output.resolve()):
                return
            self.logger.warning(
                "Existing daily archive failed verification; recreating: %s", output
            )
            output.unlink(missing_ok=True)

        temporary = output.with_name(f".{output.name}.tmp")
        temporary.unlink(missing_ok=True)
        command = [
            self.config.archive.archiver_binary,
            "a",
            "-t7z",
            "-mx=0",
            "-mmt=on",
            str(temporary.resolve()),
            date_dir.name,
        ]
        self.logger.info("Creating daily Store archive: %s -> %s", date_dir, output)
        try:
            result = subprocess.run(
                command,
                cwd=str(self.config.storage.output_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            self.record_failure(failure_marker, f"archiver could not run: {exc}")
            return

        if (
            result.returncode != 0
            or not temporary.exists()
            or not self.test_archive(temporary.resolve())
        ):
            detail = trim_stderr(result.stdout or "")
            self.record_failure(
                failure_marker,
                f"code={result.returncode} | {detail or 'archive missing or verification failed'}",
            )
            temporary.unlink(missing_ok=True)
            return

        temporary.replace(output)
        failure_marker.unlink(missing_ok=True)
        self.logger.info("Daily archive saved: %s | bytes=%s", output, output.stat().st_size)

        if self.config.archive.daily_archive_delete_source:
            shutil.rmtree(date_dir)
            self.logger.info("Deleted archived source directory: %s", date_dir)

    def test_archive(self, archive: Path) -> bool:
        try:
            result = subprocess.run(
                [self.config.archive.archiver_binary, "t", str(archive)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            self.logger.error("Cannot verify 7-Zip archive %s: %s", archive, exc)
            return False
        return result.returncode == 0

    def record_failure(self, marker: Path, reason: str) -> None:
        try:
            marker.write_text(
                json.dumps(
                    {"status": "failed", "failed_at": time.time(), "reason": reason}, indent=2
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            self.logger.error("Cannot write daily archive failure marker %s: %s", marker, exc)
        self.logger.error("Daily archive failed: %s", reason)

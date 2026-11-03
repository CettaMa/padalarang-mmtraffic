import logging
import os
import shutil
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from cctv_scraper.config import RuntimeConfig, ensure_dir, now_local
from cctv_scraper.storage_scan import parse_date_dir_name


class DiskMonitor(threading.Thread):
    def __init__(self, config: RuntimeConfig, stop_event: threading.Event):
        super().__init__(name="disk-monitor", daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.logger = logging.getLogger("disk")

    def run(self) -> None:
        self.logger.info("Disk monitor started.")

        while not self.stop_event.is_set():
            try:
                self.check_disk()
                self.cleanup_old_folders()
            except Exception as exc:
                self.logger.exception("Disk monitor error: %s", exc)

            self.stop_event.wait(self.config.storage.disk_check_seconds)

        self.logger.info("Disk monitor stopped.")

    def check_disk(self) -> None:
        ensure_dir(self.config.storage.output_root)
        usage = shutil.disk_usage(self.config.storage.output_root)
        free_gb = usage.free / (1024**3)

        if free_gb < self.config.storage.min_free_space_gb:
            self.logger.warning(
                "Low disk space: %.2f GB free. Minimum configured: %.2f GB.",
                free_gb,
                self.config.storage.min_free_space_gb,
            )
            self.emergency_purge(usage)
        else:
            self.logger.info("Disk free space: %.2f GB.", free_gb)

    def _requires_verified_archive(self) -> bool:
        """Whether a date directory must have a completed daily 7z before it can be purged.

        Only meaningful while DailyArchiver is actually running -- if daily archiving is
        disabled, no ``archives/*.7z`` will ever appear and gating on it would just let
        the disk fill up forever.
        """
        return self.config.archive.daily_archive_enabled

    def _is_day_archived(self, day_name: str) -> bool:
        return (self.config.storage.output_root / "archives" / f"{day_name}.7z").exists()

    def emergency_purge(self, initial_usage: Any = None) -> None:
        """Delete oldest complete date directories until the disk is safe again.

        Days that DailyArchiver has already packaged into ``archives/*.7z`` are reclaimed
        first, since that footage survives the deletion. Only if that is not enough to
        clear the low-space condition does this fall back to days not yet archived --
        a full disk halts every camera at once, which is worse than losing one
        unpackaged day, but it is logged as an error because that footage is now gone
        for good.
        """
        usage = initial_usage or shutil.disk_usage(self.config.storage.output_root)
        today = now_local().date()
        require_archive = self._requires_verified_archive()
        candidates: list[tuple[date, Path]] = []

        try:
            with os.scandir(self.config.storage.output_root) as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    folder_date = parse_date_dir_name(entry.name)
                    if folder_date is not None and folder_date < today:
                        candidates.append((folder_date, Path(entry.path)))
        except OSError as exc:
            self.logger.warning("Error during emergency purge scan: %s", exc)
            return

        archived = {
            target: (not require_archive) or self._is_day_archived(target.name)
            for _, target in candidates
        }
        candidates.sort(key=lambda item: (not archived[item[1]], item[0]))

        for _, target in candidates:
            if usage.free / (1024**3) >= self.config.storage.min_free_space_gb:
                return

            if archived[target]:
                self.logger.warning("Emergency low-disk purge deleting footage folder: %s", target)
            else:
                self.logger.error(
                    "Emergency low-disk purge deleting UNARCHIVED footage folder %s -- "
                    "DailyArchiver has not packaged this day yet, so this footage is now "
                    "unrecoverable. Freeing disk to keep the recorder alive took priority.",
                    target,
                )
            try:
                shutil.rmtree(target)
            except OSError as exc:
                self.logger.warning("Emergency purge failed for %s: %s", target, exc)
            usage = shutil.disk_usage(self.config.storage.output_root)

        if usage.free / (1024**3) < self.config.storage.min_free_space_gb:
            self.logger.warning(
                "Emergency purge could not reach configured free space; %.2f GB remains.",
                usage.free / (1024**3),
            )

    def cleanup_old_folders(self) -> None:
        """Prune date directories past the retention window.

        A directory is only deleted once DailyArchiver has produced a verified
        ``archives/<date>.7z`` for it (when daily archiving is enabled), so an encoding
        or archiving backlog extends retention rather than silently losing footage.
        ``emergency_purge`` is the deliberate escape hatch for when the disk cannot wait.
        """
        cutoff_date = now_local().date() - timedelta(days=self.config.storage.retention_days)

        if not self.config.storage.output_root.exists():
            return

        require_archive = self._requires_verified_archive()

        try:
            with os.scandir(self.config.storage.output_root) as it:
                for entry in it:
                    if not entry.is_dir() or entry.name in {"logs", "status"}:
                        continue

                    folder_date = parse_date_dir_name(entry.name)
                    if folder_date is None:
                        continue

                    if folder_date >= cutoff_date:
                        continue

                    target = Path(entry.path)
                    if require_archive and not self._is_day_archived(entry.name):
                        self.logger.warning(
                            "Retention cutoff reached for %s but no verified daily archive "
                            "exists yet; keeping it until DailyArchiver catches up.",
                            target,
                        )
                        continue

                    self.logger.warning("Deleting old footage folder: %s", target)
                    shutil.rmtree(target, ignore_errors=True)
        except OSError as exc:
            self.logger.warning("Error during disk cleanup scan: %s", exc)

        self.cleanup_old_files(self.config.storage.output_root / "logs")
        self.cleanup_old_files(self.config.storage.output_root / "status")

    def cleanup_old_files(self, directory: Path) -> None:
        """Prune aged logs and status files without following links."""
        if not directory.is_dir():
            return

        cutoff_ts = time.time() - self.config.storage.retention_days * 24 * 60 * 60
        for root, _, filenames in os.walk(directory, followlinks=False):
            for filename in filenames:
                path = Path(root) / filename
                try:
                    if path.stat().st_mtime < cutoff_ts:
                        self.logger.warning("Deleting old log/status file: %s", path)
                        path.unlink()
                except OSError as exc:
                    self.logger.warning("Failed deleting old log/status file %s: %s", path, exc)

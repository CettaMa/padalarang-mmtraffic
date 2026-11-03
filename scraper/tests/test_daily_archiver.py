import json
import os
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path

from cctv_scraper.archive import DailyArchiver
from cctv_scraper.config import now_local
from cctv_scraper.disk import DiskMonitor
from tests.test_characterization import make_dummy_config


def _write_segment(date_dir: Path, camera: str, mtime: float) -> Path:
    videos = date_dir / camera / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    segment = videos / f"{camera}_segment.ts"
    segment.write_bytes(b"raw")
    os.utime(segment, (mtime, mtime))
    return segment


def failure_marker_for(output: Path) -> Path:
    return output.with_name(f".{output.name}.failure.json")


# ============================================================================
# archive_completed_days: which dates are even considered
# ============================================================================


def test_archive_completed_days_skips_today(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path)
    archiver = DailyArchiver(cfg, threading.Event())
    today_str = now_local().date().strftime("%Y-%m-%d")
    _write_segment(tmp_path / today_str, "cam", time.time())

    called: list[str] = []
    monkeypatch.setattr(archiver, "archive_day", lambda date_dir: called.append(date_dir.name))
    archiver.archive_completed_days()

    assert called == []


def test_archive_completed_days_waits_for_the_safety_window(tmp_path, monkeypatch):
    """A day is only handed to archive_day once every source file is older than
    safe_age_seconds -- the recorder may still be closing the last segment."""
    cfg = make_dummy_config(tmp_path, archive_safe_age_seconds=90)
    archiver = DailyArchiver(cfg, threading.Event())
    yesterday_str = (now_local().date() - timedelta(days=1)).strftime("%Y-%m-%d")
    _write_segment(tmp_path / yesterday_str, "cam", 1000.0)

    called: list[str] = []
    monkeypatch.setattr(archiver, "archive_day", lambda date_dir: called.append(date_dir.name))

    monkeypatch.setattr("cctv_scraper.archive.time.time", lambda: 1000.0 + 50)
    archiver.archive_completed_days()
    assert called == []

    monkeypatch.setattr("cctv_scraper.archive.time.time", lambda: 1000.0 + 91)
    archiver.archive_completed_days()
    assert called == [yesterday_str]


def test_archive_completed_days_skips_directories_with_no_files(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path)
    archiver = DailyArchiver(cfg, threading.Event())
    yesterday_str = (now_local().date() - timedelta(days=1)).strftime("%Y-%m-%d")
    (tmp_path / yesterday_str / "cam" / "videos").mkdir(parents=True)

    called: list[str] = []
    monkeypatch.setattr(archiver, "archive_day", lambda date_dir: called.append(date_dir.name))
    archiver.archive_completed_days()

    assert called == []


def test_daily_archiver_run_is_a_noop_when_disabled(tmp_path):
    cfg = make_dummy_config(tmp_path, daily_archive_enabled=False)
    archiver = DailyArchiver(cfg, threading.Event())
    archiver.run()  # must return immediately without entering the scan loop


# ============================================================================
# archive_day: the actual 7z create/verify/replace flow (7z subprocess mocked)
# ============================================================================


def test_daily_archiver_creates_and_verifies_a_new_archive(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path)
    archiver = DailyArchiver(cfg, threading.Event())
    date_dir = tmp_path / "2026-06-21"
    (date_dir / "cam" / "videos").mkdir(parents=True)
    (date_dir / "cam" / "videos" / "f.ts").write_bytes(b"raw")

    def fake_run(command, **kwargs):
        if command[1] == "a":
            Path(command[5]).write_bytes(b"7z-data")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1] == "t":
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected 7z invocation: {command}")

    monkeypatch.setattr("cctv_scraper.archive.subprocess.run", fake_run)
    archiver.archive_day(date_dir)

    output = tmp_path / "archives" / "2026-06-21.7z"
    assert output.read_bytes() == b"7z-data"
    assert not failure_marker_for(output).exists()
    # Source is retained unless DAILY_ARCHIVE_DELETE_SOURCE is set.
    assert date_dir.exists()


def test_daily_archiver_records_failure_when_verification_fails(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path)
    archiver = DailyArchiver(cfg, threading.Event())
    date_dir = tmp_path / "2026-06-21"
    (date_dir / "cam" / "videos").mkdir(parents=True)
    (date_dir / "cam" / "videos" / "f.ts").write_bytes(b"raw")

    def fake_run(command, **kwargs):
        if command[1] == "a":
            Path(command[5]).write_bytes(b"broken")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1] == "t":
            return subprocess.CompletedProcess(command, 2, "CRC failed", "")
        raise AssertionError(command)

    monkeypatch.setattr("cctv_scraper.archive.subprocess.run", fake_run)
    archiver.archive_day(date_dir)

    output = tmp_path / "archives" / "2026-06-21.7z"
    marker = failure_marker_for(output)
    assert not output.exists()
    assert marker.exists()
    state = json.loads(marker.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    # The raw footage must never be touched when the archive cannot be verified.
    assert (date_dir / "cam" / "videos" / "f.ts").exists()


def test_daily_archiver_could_not_run_7z_records_failure(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path)
    archiver = DailyArchiver(cfg, threading.Event())
    date_dir = tmp_path / "2026-06-21"
    (date_dir / "cam" / "videos").mkdir(parents=True)
    (date_dir / "cam" / "videos" / "f.ts").write_bytes(b"raw")

    def fake_run(command, **kwargs):
        raise FileNotFoundError("7z binary not found")

    monkeypatch.setattr("cctv_scraper.archive.subprocess.run", fake_run)
    archiver.archive_day(date_dir)

    output = tmp_path / "archives" / "2026-06-21.7z"
    marker = failure_marker_for(output)
    assert not output.exists()
    assert marker.exists()
    assert "archiver could not run" in json.loads(marker.read_text(encoding="utf-8"))["reason"]


def test_daily_archiver_recreates_a_previously_broken_archive(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path)
    archiver = DailyArchiver(cfg, threading.Event())
    date_dir = tmp_path / "2026-06-21"
    (date_dir / "cam" / "videos").mkdir(parents=True)
    (date_dir / "cam" / "videos" / "f.ts").write_bytes(b"raw")

    archives_dir = tmp_path / "archives"
    archives_dir.mkdir()
    existing = archives_dir / "2026-06-21.7z"
    existing.write_bytes(b"corrupt-old-archive")

    call_log: list[str] = []

    def fake_run(command, **kwargs):
        call_log.append(command[1])
        if command[1] == "t":
            # First verification targets the existing broken archive and must fail;
            # the second targets the freshly created one and must succeed.
            return subprocess.CompletedProcess(
                command, 1 if call_log.count("t") == 1 else 0, "", ""
            )
        if command[1] == "a":
            Path(command[5]).write_bytes(b"fresh-archive")
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    monkeypatch.setattr("cctv_scraper.archive.subprocess.run", fake_run)
    archiver.archive_day(date_dir)

    assert call_log[0] == "t"  # the stale archive is checked before anything is rebuilt
    assert existing.read_bytes() == b"fresh-archive"


def test_daily_archiver_leaves_a_still_verified_archive_alone(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path)
    archiver = DailyArchiver(cfg, threading.Event())
    date_dir = tmp_path / "2026-06-21"
    (date_dir / "cam" / "videos").mkdir(parents=True)
    (date_dir / "cam" / "videos" / "f.ts").write_bytes(b"raw")

    archives_dir = tmp_path / "archives"
    archives_dir.mkdir()
    existing = archives_dir / "2026-06-21.7z"
    existing.write_bytes(b"already-good-archive")

    def fake_run(command, **kwargs):
        assert command[1] == "t", "a still-valid archive must not be recreated"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cctv_scraper.archive.subprocess.run", fake_run)
    archiver.archive_day(date_dir)

    assert existing.read_bytes() == b"already-good-archive"


def test_daily_archiver_deletes_source_only_when_configured(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path, daily_archive_delete_source=True)
    archiver = DailyArchiver(cfg, threading.Event())
    date_dir = tmp_path / "2026-06-21"
    (date_dir / "cam" / "videos").mkdir(parents=True)
    (date_dir / "cam" / "videos" / "f.ts").write_bytes(b"raw")

    def fake_run(command, **kwargs):
        if command[1] == "a":
            Path(command[5]).write_bytes(b"archive")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cctv_scraper.archive.subprocess.run", fake_run)
    archiver.archive_day(date_dir)

    assert (tmp_path / "archives" / "2026-06-21.7z").exists()
    assert not date_dir.exists()


# ============================================================================
# Integration: DiskMonitor only reclaims a day once DailyArchiver has packaged it
# ============================================================================


def test_disk_monitor_waits_for_daily_archiver_before_pruning(tmp_path, monkeypatch):
    cfg = make_dummy_config(tmp_path, retention_days=7)
    old_day = now_local().date() - timedelta(days=10)
    old_day_str = old_day.strftime("%Y-%m-%d")
    date_dir = tmp_path / old_day_str
    (date_dir / "cam" / "videos").mkdir(parents=True)
    (date_dir / "cam" / "videos" / "f.ts").write_bytes(b"raw")

    monitor = DiskMonitor(cfg, threading.Event())
    monitor.cleanup_old_folders()
    assert date_dir.exists(), "retention must wait for a verified archive"

    archiver = DailyArchiver(cfg, threading.Event())

    def fake_run(command, **kwargs):
        if command[1] == "a":
            Path(command[5]).write_bytes(b"archive")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cctv_scraper.archive.subprocess.run", fake_run)
    archiver.archive_day(date_dir)
    assert (tmp_path / "archives" / f"{old_day_str}.7z").exists()

    monitor.cleanup_old_folders()
    assert not date_dir.exists(), "now that it is archived, retention may reclaim it"

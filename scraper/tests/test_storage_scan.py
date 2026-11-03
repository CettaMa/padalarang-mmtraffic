import logging
import os
import threading
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cctv_scraper.config import CCTVPoint, now_local
from cctv_scraper.disk import DiskMonitor
from cctv_scraper.logging_setup import point_logger, setup_logging
from cctv_scraper.recorder import CCTVRecorder
from cctv_scraper.storage_scan import trim_stderr
from tests.test_characterization import make_dummy_config


def test_trim_stderr():
    stderr_text = "\n".join([f"line {i}" for i in range(20)])
    trimmed = trim_stderr(stderr_text, max_lines=5)
    assert trimmed == "line 15 | line 16 | line 17 | line 18 | line 19"
    assert trim_stderr("") == ""


def test_disk_emergency_purge_deletes_oldest_and_keeps_today(tmp_path: Path, monkeypatch):
    point = CCTVPoint(name="point_a", url="http://example.com/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(tmp_path, min_free_space_gb=10.0)
    monitor = DiskMonitor(cfg, threading.Event())
    today = now_local().date()
    old_dates = [today - timedelta(days=3), today - timedelta(days=2)]
    for folder_date in [*old_dates, today]:
        (tmp_path / folder_date.strftime("%Y-%m-%d") / point.name).mkdir(parents=True)

    usage = [
        SimpleNamespace(free=1 * 1024**3),
        SimpleNamespace(free=11 * 1024**3),
    ]
    monkeypatch.setattr("cctv_scraper.disk.shutil.disk_usage", lambda _: usage.pop(0))
    monitor.check_disk()

    assert not (tmp_path / old_dates[0].strftime("%Y-%m-%d")).exists()
    assert (tmp_path / old_dates[1].strftime("%Y-%m-%d")).exists()
    assert (tmp_path / today.strftime("%Y-%m-%d")).exists()


def test_disk_emergency_purge_prefers_archived_days_over_unarchived(tmp_path: Path, monkeypatch):
    """A day DailyArchiver has not packaged yet must survive as long as an already-
    archived day can be reclaimed instead -- deleting it would lose data forever."""
    point = CCTVPoint(name="point_a", url="http://example.com/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(tmp_path, min_free_space_gb=10.0)
    monitor = DiskMonitor(cfg, threading.Event())
    today = now_local().date()
    archived_day = today - timedelta(days=3)
    unarchived_day = today - timedelta(days=2)
    for folder_date in [archived_day, unarchived_day, today]:
        (tmp_path / folder_date.strftime("%Y-%m-%d") / point.name).mkdir(parents=True)
    (tmp_path / "archives").mkdir()
    (tmp_path / "archives" / f"{archived_day.strftime('%Y-%m-%d')}.7z").write_bytes(b"data")

    usage = [SimpleNamespace(free=1 * 1024**3), SimpleNamespace(free=11 * 1024**3)]
    monkeypatch.setattr("cctv_scraper.disk.shutil.disk_usage", lambda _: usage.pop(0))
    monitor.check_disk()

    # The older but already-archived day is reclaimed first, even though the
    # unarchived day is chronologically newer -- archival status wins over age.
    assert not (tmp_path / archived_day.strftime("%Y-%m-%d")).exists()
    assert (tmp_path / unarchived_day.strftime("%Y-%m-%d")).exists()
    assert (tmp_path / today.strftime("%Y-%m-%d")).exists()


def test_disk_emergency_purge_falls_back_to_unarchived_when_no_archive_is_enough(
    tmp_path: Path, monkeypatch
):
    """When reclaiming every archived day still leaves the disk critically low, the
    emergency purge must fall back to unarchived days rather than let the disk fill up
    and stop every camera at once."""
    point = CCTVPoint(name="point_a", url="http://example.com/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(tmp_path, min_free_space_gb=10.0)
    monitor = DiskMonitor(cfg, threading.Event())
    today = now_local().date()
    unarchived_day = today - timedelta(days=2)
    (tmp_path / unarchived_day.strftime("%Y-%m-%d") / point.name).mkdir(parents=True)
    (tmp_path / today.strftime("%Y-%m-%d") / point.name).mkdir(parents=True)

    usage = [SimpleNamespace(free=1 * 1024**3), SimpleNamespace(free=11 * 1024**3)]
    monkeypatch.setattr("cctv_scraper.disk.shutil.disk_usage", lambda _: usage.pop(0))
    monitor.check_disk()

    assert not (tmp_path / unarchived_day.strftime("%Y-%m-%d")).exists()
    assert (tmp_path / today.strftime("%Y-%m-%d")).exists()


def test_cleanup_old_folders_keeps_unarchived_days_past_retention(tmp_path: Path):
    """Routine retention pruning must wait for DailyArchiver, not just for age --
    otherwise an archiving backlog silently loses footage instead of just delaying it."""
    point = CCTVPoint(name="point_a", url="http://example.com/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(tmp_path, retention_days=7)
    monitor = DiskMonitor(cfg, threading.Event())
    today = now_local().date()
    old_archived = today - timedelta(days=10)
    old_unarchived = today - timedelta(days=9)
    for folder_date in [old_archived, old_unarchived]:
        (tmp_path / folder_date.strftime("%Y-%m-%d") / point.name).mkdir(parents=True)
    (tmp_path / "archives").mkdir()
    (tmp_path / "archives" / f"{old_archived.strftime('%Y-%m-%d')}.7z").write_bytes(b"data")

    monitor.cleanup_old_folders()

    assert not (tmp_path / old_archived.strftime("%Y-%m-%d")).exists()
    assert (tmp_path / old_unarchived.strftime("%Y-%m-%d")).exists()


def test_cleanup_old_folders_ignores_archive_state_when_daily_archiving_disabled(
    tmp_path: Path,
):
    """With daily archiving turned off, no archives/*.7z will ever exist, so retention
    must fall back to age-only pruning instead of keeping data forever."""
    point = CCTVPoint(name="point_a", url="http://example.com/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(tmp_path, retention_days=7, daily_archive_enabled=False)
    monitor = DiskMonitor(cfg, threading.Event())
    old_day = now_local().date() - timedelta(days=10)
    (tmp_path / old_day.strftime("%Y-%m-%d") / point.name).mkdir(parents=True)

    monitor.cleanup_old_folders()

    assert not (tmp_path / old_day.strftime("%Y-%m-%d")).exists()


def test_cleanup_prunes_old_logs_and_status_files(tmp_path: Path):
    cfg = make_dummy_config(tmp_path, retention_days=7)
    monitor = DiskMonitor(cfg, threading.Event())
    old_timestamp = time.time() - 8 * 24 * 60 * 60
    old_log = tmp_path / "logs" / "ffmpeg" / "point_2020.ffmpeg.log"
    new_log = tmp_path / "logs" / "scraper.log"
    old_status = tmp_path / "status" / "point_status.csv"
    for path in [old_log, new_log, old_status]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    os.utime(old_log, (old_timestamp, old_timestamp))
    os.utime(old_status, (old_timestamp, old_timestamp))

    monitor.cleanup_old_folders()

    assert not old_log.exists()
    assert not old_status.exists()
    assert new_log.exists()


def test_logging_uses_configured_rotating_handlers(tmp_path: Path):
    setup_logging(tmp_path, max_bytes=1234, backup_count=2)
    root = logging.getLogger()
    root_file_handlers = [
        handler for handler in root.handlers if isinstance(handler, RotatingFileHandler)
    ]
    assert len(root_file_handlers) == 1
    assert root_file_handlers[0].maxBytes == 1234
    assert root_file_handlers[0].backupCount == 2

    logger = point_logger(tmp_path, "rotation_point", max_bytes=5678, backup_count=3)
    point_handlers = [
        handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)
    ]
    assert len(point_handlers) == 1
    assert point_handlers[0].maxBytes == 5678
    assert point_handlers[0].backupCount == 3

    for handler in [*root.handlers, *logger.handlers]:
        handler.close()


def test_recorder_escalates_repeated_expired_url_failures(tmp_path: Path, caplog):
    point = CCTVPoint(
        name="expired_status_point",
        url="http://example.com/live.m3u8",
        lat=-6.85,
        lon=107.5,
    )
    cfg = make_dummy_config(tmp_path, expired_url_escalation_threshold=3)
    recorder = CCTVRecorder(point, cfg, threading.Event())
    caplog.set_level(logging.ERROR, logger=recorder.logger.name)

    for _ in range(4):
        recorder.handle_preflight_failure("not_found_or_expired_url", "404")

    status_text = recorder.status_file().read_text(encoding="utf-8")
    assert status_text.count("EXPIRED_URL_ESCALATION") == 1
    assert "expired_url_escalated" in status_text
    assert "[EXPIRED_URL_ESCALATION]" in caplog.text


def test_recorder_single_pass_latest_video(tmp_path: Path):
    point = CCTVPoint(name="point_test", url="http://example.com/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(tmp_path, stale_file_seconds=120)
    stop_event = threading.Event()
    recorder = CCTVRecorder(point, cfg, stop_event)

    video_dir = recorder.current_video_dir()
    v1 = video_dir / "point_test_20260621_100000.ts"
    v2 = video_dir / "point_test_20260621_100100.ts"
    other_cam = video_dir / "other_cam_20260621_100200.ts"

    v1.write_bytes(b"x" * 200000)
    v2.write_bytes(b"y" * 200000)
    other_cam.write_bytes(b"z" * 200000)

    now = 500000.0
    os.utime(v1, (now - 200, now - 200))
    os.utime(v2, (now - 50, now - 50))
    os.utime(other_cam, (now - 10, now - 10))

    recorder.last_restart_at = datetime.fromtimestamp(now - 300)

    with patch("time.time", return_value=now):
        latest = recorder.latest_video_file()
        assert latest == v2
        assert recorder.is_video_stale() is False

    # Simulate stale: now is 500300 (250s since v2 was modified > 120s stale_file_seconds)
    with patch("time.time", return_value=now + 200):
        assert recorder.is_video_stale() is True

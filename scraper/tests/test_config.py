import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cctv_scraper.config import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_SEGMENT_SECONDS,
    ArchiveConfig,
    MetadataConfig,
    NetworkConfig,
    RecorderConfig,
    RuntimeConfig,
    StorageConfig,
    load_runtime_config,
)


def test_load_runtime_config_defaults(tmp_path: Path):
    args = argparse.Namespace(
        config=None,
        output=None,
        segment_seconds=None,
        retention_days=None,
        min_free_space_gb=None,
        video_container=None,
    )
    with patch("cctv_scraper.config.load_dotenv"), patch.dict(os.environ, {}, clear=True):
        cfg = load_runtime_config(args)

    assert isinstance(cfg, RuntimeConfig)
    assert isinstance(cfg.recorder, RecorderConfig)
    assert isinstance(cfg.metadata, MetadataConfig)
    assert isinstance(cfg.archive, ArchiveConfig)
    assert isinstance(cfg.storage, StorageConfig)
    assert isinstance(cfg.network, NetworkConfig)

    assert cfg.config_file == Path(DEFAULT_CONFIG_FILE)
    assert cfg.storage.output_root == Path(DEFAULT_OUTPUT_ROOT)
    assert cfg.recorder.segment_seconds == DEFAULT_SEGMENT_SECONDS
    assert cfg.storage.retention_days == DEFAULT_RETENTION_DAYS
    assert cfg.recorder.video_container == "ts"
    assert cfg.recorder.ffmpeg_transport_mode == "copy"
    assert cfg.recorder.video_encoder == "hevc_nvenc"
    assert cfg.archive.daily_archive_enabled is True


def test_load_runtime_config_env_overrides(tmp_path: Path):
    args = argparse.Namespace(
        config=None,
        output=None,
        segment_seconds=None,
        retention_days=None,
        min_free_space_gb=None,
        video_container=None,
    )
    env_vars = {
        "SEGMENT_SECONDS": "120",
        "VIDEO_CONTAINER": "mp4",
        "FFMPEG_TRANSPORT_MODE": "transcode",
        "VIDEO_ENCODER": "libx264",
        "OUTPUT_FPS": "25",
        "ARCHIVE_SAFE_AGE_SECONDS": "45",
        "DAILY_ARCHIVE_ENABLED": "false",
        "DAILY_ARCHIVE_SCAN_SECONDS": "600",
        "DAILY_ARCHIVE_DELETE_SOURCE": "true",
        "ARCHIVER_BINARY": "7za",
        "RETENTION_DAYS": "14",
        "MIN_FREE_SPACE_GB": "50.5",
        "METADATA_MIN_PASS_INTERVAL_SECONDS": "7",
        "METADATA_FAILURE_BACKOFF_BASE_SECONDS": "8",
        "METADATA_FAILURE_BACKOFF_MAX_SECONDS": "90",
        "LOG_MAX_BYTES": "12345",
        "LOG_BACKUP_COUNT": "2",
        "EXPIRED_URL_ESCALATION_THRESHOLD": "4",
        "DEFAULT_LAT": "-6.123",
        "DEFAULT_LON": "107.456",
    }
    with patch("cctv_scraper.config.load_dotenv"), patch.dict(os.environ, env_vars, clear=True):
        cfg = load_runtime_config(args)

    assert cfg.recorder.segment_seconds == 120
    assert cfg.recorder.video_container == "mp4"
    assert cfg.recorder.ffmpeg_transport_mode == "transcode"
    assert cfg.recorder.video_encoder == "libx264"
    assert cfg.recorder.output_fps == 25
    assert cfg.archive.safe_age_seconds == 45
    assert cfg.archive.daily_archive_enabled is False
    assert cfg.archive.daily_archive_scan_seconds == 600
    assert cfg.archive.daily_archive_delete_source is True
    assert cfg.archive.archiver_binary == "7za"
    assert cfg.storage.retention_days == 14
    assert cfg.storage.min_free_space_gb == pytest.approx(50.5)
    assert cfg.metadata.min_pass_interval_seconds == 7
    assert cfg.metadata.failure_backoff_base_seconds == 8
    assert cfg.metadata.failure_backoff_max_seconds == 90
    assert cfg.storage.log_max_bytes == 12345
    assert cfg.storage.log_backup_count == 2
    assert cfg.network.expired_url_escalation_threshold == 4
    assert cfg.metadata.default_lat == pytest.approx(-6.123)
    assert cfg.metadata.default_lon == pytest.approx(107.456)


def test_load_runtime_config_cli_overrides_env(tmp_path: Path):
    args = argparse.Namespace(
        config="custom_points.csv",
        output="custom_dataset",
        segment_seconds=45,
        retention_days=3,
        min_free_space_gb=10.0,
        video_container="mp4",
    )
    env_vars = {
        "CCTV_CONFIG_FILE": "env_points.csv",
        "CCTV_OUTPUT_ROOT": "env_dataset",
        "SEGMENT_SECONDS": "90",
        "RETENTION_DAYS": "30",
        "MIN_FREE_SPACE_GB": "100.0",
        "VIDEO_CONTAINER": "ts",
    }
    with patch("cctv_scraper.config.load_dotenv"), patch.dict(os.environ, env_vars, clear=True):
        cfg = load_runtime_config(args)

    assert cfg.config_file == Path("custom_points.csv")
    assert cfg.storage.output_root == Path("custom_dataset")
    assert cfg.recorder.segment_seconds == 45
    assert cfg.storage.retention_days == 3
    assert cfg.storage.min_free_space_gb == pytest.approx(10.0)
    assert cfg.recorder.video_container == "mp4"


def test_load_runtime_config_validation_errors():
    args = argparse.Namespace(
        config=None,
        output=None,
        segment_seconds=None,
        retention_days=None,
        min_free_space_gb=None,
        video_container=None,
    )

    # Invalid container
    with (
        patch("cctv_scraper.config.load_dotenv"),
        patch.dict(os.environ, {"VIDEO_CONTAINER": "avi"}, clear=True),
    ):
        with pytest.raises(ValueError, match="video_container"):
            load_runtime_config(args)

    # Invalid transport mode
    with (
        patch("cctv_scraper.config.load_dotenv"),
        patch.dict(os.environ, {"FFMPEG_TRANSPORT_MODE": "invalid_mode"}, clear=True),
    ):
        with pytest.raises(ValueError, match="FFMPEG_TRANSPORT_MODE"):
            load_runtime_config(args)

    # Invalid positive number
    with (
        patch("cctv_scraper.config.load_dotenv"),
        patch.dict(os.environ, {"SEGMENT_SECONDS": "-5"}, clear=True),
    ):
        with pytest.raises(ValueError, match="lebih besar dari 0"):
            load_runtime_config(args)

    # Invalid integer
    with (
        patch("cctv_scraper.config.load_dotenv"),
        patch.dict(os.environ, {"RETENTION_DAYS": "not_an_int"}, clear=True),
    ):
        with pytest.raises(ValueError, match="harus berupa integer"):
            load_runtime_config(args)

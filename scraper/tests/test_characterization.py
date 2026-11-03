import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from cctv_scraper import (
    ArchiveConfig,
    CCTVPoint,
    CCTVRecorder,
    MetadataConfig,
    NetworkConfig,
    RecorderConfig,
    RuntimeConfig,
    StorageConfig,
    load_cctv_points,
    parse_coordinate,
    validate_video_encoder,
)


def subprocess_completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["ffmpeg"], 0, stdout, "")


def make_dummy_config(output_root: Path, **kwargs: Any) -> RuntimeConfig:
    recorder_defaults: dict[str, Any] = dict(
        segment_seconds=60,
        video_container="ts",
        restart_delay_seconds=5,
        health_check_seconds=30,
        stale_file_seconds=240,
        ffmpeg_loglevel="warning",
        ffmpeg_transport_mode="copy",
        ffmpeg_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        ffmpeg_referer="",
        ffmpeg_origin="",
        ffmpeg_analyzeduration="10000000",
        ffmpeg_probesize="10000000",
        hls_reconnect_at_eof=False,
        segment_atclocktime=False,
        hls_live_start_index="-1",
        ffmpeg_rw_timeout="10000000",
        ffmpeg_reconnect_delay_max="10",
        ffmpeg_reconnect_on_http_error="5xx",
        output_fps=10,
        transcode_preset="p4",
        segment_keyframe_seconds=2,
        video_encoder="hevc_nvenc",
        target_bitrate="650k",
        max_bitrate="900k",
        buffer_size="1300k",
        output_height=0,
        ffmpeg_http_persistent=True,
        ffmpeg_multiple_requests=True,
    )
    metadata_defaults: dict[str, Any] = dict(
        metadata_interval_seconds=60,
        tomtom_interval_seconds=300,
        openmeteo_interval_seconds=60,
        min_pass_interval_seconds=5,
        failure_backoff_base_seconds=5,
        failure_backoff_max_seconds=300,
        default_lat=-6.851117,
        default_lon=107.496586,
        tomtom_api_key=None,
    )
    archive_defaults: dict[str, Any] = dict(
        safe_age_seconds=90,
        daily_archive_enabled=True,
        daily_archive_scan_seconds=300,
        daily_archive_delete_source=False,
        archiver_binary="7z",
    )
    storage_defaults: dict[str, Any] = dict(
        output_root=output_root,
        retention_days=7,
        min_free_space_gb=20.0,
        disk_check_seconds=300,
        log_max_bytes=10 * 1024 * 1024,
        log_backup_count=5,
    )
    network_defaults: dict[str, Any] = dict(
        preflight_check=True,
        offline_retry_seconds=300,
        network_retry_seconds=60,
        expired_url_escalation_threshold=3,
        network_watchdog_seconds=600,
    )

    config_file = kwargs.pop("config_file", output_root / "cctv_points.csv")

    for k, v in kwargs.items():
        if k in recorder_defaults:
            recorder_defaults[k] = v
        elif k in metadata_defaults:
            metadata_defaults[k] = v
        elif k in archive_defaults:
            archive_defaults[k] = v
        elif k == "archive_safe_age_seconds":
            archive_defaults["safe_age_seconds"] = v
        elif k == "metadata_min_pass_interval_seconds":
            metadata_defaults["min_pass_interval_seconds"] = v
        elif k == "metadata_failure_backoff_base_seconds":
            metadata_defaults["failure_backoff_base_seconds"] = v
        elif k == "metadata_failure_backoff_max_seconds":
            metadata_defaults["failure_backoff_max_seconds"] = v
        elif k == "log_max_bytes":
            storage_defaults["log_max_bytes"] = v
        elif k == "log_backup_count":
            storage_defaults["log_backup_count"] = v
        elif k == "expired_url_escalation_threshold":
            network_defaults["expired_url_escalation_threshold"] = v
        elif k in storage_defaults:
            storage_defaults[k] = v
        elif k in network_defaults:
            network_defaults[k] = v

    return RuntimeConfig(
        config_file=config_file,
        recorder=RecorderConfig(**recorder_defaults),
        metadata=MetadataConfig(**metadata_defaults),
        archive=ArchiveConfig(**archive_defaults),
        storage=StorageConfig(**storage_defaults),
        network=NetworkConfig(**network_defaults),
    )


# ============================================================================
# 1. load_cctv_points & parse_coordinate tests
# ============================================================================
def test_parse_coordinate() -> None:
    fallback = -6.851117
    # Plain numbers
    assert parse_coordinate("-6.851117", fallback) == pytest.approx(-6.851117)
    assert parse_coordinate(107.496586, fallback) == pytest.approx(107.496586)
    # Messy comma coordinate
    assert parse_coordinate("107.4967733,237", fallback) == pytest.approx(107.4967733)
    assert parse_coordinate("-6.851117,999", fallback) == pytest.approx(-6.851117)
    # Invalid fallback
    assert parse_coordinate("invalid", fallback) == pytest.approx(fallback)
    assert parse_coordinate("", fallback) == pytest.approx(fallback)


def test_load_cctv_points_with_header(tmp_path: Path) -> None:
    csv_file = tmp_path / "cctv_points.csv"
    csv_file.write_text(
        "name,url,lat,lon\n"
        "Cam Alpha,http://example.com/cam1.m3u8,-6.123,107.456\n"
        "Cam Beta,https://example.com/cam2.m3u8,-6.789,107.999\n",
        encoding="utf-8",
    )
    cfg = make_dummy_config(tmp_path, config_file=csv_file)
    points = load_cctv_points(cfg)

    assert len(points) == 2
    assert points[0].name == "cam_alpha"
    assert points[0].url == "http://example.com/cam1.m3u8"
    assert points[0].lat == pytest.approx(-6.123)
    assert points[0].lon == pytest.approx(107.456)

    assert points[1].name == "cam_beta"
    assert points[1].url == "https://example.com/cam2.m3u8"
    assert points[1].lat == pytest.approx(-6.789)
    assert points[1].lon == pytest.approx(107.999)


def test_load_cctv_points_alternative_headers(tmp_path: Path) -> None:
    csv_file = tmp_path / "cctv_points.csv"
    csv_file.write_text(
        "name,link,latitude,longitude\nSimpang Tiga,http://example.com/s3.m3u8,-6.500,107.200\n",
        encoding="utf-8",
    )
    cfg = make_dummy_config(tmp_path, config_file=csv_file)
    points = load_cctv_points(cfg)

    assert len(points) == 1
    assert points[0].name == "simpang_tiga"
    assert points[0].url == "http://example.com/s3.m3u8"
    assert points[0].lat == pytest.approx(-6.500)
    assert points[0].lon == pytest.approx(107.200)


def test_load_cctv_points_headerless(tmp_path: Path) -> None:
    csv_file = tmp_path / "cctv_points.csv"
    csv_file.write_text(
        "cam_one,http://example.com/one.m3u8\ncam_two,https://example.com/two.m3u8\n",
        encoding="utf-8",
    )
    cfg = make_dummy_config(tmp_path, config_file=csv_file)
    points = load_cctv_points(cfg)

    assert len(points) == 2
    assert points[0].name == "cam_one"
    assert points[0].url == "http://example.com/one.m3u8"
    assert points[0].lat == pytest.approx(cfg.metadata.default_lat)
    assert points[0].lon == pytest.approx(cfg.metadata.default_lon)


def test_load_cctv_points_duplicate_name_rejection(tmp_path: Path) -> None:
    csv_file = tmp_path / "cctv_points.csv"
    csv_file.write_text(
        "name,url\nCam 1,http://example.com/1.m3u8\ncam_1,http://example.com/2.m3u8\n",
        encoding="utf-8",
    )
    cfg = make_dummy_config(tmp_path, config_file=csv_file)
    with pytest.raises(ValueError, match="duplikat"):
        load_cctv_points(cfg)


def test_load_cctv_points_invalid_url_rejection(tmp_path: Path) -> None:
    csv_file = tmp_path / "cctv_points.csv"
    csv_file.write_text(
        "name,url\nCam-1,rtsp://example.com/1\n",
        encoding="utf-8",
    )
    cfg = make_dummy_config(tmp_path, config_file=csv_file)
    with pytest.raises(ValueError, match="URL CCTV tidak valid"):
        load_cctv_points(cfg)


# ============================================================================
# 2. build_ffmpeg_command tests
# ============================================================================
def test_build_ffmpeg_command_copy_m3u8(tmp_path: Path) -> None:
    point = CCTVPoint(name="padalarang", url="https://stream.test/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(
        tmp_path,
        video_container="ts",
        ffmpeg_transport_mode="copy",
        segment_seconds=60,
        ffmpeg_loglevel="warning",
        hls_reconnect_at_eof=False,
        segment_atclocktime=False,
    )
    stop_event = threading.Event()
    recorder = CCTVRecorder(point, cfg, stop_event)
    cmd = recorder.build_ffmpeg_command()

    expected_output_pattern = str(
        tmp_path
        / recorder.current_date_folder()
        / "padalarang"
        / "videos"
        / "padalarang_%Y%m%d_%H%M%S.ts"
    )

    expected = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "+genpts+discardcorrupt+nobuffer",
        "-err_detect",
        "ignore_err",
        "-rw_timeout",
        "10000000",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_on_http_error",
        "5xx",
        "-reconnect_delay_max",
        "10",
        "-http_persistent",
        "1",
        "-multiple_requests",
        "1",
        "-user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "-headers",
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n",
        "-analyzeduration",
        "10000000",
        "-probesize",
        "10000000",
        "-live_start_index",
        "-1",
        "-i",
        "https://stream.test/live.m3u8",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-max_muxing_queue_size",
        "1024",
        "-f",
        "segment",
        "-segment_time",
        "60",
        "-reset_timestamps",
        "1",
        "-strftime",
        "1",
        "-segment_format",
        "mpegts",
        expected_output_pattern,
    ]
    assert cmd == expected


def test_build_ffmpeg_command_transcode_hevc_nvenc(tmp_path: Path) -> None:
    point = CCTVPoint(name="padalarang", url="https://stream.test/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(
        tmp_path,
        video_container="ts",
        ffmpeg_transport_mode="transcode",
        video_encoder="hevc_nvenc",
        output_fps=10,
        segment_keyframe_seconds=2,
        transcode_preset="p4",
        target_bitrate="650k",
        max_bitrate="900k",
        buffer_size="1300k",
        output_height=720,
        hls_reconnect_at_eof=True,
        segment_atclocktime=True,
    )
    stop_event = threading.Event()
    recorder = CCTVRecorder(point, cfg, stop_event)
    cmd = recorder.build_ffmpeg_command()

    expected_output_pattern = str(
        tmp_path
        / recorder.current_date_folder()
        / "padalarang"
        / "videos"
        / "padalarang_%Y%m%d_%H%M%S.ts"
    )

    expected = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "+genpts+discardcorrupt+nobuffer",
        "-err_detect",
        "ignore_err",
        "-rw_timeout",
        "10000000",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_on_http_error",
        "5xx",
        "-reconnect_delay_max",
        "10",
        "-http_persistent",
        "1",
        "-multiple_requests",
        "1",
        "-user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "-headers",
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n",
        "-analyzeduration",
        "10000000",
        "-probesize",
        "10000000",
        "-reconnect_at_eof",
        "1",
        "-live_start_index",
        "-1",
        "-i",
        "https://stream.test/live.m3u8",
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "fps=10,scale=-2:720,format=yuv420p",
        "-fps_mode",
        "cfr",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p4",
        "-rc:v",
        "vbr",
        "-b:v",
        "650k",
        "-maxrate:v",
        "900k",
        "-bufsize:v",
        "1300k",
        "-g",
        "20",
        "-keyint_min",
        "20",
        "-sc_threshold",
        "0",
        "-max_muxing_queue_size",
        "1024",
        "-f",
        "segment",
        "-segment_time",
        "60",
        "-reset_timestamps",
        "1",
        "-strftime",
        "1",
        "-segment_atclocktime",
        "1",
        "-segment_format",
        "mpegts",
        expected_output_pattern,
    ]
    assert cmd == expected


def test_build_ffmpeg_command_transcode_libx264(tmp_path: Path) -> None:
    point = CCTVPoint(name="padalarang", url="https://stream.test/live.flv", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(
        tmp_path,
        video_container="ts",
        ffmpeg_transport_mode="transcode",
        video_encoder="libx264",
        output_fps=15,
        segment_keyframe_seconds=2,
        transcode_preset="p4",  # Should convert to "veryfast" for cpu encoder
        target_bitrate="650k",
        max_bitrate="900k",
        buffer_size="1300k",
        output_height=0,
    )
    stop_event = threading.Event()
    recorder = CCTVRecorder(point, cfg, stop_event)
    cmd = recorder.build_ffmpeg_command()

    expected_output_pattern = str(
        tmp_path
        / recorder.current_date_folder()
        / "padalarang"
        / "videos"
        / "padalarang_%Y%m%d_%H%M%S.ts"
    )

    # Note: url is not .m3u8, so -live_start_index is omitted
    expected = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "+genpts+discardcorrupt+nobuffer",
        "-err_detect",
        "ignore_err",
        "-rw_timeout",
        "10000000",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_on_http_error",
        "5xx",
        "-reconnect_delay_max",
        "10",
        "-http_persistent",
        "1",
        "-multiple_requests",
        "1",
        "-user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "-headers",
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n",
        "-analyzeduration",
        "10000000",
        "-probesize",
        "10000000",
        "-i",
        "https://stream.test/live.flv",
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "fps=15,format=yuv420p",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-b:v",
        "650k",
        "-maxrate:v",
        "900k",
        "-bufsize:v",
        "1300k",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        "-max_muxing_queue_size",
        "1024",
        "-f",
        "segment",
        "-segment_time",
        "60",
        "-reset_timestamps",
        "1",
        "-strftime",
        "1",
        "-segment_format",
        "mpegts",
        expected_output_pattern,
    ]
    assert cmd == expected


def test_build_ffmpeg_command_mp4_container(tmp_path: Path) -> None:
    point = CCTVPoint(name="padalarang", url="https://stream.test/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(
        tmp_path,
        video_container="mp4",
        ffmpeg_transport_mode="copy",
    )
    stop_event = threading.Event()
    recorder = CCTVRecorder(point, cfg, stop_event)
    cmd = recorder.build_ffmpeg_command()

    expected_output_pattern = str(
        tmp_path
        / recorder.current_date_folder()
        / "padalarang"
        / "videos"
        / "padalarang_%Y%m%d_%H%M%S.mp4"
    )

    assert "-segment_format" in cmd
    idx = cmd.index("-segment_format")
    assert cmd[idx : idx + 4] == ["-segment_format", "mp4", "-movflags", "+faststart"]
    assert cmd[-1] == expected_output_pattern


# ============================================================================
# 3. validate_video_encoder: recorder-side check only, no archive fallback
# ============================================================================
def test_validate_video_encoder_skips_the_ffmpeg_probe_in_copy_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default transport mode never invokes an encoder, so this must not even
    shell out to ffmpeg -- a missing/broken ffmpeg on PATH must not block startup."""
    cfg = make_dummy_config(tmp_path, ffmpeg_transport_mode="copy")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ffmpeg should not be probed in copy mode")

    monkeypatch.setattr("cctv_scraper.app.subprocess.run", fail_if_called)
    assert validate_video_encoder(cfg) is cfg


def test_validate_video_encoder_accepts_a_compiled_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_dummy_config(tmp_path, ffmpeg_transport_mode="transcode", video_encoder="libx264")
    listed = subprocess_completed("V..... libx264 libx264 H.264 / AVC / MPEG-4 AVC\n")
    monkeypatch.setattr("cctv_scraper.app.subprocess.run", lambda *a, **k: listed)

    assert validate_video_encoder(cfg) is cfg


def test_validate_video_encoder_rejects_a_missing_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_dummy_config(tmp_path, ffmpeg_transport_mode="smooth", video_encoder="hevc_qsv")
    listed = subprocess_completed("V..... libx264 libx264 H.264 / AVC / MPEG-4 AVC\n")
    monkeypatch.setattr("cctv_scraper.app.subprocess.run", lambda *a, **k: listed)

    with pytest.raises(RuntimeError, match="hevc_qsv"):
        validate_video_encoder(cfg)


def test_validate_video_encoder_raises_when_ffmpeg_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_dummy_config(tmp_path, ffmpeg_transport_mode="transcode")

    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("cctv_scraper.app.subprocess.run", raise_missing)
    with pytest.raises(RuntimeError, match="FFmpeg tidak ditemukan"):
        validate_video_encoder(cfg)

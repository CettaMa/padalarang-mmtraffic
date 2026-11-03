import argparse
import csv
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

# =========================================================
# DEFAULT CONFIG CONSTANTS
# =========================================================
# Note: Code defaults take precedence where .env.example previously disagreed:
# - SEGMENT_SECONDS: 60 (code default)
# - METADATA_INTERVAL_SECONDS: 60 (code default)
# - OPENMETEO_INTERVAL_SECONDS: 60 (code default)
# - VIDEO_ENCODER: "hevc_nvenc" (code default)

DEFAULT_CONFIG_FILE = "cctv_points.csv"
DEFAULT_OUTPUT_ROOT = "dataset"

DEFAULT_SEGMENT_SECONDS = 60
DEFAULT_RESTART_DELAY_SECONDS = 5
DEFAULT_HEALTH_CHECK_SECONDS = 30
DEFAULT_STALE_FILE_SECONDS = 240
DEFAULT_METADATA_INTERVAL_SECONDS = 60
DEFAULT_TOMTOM_INTERVAL_SECONDS = 300
DEFAULT_OPENMETEO_INTERVAL_SECONDS = 60
DEFAULT_DISK_CHECK_SECONDS = 300
DEFAULT_HLS_RECONNECT_AT_EOF = False

DEFAULT_RETENTION_DAYS = 7
DEFAULT_MIN_FREE_SPACE_GB = 20.0

DEFAULT_LAT = "-6.851117"
DEFAULT_LON = "107.496586"

API_TIMEOUT_SECONDS = 10


# =========================================================
# DATA MODELS (Grouped Frozen Dataclasses)
# =========================================================
@dataclass(frozen=True)
class CCTVPoint:
    name: str
    url: str
    lat: float
    lon: float


@dataclass(frozen=True)
class RecorderConfig:
    segment_seconds: int
    video_container: str
    restart_delay_seconds: int
    health_check_seconds: int
    stale_file_seconds: int
    ffmpeg_loglevel: str
    ffmpeg_transport_mode: str
    ffmpeg_user_agent: str
    ffmpeg_referer: str
    ffmpeg_origin: str
    ffmpeg_analyzeduration: str
    ffmpeg_probesize: str
    hls_reconnect_at_eof: bool
    segment_atclocktime: bool
    hls_live_start_index: str
    ffmpeg_rw_timeout: str
    ffmpeg_reconnect_delay_max: str
    ffmpeg_reconnect_on_http_error: str
    output_fps: int
    transcode_preset: str
    segment_keyframe_seconds: int
    video_encoder: str
    target_bitrate: str
    max_bitrate: str
    buffer_size: str
    output_height: int
    # HTTP keep-alive for the HLS input. Disabling it opens a fresh TCP+TLS
    # connection for every playlist reload and every segment, which fills the
    # host's conntrack table and the upstream NAT session table over a 24h run.
    ffmpeg_http_persistent: bool = True
    ffmpeg_multiple_requests: bool = True


@dataclass(frozen=True)
class MetadataConfig:
    metadata_interval_seconds: int
    tomtom_interval_seconds: int
    openmeteo_interval_seconds: int
    default_lat: float
    default_lon: float
    tomtom_api_key: str | None = None
    min_pass_interval_seconds: int = 5
    failure_backoff_base_seconds: int = 5
    failure_backoff_max_seconds: int = 300


@dataclass(frozen=True)
class ArchiveConfig:
    """Settings for DailyArchiver, which packages each completed day into a 7z.

    There is no local re-encoding step: raw transport-stream segments go
    straight into the daily archive, so this config carries no encoder,
    bitrate, or hardware-device fields.
    """

    safe_age_seconds: int
    daily_archive_enabled: bool = True
    daily_archive_scan_seconds: int = 300
    daily_archive_delete_source: bool = False
    archiver_binary: str = "7z"


@dataclass(frozen=True)
class StorageConfig:
    output_root: Path
    retention_days: int
    min_free_space_gb: float
    disk_check_seconds: int
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5


@dataclass(frozen=True)
class NetworkConfig:
    preflight_check: bool
    offline_retry_seconds: int
    network_retry_seconds: int
    expired_url_escalation_threshold: int = 3
    # Every recorder failing preflight for this long means the host itself lost
    # outbound connectivity. 0 disables the watchdog.
    network_watchdog_seconds: int = 600


@dataclass(frozen=True)
class RuntimeConfig:
    config_file: Path
    recorder: RecorderConfig
    metadata: MetadataConfig
    archive: ArchiveConfig
    storage: StorageConfig
    network: NetworkConfig


# =========================================================
# UTILS & VALIDATORS
# =========================================================
def sanitize_filename(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-zA-Z0-9_\-]", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_local() -> datetime:
    return datetime.now()


def safe_float(value: object, fallback: float) -> float:
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return fallback


def parse_coordinate(value: object, fallback: float) -> float:
    """
    Ambil angka koordinat pertama dari nilai CSV.

    Beberapa baris lama memakai format seperti "107.4967733,237"; bagian
    setelah koma tampaknya bukan bagian longitude, jadi jangan langsung
    jatuh ke fallback default.
    """
    text = str(value).strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if match:
            return safe_float(match.group(0), fallback)
        return fallback


def is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def validate_positive_number(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} harus lebih besar dari 0.")


def validate_nonnegative_number(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} tidak boleh kurang dari 0.")


def validate_video_container(name: str, value: str) -> None:
    if str(value).lower().strip() not in {"ts", "mp4"}:
        raise ValueError("video_container harus 'ts' atau 'mp4'.")


def validate_transport_mode(name: str, value: str) -> None:
    if str(value).lower().strip() not in {"copy", "smooth", "transcode"}:
        raise ValueError("FFMPEG_TRANSPORT_MODE harus 'copy', 'smooth', atau 'transcode'.")


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# =========================================================
# TABLE-DRIVEN CONFIG LOADER
# =========================================================
def _parse_env(
    name: str,
    type_conv: Callable[[str], Any],
    default: Any,
    validator: Callable[[str, Any], None] | None = None,
) -> Any:
    val_str = os.getenv(name)
    if val_str is None or str(val_str).strip() == "":
        val = default
    else:
        raw = str(val_str).strip()
        if type_conv is bool:
            val = is_truthy(raw)
        elif type_conv is int:
            try:
                val = int(raw)
            except ValueError as exc:
                raise ValueError(f"Environment variable {name} harus berupa integer.") from exc
        elif type_conv is float:
            try:
                val = float(raw)
            except ValueError as exc:
                raise ValueError(f"Environment variable {name} harus berupa angka.") from exc
        elif type_conv is Path:
            val = Path(raw)
        else:
            val = type_conv(raw)

    if validator:
        validator(name, val)
    return val


def _cli_or_env(
    cli_val: Any,
    env_name: str,
    default: Any,
    type_conv: Callable[[Any], Any] = str,
    validator: Callable[[str, Any], None] | None = None,
) -> Any:
    if cli_val is not None:
        val = type_conv(cli_val) if type_conv is not None else cli_val
        if validator:
            validator(env_name, val)
        return val
    return _parse_env(env_name, type_conv, default, validator)


# Spec definitions: (field_name, env_name, type_conv, default, validator)
RECORDER_SPECS: list[tuple[str, str, type, Any, Callable[[str, Any], None] | None]] = [
    ("restart_delay_seconds", "RESTART_DELAY_SECONDS", int, DEFAULT_RESTART_DELAY_SECONDS, None),
    ("health_check_seconds", "HEALTH_CHECK_SECONDS", int, DEFAULT_HEALTH_CHECK_SECONDS, None),
    ("stale_file_seconds", "STALE_FILE_SECONDS", int, DEFAULT_STALE_FILE_SECONDS, None),
    ("ffmpeg_loglevel", "FFMPEG_LOGLEVEL", str, "warning", None),
    ("ffmpeg_transport_mode", "FFMPEG_TRANSPORT_MODE", str, "copy", validate_transport_mode),
    (
        "ffmpeg_user_agent",
        "FFMPEG_USER_AGENT",
        str,
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        None,
    ),
    ("ffmpeg_referer", "FFMPEG_REFERER", str, "", None),
    ("ffmpeg_origin", "FFMPEG_ORIGIN", str, "", None),
    ("ffmpeg_analyzeduration", "FFMPEG_ANALYZEDURATION", str, "10000000", None),
    ("ffmpeg_probesize", "FFMPEG_PROBESIZE", str, "10000000", None),
    ("hls_reconnect_at_eof", "HLS_RECONNECT_AT_EOF", bool, DEFAULT_HLS_RECONNECT_AT_EOF, None),
    ("segment_atclocktime", "SEGMENT_ATCLOCKTIME", bool, False, None),
    ("hls_live_start_index", "HLS_LIVE_START_INDEX", str, "-1", None),
    ("ffmpeg_rw_timeout", "FFMPEG_RW_TIMEOUT", str, "10000000", None),
    ("ffmpeg_reconnect_delay_max", "FFMPEG_RECONNECT_DELAY_MAX", str, "10", None),
    ("ffmpeg_reconnect_on_http_error", "FFMPEG_RECONNECT_ON_HTTP_ERROR", str, "5xx", None),
    ("output_fps", "OUTPUT_FPS", int, 10, None),
    ("transcode_preset", "TRANSCODE_PRESET", str, "p4", None),
    ("segment_keyframe_seconds", "SEGMENT_KEYFRAME_SECONDS", int, 2, None),
    ("video_encoder", "VIDEO_ENCODER", str, "hevc_nvenc", None),
    ("target_bitrate", "TARGET_BITRATE", str, "650k", None),
    ("max_bitrate", "MAX_BITRATE", str, "900k", None),
    ("buffer_size", "BUFFER_SIZE", str, "1300k", None),
    ("output_height", "OUTPUT_HEIGHT", int, 0, None),
    ("ffmpeg_http_persistent", "FFMPEG_HTTP_PERSISTENT", bool, True, None),
    ("ffmpeg_multiple_requests", "FFMPEG_MULTIPLE_REQUESTS", bool, True, None),
]

METADATA_SPECS: list[tuple[str, str, type, Any, Callable[[str, Any], None] | None]] = [
    (
        "metadata_interval_seconds",
        "METADATA_INTERVAL_SECONDS",
        int,
        DEFAULT_METADATA_INTERVAL_SECONDS,
        None,
    ),
    (
        "tomtom_interval_seconds",
        "TOMTOM_INTERVAL_SECONDS",
        int,
        DEFAULT_TOMTOM_INTERVAL_SECONDS,
        None,
    ),
    (
        "openmeteo_interval_seconds",
        "OPENMETEO_INTERVAL_SECONDS",
        int,
        DEFAULT_OPENMETEO_INTERVAL_SECONDS,
        None,
    ),
    (
        "min_pass_interval_seconds",
        "METADATA_MIN_PASS_INTERVAL_SECONDS",
        int,
        5,
        validate_positive_number,
    ),
    (
        "failure_backoff_base_seconds",
        "METADATA_FAILURE_BACKOFF_BASE_SECONDS",
        int,
        5,
        validate_positive_number,
    ),
    (
        "failure_backoff_max_seconds",
        "METADATA_FAILURE_BACKOFF_MAX_SECONDS",
        int,
        300,
        validate_positive_number,
    ),
    ("default_lat", "DEFAULT_LAT", float, float(DEFAULT_LAT), None),
    ("default_lon", "DEFAULT_LON", float, float(DEFAULT_LON), None),
]

ARCHIVE_SPECS: list[tuple[str, str, type, Any, Callable[[str, Any], None] | None]] = [
    ("safe_age_seconds", "ARCHIVE_SAFE_AGE_SECONDS", int, 90, validate_positive_number),
    ("daily_archive_enabled", "DAILY_ARCHIVE_ENABLED", bool, True, None),
    (
        "daily_archive_scan_seconds",
        "DAILY_ARCHIVE_SCAN_SECONDS",
        int,
        300,
        validate_positive_number,
    ),
    ("daily_archive_delete_source", "DAILY_ARCHIVE_DELETE_SOURCE", bool, False, None),
    ("archiver_binary", "ARCHIVER_BINARY", str, "7z", None),
]

STORAGE_SPECS: list[tuple[str, str, type, Any, Callable[[str, Any], None] | None]] = [
    ("disk_check_seconds", "DISK_CHECK_SECONDS", int, DEFAULT_DISK_CHECK_SECONDS, None),
    ("log_max_bytes", "LOG_MAX_BYTES", int, 10 * 1024 * 1024, validate_positive_number),
    ("log_backup_count", "LOG_BACKUP_COUNT", int, 5, validate_nonnegative_number),
]

NETWORK_SPECS: list[tuple[str, str, type, Any, Callable[[str, Any], None] | None]] = [
    ("preflight_check", "PREFLIGHT_CHECK", bool, True, None),
    ("offline_retry_seconds", "OFFLINE_RETRY_SECONDS", int, 300, None),
    ("network_retry_seconds", "NETWORK_RETRY_SECONDS", int, 60, None),
    (
        "expired_url_escalation_threshold",
        "EXPIRED_URL_ESCALATION_THRESHOLD",
        int,
        3,
        validate_positive_number,
    ),
    (
        "network_watchdog_seconds",
        "NETWORK_WATCHDOG_SECONDS",
        int,
        600,
        validate_nonnegative_number,
    ),
]


def _load_specs(
    specs: list[tuple[str, str, type, Any, Callable[[str, Any], None] | None]],
) -> dict[str, Any]:
    return {
        field_name: _parse_env(env_name, type_conv, default, validator)
        for field_name, env_name, type_conv, default, validator in specs
    }


def load_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    # Load .env for both secrets and runtime options. CLI arguments still take priority.
    load_dotenv()

    # CLI or ENV values
    config_file: Path = _cli_or_env(
        getattr(args, "config", None),
        "CCTV_CONFIG_FILE",
        Path(DEFAULT_CONFIG_FILE),
        Path,
    )
    output_root: Path = _cli_or_env(
        getattr(args, "output", None),
        "CCTV_OUTPUT_ROOT",
        Path(DEFAULT_OUTPUT_ROOT),
        Path,
    )
    segment_seconds: int = _cli_or_env(
        getattr(args, "segment_seconds", None),
        "SEGMENT_SECONDS",
        DEFAULT_SEGMENT_SECONDS,
        int,
        validate_positive_number,
    )
    retention_days: int = _cli_or_env(
        getattr(args, "retention_days", None),
        "RETENTION_DAYS",
        DEFAULT_RETENTION_DAYS,
        int,
        validate_positive_number,
    )
    min_free_space_gb: float = _cli_or_env(
        getattr(args, "min_free_space_gb", None),
        "MIN_FREE_SPACE_GB",
        DEFAULT_MIN_FREE_SPACE_GB,
        float,
        validate_positive_number,
    )
    video_container: str = _cli_or_env(
        getattr(args, "video_container", None),
        "VIDEO_CONTAINER",
        "ts",
        str,
        validate_video_container,
    )

    # Sub-config loading via specs
    recorder_dict = _load_specs(RECORDER_SPECS)
    recorder_dict["segment_seconds"] = segment_seconds
    recorder_dict["video_container"] = video_container.lower().strip()
    recorder_config = RecorderConfig(**recorder_dict)

    metadata_dict = _load_specs(METADATA_SPECS)
    metadata_dict["tomtom_api_key"] = os.getenv("TOMTOM_API")
    metadata_config = MetadataConfig(**metadata_dict)

    archive_config = ArchiveConfig(**_load_specs(ARCHIVE_SPECS))

    storage_dict = _load_specs(STORAGE_SPECS)
    storage_dict["output_root"] = output_root
    storage_dict["retention_days"] = retention_days
    storage_dict["min_free_space_gb"] = min_free_space_gb
    storage_config = StorageConfig(**storage_dict)

    network_config = NetworkConfig(**_load_specs(NETWORK_SPECS))

    return RuntimeConfig(
        config_file=config_file,
        recorder=recorder_config,
        metadata=metadata_config,
        archive=archive_config,
        storage=storage_config,
        network=network_config,
    )


def load_cctv_points(config: RuntimeConfig) -> list[CCTVPoint]:
    path = config.config_file

    if not path.exists():
        raise FileNotFoundError(f"File konfigurasi CCTV tidak ditemukan: {path}")

    points: list[CCTVPoint] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)

        has_header = "name" in sample.lower() and (
            "url" in sample.lower() or "link" in sample.lower()
        )

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or row.get("nama") or "").strip()
                url = (row.get("url") or row.get("link") or "").strip()

                if not name or not url:
                    continue

                point_name = sanitize_filename(name)
                if not point_name:
                    continue
                if not is_http_url(url):
                    raise ValueError(f"URL CCTV tidak valid untuk '{name}': {url}")

                lat = row.get("lat") or row.get("latitude") or config.metadata.default_lat
                lon = row.get("lon") or row.get("longitude") or config.metadata.default_lon

                points.append(
                    CCTVPoint(
                        name=point_name,
                        url=url,
                        lat=parse_coordinate(lat, config.metadata.default_lat),
                        lon=parse_coordinate(lon, config.metadata.default_lon),
                    )
                )
        else:
            csv_reader = csv.reader(f)
            for raw_row in csv_reader:
                if len(raw_row) < 2:
                    continue

                name = raw_row[0].strip()
                url = raw_row[1].strip()

                if not name or not url:
                    continue

                point_name = sanitize_filename(name)
                if not point_name:
                    continue
                if not is_http_url(url):
                    raise ValueError(f"URL CCTV tidak valid untuk '{name}': {url}")

                points.append(
                    CCTVPoint(
                        name=point_name,
                        url=url,
                        lat=config.metadata.default_lat,
                        lon=config.metadata.default_lon,
                    )
                )

    if not points:
        raise ValueError("Tidak ada titik CCTV valid di file konfigurasi.")

    seen = set()
    duplicates = []
    for p in points:
        if p.name in seen:
            duplicates.append(p.name)
        seen.add(p.name)

    if duplicates:
        raise ValueError(f"Nama titik CCTV duplikat ditemukan: {sorted(set(duplicates))}")

    return points

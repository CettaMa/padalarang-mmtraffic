import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from cctv_scraper.config import DEFAULT_CONFIG_FILE, DEFAULT_OUTPUT_ROOT, is_truthy

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_USERNAME = "admin"
DEFAULT_CACHE_TTL_SECONDS = 15
DEFAULT_STATUS_TAIL_LINES = 5000
DEFAULT_LOG_TAIL_LINES = 200
DEFAULT_HISTORY_INTERVAL_SECONDS = 60
DEFAULT_HISTORY_RETENTION_DAYS = 90

# A camera is considered to be recording while its newest raw segment is younger
# than this multiple of SEGMENT_SECONDS.  Two segments of slack absorbs one
# missed rollover without flapping the badge.
RECORDING_STALE_SEGMENT_MULTIPLIER = 3


def env_int(name: str, fallback: int) -> int:
    try:
        return int(str(os.getenv(name, fallback)).strip())
    except (TypeError, ValueError):
        return fallback


def env_float(name: str, fallback: float) -> float:
    try:
        return float(str(os.getenv(name, fallback)).strip())
    except (TypeError, ValueError):
        return fallback


def env_bool(name: str, fallback: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return fallback
    return is_truthy(raw)


class DashboardConfigError(RuntimeError):
    """Raised when the dashboard cannot start safely with the given settings."""


@dataclass(frozen=True)
class DashboardConfig:
    output_root: Path
    config_file: Path
    host: str
    port: int
    auth_enabled: bool
    username: str
    password: str
    cache_ttl_seconds: float
    status_tail_lines: int
    log_tail_lines: int
    history_enabled: bool
    history_db: Path
    history_interval_seconds: int
    history_retention_days: int
    segment_seconds: int
    min_free_space_gb: float

    @property
    def status_dir(self) -> Path:
        return self.output_root / "status"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def archives_dir(self) -> Path:
        return self.output_root / "archives"

    @property
    def recording_stale_seconds(self) -> int:
        return self.segment_seconds * RECORDING_STALE_SEGMENT_MULTIPLIER


def load_dashboard_config() -> DashboardConfig:
    load_dotenv()

    output_root = Path(os.getenv("CCTV_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser()
    auth_enabled = env_bool("DASHBOARD_AUTH_ENABLED", True)
    password = os.getenv("DASHBOARD_PASSWORD", "").strip()

    # Camera URLs are treated as credentials, so an exposed dashboard must not
    # start without a password.  Fail loudly instead of silently opening up.
    if auth_enabled and not password:
        raise DashboardConfigError(
            "DASHBOARD_PASSWORD is empty while DASHBOARD_AUTH_ENABLED is true. "
            "Set a password in .env, or set DASHBOARD_AUTH_ENABLED=false and bind "
            "DASHBOARD_HOST=127.0.0.1."
        )

    history_db_default = output_root / "dashboard" / "history.db"
    history_db = Path(os.getenv("DASHBOARD_HISTORY_DB", str(history_db_default))).expanduser()

    return DashboardConfig(
        output_root=output_root,
        config_file=Path(os.getenv("CCTV_CONFIG_FILE", DEFAULT_CONFIG_FILE)).expanduser(),
        host=os.getenv("DASHBOARD_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST,
        port=env_int("DASHBOARD_PORT", DEFAULT_PORT),
        auth_enabled=auth_enabled,
        username=os.getenv("DASHBOARD_USERNAME", DEFAULT_USERNAME).strip() or DEFAULT_USERNAME,
        password=password,
        cache_ttl_seconds=env_float("DASHBOARD_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS),
        status_tail_lines=env_int("DASHBOARD_STATUS_TAIL_LINES", DEFAULT_STATUS_TAIL_LINES),
        log_tail_lines=env_int("DASHBOARD_LOG_TAIL_LINES", DEFAULT_LOG_TAIL_LINES),
        history_enabled=env_bool("DASHBOARD_HISTORY_ENABLED", True),
        history_db=history_db,
        history_interval_seconds=env_int(
            "DASHBOARD_HISTORY_INTERVAL_SECONDS", DEFAULT_HISTORY_INTERVAL_SECONDS
        ),
        history_retention_days=env_int(
            "DASHBOARD_HISTORY_RETENTION_DAYS", DEFAULT_HISTORY_RETENTION_DAYS
        ),
        segment_seconds=env_int("SEGMENT_SECONDS", 60),
        min_free_space_gb=env_float("MIN_FREE_SPACE_GB", 20.0),
    )

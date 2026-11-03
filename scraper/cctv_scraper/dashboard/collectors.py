import csv
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from cctv_scraper.config import sanitize_filename
from cctv_scraper.dashboard.security import mask_url, scrub_text
from cctv_scraper.dashboard.settings import DashboardConfig
from cctv_scraper.storage_scan import parse_date_dir_name

STATUS_FIELDS = ("timestamp", "cctv_name", "status", "reason", "http_status", "url")
STATUS_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
RESERVED_DIRS = {"logs", "status", "archives", "dashboard"}

# Only ``online_preflight_ok`` asserts a live stream. ``recorder_restart`` is
# written the moment FFmpeg stops, so the span that follows it is genuinely
# unrecorded and must not be counted as uptime.
ONLINE_STATUSES = frozenset({"online_preflight_ok"})

HEALTH_ONLINE = "online"
HEALTH_STALE = "stale"
HEALTH_RESTARTING = "restarting"
HEALTH_OFFLINE = "offline"
HEALTH_EXPIRED = "expired_url"
HEALTH_UNKNOWN = "unknown"

# Ordered worst-first so the overview can sort attention to the top.
HEALTH_SEVERITY = {
    HEALTH_EXPIRED: 0,
    HEALTH_OFFLINE: 1,
    HEALTH_STALE: 2,
    HEALTH_RESTARTING: 3,
    HEALTH_UNKNOWN: 4,
    HEALTH_ONLINE: 5,
}


@dataclass(frozen=True)
class StatusRow:
    timestamp: str
    epoch: float
    camera: str
    status: str
    reason: str
    http_status: str
    url: str


@dataclass
class CameraSnapshot:
    name: str
    health: str = HEALTH_UNKNOWN
    status: str = ""
    reason: str = ""
    http_status: str = ""
    url: str = ""
    last_status_at: str | None = None
    last_segment_at: str | None = None
    seconds_since_segment: int | None = None
    uptime_24h: float | None = None
    raw_backlog: int = 0
    coverage_today_seconds: int = 0
    coverage_today_ratio: float = 0.0


@dataclass
class DiskSnapshot:
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    percent_used: float = 0.0
    free_gb: float = 0.0
    min_free_gb: float = 0.0
    low_space: bool = False


@dataclass
class ArchiveEntry:
    day: str
    size_bytes: int
    modified_at: str


@dataclass
class DayCoverage:
    day: str
    per_camera: dict[str, int] = field(default_factory=dict)
    hourly: dict[str, list[int]] = field(default_factory=dict)
    segment_count: int = 0
    archive_failed: bool = False


def tail_lines(path: Path, max_lines: int, max_bytes: int = 4_000_000) -> tuple[list[str], bool]:
    """Read the last ``max_lines`` lines without loading the whole file.

    Also returns whether the read started at byte 0, which tells callers whether
    the first line is the real CSV header or a record truncated mid-line.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            block = min(size, max_bytes)
            handle.seek(size - block)
            data = handle.read(block)
    except OSError:
        return [], False

    from_start = block == size
    lines = data.decode("utf-8", errors="replace").splitlines()
    if not from_start and lines:
        lines = lines[1:]
    return lines[-max_lines:], from_start


def parse_status_rows(lines: list[str], from_start: bool) -> list[StatusRow]:
    if from_start and lines and lines[0].startswith("timestamp,"):
        lines = lines[1:]

    rows: list[StatusRow] = []
    for raw in csv.reader(lines):
        if len(raw) != len(STATUS_FIELDS):
            continue
        record = dict(zip(STATUS_FIELDS, raw, strict=True))
        try:
            moment = datetime.strptime(record["timestamp"], STATUS_TIMESTAMP_FORMAT)
        except ValueError:
            continue
        rows.append(
            StatusRow(
                timestamp=record["timestamp"],
                epoch=moment.timestamp(),
                camera=record["cctv_name"],
                status=record["status"],
                reason=scrub_text(record["reason"]),
                http_status=record["http_status"],
                url=mask_url(record["url"]),
            )
        )
    rows.sort(key=lambda row: row.epoch)
    return rows


def read_status_rows(
    config: DashboardConfig, camera: str, max_lines: int | None = None
) -> list[StatusRow]:
    path = config.status_dir / f"{camera}_status.csv"
    lines, from_start = tail_lines(path, max_lines or config.status_tail_lines)
    return parse_status_rows(lines, from_start)


def uptime_ratio(
    rows: list[StatusRow],
    window_start: float,
    window_end: float,
    evidence_until: float | None = None,
) -> float | None:
    """Fraction of the window spent online, reconstructed from state transitions.

    The recorder writes ``online_preflight_ok`` once per FFmpeg start and no
    closing row when the process is killed, so a stale "online" state would
    otherwise carry forward for ever. ``evidence_until`` is the moment segments
    stopped arriving; nothing after it can be counted as uptime.
    """
    if window_end <= window_start:
        return None
    if evidence_until is not None and evidence_until <= window_start:
        return 0.0

    transitions = [(row.epoch, row.status in ONLINE_STATUSES) for row in rows]
    before = [state for epoch, state in transitions if epoch <= window_start]
    inside = [(epoch, state) for epoch, state in transitions if window_start < epoch < window_end]

    if evidence_until is not None and window_start < evidence_until < window_end:
        inside.append((evidence_until, False))
        inside.sort(key=lambda item: item[0])

    if before:
        current_state = before[-1]
    elif inside:
        # No evidence before the window; assume the first observed state held.
        current_state = inside[0][1]
    else:
        return None

    online_seconds = 0.0
    cursor = window_start
    for epoch, state in inside:
        if current_state:
            online_seconds += epoch - cursor
        cursor, current_state = epoch, state
    if current_state:
        online_seconds += window_end - cursor

    return max(0.0, min(1.0, online_seconds / (window_end - window_start)))


def date_dirs(root: Path) -> list[tuple[date, Path]]:
    found: list[tuple[date, Path]] = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir() or entry.name in RESERVED_DIRS:
                    continue
                parsed = parse_date_dir_name(entry.name)
                if parsed is not None:
                    found.append((parsed, Path(entry.path)))
    except OSError:
        return []
    return sorted(found, key=lambda item: item[0])


def discover_cameras(config: DashboardConfig) -> list[str]:
    """Union of configured points and any camera that has already written data."""
    names: set[str] = set()

    try:
        with config.config_file.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                candidate = sanitize_filename(row[0])
                if candidate and candidate not in {"name", "nama"}:
                    names.add(candidate)
    except OSError:
        pass

    try:
        with os.scandir(config.status_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith("_status.csv"):
                    names.add(entry.name[: -len("_status.csv")])
    except OSError:
        pass

    for _, day_dir in date_dirs(config.output_root):
        try:
            with os.scandir(day_dir) as entries:
                names.update(entry.name for entry in entries if entry.is_dir())
        except OSError:
            continue

    return sorted(names)


def _scan_counts(directory: Path, suffix: str) -> tuple[int, float]:
    """Return (file count, newest mtime) for one directory in a single pass."""
    count = 0
    newest = 0.0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                if not entry.name.lower().endswith(suffix):
                    continue
                count += 1
                try:
                    newest = max(newest, entry.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        return 0, 0.0
    return count, newest


def daily_archive_failed(config: DashboardConfig, day: date) -> bool:
    """Whether DailyArchiver gave up on this day's 7z after its retry budget."""
    marker = config.archives_dir / f".{day.isoformat()}.7z.failure.json"
    return marker.exists()


def camera_file_stats(config: DashboardConfig, camera: str) -> dict[str, float]:
    """Raw segment backlog and newest segment mtime for one camera.

    There is no local encoding step: raw ``.ts`` segments sit in ``videos/`` until
    the whole day is packaged into a daily 7z, so "backlog" now means "recorded but
    not yet in a completed daily archive" rather than "waiting to be encoded."
    """
    raw_backlog = 0
    newest = 0.0

    for _, day_dir in date_dirs(config.output_root):
        point_dir = day_dir / camera
        if not point_dir.is_dir():
            continue
        ts_count, ts_newest = _scan_counts(point_dir / "videos", ".ts")
        raw_backlog += ts_count
        newest = max(newest, ts_newest)

    return {
        "raw_backlog": float(raw_backlog),
        "newest_segment_epoch": newest,
    }


def classify_health(
    last_row: StatusRow | None, seconds_since_segment: float | None, stale_after: int
) -> str:
    """Combine the status CSV state with real segment freshness.

    The recorder writes ``online_preflight_ok`` once per FFmpeg start, so a
    healthy camera can stay silent for hours. Segment mtime is therefore the
    liveness signal, and the status row supplies the reason when it is not live.
    """
    recording = seconds_since_segment is not None and seconds_since_segment <= stale_after

    if last_row is None:
        return HEALTH_ONLINE if recording else HEALTH_UNKNOWN
    if last_row.status == "expired_url_escalated":
        return HEALTH_EXPIRED
    if last_row.status == "offline":
        return HEALTH_OFFLINE
    if last_row.status == "recorder_restart":
        return HEALTH_ONLINE if recording else HEALTH_RESTARTING
    if last_row.status in ONLINE_STATUSES:
        return HEALTH_ONLINE if recording else HEALTH_STALE
    return HEALTH_UNKNOWN


def build_camera_snapshot(config: DashboardConfig, camera: str, now: float) -> CameraSnapshot:
    rows = read_status_rows(config, camera)
    stats = camera_file_stats(config, camera)
    newest_epoch = stats["newest_segment_epoch"]

    seconds_since = int(now - newest_epoch) if newest_epoch > 0 else None
    last_row = rows[-1] if rows else None
    evidence_until = (
        newest_epoch + config.recording_stale_seconds if newest_epoch > 0 else now - 86400
    )

    snapshot = CameraSnapshot(
        name=camera,
        health=classify_health(last_row, seconds_since, config.recording_stale_seconds),
        raw_backlog=int(stats["raw_backlog"]),
        uptime_24h=uptime_ratio(rows, now - 86400, now, evidence_until=evidence_until),
        seconds_since_segment=seconds_since,
    )
    if newest_epoch > 0:
        snapshot.last_segment_at = datetime.fromtimestamp(newest_epoch).isoformat(
            timespec="seconds"
        )
    if last_row is not None:
        snapshot.status = last_row.status
        snapshot.reason = last_row.reason
        snapshot.http_status = last_row.http_status
        snapshot.url = last_row.url
        snapshot.last_status_at = last_row.timestamp
    return snapshot


def disk_snapshot(config: DashboardConfig) -> DiskSnapshot:
    try:
        usage = shutil.disk_usage(config.output_root)
    except OSError:
        return DiskSnapshot(min_free_gb=config.min_free_space_gb)
    free_gb = usage.free / (1024**3)
    return DiskSnapshot(
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        percent_used=round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
        free_gb=round(free_gb, 2),
        min_free_gb=config.min_free_space_gb,
        low_space=free_gb < config.min_free_space_gb,
    )


def list_archives(config: DashboardConfig, limit: int = 30) -> list[ArchiveEntry]:
    entries: list[ArchiveEntry] = []
    try:
        with os.scandir(config.archives_dir) as scan:
            for entry in scan:
                if not entry.is_file() or not entry.name.endswith(".7z"):
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                entries.append(
                    ArchiveEntry(
                        day=entry.name[: -len(".7z")],
                        size_bytes=stat.st_size,
                        modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(
                            timespec="seconds"
                        ),
                    )
                )
    except OSError:
        return []
    entries.sort(key=lambda item: item.day, reverse=True)
    return entries[:limit]


def count_daily_archive_failures(config: DashboardConfig) -> int:
    """Days DailyArchiver has given up on -- a dataset-wide alert, not per-camera."""
    count = 0
    try:
        with os.scandir(config.archives_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".7z.failure.json"):
                    count += 1
    except OSError:
        return 0
    return count


def _segment_start(camera: str, filename: str) -> datetime | None:
    """Parse the recorder's own ``{camera}_{YYYYMMDD}_{HHMMSS}.ts`` naming."""
    prefix = f"{camera}_"
    if not filename.startswith(prefix):
        return None
    stem = filename[len(prefix) :].rsplit(".", 1)[0]
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def compute_day_coverage(config: DashboardConfig, day: date, cameras: list[str]) -> DayCoverage:
    """Covered seconds per camera for one date, from raw segment filenames.

    There is no local archive-encoding step to read manifests from: each ``.ts``
    segment already represents up to ``SEGMENT_SECONDS`` of footage, so summing
    that per hour (capped at 3600s) gives the same "how much of this hour was
    actually recorded" answer without depending on anything downstream of the
    recorder itself.
    """
    coverage = DayCoverage(day=day.isoformat())
    day_dir = config.output_root / day.isoformat()
    segment_seconds = config.segment_seconds

    for camera in cameras:
        videos_dir = day_dir / camera / "videos"
        if not videos_dir.is_dir():
            continue

        hourly = [0] * 24
        segment_count = 0
        try:
            with os.scandir(videos_dir) as entries:
                for entry in entries:
                    if not entry.is_file() or entry.name.startswith("."):
                        continue
                    if not entry.name.lower().endswith(".ts"):
                        continue
                    started = _segment_start(camera, entry.name)
                    if started is None:
                        continue
                    segment_count += 1
                    hourly[started.hour] += segment_seconds
        except OSError:
            continue

        hourly = [min(seconds, 3600) for seconds in hourly]
        covered = sum(hourly)
        coverage.segment_count += segment_count
        if covered:
            coverage.per_camera[camera] = covered
            coverage.hourly[camera] = hourly

    coverage.archive_failed = daily_archive_failed(config, day)
    return coverage


def read_log_tail(
    path: Path, max_lines: int, level_filter: tuple[str, ...] | None = None
) -> list[str]:
    lines, _ = tail_lines(path, max_lines * 8, max_bytes=512_000)
    if level_filter:
        lines = [line for line in lines if any(token in line for token in level_filter)]
    return [scrub_text(line) for line in lines[-max_lines:]]


def recent_incidents(
    config: DashboardConfig, cameras: list[str], limit: int = 40
) -> list[StatusRow]:
    """Non-healthy status rows across all cameras, newest first."""
    incidents: list[StatusRow] = []
    for camera in cameras:
        for row in read_status_rows(config, camera, max_lines=400):
            if row.status not in ONLINE_STATUSES:
                incidents.append(row)
    incidents.sort(key=lambda row: row.epoch, reverse=True)
    return incidents[:limit]


def coverage_window(days: int, today: date | None = None) -> list[date]:
    anchor = today or datetime.now().date()
    return [anchor - timedelta(days=offset) for offset in range(days - 1, -1, -1)]


class TTLCache:
    """Tiny time-based cache so a browser refresh never re-walks the dataset."""

    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, tuple[float, Any]] = {}

    def get_or_set(self, key: str, producer: Callable[[], Any]) -> Any:
        now = time.monotonic()
        cached = self._entries.get(key)
        if cached is not None and now - cached[0] < self.ttl_seconds:
            return cached[1]
        value = producer()
        self._entries[key] = (now, value)
        return value

    def invalidate(self) -> None:
        self._entries.clear()

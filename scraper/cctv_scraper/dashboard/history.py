import json
import logging
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from cctv_scraper.dashboard import collectors
from cctv_scraper.dashboard.settings import DashboardConfig

SCHEMA = """
CREATE TABLE IF NOT EXISTS camera_samples (
    ts INTEGER NOT NULL,
    camera TEXT NOT NULL,
    health TEXT NOT NULL,
    online INTEGER NOT NULL,
    raw_backlog INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ts, camera)
);
CREATE INDEX IF NOT EXISTS idx_camera_samples_camera_ts ON camera_samples (camera, ts);

CREATE TABLE IF NOT EXISTS disk_samples (
    ts INTEGER PRIMARY KEY,
    total_bytes INTEGER NOT NULL,
    used_bytes INTEGER NOT NULL,
    free_bytes INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS coverage_daily (
    day TEXT NOT NULL,
    camera TEXT NOT NULL,
    covered_seconds INTEGER NOT NULL,
    hourly TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (day, camera)
);

CREATE TABLE IF NOT EXISTS coverage_meta (
    day TEXT PRIMARY KEY,
    computed_at INTEGER NOT NULL,
    finalized INTEGER NOT NULL DEFAULT 0,
    segment_count INTEGER NOT NULL DEFAULT 0,
    archive_failed INTEGER NOT NULL DEFAULT 0
);
"""


class HistoryStore:
    """SQLite roll-up of what the dataset looked like over time.

    The status CSVs and logs are pruned by ``RETENTION_DAYS`` and rotated by
    size, so long-range uptime and coverage would otherwise disappear. This
    store is the only thing the dashboard writes, and it lives in its own
    ``dashboard/`` subdirectory so the recorder's date-based cleanup never
    touches it.
    """

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.path = config.history_db
        self.logger = logging.getLogger("dashboard.history")
        self._write_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._reset_if_schema_changed(connection)
            connection.executescript(SCHEMA)

    def _reset_if_schema_changed(self, connection: sqlite3.Connection) -> None:
        """Drop tables from an older schema so ``CREATE TABLE IF NOT EXISTS`` can win.

        This store is a rebuildable cache of the dataset, not a system of record --
        every row here is re-derived from status CSVs and dataset files on the next
        sampling tick -- so dropping stale tables on a shape change is safe and
        simpler than a real migration.
        """
        expected = {
            "camera_samples": {"ts", "camera", "health", "online", "raw_backlog"},
            "coverage_meta": {"day", "computed_at", "finalized", "segment_count", "archive_failed"},
        }
        for table, columns in expected.items():
            existing = {
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if existing and existing != columns:
                self.logger.warning(
                    "History table %s has an outdated schema; rebuilding it.", table
                )
                connection.execute(f"DROP TABLE IF EXISTS {table}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        # WAL lets the sampler thread write while requests read.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    # ------------------------------------------------------------------ write

    def record_samples(
        self, snapshots: list[collectors.CameraSnapshot], disk: collectors.DiskSnapshot
    ) -> None:
        now = int(time.time())
        rows = [
            (
                now,
                snapshot.name,
                snapshot.health,
                1 if snapshot.health == collectors.HEALTH_ONLINE else 0,
                snapshot.raw_backlog,
            )
            for snapshot in snapshots
        ]
        with self._write_lock, self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO camera_samples "
                "(ts, camera, health, online, raw_backlog) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            if disk.total_bytes:
                connection.execute(
                    "INSERT OR REPLACE INTO disk_samples "
                    "(ts, total_bytes, used_bytes, free_bytes) VALUES (?, ?, ?, ?)",
                    (now, disk.total_bytes, disk.used_bytes, disk.free_bytes),
                )

    def store_coverage(self, coverage: collectors.DayCoverage, finalized: bool) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute("DELETE FROM coverage_daily WHERE day = ?", (coverage.day,))
            connection.executemany(
                "INSERT INTO coverage_daily (day, camera, covered_seconds, hourly) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        coverage.day,
                        camera,
                        seconds,
                        json.dumps(coverage.hourly.get(camera, [])),
                    )
                    for camera, seconds in coverage.per_camera.items()
                ],
            )
            connection.execute(
                "INSERT OR REPLACE INTO coverage_meta "
                "(day, computed_at, finalized, segment_count, archive_failed) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    coverage.day,
                    int(time.time()),
                    1 if finalized else 0,
                    coverage.segment_count,
                    1 if coverage.archive_failed else 0,
                ),
            )

    def prune(self) -> None:
        cutoff = int(time.time()) - self.config.history_retention_days * 86400
        cutoff_day = (
            datetime.now().date() - timedelta(days=self.config.history_retention_days)
        ).isoformat()
        with self._write_lock, self._connect() as connection:
            connection.execute("DELETE FROM camera_samples WHERE ts < ?", (cutoff,))
            connection.execute("DELETE FROM disk_samples WHERE ts < ?", (cutoff,))
            connection.execute("DELETE FROM coverage_daily WHERE day < ?", (cutoff_day,))
            connection.execute("DELETE FROM coverage_meta WHERE day < ?", (cutoff_day,))

    # ------------------------------------------------------------------- read

    def is_finalized(self, day: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT finalized FROM coverage_meta WHERE day = ?", (day,)
            ).fetchone()
        return bool(row and row["finalized"])

    def coverage_for_days(self, days: list[date]) -> dict[str, dict[str, Any]]:
        keys = [day.isoformat() for day in days]
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT day, camera, covered_seconds, hourly FROM coverage_daily "
                f"WHERE day IN ({placeholders})",
                keys,
            ).fetchall()
            meta_rows = connection.execute(
                f"SELECT day, segment_count, archive_failed FROM coverage_meta "
                f"WHERE day IN ({placeholders})",
                keys,
            ).fetchall()

        result: dict[str, dict[str, Any]] = {
            key: {
                "day": key,
                "per_camera": {},
                "hourly": {},
                "segment_count": 0,
                "archive_failed": False,
            }
            for key in keys
        }
        for row in rows:
            bucket = result[row["day"]]
            bucket["per_camera"][row["camera"]] = row["covered_seconds"]
            try:
                bucket["hourly"][row["camera"]] = json.loads(row["hourly"])
            except ValueError:
                bucket["hourly"][row["camera"]] = []
        for row in meta_rows:
            bucket = result[row["day"]]
            bucket["segment_count"] = row["segment_count"]
            bucket["archive_failed"] = bool(row["archive_failed"])
        return result

    def uptime_series(self, hours: int, buckets: int, camera: str | None = None) -> dict[str, Any]:
        """Mean online ratio per time bucket, for a sparkline or bar strip."""
        span = max(1, hours) * 3600
        bucket_seconds = max(60, span // max(1, buckets))
        start = int(time.time()) - span

        query = (
            "SELECT (ts / ?) AS bucket, AVG(online) AS ratio, MIN(ts) AS first_ts "
            "FROM camera_samples WHERE ts >= ?"
        )
        params: list[Any] = [bucket_seconds, start]
        if camera:
            query += " AND camera = ?"
            params.append(camera)
        query += " GROUP BY bucket ORDER BY bucket"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return {
            "bucket_seconds": bucket_seconds,
            "points": [
                {
                    "at": datetime.fromtimestamp(row["bucket"] * bucket_seconds).isoformat(
                        timespec="seconds"
                    ),
                    "ratio": round(float(row["ratio"]), 4),
                }
                for row in rows
            ],
        }

    def disk_series(self, hours: int, buckets: int = 96) -> list[dict[str, Any]]:
        span = max(1, hours) * 3600
        bucket_seconds = max(60, span // max(1, buckets))
        start = int(time.time()) - span
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT (ts / ?) AS bucket, AVG(free_bytes) AS free_bytes, "
                "AVG(used_bytes) AS used_bytes FROM disk_samples WHERE ts >= ? "
                "GROUP BY bucket ORDER BY bucket",
                (bucket_seconds, start),
            ).fetchall()
        return [
            {
                "at": datetime.fromtimestamp(row["bucket"] * bucket_seconds).isoformat(
                    timespec="seconds"
                ),
                "free_gb": round(float(row["free_bytes"]) / (1024**3), 2),
                "used_gb": round(float(row["used_bytes"]) / (1024**3), 2),
            }
            for row in rows
        ]

    def sample_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM camera_samples").fetchone()
        return int(row["total"]) if row else 0


class HistorySampler(threading.Thread):
    """Periodically snapshot camera health, disk, and coverage into SQLite."""

    def __init__(self, config: DashboardConfig, store: HistoryStore, stop_event: threading.Event):
        super().__init__(name="dashboard-history", daemon=True)
        self.config = config
        self.store = store
        self.stop_event = stop_event
        self.logger = logging.getLogger("dashboard.history")
        self._last_prune = 0.0

    def run(self) -> None:
        self.logger.info("Dashboard history sampler started (db=%s).", self.store.path)
        while not self.stop_event.is_set():
            try:
                self.sample_once()
            except Exception:
                self.logger.exception("History sampling failed; will retry next interval.")
            self.stop_event.wait(self.config.history_interval_seconds)
        self.logger.info("Dashboard history sampler stopped.")

    def sample_once(self) -> None:
        now = time.time()
        today = datetime.now().date()
        cameras = collectors.discover_cameras(self.config)
        snapshots = [
            collectors.build_camera_snapshot(self.config, camera, now) for camera in cameras
        ]
        self.store.record_samples(snapshots, collectors.disk_snapshot(self.config))
        self.refresh_coverage(cameras, today)

        if now - self._last_prune > 3600:
            self.store.prune()
            self._last_prune = now

    def refresh_coverage(self, cameras: list[str], today: date) -> None:
        """Recompute today, and backfill any past day not yet finalized.

        A past date's raw segments never change once the day is over, so each one
        is walked exactly once and then served from SQLite.
        """
        pending: list[date] = [today]
        for day, _ in collectors.date_dirs(self.config.output_root):
            if day < today and not self.store.is_finalized(day.isoformat()):
                pending.append(day)

        for day in pending[: 1 + 3]:  # today plus a few backfills per tick
            coverage = collectors.compute_day_coverage(self.config, day, cameras)
            self.store.store_coverage(coverage, finalized=day < today)


def path_is_writable(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / ".write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False

import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

from cctv_scraper.dashboard import collectors
from cctv_scraper.dashboard.history import HistoryStore
from cctv_scraper.dashboard.settings import DashboardConfig

SECONDS_PER_DAY = 86400
LOG_LEVEL_TOKENS = ("WARNING", "ERROR", "CRITICAL")


class DashboardService:
    """Assembles API payloads from the collectors, behind a short TTL cache."""

    def __init__(self, config: DashboardConfig, store: HistoryStore | None = None):
        self.config = config
        self.store = store
        self.cache = collectors.TTLCache(config.cache_ttl_seconds)

    # ---------------------------------------------------------------- helpers

    def cameras(self) -> list[str]:
        return self.cache.get_or_set("cameras", lambda: collectors.discover_cameras(self.config))

    def snapshots(self) -> list[collectors.CameraSnapshot]:
        def build() -> list[collectors.CameraSnapshot]:
            now = time.time()
            today = datetime.now().date()
            found = [
                collectors.build_camera_snapshot(self.config, camera, now)
                for camera in self.cameras()
            ]
            self._attach_today_coverage(found, today)
            return found

        return self.cache.get_or_set("snapshots", build)

    def _attach_today_coverage(
        self, snapshots: list[collectors.CameraSnapshot], today: Any
    ) -> None:
        per_camera = self.coverage_for_day(today)
        elapsed = max(
            1, int(time.time() - datetime.combine(today, datetime.min.time()).timestamp())
        )
        for snapshot in snapshots:
            covered = per_camera.get(snapshot.name, 0)
            snapshot.coverage_today_seconds = covered
            snapshot.coverage_today_ratio = round(min(1.0, covered / elapsed), 4)

    def coverage_for_day(self, day: Any) -> dict[str, int]:
        key = f"coverage:{day.isoformat()}"

        def build() -> dict[str, int]:
            if self.store is not None:
                stored = self.store.coverage_for_days([day]).get(day.isoformat())
                if stored and stored["per_camera"]:
                    return dict(stored["per_camera"])
            computed = collectors.compute_day_coverage(self.config, day, self.cameras())
            return dict(computed.per_camera)

        return self.cache.get_or_set(key, build)

    # --------------------------------------------------------------- payloads

    def overview(self) -> dict[str, Any]:
        snapshots = sorted(
            self.snapshots(),
            key=lambda item: (collectors.HEALTH_SEVERITY.get(item.health, 9), item.name),
        )
        disk = self.cache.get_or_set("disk", lambda: collectors.disk_snapshot(self.config))
        archives = self.cache.get_or_set("archives", lambda: collectors.list_archives(self.config))
        archive_failures = self.cache.get_or_set(
            "archive_failures", lambda: collectors.count_daily_archive_failures(self.config)
        )
        incidents = self.cache.get_or_set(
            "incidents", lambda: collectors.recent_incidents(self.config, self.cameras())
        )

        tracked = [item.uptime_24h for item in snapshots if item.uptime_24h is not None]
        coverage_ratios = [item.coverage_today_ratio for item in snapshots]

        counts: dict[str, int] = {}
        for snapshot in snapshots:
            counts[snapshot.health] = counts.get(snapshot.health, 0) + 1

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "history_enabled": self.store is not None,
            "summary": {
                "total": len(snapshots),
                "online": counts.get(collectors.HEALTH_ONLINE, 0),
                "stale": counts.get(collectors.HEALTH_STALE, 0),
                "restarting": counts.get(collectors.HEALTH_RESTARTING, 0),
                "offline": counts.get(collectors.HEALTH_OFFLINE, 0),
                "expired_url": counts.get(collectors.HEALTH_EXPIRED, 0),
                "unknown": counts.get(collectors.HEALTH_UNKNOWN, 0),
                "uptime_24h": round(sum(tracked) / len(tracked), 4) if tracked else None,
                "coverage_today_ratio": (
                    round(sum(coverage_ratios) / len(coverage_ratios), 4)
                    if coverage_ratios
                    else 0.0
                ),
                "raw_backlog": sum(item.raw_backlog for item in snapshots),
                "archive_failures": archive_failures,
            },
            "cameras": [asdict(item) for item in snapshots],
            "disk": asdict(disk),
            "archives": [asdict(item) for item in archives],
            "incidents": [asdict(item) for item in incidents],
        }

    def camera_detail(self, camera: str, days: int = 7) -> dict[str, Any]:
        snapshot = next((item for item in self.snapshots() if item.name == camera), None)
        if snapshot is None:
            snapshot = collectors.build_camera_snapshot(self.config, camera, time.time())

        rows = collectors.read_status_rows(self.config, camera)
        log_path = self.config.logs_dir / f"{camera}.log"
        window = collectors.coverage_window(days)

        if self.store is not None:
            stored = self.store.coverage_for_days(window)
            coverage = [
                {
                    "day": day.isoformat(),
                    "covered_seconds": stored[day.isoformat()]["per_camera"].get(camera, 0),
                    "hourly": stored[day.isoformat()]["hourly"].get(camera, []),
                }
                for day in window
            ]
        else:
            coverage = []
            for day in window:
                computed = collectors.compute_day_coverage(self.config, day, [camera])
                coverage.append(
                    {
                        "day": day.isoformat(),
                        "covered_seconds": computed.per_camera.get(camera, 0),
                        "hourly": computed.hourly.get(camera, []),
                    }
                )

        for entry in coverage:
            entry["ratio"] = round(min(1.0, entry["covered_seconds"] / SECONDS_PER_DAY), 4)

        return {
            "camera": asdict(snapshot),
            "status_rows": [asdict(row) for row in reversed(rows[-200:])],
            "log_tail": collectors.read_log_tail(log_path, self.config.log_tail_lines),
            "coverage": coverage,
            "uptime_series": (
                self.store.uptime_series(24, 96, camera=camera)
                if self.store is not None
                else {"bucket_seconds": 0, "points": []}
            ),
        }

    def coverage(self, days: int = 7) -> dict[str, Any]:
        window = collectors.coverage_window(days)
        cameras = self.cameras()

        if self.store is not None:
            stored = self.store.coverage_for_days(window)
        else:
            stored = {}
            for day in window:
                computed = collectors.compute_day_coverage(self.config, day, cameras)
                stored[day.isoformat()] = {
                    "per_camera": computed.per_camera,
                    "hourly": computed.hourly,
                    "archive_failed": computed.archive_failed,
                    "segment_count": computed.segment_count,
                }

        rows = []
        for day in window:
            key = day.isoformat()
            bucket = stored.get(key, {"per_camera": {}, "hourly": {}})
            per_camera = {
                camera: {
                    "covered_seconds": bucket["per_camera"].get(camera, 0),
                    "ratio": round(
                        min(1.0, bucket["per_camera"].get(camera, 0) / SECONDS_PER_DAY), 4
                    ),
                    "hourly": bucket["hourly"].get(camera, []),
                }
                for camera in cameras
            }
            total = sum(item["covered_seconds"] for item in per_camera.values())
            rows.append(
                {
                    "day": key,
                    "per_camera": per_camera,
                    "total_covered_seconds": total,
                    "ratio": (
                        round(min(1.0, total / (SECONDS_PER_DAY * len(cameras))), 4)
                        if cameras
                        else 0.0
                    ),
                    "archive_failed": bucket.get("archive_failed", False),
                }
            )

        return {"cameras": cameras, "days": rows}

    def app_log(self, lines: int | None = None) -> list[str]:
        return collectors.read_log_tail(
            self.config.logs_dir / "scraper.log",
            lines or self.config.log_tail_lines,
            level_filter=LOG_LEVEL_TOKENS,
        )

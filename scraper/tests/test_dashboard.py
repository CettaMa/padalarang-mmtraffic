import base64
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cctv_scraper.dashboard import collectors
from cctv_scraper.dashboard.api import create_app
from cctv_scraper.dashboard.history import HistorySampler, HistoryStore
from cctv_scraper.dashboard.security import mask_url, scrub_text
from cctv_scraper.dashboard.settings import (
    DashboardConfig,
    DashboardConfigError,
    load_dashboard_config,
)

SECRET_TOKEN = "7885b4db-a93f-4e0c-ada4-9f11ca8809e5"
SECRET_URL = f"https://atcs-dishubkbb.urbanaccess.net/memfs/{SECRET_TOKEN}.m3u8"
STATUS_HEADER = "timestamp,cctv_name,status,reason,http_status,url\n"


def make_config(tmp_path: Path, **overrides) -> DashboardConfig:
    root = tmp_path / "dataset"
    (root / "status").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "archives").mkdir(parents=True, exist_ok=True)
    defaults: dict[str, Any] = dict(
        output_root=root,
        config_file=tmp_path / "cctv_points.csv",
        host="127.0.0.1",
        port=8080,
        auth_enabled=True,
        username="admin",
        password="secret",
        cache_ttl_seconds=0.0,
        status_tail_lines=5000,
        log_tail_lines=50,
        history_enabled=False,
        history_db=tmp_path / "history.db",
        history_interval_seconds=60,
        history_retention_days=90,
        segment_seconds=60,
        min_free_space_gb=20.0,
    )
    defaults.update(overrides)
    return DashboardConfig(**defaults)


def write_status(config: DashboardConfig, camera: str, rows: list[tuple[str, str, str]]) -> None:
    path = config.status_dir / f"{camera}_status.csv"
    lines = [STATUS_HEADER]
    for timestamp, status, reason in rows:
        lines.append(f'{timestamp},{camera},{status},"{reason}",200,{SECRET_URL}\n')
    path.write_text("".join(lines), encoding="utf-8")


def write_segment(config: DashboardConfig, day: str, camera: str, at: datetime) -> None:
    """Write a raw .ts segment named the way the recorder itself names them.

    Coverage is computed straight from this filename convention now -- there is
    no archive-encoding step producing manifests to read instead.
    """
    videos = config.output_root / day / camera / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    path = videos / f"{camera}_{at:%Y%m%d_%H%M%S}.ts"
    path.write_bytes(b"raw")
    timestamp = at.timestamp()
    os.utime(path, (timestamp, timestamp))


# --------------------------------------------------------------- credentials


def test_mask_url_hides_the_memfs_token():
    masked = mask_url(SECRET_URL)
    assert SECRET_TOKEN not in masked
    # The host stays readable because DNS failures are diagnosed from it.
    assert "atcs-dishubkbb.urbanaccess.net" in masked
    assert masked.endswith(".m3u8")


def test_mask_url_drops_query_strings():
    masked = mask_url("https://example.invalid/live.m3u8?token=abcdefghijklmnop")
    assert "abcdefghijklmnop" not in masked


def test_scrub_text_cleans_tokens_quoted_inside_error_messages():
    reason = (
        "connection_error: HTTPSConnectionPool(host='atcs-dishubkbb.urbanaccess.net', port=443): "
        f"Max retries exceeded with url: /memfs/{SECRET_TOKEN}.m3u8 (Caused by NameResolutionError)"
    )
    assert SECRET_TOKEN not in scrub_text(reason)


def test_scrub_text_is_a_no_op_for_short_paths():
    assert scrub_text("wrote /app/dataset/logs/x.log") == "wrote /app/dataset/logs/x.log"


# ---------------------------------------------------------------- collectors


def test_tail_lines_drops_the_partial_first_record(tmp_path: Path):
    path = tmp_path / "big.csv"
    path.write_text("".join(f"line-{index}\n" for index in range(500)), encoding="utf-8")

    lines, from_start = collectors.tail_lines(path, max_lines=10, max_bytes=200)
    assert from_start is False
    assert len(lines) == 10
    assert all(line.startswith("line-") for line in lines)


def test_uptime_ratio_counts_only_online_spans():
    base = datetime(2026, 9, 1, 0, 0, 0)
    rows = collectors.parse_status_rows(
        [
            f"{base:%Y-%m-%d %H:%M:%S},cam,online_preflight_ok,ok,200,{SECRET_URL}",
            f"{base + timedelta(hours=6):%Y-%m-%d %H:%M:%S},cam,recorder_restart,down,,{SECRET_URL}",
            f"{base + timedelta(hours=12):%Y-%m-%d %H:%M:%S},cam,online_preflight_ok,ok,200,{SECRET_URL}",
        ],
        from_start=False,
    )
    window_start = base.timestamp()
    window_end = (base + timedelta(hours=24)).timestamp()

    # Online 0-6h and 12-24h => 18 of 24 hours.
    assert collectors.uptime_ratio(rows, window_start, window_end) == pytest.approx(0.75)


def test_uptime_ratio_stops_counting_when_segments_stop_arriving():
    base = datetime(2026, 9, 1, 0, 0, 0)
    rows = collectors.parse_status_rows(
        [f"{base:%Y-%m-%d %H:%M:%S},cam,online_preflight_ok,ok,200,{SECRET_URL}"],
        from_start=False,
    )
    window_start = base.timestamp()
    window_end = (base + timedelta(hours=24)).timestamp()

    # The recorder was killed after 6 hours and never wrote a closing row, so
    # the carried-forward "online" state must not become 100% uptime.
    evidence = (base + timedelta(hours=6)).timestamp()
    assert collectors.uptime_ratio(rows, window_start, window_end, evidence) == pytest.approx(0.25)
    assert collectors.uptime_ratio(rows, window_start, window_end) == pytest.approx(1.0)


def test_uptime_ratio_is_zero_when_evidence_predates_the_window():
    base = datetime(2026, 9, 1, 0, 0, 0)
    rows = collectors.parse_status_rows(
        [f"{base:%Y-%m-%d %H:%M:%S},cam,online_preflight_ok,ok,200,{SECRET_URL}"],
        from_start=False,
    )
    window_start = (base + timedelta(hours=48)).timestamp()
    window_end = (base + timedelta(hours=72)).timestamp()
    assert collectors.uptime_ratio(rows, window_start, window_end, base.timestamp()) == 0.0


def test_uptime_ratio_without_rows_is_unknown():
    assert collectors.uptime_ratio([], time.time() - 3600, time.time()) is None


def test_classify_health_prefers_segment_freshness_over_a_silent_status_file():
    row = collectors.parse_status_rows(
        [f"2026-09-01 00:00:00,cam,online_preflight_ok,ok,200,{SECRET_URL}"], from_start=False
    )[0]
    # The recorder writes this row once per FFmpeg start, so an hours-old row
    # with fresh segments still means the camera is healthy.
    assert collectors.classify_health(row, 30, 180) == collectors.HEALTH_ONLINE
    assert collectors.classify_health(row, 4000, 180) == collectors.HEALTH_STALE


def test_classify_health_escalates_expired_urls():
    row = collectors.parse_status_rows(
        [f"2026-09-01 00:00:00,cam,expired_url_escalated,EXPIRED_URL_ESCALATION,404,{SECRET_URL}"],
        from_start=False,
    )[0]
    assert collectors.classify_health(row, 10, 180) == collectors.HEALTH_EXPIRED


def test_compute_day_coverage_sums_raw_segment_durations(tmp_path: Path):
    config = make_config(tmp_path)
    write_segment(config, "2026-09-01", "cam_a", datetime(2026, 9, 1, 3, 0, 0))
    write_segment(config, "2026-09-01", "cam_a", datetime(2026, 9, 1, 3, 1, 0))
    write_segment(config, "2026-09-01", "cam_a", datetime(2026, 9, 1, 10, 0, 0))
    (config.output_root / "archives").mkdir(exist_ok=True)
    (config.output_root / "archives" / ".2026-09-01.7z.failure.json").write_text(
        "{}", encoding="utf-8"
    )

    coverage = collectors.compute_day_coverage(config, datetime(2026, 9, 1).date(), ["cam_a"])
    # Each segment is config.segment_seconds (60s here) of footage.
    assert coverage.per_camera["cam_a"] == 180
    assert coverage.segment_count == 3
    assert coverage.hourly["cam_a"][3] == 120
    assert coverage.hourly["cam_a"][10] == 60
    assert coverage.archive_failed is True


def test_discover_cameras_unions_config_and_dataset(tmp_path: Path):
    config = make_config(tmp_path)
    config.config_file.write_text(
        "name,url\nPadalarang Gedong 5,https://example.invalid/a.m3u8\n", encoding="utf-8"
    )
    write_status(config, "cam_from_status", [("2026-09-01 00:00:00", "offline", "boom")])
    (config.output_root / "2026-09-01" / "cam_from_dataset").mkdir(parents=True)

    found = collectors.discover_cameras(config)
    assert "padalarang_gedong_5" in found
    assert "cam_from_status" in found
    assert "cam_from_dataset" in found


# ------------------------------------------------------------------- history


def test_history_store_round_trips_samples_and_coverage(tmp_path: Path):
    config = make_config(tmp_path, history_enabled=True, history_db=tmp_path / "hist" / "h.db")
    store = HistoryStore(config)

    snapshot = collectors.CameraSnapshot(name="cam_a", health=collectors.HEALTH_ONLINE)
    store.record_samples(
        [snapshot], collectors.DiskSnapshot(total_bytes=100, used_bytes=40, free_bytes=60)
    )
    assert store.sample_count() == 1

    series = store.uptime_series(hours=1, buckets=12, camera="cam_a")
    assert series["points"] and series["points"][-1]["ratio"] == 1.0

    coverage = collectors.DayCoverage(day="2026-09-01", per_camera={"cam_a": 86000})
    store.store_coverage(coverage, finalized=True)
    assert store.is_finalized("2026-09-01") is True
    stored = store.coverage_for_days([datetime(2026, 9, 1).date()])
    assert stored["2026-09-01"]["per_camera"]["cam_a"] == 86000


def test_history_sampler_finalizes_past_days_once(tmp_path: Path):
    config = make_config(tmp_path, history_enabled=True, history_db=tmp_path / "hist" / "h.db")
    store = HistoryStore(config)
    yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
    write_segment(config, yesterday, "cam_a", datetime.now() - timedelta(days=1))

    sampler = HistorySampler(config, store, threading.Event())
    sampler.sample_once()

    assert store.is_finalized(yesterday) is True


# ----------------------------------------------------------------------- api


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    config = make_config(tmp_path)
    config.config_file.write_text(
        "name,url\nCam A,https://example.invalid/a.m3u8\n", encoding="utf-8"
    )
    write_status(
        config,
        "cam_a",
        [
            ("2026-09-01 00:00:00", "online_preflight_ok", "ok"),
            ("2026-09-01 06:00:00", "offline", f"connection_error /memfs/{SECRET_TOKEN}.m3u8"),
        ],
    )
    today = datetime.now().date()
    base = datetime.combine(today, datetime.min.time()) + timedelta(hours=12)
    for minute in range(5):
        write_segment(config, today.isoformat(), "cam_a", base + timedelta(minutes=minute))
    (config.logs_dir / "cam_a.log").write_text("2026-09-01 ERROR boom\n", encoding="utf-8")
    (config.logs_dir / "scraper.log").write_text("2026-09-01 WARNING low disk\n", encoding="utf-8")

    with TestClient(create_app(config, run_sampler=False)) as test_client:
        yield test_client


def auth_header(username: str = "admin", password: str = "secret") -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_healthz_needs_no_credentials(client: TestClient):
    assert client.get("/healthz").status_code == 200


def test_api_rejects_missing_and_wrong_credentials(client: TestClient):
    assert client.get("/api/overview").status_code == 401
    assert client.get("/api/overview", headers=auth_header(password="nope")).status_code == 401


def test_overview_returns_masked_cameras(client: TestClient):
    response = client.get("/api/overview", headers=auth_header())
    assert response.status_code == 200
    payload = response.json()

    assert payload["summary"]["total"] >= 1
    names = [camera["name"] for camera in payload["cameras"]]
    assert "cam_a" in names
    assert SECRET_TOKEN not in response.text


def test_camera_detail_rejects_unknown_names(client: TestClient):
    assert client.get("/api/cameras/nope", headers=auth_header()).status_code == 404
    assert client.get("/api/cameras/..%2F..%2Fetc", headers=auth_header()).status_code == 404


def test_camera_detail_masks_status_rows_and_logs(client: TestClient):
    response = client.get("/api/cameras/cam_a?days=3", headers=auth_header())
    assert response.status_code == 200
    payload = response.json()

    assert payload["camera"]["name"] == "cam_a"
    assert len(payload["coverage"]) == 3
    assert payload["log_tail"]
    assert SECRET_TOKEN not in response.text


def test_coverage_endpoint_reports_per_day_ratios(client: TestClient):
    payload = client.get("/api/coverage?days=2", headers=auth_header()).json()
    assert len(payload["days"]) == 2
    today = payload["days"][-1]
    assert today["per_camera"]["cam_a"]["covered_seconds"] == 300


def test_responses_are_not_cached(client: TestClient):
    response = client.get("/api/overview", headers=auth_header())
    assert response.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_index_is_served_behind_auth(client: TestClient):
    assert client.get("/").status_code == 401
    page = client.get("/", headers=auth_header())
    assert page.status_code == 200
    assert "CCTV" in page.text


# -------------------------------------------------------------------- config


def test_auth_without_a_password_refuses_to_start(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "true")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "")
    with pytest.raises(DashboardConfigError):
        load_dashboard_config()


def test_disabling_auth_allows_an_empty_password(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_ENABLED", "false")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "")
    assert load_dashboard_config().auth_enabled is False

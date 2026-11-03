import socket
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from cctv_scraper.config import CCTVPoint
from cctv_scraper.doh import DoHResolver
from cctv_scraper.metadata import MetadataCollector
from tests.test_characterization import make_dummy_config


def test_doh_negative_caching():
    DoHResolver._cache.clear()

    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443))],
    ) as mock_gai:
        # First call: system DNS succeeds -> returns None and caches negative result
        res1 = DoHResolver.resolve("example.com")
        assert res1 is None
        assert mock_gai.call_count == 1
        assert "example.com" in DoHResolver._cache
        assert DoHResolver._cache["example.com"][0] is None

        # Second call within TTL: should return None from cache WITHOUT calling getaddrinfo again
        res2 = DoHResolver.resolve("example.com")
        assert res2 is None
        assert mock_gai.call_count == 1  # Not incremented!


def test_doh_positive_caching():
    DoHResolver._cache.clear()

    # System DNS fails, Cloudflare DoH succeeds
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"Answer": [{"type": 1, "data": "104.21.5.6"}]}

    with (
        patch("socket.getaddrinfo", side_effect=socket.gaierror),
        patch("requests.get", return_value=mock_resp) as mock_req,
    ):
        ip1 = DoHResolver.resolve("blocked-site.test")
        assert ip1 == "104.21.5.6"
        assert mock_req.call_count == 1

        # Second call within TTL returns cached IP without HTTP request
        ip2 = DoHResolver.resolve("blocked-site.test")
        assert ip2 == "104.21.5.6"
        assert mock_req.call_count == 1


def test_doh_cache_fifo_eviction():
    DoHResolver._cache.clear()
    orig_max = DoHResolver._max_cache_size
    DoHResolver._max_cache_size = 3
    try:
        DoHResolver._set_cache("host1.test", "1.1.1.1")
        DoHResolver._set_cache("host2.test", "2.2.2.2")
        DoHResolver._set_cache("host3.test", "3.3.3.3")
        assert len(DoHResolver._cache) == 3

        # Adding 4th host evicts host1.test
        DoHResolver._set_cache("host4.test", "4.4.4.4")
        assert len(DoHResolver._cache) == 3
        assert "host1.test" not in DoHResolver._cache
        assert "host4.test" in DoHResolver._cache
    finally:
        DoHResolver._max_cache_size = orig_max
        DoHResolver._cache.clear()


def test_openmeteo_batch_maps_locations_and_uses_one_request(tmp_path: Path):
    points = [
        CCTVPoint(name="point_a", url="http://example.com/a.m3u8", lat=-6.85, lon=107.5),
        CCTVPoint(name="point_b", url="http://example.com/b.m3u8", lat=-6.86, lon=107.51),
    ]
    cfg = make_dummy_config(tmp_path, openmeteo_interval_seconds=60)
    collector = MetadataCollector(points, cfg, threading.Event())
    session = requests.Session()
    response = MagicMock()
    response.json.return_value = [
        {"current": {"temperature_2m": 27.5}},
        {"current": {"temperature_2m": 28.5}},
    ]
    response.raise_for_status = MagicMock()

    with patch.object(session, "get", return_value=response) as mock_get:
        first = collector.get_openmeteo_cached_batch(session)
        second = collector.get_openmeteo_cached_batch(session)

    assert mock_get.call_count == 1
    url = mock_get.call_args.args[0]
    assert "latitude=-6.85,-6.86" in url
    assert first["point_a"]["weather_temp"] == pytest.approx(27.5)
    assert first["point_b"]["weather_temp"] == pytest.approx(28.5)
    assert first["point_a"]["weather_cache_status"] == "fresh"
    assert second["point_a"]["weather_cache_status"] == "cached"


def test_metadata_failure_backoff_increases_between_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    point = CCTVPoint(name="backoff_point", url="http://example.com/a.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(
        tmp_path,
        metadata_interval_seconds=1,
        metadata_min_pass_interval_seconds=5,
        metadata_failure_backoff_base_seconds=5,
        metadata_failure_backoff_max_seconds=60,
    )
    stop_event = threading.Event()
    waits: list[float] = []

    def stop_after_two_passes(timeout: float) -> bool:
        waits.append(timeout)
        if len(waits) >= 2:
            stop_event.set()
        return True

    monkeypatch.setattr(stop_event, "wait", stop_after_two_passes)
    collector = MetadataCollector([point], cfg, stop_event)

    def failed_weather(session: requests.Session) -> dict[str, dict]:
        collector._record_request(True)
        return {point.name: {"weather_cache_status": "fresh_fallback"}}

    monkeypatch.setattr(collector, "get_openmeteo_cached_batch", failed_weather)
    monkeypatch.setattr(collector, "get_tomtom_cached", lambda session, item: {})
    monkeypatch.setattr(collector, "write_metadata", lambda item, row: None)
    collector.run()

    assert len(waits) == 2
    assert waits[0] >= 5
    assert waits[1] >= 10


def test_metadata_collector_session_and_caching(tmp_path: Path):
    point = CCTVPoint(name="padalarang", url="http://example.com/live.m3u8", lat=-6.85, lon=107.5)
    cfg = make_dummy_config(
        tmp_path,
        tomtom_api_key="dummy_key",
        tomtom_interval_seconds=300,
        openmeteo_interval_seconds=60,
    )
    stop_event = threading.Event()
    collector = MetadataCollector([point], cfg, stop_event)

    session = requests.Session()

    mock_tomtom_resp = MagicMock()
    mock_tomtom_resp.json.return_value = {
        "flowSegmentData": {
            "currentSpeed": 45,
            "freeFlowSpeed": 50,
            "confidence": 0.9,
            "roadClosure": False,
        }
    }
    mock_tomtom_resp.raise_for_status = MagicMock()

    mock_weather_resp = MagicMock()
    mock_weather_resp.json.return_value = {
        "current": {
            "temperature_2m": 27.5,
            "relative_humidity_2m": 80,
            "rain": 0.0,
            "wind_speed_10m": 5.2,
        }
    }
    mock_weather_resp.raise_for_status = MagicMock()

    with patch.object(session, "get") as mock_get:

        def fake_get(url, *args, **kwargs):
            if "tomtom" in url:
                return mock_tomtom_resp
            return mock_weather_resp

        mock_get.side_effect = fake_get

        ts1 = datetime(2026, 6, 21, 10, 0, 0)
        row1 = collector.collect_point_metadata(session, point, ts1)

        assert row1["traffic_cache_status"] == "fresh"
        assert row1["weather_cache_status"] == "fresh"
        assert row1["traffic_speed"] == 45
        assert row1["weather_temp"] == pytest.approx(27.5)
        assert mock_get.call_count == 2

        # Immediate second call (within cache window) should use cached values
        ts2 = datetime(2026, 6, 21, 10, 0, 30)
        row2 = collector.collect_point_metadata(session, point, ts2)

        assert row2["traffic_cache_status"] == "cached"
        assert row2["weather_cache_status"] == "cached"
        assert row2["traffic_speed"] == 45
        assert row2["weather_temp"] == pytest.approx(27.5)
        assert mock_get.call_count == 2  # No new network requests made!


def test_get_tomtom_redacts_the_api_key_from_a_logged_connection_error(tmp_path, caplog):
    """requests/urllib3 embed the full request URL -- including ?key=... -- in a
    connection error's own message, regardless of how the URL was built. The only
    reliable fix is scrubbing the known secret value before it reaches the logger."""
    secret = "sk-live-tomtom-1234567890"
    config = make_dummy_config(tmp_path, tomtom_api_key=secret)
    point = CCTVPoint(name="cam", url="https://example.invalid/live.m3u8", lat=1.0, lon=2.0)
    collector = MetadataCollector([point], config, threading.Event())

    session = requests.Session()
    error = requests.exceptions.ConnectionError(
        f"HTTPSConnectionPool(host='api.tomtom.com', port=443): Max retries exceeded with url: "
        f"/traffic/services/4/flowSegmentData/absolute/10/json?key={secret}&point=1,2"
    )
    with patch.object(session, "get", side_effect=error), caplog.at_level("WARNING"):
        result = collector.get_tomtom(session, point)

    assert result["traffic_status"] == "error"
    assert secret not in caplog.text
    assert "****" in caplog.text

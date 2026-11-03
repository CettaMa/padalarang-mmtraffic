import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from cctv_scraper.config import CCTVPoint
from cctv_scraper.netwatch import NetworkWatchdog
from cctv_scraper.recorder import CCTVRecorder
from tests.test_characterization import make_dummy_config


def make_recorder(tmp_path: Path, name: str, config) -> CCTVRecorder:
    point = CCTVPoint(name=name, url="https://example.com/a.m3u8", lat=0.0, lon=0.0)
    return CCTVRecorder(point, config, threading.Event())


def test_keep_alive_flags_default_to_on(tmp_path):
    config = make_dummy_config(tmp_path)
    recorder = make_recorder(tmp_path, "cam", config)
    command = recorder.build_ffmpeg_command()

    assert command[command.index("-http_persistent") + 1] == "1"
    assert command[command.index("-multiple_requests") + 1] == "1"


def test_keep_alive_flags_can_be_disabled(tmp_path):
    config = make_dummy_config(
        tmp_path, ffmpeg_http_persistent=False, ffmpeg_multiple_requests=False
    )
    recorder = make_recorder(tmp_path, "cam", config)
    command = recorder.build_ffmpeg_command()

    assert command[command.index("-http_persistent") + 1] == "0"
    assert command[command.index("-multiple_requests") + 1] == "0"


def test_http_error_response_still_counts_as_reachable(tmp_path):
    """A 404 means the URL expired, not that the host lost its network."""
    config = make_dummy_config(tmp_path)
    recorder = make_recorder(tmp_path, "cam", config)
    recorder.last_reachable_at = time.monotonic() - 5000

    response = MagicMock()
    response.status_code = 404
    session = MagicMock()
    session.get.return_value.__enter__.return_value = response

    with patch("requests.Session") as mock_session:
        mock_session.return_value.__enter__.return_value = session
        ok, reason, http_status = recorder.preflight_stream()

    assert ok is False
    assert reason == "not_found_or_expired_url"
    assert time.monotonic() - recorder.last_reachable_at < 5


def test_watchdog_fires_only_when_every_recorder_is_unreachable(tmp_path):
    config = make_dummy_config(tmp_path, network_watchdog_seconds=600)
    stop_event = threading.Event()
    stale = make_recorder(tmp_path, "stale", config)
    fresh = make_recorder(tmp_path, "fresh", config)
    stale.last_reachable_at = time.monotonic() - 900
    fresh.last_reachable_at = time.monotonic()

    on_unreachable = MagicMock()
    watchdog = NetworkWatchdog(
        [stale, fresh], config, stop_event, on_unreachable, check_interval_seconds=0
    )

    # One healthy camera is enough: the network is up, that stream is not.
    assert watchdog.seconds_since_any_reachable() < 600

    fresh.last_reachable_at = time.monotonic() - 700
    assert watchdog.seconds_since_any_reachable() >= 600

    watchdog.run()
    on_unreachable.assert_called_once()


def test_watchdog_disabled_when_threshold_is_zero(tmp_path):
    config = make_dummy_config(tmp_path, network_watchdog_seconds=0)
    recorder = make_recorder(tmp_path, "cam", config)
    recorder.last_reachable_at = time.monotonic() - 99999

    on_unreachable = MagicMock()
    watchdog = NetworkWatchdog(
        [recorder], config, threading.Event(), on_unreachable, check_interval_seconds=0
    )
    watchdog.run()

    on_unreachable.assert_not_called()

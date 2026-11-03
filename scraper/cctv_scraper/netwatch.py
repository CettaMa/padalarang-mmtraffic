import logging
import threading
import time
from collections.abc import Callable, Sequence

from cctv_scraper.config import RuntimeConfig
from cctv_scraper.recorder import CCTVRecorder


class NetworkWatchdog(threading.Thread):
    """Restart the process when the host itself loses outbound connectivity.

    A single camera going dark is normal and the recorder handles it. Every
    camera failing to reach its server at the same time is not a stream
    problem -- it means this host can no longer open outbound TCP connections
    (a full conntrack/NAT table does exactly that, and does not drain on its
    own). The recorders would then retry forever against a dead network, so
    the watchdog asks the supervisor for a fresh process instead: restarting
    closes every socket the old process held.

    Only a preflight that received no answer at all counts as unreachable. An
    HTTP 404 on an expired URL still proves the network works, so a batch of
    rotated URLs never triggers a restart.
    """

    def __init__(
        self,
        recorders: Sequence[CCTVRecorder],
        config: RuntimeConfig,
        stop_event: threading.Event,
        on_unreachable: Callable[[], None],
        check_interval_seconds: int = 30,
    ):
        super().__init__(name="network-watchdog", daemon=True)
        self.recorders = list(recorders)
        self.config = config
        self.stop_event = stop_event
        self.on_unreachable = on_unreachable
        self.check_interval_seconds = check_interval_seconds
        self.logger = logging.getLogger("network-watchdog")

    @property
    def threshold_seconds(self) -> int:
        return self.config.network.network_watchdog_seconds

    def seconds_since_any_reachable(self) -> float:
        """Age of the most recent successful contact across all recorders."""
        newest = max(recorder.last_reachable_at for recorder in self.recorders)
        return time.monotonic() - newest

    def run(self) -> None:
        if self.threshold_seconds <= 0:
            self.logger.info("Network watchdog disabled.")
            return
        if not self.recorders:
            self.logger.info("Network watchdog has no recorders to observe.")
            return

        self.logger.info(
            "Network watchdog started. Restart after %s seconds without any reachable stream.",
            self.threshold_seconds,
        )

        while not self.stop_event.is_set():
            self.stop_event.wait(self.check_interval_seconds)
            if self.stop_event.is_set():
                break

            offline_for = self.seconds_since_any_reachable()
            if offline_for < self.threshold_seconds:
                continue

            self.logger.error(
                "[NETWORK_DEAD] No CCTV point has been reachable for %.0f seconds "
                "(threshold %s). The host has lost outbound connectivity; "
                "requesting a process restart to drop every stale socket.",
                offline_for,
                self.threshold_seconds,
            )
            self.on_unreachable()
            return

        self.logger.info("Network watchdog stopped.")

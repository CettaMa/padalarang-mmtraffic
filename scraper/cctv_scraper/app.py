import logging
import os
import signal
import subprocess
import threading
import time

# Ensure DoH patch is loaded
import cctv_scraper.doh  # noqa: F401
from cctv_scraper.archive import DailyArchiver
from cctv_scraper.config import RuntimeConfig, load_cctv_points
from cctv_scraper.disk import DiskMonitor
from cctv_scraper.logging_setup import setup_logging
from cctv_scraper.metadata import MetadataCollector
from cctv_scraper.netwatch import NetworkWatchdog
from cctv_scraper.recorder import CCTVRecorder


def validate_video_encoder(config: RuntimeConfig) -> RuntimeConfig:
    """Fail fast if the recorder's own encoder is not compiled into FFmpeg.

    Only relevant when FFMPEG_TRANSPORT_MODE is "smooth" or "transcode" -- the
    default "copy" mode never invokes an encoder, so nothing to check there.
    There is no local archive-encoding step anymore: raw segments go straight
    into the daily 7z package, so there is no hardware/software fallback to
    resolve here either. The returned config is unchanged; it stays a function
    (rather than a plain check) so callers do not need to know that.
    """
    if config.recorder.ffmpeg_transport_mode not in {"smooth", "transcode"}:
        return config

    required_encoder = config.recorder.video_encoder

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg tidak ditemukan pada PATH.") from exc

    if required_encoder not in result.stdout:
        raise RuntimeError(
            f"Encoder FFmpeg tidak tersedia: {required_encoder}. "
            "Periksa VIDEO_ENCODER, atau gunakan FFMPEG_TRANSPORT_MODE=copy."
        )

    return config


class CCTVApp:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.exit_code = 0

    def start(self) -> None:
        setup_logging(
            self.config.storage.output_root,
            self.config.storage.log_max_bytes,
            self.config.storage.log_backup_count,
        )
        self.config = validate_video_encoder(self.config)

        logging.info("Loading CCTV points from: %s", self.config.config_file)
        points = load_cctv_points(self.config)

        logging.info("CCTV 24x7 Scraper starting.")
        logging.info("Total CCTV points: %s", len(points))
        logging.info("Output root: %s", self.config.storage.output_root.resolve())
        logging.info("Segment duration: %s seconds", self.config.recorder.segment_seconds)
        logging.info("Video container: %s", self.config.recorder.video_container)
        logging.info("FFmpeg transport mode: %s", self.config.recorder.ffmpeg_transport_mode)
        logging.info("FFmpeg referer: %s", self.config.recorder.ffmpeg_referer or "-")
        logging.info("FFmpeg origin: %s", self.config.recorder.ffmpeg_origin or "-")
        logging.info("HLS reconnect at EOF: %s", self.config.recorder.hls_reconnect_at_eof)
        logging.info("Segment at clock time: %s", self.config.recorder.segment_atclocktime)
        logging.info("HLS live start index: %s", self.config.recorder.hls_live_start_index)
        logging.info(
            "FFmpeg reconnect on HTTP error: %s",
            self.config.recorder.ffmpeg_reconnect_on_http_error,
        )
        logging.info("Output FPS: %s", self.config.recorder.output_fps)
        logging.info("Transcode preset: %s", self.config.recorder.transcode_preset)
        logging.info("Video encoder: %s", self.config.recorder.video_encoder)
        logging.info("Target bitrate: %s", self.config.recorder.target_bitrate)
        logging.info("Maximum bitrate: %s", self.config.recorder.max_bitrate)
        logging.info("Buffer size: %s", self.config.recorder.buffer_size)
        logging.info("Output height: %s", self.config.recorder.output_height or "source")
        logging.info("Daily archive safe age: %s seconds", self.config.archive.safe_age_seconds)
        logging.info("Daily 7-Zip archive enabled: %s", self.config.archive.daily_archive_enabled)
        logging.info(
            "Daily 7-Zip archive scan interval: %s seconds",
            self.config.archive.daily_archive_scan_seconds,
        )
        logging.info(
            "Daily 7-Zip delete source after archiving: %s",
            self.config.archive.daily_archive_delete_source,
        )
        logging.info("Daily 7-Zip archiver binary: %s", self.config.archive.archiver_binary)
        logging.info("Preflight check: %s", self.config.network.preflight_check)
        logging.info("Offline retry seconds: %s", self.config.network.offline_retry_seconds)
        logging.info("Network retry seconds: %s", self.config.network.network_retry_seconds)
        logging.info(
            "Network watchdog seconds: %s",
            self.config.network.network_watchdog_seconds or "disabled",
        )
        logging.info(
            "FFmpeg HTTP keep-alive: persistent=%s multiple_requests=%s",
            self.config.recorder.ffmpeg_http_persistent,
            self.config.recorder.ffmpeg_multiple_requests,
        )
        logging.info(
            "Metadata CSV write interval: %s seconds",
            self.config.metadata.metadata_interval_seconds,
        )
        logging.info(
            "TomTom API interval: %s seconds", self.config.metadata.tomtom_interval_seconds
        )
        logging.info(
            "Open-Meteo API interval: %s seconds",
            self.config.metadata.openmeteo_interval_seconds,
        )
        logging.info(
            "Metadata minimum pass interval: %s seconds",
            self.config.metadata.min_pass_interval_seconds,
        )
        logging.info(
            "Expired URL escalation threshold: %s failures",
            self.config.network.expired_url_escalation_threshold,
        )
        logging.info("Retention days: %s", self.config.storage.retention_days)
        logging.info("Minimum free disk: %.2f GB", self.config.storage.min_free_space_gb)
        logging.info("Log max bytes: %s", self.config.storage.log_max_bytes)
        logging.info("Log backup count: %s", self.config.storage.log_backup_count)

        if not self.config.metadata.tomtom_api_key:
            logging.warning("TOMTOM_API not found. TomTom metadata will be empty.")

        fallback_points = [
            p.name
            for p in points
            if p.lat == self.config.metadata.default_lat
            and p.lon == self.config.metadata.default_lon
        ]
        if fallback_points:
            logging.warning(
                "Some CCTV points use fallback coordinates. Update cctv_points.csv for accuracy: %s",
                ", ".join(fallback_points),
            )

        recorders: list[CCTVRecorder] = []
        for point in points:
            recorder = CCTVRecorder(point, self.config, self.stop_event)
            recorders.append(recorder)
            self.threads.append(recorder)
            recorder.start()

        watchdog = NetworkWatchdog(recorders, self.config, self.stop_event, self.request_restart)
        self.threads.append(watchdog)
        watchdog.start()

        metadata = MetadataCollector(points, self.config, self.stop_event)
        self.threads.append(metadata)
        metadata.start()

        daily_archiver = DailyArchiver(self.config, self.stop_event)
        self.threads.append(daily_archiver)
        daily_archiver.start()

        disk = DiskMonitor(self.config, self.stop_event)
        self.threads.append(disk)
        disk.start()

        self.wait_forever()

    def request_restart(self) -> None:
        """Stop with a non-zero code so the supervisor starts a fresh process.

        Docker's restart policy brings the container back, and the new process
        gets a clean socket table -- which is the whole point of restarting.
        """
        self.exit_code = 1
        self.stop_event.set()

    def wait_forever(self) -> None:
        def request_stop(signum=None, frame=None):
            logging.info("Stop signal received.")
            self.stop_event.set()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

        while not self.stop_event.is_set():
            time.sleep(1)

        logging.info("Stopping all workers.")
        for thread in self.threads:
            thread.join(timeout=30)

        logging.info("CCTV 24x7 Scraper stopped.")

        if self.exit_code != 0:
            # Threads are daemons and FFmpeg children are already signalled;
            # exit hard so a hung join can never keep a dead process alive.
            logging.error("Exiting with code %s to force a restart.", self.exit_code)
            logging.shutdown()
            os._exit(self.exit_code)

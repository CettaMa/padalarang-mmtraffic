import csv
import os
import subprocess
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import IO

import requests

from cctv_scraper.config import CCTVPoint, RuntimeConfig, ensure_dir, now_local
from cctv_scraper.logging_setup import point_logger


class CCTVRecorder(threading.Thread):
    def __init__(self, point: CCTVPoint, config: RuntimeConfig, stop_event: threading.Event):
        super().__init__(name=f"recorder-{point.name}", daemon=True)
        self.point = point
        self.config = config
        self.stop_event = stop_event
        self.process: subprocess.Popen | None = None
        self.logger = point_logger(
            config.storage.output_root,
            point.name,
            config.storage.log_max_bytes,
            config.storage.log_backup_count,
        )
        self.last_restart_at: datetime | None = None
        self.process_start_date: date | None = None
        self.ffmpeg_stderr_file: IO[str] | None = None
        self._ffmpeg_self_exited = False
        self.consecutive_expiry_failures = 0
        self.expiry_url_escalated = False
        # Monotonic clock: the NetworkWatchdog reads this to tell "the stream is
        # gone" apart from "this host cannot reach anything at all". Seeded at
        # construction so a recorder that never came up still ages normally.
        self.last_reachable_at = time.monotonic()

    def run(self) -> None:
        self.logger.info("Recorder watchdog started.")

        while not self.stop_event.is_set():
            ok, reason, http_status = self.preflight_stream()
            if not ok:
                self.handle_preflight_failure(reason, http_status)
                self.sleep_after_preflight_failure(reason)
                continue

            self.consecutive_expiry_failures = 0
            self.expiry_url_escalated = False
            self.write_status("online_preflight_ok", reason=reason, http_status=http_status)
            self._ffmpeg_self_exited = False
            self.start_ffmpeg()
            self.monitor_ffmpeg()

            if not self.stop_event.is_set():
                self.write_status("recorder_restart", reason="ffmpeg_stopped_or_unhealthy")
                if self._ffmpeg_self_exited:
                    # FFmpeg drop sendiri (biasanya TLS/network). Jeda singkat lalu preflight ulang.
                    self.logger.warning(
                        "FFmpeg dropped on its own. Short delay then re-preflight before restart."
                    )
                    self.stop_event.wait(self.config.recorder.restart_delay_seconds)
                    # Loop kembali ke preflight — tidak langsung spawn FFmpeg.
                else:
                    self.logger.warning(
                        "FFmpeg stopped or unhealthy. Restarting in %s seconds.",
                        self.config.recorder.restart_delay_seconds,
                    )
                    self.stop_event.wait(self.config.recorder.restart_delay_seconds)

        self.stop_ffmpeg()
        self.logger.info("Recorder watchdog stopped.")

    def current_date_folder(self) -> str:
        return now_local().strftime("%Y-%m-%d")

    def current_video_dir(self) -> Path:
        video_dir = (
            self.config.storage.output_root
            / self.current_date_folder()
            / self.point.name
            / "videos"
        )
        ensure_dir(video_dir)
        return video_dir

    def stderr_path(self) -> Path:
        ensure_dir(self.config.storage.output_root / "logs" / "ffmpeg")
        date_str = now_local().strftime("%Y-%m-%d")
        return (
            self.config.storage.output_root
            / "logs"
            / "ffmpeg"
            / f"{self.point.name}_{date_str}.ffmpeg.log"
        )

    def output_extension(self) -> str:
        return "ts" if self.config.recorder.video_container == "ts" else "mp4"

    def build_output_pattern(self) -> str:
        video_dir = self.current_video_dir()
        ext = self.output_extension()
        return str(video_dir / f"{self.point.name}_%Y%m%d_%H%M%S.{ext}")

    def input_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.config.recorder.ffmpeg_user_agent,
        }
        if self.config.recorder.ffmpeg_referer:
            headers["Referer"] = self.config.recorder.ffmpeg_referer
        if self.config.recorder.ffmpeg_origin:
            headers["Origin"] = self.config.recorder.ffmpeg_origin
        return headers

    def ffmpeg_headers_arg(self) -> str:
        return "".join(f"{key}: {value}\r\n" for key, value in self.input_headers().items())

    def build_ffmpeg_command(self) -> list[str]:
        output_pattern = self.build_output_pattern()

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            self.config.recorder.ffmpeg_loglevel,
            # Input stability for unstable HLS CCTV streams.
            "-fflags",
            "+genpts+discardcorrupt+nobuffer",
            "-err_detect",
            "ignore_err",
            "-rw_timeout",
            self.config.recorder.ffmpeg_rw_timeout,
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_on_network_error",
            "1",
            "-reconnect_on_http_error",
            self.config.recorder.ffmpeg_reconnect_on_http_error,
            "-reconnect_delay_max",
            self.config.recorder.ffmpeg_reconnect_delay_max,
            "-http_persistent",
            "1" if self.config.recorder.ffmpeg_http_persistent else "0",
            "-multiple_requests",
            "1" if self.config.recorder.ffmpeg_multiple_requests else "0",
            "-user_agent",
            self.config.recorder.ffmpeg_user_agent,
            "-headers",
            self.ffmpeg_headers_arg(),
            "-analyzeduration",
            self.config.recorder.ffmpeg_analyzeduration,
            "-probesize",
            self.config.recorder.ffmpeg_probesize,
        ]

        if self.config.recorder.hls_reconnect_at_eof:
            cmd += ["-reconnect_at_eof", "1"]

        # Start from newest live HLS segment. This reduces "expired from playlists"
        # and short/failed recordings caused by trying to fetch old .ts fragments.
        if self.point.url.lower().split("?")[0].endswith(".m3u8"):
            cmd += ["-live_start_index", self.config.recorder.hls_live_start_index]

        cmd += ["-i", self.point.url]

        if self.config.recorder.ffmpeg_transport_mode in {"transcode", "smooth"}:
            # CFR mengatasi timestamp stream sumber yang tidak stabil.
            # Bitrate-limited encoding menjaga kebutuhan storage dapat diprediksi.
            fps = max(1, int(self.config.recorder.output_fps))
            gop = max(1, fps * max(1, int(self.config.recorder.segment_keyframe_seconds)))

            filters = [f"fps={fps}"]
            if self.config.recorder.output_height > 0:
                filters.append(f"scale=-2:{self.config.recorder.output_height}")
            filters.append("format=yuv420p")

            encoder = self.config.recorder.video_encoder
            cmd += [
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                ",".join(filters),
                "-fps_mode",
                "cfr",
                "-c:v",
                encoder,
            ]

            if encoder in {"h264_nvenc", "hevc_nvenc"}:
                cmd += [
                    "-preset",
                    self.config.recorder.transcode_preset,
                    "-rc:v",
                    "vbr",
                    "-b:v",
                    self.config.recorder.target_bitrate,
                    "-maxrate:v",
                    self.config.recorder.max_bitrate,
                    "-bufsize:v",
                    self.config.recorder.buffer_size,
                ]
            elif encoder in {"libx265", "libx264"}:
                cpu_preset = self.config.recorder.transcode_preset
                if cpu_preset.startswith("p") and cpu_preset[1:].isdigit():
                    cpu_preset = "veryfast"
                cmd += [
                    "-preset",
                    cpu_preset,
                    "-b:v",
                    self.config.recorder.target_bitrate,
                    "-maxrate:v",
                    self.config.recorder.max_bitrate,
                    "-bufsize:v",
                    self.config.recorder.buffer_size,
                ]
            else:
                raise ValueError(
                    f"VIDEO_ENCODER tidak didukung: {encoder}. "
                    "Gunakan hevc_nvenc, h264_nvenc, libx265, atau libx264."
                )

            cmd += [
                "-g",
                str(gop),
                "-keyint_min",
                str(gop),
                "-sc_threshold",
                "0",
            ]
        else:
            # Recommended raw recording mode: no decode/transcode.
            # This is lighter and usually more stable for 24-hour CCTV capture.
            cmd += ["-map", "0:v:0", "-an", "-c:v", "copy"]

        cmd += ["-max_muxing_queue_size", "1024"]

        cmd += [
            "-f",
            "segment",
            "-segment_time",
            str(self.config.recorder.segment_seconds),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
        ]

        # Disabled by default. When enabled, FFmpeg cuts on wall-clock boundaries,
        # so the first segment after a restart can naturally be only 1-59 seconds.
        if self.config.recorder.segment_atclocktime:
            cmd += ["-segment_atclocktime", "1"]

        if self.config.recorder.video_container == "mp4":
            cmd += [
                "-segment_format",
                "mp4",
                "-movflags",
                "+faststart",
            ]
        else:
            cmd += ["-segment_format", "mpegts"]

        cmd += [output_pattern]
        return cmd

    def status_dir(self) -> Path:
        path = self.config.storage.output_root / "status"
        ensure_dir(path)
        return path

    def status_file(self) -> Path:
        return self.status_dir() / f"{self.point.name}_status.csv"

    def write_status(self, status: str, reason: str = "", http_status: str = "") -> None:
        path = self.status_file()
        file_exists = path.exists()
        row = {
            "timestamp": now_local().strftime("%Y-%m-%d %H:%M:%S"),
            "cctv_name": self.point.name,
            "status": status,
            "reason": reason,
            "http_status": http_status,
            "url": self.point.url,
        }

        try:
            with path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            self.logger.warning("Failed to write status CSV: %s", exc)

    def preflight_stream(self) -> tuple[bool, str, str]:
        """
        Mengecek apakah URL HLS dapat dibuka sebelum menjalankan FFmpeg.
        Tujuannya bukan menggantikan FFmpeg, tetapi mencegah restart loop agresif
        saat URL jelas 404/403 atau DNS sedang bermasalah.
        """
        if not self.config.network.preflight_check:
            self.last_reachable_at = time.monotonic()
            return True, "preflight_disabled", ""

        headers = {
            **self.input_headers(),
            "Accept": "*/*",
        }

        try:
            with requests.Session() as session:
                with session.get(
                    self.point.url,
                    headers=headers,
                    timeout=10,
                    allow_redirects=True,
                    stream=True,
                ) as response:
                    http_status = str(response.status_code)
                    # Any answer -- even 404 -- proves outbound TCP still works.
                    # Only a dead URL, not a dead host, so the watchdog must not
                    # count this as unreachable.
                    self.last_reachable_at = time.monotonic()

                    # 2xx berarti playlist dapat dibuka.
                    if 200 <= response.status_code < 300:
                        return True, "ok", http_status

                    # 404/410 biasanya link memfs sudah expired/hilang.
                    if response.status_code in {404, 410}:
                        return False, "not_found_or_expired_url", http_status

                    # 401/403 biasanya perlu header/token/link baru.
                    if response.status_code in {401, 403}:
                        return False, "forbidden_or_unauthorized", http_status

                    # 5xx kemungkinan server sedang bermasalah.
                    if 500 <= response.status_code < 600:
                        return False, "server_error", http_status

                    return False, f"http_{response.status_code}", http_status

        except requests.exceptions.ConnectionError as exc:
            return False, f"connection_error: {exc}", ""

        except requests.exceptions.Timeout:
            return False, "timeout", ""

        except requests.RequestException as exc:
            return False, f"preflight_error: {exc}", ""

    def handle_preflight_failure(self, reason: str, http_status: str) -> None:
        is_expiry = "not_found" in reason or "expired" in reason
        if is_expiry:
            self.consecutive_expiry_failures += 1
        else:
            self.consecutive_expiry_failures = 0
            self.expiry_url_escalated = False

        self.write_status("offline", reason=reason, http_status=http_status)
        self.logger.warning(
            "Preflight failed for %s | reason=%s | http_status=%s",
            self.point.name,
            reason,
            http_status or "-",
        )

        threshold = self.config.network.expired_url_escalation_threshold
        if is_expiry and self.consecutive_expiry_failures >= threshold:
            marker = "EXPIRED_URL_ESCALATION"
            if not self.expiry_url_escalated:
                self.logger.error(
                    "[%s] %s consecutive expired URL preflight failures for %s; "
                    "the stream URL likely needs rotation.",
                    marker,
                    self.consecutive_expiry_failures,
                    self.point.name,
                )
                self.write_status(
                    "expired_url_escalated",
                    reason=f"{marker}: {reason}; consecutive_failures="
                    f"{self.consecutive_expiry_failures}",
                    http_status=http_status,
                )
                self.expiry_url_escalated = True

    def sleep_after_preflight_failure(self, reason: str) -> None:
        if "not_found" in reason or "forbidden" in reason or "unauthorized" in reason:
            delay = self.config.network.offline_retry_seconds
            self.logger.warning(
                "Stream URL looks offline/expired: %s. Retrying in %s seconds.",
                reason,
                delay,
            )
        else:
            delay = self.config.network.network_retry_seconds
            self.logger.warning(
                "Stream network/server issue: %s. Retrying in %s seconds.",
                reason,
                delay,
            )

        self.stop_event.wait(delay)

    def start_ffmpeg(self) -> None:
        command = self.build_ffmpeg_command()
        self.last_restart_at = now_local()
        self.process_start_date = now_local().date()

        stderr_path = self.stderr_path()
        self.ffmpeg_stderr_file = open(stderr_path, "a", encoding="utf-8", buffering=1)

        self.ffmpeg_stderr_file.write("\n" + "=" * 100 + "\n")
        self.ffmpeg_stderr_file.write(f"START {now_local().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.ffmpeg_stderr_file.write("COMMAND:\n")
        self.ffmpeg_stderr_file.write(" ".join(command) + "\n")
        self.ffmpeg_stderr_file.write("=" * 100 + "\n")

        self.logger.info("Starting FFmpeg. stderr log: %s", stderr_path)
        self.logger.info("Output container: %s", self.config.recorder.video_container)
        self.logger.info("Transport mode: %s", self.config.recorder.ffmpeg_transport_mode)

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=self.ffmpeg_stderr_file,
            creationflags=creationflags,
        )

    def monitor_ffmpeg(self) -> None:
        while not self.stop_event.is_set():
            if self.process is None:
                return

            return_code = self.process.poll()
            if return_code is not None:
                self.logger.warning("FFmpeg exited unexpectedly with code: %s", return_code)
                self.log_recent_ffmpeg_stderr()
                self.close_ffmpeg_stderr()
                # Tandai bahwa FFmpeg keluar sendiri (bukan di-stop oleh kita).
                self._ffmpeg_self_exited = True
                return

            if self.process_start_date and now_local().date() != self.process_start_date:
                self.logger.info("Date changed. Restarting FFmpeg to switch output folder.")
                self.stop_ffmpeg()
                return

            if self.is_video_stale():
                self.logger.warning(
                    "No recent valid video file detected in the last %s seconds.",
                    self.config.recorder.stale_file_seconds,
                )
                self.log_recent_ffmpeg_stderr()
                self.stop_ffmpeg()
                return

            self.stop_event.wait(self.config.recorder.health_check_seconds)

    def get_latest_video_stat(self) -> tuple[Path, os.stat_result] | None:
        video_root = self.current_video_dir()
        if not video_root.exists():
            return None

        ext = f".{self.output_extension()}"
        prefix = f"{self.point.name}_"
        start_ts = (
            (self.last_restart_at.timestamp() - 5) if self.last_restart_at is not None else None
        )

        latest_path: Path | None = None
        latest_stat: os.stat_result | None = None

        try:
            with os.scandir(video_root) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not (name.startswith(prefix) and name.endswith(ext)):
                        continue
                    try:
                        st = entry.stat()
                    except OSError:
                        continue
                    if start_ts is not None and st.st_mtime < start_ts:
                        continue
                    if latest_stat is None or st.st_mtime > latest_stat.st_mtime:
                        latest_path = Path(entry.path)
                        latest_stat = st
        except OSError:
            return None

        if latest_path is not None and latest_stat is not None:
            return latest_path, latest_stat
        return None

    def latest_video_file(self) -> Path | None:
        res = self.get_latest_video_stat()
        return res[0] if res is not None else None

    def is_video_stale(self) -> bool:
        res = self.get_latest_video_stat()

        if res is None:
            if self.last_restart_at is None:
                return False
            age_since_restart = (now_local() - self.last_restart_at).total_seconds()
            return age_since_restart > self.config.recorder.stale_file_seconds

        latest, stat = res
        age = time.time() - stat.st_mtime

        # Jangan menilai file yang sedang aktif ditulis.
        if age > self.config.recorder.stale_file_seconds and stat.st_size < 100 * 1024:
            self.logger.warning("Latest file is too small: %s | %s bytes", latest, stat.st_size)
            return True

        return age > self.config.recorder.stale_file_seconds

    def log_recent_ffmpeg_stderr(self, lines: int = 30) -> None:
        path = self.stderr_path()
        if not path.exists():
            return

        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
            recent = content[-lines:]
            if recent:
                self.logger.warning("Recent FFmpeg stderr:")
                for line in recent:
                    self.logger.warning("FFmpeg | %s", line)
        except OSError as exc:
            self.logger.warning("Cannot read FFmpeg stderr log: %s", exc)

    def close_ffmpeg_stderr(self) -> None:
        if self.ffmpeg_stderr_file:
            try:
                self.ffmpeg_stderr_file.flush()
                self.ffmpeg_stderr_file.close()
            except OSError:
                pass
            self.ffmpeg_stderr_file = None

    def stop_ffmpeg(self) -> None:
        if self.process is None:
            self.close_ffmpeg_stderr()
            return

        if self.process.poll() is None:
            self.logger.info("Stopping FFmpeg.")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("FFmpeg did not terminate. Killing process.")
                self.process.kill()
            except OSError as exc:
                self.logger.warning("Failed to stop FFmpeg cleanly: %s", exc)

        self.process = None
        self.close_ffmpeg_stderr()

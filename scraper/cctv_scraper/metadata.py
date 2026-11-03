import csv
import logging
import threading
import time
from datetime import datetime

import requests

from cctv_scraper.config import (
    API_TIMEOUT_SECONDS,
    CCTVPoint,
    RuntimeConfig,
    ensure_dir,
    now_local,
)


class MetadataCollector(threading.Thread):
    """
    Mengambil metadata eksternal per titik.

    - TomTom dipanggil default setiap 300 detik / 5 menit.
    - Open-Meteo dipanggil default setiap 60 detik dalam satu request batch untuk semua titik.
    - CSV metadata tetap ditulis setiap METADATA_INTERVAL_SECONDS.
    - Jika belum waktunya call ulang, nilai terakhir dipakai kembali dan diberi status cached.
    - Memegang satu requests.Session selama masa hidup thread untuk connection pooling & HTTP keep-alive.
    """

    def __init__(self, points: list[CCTVPoint], config: RuntimeConfig, stop_event: threading.Event):
        super().__init__(name="metadata-collector", daemon=True)
        self.points = points
        self.config = config
        self.stop_event = stop_event
        self.logger = logging.getLogger("metadata")

        self.tomtom_cache: dict[str, dict] = {}
        self.tomtom_last_fetch: dict[str, float] = {}

        self.openmeteo_cache: dict[str, dict] = {}
        self.openmeteo_last_fetch: dict[str, float] = {}

        self.requests_this_pass = 0
        self.request_failures_this_pass = 0
        self.consecutive_failed_passes = 0

    def _record_request(self, failed: bool) -> None:
        self.requests_this_pass += 1
        if failed:
            self.request_failures_this_pass += 1

    def run(self) -> None:
        self.logger.info("Metadata collector started.")
        self.logger.info(
            "Metadata CSV write interval: %s seconds",
            self.config.metadata.metadata_interval_seconds,
        )
        self.logger.info(
            "TomTom API interval: %s seconds", self.config.metadata.tomtom_interval_seconds
        )
        self.logger.info(
            "Open-Meteo API interval: %s seconds", self.config.metadata.openmeteo_interval_seconds
        )

        session = requests.Session()
        try:
            while not self.stop_event.is_set():
                self.requests_this_pass = 0
                self.request_failures_this_pass = 0
                start = time.time()
                timestamp = now_local()
                weather_by_point = self.get_openmeteo_cached_batch(session)

                for point in self.points:
                    try:
                        tomtom = self.get_tomtom_cached(session, point)
                        weather = weather_by_point[point.name]
                        metadata = self.build_metadata_row(point, timestamp, tomtom, weather)
                        self.write_metadata(point, metadata)
                    except Exception as exc:
                        self.logger.exception("Metadata failed for %s: %s", point.name, exc)

                if self.requests_this_pass:
                    if self.request_failures_this_pass:
                        self.consecutive_failed_passes += 1
                    else:
                        self.consecutive_failed_passes = 0

                elapsed = time.time() - start
                failure_backoff = 0
                if self.consecutive_failed_passes:
                    failure_backoff = min(
                        self.config.metadata.failure_backoff_max_seconds,
                        self.config.metadata.failure_backoff_base_seconds
                        * (2 ** (self.consecutive_failed_passes - 1)),
                    )
                sleep_time = max(
                    self.config.metadata.min_pass_interval_seconds,
                    self.config.metadata.metadata_interval_seconds - elapsed,
                    failure_backoff,
                )
                self.stop_event.wait(sleep_time)
        finally:
            session.close()

        self.logger.info("Metadata collector stopped.")

    def should_fetch(
        self, cache_key: str, last_fetch: dict[str, float], interval_seconds: int
    ) -> bool:
        if cache_key not in last_fetch:
            return True

        return (time.time() - last_fetch[cache_key]) >= interval_seconds

    def collect_point_metadata(
        self, session: requests.Session, point: CCTVPoint, timestamp: datetime
    ) -> dict:
        tomtom = self.get_tomtom_cached(session, point)
        weather = self.get_openmeteo_cached(session, point)
        return self.build_metadata_row(point, timestamp, tomtom, weather)

    @staticmethod
    def build_metadata_row(
        point: CCTVPoint, timestamp: datetime, tomtom: dict, weather: dict
    ) -> dict:
        row = {
            "cctv_name": point.name,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "datestamp": timestamp.strftime("%Y%m%d"),
            "timestampsafe": timestamp.strftime("%H%M%S"),
            "latitude": point.lat,
            "longitude": point.lon,
            "stream_url": point.url,
        }
        row.update(tomtom)
        row.update(weather)
        return row

    def get_tomtom_cached(self, session: requests.Session, point: CCTVPoint) -> dict:
        cache_key = point.name

        if self.should_fetch(
            cache_key, self.tomtom_last_fetch, self.config.metadata.tomtom_interval_seconds
        ):
            data = self.get_tomtom(session, point)
            data["traffic_cache_status"] = "fresh"
            data["traffic_last_api_call"] = now_local().strftime("%Y-%m-%d %H:%M:%S")

            self.tomtom_cache[cache_key] = data
            self.tomtom_last_fetch[cache_key] = time.time()
            return data

        cached = self.tomtom_cache.get(cache_key)
        if cached:
            data = dict(cached)
            data["traffic_cache_status"] = "cached"
            return data

        # Safety fallback. Normally unreachable because first call should fetch.
        data = self.get_tomtom(session, point)
        data["traffic_cache_status"] = "fresh_fallback"
        data["traffic_last_api_call"] = now_local().strftime("%Y-%m-%d %H:%M:%S")
        self.tomtom_cache[cache_key] = data
        self.tomtom_last_fetch[cache_key] = time.time()
        return data

    def get_openmeteo_cached(self, session: requests.Session, point: CCTVPoint) -> dict:
        cache_key = point.name

        if self.should_fetch(
            cache_key, self.openmeteo_last_fetch, self.config.metadata.openmeteo_interval_seconds
        ):
            data = self.get_openmeteo(session, point)
            data["weather_cache_status"] = "fresh"
            data["weather_last_api_call"] = now_local().strftime("%Y-%m-%d %H:%M:%S")

            self.openmeteo_cache[cache_key] = data
            self.openmeteo_last_fetch[cache_key] = time.time()
            return data

        cached = self.openmeteo_cache.get(cache_key)
        if cached:
            data = dict(cached)
            data["weather_cache_status"] = "cached"
            return data

        data = self.get_openmeteo(session, point)
        data["weather_cache_status"] = "fresh_fallback"
        data["weather_last_api_call"] = now_local().strftime("%Y-%m-%d %H:%M:%S")
        self.openmeteo_cache[cache_key] = data
        self.openmeteo_last_fetch[cache_key] = time.time()
        return data

    def get_openmeteo_cached_batch(self, session: requests.Session) -> dict[str, dict]:
        """Fetch weather for every point with one request when the cache expires."""
        due = any(
            self.should_fetch(
                point.name,
                self.openmeteo_last_fetch,
                self.config.metadata.openmeteo_interval_seconds,
            )
            for point in self.points
        )
        if not due:
            return {
                point.name: {
                    **self.openmeteo_cache[point.name],
                    "weather_cache_status": "cached",
                }
                for point in self.points
            }

        fetched_at = time.time()
        fetched_at_text = now_local().strftime("%Y-%m-%d %H:%M:%S")
        fetched = self.get_openmeteo_batch(session, self.points)
        result: dict[str, dict] = {}
        for point in self.points:
            data = fetched[point.name]
            if data.get("weather_status") == "error":
                cached = self.openmeteo_cache.get(point.name)
                if cached:
                    data = {**cached, "weather_cache_status": "cached"}
                else:
                    data = {
                        **data,
                        "weather_cache_status": "fresh_fallback",
                        "weather_last_api_call": fetched_at_text,
                    }
            else:
                data = {
                    **data,
                    "weather_cache_status": "fresh",
                    "weather_last_api_call": fetched_at_text,
                }
            self.openmeteo_cache[point.name] = data
            self.openmeteo_last_fetch[point.name] = fetched_at
            result[point.name] = data
        return result

    def get_tomtom(self, session: requests.Session, point: CCTVPoint) -> dict:
        default = {
            "traffic_speed": None,
            "traffic_freeflow": None,
            "traffic_confidence": None,
            "traffic_road_closure": None,
            "traffic_source": "tomtom",
            "traffic_status": "missing_api_key"
            if not self.config.metadata.tomtom_api_key
            else "error",
        }

        if not self.config.metadata.tomtom_api_key:
            return default

        url = (
            "https://api.tomtom.com/traffic/services/4/flowSegmentData/"
            f"absolute/10/json?key={self.config.metadata.tomtom_api_key}&point={point.lat},{point.lon}"
        )

        try:
            response = session.get(url, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            flow = response.json().get("flowSegmentData", {})

            self._record_request(False)
            return {
                "traffic_speed": flow.get("currentSpeed"),
                "traffic_freeflow": flow.get("freeFlowSpeed"),
                "traffic_confidence": flow.get("confidence"),
                "traffic_road_closure": flow.get("roadClosure"),
                "traffic_source": "tomtom",
                "traffic_status": "ok",
            }

        except (requests.RequestException, ValueError, KeyError) as exc:
            self._record_request(True)
            self.logger.warning("TomTom failed for %s: %s", point.name, self._redact_api_key(exc))
            return default

    def _redact_api_key(self, exc: Exception) -> str:
        """Strip the TomTom key out of an exception message before it reaches a log file.

        requests/urllib3 embed the full request URL verbatim in connection and HTTP
        errors -- building the URL with ``params=`` instead of an f-string does not
        avoid this. Scrubbing the known secret value is the only reliable fix.
        """
        text = str(exc)
        key = self.config.metadata.tomtom_api_key
        return text.replace(key, "****") if key else text

    def get_openmeteo_batch(
        self, session: requests.Session, points: list[CCTVPoint]
    ) -> dict[str, dict]:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={','.join(str(point.lat) for point in points)}&"
            f"longitude={','.join(str(point.lon) for point in points)}"
            "&current=temperature_2m,relative_humidity_2m,rain,wind_speed_10m"
            "&timezone=Asia%2FJakarta"
        )
        try:
            response = session.get(url, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payloads = [payload] if len(points) == 1 else []
            elif isinstance(payload, list):
                payloads = payload
            else:
                payloads = []
            if len(payloads) != len(points):
                raise ValueError("Open-Meteo returned an unexpected number of locations")

            result: dict[str, dict] = {}
            for point, location in zip(points, payloads, strict=True):
                current = location.get("current", {})
                result[point.name] = {
                    "weather_temp": current.get("temperature_2m"),
                    "weather_humidity": current.get("relative_humidity_2m"),
                    "weather_rain": current.get("rain"),
                    "weather_wind_speed": current.get("wind_speed_10m"),
                    "weather_source": "open-meteo",
                    "weather_status": "ok",
                }
            self._record_request(False)
            return result
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
            self._record_request(True)
            self.logger.warning("Open-Meteo batch failed: %s", exc)
            return {
                point.name: {
                    "weather_temp": None,
                    "weather_humidity": None,
                    "weather_rain": None,
                    "weather_wind_speed": None,
                    "weather_source": "open-meteo",
                    "weather_status": "error",
                }
                for point in points
            }

    def get_openmeteo(self, session: requests.Session, point: CCTVPoint) -> dict:
        default = {
            "weather_temp": None,
            "weather_humidity": None,
            "weather_rain": None,
            "weather_wind_speed": None,
            "weather_source": "open-meteo",
            "weather_status": "error",
        }

        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={point.lat}&longitude={point.lon}"
            "&current=temperature_2m,relative_humidity_2m,rain,wind_speed_10m"
            "&timezone=Asia%2FJakarta"
        )

        try:
            response = session.get(url, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            current = response.json().get("current", {})

            self._record_request(False)
            return {
                "weather_temp": current.get("temperature_2m"),
                "weather_humidity": current.get("relative_humidity_2m"),
                "weather_rain": current.get("rain"),
                "weather_wind_speed": current.get("wind_speed_10m"),
                "weather_source": "open-meteo",
                "weather_status": "ok",
            }

        except (requests.RequestException, ValueError, KeyError) as exc:
            self._record_request(True)
            self.logger.warning("Open-Meteo failed for %s: %s", point.name, exc)
            return default

    def write_metadata(self, point: CCTVPoint, row: dict) -> None:
        date_folder = row["date"]
        metadata_dir = self.config.storage.output_root / date_folder / point.name / "metadata"
        ensure_dir(metadata_dir)

        csv_path = metadata_dir / f"{point.name}_{date_folder}_metadata.csv"
        file_exists = csv_path.exists()

        try:
            with csv_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            self.logger.warning("Failed to write metadata CSV %s: %s", csv_path, exc)

        self.logger.info(
            "Metadata saved: %s | TomTom=%s | OpenMeteo=%s",
            csv_path,
            row.get("traffic_cache_status"),
            row.get("weather_cache_status"),
        )

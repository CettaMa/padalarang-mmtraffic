"""Event modality pipeline: registry validation, candidate discovery, and feature annotation.

This module provides tools for maintaining a curated event registry (concerts,
festivals, sports, road closures, holidays, etc.), discovering candidates from
free public sources (TheSportsDB fixtures, GDELT news), and computing the six-column
event vector (Eq. 13 of the dataset paper) for traffic metadata datasets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import holidays
from holidays.constants import GOVERNMENT, PUBLIC
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Path setup and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = REPO_ROOT / "events"
CAMERAS_JSON = REPO_ROOT / "cctv_cameras.json"

CATEGORIES = (
    "concert",
    "festival",
    "sport",
    "religious",
    "community",
    "expo",
    "demonstration",
    "roadwork",
    "road_closure",
    "mudik",
    "school_holiday",
)

ATTENDANCE_CATEGORIES = (
    "concert",
    "festival",
    "sport",
    "religious",
    "community",
    "expo",
    "demonstration",
)

REGION_WIDE = ("mudik", "school_holiday")

REGISTRY_COLUMNS = [
    "event_id",
    "name",
    "category",
    "start_local",
    "end_local",
    "venue",
    "lat",
    "lon",
    "expected_attendance",
    "pre_hours",
    "post_hours",
    "severity_override",
    "source_url",
    "source_url_2",
    "verified",
    "notes",
]


# ---------------------------------------------------------------------------
# Loading and Utilities
# ---------------------------------------------------------------------------

def _write_csv_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """Write DataFrame to CSV atomically using a sibling temp file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(out_path.parent),
        prefix=f".{out_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_params(params_path: Path) -> dict[str, Any]:
    """Load JSON parameter file."""
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cameras(cameras_path: Path = CAMERAS_JSON) -> dict[str, Any]:
    """Load CCTV cameras configuration."""
    with open(cameras_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_events(events_path: Path) -> pd.DataFrame:
    """Load event registry CSV as string records."""
    if not events_path.exists():
        raise FileNotFoundError(f"events registry not found: {events_path}")
    return pd.read_csv(events_path, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two coordinates in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return 6371.0 * c


def corridor_centroid(cameras: dict[str, Any]) -> tuple[float, float]:
    """Compute corridor camera centroid (mean lat, mean lon)."""
    lats = [cam["lat"] for cam in cameras.values() if "lat" in cam]
    lons = [cam["lon"] for cam in cameras.values() if "lon" in cam]
    if not lats or not lons:
        return (0.0, 0.0)
    return (float(sum(lats) / len(lats)), float(sum(lons) / len(lons)))


def event_severity(
    category: str,
    attendance: float | None,
    d_km: float | None,
    override: int | None,
    params: dict[str, Any],
) -> int:
    """Compute event impact severity score (0-3).

    Rules in order:
    1. If override is not None, return it.
    2. mudik returns 3. school_holiday returns 1.
    3. roadwork / road_closure:
       - 3 if d_km <= corridor_radius_km
       - 1 if d_km <= roadwork_radius_km
       - else 0
    4. Attendance categories:
       - 0 if d_km is not None and d_km > region_radius_km
       - Base is first tier with attendance >= threshold, else 0
       - If d_km is not None and d_km <= local_radius_km: base = min(3, base + 1)
       - Return base
    """
    if override is not None:
        return int(override)

    if category == "mudik":
        return 3
    if category == "school_holiday":
        return 1

    corridor_r = float(params.get("corridor_radius_km", 2.0))
    roadwork_r = float(params.get("roadwork_radius_km", 10.0))
    region_r = float(params.get("region_radius_km", 40.0))
    local_r = float(params.get("local_radius_km", 5.0))

    if category in ("roadwork", "road_closure"):
        if d_km is not None and d_km <= corridor_r:
            return 3
        if d_km is not None and d_km <= roadwork_r:
            return 1
        return 0

    if category in ATTENDANCE_CATEGORIES:
        if d_km is not None and d_km > region_r:
            return 0
        tiers = params.get("attendance_tiers", [[20000, 3], [5000, 2], [1000, 1]])
        base = 0
        if attendance is not None:
            for thresh, score in tiers:
                if attendance >= thresh:
                    base = score
                    break
        if d_km is not None and d_km <= local_r:
            base = min(3, base + 1)
        return base

    return 0


def validate_events(
    df: pd.DataFrame,
    params: dict[str, Any],
    centroid: tuple[float, float],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate event registry records and return (kept, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    # Check required schema
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            errors.append(f"registry: missing column '{col}'")
    if errors:
        return (pd.DataFrame(), errors, warnings)

    seen_ids: set[str] = set()
    kept_rows: list[dict[str, Any]] = []

    known_venues = params.get("venues", {})
    halo_defaults = params.get("halo_hours", {})
    region_r = float(params.get("region_radius_km", 40.0))

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        eid = str(row_dict.get("event_id", "")).strip()
        row_errors: list[str] = []

        if not eid:
            row_errors.append("registry: row with empty event_id")
            eid = "<missing_id>"
        elif eid in seen_ids:
            row_errors.append(f"{eid}: duplicate event_id")
        else:
            seen_ids.add(eid)

        cat = str(row_dict.get("category", "")).strip()
        if cat not in CATEGORIES:
            row_errors.append(f"{eid}: unknown category '{cat}'")

        start_str = str(row_dict.get("start_local", "")).strip()
        end_str = str(row_dict.get("end_local", "")).strip()
        start_dt: pd.Timestamp | None = None
        end_dt: pd.Timestamp | None = None

        try:
            start_dt = pd.to_datetime(start_str, format="%Y-%m-%d %H:%M")
        except Exception:
            row_errors.append(f"{eid}: unparsable start_local '{start_str}'")

        try:
            end_dt = pd.to_datetime(end_str, format="%Y-%m-%d %H:%M")
        except Exception:
            row_errors.append(f"{eid}: unparsable end_local '{end_str}'")

        if start_dt is not None and end_dt is not None and end_dt <= start_dt:
            row_errors.append(f"{eid}: end_local <= start_local ({end_str} <= {start_str})")

        source_url = str(row_dict.get("source_url", "")).strip()
        if not (source_url.startswith("http://") or source_url.startswith("https://")):
            row_errors.append(f"{eid}: source_url must start with http, got '{source_url}'")

        # Fill coordinates from known venues if blank
        lat_str = str(row_dict.get("lat", "")).strip()
        lon_str = str(row_dict.get("lon", "")).strip()
        venue_str = str(row_dict.get("venue", "")).strip()

        if not lat_str and not lon_str and venue_str in known_venues:
            lat_str = str(known_venues[venue_str][0])
            lon_str = str(known_venues[venue_str][1])
            row_dict["lat"] = lat_str
            row_dict["lon"] = lon_str

        if (lat_str and not lon_str) or (lon_str and not lat_str):
            row_errors.append(f"{eid}: only one of lat/lon set")

        lat_val: float | None = None
        lon_val: float | None = None
        if lat_str and lon_str:
            try:
                lat_val = float(lat_str)
                lon_val = float(lon_str)
            except ValueError:
                row_errors.append(f"{eid}: invalid float coordinates ({lat_str}, {lon_str})")

        ver_str = str(row_dict.get("verified", "")).strip()
        if ver_str not in ("0", "1"):
            row_errors.append(f"{eid}: verified must be 0 or 1, got '{ver_str}'")

        # Attendance / override validation
        att_str = str(row_dict.get("expected_attendance", "")).strip()
        ovr_str = str(row_dict.get("severity_override", "")).strip()
        att_val: float | None = None
        ovr_val: int | None = None

        if att_str:
            try:
                att_val = float(att_str)
            except ValueError:
                row_errors.append(f"{eid}: invalid expected_attendance '{att_str}'")
        if ovr_str:
            try:
                ovr_val = int(ovr_str)
            except ValueError:
                row_errors.append(f"{eid}: invalid severity_override '{ovr_str}'")

        if cat in ATTENDANCE_CATEGORIES and att_val is None and ovr_val is None:
            row_errors.append(
                f"{eid}: attendance category '{cat}' requires expected_attendance or severity_override"
            )

        if cat in ("roadwork", "road_closure") and (lat_val is None or lon_val is None):
            row_errors.append(
                f"{eid}: category '{cat}' requires coordinates or known venue"
            )

        errors.extend(row_errors)
        if row_errors:
            continue

        # If row is verified == 0, it is not kept
        if ver_str != "1":
            continue

        # Compute distance to centroid
        d_km: float | None = None
        if lat_val is not None and lon_val is not None:
            d_km = haversine_km(centroid[0], centroid[1], lat_val, lon_val)

        severity = event_severity(cat, att_val, d_km, ovr_val, params)
        if severity < 1:
            if d_km is not None and d_km > region_r:
                warnings.append(
                    f"{eid}: dropped with severity 0 (distance {d_km:.1f} km > region radius {region_r} km)"
                )
            else:
                warnings.append(
                    f"{eid}: dropped with severity 0 (below attendance tier or zero impact)"
                )
            continue

        # Compute halo window
        pre_str = str(row_dict.get("pre_hours", "")).strip()
        post_str = str(row_dict.get("post_hours", "")).strip()
        default_pre, default_post = halo_defaults.get(cat, [0, 0])
        pre_h = float(pre_str) if pre_str else float(default_pre)
        post_h = float(post_str) if post_str else float(default_post)

        assert start_dt is not None and end_dt is not None
        win_start = start_dt - pd.Timedelta(hours=pre_h)
        win_end = end_dt + pd.Timedelta(hours=post_h)

        kept_item = dict(row_dict)
        kept_item["start"] = start_dt
        kept_item["end"] = end_dt
        kept_item["win_start"] = win_start
        kept_item["win_end"] = win_end
        kept_item["pre_hours_val"] = pre_h
        kept_item["post_hours_val"] = post_h
        kept_item["d_km"] = d_km
        kept_item["severity"] = severity
        kept_rows.append(kept_item)

    kept_df = pd.DataFrame(kept_rows)
    return (kept_df, errors, warnings)


def event_vector(
    ts: pd.Series,
    kept: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Compute the six-column event vector of Eq. (13): evt_holiday (public holidays and cuti bersama), evt_weekend, evt_local (event window incl. halo), evt_severity (0–3), evt_days_to_major, evt_days_since_major (fractional days, capped at days_cap)."""
    col_names = [
        "evt_holiday",
        "evt_weekend",
        "evt_local",
        "evt_severity",
        "evt_days_to_major",
        "evt_days_since_major",
    ]

    if ts.empty:
        return pd.DataFrame(columns=col_names)

    # Ensure ts is datetime
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts)

    n_rows = len(ts)
    years = sorted(ts.dt.year.unique())

    # Build holiday sets for all present years
    public_holidays: set[dt.date] = set()
    govt_holidays: set[dt.date] = set()

    for y in years:
        pub_for_y = holidays.ID(years=y, categories=(PUBLIC,))
        govt_for_y = holidays.ID(years=y, categories=(GOVERNMENT,))
        if not govt_for_y:
            raise ValueError(f"holidays has no GOVERNMENT dates for {y}; upgrade holidays")
        public_holidays.update(pub_for_y.keys())
        govt_holidays.update(govt_for_y.keys())

    ts_dates = ts.dt.date
    evt_holiday = ts_dates.isin(public_holidays | govt_holidays).astype(int)
    evt_weekend = (ts.dt.weekday >= 5).astype(int)

    # Pre-allocate feature arrays
    evt_local = np.zeros(n_rows, dtype=int)
    evt_severity = np.zeros(n_rows, dtype=int)

    if not kept.empty:
        for _, event in kept.iterrows():
            sev = int(event["severity"])
            win_start = event["win_start"]
            win_end = event["win_end"]

            active_mask = ((ts >= win_start) & (ts <= win_end)).to_numpy()
            if np.any(active_mask):
                evt_local[active_mask] = 1
                evt_severity[active_mask] = np.maximum(evt_severity[active_mask], sev)

    # Major event days to / since
    major_severity = int(params.get("major_severity", 3))
    days_cap = float(params.get("days_cap", 30.0))

    days_to_major = np.full(n_rows, days_cap, dtype=float)
    days_since_major = np.full(n_rows, days_cap, dtype=float)

    if not kept.empty:
        major_events = kept[kept["severity"] >= major_severity]
        if not major_events.empty:
            is_inside_major = np.zeros(n_rows, dtype=bool)
            ts_series = ts

            for _, me in major_events.iterrows():
                m_start = me["start"]
                m_end = me["end"]

                inside = ((ts_series >= m_start) & (ts_series <= m_end)).to_numpy()
                is_inside_major = is_inside_major | inside

                future_mask = (ts_series < m_start).to_numpy()
                if np.any(future_mask):
                    diff_to = (m_start - ts_series[future_mask]).dt.total_seconds().to_numpy() / 86400.0
                    days_to_major[future_mask] = np.minimum(days_to_major[future_mask], diff_to)

                past_mask = (ts_series > m_end).to_numpy()
                if np.any(past_mask):
                    diff_since = (ts_series[past_mask] - m_end).dt.total_seconds().to_numpy() / 86400.0
                    days_since_major[past_mask] = np.minimum(days_since_major[past_mask], diff_since)

            days_to_major[is_inside_major] = 0.0
            days_since_major[is_inside_major] = 0.0

    days_to_major = np.round(np.clip(days_to_major, 0.0, days_cap), 3)
    days_since_major = np.round(np.clip(days_since_major, 0.0, days_cap), 3)

    data_dict: dict[str, Any] = {
        "evt_holiday": evt_holiday.to_numpy(),
        "evt_weekend": evt_weekend.to_numpy(),
        "evt_local": evt_local,
        "evt_severity": evt_severity,
        "evt_days_to_major": days_to_major,
        "evt_days_since_major": days_since_major,
    }

    return pd.DataFrame(data_dict, index=ts.index)


# ---------------------------------------------------------------------------
# CLI Subcommands
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    """Validate event registry CSV against parameters and schema."""
    events_path = Path(args.events)
    if not events_path.exists():
        print(f"events registry not found: {events_path}")
        return 1

    df = pd.read_csv(events_path, dtype=str, keep_default_na=False)
    params = load_params(Path(args.params))
    cameras = load_cameras()
    centroid = corridor_centroid(cameras)

    kept, errors, warnings = validate_events(df, params, centroid)

    for w in warnings:
        print(f"[!] Warning: {w}")
    for e in errors:
        print(f"[X] Error: {e}")

    if errors:
        print(f"Validation failed with {len(errors)} error(s).")
        return 1

    print(f"Validation passed. Kept {len(kept)} verified active event(s):")
    if not kept.empty:
        for _, row in kept.iterrows():
            d_str = f"{row['d_km']:.1f} km" if (row["d_km"] is not None and not pd.isna(row["d_km"])) else "region-wide"
            w_start = row["win_start"].strftime("%Y-%m-%d %H:%M")
            w_end = row["win_end"].strftime("%Y-%m-%d %H:%M")
            print(
                f"  - {row['event_id']} | {row['category']} | "
                f"{row['start_local']}→{row['end_local']} | {d_str} | "
                f"sev={row['severity']} | win={w_start}→{w_end}"
            )
    return 0


def cmd_discover_football(args: argparse.Namespace) -> int:
    """Discover Liga 1 / Persib football fixtures from TheSportsDB."""
    params = load_params(Path(args.params))
    out_path = Path(args.out)
    key = os.environ.get("THESPORTSDB_KEY", "123")
    league_id = str(params.get("football_league_id", "4790"))
    home_teams = set(params.get("football_home_teams", ["Persib Bandung"]))
    match_hours = float(params.get("football_match_hours", 2.0))
    venues_map = params.get("venues", {})

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)
    cur_date = start_date

    candidates: list[dict[str, str]] = []

    print(f"Discovering football fixtures from {start_date} to {end_date}...")
    while cur_date <= end_date:
        d_str = cur_date.isoformat()
        url = f"https://www.thesportsdb.com/api/v1/json/{key}/eventsday.php?d={d_str}&l={league_id}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                raw_events = data.get("events") if isinstance(data, dict) else None
                if raw_events:
                    for ev in raw_events:
                        home = ev.get("strHomeTeam", "")
                        if home in home_teams:
                            ev_id = ev.get("idEvent", "")
                            away = ev.get("strAwayTeam", "")
                            venue = ev.get("strVenue", "")
                            ts_utc_str = ev.get("strTimestamp", "")

                            # Parse UTC timestamp and convert to local UTC+7
                            if ts_utc_str:
                                try:
                                    ts_utc = pd.to_datetime(ts_utc_str)
                                    ts_local = ts_utc + pd.Timedelta(hours=7)
                                    start_local = ts_local.strftime("%Y-%m-%d %H:%M")
                                    end_local = (ts_local + pd.Timedelta(hours=match_hours)).strftime("%Y-%m-%d %H:%M")
                                except Exception:
                                    start_local = f"{d_str} 19:00"
                                    end_local = f"{d_str} 21:00"
                            else:
                                start_local = f"{d_str} 19:00"
                                end_local = f"{d_str} 21:00"

                            lat_s = ""
                            lon_s = ""
                            if venue in venues_map:
                                lat_s = str(venues_map[venue][0])
                                lon_s = str(venues_map[venue][1])

                            candidates.append({
                                "event_id": f"cand_tsdb_{ev_id}",
                                "name": f"{home} v {away}",
                                "category": "sport",
                                "start_local": start_local,
                                "end_local": end_local,
                                "venue": venue,
                                "lat": lat_s,
                                "lon": lon_s,
                                "expected_attendance": "",
                                "pre_hours": "",
                                "post_hours": "",
                                "severity_override": "",
                                "source_url": f"https://www.thesportsdb.com/event/{ev_id}",
                                "source_url_2": "",
                                "verified": "0",
                                "notes": "thesportsdb",
                            })
                            print(f"[+] Found fixture: {home} v {away} on {start_local} at {venue}")
            else:
                print(f"[!] {d_str}: HTTP {resp.status_code}")
        except Exception as ex:
            print(f"[!] {d_str}: error {ex}")

        time.sleep(2.5)
        cur_date += dt.timedelta(days=1)

    _write_candidate_rows(candidates, out_path)
    return 0


def cmd_discover_news(args: argparse.Namespace) -> int:
    """Discover candidate local events from GDELT news API."""
    params = load_params(Path(args.params))
    out_path = Path(args.out)
    queries = params.get("news_queries", ["konser Bandung", "festival Bandung"])

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)
    today = dt.date.today()
    ninety_days_ago = today - dt.timedelta(days=90)

    # Split into 7-day chunks
    chunks: list[tuple[dt.datetime, dt.datetime]] = []
    chunk_start = dt.datetime.combine(start_date, dt.time.min)
    final_end = dt.datetime.combine(end_date, dt.time(23, 59, 59))

    while chunk_start <= final_end:
        chunk_end = min(chunk_start + dt.timedelta(days=7) - dt.timedelta(seconds=1), final_end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + dt.timedelta(seconds=1)

    candidates: list[dict[str, str]] = []

    for q in queries:
        print(f"Querying GDELT for query: '{q}'...")
        for c_start, c_end in chunks:
            if c_end.date() < ninety_days_ago:
                print(f"[!] chunk {c_start.date()}..{c_end.date()} older than GDELT 3-month window, results may be empty")

            # Convert local UTC+7 to UTC (-7h) for GDELT API format YYYYMMDDHHMMSS
            utc_start = c_start - dt.timedelta(hours=7)
            utc_end = c_end - dt.timedelta(hours=7)
            s_str = utc_start.strftime("%Y%m%d%H%M%S")
            e_str = utc_end.strftime("%Y%m%d%H%M%S")

            api_url = "https://api.gdeltproject.org/api/v2/doc/doc"
            req_params = {
                "query": f"{q} sourcelang:indonesian",
                "mode": "artlist",
                "format": "json",
                "maxrecords": "250",
                "sort": "datedesc",
                "startdatetime": s_str,
                "enddatetime": e_str,
            }

            retries = 1
            success = False
            while retries > 0:
                try:
                    resp = requests.get(api_url, params=req_params, timeout=30)
                    if resp.status_code == 429:
                        print(f"[!] GDELT 429 rate limit hit. Sleeping 10s...")
                        time.sleep(10.0)
                        retries -= 1
                        continue

                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            articles = data.get("articles", []) if isinstance(data, dict) else []
                            for art in articles:
                                url = art.get("url", "")
                                title = art.get("title", "")
                                domain = art.get("domain", "")
                                seendate = art.get("seendate", "")

                                if not url:
                                    continue

                                # Parse seendate (UTC) -> convert to local UTC+7
                                start_local = c_start.strftime("%Y-%m-%d %H:%M")
                                if seendate:
                                    try:
                                        seen_dt = pd.to_datetime(seendate)
                                        start_local = (seen_dt + pd.Timedelta(hours=7)).strftime("%Y-%m-%d %H:%M")
                                    except Exception:
                                        pass

                                url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
                                candidates.append({
                                    "event_id": f"cand_gdelt_{url_hash}",
                                    "name": title,
                                    "category": "",
                                    "start_local": start_local,
                                    "end_local": start_local,
                                    "venue": "",
                                    "lat": "",
                                    "lon": "",
                                    "expected_attendance": "",
                                    "pre_hours": "",
                                    "post_hours": "",
                                    "severity_override": "",
                                    "source_url": url,
                                    "source_url_2": "",
                                    "verified": "0",
                                    "notes": f"gdelt:{q}; {domain}",
                                })
                            success = True
                        except Exception as parse_ex:
                            print(f"[!] GDELT response parse error: {parse_ex}")
                    else:
                        print(f"[!] GDELT HTTP {resp.status_code}: {resp.text[:120]}")
                    break
                except Exception as ex:
                    print(f"[!] GDELT request error: {ex}")
                    break

            if not success and retries == 0:
                print(f"[!] gdelt 429 giving up: {q} {c_start.date()}..{c_end.date()}")

            time.sleep(6.0)

    _write_candidate_rows(candidates, out_path)
    return 0


def _write_candidate_rows(new_rows: list[dict[str, str]], out_path: Path) -> None:
    """Merge new candidates into existing candidate file, deduplicating on source_url."""
    existing_rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    if out_path.exists():
        existing_df = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        for _, r in existing_df.iterrows():
            d = r.to_dict()
            u = d.get("source_url", "").strip()
            if u:
                seen_urls.add(u)
            existing_rows.append(d)

    added = 0
    for nr in new_rows:
        u = nr.get("source_url", "").strip()
        if u and u in seen_urls:
            continue
        if u:
            seen_urls.add(u)
        existing_rows.append(nr)
        added += 1

    out_df = pd.DataFrame(existing_rows, columns=REGISTRY_COLUMNS)
    if not out_df.empty and "start_local" in out_df.columns:
        out_df = out_df.sort_values(by="start_local")

    _write_csv_atomic(out_df, out_path)
    print(f"[+] Wrote {len(out_df)} candidate rows ({added} new) to {out_path}")


def cmd_annotate(args: argparse.Namespace) -> int:
    """Annotate feature CSVs with the extended event feature matrix."""
    events_path = Path(args.events)
    params_path = Path(args.params)
    out_dir = Path(args.out)

    if not events_path.exists():
        print(f"events registry not found: {events_path}")
        return 1

    df_events = pd.read_csv(events_path, dtype=str, keep_default_na=False)
    params = load_params(params_path)
    cameras = load_cameras()
    centroid = corridor_centroid(cameras)

    kept, errors, warnings = validate_events(df_events, params, centroid)
    for w in warnings:
        print(f"[!] Warning: {w}")
    if errors:
        for e in errors:
            print(f"[X] Error: {e}")
        print("Validation errors encountered. Aborting annotation.")
        return 1

    # Locate input feature files
    features_glob = args.features_glob
    glob_path = Path(features_glob)
    if "*" in glob_path.name:
        search_dir = glob_path.parent
        pattern = glob_path.name
        matched_files = sorted(search_dir.glob(pattern))
    else:
        matched_files = [glob_path] if glob_path.exists() else []

    target_dates = set(args.dates) if args.dates else None
    if target_dates:
        matched_files = [
            f for f in matched_files
            if any(f"_{d}_" in f.name for d in target_dates)
        ]

    if not matched_files:
        print("[!] no feature CSVs matched")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_count = 0

    for fpath in matched_files:
        feat_df = pd.read_csv(fpath)
        if "timestamp" not in feat_df.columns:
            print(f"[!] Skipping {fpath.name}: no 'timestamp' column")
            continue

        # Drop any existing evt_* columns (idempotent)
        clean_df = feat_df[[c for c in feat_df.columns if not c.startswith("evt_")]].copy()
        ts_series = pd.to_datetime(clean_df["timestamp"], format="%Y-%m-%d %H:%M:%S")

        vec_df = event_vector(ts_series, kept, params)
        combined_df = pd.concat([clean_df, vec_df], axis=1)

        dest_path = out_dir / fpath.name
        _write_csv_atomic(combined_df, dest_path)
        print(f"[+] Wrote {len(combined_df)} rows to {dest_path}")
        annotated_count += 1

    print(f"Finished. Successfully annotated {annotated_count} file(s).")
    return 0


# ---------------------------------------------------------------------------
# Parser definition
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser with validate, discover, and annotate subcommands."""
    parser = argparse.ArgumentParser(
        description="Event modality tools: registry validation, discovery, and dataset annotation."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate events registry CSV")
    p_val.add_argument("--events", default=str(EVENTS_DIR / "events.csv"), help="Path to events.csv")
    p_val.add_argument("--params", default=str(EVENTS_DIR / "event_params.json"), help="Path to event_params.json")
    p_val.set_defaults(func=cmd_validate)

    # discover-football
    p_fb = subparsers.add_parser("discover-football", help="Discover football fixtures from TheSportsDB")
    p_fb.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_fb.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p_fb.add_argument("--out", default=str(EVENTS_DIR / "candidates.csv"), help="Path to candidates.csv")
    p_fb.add_argument("--params", default=str(EVENTS_DIR / "event_params.json"), help="Path to event_params.json")
    p_fb.add_argument("--events", default=str(EVENTS_DIR / "events.csv"), help="Path to events.csv")
    p_fb.set_defaults(func=cmd_discover_football)

    # discover-news
    p_news = subparsers.add_parser("discover-news", help="Discover event news from GDELT API")
    p_news.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_news.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p_news.add_argument("--out", default=str(EVENTS_DIR / "candidates.csv"), help="Path to candidates.csv")
    p_news.add_argument("--params", default=str(EVENTS_DIR / "event_params.json"), help="Path to event_params.json")
    p_news.add_argument("--events", default=str(EVENTS_DIR / "events.csv"), help="Path to events.csv")
    p_news.set_defaults(func=cmd_discover_news)

    # annotate
    p_ann = subparsers.add_parser("annotate", help="Annotate feature CSVs with event features")
    p_ann.add_argument("--events", default=str(EVENTS_DIR / "events.csv"), help="Path to events.csv")
    p_ann.add_argument("--params", default=str(EVENTS_DIR / "event_params.json"), help="Path to event_params.json")
    p_ann.add_argument(
        "--features-glob",
        default=str(REPO_ROOT / "extracted" / "*_metadata_features.csv"),
        help="Glob pattern matching feature CSV files",
    )
    p_ann.add_argument(
        "--out",
        default=str(REPO_ROOT / "extracted" / "events_annotated"),
        help="Output directory for annotated CSV files",
    )
    p_ann.add_argument("--dates", nargs="*", default=None, help="Optional subset of dates to annotate")
    p_ann.set_defaults(func=cmd_annotate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

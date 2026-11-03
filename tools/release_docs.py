"""Data dictionary and README text for the Zenodo release.

`check_dictionary(release_root)` fails when a CSV column shipped in the release is not
documented here, so the README can never silently fall behind the files.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

FEATURES = {
    "cctv_name": ("string", "-", "Identifier of the CCTV observation node (one of six)"),
    "timestamp": ("string", "UTC+7", "Row timestamp `YYYY-MM-DD HH:MM:SS`: start of the CCTV window, host clock"),
    "date": ("string", "UTC+7", "Calendar date `YYYY-MM-DD`"),
    "time": ("string", "UTC+7", "Time of day `HH:MM:SS`"),
    "datestamp": ("string", "-", "Date as `YYYYMMDD`"),
    "timestampsafe": ("string", "-", "Filesystem-safe timestamp `YYYYMMDD_HHMMSS`"),
    "latitude": ("float", "deg", "Latitude of the camera"),
    "longitude": ("float", "deg", "Longitude of the camera"),
    "stream_url": ("string", "-", "Public stream address with the access token masked"),
    "weather_temp": ("float", "deg C", "Air temperature at 2 m (Open-Meteo current conditions)"),
    "weather_humidity": ("float", "%", "Relative humidity at 2 m (Open-Meteo)"),
    "weather_rain": ("float", "mm", "Rain accumulated over the preceding 15 minutes, 0.1 mm resolution (Open-Meteo)"),
    "weather_wind_speed": ("float", "km/h", "Wind speed at 10 m (Open-Meteo)"),
    "weather_source": ("string", "-", "Weather source identifier"),
    "weather_status": ("string", "-", "Request status: `ok`, or `error` when no response was ever received (values empty)"),
    "weather_cache_status": ("string", "-", "`fresh`, or `cached` when the previous response was written again"),
    "weather_last_api_call": ("string", "UTC+7", "Time of the API response that supplied the values"),
    "cctv_status": ("string", "-", "`ok`; `partial` (truncated or failed video segments in the window); `outage` (no decodable video, ROI values are 0)"),
    "evt_holiday": ("int", "0/1", "National holiday or collective-leave day"),
    "evt_weekend": ("int", "0/1", "Saturday or Sunday"),
    "evt_local": ("int", "0/1", "Local event with documented corridor impact"),
    "evt_severity": ("int", "0-3", "Ordinal event severity (0 none, 3 major)"),
    "evt_days_to_major": ("float", "days", "Fractional days until the next major event, capped at 30"),
    "evt_days_since_major": ("float", "days", "Fractional days since the previous major event, capped at 30"),
}
_ROI_RX = re.compile(r"^ROI_([123])_(AvgSpeed_Kmh|Vol_(bus|car|motorcycle|truck))$")

METRICS = {
    "cctv": ("string", "-", "CCTV node identifier (same as `cctv_name` in the feature files)"),
    "date": ("string", "UTC+7", "Calendar date"),
    "window_ts": ("string", "UTC+7", "Window start time `HHMMSS`"),
    "roi_idx": ("int", "-", "ROI index, **0-based** (feature files use 1-based `ROI_n_*`)"),
    "lo_sec": ("float", "s", "Window start, seconds since midnight on the host clock"),
    "hi_sec": ("float", "s", "Window end, seconds since midnight"),
    "covered_sec": ("float", "s", "Seconds of the window covered by decodable video"),
    "low_coverage": ("int", "0/1", "1 when covered_sec is below 80% of the window"),
    "vol_bus": ("int", "veh", "Unique tracked buses in the ROI, Eq. (3)"),
    "vol_car": ("int", "veh", "Unique tracked cars in the ROI"),
    "vol_motorcycle": ("int", "veh", "Unique tracked motorcycles in the ROI"),
    "vol_truck": ("int", "veh", "Unique tracked trucks in the ROI"),
    "vol_total": ("int", "veh", "Sum of the four class volumes"),
    "flow_veh_h": ("float", "veh/h", "Volume scaled to one hour over covered_sec"),
    "pcu_pkji": ("float", "pcu", "Passenger-car-unit volume with PKJI 2023 factors, Eq. (6)"),
    "flow_pcu_h_pkji": ("float", "pcu/h", "PKJI pcu volume scaled to one hour"),
    "pcu_emp": ("float", "pcu", "Passenger-car-unit volume with empirical (Chandra) factors"),
    "flow_pcu_h_emp": ("float", "pcu/h", "Empirical pcu volume scaled to one hour"),
    "emp_chandra_bus": ("float", "-", "Window estimate of the bus equivalence factor (empty when too few vehicles)"),
    "emp_chandra_motorcycle": ("float", "-", "Window estimate of the motorcycle equivalence factor"),
    "emp_chandra_truck": ("float", "-", "Window estimate of the truck equivalence factor"),
    "occ_time_pct": ("float", "%", "Time occupancy, Eq. (7)"),
    "occ_any_pct": ("float", "%", "Any-occupancy, Eq. (7)"),
    "density_veh_km": ("float", "veh/km", "Flow divided by space mean speed"),
    "spd_n": ("int", "veh", "Vehicles with a valid traversal speed, Eq. (4)"),
    "spd_tms": ("float", "km/h", "Time mean speed, Eq. (5)"),
    "spd_sms": ("float", "km/h", "Space mean speed (harmonic mean), Eq. (5)"),
    "spd_std": ("float", "km/h", "Sample standard deviation of speed, Eq. (5)"),
    "spd_cv": ("float", "-", "Coefficient of variation of speed"),
    "spd_p15": ("float", "km/h", "15th percentile of speed"),
    "spd_p50": ("float", "km/h", "Median speed"),
    "spd_p85": ("float", "km/h", "85th percentile of speed"),
    "c_pkji_pcu_h": ("float", "pcu/h", "PKJI guideline capacity"),
    "ds_pkji": ("float", "-", "Degree of saturation against the PKJI capacity, Eq. (10)"),
    "c_emp_pcu_h": ("float", "pcu/h", "Empirical capacity (maximum sustained 15-minute flow at that ROI)"),
    "ds_emp": ("float", "-", "Degree of saturation against the empirical capacity"),
    "q_thr_fixed_kmh": ("float", "km/h", "Fixed queue-speed threshold (10 km/h), Eq. (8)"),
    "q_thr_data_kmh": ("float", "km/h", "Per-camera Otsu queue-speed threshold (empty when fewer than 200 speeds)"),
    "queue_len_mean_m_fixed": ("float", "m", "Mean queue length, fixed threshold, Eq. (9)"),
    "queue_len_max_m_fixed": ("float", "m", "Maximum queue length, fixed threshold"),
    "queue_len_p95_m_fixed": ("float", "m", "95th percentile queue length, fixed threshold"),
    "queued_veh_mean_fixed": ("float", "veh", "Mean number of queued vehicles, fixed threshold"),
    "queue_present_pct_fixed": ("float", "%", "Share of snapshots with a queue, fixed threshold"),
    "queue_len_mean_m_data": ("float", "m", "Mean queue length, Otsu threshold"),
    "queue_len_max_m_data": ("float", "m", "Maximum queue length, Otsu threshold"),
    "queue_len_p95_m_data": ("float", "m", "95th percentile queue length, Otsu threshold"),
    "queued_veh_mean_data": ("float", "veh", "Mean number of queued vehicles, Otsu threshold"),
    "queue_present_pct_data": ("float", "%", "Share of snapshots with a queue, Otsu threshold"),
    # capacity_empirical.csv / emp_empirical.csv
    "window_ts_start": ("string", "UTC+7", "Start of the peak 15-minute window that sets the empirical capacity"),
    "n_runs": ("int", "-", "Number of consecutive complete windows in that run"),
    "emp_bus": ("float", "-", "Median bus equivalence factor over windows"),
    "emp_motorcycle": ("float", "-", "Median motorcycle equivalence factor over windows"),
    "emp_truck": ("float", "-", "Median truck equivalence factor over windows"),
    "n_windows_bus": ("int", "-", "Windows contributing to emp_bus"),
    "n_windows_motorcycle": ("int", "-", "Windows contributing to emp_motorcycle"),
    "n_windows_truck": ("int", "-", "Windows contributing to emp_truck"),
}

OTHER = {
    "graph/nodes.csv": {
        "node_id": "Node index 0-11 (order used for adjacency and distance matrices)",
        "node_key": "`<cctv_name>|<roi_idx>` with 0-based roi_idx",
        "cctv_name": "CCTV node identifier", "roi_idx": "ROI index, 0-based", "roi_label": "ROI label as used in the paper (`ROI 1`...)",
        "start_lat": "Latitude of the first point of the traced road segment", "start_lon": "Longitude of the first point",
        "end_lat": "Latitude of the last point", "end_lon": "Longitude of the last point", "length_m": "Measured ROI length in meters",
    },
    "graph/edges.csv": {
        "src": "Source node_id (src < dst)", "dst": "Destination node_id", "src_key": "node_key of src", "dst_key": "node_key of dst",
        "distance_m": "Minimum distance between the endpoints of the two traced segments, meters",
        "type": "`intra_camera` (same camera) or `cross_camera` (endpoints within 120 m)",
    },
    "events/events.csv": {
        "event_id": "Event identifier", "name": "Event name", "category": "Event category", "start_local": "Start (UTC+7)",
        "end_local": "End (UTC+7)", "venue": "Venue name", "lat": "Venue latitude", "lon": "Venue longitude",
        "expected_attendance": "Expected attendance", "pre_hours": "Hours before the start that count as the event halo",
        "post_hours": "Hours after the end that count as the event halo", "severity_override": "Manual severity (0-3), empty when derived",
        "source_url": "Primary source", "source_url_2": "Secondary source", "verified": "1 when the record was verified by the authors", "notes": "Free text",
    },
    "validation/annotations.csv": {
        "sample_id": "`date|cctv|roi_idx|HHMMSS|release_track_id`", "rank": "Review order within the candidate pool", "date": "Date of the footage",
        "cctv_id": "CCTV node", "roi_idx": "ROI index, 0-based", "perspective": "`depan` (approaching the camera) or `belakang` (receding)",
        "stratum_class": "Class stratum the candidate was drawn for", "replay_class": "Detector class in the replay", "true_class": "Class assigned by the reviewer",
        "track_id": "Track id in the replay", "release_track_id": "Track id in the released track log", "video_name": "Source segment file name (video not released)",
        "native_fps": "Frame rate of the source video", "roi_length_meters": "ROI length used", "roi_width_meters": "ROI width used in extraction",
        "entry_frame": "Reviewer-marked entry frame", "exit_frame": "Reviewer-marked exit frame", "false_positive": "Reviewer flagged a false detection",
        "invalid_crossing": "Reviewer flagged a stopped, occluded, or identity-switched vehicle (excluded)", "delta_frames": "exit_frame - entry_frame",
        "ground_truth_speed_kmh": "Manual reference speed", "predicted_speed_kmh": "Pipeline traversal speed", "error_kmh": "predicted - ground truth",
        "mape_percent": "Absolute percentage error",
    },
    "validation/final_sample.csv": "same columns as annotations.csv",
    "validation/metrics.csv": {
        "group_type": "`overall`, `perspective`, `cctv`, `class`, or `roi`", "group_key": "Stratum value", "n": "Vehicles",
        "mae": "Mean absolute error, km/h", "mae_lo": "95% CI lower bound", "mae_hi": "95% CI upper bound", "rmse": "Root mean square error, km/h",
        "rmse_lo": "95% CI lower bound", "rmse_hi": "95% CI upper bound", "bias": "Mean of predicted - ground truth, km/h", "bias_lo": "95% CI lower bound",
        "bias_hi": "95% CI upper bound", "sd": "SD of the differences", "loa_lo": "Lower 95% limit of agreement", "loa_hi": "Upper 95% limit of agreement",
        "mape": "Mean absolute percentage error, %", "r": "Pearson correlation", "r_lo": "95% CI lower bound", "r_hi": "95% CI upper bound",
    },
    "validation/analysis/roi_diagnostics.csv": {
        "cctv_id": "CCTV node", "cctv_short": "Short name", "roi_idx": "ROI index, **1-based**", "length_m": "ROI length", "n": "Vehicles",
        "mae": "MAE", "mae_lo": "CI low", "mae_hi": "CI high", "rmse": "RMSE", "rmse_lo": "CI low", "rmse_hi": "CI high", "bias": "Bias", "r": "Pearson r",
        "r_lo": "CI low", "r_hi": "CI high", "invalid_pct": "Share of reviewed candidates marked invalid crossings",
        "coverage_median": "Median fraction of the ROI length between the marked entry and exit frames", "coverage_iqr": "Inter-quartile range of that fraction",
        "gap_pct": "Share of tracks with a detection gap over five frames", "decel_pct": "Share of tracks whose last-third speed is below 70% of the first-third speed",
        "corr_err_gt": "Correlation between error and ground-truth speed", "n_gt_lt5": "Vehicles with ground-truth speed below 5 km/h",
        "mean_err_lt5": "Mean error of those vehicles",
    },
    "validation/analysis/length_uncertainty.csv": {
        "file": "Screenshot file name", "cctv_id": "CCTV node", "roi_idx": "ROI index, 1-based", "dim": "`L` length or `W` width", "stored_m": "Value stored for extraction",
        "reading_m": "Satellite reading", "pixel_dist": "Pixel distance of the reading", "m_per_px": "Meters per screenshot pixel", "dL_m": "3*sqrt(2)*m_per_px",
        "dL_pct": "dL_m relative to the reading", "abs_diff_m": "|stored - reading|",
    },
    "validation/analysis/timing_uncertainty.csv": {
        "cctv_id": "CCTV node", "roi_idx": "ROI index, 1-based", "n": "Vehicles", "median_delta_frames": "Median frames between entry and exit",
        "median_rel_pct": "Median of 100*sqrt(2)/delta_frames", "median_abs_kmh": "Median of speed*sqrt(2)/delta_frames",
    },
    "validation/annotations_second.csv": "same columns as annotations.csv plus `annotator` (second-pass annotation by the same reviewer, without prefilled frames, for 48 vehicles)",
    "validation/analysis/interrater.csv": {
        "sample_id": "Sample id", "gt1_kmh": "Reference speed, first pass", "gt2_kmh": "Reference speed, repeat pass (empty when invalid)",
        "diff_kmh": "gt2 - gt1", "delta_entry": "Absolute entry-frame difference", "delta_exit": "Absolute exit-frame difference",
        "a2_fp": "Repeat pass flagged a false detection", "a2_invalid": "Repeat pass flagged an invalid crossing",
    },
    "validation/analysis/causes_summary.csv": {
        "sample_id": "Sample id (the third field is the 0-based ROI index)", "cctv_id": "CCTV node", "cctv_short": "Short name",
        "source": "`invalid` (invalid crossing) or `worst` (one of the 24 largest errors of the validation sample)",
        "cause": "Cause chosen by the reviewer: `partial_track`, `box_misfit`, `id_switch_fragment`, `stopped_queued`, `occluded`, or `other`",
    },
    "validation/analysis/causes_table.csv": {
        "cause": "Cause (`All` is the total)", "invalid": "Count among invalid crossings", "worst": "Count among the largest errors", "All": "Total count",
    },
    "validation/analysis/causes_by_location.csv": {
        "cause": "Cause (`All` is the total)", "G5 Atas": "Count", "G5 Bawah": "Count", "Giant \u2013 Tagog": "Count", "Giant \u2013 Tol": "Count",
        "Jl. Baru KBP": "Count", "Tagog-Bandung": "Count", "All": "Total count",
    },
    "validation/mot_metrics.csv": {
        "group": "Clip, camera, `Day`, `Night`, or `All`", "num_frames": "Frames", "num_unique_objects": "Reference vehicles", "num_objects": "Reference boxes inside the ROI",
        "num_predictions": "Tracker boxes kept after the ROI filter", "mota": "Multi-object tracking accuracy", "motp": "1 - mean IoU of matched pairs",
        "idf1": "Identity F1 score", "recall": "Matched reference boxes / reference boxes", "precision": "Matched tracker boxes / tracker boxes",
        "idp": "Identity precision", "idr": "Identity recall", "num_switches": "Identity switches", "num_false_positives": "Tracker boxes with no reference match",
        "num_misses": "Reference boxes with no tracker match", "num_fragmentations": "Track fragmentations",
        "mostly_tracked": "Reference vehicles tracked for at least 80% of their frames", "mostly_lost": "Reference vehicles tracked for at most 20% of their frames",
    },
}


def feature_doc(col: str):
    if col in FEATURES:
        return FEATURES[col]
    m = _ROI_RX.match(col)
    if m:
        n, what = m.group(1), m.group(2)
        if what == "AvgSpeed_Kmh":
            return ("float", "km/h", f"Mean over vehicles of the smoothed speed in ROI {n} (running estimate, 5-160 km/h)")
        return ("int", "veh", f"Unique tracked {m.group(3)}s entering ROI {n} in the window")
    return None


def _header(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def check_dictionary(root: Path):
    """Assert every column of every shipped CSV is documented."""
    missing = []
    for p in sorted((root / "features").glob("*.csv")):
        missing += [f"{p.name}:{c}" for c in _header(p) if feature_doc(c) is None]
    for p in sorted((root / "metrics").glob("*.csv")):
        missing += [f"{p.name}:{c}" for c in _header(p) if c not in METRICS]
    for rel, doc in OTHER.items():
        p = root / rel
        if not p.exists():
            continue
        if isinstance(doc, str):
            continue
        hdr = _header(p)
        missing += [f"{rel}:{c}" for c in hdr if c not in doc]
        missing += [f"{rel}: documented but absent: {c}" for c in doc if c not in hdr]
    ref = _header(root / "validation" / "annotations.csv")
    got = _header(root / "validation" / "final_sample.csv")
    assert ref == got, f"final_sample.csv columns differ from annotations.csv: {set(ref) ^ set(got)}"
    assert not missing, "undocumented or stale columns: " + ", ".join(missing[:20])


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([" :--- "] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def dictionary_markdown() -> str:
    parts = ["## Data Dictionary", "", "### Feature files (`features/<cctv>_<date>_metadata_features.csv`)",
             "One row per scraper cycle (nominally 300 s). Empty means missing (never imputed); see *Missing values* below. "
             "ROI columns are numbered from 1 (`ROI_1_*`, `ROI_2_*`, `ROI_3_*`); locations with fewer ROIs leave the higher ROI columns empty.", ""]
    rows = [(f"`{c}`", t, u, d) for c, (t, u, d) in FEATURES.items()]
    rows += [("`ROI_n_AvgSpeed_Kmh`", "float", "km/h", "n = 1..3. Mean over vehicles of the smoothed speed in ROI n (running estimate, 5-160 km/h)"),
             ("`ROI_n_Vol_<class>`", "int", "veh", "n = 1..3, class in `bus`, `car`, `motorcycle`, `truck`. Unique tracked vehicles entering ROI n in the window")]
    parts += [_table(rows, ["Column", "Type", "Unit", "Description"]), ""]
    parts += ["### Metric files (`metrics/<cctv>_<date>_metrics.csv`, `capacity_empirical.csv`, `emp_empirical.csv`)",
              "One row per camera, ROI and five-minute window. `roi_idx` is **0-based** here. Speeds are traversal speeds: "
              "the road-plane distance between the first and last observation of a vehicle in the ROI divided by the time between them "
              "(at least three observations over at least 0.3 s; values above 160 km/h discarded).", ""]
    parts += [_table([(f"`{c}`", t, u, d) for c, (t, u, d) in METRICS.items()], ["Column", "Type", "Unit", "Description"]), ""]
    parts += ["### Track logs (`tracks/tracks_<date>.zip`)",
              "Per window: `<HHMMSS>_meta.json` (window bounds `lo_sec`, `hi_sec`, `frames`, `covered_sec`, `n_ticks`, `target_fps`, `queue_rois`), "
              "`<HHMMSS>_veh.csv.gz` (one row per vehicle and ROI) and `<HHMMSS>_q.csv.gz` (one row per queue-zone snapshot).", "",
              _table([
                  ("`_veh.csv.gz`", "`roi_idx`, `track_id`, `class_id`", "ROI (0-based), tracker identity, detector class (0 bus, 1 mobil, 2 motor, 3 truk)"),
                  ("", "`t_first`, `t_last`, `n_obs`", "First and last video time (s) in the ROI and number of observations"),
                  ("", "`x_first_m`, `y_first_m`, `x_last_m`, `y_last_m`", "Road-plane position (m) at the first and last observation"),
                  ("", "`counted_new`", "1 when the vehicle was counted for the first time in this window"),
                  ("`_q.csv.gz`", "`t`, `roi_idx`, `track_id`, `class_id`", "Snapshot time (s), ROI, identity and class"),
                  ("", "`x_m`, `y_m`, `speed_kmh`", "Position on the road plane (m) and snapshot speed (empty on the first snapshot)"),
              ], ["File", "Columns", "Meaning"]), ""]
    parts += ["### ROI definitions (`rois/<cctv>_roi.json`)",
              "`cctv_id` and `rois[]`; each ROI has `points` (four image vertices, pixels at 1280 x 720), `length_meters`, `width_meters`, `perspective` "
              "(`depan` approaching, `belakang` receding) and `queue_zone` (`points`, `length_meters`, `width_meters`). "
              "`roi_measure_picture/` holds the satellite screenshots, `readings.csv` and `anchors.md` behind the measured dimensions.", "",
              "### Tracker evaluation clips (`validation/mot/<clip>/`)",
              "Twelve clips of 150 frames sampled at 10 Hz from 11 October 2026 (`<camera>_day` from 16:00, `<camera>_night` from 20:00). Per clip: "
              "`gt_mot.zip` (hand-corrected reference, MOT 1.1: `gt/gt.txt` with `frame, id, x, y, w, h, flag, class, visibility` in pixels at 1280 x 720, x and y the top-left corner, frames numbered from 1, class 1-based in the order of `gt/labels.txt`, "
              "and the 'gt/labels.txt' label list), `pred.csv` (pipeline tracks: `frame, track_id, x, y, w, h, conf, cls` with x, y the top-left corner and cls 0-based in the order bus, mobil, motor, truk), "
              "`masks.npz` (boolean `roi` and `zone` masks, 720 x 1280; only reference boxes whose bottom-centre lies in `roi` are scored) and `meta.json` (source segment names, sampled frame indices). "
              "The reference was corrected from the pipeline's own output. The frames are not released.", ""]
    for rel, doc in OTHER.items():
        title = {"graph/nodes.csv": "Graph nodes", "graph/edges.csv": "Graph edges", "events/events.csv": "Event calendar",
                 "validation/annotations.csv": "Validation annotations", "validation/final_sample.csv": "Validation sample",
                 "validation/metrics.csv": "Validation metrics", "validation/analysis/roi_diagnostics.csv": "ROI diagnostics",
                 "validation/analysis/length_uncertainty.csv": "ROI length uncertainty", "validation/analysis/timing_uncertainty.csv": "Timing uncertainty",
                 "validation/annotations_second.csv": "Repeat annotation", "validation/analysis/interrater.csv": "Repeat-annotation agreement",
                 "validation/analysis/causes_summary.csv": "Failure cause per inspected crossing", "validation/analysis/causes_table.csv": "Failure causes, counts",
                 "validation/analysis/causes_by_location.csv": "Failure causes by location", "validation/mot_metrics.csv": "Tracker evaluation (MOT)"}[rel]
        parts.append(f"### {title} (`{rel}`)")
        parts.append("")
        if isinstance(doc, str):
            parts += [doc[0].upper() + doc[1:] + ".", ""]
        else:
            parts += [_table([(f"`{c}`", d) for c, d in doc.items()], ["Column", "Description"]), ""]
    parts += ["`graph/adjacency.csv` and `graph/distance_m.csv` are 12 x 12 matrices in `node_id` order (binary adjacency, zero diagonal; "
              "minimum endpoint distance in meters). `events/event_params.json` holds the event-halo parameters.", ""]
    return "\n".join(parts)


def missing_values_markdown() -> str:
    return "\n".join([
        "## Missing values",
        "- `cctv_status = outage`: no decodable video in the window; ROI volumes and speeds are 0, so use this flag to separate outages from true zero flow.",
        "- `cctv_status = partial`: video segments were truncated or failed; ROI values were computed from the remaining footage.",
        "- `weather_status = error`: no weather response was ever received; the four weather values are empty.",
        "- `weather_cache_status = cached`: the previous response was written again after a failed request (the only carry-forward in the pipeline).",
        "- ROI columns that do not exist at a location (ROI 2 at Giant-Tol, ROI 3 at five locations) are structurally empty.",
        "- Rows are stamped at scraper cycle time, not on a fixed grid; re-index by flooring the timestamp to five minutes if a regular grid is needed.",
        "",
    ])

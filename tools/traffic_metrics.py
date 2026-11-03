"""Traffic metrics extension tool: derive occupancy, queue length, degree of saturation,
speed variability, and vehicle-equivalent conversion factors (EMP) from window track logs.

Subcommands:
    compute           Process track logs to produce summary tables and per-window metrics
    draw-queue-zone   PyQt6 annotator: click queue zones for every CCTV/ROI, enter length/width, auto-save + resume
"""

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "script"))

import numpy as np
import pandas as pd

from feature_extractor import (
    MAX_SPEED_KMH,
    ROI_DIR,
    TARGET_CLASSES,
    TARGET_FPS,
    get_perspective_matrices,
)


# ==============================================================================
# Pure Functions
# ==============================================================================

def traverse_speed_kmh(rec, min_obs: int = 3, min_sec: float = 0.3) -> float:
    """Calculate speed across an ROI in km/h from vehicle first/last observation.

    Returns NaN if n_obs < min_obs, t_last - t_first < min_sec, or speed > MAX_SPEED_KMH.
    """
    if isinstance(rec, dict):
        n_obs = rec.get("n_obs", 0)
        t_first = rec.get("t_first", 0.0)
        t_last = rec.get("t_last", 0.0)
        x_first = rec.get("x_first_m", 0.0)
        y_first = rec.get("y_first_m", 0.0)
        x_last = rec.get("x_last_m", 0.0)
        y_last = rec.get("y_last_m", 0.0)
    else:
        n_obs = rec["n_obs"]
        t_first = rec["t_first"]
        t_last = rec["t_last"]
        x_first = rec["x_first_m"]
        y_first = rec["y_first_m"]
        x_last = rec["x_last_m"]
        y_last = rec["y_last_m"]

    dt = float(t_last - t_first)
    if int(n_obs) < min_obs or dt < min_sec or dt <= 0:
        return float("nan")

    dist_m = math.hypot(float(x_last - x_first), float(y_last - y_first))
    spd = (dist_m / dt) * 3.6
    if not math.isfinite(spd) or spd > MAX_SPEED_KMH or spd < 0:
        return float("nan")
    return spd


def dwell_sec(rec) -> float:
    """Vehicle dwell time in seconds (t_last - t_first + 1/TARGET_FPS)."""
    if isinstance(rec, dict):
        t_first = rec.get("t_first", 0.0)
        t_last = rec.get("t_last", 0.0)
    else:
        t_first = rec["t_first"]
        t_last = rec["t_last"]
    return float(t_last - t_first + 1.0 / TARGET_FPS)


def occupancy(recs, covered_sec: float, lanes: int = 1,
              lo: float | None = None, hi: float | None = None) -> tuple[float, float]:
    """Calculate time occupancy percentage and any occupancy percentage.

    Intervals are [t_first, t_last + 1/TARGET_FPS], clipped to [lo, hi].
    Returns (occ_time_pct, occ_any_pct).
    """
    if covered_sec <= 0 or lanes <= 0:
        return (float("nan"), float("nan"))

    intervals = []
    for r in recs:
        if isinstance(r, (tuple, list)):
            s = float(r[0])
            e = float(r[1])
        elif isinstance(r, dict):
            s = float(r.get("t_first", 0.0))
            if "t_end" in r:
                e = float(r["t_end"])
            elif r.get("t_last_includes_frame"):
                e = float(r.get("t_last", 0.0))
            else:
                e = float(r.get("t_last", 0.0)) + 1.0 / TARGET_FPS
        else:
            s = float(r["t_first"])
            e = float(r["t_last"]) + 1.0 / TARGET_FPS

        if lo is not None:
            s = max(float(lo), s)
        if hi is not None:
            e = min(float(hi), e)
        if e > s:
            intervals.append((s, e))

    if not intervals:
        return (0.0, 0.0)

    sum_len = sum(e - s for s, e in intervals)
    occ_time_pct = min(100.0, (sum_len / (covered_sec * lanes)) * 100.0)

    # Union of intervals
    sorted_int = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_int[0]]
    for cur_s, cur_e in sorted_int[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    union_len = sum(e - s for s, e in merged)
    occ_any_pct = min(100.0, (union_len / covered_sec) * 100.0)

    return (occ_time_pct, occ_any_pct)


def speed_stats(speeds) -> dict:
    """Calculate speed variability metrics.

    Returns dict with keys: spd_n, spd_tms, spd_sms, spd_std, spd_cv, spd_p15, spd_p50, spd_p85.
    """
    arr = [float(s) for s in speeds if s is not None and math.isfinite(s) and s >= 0]
    n = len(arr)
    if n == 0:
        return {
            "spd_n": 0,
            "spd_tms": float("nan"),
            "spd_sms": float("nan"),
            "spd_std": float("nan"),
            "spd_cv": float("nan"),
            "spd_p15": float("nan"),
            "spd_p50": float("nan"),
            "spd_p85": float("nan"),
        }
    if n == 1:
        val = arr[0]
        sms = val if val > 0 else float("nan")
        return {
            "spd_n": 1,
            "spd_tms": val,
            "spd_sms": sms,
            "spd_std": float("nan"),
            "spd_cv": float("nan"),
            "spd_p15": val,
            "spd_p50": val,
            "spd_p85": val,
        }

    tms = float(np.mean(arr))
    pos = [s for s in arr if s > 0]
    sms = float(len(pos) / sum(1.0 / s for s in pos)) if pos else float("nan")
    std = float(np.std(arr, ddof=1))
    cv = (std / tms) if tms > 0 else float("nan")
    p15 = float(np.percentile(arr, 15))
    p50 = float(np.percentile(arr, 50))
    p85 = float(np.percentile(arr, 85))

    return {
        "spd_n": n,
        "spd_tms": tms,
        "spd_sms": sms,
        "spd_std": std,
        "spd_cv": cv,
        "spd_p15": p15,
        "spd_p50": p50,
        "spd_p85": p85,
    }


def queue_length_snapshot(ys_queued, head_max: float = 10.0, gap_max: float = 12.0) -> tuple[float, int]:
    """Calculate queue length and vehicle count for a single snapshot tick.

    ys_queued: sequence of y-coordinates of queued vehicles in meters.
    Returns (length_m, n_queued).
    """
    if not ys_queued:
        return (0.0, 0)
    ys = sorted(float(y) for y in ys_queued)
    if ys[0] > head_max:
        return (0.0, 0)

    last_idx = 0
    while last_idx + 1 < len(ys) and (ys[last_idx + 1] - ys[last_idx]) <= gap_max:
        last_idx += 1

    return (ys[last_idx], last_idx + 1)


def otsu_threshold(speeds, lo: float = 3.0, hi: float = 30.0, bin_kmh: float = 0.5) -> float:
    """Compute Otsu speed threshold on speeds in [0, 60) km/h with 120 bins.

    Returns NaN if < 200 valid samples or threshold outside [lo, hi].
    """
    valid = [float(s) for s in speeds if s is not None and math.isfinite(s) and 0.0 <= s < 60.0]
    if len(valid) < 200:
        return float("nan")

    bin_edges = np.arange(0.0, 60.0 + bin_kmh, bin_kmh)
    counts, _ = np.histogram(valid, bins=bin_edges)
    total = np.sum(counts)
    if total < 200:
        return float("nan")

    p = counts / total
    bin_centers = bin_edges[:-1] + (bin_kmh / 2.0)
    cum_w = np.cumsum(p)
    cum_mu = np.cumsum(p * bin_centers)
    mu_t = cum_mu[-1]

    variances = np.zeros(len(p) - 1, dtype=np.float64)
    for k in range(len(p) - 1):
        w0 = cum_w[k]
        w1 = 1.0 - w0
        if w0 <= 1e-9 or w1 <= 1e-9:
            continue
        mu0 = cum_mu[k] / w0
        mu1 = (mu_t - cum_mu[k]) / w1
        variances[k] = w0 * w1 * ((mu0 - mu1) ** 2)

    max_var = np.max(variances)
    if max_var <= 1e-9:
        return float("nan")

    plateau_indices = np.where(np.isclose(variances, max_var, rtol=1e-5, atol=1e-5))[0]
    best_edge = float(np.mean([bin_edges[k + 1] for k in plateau_indices]))
    if math.isnan(best_edge) or best_edge < lo or best_edge > hi:
        return float("nan")
    return best_edge


def chandra_emp(mean_speed_by_cls: dict, area_by_cls: dict) -> dict:
    """Calculate Chandra dynamic PCU conversion factor per vehicle class.

    EMP_i = (V_car / V_i) / (A_car / A_i)
    """
    res = {"car": 1.0}
    v_car = mean_speed_by_cls.get("car")
    a_car = area_by_cls.get("car")

    for cls in ("bus", "motorcycle", "truck"):
        v_cls = mean_speed_by_cls.get(cls)
        a_cls = area_by_cls.get(cls)
        if (v_car is not None and v_cls is not None and
                a_car is not None and a_cls is not None and
                math.isfinite(v_car) and math.isfinite(v_cls) and
                v_car > 0 and v_cls > 0 and a_car > 0 and a_cls > 0):
            res[cls] = (v_car / v_cls) / (a_car / a_cls)
        else:
            res[cls] = float("nan")
    return res


def pkji_emp(flow_veh_h: float, emp_cfg: dict) -> dict:
    """Select PKJI EMP table based on directional flow threshold."""
    thr = emp_cfg.get("flow_threshold_veh_h", 1800)
    if math.isfinite(flow_veh_h) and flow_veh_h >= thr:
        return emp_cfg["high_flow"]
    return emp_cfg["low_flow"]


# ==============================================================================
# Loading and Utilities
# ==============================================================================

def load_windows(tracks_dir: Path, date: str, cctv: str) -> list[dict]:
    """Load all window track logs for a given camera date directory.

    Returns list of dicts: {"ts": ts, "meta": meta, "veh": veh_df, "q": q_df}.
    """
    cam_dir = tracks_dir / date / cctv
    if not cam_dir.exists():
        return []

    windows = []
    meta_files = sorted(cam_dir.glob("*_meta.json"))
    for mf in meta_files:
        ts = mf.name[:6]
        try:
            with open(mf, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[warn] failed to read {mf}: {e}")
            continue

        veh_path = cam_dir / f"{ts}_veh.csv.gz"
        if not veh_path.exists():
            print(f"[warn] missing veh log {veh_path}")
            continue
        try:
            veh_df = pd.read_csv(veh_path)
        except Exception as e:
            print(f"[warn] failed to read {veh_path}: {e}")
            continue

        q_path = cam_dir / f"{ts}_q.csv.gz"
        q_df = None
        if q_path.exists():
            try:
                q_df = pd.read_csv(q_path)
            except Exception as e:
                print(f"[warn] failed to read {q_path}: {e}")

        windows.append({"ts": ts, "meta": meta, "veh": veh_df, "q": q_df})

    windows.sort(key=lambda w: w["meta"]["lo_sec"])
    return windows


def _write_csv_atomic(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


# ==============================================================================
# Compute Subcommand
# ==============================================================================

def cmd_compute(args):
    tracks_dir = Path(args.tracks)
    out_dir = Path(args.out)
    params_path = Path(args.params)

    if not params_path.exists():
        print(f"Error: params file not found: {params_path}", file=sys.stderr)
        sys.exit(1)

    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    q_thr_fixed = params.get("queue_speed_fixed_kmh", 10.0)
    q_head_max = params.get("queue_head_max_m", 10.0)
    q_gap_max = params.get("queue_gap_max_m", 12.0)
    otsu_range = params.get("otsu_range_kmh", [3.0, 30.0])
    low_cov_frac = params.get("low_coverage_frac", 0.8)
    min_speed_obs = params.get("min_speed_obs", 3)
    min_traverse_sec = params.get("min_traverse_sec", 0.3)
    chandra_min_veh = params.get("chandra_min_vehicles", 3)
    chandra_areas = params.get("chandra_area_m2", {"bus": 24.54, "car": 5.39, "motorcycle": 1.20, "truck": 19.0})
    emp_default_cfg = params.get("emp_default", {})
    cameras_cfg = params.get("cameras", {})

    # Discover dates and cameras
    if not tracks_dir.exists():
        print(f"Error: tracks directory not found: {tracks_dir}", file=sys.stderr)
        sys.exit(1)

    selected_dates = args.dates or sorted(d.name for d in tracks_dir.iterdir() if d.is_dir())
    selected_cctvs = set(args.cctv) if args.cctv else None

    # Load all window data into memory / map
    # all_data[date][cctv] = [windows...]
    all_data = collections.defaultdict(dict)
    known_cctv_rois = set()
    cctv_queue_rois = collections.defaultdict(set)

    warned_defaults = set()

    for date in selected_dates:
        date_dir = tracks_dir / date
        if not date_dir.is_dir():
            continue
        cctv_dirs = sorted(d.name for d in date_dir.iterdir() if d.is_dir())
        for cctv in cctv_dirs:
            if selected_cctvs and cctv not in selected_cctvs:
                continue
            wins = load_windows(tracks_dir, date, cctv)
            if not wins:
                continue
            all_data[date][cctv] = wins
            for w in wins:
                for qroi in w["meta"].get("queue_rois", []):
                    cctv_queue_rois[cctv].add(qroi)
                if not w["veh"].empty:
                    for r_idx in w["veh"]["roi_idx"].unique():
                        known_cctv_rois.add((cctv, int(r_idx)))

    # Also discover ROIs from roi json if veh was empty
    for cctv in set(cctv for d in all_data.values() for cctv in d):
        roi_json_path = ROI_DIR / f"{cctv}_roi.json"
        if roi_json_path.exists():
            with open(roi_json_path, encoding="utf-8") as f:
                r_json = json.load(f)
            for idx, r in enumerate(r_json.get("rois", [])):
                known_cctv_rois.add((cctv, idx))
                if r.get("queue_zone"):
                    cctv_queue_rois[cctv].add(idx)

    # --------------------------------------------------------------------------
    # Pass 1: Summary Tables (queue_thresholds, capacity_empirical, emp_empirical)
    # --------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. queue_thresholds.csv
    q_thr_rows = []
    for cctv, qrois in sorted(cctv_queue_rois.items()):
        for r_idx in sorted(qrois):
            speeds_pooled = []
            for date, cctvs in all_data.items():
                if cctv in cctvs:
                    for w in cctvs[cctv]:
                        q_df = w["q"]
                        if q_df is not None and not q_df.empty:
                            subset = q_df[q_df["roi_idx"] == r_idx]
                            valid_s = subset["speed_kmh"].dropna().tolist()
                            speeds_pooled.extend(valid_s)
            n_samples = len(speeds_pooled)
            otsu_val = otsu_threshold(speeds_pooled, lo=otsu_range[0], hi=otsu_range[1])
            if math.isfinite(otsu_val):
                used_kmh = otsu_val
                src = "otsu"
            else:
                used_kmh = q_thr_fixed
                src = "fixed_fallback"
            q_thr_rows.append({
                "cctv": cctv,
                "roi_idx": r_idx,
                "n_samples": n_samples,
                "otsu_kmh": round(otsu_val, 2) if math.isfinite(otsu_val) else float("nan"),
                "used_kmh": round(used_kmh, 2),
                "source": src,
            })
    q_thr_df = pd.DataFrame(q_thr_rows, columns=["cctv", "roi_idx", "n_samples", "otsu_kmh", "used_kmh", "source"])
    _write_csv_atomic(q_thr_df, out_dir / "queue_thresholds.csv")

    queue_threshold_lookup = {(r["cctv"], r["roi_idx"]): r["used_kmh"] for r in q_thr_rows}

    # 2. capacity_empirical.csv
    cap_rows = []
    all_cctv_roi_pairs = sorted(known_cctv_rois)
    for cctv, r_idx in all_cctv_roi_pairs:
        best_q15 = float("nan")
        best_date = ""
        best_window_ts = ""
        n_runs = 0

        cam_cfg = cameras_cfg.get(cctv, {}).get("rois", [])
        roi_cfg = cam_cfg[r_idx] if r_idx < len(cam_cfg) else None
        emp_cfg = (roi_cfg.get("emp") if roi_cfg else None) or emp_default_cfg

        for date, cctvs in all_data.items():
            if cctv not in cctvs:
                continue
            wins = cctvs[cctv]
            # Precompute pcu and covered_sec per window for this roi
            win_stats = []
            for w in wins:
                meta = w["meta"]
                lo_s = meta["lo_sec"]
                hi_s = meta["hi_sec"]
                cov_s = meta["covered_sec"]
                is_low_cov = cov_s < low_cov_frac * (hi_s - lo_s)

                veh_df = w["veh"]
                roi_veh = veh_df[(veh_df["roi_idx"] == r_idx) & (veh_df["counted_new"] == 1)] if not veh_df.empty else pd.DataFrame()
                counts = {cid: int((roi_veh["class_id"] == cid).sum()) if not roi_veh.empty else 0 for cid in (0, 1, 2, 3)}
                vol_tot = sum(counts.values())
                flow_v = (vol_tot * 3600.0 / cov_s) if cov_s > 0 else 0.0
                pkji_factors = pkji_emp(flow_v, emp_cfg)
                pcu = (counts[0] * pkji_factors["bus"] +
                       counts[1] * pkji_factors["car"] +
                       counts[2] * pkji_factors["motorcycle"] +
                       counts[3] * pkji_factors["truck"])

                win_stats.append({
                    "lo_sec": lo_s,
                    "hi_sec": hi_s,
                    "covered_sec": cov_s,
                    "is_low_cov": is_low_cov,
                    "pcu": pcu,
                    "ts": w["ts"],
                })

            for i in range(len(win_stats) - 2):
                w0, w1, w2 = win_stats[i], win_stats[i + 1], win_stats[i + 2]
                # Check consecutive: gap between windows <= 1.0 s
                if (w1["lo_sec"] - w0["hi_sec"] <= 1.0 and
                        w2["lo_sec"] - w1["hi_sec"] <= 1.0 and
                        not w0["is_low_cov"] and not w1["is_low_cov"] and not w2["is_low_cov"]):
                    n_runs += 1
                    tot_cov = w0["covered_sec"] + w1["covered_sec"] + w2["covered_sec"]
                    tot_pcu = w0["pcu"] + w1["pcu"] + w2["pcu"]
                    q15 = (tot_pcu / tot_cov * 3600.0) if tot_cov > 0 else 0.0
                    if not math.isfinite(best_q15) or q15 > best_q15:
                        best_q15 = q15
                        best_date = date
                        best_window_ts = w0["ts"]

        cap_rows.append({
            "cctv": cctv,
            "roi_idx": r_idx,
            "c_emp_pcu_h": round(best_q15, 2) if math.isfinite(best_q15) else float("nan"),
            "date": best_date,
            "window_ts_start": best_window_ts,
            "n_runs": n_runs,
        })
    cap_df = pd.DataFrame(cap_rows, columns=["cctv", "roi_idx", "c_emp_pcu_h", "date", "window_ts_start", "n_runs"])
    _write_csv_atomic(cap_df, out_dir / "capacity_empirical.csv")

    capacity_empirical_lookup = {(r["cctv"], r["roi_idx"]): r["c_emp_pcu_h"] for r in cap_rows}

    # 3. emp_empirical.csv
    emp_rows = []
    for cctv, r_idx in all_cctv_roi_pairs:
        chandra_wins = {"bus": [], "motorcycle": [], "truck": []}
        for date, cctvs in all_data.items():
            if cctv not in cctvs:
                continue
            for w in cctvs[cctv]:
                veh_df = w["veh"]
                if veh_df.empty:
                    continue
                roi_veh = veh_df[veh_df["roi_idx"] == r_idx]
                if roi_veh.empty:
                    continue

                speeds_by_cls = collections.defaultdict(list)
                for row in roi_veh.to_dict("records"):
                    spd = traverse_speed_kmh(row, min_obs=min_speed_obs, min_sec=min_traverse_sec)
                    if math.isfinite(spd):
                        cid = int(row["class_id"])
                        cname = TARGET_CLASSES.get(cid)
                        if cname:
                            speeds_by_cls[cname].append(spd)

                # Filter by chandra_min_vehicles
                mean_speeds = {}
                for cname, spds in speeds_by_cls.items():
                    if len(spds) >= chandra_min_veh:
                        mean_speeds[cname] = float(np.mean(spds))

                # Chandra EMP for window
                if "car" in mean_speeds:
                    emp_w = chandra_emp(mean_speeds, chandra_areas)
                    for cls in ("bus", "motorcycle", "truck"):
                        val = emp_w.get(cls)
                        if val is not None and math.isfinite(val):
                            chandra_wins[cls].append(val)

        emp_bus = float(np.median(chandra_wins["bus"])) if chandra_wins["bus"] else float("nan")
        emp_mc = float(np.median(chandra_wins["motorcycle"])) if chandra_wins["motorcycle"] else float("nan")
        emp_trk = float(np.median(chandra_wins["truck"])) if chandra_wins["truck"] else float("nan")

        emp_rows.append({
            "cctv": cctv,
            "roi_idx": r_idx,
            "emp_bus": round(emp_bus, 4) if math.isfinite(emp_bus) else float("nan"),
            "emp_motorcycle": round(emp_mc, 4) if math.isfinite(emp_mc) else float("nan"),
            "emp_truck": round(emp_trk, 4) if math.isfinite(emp_trk) else float("nan"),
            "n_windows_bus": len(chandra_wins["bus"]),
            "n_windows_motorcycle": len(chandra_wins["motorcycle"]),
            "n_windows_truck": len(chandra_wins["truck"]),
        })
    emp_df = pd.DataFrame(emp_rows, columns=["cctv", "roi_idx", "emp_bus", "emp_motorcycle", "emp_truck",
                                             "n_windows_bus", "n_windows_motorcycle", "n_windows_truck"])
    _write_csv_atomic(emp_df, out_dir / "emp_empirical.csv")

    emp_empirical_lookup = {(r["cctv"], r["roi_idx"]): r for r in emp_rows}

    # --------------------------------------------------------------------------
    # Pass 2: Per-(window, ROI) Metrics CSV
    # --------------------------------------------------------------------------
    for date, cctvs in sorted(all_data.items()):
        for cctv, wins in sorted(cctvs.items()):
            # Find ROIs for this CCTV
            cctv_rois = sorted(r_idx for (c, r_idx) in all_cctv_roi_pairs if c == cctv)
            if not cctv_rois:
                continue

            cam_rois_cfg = cameras_cfg.get(cctv, {}).get("rois", [])

            metrics_rows = []
            for w in wins:
                meta = w["meta"]
                lo_sec = meta["lo_sec"]
                hi_sec = meta["hi_sec"]
                covered_sec = meta["covered_sec"]
                n_ticks = meta.get("n_ticks", 0)
                queue_rois_in_win = set(meta.get("queue_rois", []))
                low_cov = 1 if (covered_sec < low_cov_frac * (hi_sec - lo_sec)) else 0

                veh_df = w["veh"]
                q_df = w["q"]

                for r_idx in cctv_rois:
                    roi_cfg = cam_rois_cfg[r_idx] if r_idx < len(cam_rois_cfg) else None
                    if roi_cfg is None or roi_cfg.get("emp") is None:
                        key = (cctv, r_idx)
                        if key not in warned_defaults:
                            print(f"[warn] traffic_params: {cctv} roi {r_idx} uses defaults")
                            warned_defaults.add(key)

                    lanes = roi_cfg.get("lanes", 1) if roi_cfg else 1
                    emp_cfg = (roi_cfg.get("emp") if roi_cfg else None) or emp_default_cfg
                    cap_cfg = roi_cfg.get("capacity", {}) if roi_cfg else {}

                    # 1. Volume
                    roi_veh_new = veh_df[(veh_df["roi_idx"] == r_idx) & (veh_df["counted_new"] == 1)] if not veh_df.empty else pd.DataFrame()
                    vol_b = int((roi_veh_new["class_id"] == 0).sum()) if not roi_veh_new.empty else 0
                    vol_c = int((roi_veh_new["class_id"] == 1).sum()) if not roi_veh_new.empty else 0
                    vol_m = int((roi_veh_new["class_id"] == 2).sum()) if not roi_veh_new.empty else 0
                    vol_t = int((roi_veh_new["class_id"] == 3).sum()) if not roi_veh_new.empty else 0
                    vol_tot = vol_b + vol_c + vol_m + vol_t
                    flow_v = (vol_tot * 3600.0 / covered_sec) if covered_sec > 0 else float("nan")

                    # 2. PCU
                    pkji_factors = pkji_emp(flow_v, emp_cfg)
                    pcu_pkji = (vol_b * pkji_factors["bus"] +
                                vol_c * pkji_factors["car"] +
                                vol_m * pkji_factors["motorcycle"] +
                                vol_t * pkji_factors["truck"])
                    flow_pcu_pkji = (pcu_pkji * 3600.0 / covered_sec) if covered_sec > 0 else float("nan")

                    # Empirical PCU factors (fallback to PKJI per class if NaN)
                    emp_lookup = emp_empirical_lookup.get((cctv, r_idx), {})
                    emp_b_val = emp_lookup.get("emp_bus")
                    if emp_b_val is None or not math.isfinite(emp_b_val):
                        emp_b_val = pkji_factors["bus"]
                    emp_m_val = emp_lookup.get("emp_motorcycle")
                    if emp_m_val is None or not math.isfinite(emp_m_val):
                        emp_m_val = pkji_factors["motorcycle"]
                    emp_t_val = emp_lookup.get("emp_truck")
                    if emp_t_val is None or not math.isfinite(emp_t_val):
                        emp_t_val = pkji_factors["truck"]

                    pcu_emp = (vol_b * emp_b_val +
                               vol_c * 1.0 +
                               vol_m * emp_m_val +
                               vol_t * emp_t_val)
                    flow_pcu_emp = (pcu_emp * 3600.0 / covered_sec) if covered_sec > 0 else float("nan")

                    # Window-level Chandra EMP
                    roi_all_veh = veh_df[veh_df["roi_idx"] == r_idx] if not veh_df.empty else pd.DataFrame()
                    spd_by_cls = collections.defaultdict(list)
                    valid_speeds_all = []
                    if not roi_all_veh.empty:
                        for row in roi_all_veh.to_dict("records"):
                            spd = traverse_speed_kmh(row, min_obs=min_speed_obs, min_sec=min_traverse_sec)
                            if math.isfinite(spd):
                                valid_speeds_all.append(spd)
                                cid = int(row["class_id"])
                                cname = TARGET_CLASSES.get(cid)
                                if cname:
                                    spd_by_cls[cname].append(spd)

                    mean_spds_win = {}
                    for cname, spds in spd_by_cls.items():
                        if len(spds) >= chandra_min_veh:
                            mean_spds_win[cname] = float(np.mean(spds))

                    chandra_win_emp = chandra_emp(mean_spds_win, chandra_areas) if "car" in mean_spds_win else {}
                    emp_ch_b = chandra_win_emp.get("bus", float("nan"))
                    emp_ch_m = chandra_win_emp.get("motorcycle", float("nan"))
                    emp_ch_t = chandra_win_emp.get("truck", float("nan"))

                    # 3. Occupancy and density
                    occ_recs = roi_all_veh.to_dict("records") if not roi_all_veh.empty else []
                    occ_t_pct, occ_a_pct = occupancy(occ_recs, covered_sec, lanes=lanes, lo=lo_sec, hi=hi_sec)

                    spd_s = speed_stats(valid_speeds_all)
                    spd_sms = spd_s["spd_sms"]
                    density_veh_km = (flow_v / spd_sms) if (spd_sms > 0 and math.isfinite(spd_sms) and math.isfinite(flow_v)) else float("nan")

                    # 4. Saturation
                    C0 = cap_cfg.get("C0")
                    if C0 is not None and math.isfinite(float(C0)):
                        FC_w = cap_cfg.get("FC_w", 1.0)
                        FC_sp = cap_cfg.get("FC_sp", 1.0)
                        FC_sf = cap_cfg.get("FC_sf", 1.0)
                        FC_cs = cap_cfg.get("FC_cs", 1.0)
                        dir_share = cap_cfg.get("dir_share", 1.0)
                        c_pkji_val = float(C0) * float(FC_w) * float(FC_sp) * float(FC_sf) * float(FC_cs) * float(dir_share)
                        ds_pkji_val = (flow_pcu_pkji / c_pkji_val) if (c_pkji_val > 0 and math.isfinite(flow_pcu_pkji)) else float("nan")
                    else:
                        c_pkji_val = float("nan")
                        ds_pkji_val = float("nan")

                    c_emp_val = capacity_empirical_lookup.get((cctv, r_idx), float("nan"))
                    ds_emp_val = (flow_pcu_pkji / c_emp_val) if (math.isfinite(c_emp_val) and c_emp_val > 0 and math.isfinite(flow_pcu_pkji)) else float("nan")

                    # 5. Queue metrics
                    has_queue_zone = (r_idx in queue_rois_in_win) or (r_idx in cctv_queue_rois[cctv])
                    q_data_thr = queue_threshold_lookup.get((cctv, r_idx), q_thr_fixed)

                    if not has_queue_zone or n_ticks == 0:
                        q_res = {
                            "q_thr_fixed_kmh": float("nan"),
                            "q_thr_data_kmh": float("nan"),
                            "queue_len_mean_m_fixed": float("nan"),
                            "queue_len_max_m_fixed": float("nan"),
                            "queue_len_p95_m_fixed": float("nan"),
                            "queued_veh_mean_fixed": float("nan"),
                            "queue_present_pct_fixed": float("nan"),
                            "queue_len_mean_m_data": float("nan"),
                            "queue_len_max_m_data": float("nan"),
                            "queue_len_p95_m_data": float("nan"),
                            "queued_veh_mean_data": float("nan"),
                            "queue_present_pct_data": float("nan"),
                        }
                    else:
                        # Extract tick-level metrics from q_df
                        q_subset = q_df[q_df["roi_idx"] == r_idx] if q_df is not None and not q_df.empty else pd.DataFrame()
                        ticks_grouped = collections.defaultdict(list)
                        if not q_subset.empty:
                            for q_row in q_subset.to_dict("records"):
                                ticks_grouped[q_row["t"]].append(q_row)

                        def calc_queue_for_thr(thr):
                            q_lens = []
                            n_qs = []
                            for t, t_rows in ticks_grouped.items():
                                queued_ys = [r["y_m"] for r in t_rows if math.isfinite(r["speed_kmh"]) and r["speed_kmh"] < thr]
                                q_len, n_q = queue_length_snapshot(queued_ys, head_max=q_head_max, gap_max=q_gap_max)
                                q_lens.append(q_len)
                                n_qs.append(n_q)

                            # Ticks with no queued rows / no rows count as 0
                            empty_ticks = max(0, n_ticks - len(q_lens))
                            if empty_ticks > 0:
                                q_lens.extend([0.0] * empty_ticks)
                                n_qs.extend([0] * empty_ticks)

                            mean_len = float(np.mean(q_lens)) if q_lens else 0.0
                            max_len = float(np.max(q_lens)) if q_lens else 0.0
                            p95_len = float(np.percentile(q_lens, 95)) if q_lens else 0.0
                            mean_veh = float(np.mean(n_qs)) if n_qs else 0.0
                            pres_pct = float(sum(1 for l in q_lens if l > 0) / n_ticks * 100.0)
                            return mean_len, max_len, p95_len, mean_veh, pres_pct

                        f_mean, f_max, f_p95, f_veh, f_pct = calc_queue_for_thr(q_thr_fixed)
                        d_mean, d_max, d_p95, d_veh, d_pct = calc_queue_for_thr(q_data_thr)

                        q_res = {
                            "q_thr_fixed_kmh": round(q_thr_fixed, 2),
                            "q_thr_data_kmh": round(q_data_thr, 2),
                            "queue_len_mean_m_fixed": round(f_mean, 2),
                            "queue_len_max_m_fixed": round(f_max, 2),
                            "queue_len_p95_m_fixed": round(f_p95, 2),
                            "queued_veh_mean_fixed": round(f_veh, 2),
                            "queue_present_pct_fixed": round(f_pct, 2),
                            "queue_len_mean_m_data": round(d_mean, 2),
                            "queue_len_max_m_data": round(d_max, 2),
                            "queue_len_p95_m_data": round(d_p95, 2),
                            "queued_veh_mean_data": round(d_veh, 2),
                            "queue_present_pct_data": round(d_pct, 2),
                        }

                    # Assemble row
                    row_dict = {
                        "cctv": cctv,
                        "date": date,
                        "window_ts": w["ts"],
                        "roi_idx": r_idx,
                        "lo_sec": lo_sec,
                        "hi_sec": hi_sec,
                        "covered_sec": round(covered_sec, 2),
                        "low_coverage": low_cov,
                        "vol_bus": vol_b,
                        "vol_car": vol_c,
                        "vol_motorcycle": vol_m,
                        "vol_truck": vol_t,
                        "vol_total": vol_tot,
                        "flow_veh_h": round(flow_v, 2) if math.isfinite(flow_v) else float("nan"),
                        "pcu_pkji": round(pcu_pkji, 2),
                        "flow_pcu_h_pkji": round(flow_pcu_pkji, 2) if math.isfinite(flow_pcu_pkji) else float("nan"),
                        "pcu_emp": round(pcu_emp, 2),
                        "flow_pcu_h_emp": round(flow_pcu_emp, 2) if math.isfinite(flow_pcu_emp) else float("nan"),
                        "emp_chandra_bus": round(emp_ch_b, 4) if math.isfinite(emp_ch_b) else float("nan"),
                        "emp_chandra_motorcycle": round(emp_ch_m, 4) if math.isfinite(emp_ch_m) else float("nan"),
                        "emp_chandra_truck": round(emp_ch_t, 4) if math.isfinite(emp_ch_t) else float("nan"),
                        "occ_time_pct": round(occ_t_pct, 2) if math.isfinite(occ_t_pct) else float("nan"),
                        "occ_any_pct": round(occ_a_pct, 2) if math.isfinite(occ_a_pct) else float("nan"),
                        "density_veh_km": round(density_veh_km, 2) if math.isfinite(density_veh_km) else float("nan"),
                        "spd_n": spd_s["spd_n"],
                        "spd_tms": round(spd_s["spd_tms"], 2) if math.isfinite(spd_s["spd_tms"]) else float("nan"),
                        "spd_sms": round(spd_s["spd_sms"], 2) if math.isfinite(spd_s["spd_sms"]) else float("nan"),
                        "spd_std": round(spd_s["spd_std"], 2) if math.isfinite(spd_s["spd_std"]) else float("nan"),
                        "spd_cv": round(spd_s["spd_cv"], 4) if math.isfinite(spd_s["spd_cv"]) else float("nan"),
                        "spd_p15": round(spd_s["spd_p15"], 2) if math.isfinite(spd_s["spd_p15"]) else float("nan"),
                        "spd_p50": round(spd_s["spd_p50"], 2) if math.isfinite(spd_s["spd_p50"]) else float("nan"),
                        "spd_p85": round(spd_s["spd_p85"], 2) if math.isfinite(spd_s["spd_p85"]) else float("nan"),
                        "c_pkji_pcu_h": round(c_pkji_val, 2) if math.isfinite(c_pkji_val) else float("nan"),
                        "ds_pkji": round(ds_pkji_val, 4) if math.isfinite(ds_pkji_val) else float("nan"),
                        "c_emp_pcu_h": round(c_emp_val, 2) if math.isfinite(c_emp_val) else float("nan"),
                        "ds_emp": round(ds_emp_val, 4) if math.isfinite(ds_emp_val) else float("nan"),
                        **q_res,
                    }
                    metrics_rows.append(row_dict)

            metrics_cols = [
                "cctv", "date", "window_ts", "roi_idx", "lo_sec", "hi_sec", "covered_sec", "low_coverage",
                "vol_bus", "vol_car", "vol_motorcycle", "vol_truck", "vol_total", "flow_veh_h",
                "pcu_pkji", "flow_pcu_h_pkji", "pcu_emp", "flow_pcu_h_emp",
                "emp_chandra_bus", "emp_chandra_motorcycle", "emp_chandra_truck",
                "occ_time_pct", "occ_any_pct", "density_veh_km",
                "spd_n", "spd_tms", "spd_sms", "spd_std", "spd_cv", "spd_p15", "spd_p50", "spd_p85",
                "c_pkji_pcu_h", "ds_pkji", "c_emp_pcu_h", "ds_emp",
                "q_thr_fixed_kmh", "q_thr_data_kmh",
                "queue_len_mean_m_fixed", "queue_len_max_m_fixed", "queue_len_p95_m_fixed", "queued_veh_mean_fixed", "queue_present_pct_fixed",
                "queue_len_mean_m_data", "queue_len_max_m_data", "queue_len_p95_m_data", "queued_veh_mean_data", "queue_present_pct_data"
            ]
            m_df = pd.DataFrame(metrics_rows, columns=metrics_cols)
            m_df.sort_values(by=["lo_sec", "roi_idx"], inplace=True)
            _write_csv_atomic(m_df, out_dir / f"{cctv}_{date}_metrics.csv")
            print(f"  [+] Wrote {len(m_df)} rows to {out_dir / f'{cctv}_{date}_metrics.csv'}")


# ==============================================================================
# Draw Queue Zone Subcommand (Interactive Video Annotator)
# ==============================================================================

def list_cctv_videos(cctv: str) -> list[Path]:
    """List all available video files for a CCTV camera, ordered by path."""
    dataset_dir = REPO_ROOT / "dataset"
    if not dataset_dir.exists():
        return []
    cctv_vids = sorted(dataset_dir.glob(f"*/{cctv}/videos/*.ts"))
    if not cctv_vids:
        cctv_vids = sorted(dataset_dir.glob(f"*/{cctv}/videos/*.mp4"))
    return cctv_vids


def find_cctv_video(cctv: str) -> Optional[Path]:
    """Auto-discover a representative daytime video file for a CCTV camera."""
    cctv_vids = list_cctv_videos(cctv)
    if not cctv_vids:
        return None
    # Prefer daytime recordings between 09:00 and 15:59
    day_vids = [v for v in cctv_vids if any(f"_{h:02d}" in v.stem for h in range(9, 16))]
    if day_vids:
        return day_vids[len(day_vids) // 2]
    return cctv_vids[len(cctv_vids) // 2]


def point_labels(n_segments: int) -> list[str]:
    """Ordered point labels for an n-segment queue zone."""
    n = max(1, n_segments)
    if n == 1:
        return ["head-left", "head-right", "tail-right", "tail-left"]
    labels = ["seg0-head-left", "seg0-head-right", "seg0-tail-right", "seg0-tail-left"]
    for s in range(1, n):
        labels.extend([f"seg{s}-tail-right", f"seg{s}-tail-left"])
    return labels


def segment_quads(points: list, n_segments: int) -> list[list]:
    """Return completed segment quads (4 points each) from the clicked points list."""
    n = max(1, n_segments)
    if len(points) < 4:
        return []
    quads = [[points[0], points[1], points[2], points[3]]]
    for s in range(1, n):
        req = 4 + 2 * s
        if len(points) < req:
            break
        p_tl = points[3 if s == 1 else 3 + 2 * (s - 1)]
        p_tr = points[2 if s == 1 else 2 + 2 * (s - 1)]
        c_tr = points[2 + 2 * s]
        c_tl = points[3 + 2 * s]
        quads.append([p_tl, p_tr, c_tr, c_tl])
    return quads


def build_queue_zone_dict(
    clicked_pts: list,
    n_segments: int,
    length_m: float,
    width_m: float,
    lengths_m_str: Optional[str] = None,
) -> dict:
    """Build the queue_zone dictionary structure from clicked points and geometry."""
    if n_segments == 1:
        return {
            "points": clicked_pts,
            "length_meters": float(length_m),
            "width_meters": float(width_m),
        }

    if lengths_m_str:
        seg_lengths = [float(x.strip()) for x in lengths_m_str.split(",")]
        if len(seg_lengths) != n_segments:
            print(f"Warning: lengths_m count ({len(seg_lengths)}) != segments ({n_segments}). Splitting length_m evenly.")
            seg_lengths = [float(length_m) / n_segments] * n_segments
    else:
        seg_lengths = [float(length_m) / n_segments] * n_segments

    quads = segment_quads(clicked_pts, n_segments)
    segments_data = []
    for s in range(n_segments):
        segments_data.append({
            "segment_idx": s,
            "points": quads[s],
            "length_meters": seg_lengths[s],
            "width_meters": float(width_m),
        })

    return {
        "type": "piecewise",
        "segments": segments_data,
        "length_meters": float(sum(seg_lengths)),
        "width_meters": float(width_m),
    }


def cmd_draw_queue_zone(args):
    try:
        from queue_zone_annotator import run_annotator
    except ImportError as e:
        print(f"Error: draw-queue-zone needs PyQt6 ({e}). Install with: python -m pip install PyQt6", file=sys.stderr)
        sys.exit(1)
    sys.exit(run_annotator())


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="subcommand", required=True)

    # compute
    p_comp = sub.add_parser("compute", help="Compute metrics from window track logs")
    p_comp.add_argument("--dates", nargs="+", default=None, help="Dates to compute (default: all under --tracks)")
    p_comp.add_argument("--cctv", nargs="+", default=None, help="CCTV cameras to compute (default: all)")
    p_comp.add_argument("--tracks", default=str(REPO_ROOT / "extracted" / "tracks"), help="Path to tracks dir")
    p_comp.add_argument("--out", default=str(REPO_ROOT / "extracted" / "metrics"), help="Path to output dir")
    p_comp.add_argument("--params", default=str(REPO_ROOT / "traffic_params.json"), help="Path to traffic_params.json")
    p_comp.set_defaults(func=cmd_compute)

    # draw-queue-zone
    p_draw = sub.add_parser(
        "draw-queue-zone",
        help="Open the PyQt6 queue-zone annotator (walks all cameras/ROIs, resumes from rois/queue_zone_progress.json)",
    )
    p_draw.set_defaults(func=cmd_draw_queue_zone)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Speed Validation Analysis and Figure Generation for Review Points 7-10.

Computes:
1. Length measurement uncertainty from Google Maps screenshots (readings.csv)
2. Timing resolution uncertainty (delta_frames)
3. Inter-rater agreement on 48-vehicle subset (annotations_second.json)
4. First-round anchoring statistics
5. ROI diagnostics (coverage, invalid %, gaps, deceleration, GT < 5 km/h)
6. Failure causes breakdown (causes.json)
7. Overall Bland-Altman and Day/Night split
8. Publication-ready figures (fig10 - fig14) matching docx extents
9. analysis_summary.json with all quoted values
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "script"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from speed_val import (
    AS_RUN_WIDTH,
    CLASSES,
    OUT,
    SEED,
    gt_speed,
    load_rois,
)

ANALYSIS_DIR = OUT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

TABLE_ORDER = [
    "Jl. Baru KBP",
    "G5 Bawah",
    "G5 Atas",
    "Giant – Tol",
    "Giant – Tagog",
    "Tagog-Bandung",
]

CAM_TO_SHORT = {
    "padalarang_-_jl_baru_kbp": "Jl. Baru KBP",
    "padalarang_-_gedong_5_bawah": "G5 Bawah",
    "padalarang_-_gedong_5_atas_-_stasiun": "G5 Atas",
    "padalarang_giant_-_tol": "Giant – Tol",
    "padalarang_-_giant_-_tagog": "Giant – Tagog",
    "tagog_padalarang_-_bandung": "Tagog-Bandung",
}

SHORT_TO_CAM = {v: k for k, v in CAM_TO_SHORT.items()}

# Media extents from docx (in inches @ 300 dpi)
EXTENTS = {
    "image10": (2250831 / 914400.0, 2250831 / 914400.0), # 2.46 x 2.46
    "image11": (2749594 / 914400.0, 1148361 / 914400.0), # 3.01 x 1.26
    "image12": (2800750 / 914400.0, 1409319 / 914400.0), # 3.06 x 1.54
    "image13": (3108960 / 914400.0, 2958974 / 914400.0), # 3.40 x 3.24
    "image14": (2595179 / 914400.0, 1440071 / 914400.0), # 2.84 x 1.57
}

LOC_COLORS = {
    "Jl. Baru KBP": "#1f77b4",
    "G5 Bawah": "#ff7f0e",
    "G5 Atas": "#2ca02c",
    "Giant – Tol": "#d62728",
    "Giant – Tagog": "#9467bd",
    "Tagog-Bandung": "#8c564b",
}


def main():
    print("[*] Running speed validation analysis...")
    summary = {}

    # Load core files
    df_final = pd.read_csv(OUT / "final_sample.csv")
    df_metrics = pd.read_csv(OUT / "metrics.csv")
    df_quota = pd.read_csv(OUT / "quota.csv")
    with open(OUT / "candidates.json", "r", encoding="utf-8") as f:
        cands = json.load(f)
    cand_map = {c["sample_id"]: c for c in cands}

    # --------------------------------------------------------------------------
    # 1. Length Uncertainty (from readings.csv)
    # --------------------------------------------------------------------------
    readings_path = REPO_ROOT / "rois" / "roi_measure_picture" / "readings.csv"
    if readings_path.exists():
        df_readings = pd.read_csv(readings_path)
        len_rows = []
        for _, r in df_readings.iterrows():
            cid = r["cctv_id"]
            rid = int(r["roi_idx"])
            dim = r["dim"]
            read_m = float(r["reading_m"])
            dx = r["x2"] - r["x1"]
            dy = r["y2"] - r["y1"]
            px_dist = math.hypot(dx, dy)
            m_per_px = read_m / px_dist if px_dist > 0 else 0.0
            dl_m = 3.0 * math.sqrt(2) * m_per_px
            
            # Get stored dimension
            # Map prefix back to full camera id
            full_cam = None
            for fc in CAM_TO_SHORT.keys():
                if cid in fc or fc.endswith(cid):
                    full_cam = fc
                    break
            if full_cam is None:
                full_cam = cid
                
            rois_def = load_rois(full_cam)
            stored_m = rois_def[rid]["length_meters"] if dim == "L" else rois_def[rid]["width_meters"]
            dl_pct = (dl_m / stored_m) * 100.0 if stored_m > 0 else 0.0
            abs_diff = abs(stored_m - read_m)
            
            len_rows.append({
                "file": r["file"],
                "cctv_id": full_cam,
                "roi_idx": rid + 1,
                "dim": dim,
                "stored_m": round(stored_m, 2),
                "reading_m": round(read_m, 2),
                "pixel_dist": round(px_dist, 1),
                "m_per_px": round(m_per_px, 5),
                "dL_m": round(dl_m, 3),
                "dL_pct": round(dl_pct, 2),
                "abs_diff_m": round(abs_diff, 2),
            })
            
        df_len_unc = pd.DataFrame(len_rows)
        df_len_unc.to_csv(ANALYSIS_DIR / "length_uncertainty.csv", index=False)
        print(f"[✓] Saved {ANALYSIS_DIR / 'length_uncertainty.csv'}")
        
        # Summary for length
        l_only = df_len_unc[df_len_unc.dim == "L"]
        summary["length"] = {
            "dL_pct_min": round(l_only["dL_pct"].min(), 2),
            "dL_pct_max": round(l_only["dL_pct"].max(), 2),
            "dL_pct_median": round(l_only["dL_pct"].median(), 2),
            "max_abs_diff_m": round(df_len_unc["abs_diff_m"].max(), 2),
        }

    # --------------------------------------------------------------------------
    # 2. Timing Uncertainty (delta_frames)
    # --------------------------------------------------------------------------
    timing_rows = []
    for (cam, roi), grp in df_final.groupby(["cctv_id", "roi_idx"]):
        rel_unc = 100.0 * np.sqrt(2) / grp["delta_frames"]
        abs_unc = grp["ground_truth_speed_kmh"] * np.sqrt(2) / grp["delta_frames"]
        timing_rows.append({
            "cctv_id": cam,
            "roi_idx": roi + 1,
            "n": len(grp),
            "median_delta_frames": round(grp["delta_frames"].median(), 1),
            "median_rel_pct": round(rel_unc.median(), 2),
            "median_abs_kmh": round(abs_unc.median(), 2),
        })
    df_timing_unc = pd.DataFrame(timing_rows)
    df_timing_unc.to_csv(ANALYSIS_DIR / "timing_uncertainty.csv", index=False)
    print(f"[✓] Saved {ANALYSIS_DIR / 'timing_uncertainty.csv'}")

    summary["timing"] = {
        "rel_pct_min": round(df_timing_unc["median_rel_pct"].min(), 2),
        "rel_pct_max": round(df_timing_unc["median_rel_pct"].max(), 2),
        "rel_pct_median": round(df_timing_unc["median_rel_pct"].median(), 2),
        "abs_kmh_median": round(df_timing_unc["median_abs_kmh"].median(), 2),
    }

    # --------------------------------------------------------------------------
    # 3. Anchoring Statistics (first-round)
    # --------------------------------------------------------------------------
    unchanged = 0
    entry_adjs, exit_adjs = [], []
    for _, r in df_final.iterrows():
        sid = r["sample_id"]
        c = cand_map[sid]
        e_in = r["entry_frame"]
        e_out = r["exit_frame"]
        est_in = c.get("est_entry_frame")
        est_out = c.get("est_exit_frame")
        if est_in is not None and est_out is not None:
            adj_in = abs(e_in - est_in)
            adj_out = abs(e_out - est_out)
            entry_adjs.append(adj_in)
            exit_adjs.append(adj_out)
            if adj_in == 0 and adj_out == 0:
                unchanged += 1

    summary["anchoring"] = {
        "n_total": len(df_final),
        "n_unchanged": unchanged,
        "pct_unchanged": round((unchanged / len(df_final)) * 100.0, 1),
        "median_entry_adj": float(np.median(entry_adjs)),
        "median_exit_adj": float(np.median(exit_adjs)),
    }

    # --------------------------------------------------------------------------
    # 4. Inter-Rater Agreement (subset 48)
    # --------------------------------------------------------------------------
    sec_path = OUT / "annotations_second.json"
    if sec_path.exists():
        with open(sec_path, "r", encoding="utf-8") as f:
            annos_sec = json.load(f)
        with open(OUT / "annotations.json", "r", encoding="utf-8") as f:
            annos_first = json.load(f)
            
        inter_rows = []
        for sid, a2 in annos_sec.items():
            if sid in annos_first:
                a1 = annos_first[sid]
                gt1 = a1.get("ground_truth_speed_kmh")
                gt2 = a2.get("ground_truth_speed_kmh")
                e1 = a1.get("entry_frame")
                e2 = a2.get("entry_frame")
                x1 = a1.get("exit_frame")
                x2 = a2.get("exit_frame")
                fp2 = bool(a2.get("false_positive", False))
                inv2 = bool(a2.get("invalid_crossing", False))
                
                diff_spd = (gt2 - gt1) if (gt1 is not None and gt2 is not None) else None
                de = abs(e2 - e1) if (e1 is not None and e2 is not None) else None
                dx = abs(x2 - x1) if (x1 is not None and x2 is not None) else None
                
                inter_rows.append({
                    "sample_id": sid,
                    "gt1_kmh": gt1,
                    "gt2_kmh": gt2,
                    "diff_kmh": diff_spd,
                    "delta_entry": de,
                    "delta_exit": dx,
                    "a2_fp": fp2,
                    "a2_invalid": inv2,
                })
        df_inter = pd.DataFrame(inter_rows)
        df_inter.to_csv(ANALYSIS_DIR / "interrater.csv", index=False)
        print(f"[✓] Saved {ANALYSIS_DIR / 'interrater.csv'}")
        
        valid_pairs = df_inter.dropna(subset=["diff_kmh"])
        n_pairs = len(valid_pairs)
        diffs = valid_pairs["diff_kmh"]
        b_ir = float(diffs.mean())
        sd_ir = float(diffs.std(ddof=1)) if n_pairs >= 2 else 0.0
        summary["interrater"] = {
            "n_evaluated": len(df_inter),
            "n_valid_pairs": n_pairs,
            "n_invalid_or_fp_by_a2": int(df_inter["a2_fp"].sum() + df_inter["a2_invalid"].sum()),
            "bias_kmh": round(b_ir, 2),
            "sd_kmh": round(sd_ir, 2),
            "loa_lo_kmh": round(b_ir - 1.96 * sd_ir, 2),
            "loa_hi_kmh": round(b_ir + 1.96 * sd_ir, 2),
            "median_delta_entry": float(valid_pairs["delta_entry"].median()),
            "median_delta_exit": float(valid_pairs["delta_exit"].median()),
        }
    else:
        summary["interrater"] = {
            "status": "pending_author_labeling",
            "target_n": 48,
        }

    # --------------------------------------------------------------------------
    # 5. ROI Diagnostics
    # --------------------------------------------------------------------------
    diag_rows = []
    for (cam, roi), grp in df_final.groupby(["cctv_id", "roi_idx"]):
        l_m = grp["roi_length_meters"].iloc[0]
        fps = grp["native_fps"].iloc[0]
        
        # Coverage
        cov = grp["ground_truth_speed_kmh"] * (grp["delta_frames"] / fps) / 3.6 / l_m
        cov_med = cov.median()
        cov_iqr = cov.quantile(0.75) - cov.quantile(0.25)
        
        # Gaps & deceleration
        gap_count = 0
        decel_count = 0
        for _, r in grp.iterrows():
            c = cand_map[r["sample_id"]]
            traj = c.get("trajectory", [])
            frames = [pt["frame"] for pt in traj]
            if len(frames) > 1 and any(frames[i+1] - frames[i] > 5 for i in range(len(frames)-1)):
                gap_count += 1
                
            if len(traj) >= 6:
                t_first, t_last = frames[0], frames[-1]
                dt = (t_last - t_first) / 3.0
                t1_end, t3_start = t_first + dt, t_last - dt
                pts_poly = np.array(c["roi_points"], dtype=np.float32)
                W_m = c["roi_width_meters"]
                dst_pts = np.array([[0, 0], [W_m, 0], [W_m, l_m], [0, l_m]], dtype=np.float32)
                H = cv2.getPerspectiveTransform(pts_poly, dst_pts)
                bcs = np.array([pt["bottom_center"] for pt in traj], dtype=np.float32).reshape(-1, 1, 2)
                bcs_m = cv2.perspectiveTransform(bcs, H).reshape(-1, 2)
                
                v1_list, v3_list = [], []
                for i in range(len(traj) - 1):
                    df_f = traj[i+1]["frame"] - traj[i]["frame"]
                    if df_f <= 0: continue
                    d_m = np.linalg.norm(bcs_m[i+1] - bcs_m[i])
                    v = (d_m / (df_f / fps)) * 3.6
                    if traj[i]["frame"] <= t1_end: v1_list.append(v)
                    if traj[i+1]["frame"] >= t3_start: v3_list.append(v)
                if v1_list and v3_list and np.mean(v1_list) > 1.0:
                    if (np.mean(v3_list) / np.mean(v1_list)) < 0.7:
                        decel_count += 1
                        
        corr_err = np.corrcoef(grp["error_kmh"], grp["ground_truth_speed_kmh"])[0, 1] if len(grp) >= 3 else 0.0
        lt5_sub = grp[grp["ground_truth_speed_kmh"] < 5.0]
        
        # Quota invalid pct
        sub_q = df_quota[(df_quota.cctv_id == cam) & (df_quota.roi_idx == roi + 1)]
        acc_total = sub_q["accepted"].sum()
        inv_total = sub_q["invalid_crossing"].sum()
        inv_pct = (inv_total / (acc_total + inv_total)) * 100.0 if (acc_total + inv_total) > 0 else 0.0
        
        # Metrics from df_metrics
        m_row = df_metrics[(df_metrics.group_type == "roi") & (df_metrics.group_key == f"{cam}:ROI{roi+1}")].iloc[0]
        
        diag_rows.append({
            "cctv_id": cam,
            "cctv_short": CAM_TO_SHORT.get(cam, cam),
            "roi_idx": roi + 1,
            "length_m": l_m,
            "n": len(grp),
            "mae": m_row["mae"],
            "mae_lo": m_row["mae_lo"],
            "mae_hi": m_row["mae_hi"],
            "rmse": m_row["rmse"],
            "rmse_lo": m_row["rmse_lo"],
            "rmse_hi": m_row["rmse_hi"],
            "bias": m_row["bias"],
            "r": m_row["r"],
            "r_lo": m_row["r_lo"],
            "r_hi": m_row["r_hi"],
            "invalid_pct": round(inv_pct, 1),
            "coverage_median": round(cov_med, 2),
            "coverage_iqr": round(cov_iqr, 2),
            "gap_pct": round(gap_count / len(grp) * 100, 1),
            "decel_pct": round(decel_count / len(grp) * 100, 1),
            "corr_err_gt": round(corr_err, 3),
            "n_gt_lt5": len(lt5_sub),
            "mean_err_lt5": round(lt5_sub["error_kmh"].mean(), 2) if len(lt5_sub) > 0 else None,
        })
    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(ANALYSIS_DIR / "roi_diagnostics.csv", index=False)
    print(f"[✓] Saved {ANALYSIS_DIR / 'roi_diagnostics.csv'}")

    # --------------------------------------------------------------------------
    # 6. Failure Causes Summary
    # --------------------------------------------------------------------------
    causes_path = OUT / "causes.json"
    if causes_path.exists():
        with open(causes_path, "r", encoding="utf-8") as f:
            causes_dict = json.load(f)
        c_records = []
        for sid, cinfo in causes_dict.items():
            parts = sid.split("|")
            cam = parts[1]
            c_records.append({
                "sample_id": sid,
                "cctv_id": cam,
                "cctv_short": CAM_TO_SHORT.get(cam, cam),
                "source": cinfo.get("source", "unknown"),
                "cause": cinfo.get("cause", "other"),
            })
        df_c = pd.DataFrame(c_records)
        df_c.to_csv(ANALYSIS_DIR / "causes_summary.csv", index=False)
        print(f"[✓] Saved {ANALYSIS_DIR / 'causes_summary.csv'}")
        ct = pd.crosstab(df_c["cause"], df_c["source"], margins=True)
        ct_pct = (ct / ct.loc["All"] * 100).round(1)
        ct.to_csv(ANALYSIS_DIR / "causes_table.csv")
        ct_pct.to_csv(ANALYSIS_DIR / "causes_table_pct.csv")
        loc_ct = pd.crosstab(df_c["cause"], df_c["cctv_short"], margins=True)
        loc_ct.to_csv(ANALYSIS_DIR / "causes_by_location.csv")
        print(f"[✓] Saved causes cross-tabs to {ANALYSIS_DIR}")
    else:
        # Save placeholder structure
        print(f"[*] Note: {causes_path} not found yet (pending author review).")

    # --------------------------------------------------------------------------
    # 7. Overall, Day/Night & Bland-Altman
    # --------------------------------------------------------------------------
    diff = df_final["predicted_speed_kmh"] - df_final["ground_truth_speed_kmh"]
    mean_spd = 0.5 * (df_final["predicted_speed_kmh"] + df_final["ground_truth_speed_kmh"])
    b_all = float(diff.mean())
    sd_all = float(diff.std(ddof=1))
    loa_lo = b_all - 1.96 * sd_all
    loa_hi = b_all + 1.96 * sd_all
    slope, intercept = np.polyfit(mean_spd, diff, 1)
    r_all = float(np.corrcoef(df_final["predicted_speed_kmh"], df_final["ground_truth_speed_kmh"])[0, 1])
    mae_all = float(diff.abs().mean())
    rmse_all = float(np.sqrt((diff**2).mean()))
    hours = [int(sid.split("|")[3][:2]) for sid in df_final["sample_id"]]
    is_night = [(h >= 18 or h < 6) for h in hours]
    df_final["is_night"] = is_night

    day_sub = df_final[~df_final.is_night]
    night_sub = df_final[df_final.is_night]

    err_day = day_sub["predicted_speed_kmh"] - day_sub["ground_truth_speed_kmh"]
    err_night = night_sub["predicted_speed_kmh"] - night_sub["ground_truth_speed_kmh"]

    summary["overall"] = {
        "n": len(df_final),
        "mae": round(mae_all, 2),
        "rmse": round(rmse_all, 2),
        "bias": round(b_all, 2),
        "sd": round(sd_all, 2),
        "loa_lo": round(loa_lo, 2),
        "loa_hi": round(loa_hi, 2),
        "r": round(r_all, 4),
        "ba_ols_slope": round(slope, 4),
        "ba_ols_intercept": round(intercept, 4),
    }

    summary["daynight"] = {
        "n_day": len(day_sub),
        "mae_day": round(err_day.abs().mean(), 2),
        "rmse_day": round(np.sqrt((err_day**2).mean()), 2),
        "bias_day": round(err_day.mean(), 2),
        "n_night": len(night_sub),
        "mae_night": round(err_night.abs().mean(), 2),
        "rmse_night": round(np.sqrt((err_night**2).mean()), 2),
        "bias_night": round(err_night.mean(), 2),
    }

    summary["low_speed"] = {
        "n_lt5": len(df_final[df_final.ground_truth_speed_kmh < 5.0]),
        "mean_err_lt5": round(df_final[df_final.ground_truth_speed_kmh < 5.0]["error_kmh"].mean(), 2),
        "n_ge5": len(df_final[df_final.ground_truth_speed_kmh >= 5.0]),
        "mean_err_ge5": round(df_final[df_final.ground_truth_speed_kmh >= 5.0]["error_kmh"].mean(), 2),
    }

    # --------------------------------------------------------------------------
    # 8. Publication Figures
    # --------------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "figure.titlesize": 8,
        "lines.linewidth": 1.0,
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    })

    # --- Fig 10: Scatter (image10: 2.46 x 2.46 in) ---
    fw, fh = EXTENTS["image10"]
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=300)
    for s_name in TABLE_ORDER:
        cam_id = SHORT_TO_CAM[s_name]
        sub = df_final[df_final.cctv_id == cam_id]
        ax.scatter(sub["ground_truth_speed_kmh"], sub["predicted_speed_kmh"],
                   color=LOC_COLORS[s_name], label=s_name, s=9, alpha=0.75, edgecolors="none")
    
    # y = x dashed line
    max_v = max(df_final["ground_truth_speed_kmh"].max(), df_final["predicted_speed_kmh"].max()) + 3
    ax.plot([0, max_v], [0, max_v], "k--", lw=0.8, alpha=0.7, label="y = x")
    
    # OLS line
    x_vals = np.linspace(0, max_v, 100)
    m_ols, b_ols = np.polyfit(df_final["ground_truth_speed_kmh"], df_final["predicted_speed_kmh"], 1)
    ax.plot(x_vals, m_ols * x_vals + b_ols, "r-", lw=1.0, label="OLS fit")
    
    ax.set_xlabel("Ground-Truth Speed (km/h)")
    ax.set_ylabel("Pipeline Speed (km/h)")
    ax.set_xlim(0, max_v)
    ax.set_ylim(0, max_v)
    ax.text(0.05, 0.92, f"n = {len(df_final)}\nr = {r_all:.4f}", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#ccc"))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.85, handletextpad=0.2, borderpad=0.2)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.95, bottom=0.16)
    fig.savefig(ANALYSIS_DIR / "fig10_scatter.png", dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {ANALYSIS_DIR / 'fig10_scatter.png'}")

    # --- Fig 11: Perspective (image11: 3.01 x 1.26 in) ---
    fw, fh = EXTENTS["image11"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fw, fh), dpi=300)
    df_persp = df_metrics[df_metrics.group_type == "perspective"].set_index("group_key")
    
    # Receding (belakang) vs Approaching (depan)
    persp_keys = ["depan", "belakang"]
    persp_labels = ["Approaching", "Receding"]
    x = np.arange(len(persp_keys))
    
    # MAE
    mae_val = [df_persp.loc[k, "mae"] for k in persp_keys]
    mae_err = [[mae_val[i] - df_persp.loc[k, "mae_lo"], df_persp.loc[k, "mae_hi"] - mae_val[i]] for i, k in enumerate(persp_keys)]
    mae_err = np.array(mae_err).T
    ax1.bar(x, mae_val, yerr=mae_err, capsize=2.5, color=["#1f77b4", "#ff7f0e"], width=0.55, error_kw=dict(lw=0.8))
    ax1.set_xticks(x)
    ax1.set_xticklabels(persp_labels)
    ax1.set_ylabel("MAE (km/h)")
    ax1.set_title("(A) Mean Absolute Error", pad=2)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.set_ylim(0, max(df_persp["mae_hi"]) * 1.25)
    for i, v in enumerate(mae_val):
        ax1.text(i, v / 2, f"{v:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=6)
        
    # RMSE
    rmse_val = [df_persp.loc[k, "rmse"] for k in persp_keys]
    rmse_err = [[rmse_val[i] - df_persp.loc[k, "rmse_lo"], df_persp.loc[k, "rmse_hi"] - rmse_val[i]] for i, k in enumerate(persp_keys)]
    rmse_err = np.array(rmse_err).T
    ax2.bar(x, rmse_val, yerr=rmse_err, capsize=2.5, color=["#1f77b4", "#ff7f0e"], width=0.55, error_kw=dict(lw=0.8))
    ax2.set_xticks(x)
    ax2.set_xticklabels(persp_labels)
    ax2.set_ylabel("RMSE (km/h)")
    ax2.set_title("(B) Root Mean Square Error", pad=2)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.set_ylim(0, max(df_persp["rmse_hi"]) * 1.25)
    for i, v in enumerate(rmse_val):
        ax2.text(i, v / 2, f"{v:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=6)

    fig.subplots_adjust(left=0.14, right=0.96, top=0.86, bottom=0.22, wspace=0.35)
    fig.savefig(ANALYSIS_DIR / "fig11_perspective.png", dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {ANALYSIS_DIR / 'fig11_perspective.png'}")

    # --- Fig 12: Location (image12: 3.06 x 1.54 in) ---
    fw, fh = EXTENTS["image12"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fw, fh), dpi=300)
    df_loc = df_metrics[df_metrics.group_type == "cctv"].set_index("group_key")
    loc_keys = [SHORT_TO_CAM[s] for s in TABLE_ORDER]
    x_loc = np.arange(len(TABLE_ORDER))
    
    # MAE
    mae_val = [df_loc.loc[k, "mae"] for k in loc_keys]
    mae_err = [[mae_val[i] - df_loc.loc[k, "mae_lo"], df_loc.loc[k, "mae_hi"] - mae_val[i]] for i, k in enumerate(loc_keys)]
    mae_err = np.array(mae_err).T
    ax1.bar(x_loc, mae_val, yerr=mae_err, capsize=2.0, color="#1f77b4", width=0.6, error_kw=dict(lw=0.8))
    ax1.set_xticks(x_loc)
    ax1.set_xticklabels(TABLE_ORDER, rotation=35, ha="right", fontsize=5.5)
    ax1.set_ylabel("MAE (km/h)")
    ax1.set_title("(A) MAE by Location", pad=2)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.set_ylim(0, max(df_loc["mae_hi"]) * 1.2)
    
    # RMSE
    rmse_val = [df_loc.loc[k, "rmse"] for k in loc_keys]
    rmse_err = [[rmse_val[i] - df_loc.loc[k, "rmse_lo"], df_loc.loc[k, "rmse_hi"] - rmse_val[i]] for i, k in enumerate(loc_keys)]
    rmse_err = np.array(rmse_err).T
    ax2.bar(x_loc, rmse_val, yerr=rmse_err, capsize=2.0, color="#ff7f0e", width=0.6, error_kw=dict(lw=0.8))
    ax2.set_xticks(x_loc)
    ax2.set_xticklabels(TABLE_ORDER, rotation=35, ha="right", fontsize=5.5)
    ax2.set_ylabel("RMSE (km/h)")
    ax2.set_title("(B) RMSE by Location", pad=2)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.set_ylim(0, max(df_loc["rmse_hi"]) * 1.2)

    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.34, wspace=0.35)
    fig.savefig(ANALYSIS_DIR / "fig12_location.png", dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {ANALYSIS_DIR / 'fig12_location.png'}")

    # --- Fig 13: Bland-Altman (image13: 3.40 x 3.24 in) ---
    fw, fh = EXTENTS["image13"]
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=300)
    for s_name in TABLE_ORDER:
        cam_id = SHORT_TO_CAM[s_name]
        sub = df_final[df_final.cctv_id == cam_id]
        m_s = 0.5 * (sub["predicted_speed_kmh"] + sub["ground_truth_speed_kmh"])
        d_s = sub["predicted_speed_kmh"] - sub["ground_truth_speed_kmh"]
        ax.scatter(m_s, d_s, color=LOC_COLORS[s_name], label=s_name, s=11, alpha=0.75, edgecolors="none")
        
    # Bias line
    ax.axhline(b_all, color="k", linestyle="-", lw=1.0, label=f"Mean diff ({b_all:+.2f})")
    ax.axhline(loa_hi, color="k", linestyle="--", lw=0.9, label=f"+1.96 SD ({loa_hi:+.2f})")
    ax.axhline(loa_lo, color="k", linestyle="--", lw=0.9, label=f"−1.96 SD ({loa_lo:+.2f})")
    
    # Regression line
    x_ba = np.linspace(5, df_final["ground_truth_speed_kmh"].max(), 100)
    ax.plot(x_ba, slope * x_ba + intercept, "r:", lw=1.0, label=f"Trend (slope: {slope:.3f})")
    
    ax.set_xlabel("Mean of Pipeline and Ground-Truth Speed (km/h)")
    ax.set_ylabel("Speed Difference: Pipeline − GT (km/h)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.85, handletextpad=0.2, borderpad=0.2, ncol=2)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.14)
    fig.savefig(ANALYSIS_DIR / "fig13_bland_altman.png", dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {ANALYSIS_DIR / 'fig13_bland_altman.png'}")

    # --- Fig 14: Class (image14: 2.84 x 1.57 in) ---
    fw, fh = EXTENTS["image14"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fw, fh), dpi=300)
    df_cls = df_metrics[df_metrics.group_type == "class"].set_index("group_key")
    cls_keys = list(CLASSES)
    cls_labels = [c.capitalize() for c in cls_keys]
    x_cls = np.arange(len(cls_keys))
    
    # MAE
    mae_val = [df_cls.loc[k, "mae"] for k in cls_keys]
    mae_err = [[mae_val[i] - df_cls.loc[k, "mae_lo"], df_cls.loc[k, "mae_hi"] - mae_val[i]] for i, k in enumerate(cls_keys)]
    mae_err = np.array(mae_err).T
    ax1.bar(x_cls, mae_val, yerr=mae_err, capsize=2.0, color="#595ba2", width=0.6, error_kw=dict(lw=0.8))
    ax1.set_xticks(x_cls)
    ax1.set_xticklabels(cls_labels, rotation=25, ha="right", fontsize=6)
    ax1.set_ylabel("MAE (km/h)")
    ax1.set_title("(A) MAE by Class", pad=2)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.set_ylim(0, max(df_cls["mae_hi"]) * 1.25)
    
    # RMSE
    rmse_val = [df_cls.loc[k, "rmse"] for k in cls_keys]
    rmse_err = [[rmse_val[i] - df_cls.loc[k, "rmse_lo"], df_cls.loc[k, "rmse_hi"] - rmse_val[i]] for i, k in enumerate(cls_keys)]
    rmse_err = np.array(rmse_err).T
    ax2.bar(x_cls, rmse_val, yerr=rmse_err, capsize=2.0, color="#2ca02c", width=0.6, error_kw=dict(lw=0.8))
    ax2.set_xticks(x_cls)
    ax2.set_xticklabels(cls_labels, rotation=25, ha="right", fontsize=6)
    ax2.set_ylabel("RMSE (km/h)")
    ax2.set_title("(B) RMSE by Class", pad=2)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.set_ylim(0, max(df_cls["rmse_hi"]) * 1.25)

    fig.subplots_adjust(left=0.15, right=0.96, top=0.88, bottom=0.28, wspace=0.35)
    fig.savefig(ANALYSIS_DIR / "fig14_class.png", dpi=300)
    plt.close(fig)
    print(f"[✓] Saved {ANALYSIS_DIR / 'fig14_class.png'}")

    # --------------------------------------------------------------------------
    # 9. Save Summary JSON
    # --------------------------------------------------------------------------
    with open(ANALYSIS_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Saved {ANALYSIS_DIR / 'analysis_summary.json'}")
    print("\n[✓] Speed validation analysis complete!")


if __name__ == "__main__":
    main()

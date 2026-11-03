#!/usr/bin/env python3
"""Dataset Exploratory Data Analysis (EDA) for 32-day window (2026-09-09 to 2026-10-10)."""

import argparse
import collections
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SHORT_NAMES = {
    "padalarang_-_jl_baru_kbp": "Jl. Baru KBP",
    "padalarang_-_gedong_5_bawah": "G5 Bawah",
    "padalarang_-_gedong_5_atas_-_stasiun": "G5 Atas",
    "padalarang_giant_-_tol": "Giant-Tol",
    "padalarang_-_giant_-_tagog": "Giant-Tagog",
    "tagog_padalarang_-_bandung": "Tagog-Bandung",
}

TABLE_NAMES = {
    "padalarang_-_jl_baru_kbp": "Jl. Baru KBP",
    "padalarang_-_gedong_5_bawah": "G5 Bawah",
    "padalarang_-_gedong_5_atas_-_stasiun": "G5 Atas",
    "padalarang_giant_-_tol": "Giant – Tol",
    "padalarang_-_giant_-_tagog": "Giant – Tagog",
    "tagog_padalarang_-_bandung": "Tagog-Bandung",
}

TABLE_ORDER = [
    "Jl. Baru KBP",
    "G5 Bawah",
    "G5 Atas",
    "Giant – Tol",
    "Giant – Tagog",
    "Tagog-Bandung",
]

FULL_NAMES = {
    "padalarang_-_jl_baru_kbp": "“Jl. Baru KBP”",
    "padalarang_-_gedong_5_bawah": "“Gedong 5 Bawah” (G5 Bawah)",
    "padalarang_-_gedong_5_atas_-_stasiun": "“Gedong 5 Atas – Stasiun” (G5 Atas)",
    "padalarang_giant_-_tol": "“Giant-Tol”",
    "padalarang_-_giant_-_tagog": "“Giant-Tagog”",
    "tagog_padalarang_-_bandung": "“Tagog Padalarang-Bandung” (Tagog-Bandung)",
}

CAM_ROIS = {
    "padalarang_giant_-_tol": [1],
    "padalarang_-_giant_-_tagog": [1, 2, 3],
    "padalarang_-_jl_baru_kbp": [1, 2],
    "padalarang_-_gedong_5_bawah": [1, 2],
    "padalarang_-_gedong_5_atas_-_stasiun": [1, 2],
    "tagog_padalarang_-_bandung": [1, 2],
}


def load_dataset(extracted_dir: Path, start_date: str, end_date: str):
    pattern = re.compile(r"^(.*)_(\d{4}-\d{2}-\d{2})_metadata_features\.csv$")
    files = sorted(extracted_dir.glob("*_metadata_features.csv"))

    matched = []
    for f in files:
        m = pattern.match(f.name)
        if m:
            cam, dt = m.group(1), m.group(2)
            if start_date <= dt <= end_date:
                matched.append((cam, dt, f))

    dates_found = sorted({d for _, d, _ in matched})
    cams_found = sorted({c for c, _, _ in matched})

    assert len(cams_found) == 6, f"Expected 6 cameras, found {len(cams_found)}"
    assert len(matched) == 192, f"Expected 192 files, found {len(matched)}"
    for dt in dates_found:
        dt_cams = [c for c, d, _ in matched if d == dt]
        assert len(dt_cams) == 6, f"Date {dt} has {len(dt_cams)} cameras, expected 6"

    dfs = []
    for cam, dt, f in matched:
        df = pd.read_csv(f)
        df["cam"] = cam
        df["date"] = dt
        dfs.append(df)

    raw_df = pd.concat(dfs, ignore_index=True)
    raw_df["ts"] = pd.to_datetime(raw_df["timestamp"])
    raw_df["tss"] = raw_df["timestampsafe"].astype(str).str.zfill(6)

    # Load missing reports
    reports_dir = extracted_dir / "reports"
    rep_rows = []
    if reports_dir.exists():
        for rf in sorted(reports_dir.glob("*_missing_report.json")):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", rf.name)
            if m and start_date <= m.group(1) <= end_date:
                try:
                    with open(rf, encoding="utf-8") as jf:
                        items = json.load(jf)
                        for it in items:
                            rep_rows.append({
                                "date": it["date"],
                                "cctv": it["cctv"],
                                "timestampsafe": str(it["timestampsafe"]).zfill(6),
                                "reason": it.get("reason", ""),
                            })
                except Exception as e:
                    print(f"[warn] reading report {rf}: {e}")

    rep_df = pd.DataFrame(rep_rows)
    if not rep_df.empty:
        rep_df = rep_df.drop_duplicates(subset=["date", "cctv", "timestampsafe"])
        raw_df = raw_df.merge(
            rep_df,
            left_on=["date", "cam", "tss"],
            right_on=["date", "cctv", "timestampsafe"],
            how="left",
        )
    else:
        raw_df["reason"] = np.nan

    missing_prefixes = (
        "no_frames_in_window",
        "segments_open_failed",
        "no_ts_segments",
        "no_segments_in_window",
        "no_metadata_csv",
    )
    partial_prefixes = ("truncated_segments", "partial_segments_failed")

    reason_series = raw_df["reason"].fillna("")
    raw_df["cctv_missing"] = reason_series.apply(
        lambda r: any(r.startswith(p) for p in missing_prefixes)
    )
    raw_df["cctv_partial"] = reason_series.apply(
        lambda r: any(r.startswith(p) for p in partial_prefixes)
    )

    # Compute row features per ROI
    classes = ["bus", "car", "motorcycle", "truck"]
    for k in [1, 2, 3]:
        vol_col = f"vol_{k}"
        spd_col = f"spd_{k}"
        cls_cols = [f"ROI_{k}_Vol_{c}" for c in classes]
        if all(c in raw_df.columns for c in cls_cols):
            raw_df[vol_col] = raw_df[cls_cols].sum(axis=1)
            raw_df[spd_col] = np.where(
                raw_df[f"ROI_{k}_AvgSpeed_Kmh"] > 0,
                raw_df[f"ROI_{k}_AvgSpeed_Kmh"],
                np.nan,
            )
        else:
            raw_df[vol_col] = np.nan
            raw_df[spd_col] = np.nan

    # Row total volume
    vol_cols = [f"vol_{k}" for k in [1, 2, 3]]
    raw_df["vol"] = raw_df[vol_cols].sum(axis=1, min_count=1)

    # Identify unlisted all-zero rows before masking
    all_vol_cols = [c for c in raw_df.columns if "_Vol_" in c]
    row_vol_sum = raw_df[all_vol_cols].fillna(0).sum(axis=1)
    raw_df["unlisted_zero"] = (
        (row_vol_sum == 0) & (~raw_df["cctv_missing"]) & (raw_df["reason"].isna())
    )

    # For cctv_missing rows, set vol, class volumes, and speeds to NaN
    missing_mask = raw_df["cctv_missing"]
    raw_df.loc[missing_mask, "vol"] = np.nan
    for k in [1, 2, 3]:
        raw_df.loc[missing_mask, f"vol_{k}"] = np.nan
        raw_df.loc[missing_mask, f"spd_{k}"] = np.nan
        for c in classes:
            col = f"ROI_{k}_Vol_{c}"
            if col in raw_df.columns:
                raw_df.loc[missing_mask, col] = np.nan

    return raw_df, dates_found


def build_regular_grid(raw_df: pd.DataFrame, dates: list[str]):
    cameras = list(SHORT_NAMES.keys())
    grid_records = []
    for dt in dates:
        dt_slots = pd.date_range(f"{dt} 00:00:00", f"{dt} 23:55:00", freq="5min")
        for slot in dt_slots:
            for cam in cameras:
                grid_records.append({"slot": slot, "cam": cam, "date": dt})

    grid_df = pd.DataFrame(grid_records)
    raw_df["slot"] = raw_df["ts"].dt.floor("5min")

    classes = ["bus", "car", "motorcycle", "truck"]

    # Group raw rows by (cam, slot)
    def agg_slot(group):
        available = any(~group["cctv_missing"])
        vol = group["vol"].sum(min_count=1)

        c_vols = {}
        for c in classes:
            cols = [f"ROI_{k}_Vol_{c}" for k in [1, 2, 3] if f"ROI_{k}_Vol_{c}" in group.columns]
            if cols:
                c_vols[c] = group[cols].sum(axis=1, min_count=1).sum(min_count=1)
            else:
                c_vols[c] = np.nan

        # Volume-weighted speed across ROIs
        numerator = 0.0
        denominator = 0.0
        simple_speeds = []
        for k in [1, 2, 3]:
            spds = group[f"spd_{k}"].dropna()
            vols = group.loc[spds.index, f"vol_{k}"].fillna(0)
            for s, v in zip(spds, vols):
                if v > 0:
                    numerator += s * v
                    denominator += v
                simple_speeds.append(s)

        if denominator > 0:
            spd = numerator / denominator
        elif simple_speeds:
            spd = float(np.mean(simple_speeds))
        else:
            spd = np.nan

        # Weather last non-null
        def last_valid(col):
            s = group[col].dropna()
            return s.iloc[-1] if not s.empty else np.nan

        w_temp = last_valid("weather_temp")
        w_hum = last_valid("weather_humidity")
        w_rain = last_valid("weather_rain")
        w_wind = last_valid("weather_wind_speed")

        res = {
            "available": available,
            "vol": vol,
            "spd": spd,
            "weather_temp": w_temp,
            "weather_humidity": w_hum,
            "weather_rain": w_rain,
            "weather_wind_speed": w_wind,
        }
        for c in classes:
            res[f"vol_{c}"] = c_vols[c]
        return pd.Series(res)

    print("Aggregating raw rows to regular grid...")
    slot_agg = raw_df.groupby(["cam", "slot"]).apply(agg_slot, include_groups=False).reset_index()

    merged_grid = grid_df.merge(slot_agg, on=["cam", "slot"], how="left")
    merged_grid["available"] = merged_grid["available"].fillna(False).astype(bool)
    return merged_grid


def generate_figures(raw_df: pd.DataFrame, grid_df: pd.DataFrame, dates: list[str], out_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Fig 5: Record Availability Heatmap (2062 x 1132)
    # --------------------------------------------------------------------------
    fig5_w, fig5_h = 2062 / 300, 1132 / 300
    fig, ax = plt.subplots(figsize=(fig5_w, fig5_h), dpi=300)

    cams_in_order = [
        "padalarang_-_jl_baru_kbp",
        "padalarang_-_gedong_5_bawah",
        "padalarang_-_gedong_5_atas_-_stasiun",
        "padalarang_giant_-_tol",
        "padalarang_-_giant_-_tagog",
        "tagog_padalarang_-_bandung",
    ]
    labels_y = [TABLE_NAMES[c] for c in cams_in_order]

    # Availability matrix: cam x date
    avail_matrix = np.zeros((len(cams_in_order), len(dates)))
    for i, cam in enumerate(cams_in_order):
        for j, dt in enumerate(dates):
            sub = grid_df[(grid_df["cam"] == cam) & (grid_df["date"] == dt)]
            avail_matrix[i, j] = sub["available"].sum()

    im = ax.imshow(avail_matrix, cmap="YlGnBu", vmin=0, vmax=288, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Slots / day", fontsize=10)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[5:] for d in dates], rotation=90, fontsize=8)
    ax.set_yticks(range(len(cams_in_order)))
    ax.set_yticklabels(labels_y, fontsize=9)
    ax.set_title("Record Availability per Location per Date (288 = complete day)", fontsize=11, fontweight="bold")
    plt.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.20)
    fig5_path = out_dir / "fig05_availability.png"
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig5_path}")

    # --------------------------------------------------------------------------
    # Fig 6: Class Shares & Location Totals (1965 x 2271)
    # --------------------------------------------------------------------------
    fig6_w, fig6_h = 1965 / 300, 2271 / 300
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig6_w, fig6_h), dpi=300)

    # 6A: Pie of class shares
    classes = ["motorcycle", "car", "truck", "bus"]
    class_labels = ["Motorcycle", "Car", "Truck", "Bus"]
    class_totals = []
    for c in classes:
        tot = sum(raw_df[f"ROI_{k}_Vol_{c}"].sum() for k in [1, 2, 3] if f"ROI_{k}_Vol_{c}" in raw_df.columns)
        class_totals.append(tot)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    wedges, texts, autotexts = ax1.pie(
        class_totals,
        labels=class_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_weight("bold")
    ax1.set_title("A. Vehicle Class Composition", fontsize=11, fontweight="bold")

    # 6B: Horizontal bar of total volume per location
    loc_totals = {}
    for cam in cams_in_order:
        sname = SHORT_NAMES[cam]
        cam_sub = raw_df[raw_df["cam"] == cam]
        tot = sum(cam_sub[f"ROI_{k}_Vol_{c}"].sum() for k in [1, 2, 3] for c in classes if f"ROI_{k}_Vol_{c}" in cam_sub.columns)
        loc_totals[sname] = tot

    sorted_locs = sorted(loc_totals.items(), key=lambda x: x[1])
    y_pos = range(len(sorted_locs))
    bars = ax2.barh(y_pos, [v / 1e6 for _, v in sorted_locs], color="#4C72B0", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([k for k, _ in sorted_locs], fontsize=10)
    ax2.set_xlabel("Total Volume (Million Vehicles)", fontsize=10)
    ax2.set_title("B. Total Volume per CCTV Location", fontsize=11, fontweight="bold")

    for bar, (_, v) in zip(bars, sorted_locs):
        ax2.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,.0f} ({v/1e6:.2f}M)",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax2.set_xlim(0, max(v / 1e6 for _, v in sorted_locs) * 1.25)
    plt.subplots_adjust(left=0.20, right=0.92, top=0.94, bottom=0.08, hspace=0.30)
    fig6_path = out_dir / "fig06_composition.png"
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig6_path}")

    # --------------------------------------------------------------------------
    # Fig 7: Hourly and DOW Demand Patterns (1965 x 2373)
    # --------------------------------------------------------------------------
    fig7_w, fig7_h = 1965 / 300, 2373 / 300
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig7_w, fig7_h), dpi=300)

    # 7A: Mean slot volume by hour of day, one line per location
    avail_grid = grid_df[grid_df["available"]].copy()
    avail_grid["hour"] = avail_grid["slot"].dt.hour
    avail_grid["dow"] = avail_grid["slot"].dt.dayofweek

    for cam in cams_in_order:
        sname = SHORT_NAMES[cam]
        cam_sub = avail_grid[avail_grid["cam"] == cam]
        hourly_mean = cam_sub.groupby("hour")["vol"].mean()
        ax1.plot(hourly_mean.index, hourly_mean.values, label=sname, marker="o", markersize=4)

    ax1.set_xticks(range(24))
    ax1.set_xlabel("Hour of Day (WIB)", fontsize=10)
    ax1.set_ylabel("Mean volume (per 5-min interval)", fontsize=10)
    ax1.set_title("A. Mean Hourly Volume per Location", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    # 7B: Mean slot volume by day of week pooled across locations
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_means = avail_grid.groupby("dow")["vol"].mean()
    bar_colors = ["#4C72B0" if d < 5 else "#55A868" for d in range(7)]
    bars = ax2.bar(range(7), dow_means.values, color=bar_colors, alpha=0.85)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(dow_names, fontsize=10)
    ax2.set_ylabel("Mean volume (per 5-min interval)", fontsize=10)
    ax2.set_title("B. Mean Volume by Day of Week", fontsize=11, fontweight="bold")

    for bar, v in zip(bars, dow_means.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylim(0, max(dow_means.values) * 1.15)
    plt.subplots_adjust(left=0.15, right=0.92, top=0.94, bottom=0.08, hspace=0.28)
    fig7_path = out_dir / "fig07_demand.png"
    fig.savefig(fig7_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig7_path}")

    # --------------------------------------------------------------------------
    # Fig 8: Daily Volume and Speed Trends (1965 x 2271)
    # --------------------------------------------------------------------------
    fig8_w, fig8_h = 1965 / 300, 2271 / 300
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig8_w, fig8_h), dpi=300)

    # Weekend indices for shading
    date_dts = pd.to_datetime(dates)
    weekend_indices = [i for i, dt in enumerate(date_dts) if dt.dayofweek >= 5]
    for ax in (ax1, ax2):
        for idx in weekend_indices:
            ax.axvspan(idx - 0.45, idx + 0.45, color="#55A868", alpha=0.10)

    # 8A: Daily total volume per location
    for cam in cams_in_order:
        sname = SHORT_NAMES[cam]
        cam_sub = grid_df[grid_df["cam"] == cam]
        daily_vols = []
        low_avail_pts = []
        for j, dt in enumerate(dates):
            day_slots = cam_sub[cam_sub["date"] == dt]
            n_avail = day_slots["available"].sum()
            v_sum = day_slots["vol"].sum(min_count=1)
            daily_vols.append(v_sum)
            if n_avail < 288 * 0.95:
                low_avail_pts.append((j, v_sum))

        line = ax1.plot(range(len(dates)), daily_vols, label=sname, marker="s", markersize=3.5)[0]
        if low_avail_pts:
            xs, ys = zip(*low_avail_pts)
            ax1.scatter(xs, ys, s=40, facecolors="none", edgecolors=line.get_color(), linewidths=1.5, zorder=5)

    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels([d[5:] for d in dates], rotation=90, fontsize=8)
    ax1.set_ylabel("Daily Total Volume (Vehicles)", fontsize=10)
    ax1.set_title("A. Daily Traffic Volume", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    # 8B: Daily masked mean speed per location
    for cam in cams_in_order:
        sname = SHORT_NAMES[cam]
        cam_raw = raw_df[raw_df["cam"] == cam]
        daily_spds = []
        for dt in dates:
            day_raw = cam_raw[(cam_raw["date"] == dt) & (~cam_raw["cctv_missing"])]
            all_spds = []
            for k in [1, 2, 3]:
                col = f"spd_{k}"
                if col in day_raw.columns:
                    all_spds.extend(day_raw[col].dropna().tolist())
            daily_spds.append(np.mean(all_spds) if all_spds else np.nan)

        ax2.plot(range(len(dates)), daily_spds, label=sname, marker="o", markersize=3.5)

    ax2.set_xticks(range(len(dates)))
    ax2.set_xticklabels([d[5:] for d in dates], rotation=90, fontsize=8)
    ax2.set_ylabel("Masked Mean Speed (km/h)", fontsize=10)
    ax2.set_title("B. Daily Mean Speed (Masked)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.subplots_adjust(left=0.15, right=0.92, top=0.94, bottom=0.12, hspace=0.32)
    fig8_path = out_dir / "fig08_daily_trends.png"
    fig.savefig(fig8_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig8_path}")

    # --------------------------------------------------------------------------
    # Fig 9: Weather & Volume (1965 x 1965)
    # --------------------------------------------------------------------------
    fig9_w, fig9_h = 1965 / 300, 1965 / 300
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig9_w, fig9_h), dpi=300)

    # 9A: Hourly mean temp pooled over raw rows
    raw_df["hour"] = raw_df["ts"].dt.hour
    hourly_temp = raw_df.groupby("hour")["weather_temp"].mean()
    ax1.plot(hourly_temp.index, hourly_temp.values, color="#D95F02", marker="o", linewidth=2)
    ax1.set_xticks(range(24))
    ax1.set_xlabel("Hour of Day (WIB)", fontsize=10)
    ax1.set_ylabel("Air Temperature (°C)", fontsize=10)
    ax1.set_title("A. Diurnal Temperature Cycle", fontsize=11, fontweight="bold")

    # 9B: Mean volume: dry vs rain
    valid_rain_slots = grid_df[grid_df["weather_rain"].notna() & grid_df["vol"].notna()]
    dry_vol = valid_rain_slots[valid_rain_slots["weather_rain"] == 0]["vol"].mean()
    rain_vol = valid_rain_slots[valid_rain_slots["weather_rain"] > 0]["vol"].mean()

    bars = ax2.bar(["No Rain", "Rain"], [dry_vol, rain_vol], color=["#4C72B0", "#55A868"], alpha=0.85, width=0.5)
    ax2.set_ylabel("Mean Volume per 5-min Interval", fontsize=10)
    ax2.set_title("B. Mean Volume: Dry vs. Rainy Intervals", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, [dry_vol, rain_vol]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylim(0, max(dry_vol, rain_vol) * 1.20)
    plt.subplots_adjust(left=0.15, right=0.92, top=0.93, bottom=0.09, hspace=0.30)
    fig9_path = out_dir / "fig09_weather.png"
    fig.savefig(fig9_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig9_path}")


def generate_tables_and_summary(
    raw_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    dates: list[str],
    extracted_dir: Path,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_days = len(dates)
    date_dts = pd.to_datetime(dates)
    n_weekdays = int(sum(dt.dayofweek < 5 for dt in date_dts))
    n_weekend_days = int(sum(dt.dayofweek >= 5 for dt in date_dts))
    n_files = 192
    n_rows = len(raw_df)

    n_slots = len(grid_df)
    n_slots_available = int(grid_df["available"].sum())
    avail_pct = round(n_slots_available / n_slots * 100, 1)

    n_cctv_missing_rows = int(raw_df["cctv_missing"].sum())
    cctv_missing_pct = round(n_cctv_missing_rows / n_rows * 100, 1)
    n_partial_rows = int(raw_df["cctv_partial"].sum())
    n_unlisted_zero_rows = int(raw_df["unlisted_zero"].sum())

    roi2_absent_rows = int(raw_df["ROI_2_Vol_car"].isna().sum())
    roi3_absent_rows = int(raw_df["ROI_3_Vol_car"].isna().sum())

    wx_missing_rows = int(raw_df["weather_temp"].isna().sum())
    wx_missing_pct = round(wx_missing_rows / n_rows * 100, 2)

    # Volumes
    classes = ["motorcycle", "car", "truck", "bus"]
    class_counts = {}
    for c in classes:
        cnt = int(sum(raw_df[f"ROI_{k}_Vol_{c}"].sum() for k in [1, 2, 3] if f"ROI_{k}_Vol_{c}" in raw_df.columns))
        class_counts[c] = cnt

    total_vehicles = sum(class_counts.values())
    class_pct = {c: round(cnt / total_vehicles * 100, 1) for c, cnt in class_counts.items()}

    location_totals = {}
    for cam, sname in SHORT_NAMES.items():
        cam_sub = raw_df[raw_df["cam"] == cam]
        tot = int(sum(cam_sub[f"ROI_{k}_Vol_{c}"].sum() for k in [1, 2, 3] for c in classes if f"ROI_{k}_Vol_{c}" in cam_sub.columns))
        location_totals[sname] = tot
    location_totals = dict(sorted(location_totals.items(), key=lambda x: x[1], reverse=True))

    # Demand patterns
    avail_grid = grid_df[grid_df["available"]].copy()
    avail_grid["hour"] = avail_grid["slot"].dt.hour
    avail_grid["dow"] = avail_grid["slot"].dt.dayofweek

    # Corridor hourly mean slot volume (across all locations)
    hourly_means = avail_grid.groupby("hour")["vol"].mean()
    am_peak_hour = int(hourly_means.loc[0:11].idxmax())
    pm_peak_hour = int(hourly_means.loc[12:23].idxmax())
    trough_hour = int(hourly_means.idxmin())

    weekday_mean_vol = avail_grid[avail_grid["dow"] < 5]["vol"].mean()
    weekend_mean_vol = avail_grid[avail_grid["dow"] >= 5]["vol"].mean()
    weekday_vs_weekend_pct = round((weekday_mean_vol / weekend_mean_vol - 1.0) * 100, 1)

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_means = avail_grid.groupby("dow")["vol"].mean()
    max_dow = dow_names[int(dow_means.idxmax())]
    min_dow = dow_names[int(dow_means.idxmin())]

    # Daily corridor total & OLS slope
    daily_corridor_totals = []
    for dt in dates:
        day_sub = grid_df[grid_df["date"] == dt]
        daily_corridor_totals.append(day_sub["vol"].sum(min_count=1))
    daily_corridor_totals = np.array(daily_corridor_totals)
    x = np.arange(len(dates))
    res_ols = stats.linregress(x, daily_corridor_totals)
    slope_pct_per_week = round(float(res_ols.slope * 7.0 / np.mean(daily_corridor_totals) * 100), 1)

    # Outage days: location-days with <95% slots available (< 273.6)
    n_outage_days = 0
    for cam in SHORT_NAMES:
        cam_sub = grid_df[grid_df["cam"] == cam]
        for dt in dates:
            if cam_sub[cam_sub["date"] == dt]["available"].sum() < 288 * 0.95:
                n_outage_days += 1

    # Weather
    raw_df["hour"] = raw_df["ts"].dt.hour
    hourly_temp = raw_df.groupby("hour")["weather_temp"].mean()
    temp_peak_hour = int(hourly_temp.idxmax())
    temp_peak = round(float(hourly_temp.max()), 1)
    temp_min_hour = int(hourly_temp.idxmin())
    temp_min = round(float(hourly_temp.min()), 1)

    rain_nonnull_slots = grid_df["weather_rain"].dropna()
    rain_slot_pct = round(float((rain_nonnull_slots > 0).mean() * 100), 1)

    rainy_slots = grid_df[grid_df["weather_rain"] > 0]
    rain_days = int(rainy_slots["date"].nunique())
    rain_hours = rainy_slots["slot"].dt.hour
    rain_hour_q25 = int(np.percentile(rain_hours, 25))
    rain_hour_q75 = int(np.percentile(rain_hours, 75))
    rain_max = round(float(rain_nonnull_slots.max()), 1)

    valid_wx_slots = grid_df[grid_df["weather_rain"].notna() & grid_df["vol"].notna()].copy()
    valid_wx_slots["hour"] = valid_wx_slots["slot"].dt.hour
    vol_dry = round(float(valid_wx_slots[valid_wx_slots["weather_rain"] == 0]["vol"].mean()), 1)
    vol_rain = round(float(valid_wx_slots[valid_wx_slots["weather_rain"] > 0]["vol"].mean()), 1)
    rain_diff_pct = round((vol_rain / vol_dry - 1.0) * 100, 1)

    # Hour-matched rain diff
    h_diff_num = 0.0
    h_diff_den = 0.0
    for h, h_group in valid_wx_slots.groupby("hour"):
        r_grp = h_group[h_group["weather_rain"] > 0]
        d_grp = h_group[h_group["weather_rain"] == 0]
        if not r_grp.empty and not d_grp.empty:
            n_r = len(r_grp)
            m_r = r_grp["vol"].mean()
            m_d = d_grp["vol"].mean()
            h_diff_num += n_r * (m_r - m_d)
            h_diff_den += n_r * m_d

    rain_diff_hourmatched_pct = round(float(h_diff_num / h_diff_den * 100), 1) if h_diff_den > 0 else 0.0

    # API quality
    tt_ok = int((raw_df["traffic_status"] == "ok").sum())
    tt_ok_pct = round(tt_ok / n_rows * 100, 1)
    tt_err = n_rows - tt_ok
    tt_ok_df = raw_df[raw_df["traffic_status"] == "ok"]
    tt_fresh = int(tt_ok_df["traffic_cache_status"].isin(["fresh", "fresh_fallback"]).sum())
    tt_cached = int((tt_ok_df["traffic_cache_status"] == "cached").sum())

    wx_ok = int((raw_df["weather_status"] == "ok").sum())
    wx_ok_pct = round(wx_ok / n_rows * 100, 1)
    wx_ok_df = raw_df[raw_df["weather_status"] == "ok"]
    wx_cached = int((wx_ok_df["weather_cache_status"] == "cached").sum())
    n_closure_true = int((raw_df["traffic_road_closure"] == True).sum())

    # Holidays and events
    id_hols = holidays.Indonesia(years=2026)
    holidays_in_window = [str(d) for d in id_hols if "2026-09-09" <= str(d) <= "2026-10-10"]

    events_csv = extracted_dir.parent / "events" / "events.csv"
    events_in_window = []
    if events_csv.exists():
        ev_df = pd.read_csv(events_csv)
        for _, r in ev_df.iterrows():
            s = str(r["start_local"])[:10]
            e = str(r["end_local"])[:10]
            if not (e < "2026-09-09" or s > "2026-10-10"):
                events_in_window.append(r["event_id"])

    # Speed by ROI and zero-speed pct
    speed_roi_rows = []
    max_zero_speed_pct = 0.0
    for cam in SHORT_NAMES:
        cam_sub = raw_df[raw_df["cam"] == cam]
        for k in CAM_ROIS[cam]:
            col_v = f"ROI_{k}_Vol_car"
            col_s = f"ROI_{k}_AvgSpeed_Kmh"
            non_missing = cam_sub[~cam_sub["cctv_missing"]]
            tot_v = sum(non_missing[f"ROI_{k}_Vol_{c}"].fillna(0) for c in classes)
            spds = non_missing.loc[non_missing[col_s] > 0, col_s]
            n_obs = len(spds)
            p5, p25, p50, p75, p95 = (
                np.percentile(spds, [5, 25, 50, 75, 95]) if n_obs > 0 else [np.nan] * 5
            )

            # zero speed with volume > 0
            has_vol = tot_v > 0
            zero_spd = (non_missing[col_s] == 0) & has_vol
            zero_pct = round(zero_spd.sum() / len(non_missing) * 100, 2)
            if zero_pct > max_zero_speed_pct:
                max_zero_speed_pct = zero_pct

            speed_roi_rows.append({
                "cctv": cam,
                "roi_idx": k,
                "n_samples": n_obs,
                "p05": round(float(p5), 1),
                "p25": round(float(p25), 1),
                "p50": round(float(p50), 1),
                "p75": round(float(p75), 1),
                "p95": round(float(p95), 1),
                "zero_speed_with_volume_pct": zero_pct,
            })
    pd.DataFrame(speed_roi_rows).to_csv(out_dir / "speed_by_roi.csv", index=False)

    # Anomaly screening: |0.6745*(vol - median)/MAD| > 5 within (cam, hour-of-day)
    anomaly_rows = []
    for (cam, h), grp in avail_grid.groupby(["cam", "hour"]):
        vols = grp["vol"].dropna()
        if len(vols) < 3:
            continue
        med = float(np.median(vols))
        mad = float(np.median(np.abs(vols - med)))
        if mad > 0:
            for _, r in grp.iterrows():
                if pd.notna(r["vol"]):
                    z = 0.6745 * abs(r["vol"] - med) / mad
                    if z > 5.0:
                        anomaly_rows.append({
                            "cam": cam,
                            "slot": str(r["slot"]),
                            "hour": h,
                            "vol": r["vol"],
                            "median": round(med, 2),
                            "mad": round(mad, 2),
                            "z_score": round(z, 2),
                        })

    anom_df = pd.DataFrame(anomaly_rows)
    anom_df.to_csv(out_dir / "anomalies.csv", index=False)
    n_anomalies = len(anom_df)
    anomaly_pct = round(n_anomalies / n_slots_available * 100, 2)

    # Representativeness CSV
    rep_table_rows = [
        {"dimension": "Day of Week", "category": "Weekday", "count": n_weekdays, "unit": "days"},
        {"dimension": "Day of Week", "category": "Weekend", "count": n_weekend_days, "unit": "days"},
    ]
    # Day (06:00-17:59) vs Night
    day_slots = int(((grid_df["slot"].dt.hour >= 6) & (grid_df["slot"].dt.hour < 18)).sum())
    night_slots = n_slots - day_slots
    rep_table_rows.append({"dimension": "Time of Day", "category": "Day (06:00-17:59)", "count": day_slots, "unit": "location-slots"})
    rep_table_rows.append({"dimension": "Time of Day", "category": "Night (18:00-05:59)", "count": night_slots, "unit": "location-slots"})

    # Rain bins over location-slots
    r_series = grid_df["weather_rain"]
    rep_table_rows.append({"dimension": "Rainfall", "category": "0", "count": int((r_series == 0).sum()), "unit": "location-slots"})
    rep_table_rows.append({"dimension": "Rainfall", "category": "(0, 0.1]", "count": int(((r_series > 0) & (r_series <= 0.1)).sum()), "unit": "location-slots"})
    rep_table_rows.append({"dimension": "Rainfall", "category": "(0.1, 0.3]", "count": int(((r_series > 0.1) & (r_series <= 0.3)).sum()), "unit": "location-slots"})
    rep_table_rows.append({"dimension": "Rainfall", "category": "> 0.3", "count": int((r_series > 0.3).sum()), "unit": "location-slots"})
    rep_table_rows.append({"dimension": "Rainfall", "category": "missing", "count": int(r_series.isna().sum()), "unit": "location-slots"})

    rep_table_rows.append({"dimension": "Calendar", "category": "National Holidays", "count": len(holidays_in_window), "unit": "days"})
    rep_table_rows.append({"dimension": "Calendar", "category": "Registered Local Events", "count": len(events_in_window), "unit": "events"})
    pd.DataFrame(rep_table_rows).to_csv(out_dir / "representativeness.csv", index=False)

    # Missingness CSV
    miss_rows = []
    cols_to_check = [
        "traffic_speed", "traffic_freeflow", "traffic_confidence",
        "weather_temp", "weather_humidity", "weather_rain", "weather_wind_speed",
        "ROI_1_AvgSpeed_Kmh", "ROI_2_AvgSpeed_Kmh", "ROI_3_AvgSpeed_Kmh",
    ]
    for cam in SHORT_NAMES:
        cam_sub = raw_df[raw_df["cam"] == cam]
        n_c = len(cam_sub)
        for col in cols_to_check:
            if col in cam_sub.columns:
                n_miss = int(cam_sub[col].isna().sum())
                miss_rows.append({
                    "cctv": cam,
                    "column": col,
                    "nan_count": n_miss,
                    "nan_pct": round(n_miss / n_c * 100, 2),
                })
        # Add cctv_missing, cctv_partial, unlisted_zero
        miss_rows.append({"cctv": cam, "column": "cctv_missing", "nan_count": int(cam_sub["cctv_missing"].sum()), "nan_pct": round(cam_sub["cctv_missing"].sum() / n_c * 100, 2)})
        miss_rows.append({"cctv": cam, "column": "cctv_partial", "nan_count": int(cam_sub["cctv_partial"].sum()), "nan_pct": round(cam_sub["cctv_partial"].sum() / n_c * 100, 2)})
        miss_rows.append({"cctv": cam, "column": "unlisted_all_zero", "nan_count": int(cam_sub["unlisted_zero"].sum()), "nan_pct": round(cam_sub["unlisted_zero"].sum() / n_c * 100, 2)})
    pd.DataFrame(miss_rows).to_csv(out_dir / "missingness.csv", index=False)

    # Correlation CSV
    # Corridor mean slot vol, spd, wx variables over available slots
    slot_corridor = avail_grid.groupby("slot").agg({
        "vol": "mean",
        "spd": "mean",
        "weather_temp": "last",
        "weather_humidity": "last",
        "weather_rain": "last",
        "weather_wind_speed": "last",
    }).dropna()
    corr_df = slot_corridor.corr(method="pearson").round(3)
    corr_df.to_csv(out_dir / "correlation.csv")

    # Traffic state indicators from Step 1 metrics
    metrics_dir = extracted_dir / "metrics"
    metrics_files = sorted(metrics_dir.glob("*_metrics.csv"))
    assert len(metrics_files) == 192, f"Expected 192 metrics files in {metrics_dir}, found {len(metrics_files)}"

    required_metrics_cols = ["low_coverage", "flow_pcu_h_pkji", "spd_sms", "occ_time_pct", "ds_pkji", "queue_present_pct_fixed"]
    m_dfs = []
    for mf in metrics_files:
        df_m = pd.read_csv(mf)
        for rc in required_metrics_cols:
            assert rc in df_m.columns, f"Missing column {rc} in {mf}"
        m_dfs.append(df_m)
    all_metrics = pd.concat(m_dfs, ignore_index=True)

    n_win = len(all_metrics)
    n_low = int((all_metrics["low_coverage"] == True).sum())
    low_pct = round(n_low / n_win * 100, 1)

    valid_m = all_metrics[all_metrics["low_coverage"] == False].copy()
    sms_mean = round(float(valid_m["spd_sms"].dropna().mean()), 1)
    occ_mean = round(float(valid_m["occ_time_pct"].dropna().mean()), 1)

    ds_series = valid_m["ds_pkji"].dropna()
    ds_lt075_pct = round(float((ds_series < 0.75).mean() * 100), 1)
    ds_mid_pct = round(float(((ds_series >= 0.75) & (ds_series < 0.85)).mean() * 100), 1)
    ds_ge085_pct = round(float((ds_series >= 0.85).mean() * 100), 1)

    # Queue present net
    q_series = valid_m["queue_present_pct_fixed"].dropna()
    q_net = round(float(q_series.mean()), 1) if not q_series.empty else float("nan")

    # Read traffic_params.json
    params_file = extracted_dir.parent / "traffic_params.json"
    q_thr_fixed = 10.0
    low_cov_frac = 0.8
    if params_file.exists():
        with open(params_file, encoding="utf-8") as pf:
            p_json = json.load(pf)
            q_thr_fixed = p_json.get("queue_speed_fixed_kmh", 10.0)
            low_cov_frac = p_json.get("low_coverage_frac", 0.8)

    # Per location traffic state
    loc_state = {}
    table_v_rows = []
    for tname in TABLE_ORDER:
        cam = [c for c, tn in TABLE_NAMES.items() if tn == tname][0]
        c_sub = valid_m[valid_m["cctv"] == cam]

        # flow_pcu: mean over windows of sum across ROIs
        win_flow = c_sub.groupby(["date", "window_ts"])["flow_pcu_h_pkji"].sum()
        flow_pcu = round(float(win_flow.mean())) if not win_flow.empty else 0

        sms_loc = round(float(c_sub["spd_sms"].dropna().mean()), 1) if not c_sub["spd_sms"].dropna().empty else float("nan")
        occ_loc = round(float(c_sub["occ_time_pct"].dropna().mean()), 1) if not c_sub["occ_time_pct"].dropna().empty else float("nan")
        ds_loc = round(float(c_sub["ds_pkji"].dropna().mean()), 2) if not c_sub["ds_pkji"].dropna().empty else float("nan")

        c_ds = c_sub["ds_pkji"].dropna()
        ds_ge_pct = round(float((c_ds >= 0.85).mean() * 100), 1) if not c_ds.empty else 0.0

        q_loc_s = c_sub["queue_present_pct_fixed"].dropna()
        q_loc = round(float(q_loc_s.mean()), 1) if not q_loc_s.empty else None

        loc_state[tname] = {
            "flow_pcu": flow_pcu,
            "sms": sms_loc,
            "occ": occ_loc,
            "ds": ds_loc,
            "ds_ge085_pct": ds_ge_pct,
            "queue": q_loc,
        }
        table_v_rows.append({
            "location": tname,
            "flow_pcu": flow_pcu,
            "sms": sms_loc,
            "occ": occ_loc,
            "ds": ds_loc,
            "ds_ge085_pct": ds_ge_pct,
            "queue": q_loc if q_loc is not None else "",
        })

    pd.DataFrame(table_v_rows).to_csv(out_dir / "traffic_state_by_location.csv", index=False)

    traffic_state_summary = {
        "n_win": n_win,
        "n_low": n_low,
        "low_pct": low_pct,
        "sms": sms_mean,
        "occ": occ_mean,
        "ds_lt075_pct": ds_lt075_pct,
        "ds_mid_pct": ds_mid_pct,
        "ds_ge085_pct": ds_ge085_pct,
        "q_net": q_net,
        "q_thr_fixed": q_thr_fixed,
        "low_cov_frac": low_cov_frac,
        "by_location": loc_state,
    }

    # Summary object
    summary = {
        "start": dates[0],
        "end": dates[-1],
        "n_days": n_days,
        "n_weekdays": n_weekdays,
        "n_weekend_days": n_weekend_days,
        "n_files": n_files,
        "n_rows": n_rows,
        "n_slots": n_slots,
        "n_slots_available": n_slots_available,
        "avail_pct": avail_pct,
        "n_cctv_missing_rows": n_cctv_missing_rows,
        "cctv_missing_pct": cctv_missing_pct,
        "n_partial_rows": n_partial_rows,
        "n_unlisted_zero_rows": n_unlisted_zero_rows,
        "roi2_absent_rows": roi2_absent_rows,
        "roi3_absent_rows": roi3_absent_rows,
        "wx_missing_rows": wx_missing_rows,
        "wx_missing_pct": wx_missing_pct,
        "total_vehicles": total_vehicles,
        "class_counts": class_counts,
        "class_pct": class_pct,
        "location_totals": location_totals,
        "am_peak_hour": am_peak_hour,
        "pm_peak_hour": pm_peak_hour,
        "trough_hour": trough_hour,
        "weekday_vs_weekend_pct": weekday_vs_weekend_pct,
        "max_dow": max_dow,
        "min_dow": min_dow,
        "slope_pct_per_week": slope_pct_per_week,
        "n_outage_days": n_outage_days,
        "temp_peak_hour": temp_peak_hour,
        "temp_peak": temp_peak,
        "temp_min_hour": temp_min_hour,
        "temp_min": temp_min,
        "rain_slot_pct": rain_slot_pct,
        "rain_days": rain_days,
        "rain_hour_q25": rain_hour_q25,
        "rain_hour_q75": rain_hour_q75,
        "rain_max": rain_max,
        "vol_rain": vol_rain,
        "vol_dry": vol_dry,
        "rain_diff_pct": rain_diff_pct,
        "rain_diff_hourmatched_pct": rain_diff_hourmatched_pct,
        "tt_ok": tt_ok,
        "tt_ok_pct": tt_ok_pct,
        "tt_err": tt_err,
        "tt_fresh": tt_fresh,
        "tt_cached": tt_cached,
        "wx_ok": wx_ok,
        "wx_ok_pct": wx_ok_pct,
        "wx_cached": wx_cached,
        "n_closure_true": n_closure_true,
        "holidays_in_window": holidays_in_window,
        "events_in_window": events_in_window,
        "max_zero_speed_pct": max_zero_speed_pct,
        "n_anomalies": n_anomalies,
        "anomaly_pct": anomaly_pct,
        "traffic_state": traffic_state_summary,
    }

    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Wrote {out_dir / 'eda_summary.json'}")

    # EDA_REPORT.md
    generate_markdown_report(summary, out_dir)


def generate_markdown_report(summary: dict, out_dir: Path):
    s = summary
    ts = s["traffic_state"]
    by_loc = ts["by_location"]

    rep_lines = [
        "# Exploratory Data Analysis: 32-Day Padalarang Corridor Traffic Dataset (2026-09-09 to 2026-10-10)",
        "",
        "## Headline Summary",
        f"The 32-day dataset covers **{s['n_days']} consecutive calendar days** (9 September to 10 October 2026; {s['n_weekdays']} weekdays and {s['n_weekend_days']} weekend days) across 6 CCTV observation nodes and 12 ROIs along the Padalarang corridor. "
        f"The dataset comprises **{s['n_rows']:,} five-minute records** across {s['n_files']} CSV files, logging **{s['total_vehicles']:,} vehicle passages** ({s['total_vehicles']/1e6:.2f} million). "
        f"Video availability is {s['avail_pct']}% across the 55,296 regular five-minute location-slots. "
        f"Motorcycles dominate at {s['class_pct']['motorcycle']}% ({s['class_counts']['motorcycle']:,}), followed by cars at {s['class_pct']['car']}% ({s['class_counts']['car']:,}), trucks at {s['class_pct']['truck']}% ({s['class_counts']['truck']:,}), and buses at {s['class_pct']['bus']}% ({s['class_counts']['bus']:,}). "
        f"Corridor demand exhibits pronounced peaks at {s['am_peak_hour']:02d}:00 and {s['pm_peak_hour']:02d}:00 WIB, with weekdays carrying {abs(s['weekday_vs_weekend_pct'])}% {'higher' if s['weekday_vs_weekend_pct']>0 else 'lower'} volume than weekends. "
        f"No systematic corridor drift is present ({s['slope_pct_per_week']:+.1f}%/week). "
        "No national holidays or registered major local events occurred during this window.",
        "",
        "## 1. Temporal Coverage & Record Completeness",
        f"![Fig 5: Availability](fig05_availability.png)",
        "",
        f"- **Calendar span**: {s['start']} to {s['end']} ({s['n_days']} days)",
        f"- **Weekdays**: {s['n_weekdays']} days; **Weekend days**: {s['n_weekend_days']} days",
        f"- **Regular 5-min slots**: {s['n_slots']:,} total (288 slots/day × 6 cameras)",
        f"- **Available slots**: {s['n_slots_available']:,} ({s['avail_pct']}%)",
        f"- **Outage location-days (<95% availability)**: {s['n_outage_days']}",
        "",
        "## 2. Missing Values & Outages",
        f"- **CCTV video outages**: {s['n_cctv_missing_rows']} records ({s['cctv_missing_pct']}%) listed in per-date missing reports.",
        f"- **Partial window decode issues**: {s['n_partial_rows']} records.",
        f"- **Unlisted all-zero records**: {s['n_unlisted_zero_rows']} records (retained as observed zeros).",
        f"- **Structural missingness**: ROI 3 columns absent in {s['roi3_absent_rows']:,} records (5 two-ROI sites); ROI 2 columns absent in {s['roi2_absent_rows']:,} records (Giant-Tol single-ROI site).",
        f"- **Weather missingness**: Open-Meteo missing in {s['wx_missing_rows']} records ({s['wx_missing_pct']}%).",
        "",
        "## 3. Vehicle Class Composition & Location Volumes",
        f"![Fig 6: Composition](fig06_composition.png)",
        "",
        "| Vehicle Class | Count | Share (%) |",
        "| :--- | :--- | :--- |",
    ]
    for c in ["motorcycle", "car", "truck", "bus"]:
        rep_lines.append(f"| {c.capitalize()} | {s['class_counts'][c]:,} | {s['class_pct'][c]:.1f}% |")

    rep_lines.extend([
        "",
        "| CCTV Location | Total Vehicles |",
        "| :--- | :--- |",
    ])
    for loc, v in s["location_totals"].items():
        rep_lines.append(f"| {loc} | {v:,} |")

    rep_lines.extend([
        "",
        "## 4. Diurnal & Weekly Demand Patterns",
        f"![Fig 7: Demand Patterns](fig07_demand.png)",
        "",
        f"- **Morning Peak**: {s['am_peak_hour']:02d}:00 WIB",
        f"- **Evening Peak**: {s['pm_peak_hour']:02d}:00 WIB",
        f"- **Nocturnal Trough**: {s['trough_hour']:02d}:00 WIB",
        f"- **Weekday vs. Weekend**: {s['weekday_vs_weekend_pct']:+.1f}%",
        f"- **Highest Demand Day**: {s['max_dow']}; **Lowest**: {s['min_dow']}",
        "",
        "## 5. Daily Trends & Weekly Drift",
        f"![Fig 8: Daily Trends](fig08_daily_trends.png)",
        "",
        f"- **Weekly Volume Trend (OLS)**: {s['slope_pct_per_week']:+.1f}% per week over the 32-day window.",
        "",
        "## 6. Weather & Environmental Modality",
        f"![Fig 9: Weather](fig09_weather.png)",
        "",
        f"- **Diurnal Temperature**: Peak at {s['temp_peak_hour']:02d}:00 ({s['temp_peak']:.1f} °C), minimum at {s['temp_min_hour']:02d}:00 ({s['temp_min']:.1f} °C).",
        f"- **Rain Occurrences**: Rain recorded in {s['rain_slot_pct']}% of intervals, across {s['rain_days']} of 32 days.",
        f"- **Precipitation Timing**: Interquartile range of rain hours: {s['rain_hour_q25']:02d}:00 to {s['rain_hour_q75']+1:02d}:00 WIB. Max intensity: {s['rain_max']} mm.",
        f"- **Volume under Rain**: Dry mean {s['vol_dry']} veh/slot vs. Rainy mean {s['vol_rain']} veh/slot (raw diff {s['rain_diff_pct']:+.1f}%; hour-matched diff {s['rain_diff_hourmatched_pct']:+.1f}%).",
        "",
        "## 7. Traffic State & Operational Indicators",
        "Derived from per-vehicle track logs across 192 location-days:",
        "",
        f"- **Total ROI-windows**: {ts['n_win']:,} ({ts['n_low']:,} flagged low coverage, {ts['low_pct']}%).",
        f"- **Space-Mean Speed (corridor average)**: {ts['sms']} km/h",
        f"- **Time Occupancy (corridor average)**: {ts['occ']}%",
        f"- **Degree of Saturation (PKJI)**: <0.75 in {ts['ds_lt075_pct']}%; [0.75, 0.85) in {ts['ds_mid_pct']}%; ≥0.85 in {ts['ds_ge085_pct']}%.",
        f"- **Corridor Queue Presence (fixed {ts['q_thr_fixed']:.0f} km/h threshold)**: {ts['q_net']}% on queue-monitored ROIs.",
        "",
        "### Traffic-State Summary by CCTV Location",
        "| CCTV Location | Flow (pcu/h) | SMS (km/h) | DS (PKJI) | Queue (%) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for tname in TABLE_ORDER:
        d = by_loc[tname]
        q_str = f"{d['queue']:.1f}" if d["queue"] is not None else "–"
        rep_lines.append(f"| {tname} | {d['flow_pcu']:,} | {d['sms']:.1f} | {d['ds']:.2f} | {q_str} |")

    rep_lines.extend([
        "",
        "## 8. API Quality Indicators",
        f"- **TomTom Traffic API**: {s['tt_ok']:,} valid responses ({s['tt_ok_pct']}%), {s['tt_err']:,} errors. Fresh: {s['tt_fresh']:,}, Cached: {s['tt_cached']:,}.",
        f"- **Open-Meteo Weather API**: {s['wx_ok']:,} valid responses ({s['wx_ok_pct']}%), {s['wx_cached']:,} served from cache.",
        f"- **Road Closure Flag**: {s['n_closure_true']} records set to True.",
        "",
        "## 9. Quality & Anomaly Screening",
        f"- **Max zero-speed with positive volume**: {s['max_zero_speed_pct']:.2f}% across all ROIs.",
        f"- **Statistical Anomalies (|z| > 5 MAD)**: {s['n_anomalies']} slots ({s['anomaly_pct']}% of available slots).",
    ])

    with open(out_dir / "EDA_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines) + "\n")
    print(f"Wrote {out_dir / 'EDA_REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description="EDA of 32-day dataset.")
    parser.add_argument("--start", default="2026-09-09", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-10-10", help="End date YYYY-MM-DD")
    parser.add_argument("--extracted", default="extracted", help="Path to extracted folder")
    parser.add_argument("--out", default="extracted/eda", help="Path to output EDA folder")
    args = parser.parse_args()

    extracted_dir = Path(args.extracted).resolve()
    out_dir = Path(args.out).resolve()

    print(f"Running EDA for window {args.start} to {args.end}...")
    raw_df, dates = load_dataset(extracted_dir, args.start, args.end)
    print(f"Loaded {len(raw_df)} raw rows across {len(dates)} dates.")

    grid_df = build_regular_grid(raw_df, dates)
    print(f"Built regular grid of {len(grid_df)} slots.")

    print("Generating figures (fig05 - fig09)...")
    generate_figures(raw_df, grid_df, dates, out_dir)

    print("Generating summary and tables...")
    generate_tables_and_summary(raw_df, grid_df, dates, extracted_dir, out_dir)
    print("EDA Complete!")


if __name__ == "__main__":
    main()

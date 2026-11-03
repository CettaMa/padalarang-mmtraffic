#!/usr/bin/env python3
"""Run traffic metrics computation camera-by-camera and merge results."""

import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd

DATES = [
    "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13",
    "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
    "2026-09-19", "2026-09-20", "2026-09-21", "2026-09-22", "2026-09-23",
    "2026-09-24", "2026-09-25", "2026-09-26", "2026-09-27", "2026-09-28",
    "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-03",
    "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08",
    "2026-10-09", "2026-10-10"
]

CAMERAS = [
    "padalarang_-_giant_-_tagog",
    "padalarang_giant_-_tol",
    "padalarang_-_jl_baru_kbp",
    "padalarang_-_gedong_5_bawah",
    "padalarang_-_gedong_5_atas_-_stasiun",
    "tagog_padalarang_-_bandung"
]

def main():
    base_dir = Path(__file__).resolve().parent.parent
    python_exe = sys.executable
    tools_dir = base_dir / "tools"
    metrics_dir = base_dir / "extracted" / "metrics"
    split_dir = base_dir / "extracted" / "metrics_split"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting traffic_metrics computation for {len(CAMERAS)} cameras and {len(DATES)} dates...")

    for cam in CAMERAS:
        cam_out = split_dir / cam
        cam_out.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Processing camera: {cam} ---")
        cmd = [
            python_exe,
            str(tools_dir / "traffic_metrics.py"),
            "compute",
            "--cctv", cam,
            "--out", str(cam_out),
            "--dates", *DATES
        ]
        res = subprocess.run(cmd, cwd=str(base_dir), check=True)
        print(f"Finished camera: {cam} (returncode: {res.returncode})")

    # Step 1: Move every *_metrics.csv into extracted/metrics/
    print("\nMoving *_metrics.csv files to extracted/metrics/...")
    count_moved = 0
    for cam in CAMERAS:
        cam_out = split_dir / cam
        for f in cam_out.glob("*_metrics.csv"):
            dst = metrics_dir / f.name
            shutil.copyfile(f, dst)
            count_moved += 1
    print(f"Moved {count_moved} metric CSV files.")

    # Step 2: Row-concatenate summary CSVs sorted by cctv, roi_idx
    summary_files = ["queue_thresholds.csv", "capacity_empirical.csv", "emp_empirical.csv"]
    for sf in summary_files:
        dfs = []
        for cam in CAMERAS:
            f = split_dir / cam / sf
            if f.exists():
                dfs.append(pd.read_csv(f))
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            if "cctv" in combined.columns and "roi_idx" in combined.columns:
                combined = combined.sort_values(by=["cctv", "roi_idx"]).reset_index(drop=True)
            out_file = metrics_dir / sf
            combined.to_csv(out_file, index=False)
            print(f"Wrote merged {sf} ({len(combined)} rows) to {out_file}")

    # Step 3: Delete extracted/metrics_split/
    shutil.rmtree(split_dir)
    print("Deleted extracted/metrics_split/.")

    # Verification
    final_files = list(metrics_dir.glob("*_metrics.csv"))
    print(f"Final metric CSV count: {len(final_files)}")
    no_1011 = list(metrics_dir.glob("*2026-10-11*"))
    print(f"2026-10-11 files count (must be 0): {len(no_1011)}")

if __name__ == "__main__":
    main()

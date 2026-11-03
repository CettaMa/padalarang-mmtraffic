"""Unit tests for speed validation toolkit (tools/speed_val.py)."""

from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path

tools_dir = Path(__file__).resolve().parent
repo_root = tools_dir.parent
import sys
sys.path.insert(0, str(tools_dir))
sys.path.insert(0, str(repo_root / "script"))

import numpy as np
import pandas as pd

from speed_val import (
    CLASSES,
    PER_CLASS,
    _boot_ci,
    _metrics,
    cmd_report,
    gt_speed,
    home_segment,
    select_final,
)


class TestSpeedVal(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_home_segment_containment_and_boundaries(self):
        names = ["seg_000000.ts", "seg_000028.ts", "seg_000056.ts"]
        starts = [0.0, 28.0, 56.0]

        # 1. Fits comfortably inside seg_000028: start (28) <= 35-1 (34) and last+1 (45+1=46) < next (56)
        seg = home_segment(starts, names, 35.0, 45.0)
        self.assertEqual(seg, "seg_000028.ts")

        # 2. Starts too close to segment start: 28.5 - 1.0 = 27.5 < 28.0 -> cand_idx = 0 (seg 0),
        # but end_limit is min(0+28, 28) = 28.0, and 35 + 1.0 = 36.0 >= 28.0 -> None
        seg = home_segment(starts, names, 28.5, 35.0)
        self.assertIsNone(seg)

        # 3. Last observation too close to or crossing next segment start (56.0):
        # 55.5 + 1 = 56.5 >= 56.0 -> None
        seg = home_segment(starts, names, 35.0, 55.5)
        self.assertIsNone(seg)

        # 4. Final segment has no upper boundary segment (end_limit = 56 + 28 = 84.0)
        seg = home_segment(starts, names, 62.0, 75.0)
        self.assertEqual(seg, "seg_000056.ts")

        # 5. Sparse segments regression case: only seg 0 and seg 1000 exist locally
        # cand_idx = 0, start = 0, end_limit = min(0+28, 1000) = 28.0
        # (500, 510) must return None, NOT seg_000000.ts
        seg_sparse = home_segment([0.0, 1000.0], ["seg_000000.ts", "seg_001000.ts"], 500.0, 510.0)
        self.assertIsNone(seg_sparse)

        # 6. Empty inputs
        self.assertIsNone(home_segment([], [], 10.0, 20.0))
    def test_rank_determinism(self):
        # Shuffling input rows must not alter the resulting rank per sample_id
        # Shuffling input DataFrame rows must yield identical rank per sample_id
        seed_key = "20261029|padalarang_-_giant_-_tagog|1|car"
        records = [
            {"sample_id": f"s_{i:02d}", "cctv_id": "cam", "roi_idx": 1, "class_name": "car",
             "date": "2026-10-02", "ts": "120000", "track_id": i}
            for i in range(25)
        ]
        df1 = pd.DataFrame(records)
        df2 = df1.sample(frac=1.0, random_state=42).reset_index(drop=True)

        def assign_ranks(df):
            df_sorted = df.sort_values(["cctv_id", "roi_idx", "class_name", "date", "ts", "track_id"]).copy()
            rng = random.Random(seed_key)
            order = list(range(len(df_sorted)))
            rng.shuffle(order)
            df_sorted["rank"] = order
            return dict(zip(df_sorted["sample_id"], df_sorted["rank"]))

        self.assertEqual(assign_ranks(df1), assign_ranks(df2))
    def test_gt_speed_interpolation_and_boundaries(self):
        # Build an identity-like 10m x 10m rectangular ROI on 100x100 pixels
        sample = {
            "native_fps": 10.0,
            "roi_width_meters": 10.0,
            "roi_length_meters": 10.0,
            "roi_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "trajectory": [
                {"frame": 10, "bottom_center": [50.0, 0.0]},
                {"frame": 20, "bottom_center": [50.0, 50.0]},
                {"frame": 30, "bottom_center": [50.0, 100.0]},
            ]
        }

        # 1. Exact frame match: entry at frame 10 (0m), exit at frame 30 (10m).
        # dt = 20 frames / 10 fps = 2.0 s. dist = 10 m. speed = 10 / 2 * 3.6 = 18.0 km/h.
        df, spd = gt_speed(10, 30, sample)
        self.assertEqual(df, 20)
        self.assertAlmostEqual(spd, 18.0, places=2)

        # 2. Linear interpolation between samples:
        # entry at frame 15 (halfway 10..20 -> y=25px -> 2.5m)
        # exit at frame 25 (halfway 20..30 -> y=75px -> 7.5m)
        # dist = 5.0 m. dt = 10 frames / 10 fps = 1.0 s. speed = 5.0 / 1.0 * 3.6 = 18.0 km/h.
        df, spd = gt_speed(15, 25, sample)
        self.assertEqual(df, 10)
        self.assertAlmostEqual(spd, 18.0, places=2)

        # 3. Marked frame outside trajectory bounds -> (None, None)
        df, spd = gt_speed(5, 25, sample)
        self.assertIsNone(df)
        self.assertIsNone(spd)

        df, spd = gt_speed(15, 35, sample)
        self.assertIsNone(df)
        self.assertIsNone(spd)

        # 4. exit_frame <= entry_frame -> (None, None)
        df, spd = gt_speed(20, 20, sample)
        self.assertIsNone(df)
        self.assertIsNone(spd)

    def test_report_quota_selection_and_class_overflow(self):
        # Test quota logic: 7 accepted cars with various ranks -> top 5 kept, 2 become class_overflow
        # Also test 1 car reclassified to motorcycle
        annotations = {}
        # 7 cars with ranks 9, 2, 7, 1, 0, 8, 4
        car_ranks = [9, 2, 7, 1, 0, 8, 4]
        for i, rk in enumerate(car_ranks):
            annotations[f"car_{i}"] = {
                "sample_id": f"car_{i}",
                "cctv_id": "padalarang_-_jl_baru_kbp",
                "roi_idx": 0,
                "perspective": "depan",
                "stratum_class": "car",
                "true_class": "car",
                "rank": rk,
                "predicted_speed_kmh": 20.0 + i,
                "ground_truth_speed_kmh": 21.0 + i,
                "false_positive": False,
                "invalid_crossing": False,
            }

        out_dir = self.tmp / "speed_val"
        out_dir.mkdir(parents=True, exist_ok=True)
        anno_file = out_dir / "annotations.json"
        with open(anno_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f)

        # Monkeypatch OUT in speed_val module for this test
        import speed_val
        old_out = speed_val.OUT
        speed_val.OUT = out_dir
        try:
            import argparse
            res = cmd_report(argparse.Namespace())
            self.assertEqual(res, 0)

            # Check metrics.json
            with open(out_dir / "metrics.json") as f:
                metrics = json.load(f)
            self.assertEqual(metrics["overall"]["overall"]["n"], 5)

            # Check quota.csv
            df_q = pd.read_csv(out_dir / "quota.csv")
            sub_q = df_q[(df_q.cctv_id == "padalarang_-_jl_baru_kbp") & (df_q.roi_idx == 1) & (df_q["class"] == "car")]
            self.assertEqual(int(sub_q.iloc[0]["accepted"]), 5)
            self.assertEqual(int(sub_q.iloc[0]["class_overflow"]), 2)
        finally:
            speed_val.OUT = old_out

    def test_metrics_calculation(self):
        # 3-row synthetic dataset:
        # pred = [20.0, 30.0, 40.0]
        # gt   = [22.0, 28.0, 43.0]
        # err  = [-2.0, +2.0, -3.0]
        # |err|= [ 2.0,  2.0,  3.0] -> MAE = 7/3 = 2.333... -> 2.33
        # err^2= [ 4.0,  4.0,  9.0] -> MSE = 17/3 = 5.666... -> RMSE = sqrt(5.666...) = 2.380... -> 2.38
        # bias = (-2 + 2 - 3)/3 = -1.0
        # mape = (2/22 + 2/28 + 3/43)/3 * 100 = (0.090909 + 0.071428 + 0.069767)/3 * 100 = 7.7368% -> 7.74%
        df = pd.DataFrame({
            "pred_speed": [20.0, 30.0, 40.0],
            "gt_speed": [22.0, 28.0, 43.0],
        })
        m = _metrics(df)
        expected_keys = {
            "n", "mae", "mae_lo", "mae_hi", "rmse", "rmse_lo", "rmse_hi",
            "bias", "bias_lo", "bias_hi", "sd", "loa_lo", "loa_hi", "mape",
            "r", "r_lo", "r_hi"
        }
        self.assertEqual(set(m.keys()), expected_keys)
        self.assertEqual(m["mae"], 2.33)
        self.assertEqual(m["rmse"], 2.38)
        self.assertEqual(m["bias"], -1.00)
        self.assertEqual(m["mape"], 7.74)
        self.assertGreater(m["r"], 0.95)

    def test_bootstrap_stratified_ci_and_loa(self):
        # Synthetic 2-stratum dataset
        rows = []
        # Stratum A
        for i in range(15):
            rows.append({
                "cctv_id": "cctv_1", "roi_idx": 0, "true_class": "car",
                "pred_speed": 30.0 + i * 1.5,
                "gt_speed": 28.0 + i * 1.4,
            })
        # Stratum B
        for i in range(15):
            rows.append({
                "cctv_id": "cctv_1", "roi_idx": 1, "true_class": "motorcycle",
                "pred_speed": 45.0 + i * 2.0,
                "gt_speed": 44.0 + i * 2.1,
            })
        df = pd.DataFrame(rows)

        m1 = _metrics(df, key="test|strata2")
        m2 = _metrics(df, key="test|strata2")

        # (a) CIs bracket the point estimates
        self.assertLessEqual(m1["mae_lo"], m1["mae"])
        self.assertGreaterEqual(m1["mae_hi"], m1["mae"])

        self.assertLessEqual(m1["rmse_lo"], m1["rmse"])
        self.assertGreaterEqual(m1["rmse_hi"], m1["rmse"])

        self.assertLessEqual(m1["bias_lo"], m1["bias"])
        self.assertGreaterEqual(m1["bias_hi"], m1["bias"])

        # (b) Two calls with same key give identical CIs (seeded determinism)
        self.assertEqual(m1, m2)

        # (c) LoA equals bias ± 1.96 * sd (within rounding tolerance)
        self.assertAlmostEqual(m1["loa_lo"], m1["bias"] - 1.96 * m1["sd"], delta=0.02)
        self.assertAlmostEqual(m1["loa_hi"], m1["bias"] + 1.96 * m1["sd"], delta=0.02)

if __name__ == "__main__":
    unittest.main()

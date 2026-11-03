import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

tools_dir = Path(__file__).resolve().parent
repo_root = tools_dir.parent
import sys
sys.path.insert(0, str(tools_dir))
sys.path.insert(0, str(repo_root / "script"))

from feature_extractor import WindowLog, TARGET_FPS
from traffic_metrics import (
    build_parser,
    build_queue_zone_dict,
    chandra_emp,
    dwell_sec,
    find_cctv_video,
    occupancy,
    otsu_threshold,
    pkji_emp,
    queue_length_snapshot,
    speed_stats,
    traverse_speed_kmh,
)


class TestTrafficMetrics(unittest.TestCase):
    def test_occupancy_calculations(self):
        # Two overlapping intervals [0,2] and [1,3] (t_last values already include frame)
        # with covered 10 s and lanes 1 -> occ_time = 40%, occ_any = 30%
        # with lanes 2 -> occ_time = 20%
        recs = [(0.0, 2.0), (1.0, 3.0)]

        occ_time, occ_any = occupancy(recs, covered_sec=10.0, lanes=1)
        self.assertAlmostEqual(occ_time, 40.0, places=4)
        self.assertAlmostEqual(occ_any, 30.0, places=4)

        occ_time_2, occ_any_2 = occupancy(recs, covered_sec=10.0, lanes=2)
        self.assertAlmostEqual(occ_time_2, 20.0, places=4)
        self.assertAlmostEqual(occ_any_2, 30.0, places=4)

    def test_queue_length_snapshot(self):
        # ys = [3, 9, 18, 45], head 10, gap 12 -> (18.0, 3)
        ys = [3, 9, 18, 45]
        q_len, n_q = queue_length_snapshot(ys, head_max=10.0, gap_max=12.0)
        self.assertEqual(q_len, 18.0)
        self.assertEqual(n_q, 3)

        # ys = [15] -> (0.0, 0) because head vehicle > 10.0
        q_len2, n_q2 = queue_length_snapshot([15], head_max=10.0, gap_max=12.0)
        self.assertEqual(q_len2, 0.0)
        self.assertEqual(n_q2, 0)

    def test_otsu_threshold(self):
        np.random.seed(42)
        # Bimodal synthetic data: 500 samples ~N(2,1) and 500 ~N(40,5), clipped >= 0 -> result in [8, 25]
        low_speeds = np.clip(np.random.normal(loc=2.0, scale=1.0, size=500), 0.0, 59.9)
        high_speeds = np.clip(np.random.normal(loc=40.0, scale=5.0, size=500), 0.0, 59.9)
        bimodal = np.concatenate([low_speeds, high_speeds])

        thr = otsu_threshold(bimodal, lo=3.0, hi=30.0)
        self.assertTrue(math.isfinite(thr), f"Expected finite Otsu threshold, got {thr}")
        self.assertGreaterEqual(thr, 8.0)
        self.assertLessEqual(thr, 25.0)

        # Unimodal N(40,5) -> NaN, because result falls outside [3, 30]
        unimodal = high_speeds
        thr_uni = otsu_threshold(unimodal, lo=3.0, hi=30.0)
        self.assertTrue(math.isnan(thr_uni), f"Expected NaN for unimodal, got {thr_uni}")

    def test_chandra_emp(self):
        # V_car = 40, V_mc = 40, areas 5.39/1.2 -> EMP_mc ~= 0.2226
        mean_speeds = {"car": 40.0, "motorcycle": 40.0}
        areas = {"car": 5.39, "motorcycle": 1.20}
        emp = chandra_emp(mean_speeds, areas)
        self.assertIn("motorcycle", emp)
        self.assertAlmostEqual(emp["motorcycle"], 1.20 / 5.39, places=4)
        self.assertAlmostEqual(emp["motorcycle"], 0.2226345, places=4)

    def test_speed_stats(self):
        # [10, 20, 30] -> tms 20, sms 16.3636, std 10, cv 0.5
        stats = speed_stats([10, 20, 30])
        self.assertEqual(stats["spd_n"], 3)
        self.assertAlmostEqual(stats["spd_tms"], 20.0, places=4)
        self.assertAlmostEqual(stats["spd_sms"], 16.363636, places=4)
        self.assertAlmostEqual(stats["spd_std"], 10.0, places=4)
        self.assertAlmostEqual(stats["spd_cv"], 0.5, places=4)

    def test_pkji_emp(self):
        # flow 1799 -> low_flow; 1800 -> high_flow
        emp_cfg = {
            "flow_threshold_veh_h": 1800,
            "low_flow": {"bus": 1.3, "car": 1.0, "motorcycle": 0.40, "truck": 1.3},
            "high_flow": {"bus": 1.2, "car": 1.0, "motorcycle": 0.25, "truck": 1.2},
        }
        res_low = pkji_emp(1799, emp_cfg)
        self.assertEqual(res_low, emp_cfg["low_flow"])
        res_high = pkji_emp(1800, emp_cfg)
        self.assertEqual(res_high, emp_cfg["high_flow"])

    def test_window_log(self):
        # Build WindowLog with a single ROI having a queue_zone (4-point square mapped to 10 m x 10 m)
        # Perspective maps [100, 100]..[200, 200] to (0,0)..(10,10)
        # Points: p0=(100, 100), p1=(200, 100), p2=(200, 200), p3=(100, 200)
        rois_data = [
            {
                "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "width_meters": 10.0,
                "length_meters": 10.0,
                "queue_zone": {
                    "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                    "width_meters": 10.0,
                    "length_meters": 10.0,
                },
            }
        ]

        wlog = WindowLog(lo_sec=0.0, hi_sec=300.0, rois_data=rois_data)

        # Feed observe_frame at t = 0.0, 0.5, 1.0, 2.0 with one track moving 1 m/s along y
        # Note: In the perspective transform:
        # y=100 -> y_m=0; y=110 -> y_m=1; y=120 -> y_m=2; x=150 -> x_m=5
        feed = [
            (0.0, np.array([[150.0, 100.0]])),
            (0.5, np.array([[150.0, 105.0]])),
            (1.0, np.array([[150.0, 110.0]])),
            (2.0, np.array([[150.0, 120.0]])),
        ]

        for t, pts in feed:
            wlog.observe_frame(
                t=t,
                points=pts,
                track_ids=np.array([42]),
                class_ids=np.array([1]),
            )

        # Feed roi_obs for the vehicle, collapsing to majority class
        wlog.roi_obs(roi_idx=0, track_id=42, class_id=1, t=0.0, pos_m=np.array([5.0, 0.0]), counted_new=True)
        wlog.roi_obs(roi_idx=0, track_id=42, class_id=1, t=1.0, pos_m=np.array([5.0, 1.0]), counted_new=False)
        wlog.roi_obs(roi_idx=0, track_id=42, class_id=2, t=2.0, pos_m=np.array([5.0, 2.0]), counted_new=False)

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            frames = 3000
            wlog.write(out_dir=tmp_dir, ts_str="080000", frames=frames)

            # Assert _meta.json
            meta_path = tmp_dir / "080000_meta.json"
            self.assertTrue(meta_path.exists())
            with open(meta_path) as f:
                meta = json.load(f)

            self.assertEqual(meta["n_ticks"], 3)  # ticks at t=0, 1, 2
            self.assertAlmostEqual(meta["covered_sec"], frames / TARGET_FPS)
            self.assertEqual(meta["queue_rois"], [0])

            # Assert _q.csv.gz
            q_path = tmp_dir / "080000_q.csv.gz"
            self.assertTrue(q_path.exists())
            q_df = pd.read_csv(q_path)
            self.assertEqual(len(q_df), 3)

            speeds = q_df["speed_kmh"].tolist()
            self.assertTrue(math.isnan(speeds[0]))
            self.assertAlmostEqual(speeds[1], 3.6, places=4)
            self.assertAlmostEqual(speeds[2], 3.6, places=4)

            # Assert _veh.csv.gz
            veh_path = tmp_dir / "080000_veh.csv.gz"
            self.assertTrue(veh_path.exists())
            veh_df = pd.read_csv(veh_path)
            self.assertEqual(len(veh_df), 1)
            row = veh_df.iloc[0]
            self.assertEqual(row["roi_idx"], 0)
            self.assertEqual(row["track_id"], 42)
            self.assertEqual(row["class_id"], 1)  # Majority class 1 over 2
            self.assertEqual(row["counted_new"], 1)
            self.assertEqual(row["n_obs"], 3)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_build_queue_zone_dict_single_segment(self):
        pts = [[10, 20], [30, 20], [35, 100], [5, 100]]
        q_dict = build_queue_zone_dict(pts, n_segments=1, length_m=25.0, width_m=6.5)
        self.assertEqual(q_dict["points"], pts)
        self.assertEqual(q_dict["length_meters"], 25.0)
        self.assertEqual(q_dict["width_meters"], 6.5)
        self.assertNotIn("type", q_dict)

    def test_build_queue_zone_dict_piecewise(self):
        # 6 points for 2 segments: [p0, p1, p2, p3, p4, p5]
        pts = [[10, 20], [30, 20], [35, 60], [5, 60], [40, 100], [0, 100]]
        q_dict = build_queue_zone_dict(pts, n_segments=2, length_m=30.0, width_m=7.0, lengths_m_str="12.0,18.0")
        self.assertEqual(q_dict["type"], "piecewise")
        self.assertEqual(len(q_dict["segments"]), 2)
        self.assertEqual(q_dict["segments"][0]["length_meters"], 12.0)
        self.assertEqual(q_dict["segments"][1]["length_meters"], 18.0)
        self.assertEqual(q_dict["length_meters"], 30.0)
        self.assertEqual(q_dict["width_meters"], 7.0)

    def test_find_cctv_video(self):
        vid = find_cctv_video("padalarang_-_gedong_5_bawah")
        self.assertIsNotNone(vid)
        self.assertTrue(vid.exists())
        self.assertTrue(vid.name.endswith(".ts") or vid.name.endswith(".mp4"))

    def test_parser_draw_queue_zone_no_args(self):
        parser = build_parser()
        args = parser.parse_args(["draw-queue-zone"])
        self.assertEqual(args.subcommand, "draw-queue-zone")
        self.assertTrue(callable(args.func))


if __name__ == "__main__":
    unittest.main()

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

tools_dir = Path(__file__).resolve().parent
repo_root = tools_dir.parent
import sys
sys.path.insert(0, str(tools_dir))

from event_features import (
    REGISTRY_COLUMNS,
    corridor_centroid,
    event_severity,
    event_vector,
    haversine_km,
    validate_events,
    cmd_annotate,
)


class TestEventFeatures(unittest.TestCase):
    def setUp(self):
        self.default_params = {
            "region_radius_km": 40.0,
            "local_radius_km": 5.0,
            "corridor_radius_km": 2.0,
            "roadwork_radius_km": 10.0,
            "attendance_tiers": [[20000, 3], [5000, 2], [1000, 1]],
            "major_severity": 3,
            "days_cap": 30.0,
            "halo_hours": {
                "concert": [4, 2],
                "festival": [2, 2],
                "sport": [4, 2],
                "religious": [2, 2],
                "community": [2, 1],
                "expo": [2, 1],
                "demonstration": [1, 1],
                "roadwork": [0, 0],
                "road_closure": [0, 0],
                "mudik": [0, 0],
                "school_holiday": [0, 0],
            },
            "venues": {
                "Stadion Gelora Bandung Lautan Api": [-6.95756, 107.71205],
                "Stadion Si Jalak Harupat": [-6.99648, 107.52973],
                "Sasana Budaya Ganesha": [-6.88624, 107.60827],
                "Stasiun Padalarang": [-6.84166, 107.49635],
            },
        }
        self.centroid = (-6.847436, 107.493364)

    def test_haversine_and_centroid(self):
        cameras = {
            "c1": {"lat": -6.840, "lon": 107.490},
            "c2": {"lat": -6.850, "lon": 107.500},
        }
        c = corridor_centroid(cameras)
        self.assertAlmostEqual(c[0], -6.845, places=4)
        self.assertAlmostEqual(c[1], 107.495, places=4)

        d = haversine_km(self.centroid[0], self.centroid[1], -6.84166, 107.49635)
        self.assertAlmostEqual(d, 0.72, delta=0.2)

    def test_event_severity(self):
        p = self.default_params
        # 25000 attendance at 27.1 km -> 3
        self.assertEqual(event_severity("concert", 25000, 27.1, None, p), 3)
        # 10000 at 13.5 km -> 2
        self.assertEqual(event_severity("concert", 10000, 13.5, None, p), 2)
        # 3000 at 13.5 km -> 1
        self.assertEqual(event_severity("festival", 3000, 13.5, None, p), 1)

        # 500 at 3.7 km -> 1 (local bonus)
        self.assertEqual(event_severity("community", 500, 3.7, None, p), 1)
        # 500 at 13.5 km -> 0
        self.assertEqual(event_severity("community", 500, 13.5, None, p), 0)
        # 50000 at 41 km -> 0 (> region radius)
        self.assertEqual(event_severity("concert", 50000, 41.0, None, p), 0)

        # road_closure at 0.6 km -> 3; at 8 km -> 1; at 12 km -> 0
        self.assertEqual(event_severity("road_closure", None, 0.6, None, p), 3)
        self.assertEqual(event_severity("roadwork", None, 8.0, None, p), 1)
        self.assertEqual(event_severity("road_closure", None, 12.0, None, p), 0)

        # mudik -> 3
        self.assertEqual(event_severity("mudik", None, None, None, p), 3)
        # override 2 beats tiers
        self.assertEqual(event_severity("concert", 50000, 3.7, 2, p), 2)

    def test_validate_events_errors(self):
        p = self.default_params
        centroid = self.centroid

        def make_row(**kwargs):
            base = {
                "event_id": "ev_test_1",
                "name": "Test Event",
                "category": "concert",
                "start_local": "2026-09-17 19:00",
                "end_local": "2026-09-17 22:00",
                "venue": "Sasana Budaya Ganesha",
                "lat": "",
                "lon": "",
                "expected_attendance": "10000",
                "pre_hours": "4",
                "post_hours": "2",
                "severity_override": "",
                "source_url": "https://example.com/event",
                "source_url_2": "",
                "verified": "1",
                "notes": "",
            }
            base.update(kwargs)
            return base

        # 1. Missing column in df
        bad_cols_df = pd.DataFrame([{"event_id": "e1"}])
        _, errs, _ = validate_events(bad_cols_df, p, centroid)
        self.assertTrue(any("missing column" in e for e in errs))

        # 2. Duplicate event_id
        r1 = make_row(event_id="dup_1")
        r2 = make_row(event_id="dup_1")
        df_dup = pd.DataFrame([r1, r2])
        _, errs, _ = validate_events(df_dup, p, centroid)
        self.assertTrue(any("dup_1: duplicate event_id" in e for e in errs))

        # 3. Unknown category
        df_cat = pd.DataFrame([make_row(event_id="e_cat", category="carnival")])
        _, errs, _ = validate_events(df_cat, p, centroid)
        self.assertTrue(any("e_cat: unknown category" in e for e in errs))

        # 4. Unparsable time
        df_time = pd.DataFrame([make_row(event_id="e_time", start_local="invalid-date")])
        _, errs, _ = validate_events(df_time, p, centroid)
        self.assertTrue(any("e_time: unparsable start_local" in e for e in errs))

        # 5. end_local <= start_local
        df_end = pd.DataFrame([make_row(event_id="e_end", end_local="2026-09-17 18:00")])
        _, errs, _ = validate_events(df_end, p, centroid)
        self.assertTrue(any("e_end: end_local <= start_local" in e for e in errs))

        # 6. source_url not starting with http
        df_url = pd.DataFrame([make_row(event_id="e_url", source_url="ftp://example.com")])
        _, errs, _ = validate_events(df_url, p, centroid)
        self.assertTrue(any("e_url: source_url must start with http" in e for e in errs))

        # 7. only one of lat/lon set
        df_lat = pd.DataFrame([make_row(event_id="e_lat", venue="", lat="-6.84", lon="")])
        _, errs, _ = validate_events(df_lat, p, centroid)
        self.assertTrue(any("e_lat: only one of lat/lon set" in e for e in errs))

        # 8. verified not in {0, 1}
        df_ver = pd.DataFrame([make_row(event_id="e_ver", verified="maybe")])
        _, errs, _ = validate_events(df_ver, p, centroid)
        self.assertTrue(any("e_ver: verified must be 0 or 1" in e for e in errs))

        # 9. attendance category with blank attendance and override
        df_att = pd.DataFrame([make_row(event_id="e_att", expected_attendance="", severity_override="")])
        _, errs, _ = validate_events(df_att, p, centroid)
        self.assertTrue(any("e_att: attendance category" in e for e in errs))

        # 10. roadwork without coordinates and without venue
        df_rw = pd.DataFrame([make_row(event_id="e_rw", category="roadwork", venue="", lat="", lon="", expected_attendance="")])
        _, errs, _ = validate_events(df_rw, p, centroid)
        self.assertTrue(any("e_rw: category 'roadwork' requires coordinates" in e for e in errs))

    def test_validate_events_venue_fill_and_unverified(self):
        p = self.default_params
        centroid = self.centroid

        # Verified 0 is excluded from kept
        row_unv = {
            "event_id": "unv_1",
            "name": "Unverified Concert",
            "category": "concert",
            "start_local": "2026-09-17 19:00",
            "end_local": "2026-09-17 22:00",
            "venue": "Sasana Budaya Ganesha",
            "lat": "",
            "lon": "",
            "expected_attendance": "10000",
            "pre_hours": "4",
            "post_hours": "2",
            "severity_override": "",
            "source_url": "https://example.com/unv",
            "source_url_2": "",
            "verified": "0",
            "notes": "",
        }
        df_unv = pd.DataFrame([row_unv])
        kept, errs, _ = validate_events(df_unv, p, centroid)
        self.assertEqual(len(errs), 0)
        self.assertEqual(len(kept), 0)

        # Blank lat/lon with venue Stasiun Padalarang gets coordinates filled and d_km ≈ 0.7 km
        row_pad = {
            "event_id": "pad_1",
            "name": "Station Market",
            "category": "community",
            "start_local": "2026-09-17 08:00",
            "end_local": "2026-09-17 12:00",
            "venue": "Stasiun Padalarang",
            "lat": "",
            "lon": "",
            "expected_attendance": "2000",
            "pre_hours": "",
            "post_hours": "",
            "severity_override": "",
            "source_url": "https://example.com/pad",
            "source_url_2": "",
            "verified": "1",
            "notes": "",
        }
        df_pad = pd.DataFrame([row_pad])
        kept, errs, _ = validate_events(df_pad, p, centroid)
        self.assertEqual(len(errs), 0)
        self.assertEqual(len(kept), 1)
        r = kept.iloc[0]
        self.assertAlmostEqual(r["d_km"], 0.72, delta=0.2)
        self.assertEqual(r["severity"], 2)  # base 1 + local bonus 1 = 2

    def test_event_vector_halo_boundaries(self):
        p = self.default_params
        # Concert 2026-09-17 19:00-22:00 at Sabuga, attendance 10000, halo [4, 2]
        kept_row = {
            "event_id": "concert_1",
            "category": "concert",
            "start": pd.Timestamp("2026-09-17 19:00:00"),
            "end": pd.Timestamp("2026-09-17 22:00:00"),
            "win_start": pd.Timestamp("2026-09-17 15:00:00"),
            "win_end": pd.Timestamp("2026-09-18 00:00:00"),
            "severity": 2,
            "d_km": 13.5,
        }
        kept = pd.DataFrame([kept_row])

        timestamps = pd.Series([
            pd.Timestamp("2026-09-17 14:59:59"),
            pd.Timestamp("2026-09-17 15:00:00"),
            pd.Timestamp("2026-09-17 19:00:00"),
            pd.Timestamp("2026-09-17 22:00:01"),
            pd.Timestamp("2026-09-18 00:00:00"),
            pd.Timestamp("2026-09-18 00:00:01"),
        ])

        vec = event_vector(timestamps, kept, p)

        # 14:59:59 -> all event cols 0
        self.assertEqual(vec.iloc[0]["evt_local"], 0)
        self.assertEqual(vec.iloc[0]["evt_severity"], 0)

        # 15:00:00 -> evt_local=1, evt_severity=2
        self.assertEqual(vec.iloc[1]["evt_local"], 1)
        self.assertEqual(vec.iloc[1]["evt_severity"], 2)

        # 19:00:00 -> evt_local=1
        self.assertEqual(vec.iloc[2]["evt_local"], 1)

        # 22:00:01 -> evt_local=1
        self.assertEqual(vec.iloc[3]["evt_local"], 1)

        # 2026-09-18 00:00:00 -> evt_local=1
        self.assertEqual(vec.iloc[4]["evt_local"], 1)

        # 2026-09-18 00:00:01 -> 0
        self.assertEqual(vec.iloc[5]["evt_local"], 0)
        self.assertEqual(vec.iloc[5]["evt_severity"], 0)

    def test_major_event_distances(self):
        p = self.default_params
        # Mudik row 2026-09-20 00:00 – 2026-09-21 23:59 (severity 3)
        kept_row = {
            "event_id": "mudik_1",
            "category": "mudik",
            "start": pd.Timestamp("2026-09-20 00:00:00"),
            "end": pd.Timestamp("2026-09-21 23:59:00"),
            "win_start": pd.Timestamp("2026-09-20 00:00:00"),
            "win_end": pd.Timestamp("2026-09-21 23:59:00"),
            "severity": 3,
            "d_km": None,
        }
        kept = pd.DataFrame([kept_row])

        ts = pd.Series([
            pd.Timestamp("2026-09-18 00:00:00"),
            pd.Timestamp("2026-09-20 12:00:00"),
            pd.Timestamp("2026-12-31 00:00:00"),
        ])
        vec = event_vector(ts, kept, p)

        # at 2026-09-18 00:00 -> days_to_major = 2.0
        self.assertAlmostEqual(vec.iloc[0]["evt_days_to_major"], 2.0, places=2)
        # inside the window -> both 0
        self.assertEqual(vec.iloc[1]["evt_days_to_major"], 0.0)
        self.assertEqual(vec.iloc[1]["evt_days_since_major"], 0.0)
        # at 2026-12-31 -> days_since_major capped at 30.0
        self.assertEqual(vec.iloc[2]["evt_days_since_major"], 30.0)

    def test_calendar_flags(self):
        p = self.default_params
        kept = pd.DataFrame()
        ts = pd.Series([
            pd.Timestamp("2026-08-17 12:00:00"),  # Independence Day (public holiday)
            pd.Timestamp("2026-03-20 12:00:00"),  # Cuti bersama Idul Fitri (govt holiday)
            pd.Timestamp("2026-09-19 12:00:00"),  # Saturday (weekend)
            pd.Timestamp("2026-09-17 12:00:00"),  # Thursday (not weekend)
        ])
        vec = event_vector(ts, kept, p)

        self.assertEqual(
            list(vec.columns),
            [
                "evt_holiday",
                "evt_weekend",
                "evt_local",
                "evt_severity",
                "evt_days_to_major",
                "evt_days_since_major",
            ],
        )
        self.assertEqual(vec.iloc[0]["evt_holiday"], 1)
        self.assertEqual(vec.iloc[1]["evt_holiday"], 1)
        self.assertEqual(vec.iloc[2]["evt_weekend"], 1)
        self.assertEqual(vec.iloc[3]["evt_holiday"], 0)
        self.assertEqual(vec.iloc[3]["evt_weekend"], 0)

    def test_annotate_end_to_end(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            events_csv = tmp_dir / "events.csv"
            params_json = tmp_dir / "event_params.json"
            features_dir = tmp_dir / "extracted"
            out_dir = tmp_dir / "annotated"
            features_dir.mkdir(parents=True)

            with open(params_json, "w", encoding="utf-8") as f:
                json.dump(self.default_params, f)

            # Registry with 1 verified event
            events_df = pd.DataFrame([{
                "event_id": "test_concert",
                "name": "Bandung Live",
                "category": "concert",
                "start_local": "2026-09-17 19:00",
                "end_local": "2026-09-17 22:00",
                "venue": "Sasana Budaya Ganesha",
                "lat": "-6.88624",
                "lon": "107.60827",
                "expected_attendance": "10000",
                "pre_hours": "4",
                "post_hours": "2",
                "severity_override": "",
                "source_url": "https://example.com/live",
                "source_url_2": "",
                "verified": "1",
                "notes": "",
            }], columns=REGISTRY_COLUMNS)
            events_df.to_csv(events_csv, index=False)

            # 3-row feature CSV
            feat_csv = features_dir / "camera1_2026-09-17_metadata_features.csv"
            feat_df = pd.DataFrame({
                "timestamp": [
                    "2026-09-17 12:00:00",
                    "2026-09-17 16:00:00",
                    "2026-09-17 20:00:00",
                ],
                "flow": [10.0, 25.0, 40.0],
                "density": [5.0, 12.0, 20.0],
                "evt_phase_pre": [0, 0, 0],
            })
            feat_df.to_csv(feat_csv, index=False)

            # Call annotate CLI handler
            args = argparse.Namespace(
                events=str(events_csv),
                params=str(params_json),
                features_glob=str(features_dir / "*_metadata_features.csv"),
                out=str(out_dir),
                dates=None,
            )
            code = cmd_annotate(args)
            self.assertEqual(code, 0)

            out_csv = out_dir / feat_csv.name
            self.assertTrue(out_csv.exists())
            ann_df = pd.read_csv(out_csv)

            # Assert original columns unchanged and first
            self.assertEqual(list(ann_df.columns[:3]), ["timestamp", "flow", "density"])

            # Assert expected evt_* columns appended
            expected_evt_cols = [
                "evt_holiday",
                "evt_weekend",
                "evt_local",
                "evt_severity",
                "evt_days_to_major",
                "evt_days_since_major",
            ]
            self.assertEqual(list(ann_df.columns[3:]), expected_evt_cols)

            # Assert values:
            # Row 0 (12:00:00): outside halo -> evt_local=0
            self.assertEqual(ann_df.iloc[0]["evt_local"], 0)
            # Row 1 (16:00:00): in pre-halo -> evt_local=1, evt_severity=2
            self.assertEqual(ann_df.iloc[1]["evt_local"], 1)
            self.assertEqual(ann_df.iloc[1]["evt_severity"], 2)
            # Row 2 (20:00:00): during -> evt_local=1, evt_severity=2
            self.assertEqual(ann_df.iloc[2]["evt_local"], 1)
            self.assertEqual(ann_df.iloc[2]["evt_severity"], 2)

            # Idempotence: running again yields identical output
            cmd_annotate(args)
            ann_df_2 = pd.read_csv(out_csv)
            pd.testing.assert_frame_equal(ann_df, ann_df_2)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

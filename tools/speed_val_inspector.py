"""Speed Accuracy Inspector & Annotator for JOIV Revision.

Interactive OpenCV GUI for human frame-level marking of vehicle ROI entry and exit frames.
Features:
  - In-memory JPEG frame cache for reliable frame navigation without .ts seeking glitches.
  - Linear interpolation of vehicle bottom-center centroid at arbitrary native frames.
  - Class correction keys (5: bus, 6: car, 7: motorcycle, 8: truck).
  - Invalid crossing toggle (x: stopped / occluded / ID switch).
  - Live quota tracking per stratum (target: 5 accepted vehicles per class per ROI).
  - Gamepad (Xbox via XInput) and keyboard dual-input controls.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
import platform
import sys
import time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "script"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import cv2
import numpy as np
import pandas as pd

from speed_val import OUT, PER_CLASS, gt_speed

SAMPLED_JSON = OUT / "candidates.json"
ANNOTATIONS_JSON = OUT / "annotations.json"
REPORT_CSV = OUT / "annotations.csv"

CLASS_KEYS = {
    ord("5"): "bus",
    ord("6"): "car",
    ord("7"): "motorcycle",
    ord("8"): "truck",
}

CAUSE_KEYS = {
    ord("j"): "stopped_queued",
    ord("o"): "occluded",
    ord("i"): "id_switch_fragment",
    ord("t"): "partial_track",
    ord("b"): "box_misfit",
    ord("e"): "other",
}

# (key, cause id, short label, one-line definition) shown in the causes-mode cheat sheet
CAUSE_HELP = [
    (ord("j"), "stopped_queued", "Stopped / queued", "vehicle stops or crawls in the ROI"),
    (ord("o"), "occluded", "Occluded", "hidden by another vehicle or object"),
    (ord("i"), "id_switch_fragment", "ID switch / split", "track jumps to another vehicle or splits"),
    (ord("t"), "partial_track", "Partial track", "track starts or ends inside the ROI"),
    (ord("b"), "box_misfit", "Box misfit", "box does not sit on the vehicle's road contact"),
    (ord("e"), "other", "Other", "none of the above"),
]

class SpeedAccuracyInspector:
    def __init__(self, json_path: Path = SAMPLED_JSON, mode: str = "annotate", annotator: str = "A1"):
        self.mode = mode
        self.annotator = annotator
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Candidate file {self.json_path} not found. Run 'python tools/speed_val.py prepare' first.")

        with open(self.json_path, "r", encoding="utf-8") as f:
            all_candidates = json.load(f)

        if not all_candidates:
            raise ValueError(f"Candidate file {self.json_path} is empty.")

        cand_map = {self.get_sample_key(s): s for s in all_candidates}

        self.annotations = {}
        self.causes = {}
        self.ref_annotations = {}

        if self.mode == "annotate":
            self.annotations_json = OUT / "annotations.json"
            self.report_csv = OUT / "annotations.csv"
            self.samples = all_candidates
            self.samples.sort(key=lambda s: (s["cctv_id"], s["roi_idx"], s.get("stratum_class", s.get("class_name")), s.get("rank", 0)))
            if self.annotations_json.exists():
                try:
                    with open(self.annotations_json, "r", encoding="utf-8") as f:
                        self.annotations = json.load(f)
                    print(f"[i] Loaded {len(self.annotations)} existing annotations from {self.annotations_json}")
                except Exception as e:
                    print(f"[!] Failed to load existing annotations: {e}")

        elif self.mode == "second":
            self.annotations_json = OUT / "annotations_second.json"
            self.report_csv = OUT / "annotations_second.csv"
            subset_path = OUT / "second_subset.json"
            if not subset_path.exists():
                raise FileNotFoundError(f"{subset_path} not found. Run 'python tools/speed_val.py subsets' first.")
            with open(subset_path, "r", encoding="utf-8") as f:
                subset_ids = json.load(f)
            self.samples = [cand_map[sid] for sid in subset_ids if sid in cand_map]
            if self.annotations_json.exists():
                try:
                    with open(self.annotations_json, "r", encoding="utf-8") as f:
                        self.annotations = json.load(f)
                    print(f"[i] Loaded {len(self.annotations)} existing second annotations from {self.annotations_json}")
                except Exception as e:
                    print(f"[!] Failed to load existing second annotations: {e}")

        elif self.mode == "causes":
            self.causes_json = OUT / "causes.json"
            self.ref_annotations_json = OUT / "annotations.json"
            causes_list_path = OUT / "causes_list.json"
            if not causes_list_path.exists():
                raise FileNotFoundError(f"{causes_list_path} not found. Run 'python tools/speed_val.py subsets' first.")
            with open(causes_list_path, "r", encoding="utf-8") as f:
                causes_list = json.load(f)
            self.samples = []
            for item in causes_list:
                sid = item["sample_id"]
                if sid in cand_map:
                    s = copy.deepcopy(cand_map[sid])
                    s["source"] = item["source"]
                    self.samples.append(s)
            if not self.ref_annotations_json.exists():
                raise FileNotFoundError(f"{self.ref_annotations_json} not found.")
            with open(self.ref_annotations_json, "r", encoding="utf-8") as f:
                self.ref_annotations = json.load(f)
            if self.causes_json.exists():
                try:
                    with open(self.causes_json, "r", encoding="utf-8") as f:
                        self.causes = json.load(f)
                    print(f"[i] Loaded {len(self.causes)} existing causes from {self.causes_json}")
                except Exception as e:
                    print(f"[!] Failed to load existing causes: {e}")

        self.current_idx = 0
        self.window_name = f"ROI Speed Inspector [{self.mode.upper()}] - {self.annotator}"

        # Frame cache for the currently inspected sample
        self.cached_sample_id = None
        self.frame_cache: dict[int, np.ndarray] = {}
        self.cache_min_frame = 1
        self.cache_max_frame = 1

        # Gamepad setup
        self.gamepad, self.xinput_lib = self._init_gamepad()
        self._gp_prev_buttons = 0
        self.GAMEPAD_TRIGGER_THRESHOLD = 30
        self.GAMEPAD_SCRUB_REPEAT_SEC = 0.04
        self._last_scrub_time = 0.0
        self._last_scrub_dir = 0
        self.GAMEPAD_STICK_DEADZONE = 6000
        self.GAMEPAD_STICK_SPEED_FACTOR = 3

    XINPUT_BTN = {
        "DPAD_UP":        0x0001,
        "DPAD_DOWN":      0x0002,
        "DPAD_LEFT":      0x0004,
        "DPAD_RIGHT":     0x0008,
        "START":          0x0010,
        "BACK":           0x0020,
        "LEFT_THUMB":     0x0040,
        "RIGHT_THUMB":    0x0080,
        "LB":             0x0100,
        "RB":             0x0200,
        "A":              0x1000,
        "B":              0x2000,
        "X":              0x4000,
        "Y":              0x8000,
    }

    def _init_gamepad(self):
        if platform.system() != "Windows":
            print("[i] Gamepad XInput supported on Windows only; keyboard controls active.")
            return None, None
        lib = None
        for name in ("xinput1_4", "xinput1_3", "XInput9_1_0"):
            try:
                lib = getattr(ctypes.windll, name)
                break
            except Exception:
                pass
        if lib is None:
            print("[!] XInput library not found; keyboard controls active.")
            return None, None

        class XINPUT_GAMEPAD(ctypes.Structure):
            _fields_ = [
                ("wButtons", ctypes.c_ushort),
                ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short),
                ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short),
            ]

        class XINPUT_STATE(ctypes.Structure):
            _fields_ = [("dwPacketNumber", ctypes.c_ulong), ("Gamepad", XINPUT_GAMEPAD)]

        lib.XInputGetState.argtypes = [ctypes.c_ulong, ctypes.POINTER(XINPUT_STATE)]
        lib.XInputGetState.restype = ctypes.c_ulong

        class _XInput:
            def __init__(self, lib, state_cls):
                self.lib = lib
                self.state_cls = state_cls

            def get_state(self, user_index=0):
                state = self.state_cls()
                if self.lib.XInputGetState(user_index, ctypes.byref(state)) != 0:
                    return None
                return state

        return _XInput(lib, XINPUT_STATE), lib

    def _poll_gamepad(self):
        if self.gamepad is None:
            return None
        state = self.gamepad.get_state(0)
        if state is None:
            self._gp_prev_buttons = 0
            self._last_scrub_time = 0.0
            return None

        cur = state.Gamepad.wButtons
        pressed = cur & ~self._gp_prev_buttons
        self._gp_prev_buttons = cur

        if pressed & self.XINPUT_BTN["RB"]:
            return "next_frame"
        if pressed & self.XINPUT_BTN["LB"]:
            return "prev_frame"
        if pressed & self.XINPUT_BTN["A"]:
            return "entry"
        if pressed & self.XINPUT_BTN["B"]:
            return "exit"
        if pressed & self.XINPUT_BTN["Y"]:
            return "next_object"
        if pressed & self.XINPUT_BTN["X"]:
            return "prev_object"
        if pressed & self.XINPUT_BTN["DPAD_DOWN"]:
            return "toggle_fp"
        if pressed & self.XINPUT_BTN["START"]:
            return "playpause"
        if pressed & self.XINPUT_BTN["BACK"]:
            return "quit"

        thr = self.GAMEPAD_TRIGGER_THRESHOLD
        rt = state.Gamepad.bRightTrigger
        lt = state.Gamepad.bLeftTrigger
        if rt >= thr or lt >= thr:
            now = time.monotonic()
            if now - self._last_scrub_time >= self.GAMEPAD_SCRUB_REPEAT_SEC:
                self._last_scrub_time = now
                direction = 1 if rt >= lt else -1
                self._last_scrub_dir = direction
                return ("scrub", 1, direction)
            return None

        lx = state.Gamepad.sThumbLX
        ly = state.Gamepad.sThumbLY
        mag = lx if abs(lx) >= abs(ly) else ly
        if abs(mag) >= self.GAMEPAD_STICK_DEADZONE:
            now = time.monotonic()
            if now - self._last_scrub_time >= self.GAMEPAD_SCRUB_REPEAT_SEC:
                self._last_scrub_time = now
                frac = abs(mag) / 32767.0
                direction = 1 if mag > 0 else -1
                self._last_scrub_dir = direction
                steps = max(1, round(frac * self.GAMEPAD_STICK_SPEED_FACTOR))
                return ("scrub", steps, direction)
            return None

        self._last_scrub_time = 0.0
        return None

    def get_sample_key(self, sample: dict) -> str:
        return sample["sample_id"]

    def count_accepted(self, cctv_id: str, roi_idx: int, class_name: str) -> int:
        """Count accepted annotations for this ROI and true class."""
        count = 0
        for a in self.annotations.values():
            if a.get("cctv_id") == cctv_id and a.get("roi_idx") == roi_idx:
                t_cls = a.get("true_class", a.get("class_name"))
                if t_cls == class_name:
                    if not a.get("false_positive") and not a.get("invalid_crossing") and a.get("ground_truth_speed_kmh") is not None:
                        count += 1
        return count

    def is_stratum_full(self, sample: dict) -> bool:
        """True if stratum quota is already reached (5 accepted)."""
        if self.mode != "annotate":
            return False
        cctv = sample["cctv_id"]
        roi = sample["roi_idx"]
        s_cls = sample.get("stratum_class", sample.get("class_name"))
        return self.count_accepted(cctv, roi, s_cls) >= PER_CLASS

    def load_frame_cache(self, sample: dict) -> bool:
        sample_id = self.get_sample_key(sample)
        if self.cached_sample_id == sample_id and self.frame_cache:
            return True

        video_path = sample["video_path"]
        if not os.path.exists(video_path):
            print(f"[!] Video file not found: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] Failed to open video: {video_path}")
            return False

        first_f = sample.get("first_frame", 1)
        last_f = sample.get("last_frame", first_f + 50)
        min_f = max(1, first_f - 25)
        max_f = last_f + 25

        print(f"[*] Caching frames {min_f}..{max_f} from {os.path.basename(video_path)}...")
        self.frame_cache.clear()
        f_idx = 0
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            f_idx += 1
            if f_idx >= min_f:
                success, buf = cv2.imencode(".jpg", frame, encode_params)
                if success:
                    self.frame_cache[f_idx] = buf
            if f_idx >= max_f:
                break

        cap.release()
        self.cached_sample_id = sample_id
        self.cache_min_frame = min_f
        self.cache_max_frame = max(self.frame_cache.keys()) if self.frame_cache else min_f
        print(f"[✓] Cached {len(self.frame_cache)} frames.")
        return len(self.frame_cache) > 0

    def get_frame(self, frame_idx: int) -> np.ndarray | None:
        buf = self.frame_cache.get(frame_idx)
        if buf is None:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def draw_hud(self, frame, sample, curr_frame_idx, is_playing, entry_frame, exit_frame,
                 true_class, false_positive, invalid_crossing, curr_cause=None, stick_rate=0):
        h, w = frame.shape[:2]
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, 115), (20, 20, 20), -1)
        cv2.rectangle(hud_overlay, (0, h - 85), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)

        sample_id = self.get_sample_key(sample)

        if self.mode == "causes":
            is_caused = sample_id in self.causes
            status_mark = f"[SOURCE: {sample.get('source', '').upper()}]"
            title_color = (0, 255, 0) if is_caused else (0, 255, 255)
            s_cls = sample.get("stratum_class", sample.get("class_name"))
            title_txt = f"[{self.current_idx + 1}/{len(self.samples)}] {status_mark} Rank:{sample.get('rank', 0)} | Stratum:{s_cls.upper()} [Annotator: {self.annotator}]"
        else:
            is_anno = sample_id in self.annotations and self.annotations[sample_id].get("ground_truth_speed_kmh") is not None
            if false_positive:
                status_mark = "[FALSE POSITIVE]"
                title_color = (0, 0, 255)
            elif invalid_crossing:
                status_mark = "[INVALID CROSSING]"
                title_color = (0, 165, 255)
            elif is_anno:
                status_mark = "[ANNOTATED]"
                title_color = (0, 255, 0)
            else:
                status_mark = "[PENDING]"
                title_color = (0, 255, 255)

            s_cls = sample.get("stratum_class", sample.get("class_name"))
            if self.mode == "second":
                title_txt = f"[{self.current_idx + 1}/{len(self.samples)}] {status_mark} Rank:{sample.get('rank', 0)} | Stratum:{s_cls.upper()} -> True:{true_class.upper()} [Blind Annotator: {self.annotator}]"
            else:
                q_count = self.count_accepted(sample["cctv_id"], sample["roi_idx"], true_class)
                title_txt = f"[{self.current_idx + 1}/{len(self.samples)}] {status_mark} Rank:{sample.get('rank', 0)} | Stratum:{s_cls.upper()} -> True:{true_class.upper()} (Quota:{q_count}/{PER_CLASS})"

        cv2.putText(frame, title_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 2)

        fps = sample["native_fps"]
        roi_info = f"{sample['cctv_id']} | ROI #{sample['roi_idx']+1} ({sample.get('perspective','depan')}) | {sample['roi_length_meters']}m x {sample['roi_width_meters']}m | FPS: {fps:.2f}"
        cv2.putText(frame, roi_info, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        status_play = "PLAYING" if is_playing else "PAUSED"
        frame_time = curr_frame_idx / fps
        frame_txt = f"Frame: {curr_frame_idx} ({frame_time:.3f}s) | Range: [{self.cache_min_frame}..{self.cache_max_frame}] | Status: {status_play}"
        cv2.putText(frame, frame_txt, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        delta_frames, gt_spd = gt_speed(entry_frame, exit_frame, sample)
        pred_spd = sample["predicted_speed_kmh"]

        entry_str = f"{entry_frame}" if entry_frame is not None else "[1: In]"
        exit_str = f"{exit_frame}" if exit_frame is not None else "[2: Out]"
        calc_txt = f"Entry Frame [1]: {entry_str} | Exit Frame [2]: {exit_str}"
        if delta_frames is not None:
            calc_txt += f" | Delta: {delta_frames} frames ({delta_frames/fps:.3f}s)"
        elif entry_frame is not None and exit_frame is not None:
            calc_txt += " | [GT: frame outside track]"
        cv2.putText(frame, calc_txt, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Bottom panel
        if self.mode == "causes":
            cause_name = curr_cause if curr_cause else "[NONE - Press j/o/i/t/b/e]"
            color = (0, 255, 0) if curr_cause else (0, 0, 255)
            cause_txt = f"CAUSE: {cause_name.upper()} | Source: {sample.get('source', '')} | Pred: {pred_spd:.2f} km/h | GT: {gt_spd:.2f} km/h" if gt_spd else f"CAUSE: {cause_name.upper()} | Source: {sample.get('source', '')} | Pred: {pred_spd:.2f} km/h"
            cv2.putText(frame, cause_txt, (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            ctrls = "[A/D]: -1/+1 | [W/S]: -10/+10 | [Space]: Play | [j/o/i/t/b/e]: Cause | [N/Enter]: Save & Next | [P]: Prev | [ESC]: Quit"
        elif self.mode == "second":
            cv2.putText(frame, "[BLIND VALIDATION MODE - SPEED METRICS HIDDEN]", (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            ctrls = "[A/D]: -1/+1 | [W/S]: -10/+10 | [Space]: Play | [1]: Entry | [2]: Exit | [5..8]: Class | [F]: FP | [X]: Invalid | [N/Enter]: Save | [P]: Prev | [ESC]: Quit"
        else:
            if gt_spd is not None:
                diff = pred_spd - gt_spd
                mape = (abs(diff) / gt_spd) * 100
                color = (0, 255, 0) if abs(diff) <= 5.0 else (0, 165, 255)
                spd_txt = f"GT: {gt_spd:.2f} km/h | Pred (EMA): {pred_spd:.2f} km/h | Diff: {diff:+.2f} km/h (MAPE: {mape:.1f}%)"
                cv2.putText(frame, spd_txt, (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            else:
                spd_txt = f"Pred (EMA): {pred_spd:.2f} km/h | GT: -- (Set 1: Entry & 2: Exit)"
                cv2.putText(frame, spd_txt, (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            if false_positive:
                cv2.putText(frame, "!! FALSE POSITIVE -> ignored in speed error metrics", (15, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif invalid_crossing:
                cv2.putText(frame, "!! INVALID CROSSING (stopped / occluded / ID switch) -> ignored in metrics", (15, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            ctrls = "[A/D]: -1/+1 | [W/S]: -10/+10 | [Space]: Play | [1]: Entry | [2]: Exit | [5..8]: Class | [F]: FP | [X]: Invalid | [N/Enter]: Save | [P]: Prev | [ESC]: Quit"

        cv2.putText(frame, ctrls, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        if self.mode == "causes":
            self.draw_cause_cheatsheet(frame, curr_cause)

    def draw_cause_cheatsheet(self, frame, curr_cause):
        """Key legend (top right, below the header); the currently selected cause is highlighted."""
        h, w = frame.shape[:2]
        pad, line_h = 10, 22
        rows = [(chr(k), name, label, desc) for k, name, label, desc in CAUSE_HELP]
        box_w = 470
        box_h = pad * 2 + line_h * (len(rows) + 1)
        x0, y0 = w - box_w - 10, 125
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "FAILURE CAUSE KEYS", (x0 + pad, y0 + pad + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for i, (key, name, label, desc) in enumerate(rows, start=1):
            sel = curr_cause == name
            y = y0 + pad + 14 + i * line_h
            color = (0, 255, 0) if sel else (230, 230, 230)
            cv2.putText(frame, f"[{key}] {label}", (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if sel else 1)
            cv2.putText(frame, desc, (x0 + pad + 170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    def draw_annotations_on_frame(self, frame, sample, curr_frame_idx, false_positive, invalid_crossing):
        # Draw ROI polygon
        roi_pts = np.array(sample["roi_points"], dtype=np.int32)
        cv2.polylines(frame, [roi_pts], True, (0, 255, 255), 2)

        # Highlight entry edge (pts 0->1) in GREEN and exit edge (pts 2->3) in MAGENTA
        if len(roi_pts) >= 4:
            p0, p1 = tuple(roi_pts[0]), tuple(roi_pts[1])
            p2, p3 = tuple(roi_pts[2]), tuple(roi_pts[3])
            cv2.line(frame, p0, p1, (0, 255, 0), 3)       # Entry edge
            cv2.line(frame, p2, p3, (255, 0, 255), 3)     # Exit edge
            cv2.putText(frame, "ENTRY", (p0[0] + 5, p0[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, "EXIT", (p2[0] + 5, p2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # Draw trajectory observation at current frame
        traj = sample.get("trajectory", [])
        curr_obs = next((o for o in traj if o["frame"] == curr_frame_idx), None)
        if curr_obs is not None:
            bbox = curr_obs["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = (0, 0, 255) if false_positive else ((0, 165, 255) if invalid_crossing else (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            bc_x, bc_y = int(curr_obs["bottom_center"][0]), int(curr_obs["bottom_center"][1])
            cv2.circle(frame, (bc_x, bc_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (bc_x, bc_y), 7, (255, 255, 255), 1)

            lbl = f"ID:{sample['track_id']} ({sample.get('stratum_class', 'veh')})"
            cv2.putText(frame, lbl, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def save_annotation(self, sample, entry_frame, exit_frame, true_class, false_positive, invalid_crossing):
        sample_id = self.get_sample_key(sample)
        fps = sample["native_fps"]
        delta_frames, gt_spd = gt_speed(entry_frame, exit_frame, sample)
        pred_spd = sample["predicted_speed_kmh"]

        if false_positive or invalid_crossing or gt_spd is None:
            err_kmh = None
            mape = None
            gt_out = None
            delta_out = None
        else:
            err_kmh = round(pred_spd - gt_spd, 2)
            mape = round((abs(pred_spd - gt_spd) / gt_spd) * 100, 2)
            gt_out = gt_spd
            delta_out = delta_frames

        self.annotations[sample_id] = {
            "sample_id": sample_id,
            "annotator": self.annotator,
            "rank": sample.get("rank", 0),
            "date": sample.get("date"),
            "cctv_id": sample["cctv_id"],
            "roi_idx": sample["roi_idx"],
            "perspective": sample.get("perspective", "depan"),
            "stratum_class": sample.get("stratum_class", sample.get("class_name")),
            "replay_class": sample.get("replay_class"),
            "true_class": true_class,
            "track_id": sample["track_id"],
            "release_track_id": sample.get("release_track_id"),
            "video_name": sample["video_name"],
            "video_path": sample["video_path"],
            "native_fps": fps,
            "roi_length_meters": sample["roi_length_meters"],
            "roi_width_meters": sample["roi_width_meters"],
            "entry_frame": entry_frame,
            "exit_frame": exit_frame,
            "false_positive": bool(false_positive),
            "invalid_crossing": bool(invalid_crossing),
            "delta_frames": delta_out,
            "ground_truth_speed_kmh": gt_out,
            "predicted_speed_kmh": pred_spd,
            "error_kmh": err_kmh,
            "mape_percent": mape,
        }

        # Save JSON
        with open(self.annotations_json, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2)

        # Export CSV
        df = pd.DataFrame(list(self.annotations.values()))
        df.to_csv(self.report_csv, index=False)
        print(f"[✓] Annotation saved for {sample_id} by {self.annotator} | GT: {gt_out} km/h | Pred: {pred_spd} km/h")

    def save_cause(self, sample, cause: str):
        sample_id = self.get_sample_key(sample)
        self.causes[sample_id] = {
            "cause": cause,
            "source": sample.get("source", "unknown"),
            "annotator": self.annotator,
        }
        with open(self.causes_json, "w", encoding="utf-8") as f:
            json.dump(self.causes, f, indent=2)
        print(f"[✓] Cause saved for {sample_id} by {self.annotator}: {cause} (source: {sample.get('source')})")
    def inspect_sample(self, sample_idx: int) -> str:
        sample = self.samples[sample_idx]
        sample_id = self.get_sample_key(sample)

        # Auto-skip if unannotated and stratum is already full (only in annotate mode)
        if self.mode == "annotate" and sample_id not in self.annotations and self.is_stratum_full(sample):
            print(f"[*] Skipping {sample_id} (stratum {sample.get('stratum_class')} quota already reached).")
            return "next"

        if not self.load_frame_cache(sample):
            return "next"

        curr_cause = None
        if self.mode == "second":
            saved = self.annotations.get(sample_id, {})
            entry_frame = saved.get("entry_frame")
            exit_frame = saved.get("exit_frame")
            true_class = saved.get("true_class", sample.get("stratum_class", sample.get("class_name", "car")))
            false_positive = bool(saved.get("false_positive", False))
            invalid_crossing = bool(saved.get("invalid_crossing", False))
        elif self.mode == "causes":
            saved_ref = self.ref_annotations.get(sample_id, {})
            entry_frame = saved_ref.get("entry_frame")
            exit_frame = saved_ref.get("exit_frame")
            true_class = saved_ref.get("true_class", sample.get("stratum_class", "car"))
            false_positive = bool(saved_ref.get("false_positive", False))
            invalid_crossing = bool(saved_ref.get("invalid_crossing", False))
            curr_cause = self.causes.get(sample_id, {}).get("cause")
        else:
            saved = self.annotations.get(sample_id, {})
            entry_frame = saved.get("entry_frame", sample.get("est_entry_frame"))
            exit_frame = saved.get("exit_frame", sample.get("est_exit_frame"))
            true_class = saved.get("true_class", sample.get("stratum_class", sample.get("class_name", "car")))
            false_positive = bool(saved.get("false_positive", False))
            invalid_crossing = bool(saved.get("invalid_crossing", False))
        curr_frame = max(self.cache_min_frame, sample.get("first_frame", self.cache_min_frame))
        is_playing = False
        stick_rate = 0

        while True:
            display_frame = self.get_frame(curr_frame)
            if display_frame is None:
                curr_frame = self.cache_min_frame
                display_frame = self.get_frame(curr_frame)
                if display_frame is None:
                    break

            self.draw_annotations_on_frame(display_frame, sample, curr_frame, false_positive, invalid_crossing)
            self.draw_hud(display_frame, sample, curr_frame, is_playing, entry_frame, exit_frame,
                          true_class, false_positive, invalid_crossing, curr_cause=curr_cause, stick_rate=stick_rate)

            cv2.imshow(self.window_name, display_frame)

            wait_ms = 33 if is_playing else 1
            key = cv2.waitKey(wait_ms) & 0xFF

            if is_playing:
                curr_frame += 1
                if curr_frame > self.cache_max_frame:
                    curr_frame = self.cache_min_frame

            cmd = None
            stick_rate = 0
            if key in (27, ord("q")):
                cmd = "quit"
            elif key == ord(" "):
                cmd = "playpause"
            elif key in (13, ord("n")):
                cmd = "save_next"
            elif key in (ord("d"), 83):
                cmd = "next_frame"
            elif key in (ord("a"), 81):
                cmd = "prev_frame"
            elif key in (ord("w"), 82):
                cmd = "next10"
            elif key in (ord("s"), 84):
                cmd = "prev10"
            elif key == ord("1"):
                cmd = "entry"
            elif key == ord("2"):
                cmd = "exit"
            elif key == ord("c"):
                cmd = "clear"
            elif key == ord("f"):
                cmd = "toggle_fp"
            elif key == ord("x"):
                cmd = "toggle_invalid"
            elif key == ord("p"):
                cmd = "prev_object"
            elif key in CLASS_KEYS:
                true_class = CLASS_KEYS[key]
                print(f"[+] True class set to: {true_class}")
            elif self.mode == "causes" and key in CAUSE_KEYS:
                curr_cause = CAUSE_KEYS[key]
                print(f"[+] Cause selected: {curr_cause}")
            else:
                gp_cmd = self._poll_gamepad()
                if gp_cmd:
                    if isinstance(gp_cmd, tuple) and gp_cmd[0] == "scrub":
                        _, n_steps, direction = gp_cmd
                        is_playing = False
                        if direction > 0:
                            curr_frame = min(self.cache_max_frame, curr_frame + n_steps)
                        else:
                            curr_frame = max(self.cache_min_frame, curr_frame - n_steps)
                        stick_rate = direction * n_steps
                        cmd = None
                    else:
                        cmd = gp_cmd
            if cmd == "quit":
                return "quit"
            elif cmd == "playpause":
                is_playing = not is_playing
            elif cmd == "save_next":
                if self.mode == "causes":
                    if curr_cause is None:
                        print("[!] Please select a cause (j/o/i/t/b/e) before advancing.")
                        continue
                    self.save_cause(sample, curr_cause)
                    return "next"
                else:
                    self.save_annotation(sample, entry_frame, exit_frame, true_class, false_positive, invalid_crossing)
                    return "next"
            elif cmd == "next_frame":
                is_playing = False
                curr_frame = min(self.cache_max_frame, curr_frame + 1)
            elif cmd == "prev_frame":
                is_playing = False
                curr_frame = max(self.cache_min_frame, curr_frame - 1)
            elif cmd == "next10":
                is_playing = False
                curr_frame = min(self.cache_max_frame, curr_frame + 10)
            elif cmd == "prev10":
                is_playing = False
                curr_frame = max(self.cache_min_frame, curr_frame - 10)
            elif cmd == "entry":
                entry_frame = curr_frame
                print(f"[+] Entry frame set to: {entry_frame}")
            elif cmd == "exit":
                exit_frame = curr_frame
                print(f"[+] Exit frame set to: {exit_frame}")
            elif cmd == "clear":
                entry_frame = None
                exit_frame = None
                print("[+] Cleared entry and exit frames.")
            elif cmd == "toggle_fp":
                false_positive = not false_positive
                print(f"[!] False positive: {false_positive}")
            elif cmd == "toggle_invalid":
                invalid_crossing = not invalid_crossing
                print(f"[!] Invalid crossing: {invalid_crossing}")
            elif cmd == "next_object":
                if self.mode == "causes":
                    if curr_cause is None:
                        print("[!] Please select a cause (j/o/i/t/b/e) before advancing.")
                        continue
                    self.save_cause(sample, curr_cause)
                    return "next"
                else:
                    self.save_annotation(sample, entry_frame, exit_frame, true_class, false_positive, invalid_crossing)
                    return "next"
            elif cmd == "prev_object":
                if self.mode == "causes":
                    if curr_cause is not None:
                        self.save_cause(sample, curr_cause)
                    return "prev"
                else:
                    self.save_annotation(sample, entry_frame, exit_frame, true_class, false_positive, invalid_crossing)
                    return "prev"

        return "next"

    def next_actionable_idx(self, start_idx: int) -> int:
        idx = start_idx
        while idx < len(self.samples):
            s = self.samples[idx]
            sid = self.get_sample_key(s)
            if self.mode == "annotate":
                if sid not in self.annotations and not self.is_stratum_full(s):
                    return idx
            elif self.mode == "second":
                if sid not in self.annotations:
                    return idx
            elif self.mode == "causes":
                if sid not in self.causes:
                    return idx
            idx += 1
        return idx

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        # On launch, skip all already-annotated and quota-complete entries
        self.current_idx = self.next_actionable_idx(0)
        if self.current_idx >= len(self.samples):
            if self.mode == "causes":
                print("[✓] All failure causes have been categorized! Output: OUT/causes.json")
            elif self.mode == "second":
                print("[✓] All 48 second-annotator samples annotated! Output: OUT/annotations_second.json")
            else:
                print("[✓] All strata quotas have already been met! Run 'python tools/speed_val.py report' to view results.")
            return

        print(f"[*] Resuming inspection at sample [{self.current_idx + 1}/{len(self.samples)}]: {self.get_sample_key(self.samples[self.current_idx])}")

        while 0 <= self.current_idx < len(self.samples):
            action = self.inspect_sample(self.current_idx)
            if action == "next":
                self.current_idx = self.next_actionable_idx(self.current_idx + 1)
            elif action == "prev":
                self.current_idx = max(0, self.current_idx - 1)
            elif action == "quit":
                break
        cv2.destroyAllWindows()
        print("\n==================================================")
        print("Inspection completed / paused by user.")
        if self.mode == "causes":
            print(f"Causes JSON: {self.causes_json}")
        else:
            print(f"Annotations: {self.annotations_json}")
            print(f"Report CSV : {self.report_csv}")
        print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Speed Validation Video Inspector")
    parser.add_argument("--mode", choices=["annotate", "second", "causes"], default="annotate",
                        help="Mode: annotate (default), second (blind repeat), or causes (failure causes)")
    parser.add_argument("--annotator", default="A1", help="Annotator identifier (default: A1)")
    parser.add_argument("--candidates", default=str(SAMPLED_JSON), help="Path to candidates.json")
    args = parser.parse_args()

    inspector = SpeedAccuracyInspector(Path(args.candidates), mode=args.mode, annotator=args.annotator)
    inspector.run()


if __name__ == "__main__":
    main()

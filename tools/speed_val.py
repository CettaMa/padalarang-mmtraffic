"""Speed Validation Toolkit for JOIV Revision.

Subcommands:
  pool        Scan local dates & release track logs to build candidate pool (pool.csv)
  fetch       Download and extract specific segments for scarce strata from Google Drive
  prepare     Replay release detector/tracker on candidate clips to produce candidates.json
  report      Compute evaluation metrics from annotations.json (MAE, RMSE, bias, MAPE, r)
  selfcheck   Verify replay fidelity against released window track log and features CSV
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import copy
import csv
import gzip
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "script"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch

from feature_extractor import (
    FALLBACK_FPS,
    INFER_IMGSZ,
    PROC_WIDTH,
    TARGET_CLASSES,
    TARGET_FPS,
    Detector,
    WindowLog,
    _letterbox_pad,
    _parse_hhmmss,
    _proc_size,
    build_segment_index,
    get_perspective_matrices,
    process_window,
    select_window_segments,
)
from traffic_metrics import traverse_speed_kmh

OUT = REPO_ROOT / "labeling" / "speed_val"
SEED = 20261029
BOOT_B = 10000
PER_CLASS = 5
FILL_FACTOR = 2
SEG_SEC = 28.0
HOME_PRE_SEC = 1.0
HOME_POST_SEC = 1.0
CLASSES = ("bus", "car", "motorcycle", "truck")
CLASS_TO_ID = {name: cid for cid, name in TARGET_CLASSES.items()}
MOUNT = "/tmp/sv_gdrive"
GDRIVE_SRC = "gdrive:CCTV-Archives"
# Tagog ROI 1 width was 7.0 m during the release run (track x reaches 6.93 m);
# JSON was subsequently adjusted to 6.25 m.
AS_RUN_WIDTH = {("padalarang_-_giant_-_tagog", 1): 7.0}


def load_rois(cctv_id: str) -> list[dict]:
    """Load ROI definitions with as-run geometry overrides."""
    roi_file = REPO_ROOT / "rois" / f"{cctv_id}_roi.json"
    if not roi_file.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_file}")
    with open(roi_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    rois = copy.deepcopy(data["rois"])
    for idx, roi in enumerate(rois):
        override_w = AS_RUN_WIDTH.get((cctv_id, idx))
        if override_w is not None:
            roi["width_meters"] = override_w
    return rois


def video_dir(date: str, cctv_id: str) -> Path | None:
    """Find video directory for a camera date, handling both flat and nested archive layouts."""
    dataset_root = REPO_ROOT / "dataset"
    flat = dataset_root / date / cctv_id / "videos"
    if flat.exists():
        return flat
    nested = dataset_root / date / date / cctv_id / "videos"
    if nested.exists():
        return nested
    return None


def segment_names(date: str, cctv_id: str) -> tuple[list[str], list[float]]:
    """Return sorted (names, start_secs) for a camera on a date.

    Prefers OUT/segment_lists/<date>__<cctv_id>.txt if it exists (for partially fetched dates),
    otherwise reads video_dir. Excludes files whose 8-digit date token != date.
    """
    list_file = OUT / "segment_lists" / f"{date}__{cctv_id}.txt"
    raw_names = []
    if list_file.exists():
        with open(list_file, "r", encoding="utf-8") as f:
            raw_names = [line.strip() for line in f if line.strip()]
    else:
        vdir = video_dir(date, cctv_id)
        if vdir is not None and vdir.exists():
            raw_names = [p.name for p in vdir.glob("*.ts")]

    clean_date = date.replace("-", "")
    items = []
    for name in raw_names:
        stem = Path(name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            date_token = parts[-2]
            if date_token != clean_date:
                continue
        sec = _parse_hhmmss(parts[-1])
        if sec is not None:
            items.append((name, float(sec)))
    items.sort(key=lambda x: x[1])
    return [x[0] for x in items], [x[1] for x in items]


def home_segment(starts: list[float], names: list[str], t_first: float, t_last: float) -> str | None:
    """Return home segment name if vehicle's in-ROI traversal fits strictly inside one segment.

    Requires:
      - segment start <= t_first - HOME_PRE_SEC
      - t_last + HOME_POST_SEC < min(segment_start + SEG_SEC, next_segment_start)
    """
    if not starts or not names or len(starts) != len(names):
        return None
    cand_idx = bisect.bisect_right(starts, t_first - HOME_PRE_SEC) - 1
    if cand_idx < 0:
        return None
    end_limit = starts[cand_idx] + SEG_SEC
    if cand_idx + 1 < len(starts):
        end_limit = min(end_limit, starts[cand_idx + 1])
    if t_last + HOME_POST_SEC >= end_limit:
        return None
    return names[cand_idx]


def gt_speed(entry_frame: int | None, exit_frame: int | None, sample: dict) -> tuple[int | None, float | None]:
    """Calculate semi-automatic GT speed via linearly interpolated bottom-center positions.

    Returns (delta_frames, speed_kmh) or (None, None) if inputs are invalid or outside track.
    """
    if entry_frame is None or exit_frame is None:
        return None, None
    delta_frames = exit_frame - entry_frame
    if delta_frames <= 0:
        return None, None

    fps = float(sample.get("native_fps", FALLBACK_FPS))
    if fps <= 0:
        return None, None
    dt = delta_frames / fps
    if dt <= 0:
        return None, None

    traj = sample.get("trajectory", [])
    if not traj:
        return None, None

    frames = [o["frame"] for o in traj]
    if entry_frame < frames[0] or entry_frame > frames[-1]:
        return None, None
    if exit_frame < frames[0] or exit_frame > frames[-1]:
        return None, None

    def _interp_pos(target_frame: int) -> np.ndarray:
        for i in range(len(traj)):
            if traj[i]["frame"] == target_frame:
                return np.array(traj[i]["bottom_center"], dtype=np.float32)
            if traj[i]["frame"] > target_frame:
                prev_o = traj[i - 1]
                next_o = traj[i]
                frac = (target_frame - prev_o["frame"]) / float(next_o["frame"] - prev_o["frame"])
                p0 = np.array(prev_o["bottom_center"], dtype=np.float32)
                p1 = np.array(next_o["bottom_center"], dtype=np.float32)
                return p0 + frac * (p1 - p0)
        return np.array(traj[-1]["bottom_center"], dtype=np.float32)

    bc_in = _interp_pos(entry_frame)
    bc_out = _interp_pos(exit_frame)

    pts = np.array(sample["roi_points"][:4], dtype=np.float32)
    w = float(sample["roi_width_meters"])
    l = float(sample["roi_length_meters"])
    dst = np.array([[0, 0], [w, 0], [w, l], [0, l]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts, dst)

    def _to_m(px: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(px.reshape(1, 1, 2), H)[0, 0]

    m_in = _to_m(bc_in)
    m_out = _to_m(bc_out)
    dist_m = float(np.linalg.norm(m_out - m_in))

    speed = (dist_m / dt) * 3.6
    if not math.isfinite(speed) or speed <= 0:
        return None, None
    return delta_frames, round(float(speed), 2)
def release_row_speed(r: dict, lo_sec: float, hi_sec: float) -> float | None:
    """Return finite traverse speed in km/h if row passes release filter criteria, else None.

    Filters:
      - finite traverse_speed_kmh >= 5.0 km/h
      - t_first >= 120.0 (discard first 2 min of day tracker warmup)
      - t_first - lo_sec >= 20.0 (cooldown after window start)
      - hi_sec - t_last >= 5.0 (margin before window end)
    """
    t_first = float(r["t_first"])
    t_last = float(r["t_last"])
    if t_first < 120.0:
        return None
    if (t_first - lo_sec) < 20.0 or (hi_sec - t_last) < 5.0:
        return None
    spd = traverse_speed_kmh(r)
    if not math.isfinite(spd) or spd < 5.0:
        return None
    return float(spd)


def date_complete(date: str, cctv_id: str) -> bool:
    """Return True if the date/cctv is fully extracted, False if partial."""
    d_dir = REPO_ROOT / "dataset" / date
    if (d_dir / ".unzip_ok").exists():
        return True
    if (d_dir / date / ".unzip_ok").exists():
        return True
    vdir = video_dir(date, cctv_id)
    if vdir is not None and vdir.exists():
        count = 0
        for _ in vdir.glob("*.ts"):
            count += 1
            if count >= 2000:
                return True
    return False

def to_wsl(p: Path) -> str:
    resolved = p.resolve()
    drive = resolved.drive.replace(":", "").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def wsl(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl.exe", "-e", "bash", "-lc", cmd],
        capture_output=True,
        text=True,
        check=check,
    )


@contextlib.contextmanager
def gdrive_mount():
    cmd = (
        f"mkdir -p {MOUNT} && "
        f"(mountpoint -q {MOUNT} || rclone mount {GDRIVE_SRC} {MOUNT} --vfs-cache-mode minimal --read-only --daemon)"
    )
    wsl(cmd, check=True)
    ready = False
    for _ in range(30):
        res = wsl(f"ls {MOUNT}", check=False)
        if res.returncode == 0 and res.stdout.strip():
            ready = True
            break
        time.sleep(1)
    if not ready:
        raise RuntimeError("rclone mount not ready")
    try:
        yield
    finally:
        wsl(f"fusermount3 -u {MOUNT}", check=False)


def archive_listing(date: str) -> dict[str, list[str]]:
    res = wsl(f"7z l -slt {MOUNT}/{date}.7z", check=False)
    if res.returncode != 0:
        raise RuntimeError(f"archive {date}.7z not listable: {res.stderr}")
    blocks = res.stdout.replace("\r\n", "\n").split("\n\n")
    listing: dict[str, list[str]] = {}
    for b in blocks:
        lines = b.strip().splitlines()
        d = {}
        for line in lines:
            if " = " in line:
                k, v = line.split(" = ", 1)
                d[k.strip()] = v.strip()
        p = d.get("Path", "")
        if p.endswith(".ts") and "/videos/" in p:
            sz = int(d.get("Size", 0))
            if sz > 0:
                parts = p.split("/")
                if len(parts) >= 4 and parts[0] == date and parts[2] == "videos":
                    cctv = parts[1]
                    listing.setdefault(cctv, []).append(p)
    for cctv in listing:
        listing[cctv].sort()
    return listing


def write_segment_lists(date: str, listing: dict[str, list[str]]) -> None:
    seg_list_dir = OUT / "segment_lists"
    seg_list_dir.mkdir(parents=True, exist_ok=True)
    for cctv, paths in listing.items():
        list_file = seg_list_dir / f"{date}__{cctv}.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(f"{Path(p).name}\n")


def extract_segments(date: str, cctv: str, internal_paths: list[str]) -> int:
    dest = REPO_ROOT / "dataset" / date / cctv / "videos"
    dest.mkdir(parents=True, exist_ok=True)
    existing = set(p.name for p in dest.glob("*.ts"))
    needed = [p for p in internal_paths if Path(p).name not in existing]
    if not needed:
        return 0

    listfile = OUT / f"_extract_{date}__{cctv}.txt"
    with open(listfile, "w", encoding="utf-8") as f:
        for p in needed:
            f.write(f"{p}\n")
    try:
        cmd = f"7z e {MOUNT}/{date}.7z -o'{to_wsl(dest)}' @'{to_wsl(listfile)}' -y -bso0 -bsp0"
        wsl(cmd, check=True)
    finally:
        if listfile.exists():
            listfile.unlink()
    return len(needed)


def fill_target(deficit: int, factor: int = FILL_FACTOR) -> int:
    """Return target number of pending (unannotated) candidates for a stratum.

    If deficit == 0, target is 0.
    Otherwise max(2, factor * deficit).
    """
    if deficit <= 0:
        return 0
    return max(2, factor * deficit)


def stratum_status(candidates: list[dict], annotations: dict) -> dict[tuple[str, int, str], dict]:
    """Compute (accepted, pending, deficit) for all strata across the 6 cameras."""
    cctv_rois = {}
    for roi_file in sorted((REPO_ROOT / "rois").glob("*_roi.json")):
        cctv_id = roi_file.stem.replace("_roi", "")
        cctv_rois[cctv_id] = load_rois(cctv_id)

    accepted_counts: dict[tuple[str, int, str], int] = {}
    for a in annotations.values():
        if a.get("false_positive") or a.get("invalid_crossing"):
            continue
        gt = a.get("ground_truth_speed_kmh")
        if gt is None or not math.isfinite(float(gt)) or float(gt) <= 0:
            continue
        cctv = a.get("cctv_id")
        roi = int(a.get("roi_idx", -1))
        t_cls = a.get("true_class", a.get("class_name", "car"))
        key = (cctv, roi, t_cls)
        accepted_counts[key] = accepted_counts.get(key, 0) + 1

    anno_keys = set(annotations.keys())
    pending_counts: dict[tuple[str, int, str], int] = {}
    for c in candidates:
        if c.get("sample_id") in anno_keys:
            continue
        cctv = c.get("cctv_id")
        roi = int(c.get("roi_idx", -1))
        s_cls = c.get("stratum_class", c.get("class_name", "car"))
        key = (cctv, roi, s_cls)
        pending_counts[key] = pending_counts.get(key, 0) + 1

    status: dict[tuple[str, int, str], dict] = {}
    for cctv_id, rois in sorted(cctv_rois.items()):
        for r_idx in range(len(rois)):
            for cls in CLASSES:
                k = (cctv_id, r_idx, cls)
                acc = accepted_counts.get(k, 0)
                pend = pending_counts.get(k, 0)
                defic = max(0, PER_CLASS - acc)
                status[k] = {
                    "accepted": acc,
                    "pending": pend,
                    "deficit": defic,
                }
    return status


def cmd_index(args: argparse.Namespace) -> int:
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if not dates:
        print("[!] No dates specified for indexing.")
        return 1
    print(f"[*] Indexing archives for dates: {dates}...")
    with gdrive_mount():
        for d in dates:
            print(f"[*] Listing {d}.7z...")
            listing = archive_listing(d)
            write_segment_lists(d, listing)
            summary = ", ".join(f"{cctv}={len(p)}" for cctv, p in sorted(listing.items()))
            print(f"[✓] {d}: {summary}")
    return 0





class LocalClient:
    """In-process replacement for InferClient for single-process replay without shared memory."""

    def __init__(self, detector: Detector):
        self.detector = detector
        self._lock = threading.Lock()
        self._next_slot = 0
        self._slots: dict[int, np.ndarray] = {}
        self._next_req_id = 0
        self._requests: dict[int, list[int]] = {}

    def acquire_slot(self) -> int:
        with self._lock:
            slot = self._next_slot
            self._next_slot += 1
            return slot

    def write_frame(self, slot: int, frame: np.ndarray) -> None:
        with self._lock:
            self._slots[slot] = frame.copy()

    def submit(self, height: int, width: int, slots: list[int]) -> int:
        with self._lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            self._requests[req_id] = list(slots)
            return req_id

    def fetch(self, req_id: int):
        with self._lock:
            slots = self._requests.pop(req_id)
            frames = [self._slots.pop(s) for s in slots]
        return self.detector.detect(frames)

    def close(self) -> None:
        with self._lock:
            self._slots.clear()
            self._requests.clear()


# ==============================================================================
# SUBCOMMAND: pool
# ==============================================================================

def cmd_pool(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_root = REPO_ROOT / "dataset"
    tracks_root = REPO_ROOT / "extracted" / "tracks"

    # Load ROI definitions with as-run geometry overrides
    cctv_rois = {}
    for roi_file in (REPO_ROOT / "rois").glob("*_roi.json"):
        cctv_id = roi_file.stem.replace("_roi", "")
        cctv_rois[cctv_id] = load_rois(cctv_id)

    # Discover candidate dates: dates in dataset/ with at least one camera having video_dir
    local_dates = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-" and p.name[7] == "-":
            if any(video_dir(p.name, c) is not None for c in cctv_rois):
                local_dates.append(p.name)
    print(f"[*] Local dates with video content: {local_dates}")
    # Cache segment names per (date, cctv)
    seg_cache: dict[tuple[str, str], tuple[list[str], list[float]]] = {}

    rows = []
    for date in local_dates:
        date_tracks = tracks_root / date
        if not date_tracks.exists():
            continue
        for cctv_dir in sorted(date_tracks.iterdir()):
            if not cctv_dir.is_dir():
                continue
            cctv_id = cctv_dir.name
            if cctv_id not in cctv_rois:
                continue

            vdir = video_dir(date, cctv_id)
            if vdir is None:
                continue

            complete = date_complete(date, cctv_id)
            list_file = OUT / "segment_lists" / f"{date}__{cctv_id}.txt"
            if not complete and not list_file.exists():
                print(f"[!] Skipping partial {date}/{cctv_id}: no segment list (run: speed_val.py index --dates {date})")
                continue

            key = (date, cctv_id)
            if key not in seg_cache:
                seg_cache[key] = segment_names(date, cctv_id)
            names, starts = seg_cache[key]
            if not names:
                continue

            local_ts_files = set(p.name for p in vdir.glob("*.ts"))
            all_segs = [{"path": Path(n), "start_sec": s} for n, s in zip(names, starts)]
            is_partial = not complete
            for meta_path in sorted(cctv_dir.glob("*_meta.json")):
                ts_str = meta_path.stem.replace("_meta", "")
                veh_path = cctv_dir / f"{ts_str}_veh.csv.gz"
                if not veh_path.exists():
                    continue

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                lo_sec = float(meta["lo_sec"])
                hi_sec = float(meta["hi_sec"])

                try:
                    df = pd.read_csv(veh_path)
                except Exception as e:
                    print(f"[!] Gagal membaca {veh_path}: {e}")
                    continue

                for r in df.to_dict("records"):
                    roi_idx = int(r["roi_idx"])
                    cid = int(r["class_id"])
                    if cid not in TARGET_CLASSES:
                        continue
                    class_name = TARGET_CLASSES[cid]
                    t_first = float(r["t_first"])
                    t_last = float(r["t_last"])

                    # 1-3. Traverse speed >= 5 km/h, t_first >= 120, window cooldown/margin
                    spd = release_row_speed(r, lo_sec, hi_sec)
                    if spd is None:
                        continue

                    # 4. Fits in home segment
                    home = home_segment(starts, names, t_first, t_last)
                    if home is None:
                        continue

                    # 5. Every segment in [t_first-20, t_last+5] must exist locally
                    sel_segments = select_window_segments(all_segs, starts, t_first - 20.0, t_last + 5.0)
                    if not sel_segments:
                        continue
                    if is_partial and not all(s["path"].name in local_ts_files for s in sel_segments):
                        continue

                    # Geometry guard check against as-run dimensions
                    rois = cctv_rois[cctv_id]
                    if roi_idx >= len(rois):
                        continue
                    w_max = rois[roi_idx]["width_meters"] + 0.1
                    l_max = rois[roi_idx]["length_meters"] + 0.1
                    xf, yf, xl, yl = float(r["x_first_m"]), float(r["y_first_m"]), float(r["x_last_m"]), float(r["y_last_m"])
                    if max(xf, xl) > w_max or max(yf, yl) > l_max:
                        print(f"[!] FATAL: ROI extent guard failed for {cctv_id} ROI {roi_idx}: x=({xf:.2f},{xl:.2f}) > {w_max:.2f} or y=({yf:.2f},{yl:.2f}) > {l_max:.2f}")
                        return 2

                    sample_id = f"{date}|{cctv_id}|{roi_idx}|{ts_str}|{int(r['track_id'])}"
                    rows.append({
                        "sample_id": sample_id,
                        "date": date,
                        "cctv_id": cctv_id,
                        "roi_idx": roi_idx,
                        "perspective": rois[roi_idx].get("perspective", "depan"),
                        "class_name": class_name,
                        "track_id": int(r["track_id"]),
                        "ts": ts_str,
                        "t_first": t_first,
                        "t_last": t_last,
                        "n_obs": int(r["n_obs"]),
                        "home_segment": home,
                        "traverse_speed_kmh": round(spd, 2),
                    })

    if not rows:
        print("[!] No candidate tracks found.")
        return 1

    pool_df = pd.DataFrame(rows)

    # Assign rank per stratum (cctv_id, roi_idx, class_name)
    pool_df = pool_df.sort_values(["cctv_id", "roi_idx", "class_name", "date", "ts", "track_id"])
    ranked_chunks = []
    for (cctv, roi, cls), group in pool_df.groupby(["cctv_id", "roi_idx", "class_name"]):
        group = group.copy()
        # Seeded deterministic shuffle
        rng = random.Random(f"{SEED}|{cctv}|{roi}|{cls}")
        order = list(range(len(group)))
        rng.shuffle(order)
        group["rank"] = order
        group = group.sort_values("rank")
        ranked_chunks.append(group)

    pool_df = pd.concat(ranked_chunks, ignore_index=True)
    out_csv = OUT / "pool.csv"
    pool_df.to_csv(out_csv, index=False)
    print(f"[✓] Candidate pool saved to {out_csv} ({len(pool_df)} total candidates)")

    # Print stratum count table
    print("\nCandidate Pool Counts by Stratum (target >= 8):")
    summary = pool_df.pivot_table(
        index=["cctv_id", "roi_idx"],
        columns="class_name",
        values="sample_id",
        aggfunc="count",
        fill_value=0,
    )
    # Ensure all target classes present in display
    for c in CLASSES:
        if c not in summary.columns:
            summary[c] = 0
    summary = summary[list(CLASSES)]
    print(summary.to_string())
    return 0


# ==============================================================================
# SUBCOMMAND: fetch
# ==============================================================================

def cmd_fetch(args: argparse.Namespace) -> int:
    parts = args.stratum.split(":")
    if len(parts) != 3:
        print(f"[!] Invalid stratum spec '{args.stratum}'. Expected CCTV:ROI:CLASS (e.g. padalarang_-_giant_-_tagog:1:bus)")
        return 1
    cctv_id, roi_str, class_name = parts[0], parts[1], parts[2]
    roi_idx = int(roi_str)
    if class_name not in CLASS_TO_ID:
        print(f"[!] Unknown class '{class_name}'. Must be one of {CLASSES}")
        return 1
    cid = CLASS_TO_ID[class_name]

    OUT.mkdir(parents=True, exist_ok=True)
    seg_list_dir = OUT / "segment_lists"
    seg_list_dir.mkdir(parents=True, exist_ok=True)

    tracks_root = REPO_ROOT / "extracted" / "tracks"

    # Load existing candidates and annotations to determine need if not specified
    existing_candidates = []
    candidates_path = OUT / "candidates.json"
    if candidates_path.exists():
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                existing_candidates = json.load(f)
        except Exception:
            pass

    existing_annos = {}
    anno_path = OUT / "annotations.json"
    if anno_path.exists():
        try:
            with open(anno_path, "r", encoding="utf-8") as f:
                existing_annos = json.load(f)
        except Exception:
            pass

    status_map = stratum_status(existing_candidates, existing_annos)
    st = status_map.get((cctv_id, roi_idx, class_name), {"accepted": 0, "pending": 0, "deficit": PER_CLASS})
    deficit = st["deficit"]
    cur_pending = st["pending"]
    target_needed = fill_target(deficit)
    need = args.need if getattr(args, "need", None) is not None else max(1, target_needed - cur_pending)
    print(f"[*] Stratum {args.stratum}: accepted={st['accepted']}/{PER_CLASS}, pending={cur_pending}, target={target_needed}, need={need}")

    # Identify candidate remote dates with release tracks
    candidate_dates = []
    for d in sorted(tracks_root.iterdir()):
        if not d.is_dir() or len(d.name) != 10:
            continue
        date = d.name
        if date_complete(date, cctv_id):
            continue

        cctv_track_dir = d / cctv_id
        if not cctv_track_dir.exists():
            continue

        count = 0
        for meta_path in cctv_track_dir.glob("*_meta.json"):
            ts_str = meta_path.stem.replace("_meta", "")
            veh_path = cctv_track_dir / f"{ts_str}_veh.csv.gz"
            if not veh_path.exists():
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            lo, hi = float(meta["lo_sec"]), float(meta["hi_sec"])
            try:
                df = pd.read_csv(veh_path)
            except Exception:
                continue
            sub = df[(df.roi_idx == roi_idx) & (df.class_id == cid)]
            for r in sub.to_dict("records"):
                if release_row_speed(r, lo, hi) is not None:
                    count += 1
        if count > 0:
            candidate_dates.append((date, count))

    # Sort candidate dates by count descending, date ascending
    candidate_dates.sort(key=lambda x: (-x[1], x[0]))
    print(f"[*] Candidate remote dates with release tracks for {args.stratum}: {candidate_dates}")

    if not candidate_dates:
        print(f"[!] No remote dates have candidates for {args.stratum}.")
        return 0

    max_dates = getattr(args, "max_dates", 4)
    processed = 0
    fresh = 0

    with gdrive_mount():
        for date, count in candidate_dates[:max_dates]:
            print(f"\n=== Processing remote date: {date} ({count} potential candidates) ===")
            listing = archive_listing(date)
            write_segment_lists(date, listing)

            raw_names = [Path(p).name for p in listing.get(cctv_id, [])]
            clean_date = date.replace("-", "")
            items = []
            for name in raw_names:
                parts = Path(name).stem.split("_")
                if len(parts) >= 2 and parts[-2] != clean_date:
                    continue
                sec = _parse_hhmmss(parts[-1])
                if sec is not None:
                    items.append((name, float(sec)))
            items.sort(key=lambda x: x[1])
            names = [x[0] for x in items]
            starts = [x[1] for x in items]
            all_segs = [{"path": Path(n), "start_sec": s} for n, s in zip(names, starts)]

            needed_segments = set()
            cctv_track_dir = tracks_root / date / cctv_id
            for meta_path in cctv_track_dir.glob("*_meta.json"):
                ts_str = meta_path.stem.replace("_meta", "")
                veh_path = cctv_track_dir / f"{ts_str}_veh.csv.gz"
                if not veh_path.exists():
                    continue
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                lo, hi = float(meta["lo_sec"]), float(meta["hi_sec"])
                try:
                    df = pd.read_csv(veh_path)
                except Exception:
                    continue
                sub = df[(df.roi_idx == roi_idx) & (df.class_id == cid)]
                for r in sub.to_dict("records"):
                    if release_row_speed(r, lo, hi) is None:
                        continue
                    tf, tl = float(r["t_first"]), float(r["t_last"])
                    home = home_segment(starts, names, tf, tl)
                    if home is not None:
                        segs = select_window_segments(all_segs, starts, tf - 20.0, tl + 5.0)
                        for s in segs:
                            needed_segments.add(s["path"].name)

            if not needed_segments:
                print(f"[!] No segments matched candidate filters on {date}.")
                continue

            cctv_internal = listing.get(cctv_id, [])
            extract_targets = [p for p in cctv_internal if Path(p).name in needed_segments]
            n_ext = extract_segments(date, cctv_id, extract_targets)
            print(f"[✓] {date}: extracted {n_ext} clips for {args.stratum}")
            processed += 1

            # Re-run cmd_pool and check fresh candidates
            cmd_pool(args)
            pool_csv = OUT / "pool.csv"
            if pool_csv.exists():
                df_pool = pd.read_csv(pool_csv, dtype={"ts": str})
                cand_set = {c["sample_id"] for c in existing_candidates}
                log_set = set()
                if (OUT / "prepare_log.csv").exists():
                    try:
                        log_set = set(pd.read_csv(OUT / "prepare_log.csv")["sample_id"])
                    except Exception:
                        pass
                sub_pool = df_pool[(df_pool.cctv_id == cctv_id) & (df_pool.roi_idx == roi_idx) & (df_pool.class_name == class_name)]
                fresh = len([sid for sid in sub_pool["sample_id"] if sid not in cand_set and sid not in log_set])
            print(f"[*] Fresh pool count for {args.stratum}: {fresh} (target need: {need})")
            if fresh >= need:
                print(f"[✓] Reached target fresh pool ({fresh} >= {need})!")
                break

    if fresh < need:
        print(f"[!] {args.stratum}: only {fresh} fresh candidates after {processed} dates")
    return 0


# ==============================================================================
# SUBCOMMAND: prepare
# ==============================================================================

def cmd_prepare(args: argparse.Namespace) -> int:
    pool_csv = OUT / "pool.csv"
    if not pool_csv.exists():
        print("[!] pool.csv not found. Run 'python tools/speed_val.py pool' first.")
        return 1

    pool_df = pd.read_csv(pool_csv, dtype={"ts": str})
    factor = getattr(args, "factor", FILL_FACTOR)

    # Load existing candidates and log
    candidates_path = OUT / "candidates.json"
    existing_candidates = []
    existing_ids = set()
    if candidates_path.exists():
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                existing_candidates = json.load(f)
            existing_ids = {c["sample_id"] for c in existing_candidates}
        except Exception as e:
            print(f"[!] Warning reading candidates.json: {e}")

    log_path = OUT / "prepare_log.csv"
    logged_ids = set()
    if log_path.exists():
        try:
            df_log = pd.read_csv(log_path)
            logged_ids = set(df_log["sample_id"])
        except Exception:
            pass

    # Load existing annotations
    anno_path = OUT / "annotations.json"
    existing_annos = {}
    if anno_path.exists():
        try:
            with open(anno_path, "r", encoding="utf-8") as f:
                existing_annos = json.load(f)
        except Exception:
            pass

    status_map = stratum_status(existing_candidates, existing_annos)
    tracks_root = REPO_ROOT / "extracted" / "tracks"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading Detector on {device.upper()}...")
    detector = Detector(device)
    local_client = LocalClient(detector)
    new_candidates = []
    log_rows = []

    cctv_rois_cache = {}

    for (cctv_id, roi_idx, class_name), st in sorted(status_map.items()):
        deficit = st["deficit"]
        cur_pending = st["pending"]
        target_pending = fill_target(deficit, factor)
        if cur_pending >= target_pending:
            continue

        group = pool_df[(pool_df.cctv_id == cctv_id) & (pool_df.roi_idx == roi_idx) & (pool_df.class_name == class_name)]
        if group.empty:
            continue

        if cctv_id not in cctv_rois_cache:
            cctv_rois_cache[cctv_id] = load_rois(cctv_id)
        rois_data = cctv_rois_cache[cctv_id]
        matrices = get_perspective_matrices(rois_data)
        roi_polys = [np.array(r["points"], dtype=np.int32) for r in rois_data]
        roi_boxes = [(int(p[:, 0].min()), int(p[:, 1].min()),
                      int(p[:, 0].max()), int(p[:, 1].max())) for p in roi_polys]
        target_class_ids = list(TARGET_CLASSES.keys())

        # Sort group by rank
        group = group.sort_values("rank")

        for _, row in group.iterrows():
            if cur_pending >= target_pending:
                break
            sample_id = str(row["sample_id"])
            if sample_id in existing_ids or sample_id in logged_ids:
                continue

            date = str(row["date"])
            ts_str = str(row["ts"]).zfill(6)
            track_id = int(row["track_id"])
            t_first = float(row["t_first"])
            t_last = float(row["t_last"])
            home_seg = str(row["home_segment"])

            vdir = video_dir(date, cctv_id)
            if vdir is None:
                continue
            all_segments = build_segment_index(vdir)
            seg_start_secs = [s["start_sec"] for s in all_segments]

            # Window for replay: [t_first - 20, t_last + 5]
            w_lo = t_first - 20.0
            w_hi = t_last + 5.0
            window_segments = select_window_segments(all_segments, seg_start_secs, w_lo, w_hi)
            if not window_segments:
                log_rows.append({"sample_id": sample_id, "status": "no_segments"})
                logged_ids.add(sample_id)
                continue

            # Fresh tracker & counted_ids for this candidate window
            tracker = sv.ByteTrack(frame_rate=TARGET_FPS)
            counted_ids = {i: set() for i in range(len(rois_data))}
            window_log = WindowLog(w_lo, w_hi, rois_data)

            # Frame hook to collect observations per track id
            # tid -> list of obs
            hook_obs: dict[int, list[dict]] = {}

            def hook_fn(t, scale, seg_name, frame_idx, dets):
                if dets.tracker_id is None or len(dets.tracker_id) == 0:
                    return
                tids = dets.tracker_id.astype(int)
                xyxy = dets.xyxy.astype(float)
                for tid, box in zip(tids, xyxy):
                    # Scale box back to reference 1280x720 space
                    b_scaled = [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale]
                    bc = [(b_scaled[0] + b_scaled[2]) * 0.5, b_scaled[3]]
                    in_roi = cv2.pointPolygonTest(roi_polys[roi_idx], (bc[0], bc[1]), False) >= 0
                    hook_obs.setdefault(int(tid), []).append({
                        "frame": int(frame_idx),
                        "seg": str(seg_name),
                        "time_sec": round(float(t), 3),
                        "bbox": [round(x, 1) for x in b_scaled],
                        "bottom_center": [round(x, 1) for x in bc],
                        "in_roi": bool(in_roi),
                    })

            results = process_window(
                window_segments, w_lo, w_hi, rois_data, local_client,
                tracker, counted_ids, matrices, roi_polys, roi_boxes,
                target_class_ids, window_log=window_log, frame_hook=hook_fn
            )

            # Dump temporary veh CSV to read counted_cls / first / last
            with tempfile.TemporaryDirectory() as tmpdir:
                window_log.write(Path(tmpdir), "replay", results["frames"])
                veh_gz = Path(tmpdir) / "replay_veh.csv.gz"
                if not veh_gz.exists():
                    log_rows.append({"sample_id": sample_id, "status": "no_replay_veh"})
                    logged_ids.add(sample_id)
                    continue
                df_rep = pd.read_csv(veh_gz)

            # Filter replay rows to current ROI
            df_rep_roi = df_rep[df_rep.roi_idx == roi_idx]
            if df_rep_roi.empty:
                log_rows.append({"sample_id": sample_id, "status": "unmatched"})
                logged_ids.add(sample_id)
                continue

            # Match release row by t_first, t_last within 0.15s, positions within 0.5m
            # We need release row positions:
            release_veh_path = tracks_root / date / cctv_id / f"{ts_str}_veh.csv.gz"
            df_rel = pd.read_csv(release_veh_path)
            rel_row = df_rel[(df_rel.roi_idx == roi_idx) & (df_rel.track_id == track_id)]
            if rel_row.empty:
                log_rows.append({"sample_id": sample_id, "status": "unmatched"})
                logged_ids.add(sample_id)
                continue
            rel_r = rel_row.iloc[0]
            r_xf, r_yf = float(rel_r["x_first_m"]), float(rel_r["y_first_m"])
            r_xl, r_yl = float(rel_r["x_last_m"]), float(rel_r["y_last_m"])

            best_match = None
            best_diff = 1e9
            for rep_r in df_rep_roi.to_dict("records"):
                dt_f = abs(float(rep_r["t_first"]) - t_first)
                dt_l = abs(float(rep_r["t_last"]) - t_last)
                dp_f = math.hypot(float(rep_r["x_first_m"]) - r_xf, float(rep_r["y_first_m"]) - r_yf)
                dp_l = math.hypot(float(rep_r["x_last_m"]) - r_xl, float(rep_r["y_last_m"]) - r_yl)
                if dt_f <= 0.15 and dt_l <= 0.15 and dp_f <= 0.5 and dp_l <= 0.5:
                    tot_diff = dt_f + dt_l + dp_f + dp_l
                    if tot_diff < best_diff:
                        best_diff = tot_diff
                        best_match = rep_r

            if best_match is None:
                log_rows.append({"sample_id": sample_id, "status": "unmatched"})
                logged_ids.add(sample_id)
                continue

            rep_tid = int(best_match["track_id"])
            rep_cls_id = int(best_match["class_id"])
            rep_cls_name = TARGET_CLASSES.get(rep_cls_id, class_name)

            # Check EMA speed
            latest_speeds = results.get("latest_speeds", {})
            tid_entry = latest_speeds.get(rep_tid, {}).get("by_roi", {}).get(roi_idx)
            if tid_entry is None or "speed" not in tid_entry:
                log_rows.append({"sample_id": sample_id, "status": "no_ema"})
                logged_ids.add(sample_id)
                continue
            pred_ema_spd = float(tid_entry["speed"])

            # Check observations and home segment containment
            obs_list = hook_obs.get(rep_tid, [])
            in_roi_obs = [o for o in obs_list if o["in_roi"]]
            if not in_roi_obs:
                log_rows.append({"sample_id": sample_id, "status": "no_roi_obs"})
                logged_ids.add(sample_id)
                continue

            # Verify all in-ROI observations are in home_seg
            if not all(o["seg"] == home_seg for o in in_roi_obs):
                log_rows.append({"sample_id": sample_id, "status": "spans_segments"})
                logged_ids.add(sample_id)
                continue

            # Native FPS from home segment
            home_path = vdir / home_seg
            cap = cv2.VideoCapture(str(home_path))
            native_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else FALLBACK_FPS
            if native_fps <= 0 or not math.isfinite(native_fps):
                native_fps = FALLBACK_FPS
            cap.release()

            # Filter trajectory to home segment only
            home_traj = [o for o in obs_list if o["seg"] == home_seg]
            if not home_traj:
                log_rows.append({"sample_id": sample_id, "status": "empty_home_traj"})
                logged_ids.add(sample_id)
                continue

            first_frame = min(o["frame"] for o in home_traj)
            last_frame = max(o["frame"] for o in home_traj)
            est_entry_frame = in_roi_obs[0]["frame"]
            est_exit_frame = in_roi_obs[-1]["frame"]

            cand_record = {
                "sample_id": sample_id,
                "rank": int(row["rank"]),
                "date": date,
                "cctv_id": cctv_id,
                "roi_idx": roi_idx,
                "perspective": str(row["perspective"]),
                "roi_points": rois_data[roi_idx]["points"],
                "roi_length_meters": rois_data[roi_idx]["length_meters"],
                "roi_width_meters": rois_data[roi_idx]["width_meters"],
                "stratum_class": class_name,
                "class_name": class_name,
                "replay_class": rep_cls_name,
                "track_id": rep_tid,
                "release_track_id": track_id,
                "video_path": str(home_path),
                "video_name": home_seg,
                "native_fps": round(float(native_fps), 2),
                "first_frame": int(first_frame),
                "last_frame": int(last_frame),
                "est_entry_frame": int(est_entry_frame),
                "est_exit_frame": int(est_exit_frame),
                "predicted_speed_kmh": round(pred_ema_spd, 2),
                "trajectory": [
                    {
                        "frame": o["frame"],
                        "time_sec": o["time_sec"],
                        "bbox": o["bbox"],
                        "bottom_center": o["bottom_center"],
                        "in_roi": o["in_roi"],
                    }
                    for o in home_traj
                ],
            }

            new_candidates.append(cand_record)
            existing_candidates.append(cand_record)
            existing_ids.add(sample_id)
            log_rows.append({"sample_id": sample_id, "status": "ok"})
            logged_ids.add(sample_id)
            cur_pending += 1
            print(f"  [+] Prepared candidate {sample_id} | Class: {class_name} | Pred: {pred_ema_spd:.2f} km/h (Stratum pending: {cur_pending}/{target_pending})")

        status_map[(cctv_id, roi_idx, class_name)]["pending"] = cur_pending
    # Save candidates atomically
    tmp_cand = candidates_path.with_name(candidates_path.name + ".tmp")
    with open(tmp_cand, "w", encoding="utf-8") as f:
        json.dump(existing_candidates, f, indent=2)
    os.replace(tmp_cand, candidates_path)

    # Append to prepare_log.csv
    if log_rows:
        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "status"])
            if write_header:
                writer.writeheader()
            writer.writerows(log_rows)

    print(f"\n[✓] Finished prepare. Total candidates now: {len(existing_candidates)}")

    # Deficit summary table
    print("\n--- Speed Validation Deficit Status ---")
    print(f"{'CCTV':<36} {'ROI':>3} {'Class':<10} {'Accepted':>8} {'Pending':>7} {'Target':>6} {'FreshLeft':>9}")
    exhausted = []
    for (cctv_id, roi_idx, class_name), st in sorted(status_map.items()):
        if st["deficit"] > 0:
            target = fill_target(st["deficit"], factor)
            group = pool_df[(pool_df.cctv_id == cctv_id) & (pool_df.roi_idx == roi_idx) & (pool_df.class_name == class_name)]
            fresh_left = len([s for s in group["sample_id"] if s not in existing_ids and s not in logged_ids]) if not group.empty else 0
            print(f"{cctv_id:<36} {roi_idx:>3} {class_name:<10} {st['accepted']:>8} {st['pending']:>7} {target:>6} {fresh_left:>9}")
            if st["pending"] < target and fresh_left == 0:
                exhausted.append((cctv_id, roi_idx, class_name))

    for cctv_id, roi_idx, class_name in exhausted:
        print(f"[!] pool exhausted: {cctv_id}:{roi_idx}:{class_name} -> run: speed_val.py fetch --stratum {cctv_id}:{roi_idx}:{class_name}")
    return 0


# ==============================================================================
# FINAL SAMPLE SELECTION & BOOTSTRAP METRICS
# ==============================================================================

def select_final(annotations: dict) -> tuple[pd.DataFrame, list[tuple]]:
    """Filter annotations to accepted and pick top PER_CLASS by rank per (cctv_id, roi_idx, true_class)."""
    accepted_rows = []
    quota_rejections = []

    for key, a in annotations.items():
        is_fp = bool(a.get("false_positive", False))
        is_invalid = bool(a.get("invalid_crossing", False))
        gt = a.get("ground_truth_speed_kmh")
        t_cls = a.get("true_class", a.get("class_name", "car"))
        cctv = a.get("cctv_id")
        roi = a.get("roi_idx")
        rank = a.get("rank", 999999)

        if is_fp:
            quota_rejections.append((cctv, roi, t_cls, "false_positive"))
        elif is_invalid:
            quota_rejections.append((cctv, roi, t_cls, "invalid_crossing"))
        elif gt is None:
            quota_rejections.append((cctv, roi, t_cls, "no_gt"))
        else:
            accepted_rows.append({
                "sample_id": key,
                "cctv_id": cctv,
                "roi_idx": roi,
                "perspective": a.get("perspective", "depan"),
                "true_class": t_cls,
                "rank": rank,
                "pred_speed": float(a["predicted_speed_kmh"]),
                "gt_speed": float(gt),
            })

    # Pick top 5 accepted per (cctv_id, roi_idx, true_class)
    df_acc = pd.DataFrame(accepted_rows)
    final_rows = []
    if not df_acc.empty:
        for (cctv, roi, t_cls), grp in df_acc.groupby(["cctv_id", "roi_idx", "true_class"]):
            grp = grp.sort_values("rank")
            kept = grp.iloc[:PER_CLASS]
            overflow = grp.iloc[PER_CLASS:]
            final_rows.extend(kept.to_dict("records"))
            for _ in range(len(overflow)):
                quota_rejections.append((cctv, roi, t_cls, "class_overflow"))

    return pd.DataFrame(final_rows), quota_rejections


def _boot_ci(sub: pd.DataFrame, key: str) -> dict:
    """Compute 95% stratified percentile bootstrap confidence intervals."""
    n = len(sub)
    if n == 0:
        return {
            "mae_lo": None, "mae_hi": None,
            "rmse_lo": None, "rmse_hi": None,
            "bias_lo": None, "bias_hi": None,
            "r_lo": None, "r_hi": None,
        }
    seed = SEED ^ zlib.crc32(key.encode())
    rng = np.random.default_rng(seed)

    sub_reset = sub.reset_index(drop=True)
    pred_arr = sub_reset["pred_speed"].to_numpy(dtype=np.float64)
    gt_arr = sub_reset["gt_speed"].to_numpy(dtype=np.float64)

    if all(col in sub_reset.columns for col in ["cctv_id", "roi_idx", "true_class"]):
        strata_positions = [grp.index.to_numpy() for _, grp in sub_reset.groupby(["cctv_id", "roi_idx", "true_class"])]
    else:
        strata_positions = [np.arange(n)]

    boot_indices = np.empty((BOOT_B, n), dtype=np.int64)
    col_start = 0
    for pos in strata_positions:
        size_j = len(pos)
        boot_indices[:, col_start : col_start + size_j] = rng.choice(pos, size=(BOOT_B, size_j), replace=True)
        col_start += size_j

    b_pred = pred_arr[boot_indices]
    b_gt = gt_arr[boot_indices]
    b_err = b_pred - b_gt

    b_mae = np.mean(np.abs(b_err), axis=1)
    b_rmse = np.sqrt(np.mean(b_err**2, axis=1))
    b_bias = np.mean(b_err, axis=1)

    pred_diff = b_pred - np.mean(b_pred, axis=1, keepdims=True)
    gt_diff = b_gt - np.mean(b_gt, axis=1, keepdims=True)

    denom = np.sqrt(np.sum(pred_diff**2, axis=1) * np.sum(gt_diff**2, axis=1))
    valid_r = denom > 1e-12

    mae_lo, mae_hi = np.percentile(b_mae, [2.5, 97.5])
    rmse_lo, rmse_hi = np.percentile(b_rmse, [2.5, 97.5])
    bias_lo, bias_hi = np.percentile(b_bias, [2.5, 97.5])

    degen_count = np.sum(~valid_r)
    if degen_count > 0.01 * BOOT_B:
        r_lo, r_hi = None, None
    else:
        b_r = np.full(BOOT_B, np.nan)
        b_r[valid_r] = np.sum(pred_diff[valid_r] * gt_diff[valid_r], axis=1) / denom[valid_r]
        r_lo, r_hi = np.nanpercentile(b_r, [2.5, 97.5])

    return {
        "mae_lo": round(float(mae_lo), 2),
        "mae_hi": round(float(mae_hi), 2),
        "rmse_lo": round(float(rmse_lo), 2),
        "rmse_hi": round(float(rmse_hi), 2),
        "bias_lo": round(float(bias_lo), 2),
        "bias_hi": round(float(bias_hi), 2),
        "r_lo": round(float(r_lo), 4) if r_lo is not None else None,
        "r_hi": round(float(r_hi), 4) if r_hi is not None else None,
    }


def _metrics(sub_df: pd.DataFrame, key: str | None = None) -> dict:
    n = len(sub_df)
    if n == 0:
        return {
            "n": 0, "mae": None, "mae_lo": None, "mae_hi": None,
            "rmse": None, "rmse_lo": None, "rmse_hi": None,
            "bias": None, "bias_lo": None, "bias_hi": None,
            "sd": None, "loa_lo": None, "loa_hi": None,
            "mape": None, "r": None, "r_lo": None, "r_hi": None,
        }
    err = sub_df["pred_speed"] - sub_df["gt_speed"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err**2).mean()))
    bias = float(err.mean())
    sd = float(err.std(ddof=1)) if n >= 2 else None
    loa_lo = float(bias - 1.96 * sd) if sd is not None else None
    loa_hi = float(bias + 1.96 * sd) if sd is not None else None
    mape = float((err.abs() / sub_df["gt_speed"]).mean() * 100.0)

    # Pearson correlation
    if n >= 3 and sub_df["pred_speed"].std() > 1e-6 and sub_df["gt_speed"].std() > 1e-6:
        r = float(np.corrcoef(sub_df["pred_speed"], sub_df["gt_speed"])[0, 1])
    else:
        r = None

    ci = _boot_ci(sub_df, key) if key is not None else {
        "mae_lo": None, "mae_hi": None,
        "rmse_lo": None, "rmse_hi": None,
        "bias_lo": None, "bias_hi": None,
        "r_lo": None, "r_hi": None,
    }

    return {
        "n": n,
        "mae": round(mae, 2),
        "mae_lo": ci["mae_lo"],
        "mae_hi": ci["mae_hi"],
        "rmse": round(rmse, 2),
        "rmse_lo": ci["rmse_lo"],
        "rmse_hi": ci["rmse_hi"],
        "bias": round(bias, 2),
        "bias_lo": ci["bias_lo"],
        "bias_hi": ci["bias_hi"],
        "sd": round(sd, 2) if sd is not None else None,
        "loa_lo": round(loa_lo, 2) if loa_lo is not None else None,
        "loa_hi": round(loa_hi, 2) if loa_hi is not None else None,
        "mape": round(mape, 2),
        "r": round(r, 4) if r is not None else None,
        "r_lo": ci["r_lo"],
        "r_hi": ci["r_hi"],
    }


# ==============================================================================
# SUBCOMMAND: report
# ==============================================================================

def cmd_report(args: argparse.Namespace) -> int:
    anno_path = OUT / "annotations.json"
    if not anno_path.exists():
        print(f"[!] No annotations found at {anno_path}")
        return 1
    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    df_final, quota_rejections = select_final(annotations)

    metrics_out = {}
    csv_rows = []

    def _add_metric_row(group_type: str, group_key: str, sub: pd.DataFrame):
        key = f"{group_type}|{group_key}"
        m = _metrics(sub, key=key)
        metrics_out.setdefault(group_type, {})[group_key] = m
        csv_rows.append({
            "group_type": group_type,
            "group_key": group_key,
            **m,
        })

    if not df_final.empty:
        _add_metric_row("overall", "overall", df_final)

        for p, g in df_final.groupby("perspective"):
            _add_metric_row("perspective", str(p), g)

        for c, g in df_final.groupby("cctv_id"):
            _add_metric_row("cctv", str(c), g)

        for cl, g in df_final.groupby("true_class"):
            _add_metric_row("class", str(cl), g)

        for (c, r), g in df_final.groupby(["cctv_id", "roi_idx"]):
            _add_metric_row("roi", f"{c}:ROI{r+1}", g)

    # Save metrics JSON & CSV
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    pd.DataFrame(csv_rows).to_csv(OUT / "metrics.csv", index=False)

    # Save final_sample.csv
    final_sample_rows = []
    for sid in df_final["sample_id"]:
        entry = dict(annotations[sid])
        pred = float(entry["predicted_speed_kmh"])
        gt = float(entry["ground_truth_speed_kmh"])
        entry["error_kmh"] = round(pred - gt, 4)
        final_sample_rows.append(entry)
    pd.DataFrame(final_sample_rows).to_csv(OUT / "final_sample.csv", index=False)

    # Save quota.csv
    # All 12 ROIs x 4 classes
    cctv_rois = {}
    for roi_file in (REPO_ROOT / "rois").glob("*_roi.json"):
        cctv_id = roi_file.stem.replace("_roi", "")
        cctv_rois[cctv_id] = load_rois(cctv_id)

    quota_rows = []
    for cctv_id, rois in sorted(cctv_rois.items()):
        for r_idx in range(len(rois)):
            for cls in CLASSES:
                sub_f = df_final[(df_final.cctv_id == cctv_id) & (df_final.roi_idx == r_idx) & (df_final.true_class == cls)] if not df_final.empty else []
                n_acc = len(sub_f)
                rej_fp = sum(1 for c, r, cl, rj in quota_rejections if c == cctv_id and r == r_idx and cl == cls and rj == "false_positive")
                rej_inv = sum(1 for c, r, cl, rj in quota_rejections if c == cctv_id and r == r_idx and cl == cls and rj == "invalid_crossing")
                rej_nogt = sum(1 for c, r, cl, rj in quota_rejections if c == cctv_id and r == r_idx and cl == cls and rj == "no_gt")
                rej_of = sum(1 for c, r, cl, rj in quota_rejections if c == cctv_id and r == r_idx and cl == cls and rj == "class_overflow")
                quota_rows.append({
                    "cctv_id": cctv_id,
                    "roi_idx": r_idx + 1,
                    "class": cls,
                    "accepted": n_acc,
                    "target": PER_CLASS,
                    "false_positive": rej_fp,
                    "invalid_crossing": rej_inv,
                    "no_gt": rej_nogt,
                    "class_overflow": rej_of,
                })
    pd.DataFrame(quota_rows).to_csv(OUT / "quota.csv", index=False)

    print(f"\n[✓] Speed evaluation report generated:")
    print(f"    Metrics JSON: {OUT / 'metrics.json'}")
    print(f"    Metrics CSV : {OUT / 'metrics.csv'}")
    print(f"    Final Sample: {OUT / 'final_sample.csv'}")
    print(f"    Quota CSV   : {OUT / 'quota.csv'}")
    if "overall" in metrics_out:
        print(f"    Overall: {metrics_out['overall']['overall']}")
    return 0


# ==============================================================================
# SUBCOMMAND: subsets
# ==============================================================================

def cmd_subsets(args: argparse.Namespace) -> int:
    anno_path = OUT / "annotations.json"
    if not anno_path.exists():
        print(f"[!] No annotations found at {anno_path}")
        return 1
    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    df_final, _ = select_final(annotations)
    if df_final.empty:
        print("[!] No accepted annotations found.")
        return 1

    # 1. second_subset.json: 48 sample_ids, lowest-rank row per stratum in df_final
    # sorted by (cctv_id, roi_idx, true_class)
    second_subset = []
    for (cctv, roi, t_cls), grp in df_final.groupby(["cctv_id", "roi_idx", "true_class"]):
        grp = grp.sort_values("rank")
        second_subset.append(grp.iloc[0]["sample_id"])

    with open(OUT / "second_subset.json", "w", encoding="utf-8") as f:
        json.dump(second_subset, f, indent=2)

    # 2. causes_list.json: 94 invalid + 24 worst (|pred - gt|, ties broken by rank)
    invalid_ids = [key for key, a in annotations.items() if a.get("invalid_crossing", False)]

    df_final["abs_err"] = (df_final["pred_speed"] - df_final["gt_speed"]).abs()
    worst_24 = df_final.sort_values(by=["abs_err", "rank"], ascending=[False, True]).iloc[:24]
    worst_ids = worst_24["sample_id"].tolist()

    causes_list = [{"sample_id": sid, "source": "invalid"} for sid in invalid_ids] + \
                  [{"sample_id": sid, "source": "worst"} for sid in worst_ids]

    # Assert no duplicates
    sids = [x["sample_id"] for x in causes_list]
    assert len(sids) == len(set(sids)), "Duplicate sample_id found in causes_list!"

    with open(OUT / "causes_list.json", "w", encoding="utf-8") as f:
        json.dump(causes_list, f, indent=2)

    print(f"\n[✓] Subsets generated:")
    print(f"    Second annotator subset (n={len(second_subset)}): {OUT / 'second_subset.json'}")
    print(f"    Failure causes list     (n={len(causes_list)}): {OUT / 'causes_list.json'}")
    return 0

# ==============================================================================
# SUBCOMMAND: selfcheck
# ==============================================================================

def cmd_selfcheck(args: argparse.Namespace) -> int:
    date = args.date
    cctv_id = args.cctv
    ts_str = args.ts

    tracks_dir = REPO_ROOT / "extracted" / "tracks" / date / cctv_id
    meta_path = tracks_dir / f"{ts_str}_meta.json"
    veh_path = tracks_dir / f"{ts_str}_veh.csv.gz"
    if not meta_path.exists() or not veh_path.exists():
        print(f"[!] Release track files not found: {meta_path} / {veh_path}")
        return 1

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    lo_sec, hi_sec = float(meta["lo_sec"]), float(meta["hi_sec"])

    df_rel = pd.read_csv(veh_path)
    rel_rows = df_rel[df_rel.n_obs >= 3].to_dict("records")

    vdir = video_dir(date, cctv_id)
    if vdir is None:
        print(f"[!] Video dir not found for {date} / {cctv_id}")
        return 1

    all_segments = build_segment_index(vdir)
    seg_starts = [s["start_sec"] for s in all_segments]
    window_segments = select_window_segments(all_segments, seg_starts, lo_sec, hi_sec)
    if not window_segments:
        print(f"[!] No segments found in [{lo_sec}, {hi_sec}]")
        return 1

    rois_data = load_rois(cctv_id)
    matrices = get_perspective_matrices(rois_data)
    roi_polys = [np.array(r["points"], dtype=np.int32) for r in rois_data]
    roi_boxes = [(int(p[:, 0].min()), int(p[:, 1].min()),
                  int(p[:, 0].max()), int(p[:, 1].max())) for p in roi_polys]
    target_class_ids = list(TARGET_CLASSES.keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Detector(device)
    local_client = LocalClient(detector)
    tracker = sv.ByteTrack(frame_rate=TARGET_FPS)
    counted_ids = {i: set() for i in range(len(rois_data))}
    window_log = WindowLog(lo_sec, hi_sec, rois_data)

    print(f"[*] Replaying full window {ts_str} on {cctv_id} ({len(window_segments)} segments)...")
    results = process_window(
        window_segments, lo_sec, hi_sec, rois_data, local_client,
        tracker, counted_ids, matrices, roi_polys, roi_boxes,
        target_class_ids, window_log=window_log
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        window_log.write(Path(tmpdir), "replay", results["frames"])
        df_rep = pd.read_csv(Path(tmpdir) / "replay_veh.csv.gz")

    rep_rows = df_rep[df_rep.n_obs >= 3].to_dict("records")

    # Match each release row to replay rows
    matched_all = 0
    matched_warm = 0
    total_warm = 0

    for r_rel in rel_rows:
        roi = int(r_rel["roi_idx"])
        tf, tl = float(r_rel["t_first"]), float(r_rel["t_last"])
        xf, yf = float(r_rel["x_first_m"]), float(r_rel["y_first_m"])
        xl, yl = float(r_rel["x_last_m"]), float(r_rel["y_last_m"])

        is_warm = (tf - lo_sec) >= 20.0
        if is_warm:
            total_warm += 1

        matched = False
        for r_rep in rep_rows:
            if int(r_rep["roi_idx"]) != roi:
                continue
            dt_f = abs(float(r_rep["t_first"]) - tf)
            dt_l = abs(float(r_rep["t_last"]) - tl)
            dp_f = math.hypot(float(r_rep["x_first_m"]) - xf, float(r_rep["y_first_m"]) - yf)
            dp_l = math.hypot(float(r_rep["x_last_m"]) - xl, float(r_rep["y_last_m"]) - yl)
            if dt_f <= 0.15 and dt_l <= 0.15 and dp_f <= 0.5 and dp_l <= 0.5:
                matched = True
                break

        if matched:
            matched_all += 1
            if is_warm:
                matched_warm += 1

    frac_all = matched_all / len(rel_rows) if rel_rows else 1.0
    frac_warm = matched_warm / total_warm if total_warm else 1.0

    print(f"\n--- Selfcheck Replay Fidelity Results for {cctv_id} ({ts_str}) ---")
    print(f"Matched total: {matched_all}/{len(rel_rows)} ({frac_all * 100:.1f}%)")
    print(f"Matched warm (t_first - lo_sec >= 20s): {matched_warm}/{total_warm} ({frac_warm * 100:.1f}%)")

    # Compare average speeds with released metadata features CSV
    feat_csv = REPO_ROOT / "extracted" / f"{cctv_id}_{date}_metadata_features.csv"
    max_spd_diff = 0.0
    if feat_csv.exists():
        df_feat = pd.read_csv(feat_csv)
        row_feat = df_feat[df_feat.timestampsafe.astype(str).str.zfill(6) == ts_str]
        if not row_feat.empty:
            r0 = row_feat.iloc[0]
            for r_idx in range(len(rois_data)):
                col = f"ROI_{r_idx+1}_AvgSpeed_Kmh"
                rel_spd = float(r0.get(col, 0.0))
                rep_spd = float(results["avg_speeds"].get(r_idx, 0.0))
                diff = abs(rep_spd - rel_spd)
                max_spd_diff = max(max_spd_diff, diff)
                print(f"ROI {r_idx+1} AvgSpeed: Released={rel_spd:.2f} | Replay={rep_spd:.2f} | Diff={diff:.2f} km/h")

    pass_match = frac_warm >= 0.90
    pass_speed = max_spd_diff <= 1.0
    print(f"Fidelity status: {'PASS' if (pass_match and pass_speed) else 'FAIL'}")
    return 0 if (pass_match and pass_speed) else 2


# ==============================================================================
# MAIN DISPATCHER
# ==============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pool = sub.add_parser("pool", help="Scan release track logs to build candidate pool")
    p_pool.set_defaults(func=cmd_pool)
    p_index = sub.add_parser("index", help="Index remote 7z archives to generate segment lists")
    p_index.add_argument("--dates", required=True, help="Comma-separated dates YYYY-MM-DD")
    p_index.set_defaults(func=cmd_index)

    p_fetch = sub.add_parser("fetch", help="Download and extract segments for scarce strata")
    p_fetch.add_argument("--stratum", required=True, help="Stratum specification CCTV:ROI:CLASS")
    p_fetch.add_argument("--max-dates", type=int, default=4, help="Maximum number of dates to download")
    p_fetch.add_argument("--need", type=int, default=None, help="Target number of fresh candidates")
    p_fetch.set_defaults(func=cmd_fetch)

    p_prep = sub.add_parser("prepare", help="Replay detector/tracker on candidate clips")
    p_prep.add_argument("--factor", type=int, default=FILL_FACTOR, help="Multiplier for deficit to determine pending target")
    p_prep.set_defaults(func=cmd_prepare)

    p_rep = sub.add_parser("report", help="Compute metrics from annotations")
    p_rep.set_defaults(func=cmd_report)

    p_sub = sub.add_parser("subsets", help="Generate subsets for second annotator and failure cause inspection")
    p_sub.set_defaults(func=cmd_subsets)

    p_self = sub.add_parser("selfcheck", help="Verify replay fidelity against released window")
    p_self.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    p_self.add_argument("--cctv", required=True, help="CCTV location name")
    p_self.add_argument("--ts", required=True, help="Timestamp HHMMSS")
    p_self.set_defaults(func=cmd_selfcheck)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

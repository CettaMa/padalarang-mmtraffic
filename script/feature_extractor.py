"""
Pipeline gabungan: ekstraksi fitur LANGSUNG dari dataset hasil ekstraksi
(tanpa tahap merge .ts -> mp4 seperti di script Colab).

Struktur dataset yang diharapkan:
    dataset/
      <YYYY-MM-DD>/
        <nama_cctv>/
          metadata/   -> CSV; satu baris = window 5 menit (kolom `timestampsafe`, HHMMSS)
          videos/     -> segmen .ts ~28 detik; jam mulai = token terakhir nama file (_HHMMSS)

Pendekatan (opsi 1): semua segmen .ts milik satu window di-decode berurutan
oleh satu reader thread, dan timestamp tiap frame dihitung dari jam mulai
segmen (nama file) + frame_idx/fps = waktu absolut sejak tengah malam.
Dengan waktu absolut:
- kecepatan & aging riwayat posisi tetap benar menyeberang batas segmen
  (versi lama: video_time_sec reset ke 0 tiap file -> dt negatif -> speed
  lintas file selalu ter-skip),
- gap (segmen hilang) otomatis terhandle karena waktu tetap melompat maju,
- batch inference tetap boleh menyeberang segmen -> utilisasi GPU sama.

vs opsi 2 (gabung .ts lalu remux mp4): remux `-c copy` itu lossless, piksel
dan hasil deteksi identik, jadi merge hanya menambah biaya I/O + storage
dobel. Opsi 2 baru berguna kalau butuh arsip .mp4 yang bisa diputar untuk
verifikasi manual / labeling.

Pipeline per window (reader thread + consumer thread di worker; inferensi
GPU dipisah ke proses inference server terpisah — dulu decode+infer+
postprocess berjalan bergantian di satu thread sehingga GPU menganggur):
    [worker] reader thread: decode + subsample + resize
        -> tulis frame ke slot shared_memory, kirim (req_id, slots) ke server
        -> [proses inference server: model.predict(batch)] -> hasil ke resp_queue
        -> thread pemanggil: tracker + fitur, dikonsumsi FIFO lewat req_id
Konsekuensi: hasil inferensi harus diproses berurutan waktu (FIFO per
req_id, disangga di InferClient kalau datang di luar urutan) supaya state
tracker tetap monoton, walau ada lebih dari satu inference server.

Sampling: laju ambil frame tepat TARGET_FPS (bukan pembulatan ke bawah/atas
stride integer seperti `round(fps/10)` yang menghasilkan 12.5 fps untuk
sumber 25 fps). Semua frame tetap di-decode karena rantai P-frame H.264.

Batas window: [timestamp baris, timestamp baris berikutnya). Segmen yang
menyeberang batas ikut dibaca dari frame pertamanya, lalu frame di luar
rentang dibuang -> coverage tidak bolong dan tidak dobel.

Output ditulis ke CSV terpisah `<nama_metadata>_features.csv` (file metadata
asli tidak pernah ditimpa) secara atomik.
"""

import os

# Harus di-set SEBELUM import torch/cv2 (lihat penjelasan affinity di bawah).
os.environ.setdefault("KMP_AFFINITY", "disabled")

from collections import Counter
import bisect
import contextlib
import json
import math
import multiprocessing as mp
import queue
import subprocess
from multiprocessing import shared_memory
import threading
import cv2
import numpy as np
import pandas as pd
import psutil
try:
    import torch
    import torchvision
except ImportError:
    torch = None
    torchvision = None
from pathlib import Path
from tqdm import tqdm
try:
    from ultralytics import YOLO
    from ultralytics.utils import ops as yolo_ops
except ImportError:
    YOLO = None
    yolo_ops = None
try:
    import supervision as sv
except ImportError:
    sv = None

# =================== CONFIGURATIONS ===================
# Semua path di-anchor ke root repo (parent dari folder script/), supaya
# script bisa dijalankan dari direktori mana pun.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(os.environ.get("FE_DATASET_ROOT") or (REPO_ROOT / "dataset"))
# Kosong = auto-discover semua folder tanggal. FE_DATES (dipisah koma) dipakai
# dataset/run_pipeline.sh untuk memaku satu tanggal per invocation.
DATASET_DATES: list[str] = [d.strip() for d in os.environ.get("FE_DATES", "").split(",") if d.strip()]
ROI_DIR = REPO_ROOT / "rois"
# FE_MISSING_REPORT: laporan per tanggal (dipakai run_pipeline.sh sebagai bukti
# window mana yang memang tidak punya sumber video sebelum foldernya dihapus).
MISSING_REPORT_PATH = Path(os.environ.get("FE_MISSING_REPORT") or (REPO_ROOT / "extraction_missing_report.json"))
# Sinyal stop halus: begitu file ini ada, worker berhenti mengambil window/CCTV
# baru SETELAH checkpoint window terakhir tersimpan, jadi run berikutnya lanjut
# persis dari baris terakhir yang selesai (bukan mengulang window).
STOP_FILE = Path(os.environ.get("FE_STOP_FILE") or (REPO_ROOT / ".pipeline_stop"))
# run_pipeline.sh membaca PID ini untuk taskkill /T seluruh pohon proses saat stop.
PID_FILE = Path(os.environ["FE_PID_FILE"]) if os.environ.get("FE_PID_FILE") else None
# Satu feature_extractor per mesin: dua instance masing-masing meluncurkan
# inference server berdasarkan VRAM bebas saat start -> 4 konteks TensorRT
# di GPU 6 GB -> OOM. Lock dilepas OS otomatis saat proses mati.
INSTANCE_LOCK_PATH = REPO_ROOT / ".feature_extractor.lock"

# Struktur folder dalam satu CCTV (sesuai hasil ekstraksi dataset)
METADATA_SUBDIR = "metadata"
VIDEOS_SUBDIR = "videos"
VIDEO_SUFFIXES = (".ts", ".mkv", ".mp4")  # prioritas: asli dulu

# Folder output CSV hasil ekstraksi fitur. Metadata asli di dalam
# dataset/<tanggal>/<cctv>/metadata/ TIDAK pernah ditimpa: hasil ditulis ke
# EXTRACTED_DIR/<nama_metadata><OUTPUT_SUFFIX>.csv.
EXTRACTED_DIR = Path(os.environ.get("FE_EXTRACTED_DIR") or (REPO_ROOT / "extracted"))
TRACKLOG_DIR = EXTRACTED_DIR / "tracks"
OUTPUT_SUFFIX = "_features"

# Panjang satu window (detik). Batas atas window = min(timestamp baris
# berikutnya, timestamp baris ini + WINDOW_SECONDS) supaya satu baris tidak
# "menyerap" segmen berlebih kalau ada baris metadata yang hilang di tengah.
WINDOW_SECONDS = 300.0

# CCTV yang mau dilewati (case-insensitive) — pengganti EXCLUDED_LOC_FOLDERS
# dari script Colab lama.
EXCLUDED_CCTVS: set[str] = set()

# Bobot model: engine TensorRT lebih dulu, lalu .pt.
MODEL_DIR = REPO_ROOT / "model"
MODEL_WEIGHTS = next(
    (p for p in (MODEL_DIR / "best.engine", MODEL_DIR / "best.pt") if p.exists()),
    MODEL_DIR / "best.engine",
)

TARGET_FPS = 10
INFER_BATCH_SIZE = 24
INFER_IMGSZ = 640
# Lebar frame yang dikirim ke model. Frame sumber 1280x720 (2.76 MB/frame)
# di-resize di reader supaya antrian tidak menyimpan 720p utuh; 0 = tanpa
# resize (paling aman secara numerik, paling boros memori).
PROC_WIDTH = 640
# Lebar acuan ruang koordinat ROI (rois/*.json dibuat pada frame 1280x720).
# _proc_size() memetakan skala relatif ke lebar INI, bukan ke lebar file
# sumber, supaya arsip yang sudah di-downscale tetap konsisten dengan ROI.
ROI_REF_WIDTH = 1280
QUANTIZE = 16 if (torch is not None and torch.cuda.is_available()) else None
# Paralelisme: MAX_INFER_SERVERS proses inferensi (pemilik konteks CUDA) +
# n_workers proses decode/tracking yang tidak menyentuh GPU. Terukur di RTX
# 3060 Laptop 6 GB: predict-only 1 proses = 477 fps, 2 proses = 618 fps,
# 3 proses = 179 fps (3 x 1.9 GB konteks di kartu 6 GB -> thrashing), jadi
# jumlah server dibatasi 2.
MAX_INFER_SERVERS = 2
INFER_SERVER_VRAM_MB = 2100        # terukur 1919 MiB/proses (TensorRT + torch)
# Slot frame shared memory per worker: 2x batch = double buffering, cukup
# untuk menjaga decode tetap jalan selama server memproses batch sebelumnya.
INFER_SLOTS = 2 * INFER_BATCH_SIZE
FRAME_SLOT_BYTES = (PROC_WIDTH or 1920) ** 2 * 3
# Server mati/hang tidak boleh menggantung worker selamanya.
INFER_RESPONSE_TIMEOUT_SEC = 300
WORKER_CV_THREADS = 1              # worker hanya decode (ffmpeg internal) + numpy
# Thread decoder ffmpeg per VideoCapture. Default OpenCV = jumlah core (12) per segmen dan
# TIDAK dibatasi cv2.setNumThreads; 6 worker x 12 thread -> CPU oversubscribed. 1 thread:
# frame identik (SHA-256 sama), CPU per window ~2x lebih hemat.
DECODE_THREADS = 1
SERVER_CV_THREADS = 2              # server: letterbox + NMS
FALLBACK_FPS = 30.0
HISTORY_MAX_AGE_SEC = 2.0
TARGET_CLASSES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}
CONF_THRESH = 0.4
# Default `max_wh` di ultralytics.utils.nms.non_max_suppression (offset kelas untuk NMS per-kelas).
NMS_MAX_WH = 7680
MIN_SPEED_KMH = 5
MAX_SPEED_KMH = 160
QUEUE_TICK_SEC = 1.0
QUEUE_PREV_MAX_GAP_SEC = 2.5
# ======================================================

# Argumen predict: `quantize` baru ada di ultralytics >= 8.3 (menggantikan
# half/int8). Versi lama akan menolak kwarg tak dikenal -> deteksi dulu.
try:
    from ultralytics.cfg import DEFAULT_CFG_DICT as _YOLO_CFG

    _HAS_QUANTIZE = "quantize" in _YOLO_CFG
except Exception:
    _HAS_QUANTIZE = False

PREDICT_KWARGS: dict = {"conf": CONF_THRESH, "verbose": False, "imgsz": INFER_IMGSZ}
if _HAS_QUANTIZE:
    PREDICT_KWARGS["quantize"] = QUANTIZE
elif QUANTIZE:
    PREDICT_KWARGS["half"] = True


def get_perspective_matrices(rois_data):
    matrices = []
    for roi in rois_data:
        pts = roi["points"]
        if len(pts) < 4:
            raise ValueError(
                f"ROI hanya memiliki {len(pts)} titik. Minimal 4 titik diperlukan."
            )
        w = roi["width_meters"]
        l = roi["length_meters"]
        src = np.array(pts[:4], dtype=np.float32)
        dst = np.array([[0, 0], [w, 0], [w, l], [0, l]], dtype=np.float32)
        matrices.append(cv2.getPerspectiveTransform(src, dst))
    return matrices


class WindowLog:
    def __init__(self, lo_sec: float, hi_sec: float, rois_data: list):
        self.lo_sec = float(lo_sec)
        self.hi_sec = float(hi_sec)
        self._queue_zones = []
        self._queue_rois = []
        for idx, roi in enumerate(rois_data):
            qz = roi.get("queue_zone")
            if not qz:
                continue

            segments = []
            if "segments" in qz:
                segments = qz["segments"]
            elif "points" in qz:
                segments = [{
                    "points": qz["points"],
                    "length_meters": qz.get("length_meters", 30.0),
                    "width_meters": qz.get("width_meters", 7.0),
                }]

            y_offset = 0.0
            for seg in segments:
                pts = seg.get("points", [])
                if len(pts) < 4:
                    raise ValueError(
                        f"Queue zone ROI {idx} segment has {len(pts)} points. Minimum 4 required."
                    )
                poly = np.array(pts[:4], dtype=np.int32)
                box = (int(poly[:, 0].min()), int(poly[:, 1].min()),
                       int(poly[:, 0].max()), int(poly[:, 1].max()))
                w = float(seg.get("width_meters", 7.0))
                l = float(seg.get("length_meters", 30.0))
                src = np.array(pts[:4], dtype=np.float32)
                dst = np.array([[0, 0], [w, 0], [w, l], [0, l]], dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(src, dst)
                self._queue_zones.append((idx, poly, box, matrix, y_offset))
                y_offset += l

            if segments:
                self._queue_rois.append(idx)
        self._veh = {}
        self._q_rows = []
        self._q_prev = {}
        self._next_tick = math.ceil(lo_sec)
        self._n_ticks = 0

    def observe_frame(self, t: float, points: np.ndarray, track_ids: np.ndarray, class_ids: np.ndarray) -> None:
        if not self._queue_zones:
            return
        if t < self._next_tick:
            return

        self._n_ticks += 1
        self._next_tick = math.floor(t) + QUEUE_TICK_SEC

        if len(points) == 0:
            return

        seen_in_tick = set()
        for roi_idx, poly, box, matrix, y_offset in self._queue_zones:
            x0, y0, x1, y1 = box
            in_zone = []
            for i in range(len(points)):
                px, py = float(points[i, 0]), float(points[i, 1])
                if x0 <= px <= x1 and y0 <= py <= y1 and cv2.pointPolygonTest(poly, (px, py), False) >= 0:
                    in_zone.append(i)

            if not in_zone:
                continue

            sel = np.array(in_zone, dtype=int)
            src = points[sel].reshape(-1, 1, 2).astype(np.float32)
            pos_m_all = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
            for idx_in_sel, orig_i in enumerate(sel):
                tid = int(track_ids[orig_i])
                if (roi_idx, tid) in seen_in_tick:
                    continue
                seen_in_tick.add((roi_idx, tid))

                cid = int(class_ids[orig_i])
                xm = float(pos_m_all[idx_in_sel, 0])
                ym = float(pos_m_all[idx_in_sel, 1]) + y_offset
                pos = np.array([xm, ym], dtype=np.float64)

                prev = self._q_prev.get((roi_idx, tid))
                speed_kmh = float("nan")
                if prev is not None:
                    t_prev, pos_prev = prev
                    dt = t - t_prev
                    if 0 < dt <= QUEUE_PREV_MAX_GAP_SEC:
                        dist = float(np.linalg.norm(pos - pos_prev))
                        spd = (dist / dt) * 3.6
                        if spd <= MAX_SPEED_KMH and math.isfinite(spd):
                            speed_kmh = spd
                self._q_rows.append((float(t), roi_idx, tid, cid, xm, ym, speed_kmh))
                self._q_prev[(roi_idx, tid)] = (float(t), pos)
    def roi_obs(self, roi_idx: int, track_id: int, class_id: int, t: float, pos_m: np.ndarray, counted_new: bool) -> None:
        key = (roi_idx, track_id)
        pos_copy = np.array(pos_m, dtype=np.float64, copy=True)
        rec = self._veh.get(key)
        if rec is None:
            self._veh[key] = {
                "cls": Counter([class_id]),
                "t_first": float(t),
                "t_last": float(t),
                "n": 1,
                "p_first": pos_copy,
                "p_last": pos_copy,
                "counted_new": bool(counted_new),
                "counted_cls": class_id if counted_new else None,
            }
        else:
            rec["cls"][class_id] += 1
            if t < rec["t_first"]:
                rec["t_first"] = float(t)
                rec["p_first"] = pos_copy
            if t >= rec["t_last"]:
                rec["t_last"] = float(t)
                rec["p_last"] = pos_copy
            rec["n"] += 1
            if counted_new:
                rec["counted_new"] = True
                rec["counted_cls"] = class_id
    def write(self, out_dir: Path, ts_str: str, frames: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        veh_rows = []
        for (roi_idx, track_id), rec in self._veh.items():
            if rec.get("counted_cls") is not None:
                most_common_cls = rec["counted_cls"]
            else:
                most_common_cls = rec["cls"].most_common(1)[0][0]
            veh_rows.append({
                "roi_idx": roi_idx,
                "track_id": track_id,
                "class_id": most_common_cls,
                "t_first": rec["t_first"],
                "t_last": rec["t_last"],
                "n_obs": rec["n"],
                "x_first_m": float(rec["p_first"][0]),
                "y_first_m": float(rec["p_first"][1]),
                "x_last_m": float(rec["p_last"][0]),
                "y_last_m": float(rec["p_last"][1]),
                "counted_new": 1 if rec["counted_new"] else 0,
            })
        veh_cols = ["roi_idx", "track_id", "class_id", "t_first", "t_last",
                    "n_obs", "x_first_m", "y_first_m", "x_last_m", "y_last_m", "counted_new"]
        veh_df = pd.DataFrame(veh_rows, columns=veh_cols)
        veh_path = out_dir / f"{ts_str}_veh.csv.gz"
        tmp_veh = out_dir / f"{ts_str}_veh.csv.gz.tmp"
        veh_df.to_csv(tmp_veh, index=False, compression="gzip")
        os.replace(tmp_veh, veh_path)

        if self._queue_zones:
            q_cols = ["t", "roi_idx", "track_id", "class_id", "x_m", "y_m", "speed_kmh"]
            q_df = pd.DataFrame(self._q_rows, columns=q_cols)
            q_path = out_dir / f"{ts_str}_q.csv.gz"
            tmp_q = out_dir / f"{ts_str}_q.csv.gz.tmp"
            q_df.to_csv(tmp_q, index=False, compression="gzip")
            os.replace(tmp_q, q_path)

        meta_data = {
            "lo_sec": float(self.lo_sec),
            "hi_sec": float(self.hi_sec),
            "frames": int(frames),
            "covered_sec": float(frames / TARGET_FPS),
            "n_ticks": int(self._n_ticks),
            "target_fps": TARGET_FPS,
            "queue_rois": list(self._queue_rois),
        }
        meta_path = out_dir / f"{ts_str}_meta.json"
        tmp_meta = out_dir / f"{ts_str}_meta.json.tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        os.replace(tmp_meta, meta_path)


def _proc_size(width: int, height: int):
    """(scale, ukuran_output|None) untuk resize frame di reader.
    `scale` = ROI_REF_WIDTH / lebar_proses, dipakai memetakan koordinat
    deteksi (di ruang frame yang sudah di-resize, atau ruang frame sumber
    kalau tidak di-resize) ke ruang koordinat ROI/matrix perspektif, yang
    dibuat pada frame 1280x720 -- BUKAN relatif ke lebar file sumber, supaya
    arsip yang sudah di-downscale (misal 640 lebar) tetap konsisten dengan
    ROI."""
    if not PROC_WIDTH or width <= PROC_WIDTH:
        return ROI_REF_WIDTH / width, None
    out_w = int(PROC_WIDTH)
    out_h = max(2, int(round(height * PROC_WIDTH / width)))
    return ROI_REF_WIDTH / out_w, (out_w, out_h)


def _apply_detections(detections, video_time_sec, scale, tracker, counted_ids,
                      vehicle_history, latest_speeds, volume_counts,
                      roi_polys, roi_boxes, matrices, target_class_ids,
                      window_log=None):
    """Update tracking state untuk satu frame setelah batch inference selesai.
    `video_time_sec` sekarang WAKTU ABSOLUT (detik sejak tengah malam), jadi
    kecepatan dan purging riwayat tetap benar antar segmen .ts.
    `scale` memetakan koordinat deteksi balik ke frame asli (lihat _proc_size)."""
    detections = tracker.update_with_detections(detections)

    if vehicle_history:
        cutoff = video_time_sec - HISTORY_MAX_AGE_SEC
        for tid in [t for t, h in vehicle_history.items() if h["time_sec"] < cutoff]:
            del vehicle_history[tid]

    if detections.tracker_id is None or len(detections.tracker_id) == 0:
        if window_log is not None:
            window_log.observe_frame(video_time_sec,
                                     np.zeros((0, 2), dtype=np.float64),
                                     np.zeros(0, dtype=int),
                                     np.zeros(0, dtype=int))
        return detections

    track_ids = np.asarray(detections.tracker_id).astype(int)
    class_ids = np.asarray(detections.class_id).astype(int)
    xyxy = np.asarray(detections.xyxy, dtype=np.float64)

    keep = np.isin(class_ids, target_class_ids)
    if not keep.any():
        if window_log is not None:
            window_log.observe_frame(video_time_sec,
                                     np.zeros((0, 2), dtype=np.float64),
                                     np.zeros(0, dtype=int),
                                     np.zeros(0, dtype=int))
        return detections
    xyxy, track_ids, class_ids = xyxy[keep], track_ids[keep], class_ids[keep]

    # Titik acuan: bottom-center. Deteksi dari model berada di ruang frame
    # yang sudah di-resize (PROC_WIDTH), jadi dikali `scale` (= lebar_asli /
    # lebar_proses) untuk kembali ke koordinat frame asli tempat ROI dibuat.
    points = np.stack(((xyxy[:, 0] + xyxy[:, 2]) * 0.5 * scale,
                       xyxy[:, 3] * scale), axis=1)

    if window_log is not None:
        window_log.observe_frame(video_time_sec, points, track_ids, class_ids)
    # Assign ROI. Prefilter bbox lebih dulu supaya pointPolygonTest (mahal,
    # per titik) hanya dipanggil untuk deteksi yang memang dekat ROI.
    roi_of = np.full(len(points), -1, dtype=int)
    for idx, poly in enumerate(roi_polys):
        x0, y0, x1, y1 = roi_boxes[idx]
        for i in np.nonzero(roi_of < 0)[0]:
            x, y = float(points[i, 0]), float(points[i, 1])
            if x0 <= x <= x1 and y0 <= y <= y1 and cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                roi_of[i] = idx

    # Transformasi perspektif per ROI: satu panggilan batch per ROI, bukan
    # satu panggilan per deteksi.
    positions = np.zeros((len(points), 2), dtype=np.float64)
    for idx in range(len(roi_polys)):
        sel = np.nonzero(roi_of == idx)[0]
        if sel.size == 0:
            continue
        src = points[sel].reshape(-1, 1, 2).astype(np.float32)
        positions[sel] = cv2.perspectiveTransform(src, matrices[idx]).reshape(-1, 2)

    for i in range(len(points)):
        assigned_roi = int(roi_of[i])
        if assigned_roi == -1:
            continue

        history_key = int(track_ids[i])
        class_id = int(class_ids[i])

        counted_new = False
        if history_key not in counted_ids[assigned_roi]:
            counted_ids[assigned_roi].add(history_key)
            volume_counts[assigned_roi][class_id] += 1
            counted_new = True

        pos_m = positions[i]

        if window_log is not None:
            window_log.roi_obs(assigned_roi, history_key, class_id, video_time_sec, pos_m, counted_new)
        prev = vehicle_history.get(history_key)
        if prev and prev["roi_idx"] == assigned_roi:
            dt = video_time_sec - prev["time_sec"]
            if dt > 0:
                dist_m = float(np.linalg.norm(pos_m - prev["pos_m"]))
                speed_kmh = (dist_m / dt) * 3.6
                if MIN_SPEED_KMH <= speed_kmh <= MAX_SPEED_KMH and math.isfinite(speed_kmh):
                    entry = latest_speeds.setdefault(history_key, {"roi_idx": assigned_roi, "by_roi": {}})
                    cur = entry["by_roi"].get(assigned_roi)
                    entry["by_roi"][assigned_roi] = {
                        "speed": speed_kmh if cur is None else cur["speed"] * 0.7 + speed_kmh * 0.3,
                        "t": video_time_sec,
                    }
                    entry["roi_idx"] = assigned_roi

        vehicle_history[history_key] = {
            "pos_m": pos_m,
            "time_sec": video_time_sec,
            "roi_idx": assigned_roi,
        }
    return detections


# -------------------- segmen & window helpers --------------------

def _parse_hhmmss(s: str):
    """'HHMMSS' -> detik sejak tengah malam. None kalau formatnya tidak valid."""
    s = s.strip()
    if len(s) == 6 and s.isdigit():
        h, m, sec = int(s[:2]), int(s[2:4]), int(s[4:])
        if h < 24 and m < 60 and sec < 60:
            return h * 3600 + m * 60 + sec
    return None


def build_segment_index(videos_dir: Path) -> list:
    """Index semua segmen .ts: [{start_sec, path}] terurut naik.
    Jam mulai = token terakhir nama file (konvensi yang sama dengan
    parse_ts_filename di script Colab lama)."""
    segments = []
    if not videos_dir.exists():
        return segments
    skipped = 0
    by_stem: dict = {}
    for v in videos_dir.iterdir():
        suffix = v.suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            continue
        sec = _parse_hhmmss(v.stem.split("_")[-1])
        if sec is None:
            skipped += 1
            continue
        # Mid-conversion state: file asli & hasil konversi berdampingan
        # (stem sama, ekstensi beda) -> pertahankan yang lebih diprioritaskan
        # di VIDEO_SUFFIXES (asli dulu).
        existing = by_stem.get(v.stem)
        if existing is not None:
            if VIDEO_SUFFIXES.index(suffix) >= VIDEO_SUFFIXES.index(existing["path"].suffix.lower()):
                continue
        by_stem[v.stem] = {"start_sec": sec, "path": v}
    segments = list(by_stem.values())
    if skipped:
        print(f"  [!] {skipped} file di {videos_dir} tanpa token waktu _HHMMSS, diabaikan.")
    segments.sort(key=lambda s: s["start_sec"])
    return segments


def select_window_segments(segments, seg_start_secs, lo_sec, hi_sec):
    """Segmen yang menyentuh window [lo_sec, hi_sec).
    Segmen terakhir yang MULAI sebelum lo_sec ikut disertakan (frame di luar
    rentang dibuang oleh reader) supaya tidak ada frame yang hilang di
    perbatasan window; segmen yang mulai >= hi_sec tidak ikut."""
    lo = max(0, bisect.bisect_left(seg_start_secs, lo_sec) - 1)
    hi = bisect.bisect_left(seg_start_secs, hi_sec)
    return segments[lo:hi]


class InferClient:
    """Pengganti `model.predict` di dalam worker decode/tracking.

    Frame ditulis ke slot `shared_memory` milik worker ini (tanpa copy lewat
    pipe), request berisi indeks slot dikirim ke `req_queue` yang di-`get()`
    bersama oleh semua inference server (load balancing otomatis), dan
    response dikembalikan ke `resp_queue` milik worker ini.
    """

    def __init__(self, worker_id: int, req_queue, resp_queue, shm_name: str):
        self.worker_id = worker_id
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.free_slots = queue.Queue()
        for i in range(INFER_SLOTS):
            self.free_slots.put(i)
        self._next_req_id = 0
        self._arrived: dict = {}   # response yang datang tidak berurutan

    def acquire_slot(self) -> int:
        return self.free_slots.get()            # blocking = backpressure reader

    def write_frame(self, slot: int, frame) -> None:
        if frame.nbytes > FRAME_SLOT_BYTES:
            raise ValueError(f"frame {frame.shape} = {frame.nbytes} B > FRAME_SLOT_BYTES "
                             f"{FRAME_SLOT_BYTES}; naikkan PROC_WIDTH/FRAME_SLOT_BYTES")
        view = np.ndarray(frame.shape, dtype=np.uint8, buffer=self.shm.buf,
                          offset=slot * FRAME_SLOT_BYTES)
        np.copyto(view, frame)

    def submit(self, height: int, width: int, slots: list) -> int:
        req_id = self._next_req_id
        self._next_req_id += 1
        self.req_queue.put((self.worker_id, req_id, height, width, list(slots)))
        return req_id

    def fetch(self, req_id: int):
        """Tunggu response `req_id`; slot dibebaskan segera setiap response datang."""
        while req_id not in self._arrived:
            try:
                got, slots, payload = self.resp_queue.get(timeout=INFER_RESPONSE_TIMEOUT_SEC)
            except queue.Empty:
                raise RuntimeError(f"inference server tidak merespons "
                                   f"{INFER_RESPONSE_TIMEOUT_SEC}s (req {req_id})") from None
            for s in slots:
                self.free_slots.put(s)
            self._arrived[got] = payload
        payload = self._arrived.pop(req_id)
        if isinstance(payload, str):
            raise RuntimeError(f"inference server gagal: {payload}")
        return payload

    def close(self) -> None:
        self.shm.close()


def _to_detections(xyxy, conf, cls):
    if xyxy.size == 0:
        return sv.Detections.empty()
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls.astype(int))


@contextlib.contextmanager
def _mute_native_stderr():
    """Redirect OS fd 2 (native stderr) ke devnull selama blok `with`.

    OpenCV/ffmpeg (cap_ffmpeg_impl.hpp) mencetak error decode H.264 ("error
    while decoding MB ...", "reference picture missing", "mmco: unref short
    failure", dst) langsung lewat `vfprintf(stderr, ...)` di level C, bukan
    lewat `sys.stderr` Python maupun logger OpenCV. `OPENCV_FFMPEG_LOGLEVEL`
    didokumentasikan untuk mengendalikan ini tapi terverifikasi TIDAK
    menyupresi pada build opencv-python yang dipakai (nvidia-smi/pip prebuilt
    ffmpeg 7.x) -- diuji langsung dengan segmen korup, pesan tetap muncul.
    Segmen korup itu sendiri sudah dilaporkan lewat truncated_segments /
    failed_segments di skip_report; pesan mentah ffmpeg cuma noise di log.

    Redirect fd secara OS-level (bukan patch sys.stderr) supaya frame yang
    memang gagal di-decode tetap tercatat, dan traceback Python di luar
    blok `with` ini (dicetak setelah context exit) tetap tampil normal."""
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def process_window(segments, lo_sec, hi_sec, rois_data, client, tracker, counted_ids,
                   matrices, roi_polys, roi_boxes, target_class_ids,
                   window_log=None, frame_hook=None):
    """
    Proses SEMUA segmen .ts satu window sebagai satu aliran kontinu:
    - reader thread berpindah antar file .ts; timestamp frame =
      start_segmen (dari nama file) + frame_idx/fps -> waktu absolut
    - sampling tepat TARGET_FPS (akumulasi fraksional, bukan stride integer)
    - frame di luar [lo_sec, hi_sec) tetap di-decode (wajib untuk H.264) tapi
      tidak dikirim, sehingga segmen yang menyeberang batas window terbaca
      penuh di satu sisi saja -> coverage utuh tanpa dobel
    - reader thread menulis frame ke shared memory & mengirim request ke
      inference server lewat `client`; thread pemanggil mengonsumsi response
      FIFO (per req_id) supaya state tracker tetap monoton
    """
    vehicle_history = {}
    latest_speeds = {}
    volume_counts = {idx: {cid: 0 for cid in TARGET_CLASSES} for idx in range(len(rois_data))}

    inflight = queue.Queue()
    _SENTINEL = object()
    stats = {"failed_segments": [], "truncated_segments": [],
             "fps_fallback": 0, "frames": 0}
    errors: list = []

    def submit(batch, shape):
        req_id = client.submit(shape[0], shape[1], [b[0] for b in batch])
        inflight.put((req_id, [b[1:] for b in batch]))

    def reader():
        stopped_early = False
        batch = []
        batch_shape = None
        try:
            with _mute_native_stderr():
                for seg in segments:
                    cap = cv2.VideoCapture(str(seg["path"]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, DECODE_THREADS])
                    if not cap.isOpened():
                        stats["failed_segments"].append(seg["path"].name)
                        continue
                    native_fps = cap.get(cv2.CAP_PROP_FPS)
                    if native_fps <= 0 or not math.isfinite(native_fps):
                        stats["fps_fallback"] += 1
                        native_fps = FALLBACK_FPS
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or PROC_WIDTH or 1
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or PROC_WIDTH or 1
                    scale, out_size = _proc_size(width, height)
                    expected_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    step = native_fps / TARGET_FPS
                    next_sample = 1.0
                    frame_idx = 0
                    while True:
                        frame_idx += 1
                        if frame_idx < next_sample:
                            if not cap.grab():
                                break
                            continue
                        next_sample += step
                        t = seg["start_sec"] + (frame_idx - 1) / native_fps
                        if t >= hi_sec:
                            stopped_early = True   # segmen berikutnya pasti >= hi_sec
                            break
                        if t < lo_sec:             # sudah diproses window sebelumnya
                            if not cap.grab():     # tetap decode (rantai P-frame), tanpa konversi BGR
                                break
                            continue
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if out_size is not None:
                            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
                        shape = frame.shape[:2]
                        if batch and shape != batch_shape:
                            submit(batch, batch_shape)   # resolusi segmen berubah -> flush
                            batch = []
                        batch_shape = shape
                        slot = client.acquire_slot()
                        client.write_frame(slot, frame)
                        batch.append((slot, t, scale, seg["path"].name, frame_idx))
                        stats["frames"] += 1
                        if len(batch) >= INFER_BATCH_SIZE:
                            submit(batch, batch_shape)
                            batch = []
                    cap.release()
                    if stopped_early:
                        break
                    if expected_frames > 0 and frame_idx < expected_frames - 1:
                        # Bukan gagal dibuka, tapi berhenti di tengah file:
                        # sisanya hilang tanpa error -> laporkan.
                        stats["truncated_segments"].append(seg["path"].name)
            if batch:
                submit(batch, batch_shape)
        except Exception as e:                       # noqa: BLE001
            errors.append(e)
        finally:
            inflight.put(_SENTINEL)

    reader_thread = threading.Thread(target=reader, name="fe-reader", daemon=True)
    reader_thread.start()

    # Batch dikonsumsi FIFO (per req_id) -> waktu tetap naik meski ada lebih
    # dari satu inference server yang membalas tidak berurutan.
    try:
        while True:
            item = inflight.get()
            if item is _SENTINEL:
                break
            req_id, metas = item
            payload = client.fetch(req_id)
            for (xyxy, conf, cls), (t, scale, seg_name, frame_idx) in zip(payload, metas):
                dets = _apply_detections(_to_detections(xyxy, conf, cls), t, scale, tracker,
                                         counted_ids, vehicle_history, latest_speeds,
                                         volume_counts, roi_polys, roi_boxes, matrices,
                                         target_class_ids, window_log=window_log)
                if frame_hook is not None:
                    frame_hook(t, scale, seg_name, frame_idx, dets)
    except Exception as e:                           # noqa: BLE001
        errors.append(e)
        # Postprocess berhenti: kuras sisa in-flight dan ambil response-nya
        # supaya slot kembali dan reader tidak memblokir di acquire_slot()
        # selamanya (deadlock).
        while True:
            item = inflight.get()
            if item is _SENTINEL:
                break
            try:
                client.fetch(item[0])
            except Exception:                        # noqa: BLE001
                pass
    finally:
        reader_thread.join()

    if errors:
        raise errors[0]

    if stats["fps_fallback"]:
        print(f"  [!] {stats['fps_fallback']}/{len(segments)} segmen fps tidak valid "
              f"-> pakai fallback {FALLBACK_FPS}")

    avg_speeds = {idx: 0.0 for idx in range(len(rois_data))}
    for idx in range(len(rois_data)):
        speeds = [v["by_roi"][idx]["speed"] for v in latest_speeds.values() if idx in v["by_roi"]]
        if speeds:
            avg_speeds[idx] = sum(speeds) / len(speeds)

    return {"volume": volume_counts, "avg_speeds": avg_speeds,
            "latest_speeds": latest_speeds,
            "failed_segments": stats["failed_segments"],
            "truncated_segments": stats["truncated_segments"],
            "frames": stats["frames"], "segments": len(segments)}


def _write_csv_atomic(df, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def process_cctv(cctv_dir: Path, rois_data: list, client, checkpoint: dict,
                 date_name: str, skip_report: list):
    metadata_dir = cctv_dir / METADATA_SUBDIR
    videos_dir = cctv_dir / VIDEOS_SUBDIR

    csv_files = list(metadata_dir.glob("*.csv")) if metadata_dir.exists() else []
    csv_files = [p for p in csv_files if not p.stem.endswith(OUTPUT_SUFFIX)]
    if not csv_files:
        print(f"  [!] Tidak ada CSV metadata di {metadata_dir}. Skip.")
        skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                            "timestampsafe": None, "reason": "no_metadata_csv"})
        return checkpoint

    csv_path = csv_files[0]
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXTRACTED_DIR / f"{csv_path.stem}{OUTPUT_SUFFIX}.csv"
    # Resume: kalau output sudah ada, lanjutkan dari situ; metadata asli hanya dibaca.
    df = pd.read_csv(out_path) if out_path.exists() else pd.read_csv(csv_path)

    n_rois = len(rois_data)
    for i in range(n_rois):
        if f"ROI_{i+1}_AvgSpeed_Kmh" not in df.columns:
            df[f"ROI_{i+1}_AvgSpeed_Kmh"] = 0.0
        for cid, cname in TARGET_CLASSES.items():
            col = f"ROI_{i+1}_Vol_{cname}"
            if col not in df.columns:
                df[col] = 0

    # Index segmen sekali di awal (pengganti find_video + video_index lama)
    segments = build_segment_index(videos_dir)
    if not segments:
        print(f"  [!] Tidak ada segmen .ts di {videos_dir}. Skip.")
        skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                            "timestampsafe": None, "reason": "no_ts_segments"})
        return checkpoint
    seg_start_secs = [s["start_sec"] for s in segments]

    matrices = get_perspective_matrices(rois_data)
    roi_polys = [np.array(roi["points"], dtype=np.int32) for roi in rois_data]
    roi_boxes = [(int(p[:, 0].min()), int(p[:, 1].min()),
                  int(p[:, 0].max()), int(p[:, 1].max())) for p in roi_polys]
    target_class_ids = list(TARGET_CLASSES.keys())

    # ByteTrack dan counted_ids di-share lintas semua window dalam CCTV ini
    # (semantik sama dengan versi sebelumnya: volume per baris = kendaraan
    # yang PERTAMA kali terhitung di baris itu; total sehari = unik).
    shared_tracker = sv.ByteTrack(frame_rate=TARGET_FPS)
    shared_counted_ids = {idx: set() for idx in range(n_rois)}

    # Kumpulkan baris metadata valid, terurut waktu (tracker harus jalan maju)
    rows = []
    for row_idx, row in df.iterrows():
        ts_raw = str(row.get("timestampsafe", "")).strip()
        if not ts_raw or ts_raw.lower() in ("nan", "none"):
            continue
        ts_str = ts_raw.zfill(6)   # pandas bisa buang leading zero kalau int
        sec = _parse_hhmmss(ts_str)
        if sec is None:
            skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                                "timestampsafe": ts_raw, "reason": "bad_timestampsafe"})
            continue
        rows.append((row_idx, ts_str, sec))
    rows.sort(key=lambda x: x[2])

    # buang timestampsafe duplikat (pertahankan kemunculan pertama)
    deduped, seen = [], set()
    for r in rows:
        if r[1] in seen:
            continue
        seen.add(r[1])
        deduped.append(r)
    rows = deduped

    if not rows:
        print(f"  [!] Tidak ada baris metadata valid di {csv_path.name}. Skip.")
        return checkpoint

    print(f"  [*] {len(rows)} baris metadata, {len(segments)} segmen .ts -> {out_path.name}")

    processed = list(checkpoint.get("processed_rows", []))
    processed_set = set(processed)
    changed = False

    for k, (row_idx, ts_str, row_sec) in enumerate(
            tqdm(rows, desc="  rows", leave=False)):
        if _stop_requested():
            print(f"  [!] STOP diminta ({STOP_FILE}); berhenti sebelum window {ts_str}.")
            break
        if ts_str in processed_set:
            continue

        # Batas atas window: baris berikutnya, maksimal WINDOW_SECONDS
        # (logika [t_i, t_i+1) dari script Colab, plus pengaman).
        next_sec = rows[k + 1][2] if k + 1 < len(rows) else float("inf")
        hi_sec = min(next_sec, row_sec + WINDOW_SECONDS)

        # Ikutkan segmen yang MULAI sebelum row_sec tapi menyeberang ke dalam
        # window (frame-nya difilter oleh reader) -> tidak ada frame hilang.
        window_segments = select_window_segments(segments, seg_start_secs, row_sec, hi_sec)

        if not window_segments:
            # Jangan tandai processed: segmen mungkin belum tersinkron,
            # biar run berikutnya mencoba lagi.
            skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                                "timestampsafe": ts_str, "reason": "no_segments_in_window"})
            continue

        window_log = WindowLog(row_sec, hi_sec, rois_data)
        results = process_window(window_segments, row_sec, hi_sec, rois_data, client,
                                 shared_tracker, shared_counted_ids, matrices,
                                 roi_polys, roi_boxes, target_class_ids,
                                 window_log=window_log)

        if len(results["failed_segments"]) == len(window_segments):
            print(f"  [!] Semua segmen gagal dibuka untuk window {ts_str}, dilewati.")
            skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                                "timestampsafe": ts_str, "reason": "segments_open_failed"})
            continue

        if results["frames"] == 0:
            # Semua segmen ada tapi tidak ada frame di dalam rentang window
            # (mis. jam segmen tidak sinkron) -> jangan tandai selesai.
            skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                                "timestampsafe": ts_str, "reason": "no_frames_in_window"})
            continue

        if results["failed_segments"]:
            # Sebagian segmen corrupt -> tetap diproses & ditandai selesai
            # (retry tidak akan memperbaiki file corrupt), tapi dilaporkan.
            skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                                "timestampsafe": ts_str,
                                "reason": f"partial_segments_failed: "
                                          f"{len(results['failed_segments'])}/{len(window_segments)}"})

        if results["truncated_segments"]:
            # Segmen terbaca sebagian di tengah file -> volume bisa kurang.
            skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                                "timestampsafe": ts_str,
                                "reason": f"truncated_segments: "
                                          f"{len(results['truncated_segments'])}/{len(window_segments)}"})

        vol = results["volume"]
        spd = results["avg_speeds"]
        cols, vals = [], []
        for roi_idx in range(n_rois):
            cols.append(f"ROI_{roi_idx+1}_AvgSpeed_Kmh")
            vals.append(round(spd[roi_idx], 2))
            for cid, cname in TARGET_CLASSES.items():
                cols.append(f"ROI_{roi_idx+1}_Vol_{cname}")
                vals.append(int(vol[roi_idx][cid]))
        df.loc[row_idx, cols] = vals

        processed.append(ts_str)
        processed_set.add(ts_str)
        checkpoint["processed_rows"] = processed
        changed = True

        # Save CSV dan checkpoint setelah tiap window agar resume bisa jalan.
        # Tulis atomik (tmp + replace) supaya crash tidak meninggalkan file separuh.
        # Tulis track log sebelum CSV dan checkpoint
        window_log.write(TRACKLOG_DIR / date_name / cctv_dir.name, ts_str, results["frames"])

        # Save CSV dan checkpoint setelah tiap window agar resume bisa jalan.
        # Tulis atomik (tmp + replace) supaya crash tidak meninggalkan file separuh.
        _write_csv_atomic(df, out_path)
        _save_checkpoint(cctv_dir, checkpoint)

    if not changed:
        print(f"  [=] Semua baris sudah diproses (dari checkpoint).")

    return checkpoint


# -------------------- checkpoint helpers --------------------

def _checkpoint_path(cctv_dir: Path) -> Path:
    return cctv_dir / ".extraction_checkpoint.json"

def _load_checkpoint(cctv_dir: Path) -> dict:
    p = _checkpoint_path(cctv_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"processed_rows": []}

def _save_checkpoint(cctv_dir: Path, data: dict):
    p = _checkpoint_path(cctv_dir)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)

def _stop_requested() -> bool:
    return STOP_FILE.exists()

# ------------------------------------------------------------


def _limit_threads(cv_threads: int, torch_threads: int):
    """Batasi thread internal OpenCV/torch per proses; tanpa ini setiap
    proses membuka sejumlah core thread dan 6-8 proses saling berebut CPU."""
    try:
        cv2.setNumThreads(cv_threads)
        torch.set_num_threads(torch_threads)
        return cv_threads
    except Exception as e:              # noqa: BLE001
        print(f"  [!] Gagal membatasi thread ({e})")
        return None


def _free_vram_mb():
    """VRAM bebas (MiB) via nvidia-smi. `torch.cuda.mem_get_info()` sengaja
    dihindari karena membuat konteks CUDA di proses induk (~300 MB terbuang
    selama server jalan). None = tidak terdeteksi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:                   # noqa: BLE001
        return None


def _letterbox_pad(h: int, w: int, imgsz: int, stride: int, auto: bool):
    """(top, bottom, left, right) persis seperti ultralytics LetterBox(center=True) untuk kasus
    rasio skala 1.0 (max(h, w) == imgsz): tidak ada resize, hanya padding nilai 114."""
    dw, dh = imgsz - w, imgsz - h
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    return round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)


def _fast_detect(backend, device, frames: list, pinned, pad: tuple, cls_t, conf: float, iou: float,
                 max_det: int) -> list:
    """Pengganti model.predict untuk satu batch frame BGR uint8 berukuran sama.
    Output identik dengan DetectionPredictor (letterbox+NMS+scale_boxes) tapi tanpa objek
    Results, tanpa loop Python per gambar di luar torchvision.ops.nms, dan satu .cpu() per batch."""
    n = len(frames)
    np.stack(frames, out=pinned.numpy()[:n])
    x = pinned[:n].to(device, non_blocking=True).permute(0, 3, 1, 2)
    top, bottom, left, right = pad
    x = torch.nn.functional.pad(x, (left, right, top, bottom), value=114)
    x = x.flip(1).contiguous()
    x = (x.half() if backend.fp16 else x.float()).div_(255)
    with torch.inference_mode():
        y = backend(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        p = y.transpose(-1, -2)
        p[..., :4] = yolo_ops.xywh2xyxy(p[..., :4])
        score, j = p[..., 4:].max(-1)
        keep = (score > conf) & torch.isin(j, cls_t)
        bi, ai = keep.nonzero(as_tuple=True)
        box = p[bi, ai, :4]
        sc = score[bi, ai]
        jf = j[bi, ai].float()
        off = box + (jf * NMS_MAX_WH)[:, None]
        counts = torch.bincount(bi, minlength=n).tolist()
        kept = []
        sizes = []
        s = 0
        for c in counts:
            if c > 0:
                k = torchvision.ops.nms(off[s:s + c], sc[s:s + c], iou)[:max_det] + s
                kept.append(k)
                sizes.append(len(k))
            else:
                sizes.append(0)
            s += c
        k = torch.cat(kept) if kept else torch.zeros(0, dtype=torch.long, device=device)
        det = torch.cat((box[k], sc[k, None], jf[k, None]), 1)
        det[:, :4] = yolo_ops.scale_boxes(x.shape[2:], det[:, :4], frames[0].shape[:2])
        arr = det.cpu().numpy()
    payload = []
    offset = 0
    for sz in sizes:
        chunk = arr[offset:offset + sz]
        payload.append((chunk[:, :4].astype(np.float32),
                        chunk[:, 4].astype(np.float32),
                        chunk[:, 5].astype(np.int32)))
        offset += sz
    return payload


def _child_init():
    """Dipanggil di awal setiap proses anak.
    1) Abaikan SIGINT: Ctrl-C di terminal mengenai seluruh process group; stop halus
       dikendalikan lewat STOP_FILE, bukan KeyboardInterrupt.
    2) Watchdog: bila proses induk mati (kill/crash, finally induk tidak jalan),
       anak keluar sendiri agar tidak jadi orphan."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = mp.parent_process()
    if parent is None:
        return

    def _watch():
        parent.join()                   # blok sampai induk berakhir
        os._exit(1)

    threading.Thread(target=_watch, name="fe-parent-watchdog", daemon=True).start()


class Detector:
    """Encapsulation of YOLO inference (TensorRT engine or .pt fallback) with fast letterbox/NMS path."""
    def __init__(self, device: str):
        self.device = device
        self.target_class_ids = list(TARGET_CLASSES)
        self.model = YOLO(str(MODEL_WEIGHTS))
        if str(MODEL_WEIGHTS).endswith(".pt"):
            self.model.to(device)
        self.model.predict([np.zeros((INFER_IMGSZ // 2, INFER_IMGSZ, 3), dtype=np.uint8)],
                           classes=self.target_class_ids, **PREDICT_KWARGS)     # warmup
        self.backend = self.model.predictor.model
        self.args = self.model.predictor.args
        self.infer_device = self.model.predictor.device
        self.fast_ok = (torchvision is not None and yolo_ops is not None
                        and not getattr(self.backend, "end2end", False) and not self.args.agnostic_nms)
        self.auto = self.backend.format == "pt" or (getattr(self.backend, "dynamic", False) and self.backend.format != "imx")
        self.cls_t = torch.tensor(self.target_class_ids, device=self.infer_device)
        self._pinned_by_shape = {}

    def detect(self, frames: list) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not frames:
            return []
        h, w = frames[0].shape[:2]
        if self.fast_ok and max(h, w) == INFER_IMGSZ:
            pinned = self._pinned_by_shape.get((h, w))
            if pinned is None:
                pinned = self._pinned_by_shape[(h, w)] = torch.empty(
                    (INFER_BATCH_SIZE, h, w, 3), dtype=torch.uint8
                ).pin_memory()
            pad = _letterbox_pad(h, w, INFER_IMGSZ, int(self.backend.stride), self.auto)
            return _fast_detect(self.backend, self.infer_device, frames, pinned, pad, self.cls_t,
                                self.args.conf, self.args.iou, self.args.max_det)
        else:
            results = self.model.predict(frames, classes=self.target_class_ids, **PREDICT_KWARGS)
            return [(r.boxes.xyxy.cpu().numpy().astype(np.float32),
                     r.boxes.conf.cpu().numpy().astype(np.float32),
                     r.boxes.cls.cpu().numpy().astype(np.int32)) for r in results]


def _infer_server(server_id: int, req_queue: mp.Queue, resp_queues: list,
                  shm_names: list):
    """Satu proses = satu konteks CUDA + satu model. Melayani request dari
    semua worker lewat `req_queue` bersama (load balancing otomatis lewat
    `Queue.get()`), membalas lewat `resp_queues[worker_id]`."""
    _child_init()
    _limit_threads(SERVER_CV_THREADS, SERVER_CV_THREADS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = None
    try:
        det = Detector(device)
        print(f"[server {server_id}] Model siap di {device.upper()}")
        print(f"[server {server_id}] Jalur inferensi cepat: "
              f"{'aktif' if det.fast_ok else 'nonaktif (fallback model.predict)'}")
    except Exception as e:                                            # noqa: BLE001
        det = None        # YOLO() lazy: objek setengah jadi akan memuat ulang engine di tiap predict
        print(f"[server {server_id}] Gagal memuat model {MODEL_WEIGHTS}: {e}")

    shms: dict = {}
    while True:
        item = req_queue.get()
        if item is None:                      # satu sentinel per server
            break
        wid, req_id, h, w, slots = item
        try:
            if det is None:
                raise RuntimeError(f"model gagal dimuat di server {server_id}")
            shm = shms.get(wid)
            if shm is None:
                shm = shms[wid] = shared_memory.SharedMemory(name=shm_names[wid])
            frames = [np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf,
                                 offset=s * FRAME_SLOT_BYTES) for s in slots]
            payload = det.detect(frames)
        except Exception as e:                                        # noqa: BLE001
            payload = f"{type(e).__name__}: {e}"
        resp_queues[wid].put((req_id, slots, payload))
    for shm in shms.values():
        shm.close()


def _worker(worker_id: int, job_queue: mp.Queue, done_queue: mp.Queue,
            skip_report: list, req_queue: mp.Queue, resp_queue: mp.Queue,
            shm_name: str):
    _child_init()
    _limit_threads(WORKER_CV_THREADS, WORKER_CV_THREADS)
    print(f"[worker {worker_id}] Siap (decode+tracking, inferensi lewat server)")
    client = InferClient(worker_id, req_queue, resp_queue, shm_name)

    while True:
        item = job_queue.get()          # blocking: get_nowait bisa Empty palsu
        if item is None:                # sentinel per worker
            break
        if _stop_requested():
            print(f"[worker {worker_id}] STOP diminta; {item[2]}/{Path(item[0]).name} dilewati")
            continue

        cctv_dir_str, rois_data, date_name = item
        cctv_dir = Path(cctv_dir_str)
        cctv_name = cctv_dir.name
        print(f"\n[worker {worker_id}] Memproses {date_name}/{cctv_name} ({len(rois_data)} ROI)")

        try:
            checkpoint = _load_checkpoint(cctv_dir)
            n_done = len(checkpoint.get("processed_rows", []))
            if n_done:
                print(f"[worker {worker_id}]   Resume dari checkpoint ({n_done} baris sudah selesai)")
            process_cctv(cctv_dir, rois_data, client, checkpoint, date_name, skip_report)
            print(f"[worker {worker_id}]   Selesai: {date_name}/{cctv_name}")
        except Exception as e:                                 # noqa: BLE001
            print(f"[worker {worker_id}]   Error pada {date_name}/{cctv_name}: {e}")
            import traceback; traceback.print_exc()
            skip_report.append({"date": date_name, "cctv": cctv_name,
                                "timestampsafe": None, "reason": f"exception: {e}"})
        finally:
            done_queue.put((cctv_dir_str, date_name))

    client.close()


def _acquire_instance_lock():
    """Kembalikan handle file yang memegang lock, atau None bila instance lain aktif.
    Handle HARUS tetap direferensikan selama proses hidup."""
    f = open(INSTANCE_LOCK_PATH, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _discover_date_dirs() -> list[Path]:
    root = DATASET_ROOT
    if not root.exists():
        return []
    if DATASET_DATES:
        return [root / d for d in DATASET_DATES]
    return [d for d in sorted(root.iterdir()) if d.is_dir()]


def main():
    instance_lock = _acquire_instance_lock()
    if instance_lock is None:
        print(f"[!] feature_extractor.py lain sedang berjalan (lock: {INSTANCE_LOCK_PATH}). "
              "Jalankan satu instance saja: dua instance berebut VRAM GPU dan gagal OOM.")
        raise SystemExit(3)
    if PID_FILE is not None:
        PID_FILE.write_text(str(os.getpid()))
        # Dijalankan oleh run_pipeline.sh: stop halus lewat STOP_FILE; Ctrl-C tidak boleh membunuh proses.
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    date_dirs = _discover_date_dirs()
    if not date_dirs:
        print(f"[!] Tidak ada folder tanggal ditemukan di {DATASET_ROOT}")
        return

    manager = mp.Manager()
    skip_report = manager.list()

    if not Path(MODEL_WEIGHTS).exists():
        print(f"[!] Bobot model tidak ditemukan: {MODEL_WEIGHTS}")
        skip_report.append({"date": None, "cctv": None, "timestampsafe": None,
                            "reason": f"model_weights_not_found: {MODEL_WEIGHTS}"})
        _write_missing_report(skip_report)
        return

    jobs = []
    excluded = {c.lower() for c in EXCLUDED_CCTVS}
    for date_dir in date_dirs:
        date_name = date_dir.name
        if not date_dir.exists():
            print(f"[!] Folder tanggal tidak ditemukan: {date_dir}. Skip.")
            skip_report.append({"date": date_name, "cctv": None,
                                "timestampsafe": None, "reason": "date_dir_not_found"})
            continue

        cctv_dirs = [d for d in sorted(date_dir.iterdir()) if d.is_dir()]
        print(f"[*] [{date_name}] Ditemukan {len(cctv_dirs)} CCTV: {[d.name for d in cctv_dirs]}")

        for cctv_dir in cctv_dirs:
            if cctv_dir.name.lower() in excluded:
                print(f"[!] [{date_name}] {cctv_dir.name} masuk EXCLUDED_CCTVS. Skip.")
                continue

            roi_path = ROI_DIR / f"{cctv_dir.name}_roi.json"
            if not roi_path.exists():
                print(f"[!] [{date_name}] ROI tidak ditemukan untuk {cctv_dir.name} ({roi_path}). Skip.")
                skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                                    "timestampsafe": None, "reason": "roi_not_found"})
                continue
            with open(roi_path) as f:
                rois_data = json.load(f).get("rois", [])
            if not rois_data:
                print(f"[!] [{date_name}] ROI kosong untuk {cctv_dir.name}. Skip.")
                skip_report.append({"date": date_name, "cctv": cctv_dir.name,
                                    "timestampsafe": None, "reason": "roi_empty"})
                continue
            jobs.append((str(cctv_dir), rois_data, date_name))

    if not jobs:
        print("[!] Tidak ada CCTV valid untuk diproses.")
        _write_missing_report(skip_report)
        return

    logical = psutil.cpu_count(logical=True) or 2
    max_workers_env = os.environ.get("FE_WORKERS")
    max_workers = int(max_workers_env) if max_workers_env and max_workers_env.isdigit() else (logical // 2)
    n_workers = max(1, min(len(jobs), max_workers))
    if torch.cuda.is_available():
        free_mb = _free_vram_mb()
        by_vram = MAX_INFER_SERVERS if free_mb is None else int(free_mb // INFER_SERVER_VRAM_MB)
        n_servers = max(1, min(MAX_INFER_SERVERS, by_vram, math.ceil(n_workers / 3)))
    else:
        n_servers = 1
    print(f"[*] {n_workers} worker decode/tracking + {n_servers} inference server "
          f"({logical} logical core)")

    req_queue = mp.Queue()
    resp_queues = [mp.Queue() for _ in range(n_workers)]
    shms = [shared_memory.SharedMemory(create=True, size=INFER_SLOTS * FRAME_SLOT_BYTES)
            for _ in range(n_workers)]
    shm_names = [s.name for s in shms]

    job_queue = mp.Queue()
    done_queue = mp.Queue()
    for job in jobs:
        job_queue.put(job)
    for _ in range(n_workers):
        job_queue.put(None)             # satu sentinel per worker

    servers = []
    for i in range(n_servers):
        p = mp.Process(target=_infer_server, args=(i, req_queue, resp_queues, shm_names),
                       name=f"fe-server-{i}")
        p.daemon = True
        p.start()
        servers.append(p)

    procs = []
    for i in range(n_workers):
        p = mp.Process(target=_worker,
                       args=(i, job_queue, done_queue, skip_report,
                             req_queue, resp_queues[i], shm_names[i]),
                       name=f"fe-worker-{i}")
        p.daemon = True
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()

        for _ in range(n_servers):
            req_queue.put(None)             # satu sentinel per server
        for p in servers:
            p.join()
    finally:
        for p in procs + servers:
            if p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=1)
                except Exception:
                    pass
        for s in shms:                      # handle induk wajib hidup selama anak jalan (Windows)
            try:
                s.close()
                s.unlink()
            except Exception:
                pass

    # Cek penyelesaian eksplisit: worker yang mati diam-diam tidak boleh
    # membuat CCTV hilang tanpa jejak di laporan.
    done = set()
    while not done_queue.empty():
        key, _ = done_queue.get()
        done.add(key)
    for job in jobs:
        if job[0] not in done:
            print(f"[!] Tidak diproses oleh worker mana pun: {job[2]}/{Path(job[0]).name}")
            skip_report.append({"date": job[2], "cctv": Path(job[0]).name,
                                "timestampsafe": None, "reason": "not_processed_by_worker"})
    for i, p in enumerate(procs):
        if p.exitcode not in (0, None):
            skip_report.append({"date": None, "cctv": None, "timestampsafe": None,
                                "reason": f"worker_{i}_exitcode_{p.exitcode}"})
    for i, p in enumerate(servers):
        if p.exitcode not in (0, None):
            skip_report.append({"date": None, "cctv": None, "timestampsafe": None,
                                "reason": f"infer_server_{i}_exitcode_{p.exitcode}"})
    _write_missing_report(skip_report)
    print("\n[*] Seluruh ekstraksi fitur selesai.")


def _write_missing_report(skip_report):
    entries = list(skip_report)
    with open(MISSING_REPORT_PATH, "w") as f:
        json.dump(entries, f, indent=2)
    if entries:
        print(f"\n[!] {len(entries)} titik belum ter-feature-extract, "
              f"lihat detail di {MISSING_REPORT_PATH}:")
        for e in entries[:20]:
            print(f"    - {e['date']}/{e['cctv']} {e['timestampsafe'] or ''} ({e['reason']})")
        if len(entries) > 20:
            print(f"    ... dan {len(entries) - 20} lainnya")
    else:
        print(f"\n[*] Tidak ada titik yang terlewat. Laporan kosong ditulis ke {MISSING_REPORT_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)   # sama dengan Windows; fork setelah import torch/CUDA tidak aman di Linux (Colab)
    main()

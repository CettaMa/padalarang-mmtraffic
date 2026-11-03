"""Separate evaluation of the detector and the tracker (review point 6).

Subcommands
  detect      class-wise AP / P / R of model/best.pt and model/best.engine on the
              held-out test split of dataset-yolo.
  mot-export  run the production detector+ByteTrack on 12 sequences and write
              CVAT pre-annotation packages (MOT 1.1) under labeling/mot/<seq>/.
  mot-score   score corrected CVAT exports (gt_mot.zip) with motmetrics
              (MOTA, IDF1, ID switches); run in a separate numpy<2 env.

Heavy imports are local to each subcommand so that `mot-score` needs only
numpy + motmetrics.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "script"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

CLASS_NAMES = ["bus", "mobil", "motor", "truk"]
FRAME_W, FRAME_H = 1280, 720

# (seq, cctv_dir, display, start_hhmmss)
SEQUENCES = [
    ("kbp_day", "padalarang_-_jl_baru_kbp", "Jl. Baru KBP", "160000"),
    ("kbp_night", "padalarang_-_jl_baru_kbp", "Jl. Baru KBP", "200000"),
    ("g5bawah_day", "padalarang_-_gedong_5_bawah", "G5 Bawah", "160000"),
    ("g5bawah_night", "padalarang_-_gedong_5_bawah", "G5 Bawah", "200000"),
    ("g5atas_day", "padalarang_-_gedong_5_atas_-_stasiun", "G5 Atas", "160000"),
    ("g5atas_night", "padalarang_-_gedong_5_atas_-_stasiun", "G5 Atas", "200000"),
    ("gianttol_day", "padalarang_giant_-_tol", "Giant \u2013 Tol", "160000"),
    ("gianttol_night", "padalarang_giant_-_tol", "Giant \u2013 Tol", "200000"),
    ("gianttagog_day", "padalarang_-_giant_-_tagog", "Giant \u2013 Tagog", "160000"),
    ("gianttagog_night", "padalarang_-_giant_-_tagog", "Giant \u2013 Tagog", "200000"),
    ("tagogbdg_day", "tagog_padalarang_-_bandung", "Tagog-Bandung", "160000"),
    ("tagogbdg_night", "tagog_padalarang_-_bandung", "Tagog-Bandung", "200000"),
]


# --------------------------------------------------------------------------
# pure helpers (numpy only)
# --------------------------------------------------------------------------
def sampled_frame_indices(native_fps: float, n_native: int, target_fps: float = 10.0) -> list[int]:
    """1-based native frame indices kept by feature_extractor's sampler."""
    step = native_fps / target_fps
    next_sample = 1.0
    out = []
    for frame_idx in range(1, n_native + 1):
        if frame_idx >= next_sample:
            out.append(frame_idx)
            next_sample += step
    return out


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0][None], b[:, 1][None]
    bx2, by2 = bx1 + b[:, 2][None], by1 + b[:, 3][None]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def bottom_center_in(mask: np.ndarray, boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_xywh, float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0, bool)
    h, w = mask.shape[:2]
    xs = np.clip(np.floor(boxes[:, 0] + boxes[:, 2] / 2).astype(int), 0, w - 1)
    ys = np.clip(np.floor(boxes[:, 1] + boxes[:, 3]).astype(int), 0, h - 1)
    return mask[ys, xs].astype(bool)


def filter_frame(gt_xywh, gt_ids, pr_xywh, pr_ids, roi_mask, iou_thr=0.5):
    """ROI scoring with an ignore region (see plan, Step 1)."""
    gt_xywh = np.asarray(gt_xywh, float).reshape(-1, 4)
    pr_xywh = np.asarray(pr_xywh, float).reshape(-1, 4)
    gt_ids = np.asarray(gt_ids).reshape(-1)
    pr_ids = np.asarray(pr_ids).reshape(-1)
    g_in = bottom_center_in(roi_mask, gt_xywh)
    ge, gi = gt_xywh[g_in], gt_xywh[~g_in]
    max_e = iou_xywh(pr_xywh, ge).max(axis=1, initial=0.0) if len(pr_xywh) else np.zeros(0)
    max_i = iou_xywh(pr_xywh, gi).max(axis=1, initial=0.0) if len(pr_xywh) else np.zeros(0)
    p_in = bottom_center_in(roi_mask, pr_xywh)
    keep = (p_in | (max_e >= iou_thr)) & ~((max_i >= iou_thr) & (max_i > max_e))
    return ge, gt_ids[g_in], pr_xywh[keep], pr_ids[keep]


def match_operating_point(pred_xyxy, pred_conf, pred_cls, gt_xyxy, gt_cls, iou_thr=0.5):
    """Per-class greedy matching by descending confidence -> {cls: (tp, fp, fn)}."""
    pred_xyxy = np.asarray(pred_xyxy, float).reshape(-1, 4)
    gt_xyxy = np.asarray(gt_xyxy, float).reshape(-1, 4)
    pred_conf = np.asarray(pred_conf, float).reshape(-1)
    pred_cls = np.asarray(pred_cls).reshape(-1).astype(int)
    gt_cls = np.asarray(gt_cls).reshape(-1).astype(int)

    def xywh(b):
        return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]) if len(b) else b

    res = {}
    for c in range(len(CLASS_NAMES)):
        pi = np.where(pred_cls == c)[0]
        pi = pi[np.argsort(-pred_conf[pi], kind="stable")]
        gi = np.where(gt_cls == c)[0]
        iou = iou_xywh(xywh(pred_xyxy[pi]), xywh(gt_xyxy[gi]))
        used = np.zeros(len(gi), bool)
        tp = fp = 0
        for r in range(len(pi)):
            if len(gi):
                cand = np.where(used, -1.0, iou[r])
                j = int(np.argmax(cand))
                if cand[j] >= iou_thr:
                    used[j] = True
                    tp += 1
                    continue
            fp += 1
        res[c] = (tp, fp, int((~used).sum()))
    return res


# --------------------------------------------------------------------------
# detect
# --------------------------------------------------------------------------
def _prf(tp, fp, fn):
    return (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)


def cmd_detect(args):
    import cv2
    from ultralytics import YOLO

    data = Path(args.data)
    data = data if data.is_absolute() else REPO_ROOT / data
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    root = data.parent
    imgs = sorted(p for p in (root / "test" / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    gts = {}
    for p in imgs:
        h, w = cv2.imread(str(p)).shape[:2]
        rows = []
        lf = root / "test" / "labels" / (p.stem + ".txt")
        if lf.exists():
            for line in lf.read_text().split("\n"):
                s = line.split()
                if len(s) >= 5:
                    c, cx, cy, bw, bh = int(s[0]), *map(float, s[1:5])
                    rows.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        gts[p] = rows

    result = {"split": "test", "n_images": len(imgs), "weights": {}}
    for wname in args.weights:
        wpath = Path(wname)
        wpath = wpath if wpath.is_absolute() else REPO_ROOT / wpath
        try:
            model = YOLO(str(wpath), task="detect")
            m = None
            err = None
            for bs in (16, 1):
                try:
                    m = model.val(data=str(data), split="test", imgsz=640, batch=bs, conf=0.001, iou=0.7,
                                  plots=False, save_json=False, project=str(out / "ultralytics"), workers=0,
                                  name=wpath.stem, exist_ok=True, verbose=False)
                    break
                except Exception as e:  # engine may have a static batch
                    err = e
            if m is None:
                raise err
            ap50 = {int(c): float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)}
            ap = {int(c): float(v) for c, v in zip(m.box.ap_class_index, m.box.ap)}
            tot = {c: [0, 0, 0] for c in range(4)}
            for p in imgs:
                r = model.predict(str(p), conf=0.40, iou=0.7, imgsz=640, classes=[0, 1, 2, 3],
                                  max_det=300, verbose=False)[0]
                g = gts[p]
                gx = np.array([x[1:] for x in g], float).reshape(-1, 4)
                gc = np.array([x[0] for x in g], int)
                b = r.boxes
                res = match_operating_point(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                            b.cls.cpu().numpy(), gx, gc)
                for c, (tp, fp, fn) in res.items():
                    t = tot[c]
                    t[0] += tp; t[1] += fp; t[2] += fn
            classes = {}
            for c, name in enumerate(CLASS_NAMES):
                tp, fp, fn = tot[c]
                p_, r_ = _prf(tp, fp, fn)
                classes[name] = {"boxes": tp + fn, "ap50": ap50.get(c), "ap50_95": ap.get(c),
                                 "tp": tp, "fp": fp, "fn": fn, "p": p_, "r": r_}
            TP = sum(v[0] for v in tot.values()); FP = sum(v[1] for v in tot.values()); FN = sum(v[2] for v in tot.values())
            p_, r_ = _prf(TP, FP, FN)
            classes_all = {"boxes": TP + FN, "ap50": float(m.box.map50), "ap50_95": float(m.box.map),
                           "tp": TP, "fp": FP, "fn": FN, "p": p_, "r": r_}
            result["weights"][wpath.name] = {"classes": classes, "all": classes_all}
        except Exception as e:
            result["weights"][wpath.name] = {"error": f"{type(e).__name__}: {e}"}
    (out / "detector_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out / "detector_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["weights", "class", "boxes", "ap50", "ap50_95", "tp", "fp", "fn", "p", "r"])
        for wn, d in result["weights"].items():
            if "error" in d:
                wr.writerow([wn, "ERROR", d["error"]])
                continue
            for k, v in {**d["classes"], "all": d["all"]}.items():
                wr.writerow([wn, k, v["boxes"], v["ap50"], v["ap50_95"], v["tp"], v["fp"], v["fn"], v["p"], v["r"]])
    print(json.dumps(result, indent=2))


# --------------------------------------------------------------------------
# mot-export
# --------------------------------------------------------------------------
PROTOCOL = """
CVAT annotation protocol
 1. Per sequence create one task with labels bus, mobil, motor, truk; upload cvat_images.zip as images.
 2. Actions -> Upload annotations -> format "MOT 1.1" -> cvat_preannotation_mot.zip.
 3. Correct every frame:
    - Annotate every vehicle whose box bottom-center lies inside the yellow zone of zone.png,
      in every frame where >= 30% of it is visible and its box is >= 10 px tall. Parked vehicles count too.
    - Each physical vehicle keeps ONE track id for the whole sequence. Mark frames where it is fully
      occluded or absent as "outside".
    - Delete boxes on non-vehicles, split/merge tracks to fix id swaps, add missed vehicles.
    - Classes follow Table III of the paper (angkot, minibus -> mobil; pickups -> truk); classes are not scored.
 4. Export "MOT 1.1" and save as labeling/mot/<seq>/gt_mot.zip.
"""


def _load_roi_polys(cctv):
    p = REPO_ROOT / "rois" / f"{cctv}_roi.json"
    rois = json.loads(p.read_text(encoding="utf-8")).get("rois", [])
    return [np.array(r["points"], np.int32) for r in rois]


def _segments(date, cctv):
    d = REPO_ROOT / "dataset" / date / cctv / "videos"
    out = []
    for p in sorted(d.glob("*.ts")):
        m = re.search(r"_(\d{8})_(\d{6})\.ts$", p.name)
        if m and m.group(1) == date.replace("-", ""):
            out.append((m.group(2), p))
    return out


def _decode(segs, start, n_frames, fallback_fps, target_fps):
    import cv2

    first = next((i for i, (t, _) in enumerate(segs) if t >= start), None)
    if first is None:
        return [], 0.0, []
    frames, meta, native0 = [], [], 0.0
    for t, path in segs[first:]:
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else fallback_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keep = set(sampled_frame_indices(fps, total, target_fps))
        native0 = native0 or fps
        got = []
        idx = 0
        while len(frames) < n_frames:
            if not cap.grab():
                break
            idx += 1
            if idx in keep:
                ok, fr = cap.retrieve()
                if ok:
                    frames.append(fr)
                    got.append(idx)
        cap.release()
        if got:
            meta.append({"name": path.name, "sampled_frame_idx": got})
        if len(frames) >= n_frames:
            break
    return frames, native0, meta


def cmd_mot_export(args):
    import cv2
    import supervision as sv
    from ultralytics import YOLO
    from feature_extractor import (FALLBACK_FPS, MODEL_WEIGHTS, PREDICT_KWARGS, TARGET_CLASSES,
                                   TARGET_FPS, _proc_size, _to_detections)

    out_root = Path(args.out)
    out_root = out_root if out_root.is_absolute() else REPO_ROOT / out_root
    model = YOLO(str(MODEL_WEIGHTS), task="detect")
    b = args.buffer_px
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1))
    for seq, cctv, display, start in SEQUENCES:
        segs = _segments(args.date, cctv)
        frames, native_fps, seg_meta = _decode(segs, start, args.frames, FALLBACK_FPS, TARGET_FPS)
        if len(frames) < args.frames:
            raise SystemExit(f"{seq}: only {len(frames)} frames")
        sd = out_root / seq
        (sd / "frames").mkdir(parents=True, exist_ok=True)
        tracker = sv.ByteTrack(frame_rate=TARGET_FPS)
        rows = []
        for i, fr in enumerate(frames, 1):
            cv2.imwrite(str(sd / "frames" / f"{i:06d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = fr.shape[:2]
            scale, size = _proc_size(w, h)
            proc = cv2.resize(fr, size, interpolation=cv2.INTER_AREA) if size else fr
            r = model.predict(proc, classes=list(TARGET_CLASSES), **PREDICT_KWARGS)[0]
            bx = r.boxes
            det = tracker.update_with_detections(_to_detections(
                bx.xyxy.cpu().numpy(), bx.conf.cpu().numpy(), bx.cls.cpu().numpy()))
            if det.tracker_id is None:
                continue
            for xyxy, tid, cf, c in zip(det.xyxy, det.tracker_id, det.confidence, det.class_id):
                x1, y1, x2, y2 = (xyxy * scale).tolist()
                rows.append((i, int(tid), x1, y1, x2 - x1, y2 - y1, float(cf), int(c)))
        with open(sd / "pred.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["frame", "track_id", "x", "y", "w", "h", "conf", "cls"])
            for r_ in rows:
                wr.writerow([r_[0], r_[1], *(f"{v:.2f}" for v in r_[2:6]), f"{r_[6]:.4f}", r_[7]])
        roi = np.zeros((FRAME_H, FRAME_W), np.uint8)
        polys = _load_roi_polys(cctv)
        cv2.fillPoly(roi, polys, 1)
        zone = cv2.dilate(roi, kernel)
        np.savez_compressed(sd / "masks.npz", roi=roi.astype(bool), zone=zone.astype(bool))
        vis = frames[0].copy()
        cv2.polylines(vis, polys, True, (0, 0, 255), 2)
        cnts, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
        cv2.imwrite(str(sd / "zone.png"), vis)
        (sd / "meta.json").write_text(json.dumps({
            "seq": seq, "cctv": cctv, "display": display, "date": args.date, "native_fps": native_fps,
            "target_fps": TARGET_FPS, "n_frames": len(frames), "buffer_px": b,
            "model_weights": Path(MODEL_WEIGHTS).name, "segments": seg_meta}, indent=2), encoding="utf-8")
        with zipfile.ZipFile(sd / "cvat_images.zip", "w", zipfile.ZIP_STORED) as z:
            for i in range(1, len(frames) + 1):
                z.write(sd / "frames" / f"{i:06d}.jpg", f"{i:06d}.jpg")
        pa = np.array([r_[:6] for r_ in rows], float).reshape(-1, 6)
        in_zone = bottom_center_in(zone.astype(bool), pa[:, 2:6]) if len(pa) else np.zeros(0, bool)
        lines = [f"{int(r_[0])},{int(r_[1])},{r_[2]:.2f},{r_[3]:.2f},{r_[4]:.2f},{r_[5]:.2f},1,{r_[7] + 1},1.0"
                 for r_, k in zip(rows, in_zone) if k]
        with zipfile.ZipFile(sd / "cvat_preannotation_mot.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("gt/gt.txt", "\n".join(lines) + "\n")
            z.writestr("gt/labels.txt", "\n".join(CLASS_NAMES) + "\n")
        in_roi = int(bottom_center_in(roi.astype(bool), pa[:, 2:6]).sum()) if len(pa) else 0
        print(f"{seq}: {len(frames)} frames, {len(rows)} tracked boxes, {in_roi} in ROI, "
              f"{len(lines)} in zone, tracks {len({r_[1] for r_ in rows})}; {sd}")
    print(PROTOCOL)


# --------------------------------------------------------------------------
# mot-score
# --------------------------------------------------------------------------
def _read_pred(path):
    d = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d.setdefault(int(r["frame"]), []).append((int(r["track_id"]), float(r["x"]), float(r["y"]),
                                                      float(r["w"]), float(r["h"])))
    return d


def _read_gt(zpath):
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        name = "gt/gt.txt" if "gt/gt.txt" in names else next(
            (n for n in names if n.rsplit("/", 1)[-1] == "gt.txt"), None)
        if name is None:
            raise SystemExit(f"{zpath}: no gt.txt")
        txt = z.read(name).decode("utf-8")
    d = {}
    for line in txt.splitlines():
        s = line.strip().split(",")
        if len(s) < 6:
            continue
        d.setdefault(int(float(s[0])), []).append((int(float(s[1])), *map(float, s[2:6])))
    return d


def _accumulate(mm, gt, pred, roi, n_frames):
    acc = mm.MOTAccumulator(auto_id=False)
    for f in range(1, n_frames + 1):
        g = gt.get(f, [])
        p = pred.get(f, [])
        gi = np.array([x[0] for x in g], int)
        gb = np.array([x[1:] for x in g], float).reshape(-1, 4)
        pi = np.array([x[0] for x in p], int)
        pb = np.array([x[1:] for x in p], float).reshape(-1, 4)
        gb, gi, pb, pi = filter_frame(gb, gi, pb, pi, roi)
        dist = mm.distances.iou_matrix(gb, pb, max_iou=0.5) if len(gb) and len(pb) else np.empty((len(gb), len(pb)))
        acc.update(gi, pi, dist, frameid=f)
    return acc


METRICS = ["num_frames", "num_unique_objects", "num_objects", "num_predictions", "mota", "motp", "idf1", "recall", "precision",
           "idp", "idr", "num_switches", "num_false_positives", "num_misses", "num_fragmentations",
           "mostly_tracked", "mostly_lost"]


def cmd_mot_score(args):
    import motmetrics as mm

    root = Path(args.root)
    root = root if root.is_absolute() else REPO_ROOT / root
    need = ["pred.csv", "masks.npz", "meta.json"] + ([] if args.self_check else ["gt_mot.zip"])
    seqs = [s for s in SEQUENCES if not args.self_check or s[0] == args.self_check]
    missing = [str(root / s[0] / n) for s in seqs for n in need if not (root / s[0] / n).exists()]
    if missing:
        print("missing:\n  " + "\n  ".join(missing))
        raise SystemExit(1)

    events = {}
    for seq, _, display, start in seqs:
        sd = root / seq
        meta = json.loads((sd / "meta.json").read_text())
        roi = np.load(sd / "masks.npz")["roi"]
        pred = _read_pred(sd / "pred.csv")
        if args.self_check:
            keep = {}
            for f, rows in pred.items():
                arr = np.array([r[1:] for r in rows], float).reshape(-1, 4)
                m = bottom_center_in(roi, arr)
                keep[f] = [r for r, k in zip(rows, m) if k]
            gt = keep
            if not any(gt.values()):
                print("no in-ROI predictions")
                raise SystemExit(1)
        else:
            gt = _read_gt(sd / "gt_mot.zip")
            bad = [f for f in gt if not 1 <= f <= meta["n_frames"]]
            assert not bad, f"{seq}: GT frames out of range {bad[:5]}"
        events[seq] = _accumulate(mm, gt, pred, roi, meta["n_frames"]).events

    mh = mm.metrics.create()

    def compute(evs, name):
        df = evs[0] if len(evs) == 1 else mm.MOTAccumulator.merge_event_dataframes(
            evs, update_frame_indices=True, update_oids=True, update_hids=True)
        return mh.compute(df, metrics=METRICS, name=name)

    groups = {s[0]: [s[0]] for s in seqs}
    if not args.self_check:
        for disp in dict.fromkeys(s[2] for s in seqs):
            groups[disp] = [s[0] for s in seqs if s[2] == disp]
        groups["Day"] = [s[0] for s in seqs if s[0].endswith("_day")]
        groups["Night"] = [s[0] for s in seqs if s[0].endswith("_night")]
        groups["All"] = [s[0] for s in seqs]
    import pandas as pd
    summ = pd.concat([compute([events[k] for k in ks], g) for g, ks in groups.items()])
    print(summ.to_string())
    if args.self_check:
        r = summ.iloc[0]
        ok = r["mota"] == 1.0 and r["idf1"] == 1.0 and r["num_switches"] == 0
        print("SELF-CHECK", "OK" if ok else "FAILED")
        raise SystemExit(0 if ok else 1)
    res = root / "results"
    res.mkdir(exist_ok=True)
    out = summ.reset_index().rename(columns={"index": "group"})
    out.to_csv(res / "mot_metrics.csv", index=False)
    rows = [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
             for k, v in r.items()} for r in out.to_dict("records")]
    (res / "mot_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {res}")


def cmd_describe(args):
    import ast
    import inspect
    import json
    import re
    import torch
    import ultralytics
    from ultralytics import YOLO
    import supervision as sv

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = root / "model/best.pt"
    engine_path = root / "model/best.engine"

    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    train_args_full = ckpt.get("train_args")
    if train_args_full:
        keys = [
            "model", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "patience", "hsv_h", "hsv_s", "hsv_v",
            "fliplr", "mosaic", "mixup", "seed",
        ]
        train_args = {k: train_args_full.get(k) for k in keys if k in train_args_full}
    else:
        train_args = None

    ds_root = root / "dataset-yolo/vehicle detect.v1i"
    splits = {}
    for sp in ["train", "valid", "test"]:
        sp_dir = ds_root / sp / "images"
        if sp_dir.exists():
            imgs = [p.name for p in sp_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
            source_bases = set()
            for img in imgs:
                base = img.split(".rf.")[0]
                base = re.sub(r"_(jpg|png|jpeg)$", "", base, flags=re.IGNORECASE)
                source_bases.add(base)
            splits[sp] = {
                "n_images": len(imgs),
                "n_source_frames": len(source_bases),
            }

    roboflow_readme = ds_root / "README.roboflow.txt"
    rf_lines = []
    if roboflow_readme.exists():
        rf_text = roboflow_readme.read_text(encoding="utf-8")
        for line in rf_text.splitlines():
            if any(k in line.lower() for k in ["pre-processing", "augmentation", "resize", "stretch", "flip", "exposure", "noise"]):
                rf_lines.append(line.strip())

    fe_path = root / "script/feature_extractor.py"
    fe_src = fe_path.read_text(encoding="utf-8")
    parsed = ast.parse(fe_src)
    constants = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in [
                    "TARGET_FPS", "INFER_IMGSZ", "CONF_THRESH", "MIN_SPEED_KMH", "MAX_SPEED_KMH",
                    "FALLBACK_FPS", "PROC_WIDTH", "ROI_REF_WIDTH",
                ]:
                    try:
                        val = ast.literal_eval(node.value)
                        constants[target.id] = val
                    except Exception:
                        pass

    t_inst = sv.ByteTrack(frame_rate=constants.get("TARGET_FPS", 10), lost_track_buffer=30)
    tracker_config = {
        "tracker_class": "supervision.ByteTrack",
        "supervision_version": getattr(sv, "__version__", "unknown"),
        "frame_rate": constants.get("TARGET_FPS", 10),
        "parameters": {
            "track_activation_threshold": float(getattr(t_inst, "track_activation_threshold", 0.25)),
            "lost_track_buffer": 30,
            "minimum_matching_threshold": float(getattr(t_inst, "minimum_matching_threshold", 0.8)),
            "minimum_consecutive_frames": int(getattr(t_inst, "minimum_consecutive_frames", 1)),
        },
    }

    hardware = "NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), TensorRT FP16, 2 inference processes"

    out_data = {
        "model": {
            "name": "Ultralytics YOLOv8n (Nano)",
            "ultralytics_version": getattr(ultralytics, "__version__", "unknown"),
            "weights_pt": str(pt_path.relative_to(root)),
            "weights_engine": str(engine_path.relative_to(root)),
            "layers": 73,
            "parameters": 3006428,
            "gflops": 8.1,
            "classes": {0: "bus", 1: "mobil", 2: "motor", 3: "truk"},
            "train_args": train_args,
        },
        "dataset": {
            "splits": splits,
            "roboflow_notes": rf_lines,
        },
        "inference": {
            "constants": constants,
            "iou_threshold": 0.70,
        },
        "tracker": tracker_config,
        "hardware": hardware,
    }

    out_file = out_dir / "detector_config.json"
    out_file.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[✓] Saved {out_file}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--weights", nargs="+", default=["model/best.pt", "model/best.engine"])
    d.add_argument("--data", default="dataset-yolo/vehicle detect.v1i/data.yaml")
    d.add_argument("--out", default="labeling/eval/detector")
    d.set_defaults(fn=cmd_detect)
    e = sub.add_parser("mot-export")
    e.add_argument("--date", default="2026-10-11")
    e.add_argument("--frames", type=int, default=150)
    e.add_argument("--buffer-px", type=int, default=48)
    e.add_argument("--out", default="labeling/mot")
    e.set_defaults(fn=cmd_mot_export)
    s = sub.add_parser("mot-score")
    s.add_argument("--root", default="labeling/mot")
    s.add_argument("--self-check", default=None)
    s.set_defaults(fn=cmd_mot_score)
    desc = sub.add_parser("describe")
    desc.add_argument("--out", default="labeling/eval/detector")
    desc.set_defaults(fn=cmd_describe)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

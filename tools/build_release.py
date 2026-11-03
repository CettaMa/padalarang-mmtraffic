#!/usr/bin/env python3
"""Build complete public reproducibility and release package for Zenodo and GitHub."""

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import datetime
import importlib.metadata as importlib_metadata
sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_docs import check_dictionary, dictionary_markdown, missing_values_markdown  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
ZENODO_DOI = "10.5281/zenodo.23873787"

# Import mask_url from scrapper
sys.path.insert(0, str(WORKSPACE_ROOT / "scrapper"))
try:
    from cctv_scraper.dashboard.security import mask_url
except ImportError:
    def mask_url(url: str) -> str:
        if not url:
            return ""
        # Fallback regex masking
        return re.sub(r"/memfs/[0-9a-fA-F-]+(\.m3u8)?", r"/memfs/********\1", url)

DROP_COLUMNS = {
    "traffic_speed",
    "traffic_freeflow",
    "traffic_confidence",
    "traffic_road_closure",
    "traffic_source",
    "traffic_status",
    "traffic_cache_status",
    "traffic_last_api_call",
}

START_DATE = "2026-09-09"
END_DATE = "2026-10-10"


def is_in_window(date_str: str) -> bool:
    return START_DATE <= date_str <= END_DATE
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_zenodo_package(out_root: Path, resume: bool = True):
    print(f"\n--- Building Zenodo Release Package at {out_root} ---")
    if not resume and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # 1. Features
    feat_out = out_root / "features"
    feat_out.mkdir(parents=True, exist_ok=True)
    raw_feat_dir = REPO_ROOT / "extracted"

    feat_files = sorted(raw_feat_dir.glob("*_metadata_features.csv"))
    in_window_feat = []
    for f in feat_files:
        # Match date in filename
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if m and is_in_window(m.group(1)):
            in_window_feat.append(f)

    assert len(in_window_feat) == 192, f"Expected 192 feature CSVs, found {len(in_window_feat)}"

    print(f"[*] Processing {len(in_window_feat)} feature CSVs (masking stream_url, dropping TomTom columns)...")
    for src in in_window_feat:
        dst = feat_out / src.name
        with open(src, "r", encoding="utf-8", newline="") as fin, open(dst, "w", encoding="utf-8", newline="") as fout:
            reader = csv.DictReader(fin)
            fieldnames = [k for k in reader.fieldnames if k not in DROP_COLUMNS]
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                # Mask stream_url
                if "stream_url" in row and row["stream_url"]:
                    row["stream_url"] = mask_url(row["stream_url"])
                # Remove dropped columns
                cleaned = {k: row[k] for k in fieldnames}
                writer.writerow(cleaned)

    # Assert no unmasked memfs tokens
    token_pattern = re.compile(r"memfs/[0-9a-fA-F-]{36}")
    for dst in feat_out.glob("*.csv"):
        text = dst.read_text(encoding="utf-8")
        assert not token_pattern.search(text), f"Found unmasked memfs token in {dst.name}!"

    # 2. Metrics
    metrics_out = out_root / "metrics"
    metrics_out.mkdir(parents=True, exist_ok=True)
    raw_met_dir = REPO_ROOT / "extracted" / "metrics"
    met_files = sorted(raw_met_dir.glob("*_metrics.csv"))
    in_window_met = [f for f in met_files if (m := re.search(r"(\d{4}-\d{2}-\d{2})", f.name)) and is_in_window(m.group(1))]
    assert len(in_window_met) == 192, f"Expected 192 metrics CSVs, found {len(in_window_met)}"
    for f in in_window_met:
        shutil.copy2(f, metrics_out / f.name)
    shutil.copy2(raw_met_dir / "capacity_empirical.csv", metrics_out / "capacity_empirical.csv")
    shutil.copy2(raw_met_dir / "emp_empirical.csv", metrics_out / "emp_empirical.csv")
    print(f"[✓] Copied {len(in_window_met)} metrics CSVs + 2 capacity references")

    # 3. Tracks
    tracks_out = out_root / "tracks"
    tracks_out.mkdir(parents=True, exist_ok=True)
    raw_tracks_dir = REPO_ROOT / "extracted" / "tracks"
    track_dates = [d.name for d in raw_tracks_dir.iterdir() if d.is_dir() and is_in_window(d.name)]
    print(f"[*] Packaging track logs for {len(track_dates)} in-window dates...")
    for date_str in track_dates:
        zip_path = tracks_out / f"tracks_{date_str}.zip"
        if zip_path.exists() and zip_path.stat().st_size > 1000:
            continue
        src_dir = raw_tracks_dir / date_str
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for root, _, files in os.walk(src_dir):
                for file in files:
                    full_p = Path(root) / file
                    arcname = full_p.relative_to(src_dir)
                    zf.write(full_p, arcname)
    print(f"[✓] Created {len(track_dates)} track archive zips")

    # 4. Reports
    rep_out = out_root / "reports"
    rep_out.mkdir(parents=True, exist_ok=True)
    raw_rep_dir = REPO_ROOT / "extracted" / "reports"
    rep_files = [f for f in raw_rep_dir.glob("*_missing_report.json") if (m := re.search(r"(\d{4}-\d{2}-\d{2})", f.name)) and is_in_window(m.group(1))]
    for f in rep_files:
        shutil.copy2(f, rep_out / f.name)
    print(f"[✓] Copied {len(rep_files)} missing data quality reports")

    # 5. ROIs
    rois_out = out_root / "rois"
    rois_out.mkdir(parents=True, exist_ok=True)
    for p in (REPO_ROOT / "rois").glob("*_roi.json"):
        shutil.copy2(p, rois_out / p.name)
    pic_src = REPO_ROOT / "rois" / "roi_measure_picture"
    if pic_src.exists():
        pic_out = rois_out / "roi_measure_picture"
        pic_out.mkdir(parents=True, exist_ok=True)
        for p in pic_src.iterdir():
            if p.is_file():
                shutil.copy2(p, pic_out / p.name)
    print(f"[✓] Copied {len(list(rois_out.glob('*_roi.json')))} ROI polygon definitions and ROI measurement evidence")

    # 6. Graph
    graph_out = out_root / "graph"
    graph_out.mkdir(parents=True, exist_ok=True)
    for f in (REPO_ROOT / "graph").iterdir():
        if f.is_file():
            shutil.copy2(f, graph_out / f.name)
    print(f"[✓] Copied road graph files to {graph_out}")

    # 7. Events
    events_out = out_root / "events"
    events_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "events" / "events.csv", events_out / "events.csv")
    shutil.copy2(REPO_ROOT / "events" / "event_params.json", events_out / "event_params.json")
    print(f"[✓] Copied event metadata files")

    # 8. Model
    model_out = out_root / "model"
    model_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "model" / "best.pt", model_out / "best.pt")
    shutil.copy2(REPO_ROOT / "dataset-yolo" / "vehicle detect.v1i" / "data.yaml", model_out / "data.yaml")
    shutil.copy2(REPO_ROOT / "labeling" / "eval" / "detector" / "detector_metrics.json", model_out / "detector_metrics.json")
    shutil.copy2(REPO_ROOT / "labeling" / "eval" / "detector" / "detector_config.json", model_out / "detector_config.json")
    print(f"[✓] Copied detector weights and evaluation files")

    # 9. Validation
    val_out = out_root / "validation"
    val_out.mkdir(parents=True, exist_ok=True)
    fs_src = REPO_ROOT / "labeling" / "speed_val" / "final_sample.csv"
    with open(fs_src, "r", encoding="utf-8", newline="") as fin, open(val_out / "final_sample.csv", "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = [c for c in reader.fieldnames if c != "video_path"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow({k: row[k] for k in fieldnames})
    shutil.copy2(REPO_ROOT / "labeling" / "speed_val" / "metrics.csv", val_out / "metrics.csv")

    # Sanitize annotations.csv (remove video_path)
    ann_src = REPO_ROOT / "labeling" / "speed_val" / "annotations.csv"
    ann_dst = val_out / "annotations.csv"
    with open(ann_src, "r", encoding="utf-8", newline="") as fin, open(ann_dst, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = [c for c in reader.fieldnames if c != "video_path"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            cleaned = {k: row[k] for k in fieldnames}
            writer.writerow(cleaned)

    val_analysis_out = val_out / "analysis"
    val_analysis_out.mkdir(parents=True, exist_ok=True)
    src_ana = REPO_ROOT / "labeling" / "speed_val" / "analysis"
    for fn in ["analysis_summary.json", "roi_diagnostics.csv", "length_uncertainty.csv", "timing_uncertainty.csv"]:
        p = src_ana / fn
        if p.exists():
            shutil.copy2(p, val_analysis_out / fn)
    print(f"[✓] Copied speed validation benchmark files")

    # Phase 3 additions: repeat annotation, failure causes, tracker evaluation
    for fn in ["interrater.csv", "causes_summary.csv", "causes_table.csv", "causes_by_location.csv"]:
        p = src_ana / fn
        if not p.exists():
            raise SystemExit(f"missing {p}; run tools/speed_val_analysis.py")
        shutil.copy2(p, val_analysis_out / fn)
    sec = json.loads((REPO_ROOT / "labeling" / "speed_val" / "annotations_second.json").read_text(encoding="utf-8"))
    with open(ann_src, "r", encoding="utf-8", newline="") as fin:
        base_fields = [c for c in csv.DictReader(fin).fieldnames if c != "video_path"]
    with open(val_out / "annotations_second.csv", "w", encoding="utf-8", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=base_fields + ["annotator"])
        w.writeheader()
        for a in sec.values():
            w.writerow({k: ("" if a.get(k) is None else a.get(k)) for k in base_fields + ["annotator"]})
    mot_src = REPO_ROOT / "labeling" / "mot"
    if not (mot_src / "results" / "mot_metrics.csv").exists():
        raise SystemExit("missing labeling/mot/results/mot_metrics.csv; run tools/component_eval.py mot-score")
    shutil.copy2(mot_src / "results" / "mot_metrics.csv", val_out / "mot_metrics.csv")
    n_clips = 0
    for sd in sorted(p for p in mot_src.iterdir() if (p / "gt_mot.zip").exists()):
        dst = val_out / "mot" / sd.name
        dst.mkdir(parents=True, exist_ok=True)
        for fn in ("gt_mot.zip", "pred.csv", "masks.npz", "meta.json"):
            shutil.copy2(sd / fn, dst / fn)
        n_clips += 1
    assert n_clips == 12, n_clips
    print(f"[✓] Copied repeat annotation, failure causes, and {n_clips} tracker evaluation clips")

    check_dictionary(out_root)
    print("[✓] Data dictionary covers every column of the shipped CSV files")
    # 10. Data Dictionary & README
    readme_path = out_root / "README.md"
    readme_content = _generate_zenodo_readme()
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"[✓] Generated dataset README.md at {readme_path}")

    # 11. LICENSE (CC BY 4.0) & CITATION.cff
    license_path = out_root / "LICENSE"
    license_path.write_text((Path(__file__).resolve().parent / "assets" / "cc_by_4.0_legalcode.txt").read_text(encoding="utf-8"), encoding="utf-8")

    cff_path = out_root / "CITATION.cff"
    cff_path.write_text(_generate_citation_cff(), encoding="utf-8")

    # 11b. Normalise text files to LF so that `sha256sum -c` agrees on every platform
    text_suffixes = {".csv", ".json", ".md", ".txt", ".cff", ".yaml", ".yml", ""}
    n_lf = 0
    for p in out_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in text_suffixes and p.name != "checksums.sha256":
            raw = p.read_bytes()
            if b"\r\n" in raw:
                p.write_bytes(raw.replace(b"\r\n", b"\n"))
                n_lf += 1
    print(f"[\u2713] Normalised line endings to LF in {n_lf} text files")

    # 12. Checksums
    print("[*] Generating SHA-256 checksums for all release files...")
    all_files = sorted([p for p in out_root.rglob("*") if p.is_file() and p.name != "checksums.sha256"])
    checksum_lines = []
    for f in all_files:
        rel = f.relative_to(out_root).as_posix()
        csum = sha256_file(f)
        checksum_lines.append(f"{csum} *{rel}")
    (out_root / "checksums.sha256").write_bytes(("\n".join(checksum_lines) + "\n").encode("utf-8"))
    print(f"[✓] Generated checksums.sha256 ({len(checksum_lines)} files)")


def build_github_repo(out_root: Path):
    print(f"\n--- Building GitHub Code Repository at {out_root} ---")
    if out_root.exists():
        import stat
        shutil.rmtree(out_root, onexc=lambda fn, p, exc: (os.chmod(p, stat.S_IWRITE), fn(p)))
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Scraper files from scrapper/ (tracked files)
    scraper_dir = WORKSPACE_ROOT / "scrapper"
    proc = subprocess.run(["git", "-C", str(scraper_dir), "ls-files"], capture_output=True, text=True, check=True)
    tracked_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    target_scraper = out_root / "scraper"
    target_scraper.mkdir(parents=True, exist_ok=True)

    memfs_pattern = re.compile(r"memfs/[0-9a-fA-F-]{36}")

    for rel_path in tracked_files:
        src = scraper_dir / rel_path
        if not src.exists():
            continue
        dst = target_scraper / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Check that no memfs credentials leak (except mock test files)
        if "test" not in rel_path:
            try:
                txt = src.read_text(encoding="utf-8")
                assert not memfs_pattern.search(txt), f"Secret memfs token detected in tracked file: {rel_path}"
            except UnicodeDecodeError:
                pass
        shutil.copy2(src, dst)
    print(f"[✓] Packaged {len(tracked_files)} scraper source files")

    # 2. Extractions scripts and tools
    scripts_to_copy = [
        "script/feature_extractor.py",
        "tools/traffic_metrics.py",
        "tools/run_all_metrics.py",
        "tools/cctv_status.py",
        "tools/event_features.py",
        "tools/dataset_eda.py",
        "tools/speed_val.py",
        "tools/speed_val_analysis.py",
        "tools/speed_val_inspector.py",
        "tools/component_eval.py",
        "tools/build_road_graph.py",
        "tools/build_release.py",
        "tools/release_docs.py",
        "tools/assets/cc_by_4.0_legalcode.txt",
        "tools/test_cctv_status.py",
        "tools/test_event_features.py",
        "tools/test_speed_val.py",
        "tools/test_component_eval.py",
        "tools/test_traffic_metrics.py",
    ]

    for rel in scripts_to_copy:
        src = REPO_ROOT / rel
        if src.exists():
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"[✓] Packaged {len(scripts_to_copy)} extraction tools and test scripts")

    # Config files required by tools
    shutil.copy2(REPO_ROOT / "cctv_cameras.json", out_root / "cctv_cameras.json")
    target_rois = out_root / "rois"
    target_rois.mkdir(parents=True, exist_ok=True)
    for p in (REPO_ROOT / "rois").glob("*_roi.json"):
        shutil.copy2(p, target_rois / p.name)
    target_events = out_root / "events"
    target_events.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "events" / "events.csv", target_events / "events.csv")
    shutil.copy2(REPO_ROOT / "events" / "event_params.json", target_events / "event_params.json")

    # 3. requirements.txt
    pinned = [f"{dist}=={importlib_metadata.version(dist).split('+')[0]}" for dist in
              ("torch", "ultralytics", "supervision", "opencv-python", "numpy", "pandas", "scipy", "matplotlib", "requests", "pytest")]
    reqs_text = "\n".join(pinned) + "\n"
    (out_root / "requirements.txt").write_text(reqs_text, encoding="utf-8")

    # 4. GitHub README.md
    readme_text = _generate_github_readme()
    (out_root / "README.md").write_text(readme_text, encoding="utf-8")

    # 5. MIT LICENSE
    mit_license = (
        "MIT License\n\n"
        "Copyright (c) 2026 Cetta Maulana Andhika, Surya Michrandi Nasution, Reza Rendian Septiawan\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n"
    )
    (out_root / "LICENSE").write_text(mit_license, encoding="utf-8")

    (out_root / ".gitignore").write_text("__pycache__/\n*.pyc\n.venv/\n.env\n.pytest_cache/\n", encoding="utf-8")

    # 6. Git init, commit, and tag v1.0.0
    print("[*] Initializing local git repository and creating v1.0.0 tag...")
    subprocess.run(["git", "init"], cwd=out_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Cetta Maulana Andhika"], cwd=out_root, check=True)
    subprocess.run(["git", "config", "user.email", "cettamaulanaandhika@student.telkomuniversity.ac.id"], cwd=out_root, check=True)
    subprocess.run(["git", "add", "."], cwd=out_root, check=True)
    subprocess.run(["git", "commit", "-m", "Release v1.0.0: Padalarang Multi-Modal Traffic Dataset Construction Pipeline"], cwd=out_root, check=True)
    subprocess.run(["git", "tag", "-a", "v1.0.0", "-m", "Version 1.0.0 release"], cwd=out_root, check=True)
    print("[✓] Local repository initialized with commit and tag v1.0.0")


def _generate_zenodo_readme() -> str:
    return """# Padalarang Multi-Modal Mixed Urban Traffic Dataset (v1.0.0)

## Overview
This repository contains the complete open dataset release described in the publication:
**"A Synchronized Multi-Modal Dataset of CCTV, Weather, and Event Records for Mixed Urban Traffic: Construction and Speed Validation"**
*by Cetta Maulana Andhika, Surya Michrandi Nasution, and Reza Rendian Septiawan.*

**DOI:** [10.5281/zenodo.23873787](https://doi.org/10.5281/zenodo.23873787) · **Code:** https://github.com/CettaMa/padalarang-mmtraffic (tag v1.0.0)

The dataset captures 32 consecutive calendar days (9 September – 10 October 2026) across six strategic closed-circuit television (CCTV) monitoring installations (12 Regions of Interest, ROIs) along the Padalarang arterial corridor at the western gateway of the Bandung metropolitan area, West Java, Indonesia.

## Directory Structure
- `features/`: 192 daily CSV files containing 58,358 five-minute synchronized multi-modal observation records.
- `metrics/`: 192 daily traffic metrics files containing high-resolution hydrodynamic traffic variables (occupancy, queue, degree of saturation, passenger car units) per ROI, plus empirical capacity reference tables.
- `tracks/`: 32 daily ZIP archives containing per-vehicle trajectory and detection logs.
- `rois/`: 6 JSON files defining measured road dimensions, homography vertices, camera perspectives, and queue zones for all 12 ROIs, plus `roi_measure_picture/` (satellite screenshots, `readings.csv`, `anchors.md`) behind the measured dimensions.
- `graph/`: Road network topology files including node attributes (`nodes.csv`), 18 undirected adjacency links (`edges.csv`), binary adjacency matrix (`adjacency.csv`), and pairwise distance matrix (`distance_m.csv`).
- `events/`: Event metadata definition and holiday schedule records.
- `model/`: YOLOv8n fine-tuned weights (`best.pt`), dataset split configuration (`data.yaml`), held-out test detection metrics (`detector_metrics.json`), and pipeline configuration (`detector_config.json`).
- `validation/`: Ground-truth speed validation dataset across 240 vehicles, evaluation metrics with 95% bootstrap confidence intervals, Bland–Altman agreement parameters, spatial/temporal uncertainty measurements, a repeat annotation of 48 vehicles, the failure causes of 118 inspected crossings, and the tracker evaluation (`mot_metrics.csv` and the 12 hand-corrected clips in `mot/`).

## How the files are delivered on Zenodo
Zenodo limits the number of files per record, so each directory above except `tracks/` is delivered as one ZIP of the same name (`features.zip`, `metrics.zip`, `rois.zip`, `reports.zip`, `graph.zip`, `events.zip`, `model.zip`, `validation.zip`), and the 32 `tracks_<date>.zip` archives are delivered unchanged. Unzip each directory archive in one common folder (each contains its own top-level directory), and move the 32 `tracks_<date>.zip` files into a `tracks/` folder next to them, without unzipping them, to obtain the tree above. `checksums.sha256` lists every file of that tree: run `sha256sum -c checksums.sha256` from the folder that contains `features/`, `metrics/`, and the others.

""" + dictionary_markdown() + "\n" + missing_values_markdown() + """

### Recommended Chronological Split
For benchmarking spatio-temporal and sequence learning models, the following non-overlapping chronological partition is recommended:
- **Training Set**: 2026-09-09 to 2026-10-01 (23 days, 71.9%)
- **Validation Set**: 2026-10-02 to 2026-10-05 (4 days, 12.5%)
- **Testing Set**: 2026-10-06 to 2026-10-10 (5 days, 15.6%)

## Access & Licensing Restrictions
1. **Raw CCTV Video Footage**: Raw video segments (`.ts`) are not redistributed: they come from third-party streams of the local transportation agency (Dishub Kabupaten Bandung Barat) and show identifiable pedestrians and licence plates. Instead, the per-vehicle track logs (`tracks/`), ROI geometry (`rois/`), detector weights (`model/`), and validation annotations (`validation/`) are provided so that every step after detection can be inspected and repeated.
2. **Third-Party Floating-Car Data**: Commercial TomTom API traffic fields used during internal quality control have been excluded from this public distribution in compliance with TomTom Terms of Service.
3. **Data License**: This dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
4. **Attribution**: Environmental data retrieved from Open-Meteo under CC BY 4.0. Road segment geometry derived from OpenStreetMap data (c) OpenStreetMap contributors under ODbL.
"""


def _generate_github_readme() -> str:
    return """# Padalarang Multi-Modal Mixed Urban Traffic Pipeline

This repository contains the complete open-source codebase for capturing, processing, synchronizing, and validating the Padalarang multi-modal urban traffic dataset.

## Associated Publication
**"A Synchronized Multi-Modal Dataset of CCTV, Weather, and Event Records for Mixed Urban Traffic: Construction and Speed Validation"**  
*Cetta Maulana Andhika, Surya Michrandi Nasution, Reza Rendian Septiawan (2026).*

- **Dataset Download**: [Zenodo record, DOI 10.5281/zenodo.23873787](https://doi.org/10.5281/zenodo.23873787)
- **Code License**: MIT License

## Repository Layout
- `scraper/`: Continuous 24/7 asynchronous recording and metadata acquisition service for CCTV, Open-Meteo, and calendar sources.
- `script/feature_extractor.py`: Real-time vehicle detection (YOLOv8), multi-object tracking (ByteTrack), planar homography projection, and instantaneous speed estimation pipeline.
- `tools/`:
  - `traffic_metrics.py` & `run_all_metrics.py`: Computes hydrodynamic traffic features (time occupancy, queue length, PKJI 2023 degree of saturation, dynamic Chandra PCU).
  - `speed_val.py` & `speed_val_analysis.py`: Ground-truth speed validation harness with stratified percentile bootstrap confidence intervals and Bland–Altman agreement analysis.
  - `speed_val_inspector.py`: Desktop visual inspection tool for single-frame validation at 25 fps.
  - `component_eval.py`: Detector held-out test evaluation and MOT tracking evaluation runner.
  - `build_road_graph.py`: Generates the 12-node corridor network graph and distance matrices.

## Installation & Setup
```bash
git clone https://github.com/CettaMa/padalarang-mmtraffic.git
cd padalarang-mmtraffic
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Running Verification Tests
```bash
pytest tools/test_cctv_status.py tools/test_event_features.py tools/test_speed_val.py tools/test_component_eval.py
```
`tools/test_traffic_metrics.py` needs raw video segments, which are not distributed, and is therefore not part of this command.
"""



def _generate_citation_cff() -> str:
    return """cff-version: 1.2.0
message: "If you use this dataset, please cite it as below."
title: "A Synchronized Multi-Modal Dataset of CCTV, Weather, and Event Records for Mixed Urban Traffic: Construction and Speed Validation"
version: 1.0.0
date-released: DATE_PLACEHOLDER
license: CC-BY-4.0
doi: DOI_PLACEHOLDER
repository-code: "https://github.com/CettaMa/padalarang-mmtraffic"
authors:
  - family-names: Andhika
    given-names: Cetta Maulana
    affiliation: "School of Electrical Engineering, Telkom University"
    email: cettamaulanaandhika@student.telkomuniversity.ac.id
  - family-names: Nasution
    given-names: Surya Michrandi
    affiliation: "School of Electrical Engineering, Telkom University"
    email: michrandi@telkomuniversity.ac.id
  - family-names: Septiawan
    given-names: Reza Rendian
    affiliation: "School of Electrical Engineering, Telkom University"
""".replace("DATE_PLACEHOLDER", datetime.date.today().isoformat()).replace("DOI_PLACEHOLDER", ZENODO_DOI)


def bundle_for_upload(zen_dir: Path, up_dir: Path):
    """Zenodo caps the number of files per record: one ZIP per directory, tracks/ archives and top-level files unchanged."""
    if up_dir.exists():
        shutil.rmtree(up_dir, onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
    up_dir.mkdir(parents=True)
    for name in ("README.md", "LICENSE", "CITATION.cff", "checksums.sha256"):
        shutil.copy2(zen_dir / name, up_dir / name)
    for d in sorted(p for p in zen_dir.iterdir() if p.is_dir() and p.name != "tracks"):
        with zipfile.ZipFile(up_dir / f"{d.name}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                z.write(f, f.relative_to(zen_dir).as_posix())
    for f in sorted((zen_dir / "tracks").glob("*.zip")):
        try:
            os.link(f, up_dir / f.name)
        except OSError:
            shutil.copy2(f, up_dir / f.name)
    files = sorted(p.name for p in up_dir.iterdir())
    assert len(files) < 100, len(files)
    print(f"[✓] Upload bundle: {len(files)} files in {up_dir}")


def main():
    release_root = WORKSPACE_ROOT / "release"
    zenodo_dir = release_root / "zenodo" / "padalarang-mmtraffic-v1.0.0"
    github_dir = release_root / "github" / "padalarang-mmtraffic"

    build_zenodo_package(zenodo_dir)
    build_github_repo(github_dir)
    bundle_for_upload(zenodo_dir, release_root / "zenodo_upload" / "padalarang-mmtraffic-v1.0.0")
    print(f"\n[✓] Successfully built complete release package in {release_root}!")


if __name__ == "__main__":
    main()

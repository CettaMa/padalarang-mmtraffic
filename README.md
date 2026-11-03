# Padalarang Multi-Modal Mixed Urban Traffic Pipeline

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
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running Verification Tests
```bash
pytest tools/test_cctv_status.py tools/test_event_features.py tools/test_speed_val.py tools/test_component_eval.py
```
`tools/test_traffic_metrics.py` needs raw video segments, which are not distributed, and is therefore not part of this command.

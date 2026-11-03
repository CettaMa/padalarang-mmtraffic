import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cctv_status import annotate_text, classify, load_status_map  # noqa: E402

CSV = ("cctv_name,date,timestampsafe,ROI_1_Vol_car,evt_holiday\n"
       "cam,2026-09-09,422,3,0\n"
       "cam,2026-09-09,922,0,0\n"
       "cam,2026-09-09,1422,5,0\n")


def test_classify():
    assert classify("no_frames_in_window") == "outage"
    assert classify("truncated_segments: 1/10") == "partial"
    assert classify("roi_not_found") == "ok"
    assert classify(None) == "ok"


def test_annotate_inserts_before_evt_and_preserves_cells():
    st = {("2026-09-09", "cam", "000922"): "outage", ("2026-09-09", "cam", "001422"): "partial"}
    out, counts = annotate_text(CSV, st)
    lines = out.splitlines()
    assert lines[0] == "cctv_name,date,timestampsafe,ROI_1_Vol_car,cctv_status,evt_holiday"
    assert [l.split(",")[4] for l in lines[1:]] == ["ok", "outage", "partial"]
    assert lines[1].split(",")[2] == "422" and counts == {"ok": 1, "partial": 1, "outage": 1}
    # idempotent
    assert annotate_text(out, st)[0] == out


def test_outage_wins_over_partial(tmp_path):
    (tmp_path / "2026-09-09_missing_report.json").write_text(
        '[{"date":"2026-09-09","cctv":"cam","timestampsafe":"922","reason":"truncated_segments: 1/2"},'
        '{"date":"2026-09-09","cctv":"cam","timestampsafe":922,"reason":"no_frames_in_window"}]')
    assert load_status_map(tmp_path)[("2026-09-09", "cam", "000922")] == "outage"

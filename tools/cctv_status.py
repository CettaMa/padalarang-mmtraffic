"""Add a per-row `cctv_status` flag to the released feature CSVs (review point 4).

Outage rows are stored with zero ROI features; the only record of them is
extracted/reports/<date>_missing_report.json. This tool copies that
information into the CSV so each file is self-describing:

  ok      window fully decoded (no report entry)
  partial some segments truncated or failed, features cover part of the window
  outage  no decodable video in the window, ROI features are zero placeholders

Values of every existing column are preserved byte-for-byte (all cells are
handled as strings); the new column is inserted immediately before
`evt_holiday`.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = REPO_ROOT / "extracted"
COLUMN = "cctv_status"
BEFORE = "evt_holiday"
OUTAGE = ("no_frames_in_window", "segments_open_failed", "no_ts_segments",
          "no_segments_in_window", "no_metadata_csv")
PARTIAL = ("truncated_segments", "partial_segments_failed")


def classify(reason: str | None) -> str:
    r = (reason or "").strip()
    if any(r.startswith(p) for p in OUTAGE):
        return "outage"
    if any(r.startswith(p) for p in PARTIAL):
        return "partial"
    return "ok"


def load_status_map(reports_dir: Path) -> dict[tuple[str, str, str], str]:
    """(date, cctv, timestampsafe) -> status; outage wins over partial."""
    rank = {"ok": 0, "partial": 1, "outage": 2}
    out: dict[tuple[str, str, str], str] = {}
    for rf in sorted(reports_dir.glob("*_missing_report.json")):
        for it in json.loads(rf.read_text(encoding="utf-8")):
            ts = it.get("timestampsafe")
            if ts in (None, ""):
                continue
            key = (str(it["date"]), str(it["cctv"]), str(ts).zfill(6))
            st = classify(it.get("reason"))
            if rank[st] > rank[out.get(key, "ok")]:
                out[key] = st
    return out


def annotate_text(text: str, status: dict) -> tuple[str, dict]:
    rows = list(csv.reader(io.StringIO(text, newline="")))
    head = rows[0]
    if COLUMN in head:
        i = head.index(COLUMN)
        head = head[:i] + head[i + 1:]
        rows = [head] + [r[:i] + r[i + 1:] for r in rows[1:]]
    pos = head.index(BEFORE)
    ci, di, ti = head.index("cctv_name"), head.index("date"), head.index("timestampsafe")
    counts = {"ok": 0, "partial": 0, "outage": 0}
    new = [head[:pos] + [COLUMN] + head[pos:]]
    for r in rows[1:]:
        st = status.get((r[di], r[ci], r[ti].zfill(6)), "ok")
        counts[st] += 1
        new.append(r[:pos] + [st] + r[pos:])
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\n").writerows(new)
    return buf.getvalue(), counts


def cmd_annotate(args) -> int:
    ext = Path(args.extracted)
    status = load_status_map(ext / "reports")
    bak = ext / "_cctv_bak"
    have = {rf.name[:10] for rf in (ext / "reports").glob("*_missing_report.json")}
    files = sorted(p for p in ext.glob("*_metadata_features.csv") if p.stem.split("_")[-3] in have)
    if not args.dry_run:
        bak.mkdir(exist_ok=True)
    tot = {"ok": 0, "partial": 0, "outage": 0}
    for p in files:
        raw = p.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        text, c = annotate_text(raw.replace("\r\n", "\n"), status)
        for k, v in c.items():
            tot[k] += v
        if args.dry_run:
            continue
        b = bak / p.name
        if not b.exists():
            shutil.copy2(p, b)
        p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print(f"{len(files)} files {'(dry run) ' if args.dry_run else ''}{tot}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracted", default=str(EXTRACTED))
    ap.add_argument("--dry-run", action="store_true")
    return cmd_annotate(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

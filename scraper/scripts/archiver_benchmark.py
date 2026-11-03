#!/usr/bin/env python3
"""Benchmark file archivers / compression tools on the CCTV dataset.

Measures per method:
  - archive (compress) wall time and CPU time
  - decompress wall time
  - archive size and compression ratio
  - throughput (MB/s) compress & decompress
  - peak memory usage (max RSS)
  - integrity verification (roundtrip byte-exact check)

Usage:
    python3 scripts/archiver_benchmark.py [--data-dir DIR] [--out CSV] [--methods ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


ZIP_PACK_FMT = (
    "import sys, zipfile, os; arc, src = sys.argv[1], sys.argv[2]; "
    "zf = zipfile.ZipFile(arc, 'w', zipfile.ZIP_DEFLATED, compresslevel={level}); "
    "[zf.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), src)) "
    "for r, _, fs in os.walk(src) for f in fs]; zf.close()"
)
ZIP_UNPACK = "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])"
ZIP_TEST = (
    "import sys, zipfile; sys.exit(0 if zipfile.ZipFile(sys.argv[1]).testzip() is None else 1)"
)


@dataclass
class Row:
    method: str
    src_bytes: int
    arc_bytes: int = 0
    ratio: float = 0.0  # src / arc
    saved_pct: float = 0.0  # 1 - arc/src
    archive_s: float = 0.0
    archive_cpu_s: float = 0.0
    archive_mbps: float = 0.0
    decompress_s: float = 0.0
    decompress_mbps: float = 0.0
    verify_s: float = 0.0
    peak_rss_mb: float = 0.0
    verified: bool = False


# method -> (pack cmd list template, unpack cmd list template, test cmd)
# placeholders: {SRC} {ARC} {OUT}
METHODS: dict[str, tuple[list[str], list[str], list[str] | None]] = {
    # store baseline (tar, no compression)
    "tar_store": (
        ["tar", "-cf", "{ARC}", "-C", "{DATA}", "."],
        ["tar", "-xf", "{ARC}", "-C", "{OUT}"],
        None,
    ),
    "tar_gzip6": (
        ["tar", "-czf", "{ARC}", "-C", "{DATA}", "."],
        ["tar", "-xzf", "{ARC}", "-C", "{OUT}"],
        ["gzip", "-t", "{ARC}"],
    ),
    "tar_pigz6": (
        ["tar", "-cf", "-", "-C", "{DATA}", ".", "|", "pigz", "-6", ">", "{ARC}"],
        ["pigz", "-dc", "{ARC}", "|", "tar", "-xf", "-", "-C", "{OUT}"],
        ["gzip", "-t", "{ARC}"],
    ),
    "tar_bzip2": (
        ["tar", "-cjf", "{ARC}", "-C", "{DATA}", "."],
        ["tar", "-xjf", "{ARC}", "-C", "{OUT}"],
        ["bzip2", "-t", "{ARC}"],
    ),
    "tar_xz6": (
        ["tar", "-cJf", "{ARC}", "-C", "{DATA}", "."],
        ["tar", "-xJf", "{ARC}", "-C", "{OUT}"],
        ["xz", "-t", "{ARC}"],
    ),
    "tar_zstd3": (
        ["tar", "--zstd", "-cf", "{ARC}", "-C", "{DATA}", "."],
        ["tar", "--zstd", "-xf", "{ARC}", "-C", "{OUT}"],
        ["zstd", "-t", "{ARC}"],
    ),
    "tar_zstd19": (
        ["tar", "-cf", "-", "-C", "{DATA}", ".", "|", "zstd", "-19", "-T0", "-q", "-o", "{ARC}"],
        ["zstd", "-dc", "{ARC}", "|", "tar", "-xf", "-", "-C", "{OUT}"],
        ["zstd", "-t", "{ARC}"],
    ),
    "zip_deflate": (
        ["python3", "-c", ZIP_PACK_FMT.format(level=6), "{ARC}", "{DATA}"],
        ["python3", "-c", ZIP_UNPACK, "{ARC}", "{OUT}"],
        ["python3", "-c", ZIP_TEST, "{ARC}"],
    ),
    "zip_deflate9": (
        ["python3", "-c", ZIP_PACK_FMT.format(level=9), "{ARC}", "{DATA}"],
        ["python3", "-c", ZIP_UNPACK, "{ARC}", "{OUT}"],
        ["python3", "-c", ZIP_TEST, "{ARC}"],
    ),
}


def run_meas(cmd: list[str], cwd: Path | None = None) -> tuple[float, float, float]:
    """Run cmd; return (wall_s, cpu_s, peak_rss_mb) of children."""
    if "|" in cmd or ">" in cmd:  # pipeline -> needs a shell
        cmd = " ".join(cmd)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str))
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    rss = after.ru_maxrss / 1024  # Linux: KB -> MB
    return wall, cpu, rss


def fill(t: list[str], **kw) -> list[str]:
    return [s.format(**kw) for s in t]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=REPO / "dataset/2026-09-03")
    ap.add_argument("--out", type=Path, default=REPO / "benchmark_archiver_results.csv")
    ap.add_argument("--methods", nargs="*", default=None, help="subset of: " + ", ".join(METHODS))
    ap.add_argument("--tmpdir", type=Path, default=Path.home() / ".cache/archbench")
    args = ap.parse_args()

    methods = args.methods or list(METHODS)
    data = args.data_dir.resolve()
    src_bytes = sum(f.stat().st_size for f in data.rglob("*") if f.is_file())
    print(
        f"Data: {data}  ({src_bytes / 1e6:.1f} MB, "
        f"{sum(1 for _ in data.rglob('*') if _.is_file())} files)"
    )
    print(f"Methods: {', '.join(methods)}\n")

    rows: list[Row] = []
    tmp = args.tmpdir
    tmp.mkdir(parents=True, exist_ok=True)

    for name in methods:
        if name not in METHODS:
            sys.exit(f"unknown method: {name}")
        pack, unpack, verify = METHODS[name]
        arc = (
            tmp
            / f"{name}.{'tar' if name == 'tar_store' else 'tar.gz' if 'gzip' in name or 'pigz' in name else 'tar.bz2' if 'bzip2' in name else 'tar.xz' if 'xz' in name else 'tar.zst' if 'zstd' in name else 'zip'}"
        )
        outdir = tmp / f"out_{name}"
        outdir.mkdir(exist_ok=True)
        kw = {"ARC": str(arc), "DATA": str(data), "OUT": str(outdir)}

        print(f"== {name} ==", flush=True)
        try:
            # warm page cache is fine; all methods treated equally
            wall, cpu, rss = run_meas(fill(pack, **kw))
            size = arc.stat().st_size
            w2, _, _ = run_meas(fill(unpack, **kw))
            # verify: re-hash extracted vs source
            t0 = time.perf_counter()

            def hash_dir(d: Path) -> str:
                h = hashlib.sha256()
                for p in sorted(d.rglob("*")):
                    if p.is_file():
                        h.update(str(p.relative_to(d)).encode())
                        with open(p, "rb") as fh:
                            for chunk in iter(lambda: fh.read(1 << 20), b""):
                                h.update(chunk)
                return h.hexdigest()

            ok = hash_dir(data) == hash_dir(outdir)
            verify_t = time.perf_counter() - t0

            row = Row(
                method=name,
                src_bytes=src_bytes,
                arc_bytes=size,
                ratio=round(src_bytes / size, 3),
                saved_pct=round(100 * (1 - size / src_bytes), 2),
                archive_s=round(wall, 2),
                archive_cpu_s=round(cpu, 2),
                archive_mbps=round(src_bytes / wall / 1e6, 2),
                decompress_s=round(w2, 2),
                decompress_mbps=round(src_bytes / w2 / 1e6, 2),
                verify_s=round(verify_t, 2),
                peak_rss_mb=round(rss, 1),
                verified=ok,
            )
            rows.append(row)
            print(
                f"   {size / 1e6:8.1f} MB  ratio {row.ratio:5.2f}x  "
                f"saved {row.saved_pct:5.1f}%  arch {wall:6.1f}s "
                f"({row.archive_mbps:6.1f} MB/s)  unarch {w2:6.1f}s "
                f"({row.decompress_mbps:6.1f} MB/s)  RSS {rss:6.0f} MB  "
                f"verified={ok}"
            )
        except Exception as e:
            print(f"   FAILED: {e}", file=sys.stderr)
        finally:
            arc.unlink(missing_ok=True)
            shutil.rmtree(outdir, ignore_errors=True)
        print(flush=True)

    shutil.rmtree(tmp, ignore_errors=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        w.writerows([asdict(r) for r in rows])
    print(f"Results written to {args.out}\n")

    hdr = (
        f"{'method':<15}{'size MB':>10}{'ratio':>8}{'saved%':>8}"
        f"{'arch s':>9}{'MB/s':>9}{'unc s':>8}{'MB/s':>9}{'RSS MB':>9}{'ok':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r.ratio):
        print(
            f"{r.method:<15}{r.arc_bytes / 1e6:>10.1f}{r.ratio:>8.2f}"
            f"{r.saved_pct:>8.1f}{r.archive_s:>9.1f}{r.archive_mbps:>9.1f}"
            f"{r.decompress_s:>8.1f}{r.decompress_mbps:>9.1f}"
            f"{r.peak_rss_mb:>9.0f}{str(r.verified):>5}"
        )


if __name__ == "__main__":
    main()

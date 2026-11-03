#!/usr/bin/env python3
"""Benchmark various video compression codecs on CCTV dataset clips.

Usage:
    python3 scripts/compression_benchmark.py [--sample-dir DIR] [--n-samples N] [--out CSV]

For each sample clip, transcoding is done with several codecs at
quality-equivalent settings, measuring:
  - compressed size / compression ratio
  - encode time (wall clock, single run)
  - decode time
  - PSNR / SSIM (quality vs. decoded source)

Results are written to CSV and a summary table printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (name, ffmpeg vcodec args, output ext)
CODECS: dict[str, list[str]] = {
    # H.264: baseline, main-equivalent quality, high quality
    "h264_crf28": ["-c:v", "libx264", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p"],
    "h264_crf23": ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"],
    "h264_ultrafast": [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
    ],
    # H.265/HEVC
    "h265_crf32": ["-c:v", "libx265", "-preset", "medium", "-crf", "32", "-pix_fmt", "yuv420p"],
    "h265_crf28": ["-c:v", "libx265", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p"],
    # AV1 (SVT-AV1, fast preset)
    "av1_crf35": ["-c:v", "libsvtav1", "-preset", "8", "-crf", "35", "-pix_fmt", "yuv420p"],
    "av1_crf30": ["-c:v", "libsvtav1", "-preset", "8", "-crf", "30", "-pix_fmt", "yuv420p"],
}


@dataclass
class Row:
    clip: str
    codec: str
    src_bytes: int
    dst_bytes: int = 0
    ratio: float = 0.0
    duration_s: float = 0.0
    encode_time_s: float = 0.0
    encode_fps: float = 0.0
    decode_time_s: float = 0.0
    psnr_db: float = field(default=float("nan"))
    ssim: float = field(default=float("nan"))


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe_duration(path: Path) -> float:
    p = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        return float(json.loads(p.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def transcode(src: Path, dst: Path, codec_args: list[str]) -> float:
    """Transcode src -> dst, return wall-clock encode time (seconds)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        *codec_args,
        "-an",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    t0 = time.perf_counter()
    p = run(cmd)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed:\n{p.stderr[-2000:]}")
    return dt


def decode_time(src: Path) -> float:
    """Time a pure decode (null output) pass."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-f", "null", "-"]
    t0 = time.perf_counter()
    p = run(cmd)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{p.stderr[-2000:]}")
    return dt


def quality_metrics(ref: Path, dist: Path) -> tuple[float, float]:
    """PSNR (dB) and SSIM of dist vs ref (both decoded)."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(ref),
        "-i",
        str(dist),
        "-lavfi",
        "[0:v]setpts=PTS-STARTPTS,format=yuv420p,split=2[r1][r2];"
        "[1:v]setpts=PTS-STARTPTS,format=yuv420p,split=2[d1][d2];"
        "[r1][d1]psnr;[r2][d2]ssim",
        "-f",
        "null",
        "-",
    ]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg metrics failed:\n{p.stderr[-2000:]}")
    err = p.stderr
    psnr = ssim = float("nan")
    for line in err.splitlines():
        if "PSNR" in line and "average:" in line:
            try:
                psnr = float(line.split("average:")[1].split()[0])
            except Exception:
                pass
        if "SSIM" in line and "All:" in line:
            try:
                ssim = float(line.split("All:")[1].split()[0])
            except Exception:
                pass
    return psnr, ssim


def collect_clips(root: Path, n: int, seed: int = 42) -> list[Path]:
    # skip empty/corrupt placeholder files (some failed downloads are 0 bytes)
    ts_files = sorted(f for f in root.glob("*/videos/*.ts") if f.stat().st_size > 10_000)
    if not ts_files:
        sys.exit("No .ts clips found under " + str(root))
    by_dir: dict[Path, list[Path]] = {}
    for f in ts_files:
        by_dir.setdefault(f.parent.parent, []).append(f)
    rng = random.Random(seed)
    picked: list[Path] = []
    # round-robin across locations for a balanced sample
    for files in by_dir.values():
        if len(picked) >= n:
            break
        picked.append(rng.choice(files))
    while len(picked) < n:
        picked.append(rng.choice(ts_files))
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=REPO / "dataset/2026-09-03")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out", type=Path, default=REPO / "benchmark_compression_results.csv")
    ap.add_argument("--keep-outputs", action="store_true")
    args = ap.parse_args()

    clips = collect_clips(args.dataset_dir, args.n_samples)
    print(f"Benchmarking {len(clips)} clips x {len(CODECS)} codec settings\n")

    rows: list[Row] = []
    tmp = Path(tempfile.mkdtemp(prefix="compbench_"))
    try:
        for clip in clips:
            src_size = clip.stat().st_size
            print(f"== {clip.name} ({src_size / 1e6:.2f} MB) ==")
            ref_decoded = tmp / "ref_decoded.ts"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(clip),
                    "-c",
                    "copy",
                    "-f",
                    "mpegts",
                    str(ref_decoded),
                ]
            )
            dur = probe_duration(clip)

            for name, cargs in CODECS.items():
                dst = tmp / f"{name}.mp4"
                try:
                    enc_t = transcode(clip, dst, cargs)
                    dec_t = decode_time(dst)
                    psnr, ssim = quality_metrics(clip, dst)
                    dst_size = dst.stat().st_size
                    row = Row(
                        clip=clip.name,
                        codec=name,
                        src_bytes=src_size,
                        dst_bytes=dst_size,
                        ratio=src_size / dst_size if dst_size else 0.0,
                        duration_s=dur,
                        encode_time_s=round(enc_t, 3),
                        encode_fps=round(dur / enc_t, 2) if enc_t else 0.0,
                        decode_time_s=round(dec_t, 3),
                        psnr_db=round(psnr, 3) if psnr == psnr else psnr,
                        ssim=round(ssim, 5) if ssim == ssim else ssim,
                    )
                    rows.append(row)
                    print(
                        f"   {name:<18} {dst_size / 1e6:6.2f} MB  "
                        f"ratio {row.ratio:5.2f}x  enc {enc_t:6.2f}s  "
                        f"PSNR {psnr:6.2f}  SSIM {ssim:.4f}"
                    )
                except Exception as e:
                    print(f"   {name:<18} FAILED: {e}", file=sys.stderr)
                finally:
                    if not args.keep_outputs:
                        dst.unlink(missing_ok=True)
            print()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        w.writerows([asdict(r) for r in rows])
    print(f"Per-clip results written to {args.out}\n")

    # ---- aggregate summary ----
    agg: dict[str, list[Row]] = {}
    for r in rows:
        agg.setdefault(r.codec, []).append(r)

    print(
        f"{'codec':<18}{'avg size MB':>12}{'avg ratio':>11}{'enc s':>8}"
        f"{'enc fps':>9}{'dec s':>8}{'PSNR dB':>10}{'SSIM':>9}"
    )
    print("-" * 85)
    for name, rs in agg.items():
        n = len(rs)
        print(
            f"{name:<18}"
            f"{sum(r.dst_bytes for r in rs) / n / 1e6:>12.2f}"
            f"{sum(r.ratio for r in rs) / n:>11.2f}"
            f"{sum(r.encode_time_s for r in rs) / n:>8.2f}"
            f"{sum(r.encode_fps for r in rs) / n:>9.2f}"
            f"{sum(r.decode_time_s for r in rs) / n:>8.2f}"
            f"{sum(r.psnr_db for r in rs) / n:>10.2f}"
            f"{sum(r.ssim for r in rs) / n:>9.4f}"
        )


if __name__ == "__main__":
    main()

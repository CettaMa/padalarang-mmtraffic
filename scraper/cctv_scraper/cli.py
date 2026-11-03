import argparse

from cctv_scraper.app import CCTVApp
from cctv_scraper.config import load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCTV 24x7 scraper with FFmpeg segment mode.")
    parser.add_argument("--config", default=None, help="Path to cctv_points.csv")
    parser.add_argument("--output", default=None, help="Output root directory")
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=None,
        help="Durasi tiap segment video. Jika kosong, memakai SEGMENT_SECONDS dari .env atau default 60.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Jumlah hari penyimpanan sebelum folder lama dihapus.",
    )
    parser.add_argument(
        "--min-free-space-gb",
        type=float,
        default=None,
        help="Batas minimum sisa storage dalam GB.",
    )
    parser.add_argument(
        "--video-container",
        choices=["ts", "mp4"],
        default=None,
        help="Use 'ts' for robust HLS capture or 'mp4' for direct MP4 segments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args)
    app = CCTVApp(config)
    app.start()

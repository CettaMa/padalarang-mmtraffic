import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from cctv_scraper.config import ensure_dir


def setup_logging(
    output_root: Path, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5
) -> None:
    ensure_dir(output_root)
    ensure_dir(output_root / "logs")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        output_root / "logs" / "scraper.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def point_logger(
    output_root: Path,
    point_name: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(f"cctv.{point_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = True

    marker = f"{point_name}.log"
    log_path = (output_root / "logs" / marker).resolve()
    for handler in logger.handlers:
        if (
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename).resolve() == log_path
        ):
            return logger

    ensure_dir(output_root / "logs")
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger

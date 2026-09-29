"""Logging configuration utilities.

Google-style docstrings are used throughout the project.
"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure project logging to console and a rotating-style file target.

    Args:
        log_file: Destination file for persistent logs.
        level: Root logging level.

    Returns:
        A configured project logger.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("chess_behavior")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

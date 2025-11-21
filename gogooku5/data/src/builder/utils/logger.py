"""Logging utilities for the dataset builder."""

from __future__ import annotations

import logging
from logging import Logger
from typing import Optional

_DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"


def configure_logger(name: str = "builder", level: int = logging.INFO) -> Logger:
    """Configure and return a logger with standardized formatting."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_DEFAULT_LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> Logger:
    """Return a child logger derived from the root builder logger."""

    root = configure_logger()
    return root if name is None else root.getChild(name)

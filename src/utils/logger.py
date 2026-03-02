"""Structured logging setup for the project."""

import os
import sys
import logging
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "ta-pipeline",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure and return a logger with file + console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files. If None, console only.
        level: Logging level
        console: Whether to add console handler

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Log file: {log_file}")

    return logger


def get_logger(name: str = "ta-pipeline") -> logging.Logger:
    """Get existing logger by name. If not set up, returns a basic one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class PhaseLogger:
    """Context manager for logging phases with timing."""

    def __init__(self, phase_name: str, logger: Optional[logging.Logger] = None):
        self.phase_name = phase_name
        self.logger = logger or get_logger()
        self._start = None

    def __enter__(self):
        import time
        self._start = time.time()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"PHASE START: {self.phase_name}")
        self.logger.info(f"{'='*60}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self._start
        if exc_type is None:
            self.logger.info(f"PHASE DONE: {self.phase_name} ({elapsed:.1f}s)")
        else:
            self.logger.error(f"PHASE FAILED: {self.phase_name} ({elapsed:.1f}s) - {exc_val}")
        return False

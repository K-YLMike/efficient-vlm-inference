"""Structured logging to stdout.

Logs land in ``logs/%x_%j.out`` under Slurm; real tracebacks land in the
matching ``.err``. A single timestamped format keeps job logs greppable.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to stdout once per process."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

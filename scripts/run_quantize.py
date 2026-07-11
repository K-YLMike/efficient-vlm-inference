"""Produce every quantized checkpoint that is not already done.

Iterates the quant configs, skipping finished ones (via `_DONE.json`), and
quantizes the rest one at a time. Safe to resubmit: a job killed mid-config
recomputes only that config on the next run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import load_calibration_texts, prepare_calibration  # noqa: E402,E501
from src.quantize import quantize_one  # noqa: E402
from src.utils.config import load_config, quant_configs  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

LOGGER = get_logger("run_quantize")


def main() -> None:
    """Quantize all pending configs using the shared calibration set."""
    cfg = load_config()
    prepare_calibration(cfg)
    calib_texts = load_calibration_texts(cfg)

    did_any = False
    for entry in quant_configs(cfg):
        did_any = quantize_one(entry, cfg, calib_texts) or did_any
    if not did_any:
        LOGGER.info("all quant configs already complete")


if __name__ == "__main__":
    main()

"""Build a calibration set for post-training quantization.

GPTQ, AWQ, and SmoothQuant all need a small batch of forward passes to
estimate activation statistics. The calibration corpus is wikitext-2, a
general text source with no overlap with the MMLU test split or the
HumanEval problems. Calibrating on the evaluation data would contaminate
the accuracy numbers, so it is deliberately avoided.
"""

import os
from typing import List

from src.utils.io_atomic import atomic_write_json, is_done, write_done
from src.utils.logging_setup import get_logger

LOGGER = get_logger("calibration")
_CALIB_FILE = "calib_texts.json"


def build_calibration_texts(
    num_samples: int,
    min_chars: int,
    cache_dir: str,
) -> List[str]:
    """Return ``num_samples`` non-trivial text chunks from wikitext-2.

    Chunks shorter than ``min_chars`` are dropped so calibration sees
    dense text rather than headers or blank lines.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
    texts: List[str] = []
    for row in dataset:
        text = row["text"].strip()
        if len(text) >= min_chars:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    if len(texts) < num_samples:
        LOGGER.warning(
            "only %d calibration chunks available (< %d requested)",
            len(texts), num_samples)
    return texts


def prepare_calibration(cfg: dict) -> str:
    """Materialize the calibration set to disk once and return its path.

    Idempotent: a `_DONE.json` marker in the calibration directory makes a
    re-run an instant no-op.
    """
    calib_cfg = cfg.get("calibration", {})
    calib_dir = cfg["resolved"]["calib_dir"]
    out_path = os.path.join(calib_dir, _CALIB_FILE)

    if is_done(calib_dir) and os.path.isfile(out_path):
        LOGGER.info("calibration already prepared at %s", out_path)
        return out_path

    texts = build_calibration_texts(
        num_samples=calib_cfg.get("num_samples", 512),
        min_chars=calib_cfg.get("min_chars", 256),
        cache_dir=cfg["resolved"]["data_dir"],
    )
    atomic_write_json(out_path, texts)
    write_done(calib_dir, {"num_samples": len(texts), "source": "wikitext-2"})
    LOGGER.info("wrote %d calibration texts to %s", len(texts), out_path)
    return out_path


def load_calibration_texts(cfg: dict) -> List[str]:
    """Load the previously materialized calibration texts."""
    from src.utils.io_atomic import load_json
    out_path = os.path.join(cfg["resolved"]["calib_dir"], _CALIB_FILE)
    return load_json(out_path)

"""Download the base model, calibration corpus, and eval datasets once.

Run this in an interactive session with internet access. Later GPU jobs set
HF_HUB_OFFLINE=1 and read only from the populated cache, so compute nodes
never need network access.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import prepare_calibration  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

LOGGER = get_logger("run_download")


def main() -> None:
    """Fetch model weights and datasets into the local cache."""
    cfg = load_config()
    data_dir = cfg["resolved"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    base_model = cfg["project"]["base_model"]
    LOGGER.info("downloading model %s", base_model)
    snapshot_download(repo_id=base_model)

    LOGGER.info("downloading MMLU")
    load_dataset("cais/mmlu", "all", split="test", cache_dir=data_dir)

    LOGGER.info("downloading HumanEval")
    load_dataset("openai_humaneval", split="test", cache_dir=data_dir)

    LOGGER.info("building calibration set")
    prepare_calibration(cfg)

    LOGGER.info("download stage complete")


if __name__ == "__main__":
    main()

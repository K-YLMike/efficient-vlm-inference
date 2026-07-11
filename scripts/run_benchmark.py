"""Benchmark throughput, latency, and memory for every pending config.

Each config is a separate work unit with its own `_DONE.json`, so the stage
is resumable and quick to rerun.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark import run_benchmark_unit  # noqa: E402
from src.utils.config import load_config, quant_configs  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

LOGGER = get_logger("run_benchmark")


def main() -> None:
    """Run the vLLM benchmark for each config not already measured."""
    cfg = load_config()
    did_any = False
    for entry in quant_configs(cfg):
        did_any = run_benchmark_unit(entry, cfg) or did_any
    if not did_any:
        LOGGER.info("all benchmarks already complete")


if __name__ == "__main__":
    main()

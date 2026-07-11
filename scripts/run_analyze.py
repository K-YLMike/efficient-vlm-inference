"""Build the Pareto frontier, degradation table, and figures.

Reads only JSON results, so it runs on CPU (the `short` partition) after
the GPU stages have populated ``results/``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyze import run_analysis  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

LOGGER = get_logger("run_analyze")


def main() -> None:
    """Aggregate results and write summary.json plus Pareto figures."""
    cfg = load_config()
    run_analysis(cfg)


if __name__ == "__main__":
    main()

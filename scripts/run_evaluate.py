"""Evaluate every pending (config, task) unit on MMLU and HumanEval.

Work units are (config x task). MMLU is internally sharded by subject and
HumanEval by task id, so a job killed at the 1-hour wall clock resumes on
only the missing shards. One config is loaded into vLLM at a time to bound
GPU memory; finished units are skipped.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate_humaneval import evaluate_humaneval  # noqa: E402
from src.evaluate_mmlu import evaluate_mmlu  # noqa: E402
from src.utils.config import checkpoint_dir, load_config, \
    quant_configs  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

LOGGER = get_logger("run_evaluate")


def main() -> None:
    """Run enabled evaluations for each config, one unit per invocation.

    Returns after doing a single unit of work so that each chained job
    makes bounded progress; the chain drives the stage to completion.
    """
    cfg = load_config()
    eval_cfg = cfg.get("evaluate", {})
    mmlu_on = eval_cfg.get("mmlu", {}).get("enabled", True)
    he_on = eval_cfg.get("humaneval", {}).get("enabled", True)

    for entry in quant_configs(cfg):
        name = entry["name"]
        model_path = checkpoint_dir(cfg, name)
        quant_kind = entry.get("vllm_quantization", "auto")

        if mmlu_on and evaluate_mmlu(name, model_path, quant_kind, cfg):
            return
        if he_on and evaluate_humaneval(name, model_path, quant_kind, cfg):
            return

    LOGGER.info("all evaluations already complete")


if __name__ == "__main__":
    main()

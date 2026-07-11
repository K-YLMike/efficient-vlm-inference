"""Measure serving-side throughput, latency, and memory for one config.

Inference runs through vLLM (batched, the production serving path the
proposal targets), so throughput reflects continuous batching rather than
one-request-at-a-time decoding.

Metrics reported per config:

* decode_tokens_per_s : total generated tokens / wall time over a fixed
                        batch of prompts.
* ttft_ms_median      : median time-to-first-token, taken from vLLM's
                        per-request metrics (first_token_time - arrival).
* checkpoint_gb       : on-disk weight footprint, the memory axis of the
                        Pareto plot. It is exact, framework-independent,
                        and comparable across bit-widths; end-to-end
                        serving also adds a separately-quantizable KV
                        cache, noted in the README.
"""

import glob
import os
import statistics
import time
from typing import Any, Dict, List

from src.utils.io_atomic import is_done, load_json, atomic_write_json, \
    write_done
from src.utils.logging_setup import get_logger

LOGGER = get_logger("benchmark")


def checkpoint_size_gb(path: str) -> float:
    """Return the summed size of weight shards under ``path`` in GB.

    Counts safetensors and bin shards; falls back to the whole directory
    if neither pattern matches (covers formats that name files unusually).
    """
    if not os.path.isdir(path):
        return 0.0
    patterns = ["*.safetensors", "*.bin"]
    total = 0
    for pattern in patterns:
        for shard in glob.glob(os.path.join(path, pattern)):
            total += os.path.getsize(shard)
    if total == 0:
        for root, _dirs, files in os.walk(path):
            for name in files:
                total += os.path.getsize(os.path.join(root, name))
    return total / 1e9


def _build_prompts(tokenizer, num_prompts: int, input_len: int) -> List[str]:
    """Build ``num_prompts`` prompts of roughly ``input_len`` tokens.

    A fixed synthetic prompt controls input length so throughput and TTFT
    are compared at the same working point across configs.
    """
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    token_ids = tokenizer(filler)["input_ids"][:input_len]
    prompt = tokenizer.decode(token_ids)
    return [prompt for _ in range(num_prompts)]


def benchmark_one(
    name: str,
    model_path: str,
    quant_kind: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the vLLM benchmark for a single config and return its metrics."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    bench = cfg.get("benchmark", {})
    num_prompts = bench.get("num_prompts", 128)
    input_len = bench.get("input_len", 512)
    output_len = bench.get("output_len", 128)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = _build_prompts(tokenizer, num_prompts, input_len)

    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "dtype": "auto",
        "gpu_memory_utilization": bench.get("gpu_memory_utilization", 0.85),
        "max_model_len": bench.get("max_model_len", 4096),
        "enforce_eager": bench.get("enforce_eager", False),
    }
    if quant_kind and quant_kind != "auto":
        llm_kwargs["quantization"] = quant_kind
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(temperature=0.0, max_tokens=output_len)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    wall = time.perf_counter() - start

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    ttft_ms: List[float] = []
    for out in outputs:
        metrics = getattr(out, "metrics", None)
        if metrics and metrics.first_token_time and metrics.arrival_time:
            ttft_ms.append(
                (metrics.first_token_time - metrics.arrival_time) * 1000.0)

    result = {
        "name": name,
        "decode_tokens_per_s": total_out_tokens / wall if wall > 0 else 0.0,
        "ttft_ms_median": statistics.median(ttft_ms) if ttft_ms else None,
        "checkpoint_gb": checkpoint_size_gb(model_path),
        "num_prompts": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "total_output_tokens": total_out_tokens,
        "wall_s": wall,
    }
    return result


def run_benchmark_unit(
    entry: Dict[str, Any],
    cfg: Dict[str, Any],
) -> bool:
    """Benchmark one config if not already done; return whether work ran."""
    name = entry["name"]
    unit_dir = os.path.join(cfg["resolved"]["results_dir"], "benchmark", name)
    if is_done(unit_dir):
        LOGGER.info("[%s] benchmark already done, skipping", name)
        return False

    from src.utils.config import checkpoint_dir
    model_path = checkpoint_dir(cfg, name)
    quant_kind = entry.get("vllm_quantization", "auto")

    LOGGER.info("[%s] benchmarking (model=%s)", name, model_path)
    result = benchmark_one(name, model_path, quant_kind, cfg)

    out_path = os.path.join(unit_dir, "metrics.json")
    atomic_write_json(out_path, result)
    write_done(unit_dir, {"name": name})
    LOGGER.info("[%s] tokens/s=%.1f ttft_ms=%s size_gb=%.2f",
                name, result["decode_tokens_per_s"],
                result["ttft_ms_median"], result["checkpoint_gb"])
    return True


def load_benchmark(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Load a previously written benchmark result for ``name``."""
    path = os.path.join(
        cfg["resolved"]["results_dir"], "benchmark", name, "metrics.json")
    return load_json(path)

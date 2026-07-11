"""Produce quantized checkpoints from an FP16 base model.

Two library families cover the four quantized configs so that GPTQ and AWQ
are genuinely different algorithms in the comparison, not two wrappers over
one implementation:

* llm-compressor  -> W8A16 (INT8 weight-only), W4A16 (INT4 GPTQ),
                     W8A8 (INT8 weight + activation, with SmoothQuant)
* autoawq         -> W4A16 AWQ (activation-aware, group-wise)

All heavy imports are lazy so the module imports cleanly without the
quantization backends installed. Each config writes its checkpoint to its
own directory and a `_DONE.json` marker only after the save succeeds, so a
timed-out job resumes by recomputing only the unfinished config.
"""

import os
from typing import Any, Dict, List

from src.utils.io_atomic import is_done, write_done
from src.utils.logging_setup import get_logger

LOGGER = get_logger("quantize")


def _quantize_llmcompressor(
    base_model: str,
    scheme: str,
    calib_texts: List[str],
    max_seq_len: int,
    out_dir: str,
    ignore: List[str],
    smoothquant: bool,
) -> None:
    """Quantize with llm-compressor's one-shot GPTQ / SmoothQuant recipe.

    ``scheme`` is one of ``W8A16``, ``W4A16``, ``W8A8``. SmoothQuant is
    prepended only for the W8A8 (activation-quantized) scheme, where it
    migrates activation outliers into the weights before quantization.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor.transformers import oneshot
    except ImportError:
        from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto")

    dataset = Dataset.from_dict({"text": calib_texts})

    recipe: List[Any] = []
    if smoothquant:
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        recipe.append(SmoothQuantModifier(smoothing_strength=0.8))
    recipe.append(
        GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore))

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=len(calib_texts),
        output_dir=out_dir,
    )
    tokenizer.save_pretrained(out_dir)


def _quantize_awq(
    base_model: str,
    group_size: int,
    calib_texts: List[str],
    out_dir: str,
) -> None:
    """Quantize with autoawq (4-bit, activation-aware, group-wise)."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoAWQForCausalLM.from_pretrained(base_model)
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": 4,
        "version": "GEMM",
    }
    model.quantize(
        tokenizer, quant_config=quant_config, calib_data=calib_texts)
    model.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)


def quantize_one(
    entry: Dict[str, Any],
    cfg: Dict[str, Any],
    calib_texts: List[str],
) -> bool:
    """Produce the checkpoint for a single quant config.

    Returns True if work was done, False if the config was already
    complete or requires no checkpoint (the fp16 baseline).
    """
    name = entry["name"]
    method = entry.get("method", "none")

    if method == "none":
        LOGGER.info("[%s] fp16 baseline uses the base model directly", name)
        return False

    out_dir = os.path.join(cfg["resolved"]["quantized_dir"], name)
    if is_done(out_dir):
        LOGGER.info("[%s] already quantized, skipping", name)
        return False

    base_model = cfg["project"]["base_model"]
    calib_cfg = cfg.get("calibration", {})
    max_seq_len = calib_cfg.get("max_seq_len", 2048)
    ignore = entry.get("ignore", ["lm_head"])

    LOGGER.info("[%s] quantizing with method=%s scheme=%s",
                name, method, entry.get("scheme"))

    if method == "llmcompressor":
        _quantize_llmcompressor(
            base_model=base_model,
            scheme=entry["scheme"],
            calib_texts=calib_texts,
            max_seq_len=max_seq_len,
            out_dir=out_dir,
            ignore=ignore,
            smoothquant=bool(entry.get("smoothquant", False)),
        )
    elif method == "autoawq":
        _quantize_awq(
            base_model=base_model,
            group_size=entry.get("group_size", 128),
            calib_texts=calib_texts,
            out_dir=out_dir,
        )
    else:
        raise ValueError("unknown quant method: {}".format(method))

    write_done(out_dir, {
        "name": name,
        "method": method,
        "scheme": entry.get("scheme"),
        "group_size": entry.get("group_size"),
    })
    LOGGER.info("[%s] checkpoint written to %s", name, out_dir)
    return True

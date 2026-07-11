"""Full MMLU evaluation for one quantized config, batched through vLLM.

MMLU is 4-way multiple choice across 57 subjects. Each question is rendered
as an instruction that asks for a single answer letter; the model's first
emitted letter is parsed and compared to the gold answer. The identical
prompt and parser are used for every quant config, so cross-config
comparisons stay apples-to-apples.

Resumability: work is sharded by subject. Each subject's result is written
atomically to its own file, and the task is marked done only once all
subjects are present. A job killed at the 1-hour wall clock resumes by
recomputing only the subjects still missing.
"""

import os
import re
import string
from typing import Any, Dict, List, Tuple

from src.utils.io_atomic import atomic_write_json, is_done, load_json, \
    write_done
from src.utils.logging_setup import get_logger

LOGGER = get_logger("evaluate_mmlu")
_LETTERS = string.ascii_uppercase[:4]
# Match an A-D letter only when it stands alone (bounded by non-word
# characters), so letters inside words like "ANSWER" are not picked up.
_LETTER_RE = re.compile(r"\b([A-D])\b")


def format_question(question: str, choices: List[str]) -> str:
    """Render one MMLU item as a single-letter-answer instruction prompt."""
    lines = [question.strip(), ""]
    for letter, choice in zip(_LETTERS, choices):
        lines.append("{}. {}".format(letter, choice))
    lines.append("")
    lines.append("Answer with a single letter (A, B, C, or D).")
    return "\n".join(lines)


def parse_answer(text: str) -> str:
    """Return the first standalone A-D letter in ``text``, else empty.

    An instruct model asked for a single letter usually emits either the
    bare letter or a short phrase such as "The answer is B"; matching only
    isolated letters avoids picking the "A" out of a word like "ANSWER".
    """
    match = _LETTER_RE.search(text.strip().upper())
    return match.group(1) if match else ""


def _chat_prompt(tokenizer, user_text: str) -> str:
    """Wrap ``user_text`` in the model's chat template when available."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return user_text


def _subjects(dataset) -> List[str]:
    """Return the sorted unique subject names present in the dataset."""
    return sorted(set(dataset["subject"]))


def _evaluate_subject(
    llm,
    tokenizer,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score every question for one subject and return its summary."""
    from vllm import SamplingParams

    prompts = [
        _chat_prompt(tokenizer, format_question(r["question"], r["choices"]))
        for r in rows
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=8)
    outputs = llm.generate(prompts, sampling)

    correct = 0
    records: List[Tuple[str, str, bool]] = []
    for row, out in zip(rows, outputs):
        pred = parse_answer(out.outputs[0].text)
        gold = _LETTERS[row["answer"]]
        hit = pred == gold
        correct += int(hit)
        records.append((pred, gold, hit))
    total = len(rows)
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


def evaluate_mmlu(
    name: str,
    model_path: str,
    quant_kind: str,
    cfg: Dict[str, Any],
) -> bool:
    """Evaluate one config on full MMLU; return whether any work ran.

    Loads vLLM once, then iterates subjects, skipping any already written.
    """
    unit_dir = os.path.join(
        cfg["resolved"]["results_dir"], "mmlu", name)
    if is_done(unit_dir):
        LOGGER.info("[%s] mmlu already done, skipping", name)
        return False

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM

    eval_cfg = cfg.get("evaluate", {}).get("mmlu", {})
    split = eval_cfg.get("split", "test")
    dataset = load_dataset(
        "cais/mmlu", "all", split=split,
        cache_dir=cfg["resolved"]["data_dir"])
    subjects = _subjects(dataset)

    subj_dir = os.path.join(unit_dir, "subjects")
    pending = [
        s for s in subjects
        if not os.path.isfile(os.path.join(subj_dir, s + ".json"))
    ]
    if not pending:
        _aggregate_mmlu(name, subjects, unit_dir)
        return False

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "dtype": "auto",
        "gpu_memory_utilization": eval_cfg.get("gpu_memory_utilization", 0.85),
        "max_model_len": eval_cfg.get("max_model_len", 4096),
    }
    if quant_kind and quant_kind != "auto":
        llm_kwargs["quantization"] = quant_kind
    llm = LLM(**llm_kwargs)

    by_subject = {}
    for subject in subjects:
        by_subject.setdefault(subject, [])
    for row in dataset:
        by_subject[row["subject"]].append(row)

    for subject in pending:
        LOGGER.info("[%s] scoring subject %s", name, subject)
        summary = _evaluate_subject(llm, tokenizer, by_subject[subject])
        atomic_write_json(
            os.path.join(subj_dir, subject + ".json"), summary)

    _aggregate_mmlu(name, subjects, unit_dir)
    return True


def _aggregate_mmlu(name: str, subjects: List[str], unit_dir: str) -> None:
    """Combine per-subject files into an overall accuracy and mark done."""
    subj_dir = os.path.join(unit_dir, "subjects")
    total_n = 0
    total_correct = 0
    per_subject = {}
    for subject in subjects:
        path = os.path.join(subj_dir, subject + ".json")
        if not os.path.isfile(path):
            LOGGER.warning("[%s] subject %s missing, cannot aggregate",
                           name, subject)
            return
        summary = load_json(path)
        per_subject[subject] = summary["accuracy"]
        total_n += summary["n"]
        total_correct += summary["correct"]

    result = {
        "name": name,
        "task": "mmlu",
        "accuracy": total_correct / total_n if total_n else 0.0,
        "n": total_n,
        "per_subject": per_subject,
    }
    atomic_write_json(os.path.join(unit_dir, "result.json"), result)
    write_done(unit_dir, {"name": name, "task": "mmlu", "n": total_n})
    LOGGER.info("[%s] mmlu accuracy=%.4f over %d questions",
                name, result["accuracy"], total_n)

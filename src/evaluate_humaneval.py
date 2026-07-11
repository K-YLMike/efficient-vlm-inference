"""HumanEval pass@1 for one quantized config, batched through vLLM.

The model completes each function signature greedily (one sample per task,
so the metric is pass@1). Completions are checked by running the problem's
own unit tests. Generated code is executed in a separate short-lived
subprocess with a wall-clock timeout, which isolates the parent process
from crashes, hangs, and stray side effects in model output.

Resumability: each task's pass/fail record is written atomically to its own
file; the task set is marked done only when every problem has a record, so a
killed job resumes on the remaining problems.
"""

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

from src.utils.io_atomic import atomic_write_json, is_done, load_json, \
    write_done
from src.utils.logging_setup import get_logger

LOGGER = get_logger("evaluate_humaneval")
_STOP = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]


def _truncate_completion(text: str) -> str:
    """Cut a completion at the first token that leaves the target function.

    The model often keeps generating extra definitions after the requested
    function body; those would break execution, so generation is trimmed at
    the earliest stop marker.
    """
    cut = len(text)
    for marker in _STOP:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def build_program(problem: Dict[str, Any], completion: str) -> str:
    """Assemble the full runnable program for one HumanEval task."""
    body = _truncate_completion(completion)
    return "\n".join([
        problem["prompt"] + body,
        problem["test"],
        "check({})".format(problem["entry_point"]),
        "",
    ])


def run_program(source: str, timeout_s: float) -> Dict[str, Any]:
    """Execute ``source`` in an isolated subprocess and return its outcome.

    Returns a dict with ``passed`` and a short ``detail`` string. A non-zero
    exit code, a timeout, or any exception all count as a failure.
    """
    with tempfile.TemporaryDirectory() as work_dir:
        prog_path = os.path.join(work_dir, "candidate.py")
        with open(prog_path, "w", encoding="utf-8") as handle:
            handle.write(source)
        try:
            proc = subprocess.run(
                [sys.executable, prog_path],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "detail": "timeout"}
        if proc.returncode == 0:
            return {"passed": True, "detail": "ok"}
        return {"passed": False, "detail": proc.stderr.strip()[-200:]}


def evaluate_humaneval(
    name: str,
    model_path: str,
    quant_kind: str,
    cfg: Dict[str, Any],
) -> bool:
    """Evaluate one config on HumanEval; return whether any work ran."""
    unit_dir = os.path.join(
        cfg["resolved"]["results_dir"], "humaneval", name)
    if is_done(unit_dir):
        LOGGER.info("[%s] humaneval already done, skipping", name)
        return False

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    eval_cfg = cfg.get("evaluate", {}).get("humaneval", {})
    dataset = load_dataset(
        "openai_humaneval", split="test",
        cache_dir=cfg["resolved"]["data_dir"])
    problems = list(dataset)

    task_dir = os.path.join(unit_dir, "tasks")
    pending = [
        p for p in problems
        if not os.path.isfile(
            os.path.join(task_dir, p["task_id"].replace("/", "_") + ".json"))
    ]
    if not pending:
        _aggregate_humaneval(name, problems, unit_dir)
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

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=eval_cfg.get("max_new_tokens", 512),
        stop=_STOP,
    )
    prompts = [p["prompt"] for p in pending]
    outputs = llm.generate(prompts, sampling)

    timeout_s = eval_cfg.get("exec_timeout_s", 10.0)
    for problem, out in zip(pending, outputs):
        completion = out.outputs[0].text
        program = build_program(problem, completion)
        outcome = run_program(program, timeout_s)
        record = {
            "task_id": problem["task_id"],
            "passed": outcome["passed"],
            "detail": outcome["detail"],
        }
        fname = problem["task_id"].replace("/", "_") + ".json"
        atomic_write_json(os.path.join(task_dir, fname), record)

    _aggregate_humaneval(name, problems, unit_dir)
    return True


def _aggregate_humaneval(
    name: str,
    problems: List[Dict[str, Any]],
    unit_dir: str,
) -> None:
    """Combine per-task records into pass@1 and mark the task done."""
    task_dir = os.path.join(unit_dir, "tasks")
    passed = 0
    total = 0
    failures: List[str] = []
    for problem in problems:
        fname = problem["task_id"].replace("/", "_") + ".json"
        path = os.path.join(task_dir, fname)
        if not os.path.isfile(path):
            LOGGER.warning("[%s] task %s missing, cannot aggregate",
                           name, problem["task_id"])
            return
        record = load_json(path)
        total += 1
        if record["passed"]:
            passed += 1
        else:
            failures.append(problem["task_id"])

    result = {
        "name": name,
        "task": "humaneval",
        "pass_at_1": passed / total if total else 0.0,
        "passed": passed,
        "n": total,
        "failed_task_ids": failures,
    }
    atomic_write_json(os.path.join(unit_dir, "result.json"), result)
    write_done(unit_dir, {"name": name, "task": "humaneval", "n": total})
    LOGGER.info("[%s] humaneval pass@1=%.4f over %d tasks",
                name, result["pass_at_1"], total)

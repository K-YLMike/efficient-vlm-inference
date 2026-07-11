"""GPU-free smoke tests for the pure-logic parts of the pipeline.

Covers atomic IO and done markers, resume behaviour, the Pareto frontier,
the degradation table, the MMLU prompt/parser, and the HumanEval completion
truncation and sandbox executor. Run with: python -m pytest tests -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyze import degradation_table, pareto_frontier  # noqa: E402
from src.evaluate_humaneval import build_program, run_program, \
    _truncate_completion  # noqa: E402
from src.evaluate_mmlu import format_question, parse_answer  # noqa: E402
from src.utils.io_atomic import atomic_write_json, is_done, load_json, \
    write_done  # noqa: E402


def test_atomic_write_and_marker(tmp_path):
    path = os.path.join(str(tmp_path), "sub", "value.json")
    atomic_write_json(path, {"a": 1})
    assert load_json(path) == {"a": 1}

    unit = os.path.join(str(tmp_path), "unit")
    assert not is_done(unit)
    write_done(unit, {"ok": True})
    assert is_done(unit)


def test_pareto_frontier_basic():
    # b dominates a (higher value, lower cost); c trades value for cost.
    points = [
        {"name": "a", "v": 0.70, "c": 20.0},
        {"name": "b", "v": 0.75, "c": 10.0},
        {"name": "c", "v": 0.60, "c": 5.0},
    ]
    front = pareto_frontier(points, "v", "c")
    assert "a" not in front
    assert "b" in front
    assert "c" in front


def test_pareto_ignores_missing_keys():
    points = [
        {"name": "a", "v": 0.7, "c": 10.0},
        {"name": "b", "v": None, "c": 5.0},
    ]
    assert pareto_frontier(points, "v", "c") == ["a"]


def test_degradation_table_relative_drop():
    points = [
        {"name": "fp16", "mmlu_acc": 0.80, "humaneval_pass1": 0.50},
        {"name": "int4", "mmlu_acc": 0.78, "humaneval_pass1": 0.40},
    ]
    rows = degradation_table(points, "fp16", "mmlu_acc", "humaneval_pass1")
    int4 = next(r for r in rows if r["name"] == "int4")
    assert abs(int4["headline_rel_drop"] - 0.025) < 1e-6
    assert abs(int4["hard_rel_drop"] - 0.20) < 1e-6
    # The multi-step code task degrades more than multiple choice.
    assert int4["hard_rel_drop"] > int4["headline_rel_drop"]


def test_mmlu_prompt_and_parser():
    prompt = format_question("2+2?", ["3", "4", "5", "6"])
    assert "A. 3" in prompt and "D. 6" in prompt
    assert parse_answer("The answer is B.") == "B"
    assert parse_answer("no letter here") == ""


def test_humaneval_truncation():
    completion = "    return x + 1\n\ndef other():\n    return 0\n"
    trimmed = _truncate_completion(completion)
    assert "def other" not in trimmed
    assert "return x + 1" in trimmed


def test_humaneval_executor_pass_and_fail():
    problem = {
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
        ),
    }
    good = build_program(problem, "    return a + b\n")
    assert run_program(good, timeout_s=10.0)["passed"] is True

    bad = build_program(problem, "    return a - b\n")
    assert run_program(bad, timeout_s=10.0)["passed"] is False


def test_humaneval_executor_timeout():
    problem = {
        "prompt": "def loop():\n",
        "entry_point": "loop",
        "test": "def check(candidate):\n    candidate()\n",
    }
    program = build_program(problem, "    while True:\n        pass\n")
    assert run_program(program, timeout_s=2.0)["passed"] is False

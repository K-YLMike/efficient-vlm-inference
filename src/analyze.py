"""Aggregate all results into the Pareto frontier and degradation table.

Two headline artifacts:

* Pareto frontier over accuracy vs latency and accuracy vs memory. A config
  is on the frontier if no other config is at least as good on every axis
  and strictly better on one.
* Degradation analysis contrasting the multiple-choice task (MMLU) with the
  multi-step code task (HumanEval). The comparison shows where low-bit
  quantization first breaks: multi-step generation degrades more than
  single-token multiple choice.

The Pareto and degradation functions are pure and depend only on plain
dicts, so they run and are tested without a GPU. Plotting imports
matplotlib lazily.
"""

import os
from typing import Any, Dict, List

from src.utils.io_atomic import atomic_write_json, load_json
from src.utils.logging_setup import get_logger

LOGGER = get_logger("analyze")


def pareto_frontier(
    points: List[Dict[str, Any]],
    value_key: str,
    cost_key: str,
) -> List[str]:
    """Return names of points on the max-value / min-cost frontier.

    ``value_key`` is maximized (accuracy), ``cost_key`` is minimized
    (latency or memory). Points missing either key are ignored.
    """
    usable = [
        p for p in points
        if p.get(value_key) is not None and p.get(cost_key) is not None
    ]
    frontier: List[str] = []
    for cand in usable:
        dominated = False
        for other in usable:
            if other is cand:
                continue
            no_worse = (
                other[value_key] >= cand[value_key]
                and other[cost_key] <= cand[cost_key]
            )
            strictly_better = (
                other[value_key] > cand[value_key]
                or other[cost_key] < cand[cost_key]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(cand["name"])
    return frontier


def degradation_table(
    points: List[Dict[str, Any]],
    baseline_name: str,
    headline_key: str,
    hard_key: str,
) -> List[Dict[str, Any]]:
    """Return per-config relative drop on the headline and hard tasks.

    Drops are relative to the baseline config's score on each task, so the
    table reads as "how much accuracy this quant level costs".
    """
    base = next(p for p in points if p["name"] == baseline_name)
    rows: List[Dict[str, Any]] = []
    for point in points:
        row: Dict[str, Any] = {"name": point["name"]}
        for label, key in (("headline", headline_key), ("hard", hard_key)):
            score = point.get(key)
            base_score = base.get(key)
            row[label + "_score"] = score
            if score is not None and base_score:
                row[label + "_rel_drop"] = (base_score - score) / base_score
            else:
                row[label + "_rel_drop"] = None
        rows.append(row)
    return rows


def collect_points(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge benchmark, MMLU, and HumanEval results per config."""
    from src.utils.config import quant_configs
    results_dir = cfg["resolved"]["results_dir"]
    points: List[Dict[str, Any]] = []
    for entry in quant_configs(cfg):
        name = entry["name"]
        point: Dict[str, Any] = {"name": name}

        bench_path = os.path.join(
            results_dir, "benchmark", name, "metrics.json")
        if os.path.isfile(bench_path):
            bench = load_json(bench_path)
            point["tokens_per_s"] = bench.get("decode_tokens_per_s")
            point["ttft_ms"] = bench.get("ttft_ms_median")
            point["memory_gb"] = bench.get("checkpoint_gb")

        mmlu_path = os.path.join(results_dir, "mmlu", name, "result.json")
        if os.path.isfile(mmlu_path):
            point["mmlu_acc"] = load_json(mmlu_path).get("accuracy")

        he_path = os.path.join(results_dir, "humaneval", name, "result.json")
        if os.path.isfile(he_path):
            point["humaneval_pass1"] = load_json(he_path).get("pass_at_1")

        points.append(point)
    return points


def _plot_pareto(points, frontier_names, x_key, x_label, out_path) -> None:

    """Scatter MMLU accuracy vs a cost axis and connect the frontier.



    Both memory and latency are costs to minimize, so the preferred corner

    is lower cost and higher accuracy; the axes are annotated accordingly.

    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt



    usable = [

        p for p in points

        if p.get("mmlu_acc") is not None and p.get(x_key) is not None

    ]

    accs = [p["mmlu_acc"] for p in usable]

    xs = [p[x_key] for p in usable]



    fig, axis = plt.subplots(figsize=(6.5, 4.8))



    front = sorted(

        [p for p in usable if p["name"] in frontier_names],

        key=lambda p: p[x_key])

    if len(front) >= 2:

        axis.plot([p[x_key] for p in front], [p["mmlu_acc"] for p in front],

                  linestyle="--", color="#1f77b4", linewidth=1.5,

                  zorder=2, label="Pareto frontier")



    colors = {"fp16": "#7f7f7f", "int8_w": "#2ca02c",

              "int4_gptq": "#d62728", "w8a8": "#1f77b4"}

    for point in usable:

        on_front = point["name"] in frontier_names

        axis.scatter(

            point[x_key], point["mmlu_acc"],

            s=130 if on_front else 90,

            marker="o" if on_front else "X",

            color=colors.get(point["name"], "#333333"),

            edgecolors="black", linewidths=0.8,

            zorder=4)



    offsets = {

        "fp16": (8, -14), "int8_w": (8, 8),

        "int4_gptq": (10, 6), "w8a8": (-12, 10),

    }

    for point in usable:

        dx, dy = offsets.get(point["name"], (8, 6))

        axis.annotate(

            point["name"], (point[x_key], point["mmlu_acc"]),

            textcoords="offset points", xytext=(dx, dy),

            fontsize=9, fontweight="bold")



    lo, hi = min(accs), max(accs)

    pad = max(0.01, (hi - lo) * 0.35)

    axis.set_ylim(lo - pad, hi + pad)

    xlo, xhi = min(xs), max(xs)

    xpad = (xhi - xlo) * 0.12

    axis.set_xlim(xlo - xpad, xhi + xpad * 1.4)



    axis.set_xlabel(x_label + "  (lower is better)", fontsize=10)

    axis.set_ylabel("MMLU accuracy  (higher is better)", fontsize=10)

    axis.set_title("Accuracy vs {}".format(x_label), fontsize=12,

                   fontweight="bold")

    axis.grid(True, alpha=0.3)

    axis.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()

    fig.savefig(out_path, dpi=150)

    plt.close(fig)


def run_analysis(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the summary JSON and Pareto figures; return the summary."""
    points = collect_points(cfg)
    analysis_cfg = cfg.get("analysis", {})
    baseline = analysis_cfg.get("baseline", "fp16")

    latency_points = [
        {"name": p["name"], "acc": p.get("mmlu_acc"),
         "lat": p.get("ttft_ms")}
        for p in points
    ]
    memory_points = [
        {"name": p["name"], "acc": p.get("mmlu_acc"),
         "mem": p.get("memory_gb")}
        for p in points
    ]
    frontier_latency = pareto_frontier(latency_points, "acc", "lat")
    frontier_memory = pareto_frontier(memory_points, "acc", "mem")

    degradation = degradation_table(
        points,
        baseline_name=baseline,
        headline_key="mmlu_acc",
        hard_key="humaneval_pass1",
    )

    summary = {
        "points": points,
        "pareto_frontier_latency": frontier_latency,
        "pareto_frontier_memory": frontier_memory,
        "degradation": degradation,
        "baseline": baseline,
    }

    results_dir = cfg["resolved"]["results_dir"]
    atomic_write_json(
        os.path.join(results_dir, "summary.json"), summary)

    docs_dir = cfg["resolved"]["docs_dir"]
    os.makedirs(docs_dir, exist_ok=True)
    try:
        _plot_pareto(points, frontier_memory, "memory_gb",
                     "weight memory (GB)",
                     os.path.join(docs_dir, "pareto_memory.png"))
        _plot_pareto(points, frontier_latency, "ttft_ms",
                     "TTFT (ms)",
                     os.path.join(docs_dir, "pareto_latency.png"))
    except ImportError:
        LOGGER.warning("matplotlib unavailable; skipped figures")

    atomic_write_json(
        os.path.join(docs_dir, "summary.json"), summary)
    LOGGER.info("analysis written; memory frontier=%s", frontier_memory)
    return summary

<div align="center">

# efficient-vlm-inference

**Post-training compression of a 14B instruction model, measured as an accuracy, latency, and memory Pareto frontier on the vLLM serving stack.**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM-0.6.6-6f42c1?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

This project takes a single open model, Qwen2.5-14B-Instruct, and runs it through four post-training quantization schemes, then measures what each one costs and saves. The output is a Pareto frontier over accuracy, throughput, and memory, plus a per-task degradation analysis. The whole pipeline is single-GPU post-training: no pretraining, no multi-GPU parallel training. That boundary is stated plainly, and the last section says exactly what a production or frontier setup would add on top.

Accuracy is measured on two deliberately different tasks. MMLU is 4-way multiple choice and stresses a single decision token. HumanEval is code generation and stresses multi-step output whose errors compound across tokens. Reporting both, rather than one aggregate score, is what makes the degradation analysis honest: it shows whether low-bit quantization hits one kind of task harder than the other.

## Results

All numbers are measured on Qwen2.5-14B-Instruct. Accuracy is full MMLU (14,042 questions) and HumanEval pass@1 (164 problems). Speed and memory come from vLLM on a single GPU; the four configs were benchmarked on one card so throughput is comparable.

| Config | MMLU | HumanEval pass@1 | tokens/s | TTFT (ms) | weight mem (GB) | bits/weight |
|---|---|---|---|---|---|---|
| FP16 baseline | 75.59% | 69.51% | 2908 | 1671 | 29.5 | 16 |
| INT8 weight-only | 75.60% | 70.12% | 2755 | 1717 | 16.3 | 8 |
| INT4 GPTQ | 73.49% | 68.29% | 2772 | 1692 | 9.9 | ~4 |
| W8A8 | 74.80% | 69.51% | 3763 | 1060 | 16.3 | 8 |

![Accuracy vs memory](docs/pareto_memory.png)
![Accuracy vs latency](docs/pareto_latency.png)

**INT8 is nearly free.** Weight-only INT8 matches FP16 on both tasks (MMLU +0.01, HumanEval +0.6, both within noise) while cutting weight memory by 45% (29.5 to 16.3 GB). It is the obvious default.

**INT4 buys a 3x memory cut for a small, honest accuracy cost.** INT4-GPTQ drops MMLU by 2.1 points and HumanEval by 1.2 points in exchange for shrinking the model from 29.5 GB to 9.9 GB.

**W8A8 is the throughput winner.** Quantizing activations as well as weights gives the highest throughput (3763 tok/s) and lowest first-token latency (1060 ms), with accuracy within about one point of FP16.

**The honest degradation finding.** A common assumption is that low-bit quantization damages multi-step generation (code) more than single-token tasks (multiple choice). The measurements do not support that here: INT4-GPTQ degrades MMLU (2.1 pt) slightly more than HumanEval (1.2 pt), and both drops are small and close to evaluation noise. At this model size and quantization strength the weight-only degradation is roughly uniform across task types rather than concentrated in code. The full per-task relative-drop table, including failing HumanEval task ids, is in `results/summary.json`.

Note on the Pareto frontiers: FP16 is off the memory frontier entirely, since INT8 matches its accuracy at 45% less memory and dominates it.

## Getting Started

Built for Northeastern's Explorer cluster (Slurm), and runnable anywhere with a single GPU that fits a 14B model in FP16 (about 30 GB; an L40S, A6000, A100, or H200 all work) by calling the stage scripts directly.

### 1. Build the environment (once, with internet access)

Do this on a compute node, not the login node: environment builds and package installs are heavy and get killed on login nodes.

```bash
srun -p short --cpus-per-task=8 --mem=32G --time=02:00:00 --pty /bin/bash
cd <project-dir>
bash environment_setup.sh
```

`environment_setup.sh` creates a conda env under `envs/evi` and installs the pinned stack from `requirements.txt`. The versions are pinned deliberately; in particular transformers is held at 4.46.3, because newer versions break llm-compressor's GPTQ calibration on Qwen2. torch 2.5.1 is installed from the CUDA 12.1 wheel index inside the script.

Always call the environment's Python by absolute path (`envs/evi/bin/python`); do not rely on `conda activate`, which is unreliable on compute nodes.

### 2. Download the model and datasets (once, with internet access)

```bash
export PROJECT_BASE=$PWD HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=0
envs/evi/bin/python scripts/run_download.py
```

This caches Qwen2.5-14B-Instruct, MMLU, HumanEval, and the wikitext calibration set. Later GPU jobs set `HF_HUB_OFFLINE=1` and read only from this cache, so compute nodes never need network access.

### 3. Quantize

Produces the INT8, INT4-GPTQ, and W8A8 checkpoints (FP16 needs none). Each config is written atomically and marked with a `_DONE.json` on success, so the stage is idempotent: rerunning skips finished configs and only recomputes the rest. A single 14B GPTQ config can take 30-90 minutes and has no mid-config checkpoint, so give it a partition with enough wall clock (an 8-hour H200 job finishes all configs in one shot):

```bash
sbatch slurm/run_stage_h200.sbatch scripts/run_quantize.py
```

Check progress; three DONE markers (INT8, INT4-GPTQ, W8A8) means the stage is complete:

```bash
find quantized -name _DONE.json | wc -l
```

### 4. Benchmark throughput, latency, and memory

All configs must be benchmarked on the same GPU so speeds are comparable.

```bash
sbatch slurm/run_stage_gpushort.sbatch scripts/run_benchmark.py
find results/benchmark -name _DONE.json | wc -l   # target: 4
```

### 5. Evaluate MMLU and HumanEval

Evaluation is one work unit per (config, task) and is sharded internally (MMLU by subject, HumanEval by problem), so it resumes cleanly after a timeout. Accuracy does not depend on GPU type, so any fitting card works. `run_evaluate.py` does one pending unit per invocation, so drive the eight units (4 configs x 2 tasks) with a Slurm job array that runs one task at a time and auto-advances:

```bash
sbatch --array=1-8%1 slurm/run_stage.sbatch scripts/run_evaluate.py
find results/mmlu results/humaneval -name _DONE.json | wc -l   # target: 8
```

Extra array tasks after the eight units are done become instant no-ops.

### 6. Build the Pareto frontier and figures

```bash
envs/evi/bin/python scripts/run_analyze.py
```

Writes `results/summary.json` (both Pareto frontiers plus the degradation table) and `docs/pareto_memory.png` / `docs/pareto_latency.png`.

## How production would differ

- **Scale.** This is a single 14B model on one GPU. A frontier setup is 70B and up with multi-GPU tensor and pipeline parallelism. That axis is out of scope here and is the direction I most want to work on next.
- **Quantization frontier.** Production also reaches for FP8 on Hopper and Blackwell, activation-aware joint weight-activation schemes, and dynamic range management. This project covers mainstream INT8 and INT4 plus the SmoothQuant path for W8A8.
- **Serving.** Real serving adds paged-KV, continuous batching at load, and speculative decoding. This project uses vLLM's existing batching and metrics rather than building a serving layer.

## Design Notes

- **Weight-only vs weight+activation.** INT4 is weight-only because activations carry outlier channels whose dynamic range is hard to fit in 4 bits. W8A8 quantizes activations too (with SmoothQuant migrating the outliers into the weights first), trading a little accuracy for a faster INT8 kernel, which is why it wins on throughput and TTFT.
- **Why GPTQ and not AWQ.** The plan included an INT4-AWQ config, but autoawq is officially deprecated and its current release imports transformers internals not present in the 4.46.3 pin that GPTQ calibration requires. The AWQ path is left in `src/quantize.py` but disabled in the config; INT4 is produced with GPTQ. This is a dependency-compatibility call, not an algorithmic one.
- **Calibration.** GPTQ and SmoothQuant calibrate on wikitext-2, which does not overlap the MMLU test split or HumanEval, so the accuracy numbers are not contaminated by the calibration data.
- **Measuring on vLLM.** Throughput is batched decode tokens over wall time; TTFT is prefill latency to the first token, read from vLLM per-request metrics, and is a separate number from steady-state throughput.
- **Memory axis.** The Pareto memory axis is the on-disk weight footprint, which is exact and comparable across bit-widths. End-to-end serving also holds a KV cache, which is quantizable separately.

## Future Work

- INT4-AWQ once a maintained toolchain (for example the AWQ support in llm-compressor) is compatible with the pinned stack, to compare against INT4-GPTQ directly.
- KV-cache quantization and its effect on long-context memory.
- Per-layer sensitivity analysis feeding a mixed-precision assignment that keeps the most sensitive layers at higher precision.
- Extending the same pipeline to a small VLM with an OCR-heavy task, where quantization sensitivity is easier to surface.

## License

MIT. See `LICENSE`.

# Phase 1: dLLM Inference Diagnostics

`research/phase1` has been refactored into a package-style layout so that:

- CLI entrypoints live in one place
- benchmark execution and suite parsing live in one place
- task formatting and scoring live in one place
- trace analysis and visualization live in separate modules

## Layout

```text
research/phase1/
  download_phase1_assets.sh
  assets/
  suites/
  diagbench/
    cli/
      prepare_phase1_subsets.py
      collect_traces.py
      collect_mask_rate_oracle.py
      run_minimal_baselines.py
      run_benchmark_suite.py
    core/
      benchmark_spec.py
      benchmark_core.py
      trace_metrics.py
    tasks/
      data_formats.py
      scorers.py
    methods/
      minimal_decoders.py
    analysis/
      analysis_utils.py
      analyze_finalization.py
      analyze_step_drift.py
      analyze_quadrants.py
      analyze_top1_vs_distribution.py
      analyze_router_stability.py
    viz/
      visualize_finalization_gap.py
      visualize_spatial_dynamics.py
      visualize_early_stop_dynamics.py
      visualize_suite_results.py
```

## Why stay in this repository

Start from this DMax repository first.

Reasons:

- it already contains the DMax / LLaDA block-diffusion runtime we need to instrument
- it already contains the MoE router outputs we need for DMax-specific diagnostics
- it already contains the main first-phase model family and evaluation code paths

Dream / thinking models remain second-phase comparisons, but they can still be plugged into the same benchmark suite framework.

## Stable entrypoints

Use module entrypoints instead of direct file paths.

Set:

```bash
export PYTHONPATH="research/phase1:dInfer/python:${PYTHONPATH}"
```

Then use:

- `python -m diagbench.cli.prepare_phase1_subsets`
- `python -m diagbench.cli.collect_traces`
- `python -m diagbench.cli.collect_mask_rate_oracle`
- `python -m diagbench.cli.run_minimal_baselines`
- `python -m diagbench.cli.run_benchmark_suite`

And for offline analysis:

- `python -m diagbench.analysis.analyze_finalization`
- `python -m diagbench.analysis.analyze_step_drift`
- `python -m diagbench.analysis.analyze_quadrants`
- `python -m diagbench.analysis.analyze_top1_vs_distribution`
- `python -m diagbench.analysis.analyze_router_stability`

And for visualizations:

- `python -m diagbench.viz.visualize_finalization_gap`
- `python -m diagbench.viz.visualize_spatial_dynamics`
- `python -m diagbench.viz.visualize_early_stop_dynamics`
- `python -m diagbench.viz.visualize_suite_results`

## Current capabilities

This package currently supports three layers of work:

1. Data preparation
   - deterministic subset building for:
     - `gsm8k`
     - `math500`
     - `aime24`
     - `gpqa_cot`
     - `humaneval`
     - `mbpp`
2. Controlled decoder comparison
   - unified benchmark suites with shared runtime controls
   - shared scoring outputs:
     - `Score`
     - `TPF`
     - `TPS`
     - `NFE`
     - `premature_finalization_rate`
3. Trace-driven analysis
   - per-step trace collection
   - finalization-gap analysis
   - step-drift analysis
   - confidence/consistency quadrant analysis
   - top-1 vs distribution stability analysis
   - optional MoE router stability analysis

## Supported model status

### Primary supported path

These are the main models the runtime is designed around:

- `LLaDA2.0-mini`
- `DMax-Math-16B`
- `DMax-Coder-16B`

### Experimental suite support

These models are already wired into suite skeletons, but should be treated as compatibility targets rather than guaranteed stable paths:

- `Dream-7B`
- `Dream-1-8B-Thinking`

The reason is simple: the current benchmark runtime still uses the DMax / LLaDA execution stack under the hood.

## Supported decoder families

Current method families include:

- single-step static scores
  - `confidence`
  - `margin`
  - `negative_entropy`
- temporal heuristics
  - `stable_top1`
- time + static combinations
  - `margin_plus_stable`
  - `entropy_plus_stable`
- logits modification
  - `credit`
- threshold modification
  - `stdec`
- early stop approximations
  - `jot`
  - `prophet_like`
- combinations
  - `credit_plus_margin`
  - `credit_plus_entropy`
  - `credit_plus_stdec`
- ordering / locality controls
  - `selection_mode = prefix`
  - `selection_mode = arbitrary`
  - `frontier_window = w`

Important:

- `STDec / Jot / Prophet` here are research approximations for controlled comparison.
- They should not be described as official reproductions unless separately validated.

## Assets

The download script still lives at:

- [download_phase1_assets.sh](/Users/ira/Document/DMax/research/phase1/download_phase1_assets.sh)

It prepares:

```text
research/phase1/assets/
  models/
  datasets/
    subsets/
    manifests/
```

Run:

```bash
bash research/phase1/download_phase1_assets.sh
```

Useful variants:

```bash
DOWNLOAD_MODELS=0 bash research/phase1/download_phase1_assets.sh
MODEL_REPOS="inclusionAI/LLaDA2.0-mini" bash research/phase1/download_phase1_assets.sh
MODEL_REPOS="Zigeng/DMax-Math-16B" bash research/phase1/download_phase1_assets.sh
```

## Controlled benchmark suite

The main benchmark path is:

```bash
PYTHONPATH="research/phase1:dInfer/python:${PYTHONPATH}" \
python -m diagbench.cli.run_benchmark_suite \
  --suite research/phase1/suites/phase1_controlled_example.json
```

Dry run:

```bash
PYTHONPATH="research/phase1:dInfer/python:${PYTHONPATH}" \
python -m diagbench.cli.run_benchmark_suite \
  --suite research/phase1/suites/phase1_controlled_example.json \
  --dry-run
```

This suite framework is designed to keep comparisons controlled:

- same model
- same prompt formatting
- same generation length cap
- same block size
- same cache setting
- same backend early-stop setting
- same runtime backend
- same unroll budget

`NFE` is not fixed, because it is one of the comparison targets. What is fixed is the refinement environment.

## Runtime assumptions

The suite expects:

- a local Hugging Face-style checkpoint directory
- `config.json` present in the model root
- tokenizer files present in the same directory, or an explicit `tokenizer_path`

The runner now checks this early and fails with a local-path error instead of falling back to a Hugging Face repo-id parse error.

## Prefix-controlled matrix

For math / code prefix experiments:

- [prefix_control_matrix.json](/Users/ira/Document/DMax/research/phase1/suites/prefix_control_matrix.json)

Example:

```bash
PYTHONPATH="research/phase1:dInfer/python:${PYTHONPATH}" \
python -m diagbench.cli.run_benchmark_suite \
  --suite research/phase1/suites/prefix_control_matrix.json \
  --experiments dmax_math_math500 \
  --methods credit_prefix credit_prefix_w8 credit_arbitrary stdec_prefix stdec_prefix_w8 stdec_arbitrary
```

## Dream / thinking suites

Short / medium reasoning skeleton:

- [dream_thinking_skeleton.json](/Users/ira/Document/DMax/research/phase1/suites/dream_thinking_skeleton.json)

Long-CoT skeleton with AIME / GPQA adapters:

- [dream_thinking_longcot_skeleton.json](/Users/ira/Document/DMax/research/phase1/suites/dream_thinking_longcot_skeleton.json)
- [DREAM_THINKING_NOTES.md](/Users/ira/Document/DMax/research/phase1/suites/DREAM_THINKING_NOTES.md)

Dry run:

```bash
PYTHONPATH="research/phase1:dInfer/python:${PYTHONPATH}" \
python -m diagbench.cli.run_benchmark_suite \
  --suite research/phase1/suites/dream_thinking_longcot_skeleton.json \
  --dry-run
```

## Trace collection

Example:

```bash
PYTHONPATH="research/phase1:dInfer/python:${PYTHONPATH}" \
python -m diagbench.cli.collect_traces \
  --model-path research/phase1/assets/models/LLaDA2.0-mini \
  --task gsm8k \
  --subset-path research/phase1/assets/datasets/subsets/gsm8k_200.jsonl \
  --output-dir research/phase1/runs/llada2mini_gsm8k \
  --gpus 0 \
  --block-length 32 \
  --threshold 0.3 \
  --disable-compile
```

Current task adapters include:

- `gsm8k`
- `math500`
- `aime24`
- `gpqa_cot`
- `humaneval`
- `mbpp`

Current per-step trace fields include:

- block / iteration metadata
- active positions
- masked flags
- committed flags
- changed flags
- previous and current discrete state tokens
- `topk_ids`
- `topk_probs`
- `margin`
- `entropy`
- optional `router_topk`
- decoder-side extras such as:
  - `score`
  - `adjusted_score`
  - `streak`
  - `eligible`
  - `frozen`
  - `within_frontier_window`

Interpretation note:

- `topk_ids / topk_probs` describe the model output distribution at the current refinement step.
- They do **not** represent a separate "soft-label target".
- Under `credit` methods they reflect the credit-modified logits, not raw model logits.

## Mask-rate oracle

Example:

```bash
PYTHONPATH="research/phase1:dInfer/python:${PYTHONPATH}" \
python -m diagbench.cli.collect_mask_rate_oracle \
  --model-path research/phase1/assets/models/DMax-Math-16B \
  --task math500 \
  --subset-path research/phase1/assets/datasets/subsets/math500_200.jsonl \
  --output-path research/phase1/oracles/dmax_math_math500.csv
```

## Minimal baselines

For quick smoke tests only:

```bash
PYTHONPATH="research/phase1:dInfer/python:${PYTHONPATH}" \
python -m diagbench.cli.run_minimal_baselines \
  --model-path research/phase1/assets/models/LLaDA2.0-mini \
  --task gsm8k \
  --subset-path research/phase1/assets/datasets/subsets/gsm8k_200.jsonl \
  --output-dir research/phase1/baselines/llada2mini_gsm8k_standard \
  --decoder standard
```

## Offline analysis

```bash
python -m diagbench.analysis.analyze_finalization \
  --trace-path research/phase1/runs/llada2mini_gsm8k/traces.jsonl \
  --output-dir research/phase1/analysis/llada2mini_gsm8k

python -m diagbench.analysis.analyze_step_drift \
  --trace-path research/phase1/runs/llada2mini_gsm8k/traces.jsonl \
  --output-dir research/phase1/analysis/llada2mini_gsm8k
```

## Visualization

```bash
python -m diagbench.viz.visualize_finalization_gap \
  --trace-path research/phase1/benchmarks/llada2mini_gsm8k/credit/traces.jsonl \
  --output-dir research/phase1/figures/credit_gap

python -m diagbench.viz.visualize_spatial_dynamics \
  --trace-path research/phase1/benchmarks/dmax_math_gsm8k/stdec_prefix_w8/traces.jsonl \
  --output-dir research/phase1/figures/stdec_spatial \
  --sample-index 0

python -m diagbench.viz.visualize_early_stop_dynamics \
  --trace-path research/phase1/benchmarks/llada2mini_gsm8k/jot/traces.jsonl \
  --output-dir research/phase1/figures/jot_dynamics

python -m diagbench.viz.visualize_suite_results \
  --suite-results research/phase1/benchmarks/suite_results.csv \
  --output-dir research/phase1/figures/suite
```

## Common failure modes

### 1. Model path looks valid in the suite, but `transformers` says it is an invalid repo id

This almost always means the local checkpoint directory does not really exist or is incomplete.

Check:

```bash
ls -lah research/phase1/assets/models
ls -lah research/phase1/assets/models/DMax-Math-16B
```

At minimum, the directory should contain:

- `config.json`
- tokenizer files
- model weight shards

### 2. `save_gemlite_cache` import error from `sglang`

The code now treats this as an optional optimization and falls back to a no-op if your `sglang` version does not provide it.

If you still see this error, you are likely running an older checkout that does not include the compatibility patch.

### 3. NVIDIA driver / CUDA mismatch

If you see warnings like:

```text
The NVIDIA driver on your system is too old
```

then the issue is environment-level, not suite-level.

You need one of:

- a newer NVIDIA driver
- or a PyTorch build compatible with your current driver

### 4. `Dream` suites fail while `DMax` suites run

That usually means the checkpoint is not fully compatible with the current DMax / LLaDA runtime stack.

Treat Dream support as experimental unless you have verified the exact checkpoint/runtime combination.

## Design rule

If you add a new benchmark:

1. add or update a subset in `diagbench/cli/prepare_phase1_subsets.py`
2. add or update a task adapter in `diagbench/tasks/data_formats.py`
3. add or update a scorer in `diagbench/tasks/scorers.py`
4. wire it into a suite JSON under `research/phase1/suites/`

This keeps the system configurable without growing new one-off scripts.

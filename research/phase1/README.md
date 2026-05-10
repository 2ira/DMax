# Phase 1: dLLM Inference Diagnostics

This directory is the starting point for the first phase of the diagnostics study.

## Why start from this repository

Start from this DMax repository first.

Reasons:

- It already contains evaluation entrypoints for the exact model family we want to study first:
  - `LLaDA-2.0-mini`
  - `DMax-Math-16B`
  - `DMax-Coder-16B`
- It already contains the block-diffusion decoding code we need to instrument:
  - `/Users/ira/Document/DMax/dInfer/python/dinfer/decoding/generate_uniform.py`
  - `/Users/ira/Document/DMax/dInfer/python/dinfer/decoding/parallel_strategy.py`
- It already contains the MoE model implementation and router outputs needed for DMax-specific analysis:
  - `/Users/ira/Document/DMax/dInfer/python/dinfer/model/modeling_llada2_moe.py`
- It already contains task configs for the first-phase datasets:
  - GSM8K
  - MATH500
  - HumanEval
  - MBPP

Dream/thinking models should be treated as second-phase external comparisons, not the main first-phase execution environment.

## Files

- `download_phase1_assets.sh`
  - downloads the first-phase model checkpoints
  - prepares frozen subset files for diagnostics
- `prepare_phase1_subsets.py`
  - creates deterministic dataset subsets for the trace study
- `collect_traces.py`
  - runs the DMax/LLaDA decoding path on a frozen subset
  - records per-step trace JSONL with top-k, margin, entropy, commit/change flags
- `collect_mask_rate_oracle.py`
  - measures one-step masked-token accuracy at multiple mask rates
- `analyze_finalization.py`
  - computes token-level finalization / commit gap statistics
- `analyze_step_drift.py`
  - computes per-step approximate JS drift, margin, and entropy curves
- `analyze_quadrants.py`
  - computes confidence vs consistency quadrant statistics
- `analyze_top1_vs_distribution.py`
  - compares top-1 stability against top-k distribution stability
- `analyze_router_stability.py`
  - computes MoE router top-2 stability across refinement steps

## Recommended directory layout

The download script defaults to:

```text
research/phase1/assets/
  models/
    LLaDA2.0-mini/
    DMax-Math-16B/
    DMax-Coder-16B/
  datasets/
    subsets/
      gsm8k_200.jsonl
      math500_200.jsonl
      humaneval_164.jsonl
      mbpp_sanitized_200.jsonl
    manifests/
      phase1_manifest.json
```

## Quick start

From the repo root:

```bash
bash research/phase1/download_phase1_assets.sh
```

This does two things:

1. Downloads the model checkpoints needed for the first diagnostics phase.
2. Creates deterministic evaluation subsets for the planned trace experiments.

## Notes

- The script uses the repository's existing Hugging Face helper:
  - `/Users/ira/Document/DMax/dInfer/evaluations/download_hf_model.py`
- The subset builder uses Hugging Face `datasets`.
- If you want to skip model downloads and only prepare subsets:

```bash
DOWNLOAD_MODELS=0 bash research/phase1/download_phase1_assets.sh
```

- If you want to skip subset generation and only download models:

```bash
PREPARE_SUBSETS=0 bash research/phase1/download_phase1_assets.sh
```

## Trace collection

Example: collect GSM8K traces on a local LLaDA2.0-mini checkpoint.

```bash
PYTHONPATH=dInfer/python python3 research/phase1/collect_traces.py \
  --model-path research/phase1/assets/models/LLaDA2.0-mini \
  --task gsm8k \
  --subset-path research/phase1/assets/datasets/subsets/gsm8k_200.jsonl \
  --output-dir research/phase1/runs/llada2mini_gsm8k \
  --gpus 0 \
  --block-length 32 \
  --threshold 0.3 \
  --disable-compile
```

For DMax-Math-16B or DMax-Coder-16B on tensor parallel GPUs, pass `--gpus 0;1` and optionally `--record-router`.
When `--record-router` is enabled, prefer trace-mode settings such as `--disable-compile`.

## Analysis

Example:

```bash
python3 research/phase1/analyze_finalization.py \
  --trace-path research/phase1/runs/llada2mini_gsm8k/traces.jsonl \
  --output-dir research/phase1/analysis/llada2mini_gsm8k

python3 research/phase1/analyze_step_drift.py \
  --trace-path research/phase1/runs/llada2mini_gsm8k/traces.jsonl \
  --output-dir research/phase1/analysis/llada2mini_gsm8k
```

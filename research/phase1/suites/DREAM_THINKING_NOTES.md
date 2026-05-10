# Dream / Thinking Suite Notes

This suite is intentionally a skeleton.

What is runnable immediately:

- `Dream-7B` on `gsm8k_200`
- `Dream-7B` on `math500_200`
- `Dream-1-8B-Thinking` on `gsm8k_200`
- `Dream-1-8B-Thinking` on `math500_200`

Assumptions:

- You have local checkpoints placed at:
  - `research/phase1/assets/models/Dream-7B`
  - `research/phase1/assets/models/Dream-1-8B-Thinking`
- The checkpoints are compatible with the same runtime path used by the current benchmark suite.

What is not wired in yet:

- AIME-style long-CoT subsets
- GPQA CoT subsets
- Task adapters specialized for long-form reasoning traces
- Length-bucketed reporting (`short / medium / long CoT`)

Recommended next extensions:

1. Add deterministic subsets:
   - `aime24_100.jsonl`
   - `gpqa_cot_100.jsonl`
2. Extend `research/phase1/data_formats.py` with:
   - `aime24`
   - `gpqa_cot`
3. Add a second Dream suite that groups outputs by generated trace length:
   - `<256`
   - `256-768`
   - `>768`

Why keep this separate from the main DMax suite:

- Dream / thinking checkpoints are second-phase comparisons.
- They are useful for dynamics contrast, but should not block the first controlled DMax experiments.

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

- Length-bucketed reporting (`short / medium / long CoT`)

Recommended next extensions:

1. Add a second Dream suite that groups outputs by generated trace length:
   - `<256`
   - `256-768`
   - `>768`
2. Validate the exact HF schemas we use for:
   - `HuggingFaceH4/aime_2024`
   - `llamastack/gpqa_0shot_cot`
3. Stress-test long-CoT prompts on:
   - `aime24_30`
   - `gpqa_cot_100`
   using:
   - [dream_thinking_longcot_skeleton.json](/Users/ira/Document/DMax/research/phase1/suites/dream_thinking_longcot_skeleton.json)

What is wired in now:

- deterministic subsets:
  - `aime24_30.jsonl`
  - `gpqa_cot_100.jsonl`
- task adapters:
  - `aime24`
  - `gpqa_cot`
- a long-CoT comparison suite skeleton:
  - [dream_thinking_longcot_skeleton.json](/Users/ira/Document/DMax/research/phase1/suites/dream_thinking_longcot_skeleton.json)

Why keep this separate from the main DMax suite:

- Dream / thinking checkpoints are second-phase comparisons.
- They are useful for dynamics contrast, but should not block the first controlled DMax experiments.

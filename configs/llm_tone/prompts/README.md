# LLM Tone Prompts

`forecast/` stores templates used to generate future-event descriptions.
`forecast/baseline.yaml` is the original round_00 baseline forecast prompt that was previously embedded in code.
`forecast/baseline_one_shot_v1.yaml` adds one anonymized format-only training example while preserving the original actual-case task structure.

`optimizer/` stores templates used by the prompt-improvement agent that analyzes high-loss training examples and proposes new forecast guidance.
`optimizer/legacy.yaml` is the original optimizer prompt that was previously embedded in code.

The generated per-round forecast prompts are still saved under `outputs/llm_event_tone/prompt_optimization/prompt_versions.json` and `.md`.

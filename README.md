# q27b_on_12gb

Experiments on running a 27B-class dense language model (Qwen3.8-27B) on a single 12 GB RTX 3060: quantisation format, KV-cache precision, context size, speculative decoding, scheduling and quality gates.

This is the research side of [revv](https://github.com/mericanii-technologies/revv), which ships the measured configuration.

Start with:

- `ARCHITECTURE.md`, `STRATEGY.md` — what was tried and why.
- `GATES.md`, `PROJECT_CONTRACT.md` — pre-registered acceptance gates.
- `FINDINGS.md` — measured results and corrections.
- `pub/PAPER.md` — paper draft.
- `harness/` — evaluation harness (HumanEval, GSM8K subsets).
- `results/` — raw measurements.

Model files and large logs are not tracked.

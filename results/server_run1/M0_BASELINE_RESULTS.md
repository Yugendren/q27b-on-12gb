# M0 Baseline Results — RTX 3060 12GB (host: ollama)

Hardware: RTX 3060 12GB (sm_86, driver 535.309.01, CUDA 12.0) ·
AMD Ryzen 5 3600 · 47 GB DDR4 · Ubuntu 24.04
llama.cpp build 10566 (bb4caa754), CUDA backend, built from source
Correctness smoke test: PASSED (3/3 coherent, no stale-CUDA gibberish)

## Measured

| Build | bpw | File | VRAM | Prompt t/s | Gen t/s | GSM8K (50) | HumanEval (20) |
|---|---|---|---|---|---|---|---|
| UD-Q2_K_XL (resident, -ngl 99) | 2.87 | 9.14 GiB | 10.14 GB @8K | 518.9 +-8.0 | **21.75 +-0.03** | **0.96** (48/50) | **0.80** (16/20) |
| UD-Q4_K_S (FFN CPU offload recipe) | ~4.5 | 15.4 GiB | 8.98 GB @8K | running | running | running | running |

Eval: temp 0, chat template froggeric fixed jinja, server /v1/chat/completions.
Q2_K_XL eval wall time 1281.9 s. Live-serving gen speed 21.05 t/s (matches bench).

## Findings so far

1. **The >=15 raw tok/s speed gate is ALREADY MET by an off-the-shelf quant**
   (21.75 measured vs 18.7 roofline prediction -- roofline was 16% pessimistic).
   No adapters, no search, no MTP speculation applied yet.
2. **2.87 bpw is NOT quality-wrecked**: 96% GSM8K / 80% HumanEval. The
   community assumption that sub-Q4 of this model is broken is unverified
   folklore; these are the first published numbers.
3. Model defaults to heavy reasoning (500-600 tokens for grade-school math)
   -- shipped bundle must set reasoning_effort sanely.
4. Implication for the project spine: contribution 1 (independent sub-Q4
   quality eval) may be the headline, not the warm-up. The Q4 reference
   (running) decides whether a quality gap exists to close at all.

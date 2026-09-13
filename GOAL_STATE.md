# Goal State — frozen 2026-08-21

## The Mericanii thesis test (added 2026-08-22)

The methodology (harnesses, verifier, rails) is human-built lab equipment.
The **solutions** are machine-discovered: the engine's per-layer allocation ×
adapter-rank × format configuration, found by search against measured
experience, no human priors on what a good config looks like. The thesis
experiment is the frozen head-to-head: **engine-discovered config vs Unsloth
Dynamic (human expert hand-tuning) vs llama.cpp `--target-bpw` (human-designed
greedy heuristic)** at equal size. A win is a direct, certificated instance of
Bet 3 (machine-created design surpasses best human design on a public
frontier). A loss is reported honestly; the product ships on engineering
merits and the thesis test escalates in paper 2. Contributions 1/2/4 are
product engineering; contribution 3 is the science. Never conflate them in
the paper.

## The one-sentence goal

Produce and openly release a ~2.9-effective-bit build of **Qwen3.8-27B** that
runs **fully resident on 12 GB consumer GPUs** (≤11.5 GB incl. KV + overhead)
at codable speed, with quality within a frozen budget of the fp16 original —
discovered by a machine-search engine (LLM proposes quantization/kernel
candidates → measured verifier judges), not by hand-tuning.

## In scope (and nothing else)

1. One flagship model: Qwen3.8-27B, quantized directly from the bf16 original.
2. Techniques: engine-searched per-layer bit allocation; low-rank correction
   adapters; KV-cache quantization; consumer-Ampere/GDDR7 kernels; vision-tower
   excision (text-only variant); MTP/self-speculation if the model ships a head.
3. Five-metric scorecard on every claim: peak VRAM, decode tok/s, TTFT,
   context length, quality vs fp16 on a small frozen benchmark suite.
4. Frozen gates (reference RTX 3060): ≤11.5 GB **at a named context length**
   (default 32K coding-relevant → ~9.0 GB weight budget ≈ 2.6 bpw; plus an 8K
   "max-quality" variant ≈ 3.0 bpw). Context and residency compete for the
   same gigabytes — a gate without a stated context is unadjudicable.
   ≥15 raw / ≥25 code-effective tok/s; TTFT budget set at M0; quality within
   budget. Never claim >23 raw tok/s on 12 GB (bandwidth ceiling); calibrated
   roofline says ~18.7 raw at 2.9 bpw, effective 24–34 straddling the ≥25 gate
   — budget for 2.7 bpw or a faster-dequant format if quality permits.
   16 GB tier: ~28 raw / ~50–70 effective.
5. Demo 1 vs honest baselines (fp16 won't load / 8-bit won't load / best
   4.3-bit GGUF offloaded 2–4 tok/s / extreme 2-bit quality collapse).
6. Week-4 universality run: frozen engine re-applied unchanged to one
   ~8B-class model — repeatability evidence AND coverage of the 8 GB tier.
7. Release: HF model files (ollama-pullable), llama.cpp PR if a new format
   needs kernels, open engine repo, short paper with failures included.
8. Timeline: ~4 weeks. M0 measurement → M1 engine → M2 kernels+KV → M3 freeze
   and release.

## Explicitly out of scope (paper 2+ or never)

- Kimi K3 / frontier-MoE class (2.8T; physics-impossible on consumer VRAM).
- The 80–120B-MoE-on-5090 demo — reserved as paper 2, frozen only after ship.
- Training/fine-tuning products, new model architectures, runtime products
  competing with Ollama/llama.cpp, multi-model × multi-GPU matrices.
- Any claim of Opus speed parity ("same league as cloud streaming" only) or
  agentic-workload parity.
- 27B-on-8GB claims (physically impossible; 8 GB is served by the 8B run).
- **Dynamic/streaming weight loading in any form** (activation-sparsity
  streaming, PCIe prefetch, NVMe streaming, relufication, layer pruning /
  early exit). Researched 2026-08-23 and rejected on physics — see
  ARCHITECTURE.md "Dynamic weight loading — VERDICT". Do not re-litigate.
- Placement/offload as the *headline* claim — partially scooped by ATSInfer
  (arXiv 2607.10183). Placement stays a product feature and a side result
  (the auto-vs-hand-tuned baseline nobody has run), never the paper's spine.

## Current state

- [DONE] Contract, bottleneck ladder, scorecard, market data, gates — see
  PROJECT_CONTRACT.md.
- [DONE 2026-08-21] Landscape + real-world research; **architecture decided**
  (ARCHITECTURE.md Decision section): dual-rail (GGUF + EXL3), four
  contributions = independent sub-Q4 eval → adapters-to-mainstream-rails
  (core novelty) → verifier-first co-optimization → productized 12 GB bundle.
  Headline revised: "first *verified-quality, mainstream-usable* 27B on 12 GB"
  (EXL3 already fits, unverified and niche — never claim "first to fit").
- [DONE 2026-08-23] Harness built + smoke-tested on the M4 Mac (GSM8K +
  HumanEval quality suite, llama-bench speed, VRAM polling, matrix runner) —
  `harness/`. M0_RUNBOOK.md written (verified commands, 9 flagged unknowns).
  Dynamic-loading research closed: NOT VIABLE, plan validated not changed.
- [NEXT] **M0 execution**: run the baseline matrix on the reference RTX 3060.
  Blocked only on access to the 3060. Added to the matrix (cheap, publishable):
  **hand-tuned `-ot` vs `llama-fit-params` head-to-head** — establishes the
  missing literature baseline in one afternoon.
- [PENDING] Mericanii vault (mericanii_knowledge/CURRENT_STATE.md) still names
  the sorting-network project as active; founder should update it to point
  here when convenient.

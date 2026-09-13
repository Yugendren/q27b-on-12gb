# Project Contract — Qwen3.8-27B on 12 GB Consumer GPUs

**Status:** M0 pending · **Opened:** 2026-08-21 · **Owner:** Mericanii (solo founder + Claude as engineering team)

## Claim under attack

[FACT, 2026-08-21] Qwen3.8-27B (released 2026-08-14, Apache 2.0) is the first
open model with credible Opus-4.6-class coding claims. Best current
quantizations (Unsloth Dynamic GGUF) need **16–17 GB**, fitting 3090/4090-class
cards. **No usable configuration exists for 12 GB cards** (RTX 3060 12GB /
4070 / most consumer hardware).

**Target:** an open, reproducible configuration of Qwen3.8-27B that runs on a
12 GB GPU at usable speed with near-baseline quality — via machine-discovered
quantization (per-layer bit allocation, rotation/outlier schemes, KV-cache
compression) and kernels tuned for consumer Ampere.

## Why this is a Mericanii problem

The discovery engine (LLM proposes quantizer/kernel candidates → cheap search
keeps what measures better → frozen verifier judges) is the reusable asset.
This project is its first public exam. Paper 2 (KV-cache frontier, TurboQuant
class) and paper 3 (third domain) reuse the same engine.

## Frozen success metrics (define exactly in M0, before any search)

1. **Fits:** peak VRAM ≤ 11.5 GB including KV cache at a stated context length.
2. **Usable:** on the reference 3060 — raw decode ≥ 15 tok/s and
   code-effective (with speculation/MTP if available) ≥ 25 tok/s, plus a frozen
   TTFT budget for long prompts. Consumer benchmark: Claude Opus 5 streams
   ~52 tok/s standard / ~130 fast mode (Aug 2026); the product competes on
   cost/privacy, not peak speed, but must clear the "codable" bar above.
3. **Quality:** ≤ a frozen degradation budget vs. the fp16 reference on a fixed
   local eval suite (perplexity + small high-signal benchmark subset), measured
   identically for baselines and candidates.
4. **Honesty:** every baseline (llama.cpp IQ2/IQ3 quants, Unsloth Dynamic,
   CPU-offload configs) measured on the same rig with the same harness; results
   published including failures.

## Milestones

- **M0 — Baseline truth (days 1–4):** eval harness + speed harness (the two
  verifiers); measure what 12 GB looks like *today* with every existing method.
  If an existing config already meets all metrics, the project claim collapses
  — publish the measurement note and re-scope.
- **M1 — Engine v1 (week 2):** search over quantization configurations
  (per-layer/per-tensor bit allocation, outlier handling, rotations) against
  the frozen quality verifier. Beat the best hand-made quant at equal size, or
  match it at smaller size.
- **M2 — Kernels + KV (week 3):** low-bit kernels tuned for consumer Ampere;
  KV-cache quantization for real context lengths within the VRAM budget.
- **M3 — Freeze and release (week 4):** frozen configs, reproducible manifest,
  GitHub release, short paper. No tuning after freeze.

## Scorecard and demo narrative (2026-08-21)

Five frozen metrics, always reported together: **fits (peak VRAM), decode
tok/s, TTFT, context length, quality retention vs fp16.** The first four can
always be bought by sacrificing the fifth; every claim states its quality cost.

**Demo 1 (3060):** before/after where "before" = the *best* existing option at
each precision — fp16 (won't load), 8-bit (won't load), best ~4.3-bit GGUF
(offload, 2–4 tok/s), extreme ~2-bit quants that fit (show the measured quality
collapse). "After" = ours, resident, ≥15/≥25 tok/s, within quality budget.
Never compare against a strawman. Speed claim language: "conversational,
codable, same league as cloud streaming" — never "as fast as Opus."

**Demo 2 (5090, paper 2):** same structure one tier up — an 80–120B-class open
MoE at ~3 bits, resident on the largest consumer card (32 GB), that today
requires multi-GPU or datacenter hardware. Frozen only after paper 1 ships.

## Known risks

- [RISK] Sub-3-bit quality cliff: 27B at ~2.7–3.0 effective bits may degrade
  beyond the budget. Mitigation: the engine's entire job is finding non-uniform
  allocations humans didn't; even matching Unsloth quality at −4 GB is a win.
- [RISK] Unsloth/llama.cpp community ships a 12 GB config first. Mitigation:
  speed; the model is one week old. If scooped on "fits", the machine-discovery
  angle and measured study remain the paper.
- [RISK] Eval compute on a 3060 is slow. Mitigation: small frozen high-signal
  suite; full benchmarks only at freeze.

## Additional bottlenecks (2026-08-21 audit)

- **Vision tower excision:** Qwen3.8-27B is multimodal; the vision encoder
  occupies VRAM even for text-only use. M0 measures its size; a text-only
  build is likely worth 1–2 GB free. Ship both variants.
- **Overhead budget:** CUDA context + activations + fragmentation ≈ 0.5–1 GB;
  display output ≈ 0.3–0.5 GB more. Target stays 11.5 GB; docs recommend
  headless.
- **Time-to-first-token:** prefill is compute-bound and needs its own frozen
  metric at M0 (long-prompt TTFT), separate from decode tok/s.
- **Self-speculation:** check at M0 whether the model ships a multi-token
  prediction head — if yes, near-free 1.5–2× decode; prefer over a separate
  draft model on 12 GB.
- **Plan B (cliff softening):** if ≤11.5 GB fails the quality budget,
  hot/cold weight splitting (PowerInfer-style) is the documented fallback.

## Market facts (2026-08 Steam survey, July data)

- VRAM share: 16 GB 25.9% (#1, the volume tier) · 8 GB 25.3% (unreachable for
  27B at any bits — state this boundary) · 12 GB 12.9% (the floor tier) ·
  24 GB 5.4%. Target ≤11.5 GB serves every tier ≥12 GB (~45% of market).
- #1 single GPU: RTX 4060 Laptop (8 GB). #2: desktop RTX 3060 (12 GB).
- Two-tier promise: 12 GB floor = "runs at all, codable" (~15–20 raw /
  25–40 effective tok/s; raw ceiling ~23 by bandwidth physics — never claim
  50 raw here). 16 GB GDDR7 majority tier = "Opus-standard feel"
  (~28 raw / ~50–70 effective tok/s).
- Local-AI enthusiast standard: used RTX 3090 (24 GB) — speed/context wins only.
- **8 GB tier (25.3%, incl. the #1 GPU RTX 4060 Laptop):** unreachable for 27B
  at any bits; served instead by applying the same frozen engine to an
  ~8B-class model (~3.5 GB at target bits). This is the week-4 universality
  run — it doubles as market coverage for the largest single tier.

## Non-goals

- No architecture changes, no fine-tuning quality claims, no training runs.
- No claims about SWE-bench Pro parity — quality is measured only on the
  frozen local suite.
- The sorting-network project's criteria do not govern this repo (founder
  decision, 2026-08-21); its verification discipline (two verifiers, frozen
  metrics, honest baselines) does.

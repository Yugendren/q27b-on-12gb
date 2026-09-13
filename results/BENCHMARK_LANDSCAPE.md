# Benchmark landscape + product comparison (2026-08-23, Opus research)

## Product comparison

| | Claude Opus 5 | Qwen3.8-27B |
|---|---|---|
| SWE-bench Verified | **97.0%** [M] vals.ai (rank 1/86) | **86.0%** [M] vals.ai (430/500, rank 14/84) |
| SWE-bench Pro | not published | 61.7 [C] vendor-only, Qwen-modified harness |
| LiveCodeBench | 89.03 [M] | 90.3 [C] / **84.0 [M]** (−6.3 vs claim) |
| Terminal-Bench 2.1 | 84.64 [M] | 73.0 [C] / 58.4 [M] / 79.8 [M] @xhigh |
| GPQA-Diamond | rank 7/135 [M] | 89.2 [C] / 88.9 [M] — replicates |
| Output speed | 55.5 tok/s [M]; fast ~139 (derived) | 55.7 tok/s [M] hosted API |
| Price | $5/$25 per MTok; fast $10/$50 | $0.50/$3.00 per MTok |
| Consumer | Pro $17–20/mo · Max from $100/mo | Apache 2.0, free to self-host |

**Critical caveat:** Qwen never states precision for ANY published benchmark.
The FP8 card republishes the BF16 table verbatim with unbacked "nearly
identical" prose. No arXiv tech report exists.

## CORRECTION: quantized agentic scores DO exist

My earlier claim that this is unmeasured territory was WRONG. 11 independent
sources; 6 cover Qwen3.8-27B. Key data:

**Qwen3.8-27B GGUF ladder, aider-polyglot-style** [M, quanteval.ai, 4156 rows]:
Q6_K **63.0** · Q5_K_M **61.0** · Q4_K_M **54.25** · UD-Q3_K_XL **49.0** ·
UD-IQ2_XXS **32.5**  ← steep, monotonic degradation. No BF16 anchor.

**Qwen3.8-27B SWE-bench Verified (100-task, mini-swe-agent)** [M, WonderRico]:
BF16 **81** → NVFP4/INT8 81 · INT8-W8A16 80 · NVFP4/FP8 76 · FP8 74–78.
Re-runs moved 6–7 pts, so ±5 is noise at n=100. No GGUF rows.

**RedHatAI Gemma-4-31B-FP8** [C] — the most important single datapoint:
MMLU-Pro / MATH-500 / AIME / GPQA / LiveCodeBench all recover **99.2–103.8%**
while **BFCLv4-Agentic falls 65.82 → 56.53 = 85.9%**. Damage concentrates in
the agentic channel and is INVISIBLE to standard evals.

**The gap that IS still open:** no same-checkpoint **GGUF-vs-BF16** comparison
for any 24–35B model on **official-protocol** SWE-bench Verified or Aider
Polyglot. Every existing source is missing a leg (quanteval has the ladder but
no BF16 anchor; WonderRico has the anchor but no GGUF; RedHat has
anchor+protocol but one quant point).

## THIS INVALIDATES OUR CURRENT MEASUREMENT APPROACH

Our result "Q2_K_XL 80% = Q4_K_S 80% on HumanEval" is almost certainly an
artifact of HumanEval being **quant-blind**. Evidence:

- quanteval's agentic-coding ladder separates Q4 (54.25) from Q3 (49.0) —
  a gap HumanEval does not show at all.
- `agentbench_os` is flat 81.9–83.3 across Q6→IQ2 (completely quant-blind).
- Unsloth's own data: Gemma-3-27B MMLU moves 2.8 pts Q2→Q4 while KLD moves 2.7x.
- **KLD and perplexity RANK QUANTS WRONG for agentic use** (Spearman +0.24,
  p=0.57; a higher-KLD AWQ build beat a lower-KLD one on tool-arg accuracy by
  +54%). Validating 12 GB work on perplexity = validating on the wrong metric.

**Where the damage actually lives — the tool-error tail, not the mean:**
arXiv 2607.27275 (τ²-bench, 456 episodes/arm): scores statistically FLAT
while **tool-error rate doubles** (19.51 → 38.26). The 10-error budget absorbs
it; **tighten to 2 errors and a 17-point gap appears.** Cliff sits between 8
and 4 bits. arXiv 2608.06564: decision margin ×0.86 @4-bit, ×0.33 @3-bit,
**×0.00 @2-bit** — and "no label-free repair recovers more than one more bit."
ACBench: 4-bit costs tool use 1–3% but real applications 10–15% (~5x
component→trajectory amplification).

Counter-evidence to engage honestly: arXiv 2411.02355 (>500k evals) finds FP8
lossless and W4A16 rivalling 8-bit with RULER long-context recovery ≥98% —
but at 70B/405B. Scope: quantization damage bites at ≤4-bit, hardest ≤3-bit,
small models > large, long context > short, reasoning > instruct.

## Revised measurement plan

1. **Drop GSM8K** (at ceiling, 94–100% across every build — discriminates nothing).
2. **HumanEval is necessary but not sufficient** — keep as a cheap screen only.
3. **Add discriminating instruments** (all cheap, all from the literature):
   - aider-polyglot-style edit tasks (proven to separate this exact ladder)
   - BFCLv4-**Agentic** (not single-turn BFCL, which is quant-blind)
   - IFEval (instruction-following breaks first)
   - **error-budget tightening** (10→2) on logs we already collect
   - **flips + KL vs BF16 reference** rather than raw accuracy deltas
4. Freeze `reasoning_effort` as a constant across all builds and state it.

## SWE-bench local cost (measured from 838 public trajectories)

24B-class local model via mini-swe-agent: **15,389 output tok/instance mean**
(Devstral-Small-24B row; a weak model costs MORE, not less — it fails slowly).
Input:output ratio 75:1 to 147:1.

@20 tok/s decode-bound with prefix caching ON: **50 inst = 10.7 h · 100 inst =
21.4 h · 500 inst = 107 h.** WARNING: **prefill may dominate** — un-cached
prefill is 23–125 min/instance (6–11x decode) for a 27B on 12 GB. Verify cache
reuse before budgeting.

Cheap standard subset: `MariusHobbhahn/swe-bench-verified-mini` (50 instances,
distribution-preserving, 5 GB vs 130 GB Docker); mini-swe-agent `--slice 0:50`.
Do NOT cite SWE-agent's `tokens_received` — undercounts decode 4.1x.

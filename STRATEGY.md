# Strategy — Local Intelligence for Productive Work

Status: founder thesis, 2026-08-27. Supersedes ad-hoc framing in GOAL_STATE.md.
Parent: Mericanii thesis (machine-invention lab). This is the commercial wedge.

## 1. The customer

**Working programmers and serious hobbyists who currently pay for Claude Code
/ Codex / Cursor.** They pay because of two things, and it is important not to
conflate them:

1. **The model is smarter.** Opus 5 scores 97.0 SWE-bench Verified vs 86.0 for
   the best open model at full precision [M]. An 11-point gap before any
   compression.
2. **The harness is better.** Tool loops, context management, file editing,
   error recovery, permissions. The scaffolding is a large fraction of the
   delivered quality and is almost never measured separately.

[INSIGHT] Existing "coder models" fail this customer not because they cannot
write a function — they can — but because they are unreliable across the
multi-step agentic loop where real work happens. This matches our own
measurement: single-turn coding benchmarks are quant-blind while agentic
benchmarks separate builds sharply.

## 2. Market tailwinds (why the window is opening)

- **Open models improve and shrink.** Chinese labs (Qwen, DeepSeek, Kimi,
  GLM) ship frontier-adjacent open weights on a monthly cadence, Apache-2.0.
  The barrier to entry that matters — pretraining cost — is being paid by
  someone else and given away.
- **Closed models get more expensive.** US labs are pricing toward IPO. Opus 5
  is $5/$25 per MTok, $10/$50 in fast mode; consumer plans $17–20/mo (Pro) and
  $100+/mo (Max).
- Net: the value gap between "free local" and "paid cloud" narrows from the
  model side while the price gap widens from the cost side.

## 3. Hardware targets (explicit, non-negotiable)

**In scope:** NVIDIA consumer 30–50 series, **12–16 GB** — RTX 3060 12GB,
4060 Ti 16GB, 5060 Ti 16GB, 4070/Ti, 4080, 5080. And **Apple Silicon** M-series
(M4 / Pro / Max), which is what most working programmers actually carry.

**Out of scope (for now):** RTX 5090 (too expensive to be the target market),
datacenter GPUs, multi-GPU rigs. AMD consumer is a possible later addition.

Consequence: the design point is **10–14 GB of weights**, which caps a 27B
dense model at ~3.2 bits per weight. See ARCHITECTURE.md for the arithmetic.

## 4. Workflow targets

Only two workloads justify frontier-class local intelligence:

1. **Coding** — the primary wedge. Verifiable (tests pass or fail), high
   willingness to pay, and the customer is already paying someone else.
2. **Research** — mathematics, physics, engineering, literature synthesis.
   Secondary; harder to verify, lower willingness to pay individually.

Possible later: CAD, visual/design. Not now.

## 5. Product options, with honest assessment

### Option A — Optimized model builds (current work)
Ship the best possible local build of the best open model.
- **Evidence:** we reached 35.3 tok/s at 85% HumanEval on a 3060 (UD-IQ3_XXS +
  built-in MTP n=3 + q8_0 KV, 11.8 GB) — a config we have not found published.
- **Risk:** four groups already publish competing per-tensor quantizations;
  this is a contested, fast-moving space with weeks of lead time at best.

### Option B — Post-trained specialist ("our own model")
Take an open base, post-train it (RL with verifiable rewards, distillation
from a frontier teacher, error-correction adapters) into a coding/research
specialist that is better *per byte* than the generalist it came from.
- **Rationale:** at a fixed 10–14 GB budget, capacity spent on 100+ languages
  and general world knowledge is capacity not spent on the two workflows.
  Specialisation is a bad trade at the frontier and a good trade at the edge.
- **[DECISION] "Combining/merging models" is NOT this.** Model merging rarely
  exceeds its parents and is not a defensible technique. The defensible
  version is *post-training a single strong base toward a verifiable
  objective.*
- **Cost:** requires rented GPU (the only part of the plan that leaves the
  home-hardware constraint). Order of hundreds of dollars per run.
- **Precedent gap:** specialist distillation from a 27–32B generalist has
  **zero shipped artifacts** — the literature claims 50–60% compression at no
  accuracy loss (TrimLLM) and nobody has ever shipped from it.

### Option C — The local-first harness  [UNDERWEIGHTED, possibly the real product]
The founder's own observation: half the reason people pay is the harness, not
the model. **Every serious agentic harness assumes cloud economics** — huge
context, fast generation, cheap retries. None is designed for the local
constraint set: 16–32K usable context, 35 tok/s, no per-token cost, full
privacy, and a model that is weaker per step but free to run for hours.
- A harness designed *for* those constraints (aggressive context compaction,
  cheap retry loops, local-model-aware prompting, test-driven self-correction)
  could recover a meaningful part of the 11-point model gap.
- **Costs no GPU. Solo-founder shaped. Immediately shippable. Unclaimed.**

### Option D — The pipeline (transferable method)
Whatever wins in A/B/C, packaged so it re-applies to each new open model the
week it drops. This is what makes the work a company rather than a one-off,
and it is directly leveraged by tailwind #1.

## 6. Honest constraints

- Training a base model from scratch: **impossible** (5×10^24 FLOP; ~1,600
  years on a 5080). Never revisit.
- "Significantly faster" at equal size: bounded. Decode is memory-bandwidth
  bound; we measured 59.3% of peak and kernel tuning moved nothing. The only
  lossless speed lever is speculation, already applied (+81%).
- Faster therefore means **smaller**, and smaller means **quality per byte** —
  which is why Option B is the only path to a large further gain.

## 7. Open questions

- Does a specialist post-trained at 10 GB beat a generalist at 14 GB on
  agentic coding? (The core Option B hypothesis. Unmeasured by anyone.)
- How much of the Opus gap is harness rather than model? (Option C's premise.
  Measurable: run the same local model under a naive harness and a good one.)
- Which of A/B/C do customers actually pay for?

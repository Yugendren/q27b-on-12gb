# Architecture — DECIDED 2026-08-21 (see Decision section at bottom)

## Design constraint zero: transferability

The deliverable is a **pipeline, not a model**. Input: any open-weight HF
checkpoint. Output: a compressed artifact + manifest that runs in mainstream
runtimes. A stranger must be able to run `pipeline compress <model>` on their
own hardware against their own quality suite and publish the result. Every
architecture below is evaluated against this constraint first.

## The complete bottleneck ledger (what must be solved, per goal)

**Space (fit ≤11.5 GB on the reference 3060):**
- S1. Weight bytes: 4.3 effective bits (17.6 GB) → ~2.9 bits (~10.5 GB).
- S2. Quantization metadata overhead (scales/codebooks) — the double-quant
  lesson; must be counted in "effective bits".
- S3. KV cache at target context (32k) within ~1 GB budget → ~2.5–3-bit KV.
- S4. Vision tower: measure, excise for text-only variant (est. 1–2 GB free).
- S5. Runtime overhead: CUDA context + activations + fragmentation (~0.5–1 GB
  reserve; headless recommended).

**Quality (stay within frozen budget vs bf16):**
- Q1. Which layers/tensors are fragile at low bits (non-uniform allocation —
  the core search problem).
- Q2. Quantization error recovery: low-rank correction adapters (EoRA-class),
  rank allocation per layer is part of the search space.
- Q3. Calibration data choice (imatrix-style importance) — affects everything.
- Q4. Honest measurement: small frozen suite, three reference points
  (bf16 / best existing quant / ours) on every result.

**Speed (≥15 raw, ≥25 code-effective tok/s on 3060):**
- P1. Dequant+matmul kernel efficiency for the chosen format on Ampere
  (target ≥65–70% of 360 GB/s).
- P2. Speculation: MTP head if the model ships one, else consider tiny draft;
  costs VRAM — must fit inside 11.5 GB envelope.
- P3. TTFT: prefill compute path (batch kernels, flash attention) within
  frozen budget.
- P4. No PCIe fallback in the shipped config (the cliff is the enemy).

## Candidate architectures

### A — "GGUF-native": search within the existing ecosystem
Engine searches over llama.cpp's existing quant types (IQ2/IQ3/Q4 mixes),
per-tensor assignment, and imatrix calibration variants. Adapters optional
(as GGUF LoRA sidecar).
- **Pros:** zero new kernels; output is instantly ollama-pullable; maximum
  transferability (llama.cpp already runs everywhere); fastest to ship.
- **Cons:** quality ceiling capped by existing IQ formats — if IQ3-class
  quality at ~2.9 bits is insufficient, we can't fix it from inside;
  differentiation vs Unsloth Dynamic is only the search, not the format.

### B — "New format": codebook/trellis format + custom kernels
QTIP/EXL3-class quantization (trellis/codebook, near-theoretical distortion)
with our searched allocation on top; write/port Ampere kernels; upstream to
llama.cpp as a new quant type.
- **Pros:** highest quality ceiling at 2.9 bits; a real technical moat.
- **Cons:** kernel engineering dominates the month; upstreaming is slow
  (PR review latency is weeks); transferability suffers until merged; we
  compete head-on with exllamav3 which may already own this ground.

### C — "Hybrid": best existing formats + searched allocation + adapters
Base = strongest *already-supported* low-bit formats (llama.cpp IQ-family
and/or EXL3 where it runs); engine's novelty = (1) global search over
per-layer format/bit assignment against the measured quality verifier,
(2) trained low-rank correction adapters co-optimized with the allocation,
(3) KV quant config, (4) speculation config. Ship GGUF where possible,
EXL3 build as enthusiast variant.
- **Pros:** highest probability of hitting gates in 4 weeks; novelty lives in
  the search+adapters (our actual thesis) not in format engineering;
  transferable — the pipeline emits configs for formats every runtime knows.
- **Cons:** ceiling between A and B; two output formats = more release work.

## Real-world baseline findings (community research, 2026-08-21)

Measured user data (sources in research log) that revises our assumptions:

1. **KV is cheaper than we budgeted.** The model uses hybrid attention (48/64
   layers linear DeltaNet + 16 full attention) → **~64 KB/token measured**:
   32K ctx ≈ 2 GB, 96K ≈ 6 GB. KV compression matters less; weight bits and
   placement matter more.
2. **MTP confirmed.** The checkpoint ships an MTP draft head; llama.cpp merged
   support (July 2026). Measured: 1.3–1.8× on NVIDIA GPUs, ~88% acceptance in
   the one known 3060 recipe; n=3 costs ~1 GB VRAM (must fit our envelope).
   Regression on Ollama/Metal — MTP config must be per-backend.
3. **The 3060 is not virgin territory — best known result is ~9.7 tok/s** via
   Q4_K_S + hand-picked FFN-tensor CPU offload + MTP at 96K ctx (community
   recipe, replicated at 6.6–8.5). Naive offload remains 2–4.5. **Our gates
   (≥15 raw / ≥25 effective) mean beating the best community recipe ~2×, not
   beating naive 4×.** Update Demo 1 baselines to include this recipe.
4. **The quality cliff is measured and real exactly at our target size.**
   Community KLD study: Q4 ≈ lossless (KLD 0.0137), Q3 borderline, and every
   quant below ~10 GB "degrades fast"; the model is described as "very easy to
   break with quants." IQ3_XXS (10.9 GB) technically nearly fits 12 GB but is
   borderline-degraded and IQ dequant is slow. **This is the wedge, precisely:
   nobody has Q4-class quality at ≤11.5 GB. It is also the project's main
   risk** — correction adapters must carry real weight; brief
   quantization-aware tuning (EfficientQAT-style, possibly rented-GPU) is the
   contingency if pure post-training search misses the quality gate.
5. **Vision overhead confirmed worse than expected:** tools auto-loading the
   mmproj pushed a user into offload — removing it took them 7–10 → 57 tok/s.
   Text-only variant is mandatory, not optional.
6. **TTFT at long context is brutal everywhere** (78 s @128K even on a 5090;
   3060 recipe prefills at ~225 tok/s → minutes at long ctx). Frozen TTFT
   budget must be stated at a realistic ctx (e.g. 8–16K coding prompts), and
   long-ctx limits documented honestly.
7. **The tooling is raw** (chat-template 500s, stale-CUDA gibberish, broken
   week-1 quants, xhigh reasoning burning 20–60K think-tokens by default).
   Shipping a *polished, verified* 12 GB experience — correct template, MTP
   flags, reasoning_effort preset — is itself part of the product value.

## Decision rule (to apply when landscape research lands)

- If existing sub-Q4 quants of Qwen3.8-27B already fit 12 GB with acceptable
  measured quality → the "fits" claim is dead; wedge shifts entirely to
  quality-at-bits (search+adapters vs their hand allocation) and speed.
- If EXL3 at 3.0 bpw already runs this model well on a 3060 → B is dead;
  C absorbs EXL3 as a base format; our contribution = search + adapters +
  GGUF-mainstreaming.
- If nobody is doing automated allocation search → C's novelty claim is clear.
- Default expectation: **C**, pending evidence.

## Universal formulation (2026-08-23, founder reframe — supersedes "fit in VRAM")

The real problem is **hierarchy-aware placement**: given (model, hardware
profile = VRAM + RAM + SSD + CPU), jointly optimize per-tensor (precision,
location) + adapter ranks + speculation config to maximize quality × speed.
"Fit in 11.5 GB" is one instance; the hand-tuned 46-FFN-tensor offload recipe
(~9.7 tok/s) is proof a human can find one good point slowly — the pipeline
finds the point for ANY machine automatically. Incumbent weaknesses this
exploits: EXL3 = all-or-nothing VRAM fit, CUDA-niche, quality unmeasured;
Unsloth/GGUF = frozen one-size-fits-all files ignoring the user's machine;
recipes = non-transferable single points. **Nobody ships the optimizer;
everybody ships artifacts.** We ship both: pre-baked configs for common
profiles + the optimizer for everything else.

Paper-1 scope guard: same model, CUDA rail, three reference profiles
(3060 12GB+32GB RAM floor · 16 GB tier · 5080 16GB comfortable — founder owns
two). Mac/MLX rail = paper 2 (Metal kernels are a separate world). Search
space additions: placement per tensor (VRAM/RAM), speculation config per
profile. Hardware profile is an INPUT to the pipeline, never a hardcode.

## Efficiency amendment (2026-08-22, founder solution-agnostic review)

- **Cost reality:** LLM tokens are negligible (~$10s); candidate evaluation is
  the budget (quantize ≈ up to 1 h + benchmark). Naive evaluate-everything is
  infeasible on our hardware.
- **Staged fidelity (mandatory):** (i) per-tensor sensitivity table measured
  once → search runs on the table at ~ms/candidate; (ii) promote few to
  small-sample KLD (minutes); (iii) finalists only to full task suite (hours).
- **Dumbest-that-works laddering:** greedy + knapsack/ILP on the table are the
  workhorse AND the population seed. LLM-guided evolution operates only in the
  interaction space the table can't model (cross-layer error compounding,
  adapter-rank × allocation trades, format mixing, calibration choice).
- **Honesty rule:** if classical solvers alone match the LLM-guided search,
  report that; ablations of our own machinery are part of the frozen
  comparison. Solution-agnostic means our preferred method is allowed to lose.

## Pipeline (step-by-step, architecture C shape)

1. **Ingest:** HF checkpoint → canonical tensor map (any model, any size).
2. **Profile:** per-layer sensitivity scan (cheap perturbation probes) +
   calibration set → fragility priors for the search.
3. **Search:** engine loop — LLM proposes allocation/config candidates from
   priors + past results; each candidate is materialized and scored by the
   frozen verifier (quality suite + VRAM + tok/s on reference GPU); population
   keeps Pareto frontier (size × quality × speed).
4. **Repair:** train low-rank correction adapters on the winning allocation;
   optionally re-enter search with adapters in place (co-optimization).
5. **Package:** emit runtime-native artifacts (GGUF / EXL3) + manifest with
   hashes, measured scorecard, reproduction command.
6. **Verify:** independent replay — second machine, scripted, must reproduce
   the scorecard before release. (Mericanii trust boundary.)

Steps 1–6 contain nothing Qwen-specific: that is the transferability claim,
proven in week 4 by re-running unchanged on an 8B model.

## DECISION (2026-08-21, after landscape research)

### What the landscape research established (sources in research log)

1. **"Fits on a 3060" is already technically solved — in a niche.** turboderp
   publishes EXL3 quants of Qwen3.8-27B at 2.0–3.5 bpw; 27–30B @3.0 bpw
   (~10.5 GB) is demonstrated working on a 3060 12GB (est. ~20 tok/s, never
   measured there). EXL3 carries a ~0.5 bpw quality advantage over GGUF
   i-quants. BUT: CUDA-only, no ollama/LM Studio, no dense-model CPU offload,
   ~1.8k HF repos vs >15k GGUF — quality inverted against adoption.
2. **Mainline llama.cpp is frozen at 2024-era i-quants** (Feb 2026 rejection
   of the ik_llama.cpp port; trellis quants live only in EXL3 + the ik fork).
3. **Automated per-tensor bit allocation is commoditized**: llama.cpp
   `--target-bpw` (merged Jan 2026), EXL3 optimize.py, AutoRound AutoScheme.
   "We search the allocation" is NOT novel by itself.
4. **Four documented open gaps nobody holds:**
   (a) zero independent sub-Q4 quality evaluations of Qwen3.8-27B exist;
   (b) EoRA-class error-correction adapters (measured +11 GSM8K pts at 3-bit)
       have NO pathway into GGUF or EXL3 — they live only in GPTQ/vLLM;
   (c) nobody has benchmarked Unsloth "Dynamic" vs mainline `--target-bpw`
       auto-search on the same model;
   (d) same-PPL quants measurably diverge on task benchmarks (GSM8K/IFEval) —
       the field's proxy metrics are unreliable, and nobody ships
       task-verified low-bit quants.

### Decision rules applied

- Architecture **B is dead** (rule fired: EXL3 already owns the
  trellis-format ground; a third format is a wasted month).
- Architecture **A alone is insufficient** (search is commoditized; GGUF
  i-quant ceiling is real).
- **C is selected, revised**: both rails as base formats — GGUF (mainstream
  rail: universality, ollama, offload fallback) and EXL3 (quality rail:
  best bits-per-quality on CUDA). Novelty relocated from "search exists" to
  the four gaps above.

### The four contributions (= the paper, in order of execution)

1. **The independent sub-Q4 eval** (fills gaps a + c). M0's quality harness,
   run across unsloth UD ladder, bartowski, EXL3 2.0–4.0, and `--target-bpw`
   outputs, on task benchmarks not just PPL/KLD. Publishable standalone;
   de-risks everything downstream.
2. **Adapters onto mainstream rails** (fills gap b — the core novelty).
   Train EoRA-class quantization-error-correcting low-rank adapters and make
   them consumable in llama.cpp (GGUF LoRA sidecar / merged) and EXL3.
   This is the "layer we ship" that buys back quality below 3.5 bpw.
3. **Verifier-first co-optimization** (exploits gap d). The engine searches
   allocation × adapter-rank jointly, scored on the task-benchmark verifier —
   compared head-to-head against Unsloth Dynamic and `--target-bpw`.
4. **The productized 12 GB bundle**: verified-quality builds on both rails +
   correct chat template, MTP flags, reasoning_effort preset, context sizing —
   fixing the documented week-1 tooling chaos. Ollama-pullable GGUF variant.

### Revised claim language

- Headline is NOT "first to fit 27B on 12 GB" (EXL3 got there). It is:
  **"first verified-quality, mainstream-usable Qwen3.8-27B on 12 GB — and the
  open pipeline that does this to any model."**
- Speed gates unchanged (≥15 raw / ≥25 effective on 3060) — EXL3 rail est.
  ~20 raw makes them plausibly achievable; M0 measures.

### M0 baseline matrix (expanded)

On the reference 3060, measure: naive UD-Q4_K_XL offload · community
FFN-offload recipe (~9.7 claimed) · UD-IQ3_XXS resident · UD-Q2_K_XL ·
bartowski IQ2_S/IQ3_XXS · EXL3 2.5/3.0/3.5 bpw · `--target-bpw` @~2.9 ·
each × {quality suite, VRAM, decode, TTFT, MTP on/off}. bf16 quality
reference computed on rented GPU once.

## M0 runbook corrections (2026-08-23, Opus verification pass)

Primary-source verification overturned earlier research claims:

1. **EXL3 3.00bpw of Qwen3.8-27B is 12.87 GiB — it does NOT fit a 3060.**
   Existing EXL3 options for 12 GB are ≤2.20 bpw (quality-collapse zone).
   → Strategic upgrade: nobody — including the niche — fits this model on
   12 GB above ~2.2 bpw today. Our ≤11.5 GB @ ~2.9 effective bits target is
   genuinely unclaimed even by EXL3. "First to fit at usable quality" is
   partially back on the table (still claim it only after quality verifies).
2. **`--target-bpw` is NOT merged** (PR #15550 open, draft) — the landscape
   agent's "merged Jan 2026" was wrong or refers to something else; treat as
   CONFLICT to resolve at execution. Comparison baseline options: build the
   PR branch, or use AutoRound/Thireus as the automated-search baseline.
   The "search is commoditized" threat is weaker than assessed.
3. exllamav3 DOES support the MTP head (`draft_mode: mtp`) — quality rail
   gets speculation after all.
4. Flag corrections: MTP is `--spec-draft-n-max` (not `--draft-n`);
   `--jinja` is now default; community recipe's Q4_K_S filename is actually
   UD-Q4_K_S; build "~10450 fixes DeltaNet" unverified — pin exact PR at M0.
5. Nine items marked ⚠️ unverified in M0_RUNBOOK.md §10 — resolve on the
   GPU machines, not by further web research.

## Dynamic weight loading — VERDICT: NOT VIABLE (2026-08-23, Opus research)

Founder's "sub-model streams only needed weights" idea researched to primary
sources. **Rejected on physics.** Recorded so it is never re-litigated.

**Architecture facts (verified from config.json, not secondary reporting):**
`hidden_act: silu` (SwiGLU — zero natural sparsity, invalidates the entire
PowerInfer/LLM-in-a-flash family) · 64 layers, 16 full + 48 linear attention ·
FFN = 17.1B params = **66% of model** (caps any sparsity win at 1.95× at a
realistic 70%) · untied LM head = 1.27B = 0.72 GB read every token ·
`mtp_num_hidden_layers: 1` (MTP head CONFIRMED) · vision tower confirmed.

**The roofline, calibrated and validated:** anchored on a measured 3090 run of
this model (488 GB/s achieved = 52% of peak) → 3060 effective **188 GB/s**;
DDR4-3200 dual-channel ≈ **30 GB/s**; PCIe 4.0 x16 pinned ≈ **21–25 GB/s**;
NVMe ≈ 6 GB/s. Model predicts the community recipe at 3.55 raw / 9.81
effective vs measured 3.51 / 9.7 — **agreement within 1%**.

1. **The killer inequality: PCIe (21–25) < DDR4 (30).** Streaming a weight to
   the GPU costs MORE than computing it in place on the CPU. Prefetch/overlap
   cannot fix a bandwidth deficit (measured: llama.cpp's unmerged prefetch PR
   = decode −4.8%; DeepSpeed's engineered prefetch = 1.13–1.21× only).
2. **Economics: moving 1 GB VRAM→RAM costs +36 ms/token; halving that same
   tensor's bits and keeping it resident saves 2 ms/token. Offloading is ~18×
   more expensive than aggressive quantization of the same bytes.**
   *On this hardware, size dominates placement.*
3. **Sparsity requirement is unreachable:** 15 raw tok/s with the recipe's
   7.37 GB off-GPU needs 77–89% sparsity under perfect prediction; SOTA on
   SwiGLU (Prox, Jul 2026) is 70% **GPU-resident** at −7.1 points on Qwen3-8B.
   Better-trained models are LESS sparsifiable; this model is trained far past
   the ones those papers use.
4. **Sparsity and MTP are SUBSTITUTES, not complements.** MTP already amortizes
   each weight read over ~2.76 accepted tokens; sparsity then requires the
   UNION of neurons across drafted positions. At 50% sparsity / 50% overlap the
   union is 1.0 — **zero gain**. Assume no stacking without measurement.
5. **Layer pruning / early exit: Qwen prunes worst of any family** (MMLU
   collapse ~20% vs Llama-2's 45–55%), and MMLU hides generative collapse
   (LayerSkip: MMLU 49.2 while GSM8K 0.08, HumanEval 0). Dead.
6. **MoE offload is real but architectural, not transferable:** same 3060 runs
   a 35B-A3B MoE at ~51–53 tok/s vs ~10 for this dense 27B. Kimi-class
   SSD/RAM streaming = a future MoE paper, never this dense model.

**Ranked ceilings on 3060+32GB (gates: ≥15 raw / ≥25 effective):**
| Path | Ceiling | Verdict |
|---|---|---|
| **Quantize to residency (~2.9 bpw) — existing plan** | **~18.7 raw / 24–34 eff** | **only path clearing the gates** |
| Static offload + MTP (community recipe) | 9.7 → 10.9 max at k=3 | CPU-FLOPs bound; fails gates |
| Learned/searched placement | 12–14 eff | real, bounded, cannot reach 15 raw |
| SubSpec-style quantized-substitute speculation | 4–8 | substitute copy doesn't fit 11 GB |
| Activation-sparsity streaming | <9 | needs 77–89%, SOTA 70% resident |
| PCIe streaming + prefetch | <4 | PCIe < DDR4 |
| NVMe streaming | 0.4–1.5 | AirLLM measures 0.075 on a 32B |

## Two corrections this forces into the plan

**A. NOVELTY: hierarchy-aware placement is PARTIALLY SCOOPED.**
ATSInfer (arXiv 2607.10183, Jul 2026) does per-tensor placement on llama.cpp
via measured performance-density + knapsack DP, **evaluated on RTX 3060 +
32GB DDR4 with dense Qwen3-14B INT4** (1.94× prefill / 3.29× decode). Read it
before writing any placement claim. **What survives as genuinely open:**
(i) black-box search against *measured tok/s* over discrete per-tensor
placement (analytical/DP solvers are crowded; measured-throughput search is
empty); (ii) **the missing baseline nobody has run: automatic placement vs a
competent human's hand-tuned `-ot` regex** — cheap, high-credibility, add to
M0; (iii) joint bits×tier under a *bandwidth* objective rather than a size
budget; (iv) transfer of a found placement across hardware profiles.
Placement is a real product feature; it is NOT the path to the speed gate and
must not be the headline claim.

**B. THE GATE IS AMBIGUOUS — context and residency compete for the same GB.**
At ~64 KB/token: 8K ctx leaves ~10.5 GB for weights (3.03 bpw, ~17.8 raw);
32K leaves ~9.0 GB (2.60 bpw, ~20.8 raw); 96K leaves ~5.0 GB (1.45 bpw —
impossible). The community recipe bought its 96K context *by paying* the
offload that caps it at 9.7. **Action: state the residency target AT a named
context length** (proposal: 32K as the coding-relevant default, with an 8K
"max-quality" variant), or the gate cannot be adjudicated.
Also: effective speed at 2.9 bpw lands 24–34 vs the ≥25 gate — IQ-style quants
dequantize slower and push this down. Budget for 2.7 bpw or a faster format.

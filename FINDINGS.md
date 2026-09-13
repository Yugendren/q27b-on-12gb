# M0 Findings — Consolidated (2026-08-28)

All measurements: RTX 3060 12GB (sm_86, driver 535.309.01), Ryzen 5 3600,
47 GB DDR4, Ubuntu 24.04, llama.cpp build 10566 (bb4caa754) CUDA, headless.
Correctness smoke test PASSED before any measurement (3/3 coherent).

## 1. Quality vs size curve (⚠️ SEE §6 — THESE ARE NOISY)

| Build | bpw | GiB | decode t/s | GSM8K(50) | HumanEval(20) |
|---|---|---|---|---|---|
| UD-IQ2_XXS | 2.12 | 6.76 | 23.33 | 0.94 | 0.35 |
| UD-IQ2_S | 2.45 | 7.79 | 22.48 | 0.96 | **0.70 then 0.80** |
| UD-Q2_K_XL | 2.87 | 9.14 | 21.75 | 0.96 | 0.80 |
| UD-IQ3_XXS | 3.20 | 10.17 | 20.69 | 0.98 | 0.85 |
| UD-Q4_K_S (CPU offload) | ~4.5 | 15.4 | 2.12 | 1.00 | 0.80 |
| Minitron-20B (i1-IQ4_XS) | — | 10.23 | 28.44 | 0.90 | 0.50 |

## 2. Speculative decoding — the dominant speed lever

16K ctx, q8_0 KV. Built-in MTP head (blk.64.nextn), NOT the external MTP file.

| Build | config | VRAM MiB | decode t/s | accept |
|---|---|---|---|---|
| Q2_K_XL | none | 10212 | 21.09 | — |
| Q2_K_XL | mtp n=2 | 10588 | 35.11 | 0.786 |
| Q2_K_XL | mtp n=3 | 10740 | 35.39 | 0.707 |
| Q2_K_XL | mtp n=4 | 10894 | 33.63 | 0.581 |
| **Q2_K_XL** | **ngram-mod** | **9762** | **86.12** | 0.754 |
| IQ3_XXS | none | 11266 | 20.25 | — |
| IQ3_XXS | mtp n=2 | 11642 | 33.83 | 0.759 |
| IQ3_XXS | mtp n=3 | 11794 | 33.28 | 0.651 |
| **IQ3_XXS** | **ngram-mod** | **10816** | **134.40** | 0.958 |

- MTP: +68%, lossless, optimum at n=2–3; n=4 loses acceptance and is net slower.
- **ngram-mod: 4–6.6×, uses LESS VRAM than MTP.** ⚠️ measured on a repeated
  prompt (warm cache). Warm-cache IS the code-editing case but this must be
  re-measured on novel generation before any public claim.
- DFlash2 drafter: FAILS to load on build 10566 ("expected 81, got 58") for
  all three quant variants. `--spec-type draft-dflash` exists but loader is
  incomplete. Untested — needs a newer llama.cpp.

## 3. Context scaling — NO decode cliff (negative result)

Q2_K_XL, q4_0 KV, fully resident:

| ctx | 8192 | 16384 | 32768 | 49152 | 65536 | 81920 | 98304 |
|---|---|---|---|---|---|---|---|
| VRAM MiB | 9782 | 9956 | 10324 | 10692 | 11060 | 11428 | 11796 |
| decode t/s | 21.05 | 20.98 | 21.00 | 21.00 | 20.99 | 21.00 | 21.02 |

**Flat to 98K.** The decode cliff reported in llama.cpp #27623 (35.6→1.4 t/s
past ~90K) does NOT reproduce here. KV grows ~368 MiB per 16K at q4_0
(≈23 KiB/token), matching the hybrid-attention architecture.

## 4. KV precision — memory only, zero speed effect

IQ3_XXS, 16K ctx: f16 11718 MiB / 20.28 t/s · q8_0 11266 / 20.21 ·
q4_0 11010 / 20.18. **Speed identical within noise.** Quantize KV purely to
buy weight budget or context.

## 5. Structural pruning loses to quantization (negative result)

`exnivo/Qwen3.8-20B-Minitron` (64→44 layers, KD-healed), i1-IQ4_XS 10.23 GiB:
faster (28.44 decode / 38.61 prefill vs 21.09 / 25.63) but **HumanEval 0.50 vs
0.80 and GSM8K 0.90 vs 0.96** against UD-Q2_K_XL at a similar size.
Someone else paid the pruning compute; the result is worse than quantizing.

## 6. ⚠️ MEASUREMENT NOISE — invalidates fine distinctions

UD-IQ2_S, identical build/flags/harness, measured twice:
**HumanEval 0.70, then 0.80.** At n=20 one task = 5 points, so the observed
±10-point swing is pure sampling noise.

**Consequence: every ≤10-point HumanEval difference reported so far is not
significant.** Specifically IQ3_XXS (0.85) vs Q2_K_XL (0.80) is NOT a
demonstrated difference. GSM8K is separately unusable — 0.94–1.00 across every
build is at ceiling and discriminates nothing.

**Action: re-run on the full 164-task HumanEval; drop GSM8K for a harder set.**

## 7. Effective bandwidth (calibration)

Q2_K_XL 9.82 GB at 21.748 t/s → **213.5 GB/s = 59.3% of the 3060's 360 GB/s
peak.** IQ-format quants achieve only ~169 GB/s (−21%) — they dequantize more
slowly, so a smaller IQ build is not proportionally faster. Kernel tuning
(flash-attn, batch/ubatch sweeps) moved decode <1%: this is the achievable
ceiling, not recoverable overhead.

## 8. Corrections filed against earlier project assumptions

- MTP costs ~530 MiB (built-in head), NOT 1.3 GB. The external MTP file is
  +776 MiB for identical performance — never download it.
- Vision-tower excision is a NON-WIN: mmproj is a separate file llama.cpp
  never loads unless asked. Struck from scope.
- Weight streaming / RAM offload: rejected on physics (PCIe 21–25 GB/s <
  DDR4 30 GB/s). Offloading 1 GB costs +36 ms/token; halving that tensor's
  bits costs 2 ms/token — ~18× worse.
- Speed gate ≥15 raw t/s is met by off-the-shelf quants; the binding
  constraint is quality-per-gigabyte, not speed.

## FINAL: Full HumanEval curve (n=164, 2026-08-27) — supersedes §1 and §6

| Build | GiB | HumanEval (164) | MTP head? |
|---|---|---|---|
| UD-IQ2_XXS | 6.76 | **49.4%** (81/164) | ❌ stripped |
| UD-IQ2_S | 7.79 | **62.2%** (102/164) | ❌ stripped |
| UD-Q2_K_XL | 9.14 | **72.0%** (118/164) | ✅ |
| UD-IQ3_XXS | 10.17 | **82.3%** (135/164) | ✅ |

**THE EXCHANGE RATE: ~9.7 points of coding ability per GiB, near-linear
across 6.8–10.2 GiB.** (Steps: +12.8, +9.8, +10.3 per ~1.1 GiB.)

- The n=20 samples both understated the gaps and mis-ranked nothing; the
  "staircase" hypothesis from IQ2_S's half-way tracking (~72%) died — it
  finished at 62.2%. Full-set only, from here on.
- Rule confirmed with 4 points: **on a 12 GB card, run the largest build
  that fits your context need.** Weight bits >> context, at ~10 pts/GiB.
- Builds < ~8.4 GiB also lose the MTP head (no +68% speculation): going
  small is doubly penalized.
- Reference: same model at BF16 claims 86% SWE-bench Verified [vendor];
  our 10.17 GiB build does 82.3% HumanEval at 33 t/s on a $250 GPU.

## THE ANCHOR (2026-09-01, rented L40S, ~$6): the original's ceiling measured
Q8_0 (29GB, ~=BF16: ppl 6.9554 vs 6.9540) full HumanEval-164: **153/164 = 93.3%**.
Retention table (same protocol, same seed discipline):
| Build | Size | HumanEval | % of original retained |
|---|---|---|---|
| Q8 anchor | 29 GB | 93.3% | 100% (def.) |
| IQ3_XXS flagship | 10.17 GiB | 82.3% | **88.2%** |
| Q2_K_XL | 9.14 GiB | 72.0% | 77.2% |
| IQ2_S | 7.79 GiB | 62.2% | 66.7% |
| IQ2_XXS | 6.76 GiB | 49.4% | 52.9% |
HEADLINE: a $250 card at 1/5.5 the bytes keeps 88% of the original's measured
coding ability (and with certified-lossless +68% speculation on top).
The 11-point gap (93.3->82.3) is the measured target for healing/EoRA.
Reference responses saved (results/anchor/) = canary-v2 ground truth + EoRA prereq.
Adjudicator (same day): nospec 136/164=82.9% vs spec 135/164=82.3% -> MTP
speculation CERTIFIED quality-neutral. Embedding offload: NULL (llama.cpp
already hosts token_embd off-VRAM; folklore tip debunked). DFlash2 squeeze:
plain Q2 draft still OOMs at 12GB; pruned+pruned test pending.
Vocab prune of flagship: -555MB (5.08%), verification rerun pending.

## DFlash2-on-12GB: the full adaptation arc (2026-09-01) — BLOCKED UPSTREAM
Chain: plain draft OOM (12.6GB) -> Q2 draft OOM -> vocab-pruned target (-555MB)
+ vocab-pruned drafter (-27MB, both selector tensors, byte-exact) + q4 KV
(-256MB) + small ubatch (-230MB compute) -> **LOADS at 12,002/12,044 MiB**
with draft-dflash active... then SEGFAULTS in graph compute during decode.
Same crash family as the -devd none (CPU-draft) attempt with the UNPRUNED
drafter -> upstream llama.cpp DFlash2 instability (b10712), not our artifacts.
Two clean repro stack traces held (results/dflash_cpu.log, dflash_pruned5.log)
-> file upstream issue. Meanwhile: DFlash2 = 16GB-tier lever + post-fix 12GB
candidate; MTP (certified lossless, +68%) remains the shipping 12GB config.
Tool hardening banked en route: prune handles arbitrary vocab-dim tensors +
preserves 3D shapes (two real bugs found by our own verify gates).

## Full Test verdict — the ASCII vocab prune is NOT quality-free (2026-09-01)

Full HumanEval-164, matched protocol (IQ3_XXS, MTP n=2, q8_0 KV, greedy):

| Build | HumanEval | Size | Δ |
|---|---|---|---|
| IQ3_XXS unpruned | **82.3%** (135/164) | 10.92 GB | — |
| IQ3_XXS-ASCII pruned (vocab 248,320 → 127,947) | **77.4%** (127/164) | 10.37 GB | **−4.9 pts** |

9 tasks pass→fail vs 1 fail→pass; **McNemar exact p = 0.0215 — real damage, not noise.**

**Mechanism (visible in the identity probe):** outputs are byte-identical until the
first moment the model's greedy argmax is a pruned token (`→`, `–`, typographic
punctuation in comments/docstrings). The prune forces a detour onto the second-best
token; most detours stay coherent, some derail solutions.

**The physics punchline:** the measured cost matches the quality-per-byte exchange
rate almost exactly. −0.54 GiB × ~10 pts/GiB ≈ −5.4 expected; −4.9 measured.
**Vocab pruning is not an arbitrage — it buys bytes at the same price as
quantization.** The embedding rows of "unused" tokens were not dead weight; they
were load-bearing through the greedy path.

**64K fit test: FAILED (CUDA OOM)** even on the pruned build with q4_0 KV and
-b 256 -ub 128 — so the prune's intended payoff (context headroom) did not
materialize at 64K. Context-fit bisect queued (ctx_bisect.tsv) to find the true
max for both builds.

**Decision consequence:** the pruned build does NOT ship as flagship. Unpruned
IQ3_XXS (82.3%, 35.3 t/s) remains the certified baseline. A gentler
frequency-based prune (keep top non-ASCII tokens seen in real generations) is the
only re-attempt worth considering, and only if the bisect shows the reclaimed
VRAM actually converts into a context tier we can't otherwise reach.

**Silver lining:** canary2 running on a build with *known, quantified* damage
(p=0.0215) is a true-positive calibration point for the gauge — if canary2 fails
to flag the pruned build, the gauge is broken.

## Context bisect + canary2 status (2026-09-01, follow-up to Full Test)

**Context bisect (q4_0 KV, -b 256 -ub 128) — the prune DOES buy context:**

| Build | Max ctx w/ MTP | Max ctx w/o MTP |
|---|---|---|
| IQ3_XXS-ASCII pruned | **57,344** (11,952 MiB) | **65,536** (11,676 MiB, 368 MiB headroom) |
| IQ3_XXS unpruned | 40,960 (11,860 MiB) | 65,536 at **12,006/12,044 MiB — knife-edge, not shippable** |

Revised prune verdict: not "failed" but a real, quantified trade — **−4.9 HumanEval
pts ⟷ +16K context with speculation (+42%)**. This becomes an optimizer *tier*,
not the flagship: default users get unpruned 82.3% @ 40K; max-context users can
opt into pruned 77.4% @ 57K. Both at ~35 t/s with MTP.

**Canary2 first pass was invalid as a divergence test:** `anchor_reference.json`
was never generated, so all three runs silently fell back to judgment-probes-only
(absolute) mode. Result confirms the v0 lesson from the other side: absolute
probes alone scored flagship ≡ pruned identically and ranked Q2_K_XL *above*
flagship — non-discriminative. The reference-divergence leg is the load-bearing
half. Fix running: Q8 anchor (partial offload, ~24/64 layers on GPU) generates
the reference, then all three builds re-score against it
(`anchor_then_divergence.sh` → results/canary2_div_*.json).

## Chain complete — canary2 divergence verdict + adaptive-depth verdict (2026-09-01)

**Canary2 vs Q8 anchor (the gauge's first real test) — PARTIAL PASS:**

| Build | HumanEval (truth) | Divergence from anchor |
|---|---|---|
| IQ3_XXS flagship | 82.3% | **0.288** (closest — correct) |
| Q2_K_XL | 72.0% | 0.302 |
| IQ3_XXS-ASCII pruned | 77.4% | 0.308 |

- PASS: flagship correctly closest to anchor; both damaged builds diverge more.
  The gauge detects the prune damage that absolute probes scored as identical
  (0.6842 == 0.6842) — divergence is confirmed as the load-bearing signal.
- FAIL: misorders pruned vs q2kxl relative to HumanEval, and the
  flagship-vs-damaged margin is ~2 probe items (n=54) — a detector, not a
  ranker, at this battery size. Caveat: the pruned build has a *structural*
  divergence source (forced token detours) beyond its quality loss, so some
  misordering is expected physics, not pure gauge error.
- Gate B: divergence leg validated in direction; needs a larger probe battery
  (n≈200+) before quantitative use.

**Adaptive MTP draft depth (merged upstream since our cherry-pick) — UNUSABLE
on 12GB as implemented.** `--spec-type draft-mtp-adaptive` creates a SECOND
full context against the target model (log: "creating MTP draft context
against the target model") and OOMs even at c=8192 with ~900 MiB free and
q4_0 draft KV. Plain `draft-mtp` shares the main context and is unaffected.
Upstream contribution opportunity: make the adaptive path share the main
context. Until then the 2.60×-on-code claim is datacenter-only — same
consumer-gap pattern as DFlash2.

Baseline reconfirmed in these runs: fixed MTP n=2 = 33.7–33.9 t/s
(c=8–16K, q8 KV, small buffers), acceptance 0.759.

## Predictive-KV lane, experiment 0: KV-read share of decode at depth (2026-09-01)

llama-bench, IQ3_XXS flagship, -fa 1, tg64 at filled depths (raw decode, no speculation):

| KV type | d=0 | d=8K | d=16K | d=24K | d=32K |
|---|---|---|---|---|---|
| f16 | 20.36 | 19.81 | **19.26** | OOM | OOM |
| q8_0 | 20.53 | 19.22 | 18.12 | 17.14 | 16.27 |
| q4_0 | 20.26 | 18.92 | 17.75 | 16.69 | 15.72 |

**Finding 1 — KV bytes are NOT the long-context bottleneck. Quantized KV is
SLOWER, not faster:** q4_0 (half the bytes of q8_0) loses ~0.3-0.6 t/s at every
depth, and f16 is the FASTEST at depth (+6.3% over q8 at 16K). The depth
slowdown in quantized KV is dequantization COMPUTE, not memory reads. Folklore
inverted: KV quantization is purely a capacity trade, bought at a speed cost.

**Finding 2 — the eviction ceiling on this hybrid model is small:** perfect
oracle eviction (depth→0) recovers only +26% at 32K (16.3→20.5). Real methods
(SnapKV-class keep 10-20%) would get a fraction. The architecture already did
the big eviction: 48/64 layers hold no KV at all. Predictive KV eviction =
LOW priority on hybrid models; re-rank for dense models (5080/16GB tier).

**Optimizer rule extracted:** on hybrid models choose the LARGEST KV precision
that fits the target context — f16 for ≤16K, q8_0 beyond — never q4_0 for
speed reasons. (f16 OOMs above ~16K on 12GB with this model.)

**Residual slowdown even at f16 (−5.4% at 16K) is linear-layer state/compute
growth — invisible to any KV technique; consistent with the hybrid-overhead
finding from S1.**

## Inverted twin (axis B): mechanism confirmed, magnitude insufficient on DDR4 (2026-09-01)

Q8 (29GB, 93.3% HE) hosted in RAM as VERIFIER, IQ3_XXS on GPU as drafter:

| Config | decode t/s | acceptance | vs baseline |
|---|---|---|---|
| Q8 pure-CPU baseline | 1.15 | — | — |
| + IQ3 GPU draft, n=8 | **1.63** | 0.331 | **+42%** |
| + IQ3 GPU draft, n=16 | 0.99 | 0.164 | −14% (over-drafting) |

**Mechanism works** (speculative verification lifts a RAM-hosted model), but two
walls: (1) IQ3↔Q8 greedy agreement is far lower than same-model intuition
suggests (33% acceptance at depth 8 — the α-vs-precision curve, which run C
measures, is exactly this number's anatomy); (2) the Ryzen 3600 + DDR4 verify
pass is slow. Even DDR5 scaling (~2.5×) lands ~4 t/s — below the 15 t/s gate.
**Parked as interactive tier on dense 27B/DDR4.** Stays alive as: (a) async
"quality pass" mode, (b) the MoE-verifier variant (3-6GB active per token
changes the math ~10×), (c) Mac unified-memory variant (400+ GB/s).

## MTP profiling + CPU drafter latency — two verdicts and one MAJOR correction (2026-09-01)

Full reports: results/mtp_profile.md, results/cpu_drafter_latency.md.

**1. The speculation overhead is found and measured.** Per-round cost fits
T(n) = 47.8ms + 13.43ms·n (R²=0.999). Each drafted token costs 13.43ms vs the
1.48ms bandwidth predicts; a draft-forward-free control (pinned-depth ngram)
shows **≥64% of the overhead is runtime GDN state management (snapshot/rollback),
not the MTP head**. Removing just that term: n=3 goes 35.7 → **50.8 t/s (+42%)**.
This is now the program's #1 speed lever, and it's an llama.cpp engineering
project (TreeWY-style state handling), not a training project. Also: verify
batching is free (T0 = 0.97× base), MTP depth is VRAM-capped at n≤5 on 12GB,
--spec-draft-p-min 0.50 buys +1-3%.

**2. CORRECTION — the ngram-mod 86-148 t/s numbers were a benchmark artifact.**
Re-firing the SAME prompt lets ngram replay its own previous answer (mean
accepted run 56.7 tokens). On 5 DISTINCT prompts ngram-mod issued **zero
drafts: 1.00×, no gain**. All prior "119-134 t/s warm" and today's "143-148
sustained" claims are hereby retracted for novel generation. ngram remains
real ONLY for re-emission workloads (a model rewriting file content it has
already seen in-context — genuine in edit loops, but must be re-measured on
honest edit tasks before any number is quoted). Earlier public/paper framing
must not cite the old numbers.

**3. CPU drafter (Companion shape b): viable at b2 (314M), b1 is dead.**
Measured on the Ryzen: b2 = 139ms/block vs the honest 224ms deadline (which is
"beat shipping MTP", not "beat nothing"). Requirements: the **32k-vocab head is
mandatory** (248k head alone costs 306ms > whole deadline; our 128k ASCII prune
halves it — not enough), 6 threads not 12 (SMT contention), fp32 for the big
GEMM on Zen 2 (no bf16 units), context projection must be cached. To beat MTP,
a b2 drafter needs mean 4.97/7 slots accepted (~71%) on the user's code — this
is the concrete bar the trained drafter must clear. Achieved DDR4 bandwidth:
8.3 GB/s effective, not the 30 nominal — folklore corrected again.

## Run B verdict: EoRA-class correction adapters DO NOT PAY on this model (2026-09-01, $3.10)

Pathway PROVEN (first GGUF-rail LoRA correction pipeline: BF16-vs-dequant residual,
imatrix-whitened, direct GGUF LoRA write — 505/866 tensors correctable incl. all
DeltaNet layers; artifacts + runbook in local_inference_program/eora/). Method
KILLED on measurement: best adapter (445 MiB) buys −0.44% perplexity while a byte
spent on quantization bits removes 1.2–3.3× more error at every rank — and the
445 MiB doesn't exist in our VRAM budget without dropping MTP (+68%).

**Transferable law: correction adapters and dynamic bit-allocation are
SUBSTITUTES, not complements.** EoRA's published wins were vs uniform GPTQ;
Unsloth Dynamic already spent its bits along the directions EoRA would recover
(capture 4.9% vs 1.5% noise floor). Kills ARCHITECTURE.md contribution #2 as
scoped; effort redirects to #3 (verifier-first co-optimization).

Byproducts: (1) METHODOLOGICAL WARNING — small-model pilots overstate low-rank
capture by ~the hidden-dim ratio (0.6B said 29.5% go; 27B measured 4.9% no-go);
(2) spectra-based capture metric correctly ranked all three adapters → optimizer
can pre-screen allocation candidates at ms/candidate without inference;
(3) suspected upstream bug: llama-perplexity at -c 512 on hybrid arch returns
garbage (uniform logits) — perplexity-testing this family needs -c 2048+.
Open door (only one): full-covariance whitening must lift capture 4.9%→8.5% to
overturn; well-defined ~$3 test, low prior.

## Run C verdict: the α-precision curve — a law, a placement theorem, and 3 corrections (2026-09-01, $7.66)

Full doc: inference_optimization_research/13-ALPHA-PRECISION-CURVE.md; raw: local_inference_program/alpha_curve/ (34 configs, ~60,700 tokens, all byte-exact lossless).

**The law [M]:** same-model twin disagreement follows 1−α₁ = 1.35·2^(−1.13·bpw)
(R²=0.965) — halves per ~0.9 bits. Knee (not cliff) below 2.1 bpw. Family-specific
constant; dense-model exponent unmeasured.

**The verdict [M]:** GPU-hosted twin drafting is DEAD on this architecture — 0 of
25 configs beat no-spec. Cause is c not α: a twin draft step costs 0.70-0.97 of a
target step (latency-bound at batch 1 — quartering drafter bytes cut step time
only 28%). MTP wins with one of the WORST α values (0.738 ≈ a 2-bpw twin)
because its c ≈ 0.07-0.09. Confirms: viable drafters must be architecturally
small (b2-class), not numerically small. Twin-as-CPU-drafter dead by 50×.

**The placement theorem [M]:** same drafter, same α₁=0.955 → 0.88× on one GPU
but 1.77× with the target on CPU. α is a property of the model pair; the
VERDICT is a property of the placement. (And heterogeneous CPU/GPU placement is
numerically free — byte-identical greedy output over ~3,000 tokens.)

**CORRECTIONS to prior entries:**
1. Inverted twin (axis B) is STRONGER than banked: on L40S+CPU it measures
   α_raw 0.745, +77% at n=8 and +65% at n=16 (vs our 3060's 0.331/+42%/−14%).
   The 3060 numbers likely suffered the weak-CPU verify + short battery;
   RE-TAKE on the 3060 before quoting either. Lane upgraded: parked → active.
2. MTP acceptance 0.759/0.786 in earlier entries is a warm-repeated-prompt
   artifact (same failure family as the retracted ngram numbers). Novel-battery
   value: α_raw 0.640. All speed math derived from 0.76-0.79 needs rechecking.
3. α is deterministic and generation-length-dependent (0.955@128tok →
   0.887@320tok) — batteries must state length.

## State-rewind verdict: math exact, numerics dead — and the +42% RETRACTED (2026-09-01)

**Algebraic rewind: KILLED, five ways.** The GDN update inverts exactly via
Sherman-Morrison (derivation verified against modeling source), but the inverse
is an EXPANSIVE map while forward replay is contractive (proven as a dominance
theorem). On the model's real trained gates: fp32 depth-3 rewind p99 error
1.2e-3, worst 183%; bf16 → NaN. Forward replay is bit-identical (0 ULP).
Also pre-empted: TreeWY (arXiv 2608.20961) covers this family. Surviving
original claim: TreeWY's ~1e-7 approximation silently breaks speculation's
losslessness contract; exact checkpoint-and-replay does not — an argument for
the simpler method not yet made in the literature.

**MAJOR CORRECTION — the "+42% (35.7→50.8 t/s)" claim is RETRACTED.** The
mtp_profile control was flawed: ngram-mod takes ZERO state snapshots (absent
from need_n_rs_seq(); confirmed by its own VRAM column — depth 63 at 410 MiB
BELOW baseline). So the 8.60ms/token attributed to "state management" is
actually verify+graph cost that NO checkpoint scheme can touch. Realistic gain
from CPR checkpoint-and-replay: **+3-7%, not +42%.** Its real payoff is VRAM:
~303 MiB at n=3 (144 MiB/depth GDN state now exactly accounted), unblocking
deeper MTP (n≥7 OOM).

**Speed levers standing after this retraction:** kernel-level GDN efficiency
(profile pending), drafter α (run A), grammar-forced fast path, placement
plays (run C theorem), prefill/session latency. The 50 t/s headline moves from
"one fix away" back to "sum of smaller measured gains."

## Ladder extension verdict: the no-arbitrage law holds at EVERY rung (2026-09-01, +$0.94)

Founder's hypothesis (stupider base + bigger adapter at matched bytes) tested at
IQ2_XXS and IQ2_S vs IQ3_XXS raw at matched ~10.93GB footprint: adapters lose by
2.0-2.7× at every rung; break-even needs 57-69% residual capture, measured 13-17%.
**Mechanism (the real find): the quantization residual's spectral SHAPE is
invariant along the ladder** — capture at matched rank identical to 0.3pp across
bases while error magnitude spans 3.24×. Adapter headroom is a constant of the
architecture, not the bit-rate. With the exchange rate and the prune result:
three independent confirmations that **bytes buy quality at one price; there is
no arbitrage on this model.** Total cost of the entire adapter investigation: $3.99.

## Kernel profile verdict: the 41% shortfall is the QUANT FORMAT, not the hybrid arch (2026-09-01)

Full report: results/kernel_profile.md (nsys + ncu, both worked).

**CORRECTION to prior belief:** the bandwidth gap we attributed to hybrid-GDN
compute is actually the **sub-4-bit i-quant codebook kernels**. One kernel
(mul_mat_vec_q) is 89.6% of the decode step; GPU is 99% busy (NOT launch-bound).
The model's own 4-bit+ tensors achieve 90.1% of peak — the "dense-model 90%"
reproduced inside our model — while its sub-4-bit LUT quants run COMPUTE-bound
at 59.5% (SM 65-84% busy, DRAM 30-47% idle). Confirmed profiler-free: achieved
bandwidth FALLS as quants shrink (62.8%→47.0%); IQ2_XXS is 34% smaller than
IQ3_XXS but only 12.8% faster. **Smaller-is-faster saturates below 4 bits —
a new term in the optimizer's speed model.**

**KV mechanism found:** quantized KV switches attention to a compute-bound
kernel (57 GB/s, 16% peak) vs f16's bandwidth-bound path (314 GB/s, 87%).
q4_0 reads 47% fewer bytes and takes 2.9× LONGER. Shipping config impact:
f16 KV is a free **+7%** over q8_0 at d=16K (19.33 vs 18.07 t/s) — but f16
only fits to ~16-20K on 12GB. Optimizer rule sharpened.

**The one kernel worth writing:** sub-4-bit GEMV at 4-bit efficiency = **+33%
ceiling (20.4→27.2 t/s raw)**. Pure CUDA engineering, no quality cost, benefits
every sub-4-bit GGUF user on every model. No third target exists (everything
else ≤2.75%; perfect fusion bounds +7.4%).

**Also:** GPU-side GDN state work is only 2.386 ms/token — confirms the 8.60ms
draft overhead is host-side runtime cost (third independent confirmation).

## Run A verdict: the trainer WORKS; the drafter isn't deployable yet (2026-09-01, $10.89)

**Gate: PASS, decisively.** b2core (367.8M) trained 1600 steps on the founder
corpus: block acceptance **0.7343 vs bigram 0.2660** (+0.4683, 95% CI
[+0.4346,+0.5025], n=3741 windows, 10k paired bootstrap) — 2.76× bigram;
untrained control exactly 0.0000. **Reverses the toy null.** The invented
selector loss earns +0.0342 [CI clear of zero]. λ monotone: 0.25 > 1.0 > 2.0
(in-training 512-window evals ranked them BACKWARDS — paired bootstrap
required; standing warning). λ=0 is the top next ablation.

**Deployment bar: NOT cleared.** Mean accepted 0.734/7 slots vs the 3.85/7
b2 latency bar (~19% of it); per-slot decay steep (0.435→0.067). Verdict:
$11 bought proof that the reconstructed trainer + personalization signal are
REAL; a deployable drafter needs the v2 program (λ→0, 10-50× training compute,
32k pruned head, on-policy/self-distillation from live traffic).

**Two upstream-relevant bugs found:** (1) Qwen3.8's published config lists
phantom MoE routers (mlp.gate) and transformers matches skip-patterns by
unanchored prefix → all 64 gate_proj tensors silently load as raw FP8 bytes
(~10,000× too large) with no error — affects anyone fine-tuning this model;
(2) output_loading_info reported missing_keys=0 with 108/109 tensors unloaded.
Both worth public reports alongside the DFlash segfault.

AWS campaign total: **$25.9** (A $10.89, B+ladder $3.99, C $7.66, rewind $0.24,
misc EBS ~$3). All instances terminated+verified, zero strays.

## Sweep verdict: EXL3 fit reversal, REST adoptable-but-weak, HYPIC opens, MInference dead (2026-09-02)

Full doc: inference_optimization_research/15-RETRIEVAL-CACHE-SPARSE-EXL3.md (6 corrections filed).

1. **EXL3 rail: GATE PASSES — and our "doesn't fit" was wrong.** ARCHITECTURE.md
compared download size to VRAM: exllamav3 forces the 2.37GiB embedding to host
RAM and never instantiates the 0.86GiB vision tower → 3.0bpw text VRAM =
**9.72GiB vs our GGUF flagship's 11.00** — fits to ~40K with +1.3GiB headroom,
hybrid GDN first-class, MTP head preserved. No published hybrid quality numbers
exist (pre-Aug-23 numbers stale: 144 GDN linears were silently unquantized).
Open risks: sm_86 speed (nobody has run this model on a 3060), realized MTP
gain. Quality head-to-head → AWS battery; speed check → 3060 queue.
2. **REST/ngram-cache: adoptable TODAY** (--spec-type ngram-cache -lcs) but
expected ≈0 gain on hybrids: the 8.60ms/position runtime tax means break-even
needs mean accepted >4.41 vs REST's best published 2.65. One-number pass/fail
in the battery. **Corollary: the entire draft-TREE family is dead on hybrids**
(a 64-token trie = one forward on dense, 550ms here).
3. **CacheBlend dead, but our "can't relocate recurrent state" objection was
FALSE**: GDN updates are affine in the state → chunk states compose exactly if
the transition operator is cached. HYPIC (2607.01299) delivers 3.25× TTFT on
our model family. 30-60 day project, largest absolute payoff on the board;
gated on a 1-day prefix-cache measurement + 2-day drift sim.
4. **MInference/sparse-prefill: DEAD by our own arithmetic** — first-ever fit of
our prefill curve shows the attackable N-scaling term is only 4.5%/8.5%/18.9%
of prefill at 8K/16K/41K. Perfect implementation saves ≤18s at our ceiling. Do
not start.

## 3060 overnight triple: admission ships, restore deploys, twin re-corrected (2026-09-02)

**1. Companion admission on the real 27B: VIABLE — and the winner is the free
scorer.** lexical@0.25 keep: **97.8% accuracy vs the 93.3% full-context
reference at 3.11× net prefill speedup**; at 32K, lexical@0.10 = **100% at
7.01×**. Controls crushed (random 57.8%, last 35.6%). The model-based scorers
are dead weight (cheap-helper theorem, third confirmation: embedder costs
+325ms for negative accuracy). SHIP: lexical scorer on CPU, zero VRAM.
Constants settled: prefill = **~500 t/s** (225 and 25.63 both wrong) and
essentially LINEAR on hybrids → admission buys ~1/k, no quadratic bonus.
Caveats: synthetic NIAH flatters lexical; real-traffic replay is the top
follow-up. CORRECTION: -c 40960 ceiling was measured WITHOUT MTP at q4 KV
small-buffers; with MTP at q8 KV the ceiling is 24-32K (OOMs at 32768) —
context claims must state the full config.

**2. Session restore: DEPLOY-READY at 18.0×** (0.925s vs 16.7s re-prefill).
Byte-correctness proven against a batch-split noise floor + 6/6 ground-truth
probes at three depths; MTP-compatible at full speed; dense models unaffected
(bug is hybrid-specific). KEY INVERSION: the RAM prompt-cache delivers ZERO
reuse on hybrids — the patch isn't an optimization, it's the only working
mechanism. Honest number: 18×, not 40-50×. Deploy caveats: ~1GB per 8K-token
save; llama-server is a thin wrapper (deploy the .so, not the binary).

**3. Inverted twin: our original 3060 numbers were RIGHT** (retake: α 0.320,
+37.6% @ n=8 vs original 0.331/+42%). Correction-to-the-correction: run C's
L40S α=0.745 is likely a warm-repeat/code-only artifact (α here spans
0.186-0.614 by CONTENT) — re-take before ever quoting +77%. **The real law:
speculation speedup is a property of the CONTENT**: +129% structured, +110%
code, −2 to −4% prose. Optimizer rule: acceptance-gated speculation (disable
below α≈0.25). 6-thread pinning is a baseline win (+11.5%), not a speculation
win. Lane verdict: batch/unattended work only (1.73 t/s best).

## AWS quality battery (2026-09-02, ~$19): five verdicts and ONE POTENTIAL EARTHQUAKE

**THE OPEN EARTHQUAKE — sub-4-bit quants may be BROKEN on the 3060, not weak.**
Same GGUF files, same harness lineage, L40S: IQ3_XXS **93.9%** (3060 recorded
82.3%), Q2_K_XL **93.9%** (vs 72.0%), IQ2_XXS **80.5%** (vs 49.4%) — while the
Q8 anchor reproduces EXACTLY (153/164, harness validated). Version, speculation,
and template ruled out as confounds on-instance. Remaining variable: the
platform (sm_86 vs sm_89) — consistent with kernel_profile.md's finding that
sub-4-bit i-quants take a separate LUT code path. If the 3060 arm confirms:
(a) the flagship's 11-pt gap to anchor is a KERNEL DEFECT, not quantization
cost; (b) the 9.7pts/GiB HumanEval exchange rate collapses (Q2_K_XL, IQ3_XXS,
Q4_K_S, Q8 mutually indistinguishable, all McNemar p=1.0 on L40S); (c) it's a
major upstream correctness bug affecting every sm_86 user of sub-4-bit GGUFs.
STATUS: strong inference, NOT proof — decisive 3060 re-run launched.

**Test-time reliability curve [M, L40S]:** 93.9% pass@1 → **96.95% at N=3,
97.56% at N=5** verified retries; 289 tok/task, 1819 tok per extra solve.
Retries do NOT rescue cheaper quants (Q2_K_XL: 2.9× worse tokens/solve) — the
quant difference lives in the retry channel.

**Agentic ladder [M]:** the damage below ~3bpw is FORMAT/ADHERENCE, not
reasoning: edit-format compliance separates decisively (Q2_K_XL 67.6% vs
IQ3_XXS 94.1%, p=0.0117; IQ2_S emits unclosed fences on 52% of tasks with
correct code inside). IFEval blind in the flat region (Q8 ranks LAST).
Lenient benchmarks cannot see this damage — instrument rule for all future certs.

**Long-context [M]:** NIAH 15/15 at all depths to 48K both KV tiers; multi-hop
100%@8K → 65-75%@32-48K. KV q4 vs q8: p=0.375 — **no quality difference;
KV quantization confirmed as pure capacity trade.**

**Dense α-curve [M]:** twin drafting dead on dense too (0/6 beat no-spec,
acceptance 0.70 — c kills it, not the GDN tax). The cheap-helper theorem is
UNIVERSAL at batch 1. (Exponent 0.741 is aggregate-acceptance, not α₁ — not
comparable to run C's 1.13 without re-derivation.)

**EXL3 leg:** skipped by gating race (doc 15 landed after the check) — still
pending, folds into the 3060 platform investigation.

## PLATFORM ARM VERDICT: earthquake closed NEGATIVE — the cause was OUR harness flag (2026-09-02)

Full forensics: results/platform_arm/VERDICT.md.

**The GPU is exonerated with the strongest possible evidence:** Q8 greedy on
both platforms → 54/54 paired tasks with IDENTICAL completion-token counts,
0 discordant. No sm_86 defect exists. No upstream report.

**Root cause, positively identified:** the legacy harness never sent
enable_thinking=false, and the served chat template DEFAULTS THINKING ON.
Every historical quality number was measured with the model burning its
1024-token budget on reasoning — 26/29 legacy failures are truncations.
Reconstruction reproduced the old 82.3% task-for-task (164/164 match);
flipping the single flag = +18 tasks (p=0.00053). Every other suspected
confound combined was worth exactly one task (p=1.0).

**CORRECTED QUALITY LADDER (3060, HumanEval-164, thinking off, greedy):**
IQ2_XXS 78.0% · Q2_K_XL **93.3%** · IQ3_XXS **92.7%** · Q8 anchor 93.3% —
flagship is statistically indistinguishable from the uncompressed anchor.
**Retention ≈ 99%. The ~10pts/GiB HumanEval exchange rate is RETIRED.**
Above ~2.9bpw, quantization damage on this model is visible ONLY in
format/adherence instruments (battery: Q2_K_XL edit-compliance 67.6% vs
IQ3_XXS 94.1%, p=0.0117) — which is now the flagship's justification and the
certification instrument of record.

**Entries requiring re-audit under the corrected protocol:** the ASCII-prune
−4.9pts verdict (legacy protocol both arms; relative flips may be
truncation-driven — DO NOT QUOTE until re-run), the retention table, canary2
anchor references, GATES quality gates. Speed physics unaffected.

**New standing rules:** (1) harness canary — HumanEval mean tokens/task >350
means the thinking switch is not taking effect; (2) sub-2-bit i-quants are NOT
numerically reproducible across GPU architectures (IQ2_XXS 67% identical-token
rate cross-platform, unbiased p=0.45; same-GPU control 100%) — never A/B two
quants across two different cards.

## Speed re-certification under corrected protocol (2026-09-02)

Full doc: results/speed_recert.md. Noise floor ±1%, drift-controlled.

**Official config UNCHANGED, now for measured reasons: IQ3_XXS + MTP n=2 +
q8_0 KV + c=16384 → 34.39 t/s (1.719×), 11,958 MiB peak.** At working depth
(13.5K filled) it reaches **36.00 t/s, α=0.981, 1.99×** — deep-context is
speculation's BEST regime under the corrected protocol.

Config-lever corrections: f16 KV at c=16K is NOT a config (passes load, OOMs
on first request — new rule: certify VRAM from peak-during-requests, three
configs failed this); f16 ceiling is c=12288. The +7% f16 gain is depth-scoped
(+5.4% measured at 13.5K, noise at shallow). Optimal n stays 2; α SHIFTED down
(thinking tokens drafted easier than answers — legacy α inflated). Q2_K_XL is
NOT faster (+0.75%, kernel-profile prediction wrong) but buys 1,254 MiB → the
capability tier (f16 KV @16K, MTP n=4). Content-gating: unnecessary in this
lane — α* = 0.2445 derived from this run's own cost model (predicts all cells
within 1.4%); worst content sits 2.45× above the gate. MTP stays ON always.

**THE dominant speed finding: the thinking fix is worth ~2.8× wall-clock per
task** (4.79s vs 13.38s per HumanEval task; 158.8 vs 474 tok) — ~4× larger
than every config lever combined. Legacy protocol posted HIGHER raw t/s while
doing the job 2.8× slower: raw t/s is not work speed.

**Path to the 55 t/s goal, honestly revised:** config levers are exhausted at
~34-36. The kernel offensive (raw 20→27 ceiling → ~46-50 with speculation) is
now effectively the ONLY path, with drafter v2 closing the gap. 
**Open flag:** at n≥3-4, greedy output showed non-reproducibility/divergence
in one-prompt observations — deep-MTP losslessness needs a dedicated check
before any n>2 config ever ships.

## Kernel offensive verdict: +10.1% raw CAPTURED, byte-identical — attenuated to +2.5% under speculation (2026-09-02)

Full evidence: local_inference_program/kernel_offensive/ (patches, exhaustive
identity proofs, ncu counters, SASS histograms).

**Root cause corrected:** not the codebook gather — sm_75+ removed SIMD-video
instructions and nvcc EMULATES them (4-5 instructions each); the i-quant sign
machinery is 18 of every 20 inner-loop instructions. Fix: a carry-free SWAR
multiply spreads sign bits (valid because no i-quant codebook byte is zero) +
2-rows-per-block tiling. Superadditive: +4.2% and +2.7% alone, **+10.1%
together** (20.48→22.54 raw; +10.2% at depth; prefill untouched — clean
control). Correctness: 133,392-case exhaustive proof, backend tests, 3×200-tok
greedy transcripts byte-identical across all four builds, SASS of untouched
types byte-for-byte unchanged. Also caught nvcc silently compiling
__byte_perm sign-mode to identity (upstream-noteworthy on its own).

**MTP-path A/B (main-loop measurement):** shipping config stock 35.8 →
patched **36.7 t/s (+2.5%)**, acceptance identical (0.8356 both — correctness
corroborated). **The attenuation is the finding: speculation amortizes the
GEMV, so raw-decode gains divide by the round structure.** Per the recert cost
model (T_round n=2 = 74.45ms, one verify forward ≈ 45-50ms), the ROUND
OVERHEAD (~24ms/round: draft forwards + host graph/sync) is now CO-DOMINANT
with the GEMV. Path to 55 t/s revised: remaining kernel headroom (+21% raw →
~+5% effective) + **round-overhead decomposition and attack** (the new #1
unknown: how much of 24ms/round is reducible?) + drafter v2.

**Upstream:** two PRs staged (SWAR signs = low-risk first; rows-per-block
needs multi-GPU validation). llama.cpp's AGENTS.md requires HUMAN-owned
contributions — the founder must understand, own, and submit these patches
personally; both ideas are two sentences each. FOUNDER DECISION PENDING.

## EXL3 head-to-head: KILL — flagship stays GGUF (2026-09-02, $2.33)

EXL3 3.0bpw = 92.68% / 3.5bpw = 93.29% HumanEval vs GGUF IQ3_XXS 93.90% — all
McNemar p≥0.73, mutually indistinguishable; both rails sit on the same ~93%
saturation ceiling. VRAM: doc 15's "+1.3GiB headroom" was an apples-to-oranges
KV comparison — matched-KV measured saving is only **~0.2GiB**, and its [D]
budget under-counted real overhead by +0.61GiB (measured beats derived, again).
Speed: EXL3 ~9% SLOWER at matched settings on L40S (sm_86 untested, moot).
Dual-rail contract item: resolved — single GGUF rail for this model.
Side-finding: all 26 published EXL3 branches of this model predate the GDN
quantization fix (v1.4.2 < PR#298); null applies to published quants, though
the 3.5bpw arm suggests the ceiling, not the bug, binds.

## 3060 chain complete: stages 1-4 (2026-09-02)

**Stage 1 — demo number FINAL: 19.35× wall-clock, 5/12→12/12** (stock defaults
vs our stack, same card). Pooled decode 41.4 t/s on HumanEval content (α=0.99).
Session restore adds 18.8-30.8× end-to-end on long-context re-entry, warm
outputs byte-identical. CAVEAT: restore build and kernel patch never merged —
integration task for revv.

**Stage 2 — prune EXONERATED AND REVERSED:** corrected protocol: pruned 93.29%
vs flagship 92.68% (p=1.0, all 9 legacy losses recovered). Free 555MB.
canary2: reproducible bit-for-bit but resolution floor ~11pts at n=54 — never
quote sub-11pt canary deltas.

**Stage 3 — round overhead SOLVED and the lane CLOSED:** T_round = 47.31 +
13.83n (R²=0.9998). Host overhead ZERO (intercept below T_base); rollback
free; 97% GPU busy; no knob helps. Budget: verify 49.6 + batch-widening 16.0
(8.01ms/extra verified token — ALU-bound) + 2 MTP drafts 11.6. **Ceiling
formula: (n+1)/(47.31+13.83n) → 40 t/s @n=2, 45 @n=3, 51.5 @n=5 at PERFECT
acceptance; 55 needs n≥7.** The drafter program is the only road to 55.
**ACTIONABLE: the ASCII prune cuts draft cost 10.5% (LM head) → shipping
34.15→36.10 t/s (+5.7%), −332MB, at identical acceptance.**

**Stage 4 — kernel lane CLOSED for our models:** Q3_K register-cliff fix =
+26.6% kernel-level (4-gate verified) but Q3_K is 1-7% of our builds' bytes →
end-to-end noise. Worth upstreaming for Q3_K_M/S users (3rd PR candidate).
Q2_K: no clean win exists. Sub-4-bit kernel work: done.

**NEW CERTIFIED-CONFIG CANDIDATE (revv v1.1): ASCII-pruned flagship + kernel
patch + MTP n=2 → ~38 t/s [est, patches unmerged], 93.3% HumanEval, −887MB
total headroom.** The pruned GGUF is OUR artifact → publish on HF under
Mericanii. Needs: patch merge + one certification pass.

## MoE tiering campaign verdict (2026-09-03, $1.75 of $50): hypothesis KILLED, better door found

Full doc: inference_optimization_research/17-MOE-TIERING-PROOF.md. 16.19M routed
calls traced on Qwen3.8-Flash-Next (60 prompts, 4 domains).

**Hot-expert/SSD-tiering: DEAD by measurement.** Pooled expert skew 1.83×
(kill gate was 2×); 90% of calls need 85.7% of experts (uniform: 90%); hot set
neither code-specific (Jaccard 0.239 ≈ cross-domain 0.247) nor persistent
(0.200 decile stability); LRU beats any static hot-set at every cache size —
the OS already runs a better policy than we'd ship. Measured caps: uncapped
6.1 t/s (CPU box) → 32GiB 4.5 t/s; pinning +0.4 t/s but break-even ~7.5k tokens.
**Methodological exports:** (1) per-layer imbalance factors (6-80× in papers)
are NOT cacheability — caches hold (layer,expert) pairs and the union washes
out; (2) our own doc-14 |W(τ)| instrument corrected — distinct-expert sets
saturate; call-weighted miss-rate is the statistic.

**Two bigger discoveries:** (a) 40% of Flash-Next is a per-layer embedding
table (28.8GB) that llama.cpp ALREADY lazy-streams from disk by default — the
tierable mass was never the experts and the tiering already shipped;
(b) **LICENSE KILL: Flash-Next's Qwen Community License requires a separately
negotiated license for AI-coding products (no revenue floor)** — dead for
Mericanii regardless of physics.

**The door that opened: Qwen3.6-35B-A3B** — 21.3GiB @4bit, BUILT-IN MTP,
Apache-2.0, hybrid GDN (our stack's architecture family), 3B active. Fits
64GB-RAM rigs outright and 32GB+12GB-VRAM rigs split, no tiering needed; field
report (P40 pair): 50-70 t/s. NO HumanEval-164 exists for it anywhere — our
certification harness on the ollama box (47GB RAM) is the natural next
campaign, $0. Candidate revv v2 flagship.

## DRAFTER V2 FINAL VERDICT: killed at this scale — architecture, not scaling (2026-09-03, $56.78)

Full artifacts: drafter_training/dflash_custom/runs/run_v2/ (705MB, all evals + best ckpt sha-verified).

**The curve [M]:** block acceptance plateaus at fitted ceiling 0.767/7 (power-law
fit, rms 0.008); best checkpoint (decay tail) **0.798 ± 0.019** vs run A's
0.734 — 10× compute + anneal = +8.7% relative, then flat. Patience watchdog
stopped the arm correctly (saved ~$12).
**Structural bound [M]:** block acc ≤ 7·p₁; bars need p₁ ≥ 0.550 (b3) / 0.710
(b2); best observed p₁ = 0.459 and unresponsive to both levers. NOT a near
miss — ~5× away in block acceptance.
**The probe [M, weakly powered]:** 2× data at fixed compute scored WORSE
(0.660 vs 0.734 — halved epochs dominate). Confounded by design (flagged
before running); reads "epoch/compute-limited at small budgets", not a clean
data law.
**v3 recommendation (parked):** architecture bet — stock 5-layer DFlash2 shape
or init_from the z-lab checkpoint, not more scaling of b2core-2L. Full 14.3M
corpus needs 440GB harvest (EBS, ~$1.30/day) if ever run.

**PROGRAM CONSEQUENCE: the 55 t/s road on the current flagship closes.**
Certified reality: 38-40 t/s (v1.1 pending full-164 cert). The speed frontier
moves to the model layer: Qwen3.6-35B-A3B certification (3B active, Apache,
built-in MTP, our architecture family) is the live candidate for 50+ natively.
Drafter program legacy: proven trainer, proven personalization signal (2.76×
bigram), three loader bugs found, the b2-bar methodology — reusable when the
architecture bet is worth $60.

## 35B-A3B CERTIFICATION VERDICT: quality TIE, +32.5% speed — v2 speed tier adopted (2026-09-03, $0)

Full report: results/cert_35ba3b.md. Best config: --n-cpu-moe 26 + MTP n=2,
q8 KV, c=16K → **45.58 t/s** (44.07 @13K depth), HumanEval **93.29%** and
edit-compliance **94.12%** — both statistically tied with the 27B flagship.
VRAM 10,524 MiB (+1.4GiB headroom); mainstream 32GB-RAM rig: comfortable
(host working set only 14.7GiB, 0 reclaim events). Apache-2.0 clean.
Weak flank: prefill 128 t/s shallow (−54%) / 365 @depth. MTP pays only 1.23×
on MoE (structural: verify batch fans across experts) — the speed comes from
3B active params. **Placement law: T(N)=14.45ms+0.4845ms·N (±1.1%) → N=0
extrapolates to 69 t/s (reconciles the P40 prior); UD-Q3_K_XL rung predicted
~55-58 t/s** — blocked only by /data at 99% (second quant not downloadable).

**FLAG — draft-mtp is NOT bit-exact (n=2, this model): 4/5 greedy probes
diverged (no-spec control 5/5 identical incl. across restart). Quality-neutral
(93.29 vs 92.07 no-spec, p=0.625). The 27B's "certified lossless" MTP claim
must be re-verified with the same protocol** — queued. Recommendation stands:
v2 SPEED TIER now; flagship promotion after an agentic wall-clock test
(prefill-heavy workloads may favor the 27B).

## 27B MTP losslessness recheck: NOT bit-exact — claim corrected everywhere (2026-09-03)

3/5 greedy probes diverge MTP-on vs off (control 5/5 identical across restart).
Same as the 35B finding; cause = batch-shape floating-point order (known
llama.cpp nondeterminism class). Honest claim, now in all docs/README:
**quality-neutral (full-164 per-task identical, p=1.0), not byte-identical.**
The historical "certified lossless" wording is retracted.

## Q3_K_XL rung verdict: 48.5 t/s at FULL certified quality — new best ship (2026-09-04)

Full sweep: results/q3kxl/ (ncm 8-41 × nospec/mtp2/mtp4, depth, VRAM traces,
tripwire clean). **Best: 35B-A3B UD-Q3_K_XL, --n-cpu-moe 16, MTP n=2 →
48.54 t/s decode, 205 t/s prefill, 11,832 MiB peak, host RSS only 7.2 GiB.**
Quality FULL-164 certified: **92.68% (MTP) / 93.90% (no-spec)** — ties the 27B
flagship and the Q4_K_XL rung. vs prior bests: +41% over the 27B (34.4), +6.5%
over 35B-Q4 (45.6), and prefill +60% over Q4's weak flank (205 vs 128).

**The 55-58 prediction MISSED (measured 48.5).** Law refined: the placement
slope did NOT scale with file bytes (slope ratio 1.10 vs file ratio 0.75) —
the CPU-side expert stream pays a per-quant DECODE tax (sub-4-bit unpacking on
CPU, mirroring our GPU kernel finding), and MTP speedup at deep GPU placement
compressed to 1.106×. Below ncm14: OOM wall. Optimizer note: the placement law
needs a quant-specific slope term.

**New shippable line: 48.5 t/s at 92.7-93.9% HumanEval on 12GB VRAM + ~8GB
host RAM — comfortably inside a 16GB-RAM mainstream machine.** Pending before
flagship promotion: agentic wall-clock A/B vs the 27B (prefill/tool-loop
weighted), adherence instrument on this rung, PR #28223/#27861 stack test.

## Turbo4 KV + overclock evaluation (2026-09-04, $0)

Full doc: results/turbo4_oc_eval.md.
**Turbo4 (buun fork): real, and it refines our KV law without breaking it.**
Mechanism (read from source): turbo KV materializes into f16 scratch → f16
attention + decode pass → structurally can NEVER beat f16 (confirmed: −1.7%
vs f16 at matched config). Among quantized KV it DOMINATES: 4.125 bpv (< q4_0)
AND faster than q8_0/q4_0 (+2.3-2.5% at 16K, −246 MiB) — retires "q8 beyond
16K" as a rule *when the codec is available*. Real prize: **c=32768 runs
(35.0 t/s) where q8_0 OOMs.** BUT the fork carries a global −1.70%/+182 MiB
tax that cancels the win at 16K and regresses the f16 ceiling to c=8192.
VERDICT: don't adopt now; track the codec (nothing upstreamed; PR closed);
reconsider for long-context tiers only. Quality: n=10 probes inconclusive-
cosmetic; full battery required before any claim.

**Overclock lane: effectively CLOSED on 30-series/driver-535 headless.** No
clock offsets without X; only lever = power limit 170→190W = **+1.4%** for
+11.8% power. revv must not promise tuning gains here. AUTOMATION TRAP:
nvidia-smi silently clamps above-spec requests while reporting success —
any tooling must verify clocks.sm/clocks.mem UNDER LOAD.

**Operational flag (bit us, will bite users): upstream context-checkpoint
default (32/slot × ~150MiB) OOMs configs near the VRAM wall** — first
checkpoint kills request #2 with a misleading cudaGraph error. Our v11
lineage carries the same default. Rule: set -ctxcp 0 (or size-aware) on any
config within ~500MiB of the ceiling. → revv serve fix queued.

## Provenance closed: the speed-tier cert file verified; identical-filename trap documented (2026-09-04)

The 35B-A3B Q3_K_XL cert ran on /data/models/coding/q36_35ba3b/ (17,227,569,440
bytes, 753 tensors = unsloth Qwen3.6-35B-A3B-**MTP**-GGUF) — the 48.5 t/s
certification is VALID as configured. TRAP DOCUMENTED: unsloth publishes
MTP and non-MTP repos with IDENTICAL filenames (16.85GB/733t headless vs
17.23GB/753t with blk.40.nextn); the headless twin silently no-ops
--spec-type draft-mtp. revv's registry pins repo+size; inspect gates on
tensors. A redundant headless download was deleted.

## 2026-09-04 — CPU-utility research sweep (literature, no new measurements)

Three-lane sweep (MoE overlap / attention-KV / CPU kernels); full report:
`experiments/inference_optimization_research/18-CPU-UTILITY-RESEARCH.md`.
Two corrections to prior entries:

1. **KV-offload premise CORRECTED**: llama.cpp `-nkvo` does not stream KV over
   PCIe — attention between KV store and output runs on the CPU
   (llama-graph.cpp:2666); only Q/output cross the bus. Host-KV is an ~8× DRAM
   penalty, not a PCIe wall. "Naive KV offload dead on PCIe" was wrong in
   mechanism (still slow, but beatable by ~5% sparse selection — see doc 18 §Tier 3).
2. **Q3 CPU decode tax mechanism identified**: ggml repacks Q4_K experts to an
   8x8 AVX2 GEMV (applies to MUL_MAT_ID, no batch threshold) but Q3_K has no
   repack path on any arch → nrows=1 fallback. Explains the 0.5332 vs 0.4845
   ms/layer placement-law slopes. Candidate fix is a requantize of CPU-resident
   expert tensors (--tensor-type "ffn_.*_exps=q4_K"), not a kernel.

Also flagged: MTP verify passes activate the UNION of drafted tokens' experts
(DraftExpert 2607.24434) → placement law is likely n-dependent; 48.5 t/s
operating point unverified as the (n, n_cpu_moe) optimum. Tier-1 measurement
ladder in doc 18 §Ranked actionables; nothing here is measured yet.

## 2026-09-05 — Experiment 3.2: CPU/CUDA split concurrency in ggml — GATE PASSED

Toy on AWS g4dn.2xlarge (ap-northeast-1, 41 min, ~$0.70; instance verified
terminated, all regions swept clean). Verdict: **concurrent CPU+CUDA split
execution works with ZERO ggml internal changes and NO threads** — the CPU
backend's graph_compute blocks (its "async" path is synchronous, synchronize is
a no-op), while CUDA's is genuinely async; so issuing the CUDA split FIRST and
then running the CPU split on the same thread yields ~100% overlap efficiency
(13.91 ms vs serial 21.72, ideal 13.84). The deferral-shaped topology (layer-N
CPU experts joining the residual one step late while layer-N+1 GPU attention
runs) verified BIT-IDENTICAL to serial, 0/262144 floats differ, with CUDA
graphs active. Per-layer granularity survives: async-only overhead ≤0.023
ms/split-pair at 32 pairs/token (worker-thread designs collapse — up to 2.7×
slower than serial at fine grain; do not use threads). Integration rec: custom
wrapper in llama-context/llama-graph (~400-700 LoC, 0 ggml internals), NOT
in-sched surgery (ggml-alloc buffer aliasing across splits is the landmine —
designed around, not tested; deferred tensors need their own buffers/liveness).
Toy source preserved: q27b_on_12gb/exp3.2_sched_overlap/. Remaining gate for
the build: 3.1 overlap ceiling on the 3060 (queued on the box campaign).
OPS NOTE: us-east-1 GPU quota is now ZERO for G-family (spot exhausted too);
only ap-northeast-1 has quota (8 vCPU G on-demand). Quota increase filed,
pending — standing blocker for future GPU-on-AWS work.

## 2026-09-05 — CPU Wave 1 + 3.1 campaign (3060, full report: results/cpu_wave1/RESULTS.md)

Noise floor 0.1% (n=3 argv-identical cells). B1 reproduced 48.815 (+0.6% vs cert).
B2 nominal (stock llama-server, -ngl 30, no tuning) = 22.17 t/s.

1. **NEW BEST SPEED-TIER CONFIG — 55.86 t/s code-gen (+14.4% vs B1, 2.52× stock)**
   from `-t 8` alone. Box host is a 6c/12t Ryzen 5 3600 exposed as 10 vCPUs; the
   default -t 10 oversubscribes. Sweep peak t=8 (55.86), t=6 55.06, t=10 48.50,
   t=12 41.10. Output BIT-IDENTICAL across t=3..12 (one SHA1, 22 cells).
   MECHANISM SURPRISE: slope of placement law is thread-INVARIANT (0.5360 vs
   0.5353 ms/layer at t=8/t=10); the win is ~1.0-1.4 ms of fixed per-forward-pass
   overhead, not faster expert compute.
2. **n-gram+MTP stack CONFIRMED and large on editing workloads**: --spec-type
   ngram-simple,draft-mtp + size_m 256 → +197% mean on 3 editing tasks (63→188
   t/s, peak 243.6), ZERO effect on pure generation (acceptance unchanged
   0.7732 — complementary, not additive), zero VRAM. Build default m=48 leaves
   most of the win on the table; m=8 inert; gains still rising at m=256.
   m=256 is byte-identical to the MTP-only control on all 3 editing tasks.
3. **Recommended config** (B1 + `-t 8 --spec-type ngram-simple,draft-mtp
   --spec-ngram-simple-size-m 256`): 55.65-55.86 code-gen / ~188 editing;
   VRAM peak 11832 MiB (456 MiB headroom → -ctxcp 0 guard applies).
   Quality: HE-50 96.0% vs 98.0% prior same-50 (McNemar p=1.0,
   indistinguishable; tripwire clear 181 tok/task). Full 164 battery required
   before shipping. **≥55 t/s speed goal: MET on the speed tier.**
4. **S1.4 bandwidth split**: STREAM Triad 25.7-26.4 GB/s; active expert bytes
   10.38 MiB/layer (8/256 top-k — NOT the full 332 MiB) → CPU expert decode is
   **77-79% bandwidth-bound, 21-23% instruction**. Kernel-level offensives
   (iqk vendoring etc.) capped at ~0.11 ms/layer → mostly dead. Repack A/B: no
   delta (expected — IQ3_XXS/IQ4_XS experts have no repack path either way).
5. **S1.2 re-opt**: n=2/ncm16 re-confirmed optimal at t=8 (n1 53.10, n2 55.83,
   n3 48.59, n4 47.23; ncm18 50.85). Acceptance IS n- and ncmoe-dependent
   (0.839/0.773/0.582/0.545) — DraftExpert effect real but the optimum didn't move.
6. **3.1 OVERLAP CEILING — the deferral economics changed**: nsys node-level
   trace (first pass invalid — cuda-graph-trace=graph hides replayed kernels;
   corrected run 1.6M events): GPU busy 11.29 ms/token of 21.73 wall → **GPU
   idle ~48% of decode**. g = 0.20 ms/layer GPU work vs c = 0.536 CPU expert →
   depth-1 deferral hides only g/c ≈ 37%: ceilings 54.0 nospec / 60.1 best-MTP /
   52.0 B1 (+7%-ish). BUT total CPU 10.44 < total GPU 11.29, so full overlap is
   feasible at deferral depth d ≥ c/g = 2.68 → **depth-3: ~76 t/s nospec (+65%),
   theoretical max 88.6**. Verdict: do NOT build depth-1; cost depth-3 first
   (3 deferred-buffer sets, 3-deep residual dependency — bigger than the 400-700
   LoC wrapper). Caveat: ceilings assume the dependency graph offers the
   independent work — 3.2 proved ggml CAN overlap, not that this graph does.
   Structural finds: 40 active blocks (blk.40 = MTP head, inert under nospec);
   attention is hybrid gated-delta-net (~30/token) + full FA (~6/token).
7. **Wave 2 (placement-aware quant) NOT RUN — two blockers**: (a) space gate:
   19.12 GiB free < 23 needed (94 GiB /data/swapfile is the elephant; Q4_K_XL
   21.3 GiB source sits beside it — also the better source for a shippable
   artifact); (b) MY RECIPE WAS WRONG, caught by dry-run: --tensor-type with a
   positional ftype requantizes EVERYTHING (incl. degrading the q8 MTP head to
   q3_K), and COPY base silently ignores --tensor-type. Correct recipe needs
   --tensor-type-file pinning all groups. Nothing deleted; awaiting human call.
8. **Losslessness audit** (SHA1 of outputs per cell): threads numerically inert;
   --n-cpu-moe value changes bytes (kernel rounding); MTP n≥3 changes bytes;
   ngram size_m byte-stable on 2/3 tasks and m=256 matches control on 3/3.

## 2026-09-05 — CERT: recommended config SHIPS (full battery, results/cpu_wave1/RESULTS.md §CERT)

Config: B1 + -t 8 + --spec-type ngram-simple,draft-mtp + size_m 256.
HE-164 **153/164 = 93.29%** (vs B1 152/164, McNemar p=1.0); edit-compliance
**34/34 = 100%** (vs B1 33/34, p=1.0); 198 paired tasks total, 2 wins each way,
smallest p=0.25 → no detectable quality change either direction. Tripwires clear
(265 tok/task). Peak VRAM identical (11832 MiB, no OOM across ~50 launches).
Cert itself ran 1.62× faster wall (832s vs 1351s) — the speedup pays for its own
verification. Speed tier is now **55.9 t/s decode / ~188 t/s editing** on the
3060, 2.52× stock, above the ~52 t/s Opus-reference bar.

Corrections from the cert (lead's audit):
- **-ctxcp 0 guard nuance**: peak VRAM measured IDENTICAL with/without -ctxcp 0
  on this config (11832), no request-#2 OOM either way; -ctxcp 0 costs 2.7% at
  short prompts but GAINS 6.7% at 13.5k context. Certified both ways. revv keeps
  its <500MiB-headroom guard (long-context favors it; coding agents run long).
- **Provenance fix**: the 94.1%-vs-67.6% edit-compliance pair was the 27B on
  L40S, NOT an on-box 35B baseline; no prior edit-compliance existed for any
  Q3_K_XL config — B1's arm was measured fresh (33/34), not cited.
- Soft cell flagged honestly: edit pass_2 under -ctxcp 0 was 0W/3L vs B1
  (p=0.25, n=34) — not significant, watch on future runs.

## 2026-09-05 — Expert Deferral: KILLED AT ALL DEPTHS (quality smoke, full writeup: deferral_smoke/RESULTS.md)

The staleness simulation (graph-only, serial, d-layers-late expert residual
joins on the 16 CPU-resident blocks) delivered a decisive RED before any
scheduler was built. HumanEval-164: d=0 154/164 (reproduces historical baseline
EXACTLY — rig validated); d=1 153 (p=1.0); d=2 151 (p=0.61); d=3 139
(p=0.0015). But **aider-polyglot editing (34 multi-file tasks) collapses at
d=1**: first-attempt 13/34 → 3/34, ten regressions ZERO improvements, McNemar
p=0.00195 (d=2: 1/34; d=3: 1/34). Zero improvements = systematic capability
loss, not churn. d=1 is the minimum deferral that buys any overlap ⇒ **no
surviving configuration. Wave-4.1 deferral is dead as specified.** Total kill
cost ≈ $5 + a day, vs a 1-2 week build.

Instrument findings (as important as the verdict):
1. **HumanEval would have GREEN-LIT d=1** (p=1.0, balanced churn). Short
   self-contained ~265-token tasks structurally cannot see damage that
   compounds over ~1100-token multi-file generations. Benchmark shape, not
   benchmark quality, was the blindness. Standing rule: NO config ships on
   HE alone; the editing instrument is mandatory for any graph/numeric change.
2. **edit_compliance_1 is a FORMAT check, not correctness** — it stayed 34/34
   → 32/34 (p=0.5) at every depth while actual solve rate collapsed 77%. The
   2026-09-05 cert's edit-compliance leg measured well-formedness; its
   conclusion stands only because the cert config also matched on HE AND is
   byte-identical-classed, but future certs must use polyglot solve rate.
3. Perplexity mis-ordered the risk in the other direction (+14.5% at the
   HE-clean d=1) — ordering signal, never a verdict (reconfirmed).

Physics conclusion: the CPU expert latency on the speed tier cannot be hidden
by residual-stream staleness at any quality-safe depth. The overlap lane now
has exactly one open door: schemes that overlap WITHOUT staleness (intra-layer
expert splitting, llama.cpp discussion #20528 — exact arithmetic, GGUF format
work required). 55.9 t/s stands as the config-lane ceiling on this hardware.

## 2026-09-05 — Research sweep #2 (lossless speed): see 19-LOSSLESS-SPEED-SWEEP.md

Headlines: (1) our MoE speculation numbers were measured under a kernel
restriction — MoE fusion hard-disabled for verify batches until PR #27621
(merged hours after our pin); the "1.23× MTP ceiling" and n=2 optimum need
re-measurement on v0.4.0 (expect +1-4% honest, fusions NOT bit-exact → full
battery gate). (2) Block drafters (dspark/dflash) beat MTP at n=2 in an
audited 3090 study on our exact model → predicted 60-64 t/s; artifacts on HF;
afternoon A/B. (3) Dynamic expert cache measured 1.91× on a 12GB 4070Ti with
a near-identical MoE (FATE fork) — distinct from our killed STATIC cache;
draft-driven expert prefetch is an unclaimed lossless research gap. (4) Model
space: nothing beats incumbents; NVFP4 settled negative incl. 5080; engram =
qwen4exp confirmed (license+RAM still kill it); REAP dead at 12GB (independent
lit: ≥3-bit quant beats pruning at equal bytes — confirms no-arbitrage law).
(5) PRODUCT-CLAIM CORRECTION: "27B SWE ~86" was a mis-mapped Max number (card:
SWE-Pro 61.7, unreplicated) and "35B meaningfully dumber" was a
benchmark-namespace artifact (Verified≠Pro) — our instruments measured a TIE;
update any copy that claims otherwise. (6) Release blocker: upstream #26425
reports MTP inter-request state corruption against our exact speed-tier GGUF —
reproduce-or-refute before next revv release.

## 2026-09-05 — Expert-cache lane: NO-GO from trace simulation ($0 smoke)

LRU replay of our only sequential routing trace (Flash-Next proxy, 1.6M rows;
fraction-matched to 256-expert/top-8; caveat: no trace exists for 35B-A3B
itself — capturing one is ~1h box time if ever needed). The killer is dispatch
granularity: llama.cpp runs a layer on GPU only if ALL top-8 experts are
resident (all-or-nothing), and the JOINT hit rate at practical cache sizes is
tiny (12.5% of pool cached → 12% joint hits → 1.12×; 25% → 1.38×). The
published 1.91× needs ~50% of the pool cached — at which point it's just
static placement, already shipped. PCIe-fetch-on-miss loses to our measured
compute-on-CPU 0.536 ms (Gen4 fetch 0.57ms, Gen3 1.2ms). Sensitivity with
g=0.20 (our nsys figure) makes it MORE negative. T3 fork test cancelled.
THE REDIRECT: per-ACCESS hit rate clears 1.91× at only 12.5% cache — but that
requires computing some of a layer's experts on GPU and some on CPU
simultaneously = intra-layer expert splitting (llama.cpp #20528). Two
independent analyses (this sim + the overlap research) now converge on
intra-layer split as the one remaining structural speed door. Sim + data:
local_inference_program/moe_tiering/cache_sim/.

## 2026-09-05 — DECISION: flagship (Qwen3.8-27B) decode-speed work FROZEN

Every lever measured to a wall (ledger review): raw 20.3 (floor ~23), MTP n=2
34-36 (the only big lever; n≥3 loses), p-min +1-3%, ngram dead for novel gen,
DFlash2 crashes on 12GB, custom drafter killed ($68, architecture-bound 5×
short), state-rewind retracted, kernel +10.1% raw → +2.5% effective (+21%
raw headroom ≈ +5%), config levers exhausted. No 55 t/s road exists for a
dense 27B on a 3060. The 27B stays the QUALITY flagship; its remaining levers
are non-decode: context (TurboQuant-class KV), work-speed features (session
restore 18×, thinking fix), and the two upstream kernel PRs. Speed research
moves to the MoE tier + model layer. Two sweeps commissioned: (1) GDN
verify-widening (the 8.01 ms/drafted-token ALU-bound tax — the one measured
wall never attacked; transfers to every Qwen3.x hybrid incl. the speed tier);
(2) TurboQuant KV for our use case (second look after the 09-04 Turbo4 eval).

## 2026-09-05 — GDN verify-widening sweep: GDN exonerated, MMVQ small-batch routing is the real tax

ATTRIBUTION CORRECTED: the 8.01 ms/drafted-token verify-widening cost is NOT
GDN-ALU-bound. Source reading + our own batch_probe: verify-2 vs verify-1 costs
1.108× (vLLM's true GDN snapshot pathology measures 1.79×; we don't have it).
GDN's share of the 8.01 ms is bounded ~9-11% (0.30 ms ALU + 0.45 ms per-token
state-snapshot writes) → a GDN kernel rewrite buys +1.4%. Independent
confirmation: upstream recurrent-kernel rewrites (#22675 Mamba-2, #22587 GDN)
shipped "TG unchanged / within noise". GDN lane DEAD as posed. (FLA has no
short-T kernel either; vLLM's kMaxMtpTokens=8 kernel still writes a full
state per token; at batch-1 weight traffic outweighs state traffic 68:1.)
Side finding: the ~150 MiB VRAM step per draft depth (n=4/5 OOM wall) IS the
per-position GDN snapshot planes (48 × 3.146 MB = 144 MiB, matches measured
152-156). ReplaySSM-style checkpoint+ring buffer would free it — only worth
it AFTER the widening fix (n≥4 pays only once widening <2 ms).
THE LIVE LEVER: `ggml_cuda_should_use_mmvq` (ggml-cuda/mmvq.cu:289) has no
Ampere branch and no IQ-type tuning → our 3-token verify batch routes to MMVQ,
which re-executes the sub-4-bit codebook decode PER COLUMN (mmvq.cu:665-678).
Same disease as the kernel offensive, in the batch>1 path. Fix ≈10 LoC
(route small IQ batches to MMQ / sm_86 cutoff), lossless (exact arithmetic,
not bit-identical → battery). Predicted **+12-19% on the flagship if it
holds** (→ ~40-43 t/s); also applies to the speed tier's verify batches.
CONVERGENT with upstream issue #28090 (open, asks for exactly an sm_86 MMVQ
cutoff entry, ~9% on Q4_0/A10). Settling runs: nsys diff at -p 3 vs -p 1;
batch_probe slope with the new routing. Flagship freeze holds for
drafting/architecture work; this kernel-routing item is exempted as a
bounded, cheap, convergent lever.

## 2026-09-05 — TurboQuant KV second look: DEAD as a lane; flagship context is cheap WITHOUT it

Extraction is feasible (TheTom fork, ~2,700 LoC incl. 44 hunks on the shared
fattn-mma-f16.cuh; zero conflict with our two patches — vecdotq.cuh/mmvq.cu
turbo lines = 0) but the prize is tiny: turbo4 25.4 MiB/1K ctx vs q4_0 27.0
→ ~50.7K vs ~47.8K reachable — **6% more context than a KV type we can
already use**. The fused (no-f16-scratch) kernel exists and measured −1.73%
vs f16. 2,700 LoC of hot-kernel surgery for +3K context = bad trade. DEAD.
New facts: (1) the buun fork silently alters f16/q8_0 attention numerics
(unguarded sparse_v_threshold compare in fattn-vec.cuh:380/428 compiled into
every type instance) — leading suspect for the −1.70% fork tax AND it weakens
our turbo4_oc_eval §6 cross-check cell (compared against non-upstream f16).
(2) Fork VRAM tax suspects: CUDA-graph exec cache re-keyed by shape (64
entries) + ~406 extra kernel TUs of __constant__ tables; the RSS explanation
is ruled out (our VRAM numbers are nvidia-smi device peaks).
THE LIVE ITEM: on the hybrid 27B only 16/64 layers hold KV, so KV/token is
small; the 16K certified limit was likely set by the context-checkpoint
lazy allocation (32×~150MiB) rather than KV bytes. Prediction to test at $0,
no rebuild: `-fitp on` with -c omitted (+ -ctxcp 0) → q8_0 fits ~28-30K,
q4_0 ~46-50K; confirm at -c 24576 with decode t/s at depth (q4_0's
compute-bound attention kernel is the speed cost to measure). Context is
capability, not speed — still the flagship's most valuable open lever.

## 2026-09-05 — W3a bake-off + context-fit (box; results/w3a_bakeoff/RESULTS.md, results/ctx_fit/RESULTS.md)

Tonight's B1 = 55.16 t/s (3 sessions, acc 0.7732; within 1% of cert). All
screening; harness gained a host-CPU quiet gate + interleaved sessions after
an Explore subagent's recursive grep contaminated one run by 9% (retained).
1. **v12 / PR #27621 fusion: −4.6% at n=2, slower at every n. KILL, with
   mechanism**: the non-bit-exact fusion perturbs the MTP head's logits →
   acceptance 0.773→0.704; it gives back more in re-drafted tokens than it
   saves in kernel time. Lesson: "close enough" kernels are NOT free under
   speculation — acceptance is the canary. Rebase verdict: don't.
2. **Block drafters on the speed tier: DSpark −8.3%, DFlash −9.3% vs built-in
   MTP. DEAD.** (The 60-64 prediction from the 3090 study did not transfer —
   12GB forces ncm19 to fit the drafter, and the drafter must beat MTP AND
   repay displaced layers.) TRAP sharper than #25345: the DSpark GGUF declares
   general.architecture=dflash; via the dflash loader it loads, reports 4,722
   healthy-looking drafted tokens, and runs at **0.0051 acceptance** (−62%).
   "draft_n>0" is NOT sufficient — acceptance is the required canary. → harness.
3. **DFlash2 on the flagship: v12 fixed the segfault; runs at −8.1% and
   +844 MiB. DEAD** (confirms freeze).
4. **p-min × n-max flagship grid: monotone negative** (c=8192; n=4 OOMs at
   16K on request #1 as recorded). Brief corrections by the lead: flagship uses
   the repo chat template not --jinja; -t is irrelevant (fully GPU-resident);
   the flagship's MTP is the BUILT-IN head — the external MTP file has never
   loaded on this box.
5. **n-gram chain on FLAGSHIP editing: 40.25 → 222.92 t/s (+454%)** on one
   editing task (LF, file-in-prompt). PENDING CERT across all 3 EDIT_TASKS +
   output identity (SHA1 vs MTP-only control) before any claim/ship.
6. **#26425 (MTP inter-request state) NOT REPRODUCED**: 3 legs × 30
   same-prompt generations byte-identical; original re-run after 10 different
   prompts identical. Release blocker cleared at this depth.
7. **Context-fit (flagship, -ctxcp 0)**: q8_0 → 20,480 fits but peaks 12,028
   /12,044 MiB (16 MiB headroom — NOT shippable); **q4_0 → 24,576 at 35.5 t/s
   at depth, 224 MiB slack — shippable +50% context** pending KV-quality check
   (HE + polyglot at depth). My 28-30K/46-50K prediction MISSED: checkpoints
   were a limiter, not the only one.
Disk used 2.0GB. MMVQ probe: staged but stopped by permission (patch targets
the v11 source tree under revv-home) → re-routed to a scratch copy.

## 2026-09-05 — Flagship n-gram editing chain CERTIFIED; MMVQ→MMQ routing REGRESSES

**Flagship (27B IQ3_XXS) + `--spec-type ngram-simple,draft-mtp --spec-ngram-simple-size-m 256`**
(results/flagship_ngram_cert/RESULTS.md): editing workloads 40.3 → **222.8 /
246.0 / 113.3 t/s (5.53× / 6.10× / 2.81×)**; code-gen control 35.17 → 35.16
(1.00×, inert where it doesn't apply). 48 generations, 8 cells, ONE SHA1 per
cell, chain byte-IDENTICAL to MTP-only control on 4/4 workloads (same as the
35B cert). Insight: acceptance is the WRONG scalar for block drafters — chain
acceptance 0.48-0.93 vs MTP's ~1.0 yet 2.8-6.1× faster (long verbatim runs
land more accepted tokens per verify; 0.4832 acceptance still buys 2.81×).
SHIP-POINT CAVEAT: chain costs ~100 MiB → 88 MiB headroom at c=16384 (thin by
the standard that disqualified q8_0@20480); c=8192 = −0.3% with ~430 MiB.
Pending: confirm cell WITH -ctxcp 0 (revv's guard already emits it for this
config) — expected to restore headroom; then stage into revv.

**MMVQ→MMQ small-batch routing: REGRESSION — do not ship, do not PR**
(results/mmvq_routing/RESULTS.md): flagship 35.33 → 29.11 (**−17.6%**), speed
tier −5.6%, acceptance canary moved 0.0187 (3.7× tolerance). Mechanism
measured: the patch DOES remove the per-column decode (marginal 7.81 → 1.52
ms/token) but MMQ pays a ~73 ms fixed entry cost at small ne11 — crossover at
N=6, we run ne11=3. Falsification arm MIN_BATCH=6 reproduces OFF exactly
(35.38, identical acceptance to 16 digits, identical SHA1) → regression pinned
to ne11=2-5 routing. METHOD LESSON: the runbook's linear-slope threshold would
have PASSED this (w_on=3.98, r²=0.561 — a step + shallow slope, not a line);
the lead's pre-registered operating-point prediction (28.85) matched measured
(29.11). Judge at the operating point, never by a fitted slope.
WHAT SURVIVES: the lever is now SIZED — MMVQ's per-column codebook decode
costs 7.81 ms/drafted token; a small-tile MMQ or a batched MMVQ that decodes
the codebook once across columns would capture ~6.3 ms/token ≈ **+5.5 t/s on
the flagship**. Kernel-engineering backlog item (same family as the SWAR PR),
not a routing flag. Process: llama-bench swallows GGML_LOG_INFO without -v —
every gate-liveness check now runs with -v.

## 2026-09-05 — Flagship n-gram chain SHIP POINT: c=12288 (results/flagship_ngram_cert/)

Headroom cells (chain on, 3 consecutive 81%-filled requests): c=16384 →
11,956 MiB (332 headroom ✗); **c=12288 → 11,822 (466 ✓)**; c=8192 → 11,666
(622). Throughput FLAT in context (222.62/222.78/222.94 on the same task,
0.14% spread) → the ship point is chosen purely on headroom, no speed cost.
PREMISE CORRECTED: -ctxcp 0 frees NOTHING here (11,956 with vs 11,954-956
without) — checkpoints weren't holding that memory; same under-delivery as the
ctx-fit probe. The chain drafts a mean 74.85 tokens/step → ne11≈76 verify
batches (~25× MTP-only). Extra cell: MMQ at ne11≈76 (12× past its N=6
crossover) STILL loses −2.76% (numerically harmless) → **MMQ closed in all
three regimes** (−17.6/−5.6/−2.8%); the batch-probe crossover does not
transfer to the main loop. Quality gate: outputs byte-identical to MTP-only
on 3/3 editing + generation control → HE/edit scores CANNOT move; no battery
needed. Shipped flagship argv: -c 12288 -ctxcp 0 + repo chat template +
--spec-type ngram-simple,draft-mtp --spec-draft-n-max 2
--spec-ngram-simple-size-m 256. Flagship: ~35-38 t/s generation, 113-246
t/s editing (2.8-6.1×), 1.00× and free where the chain doesn't apply.

## 2026-09-05 — WAVE 2: placement-aware quantization CONFIRMED (results/wave2_placement_quant/RESULTS.md)

Retyping ONLY the 16 CPU-resident blocks' expert tensors (48 tensors; GPU
tensors + MTP head byte-identical types):
| arm | CPU expert type | t/s | Δ | RSS | load |
| A stock | iq3_xxs/iq4_xs | 55.84 | — | 7.4G | 5.0s |
| B | q4_K | **59.68** | **+3.84 (+6.9%)** | 9.0G | 9.3s |
| C (falsification) | q6_K | 53.03 | **−2.82** | 12.1G | 16.1s |
| D | iq4_nl | **60.10** | +4.26 | 9.0G | 9.8s |
Pre-registered prediction +2 t/s → measured +3.8-4.3 (~2×). **Arm C is the
proof**: MORE bits, NO x86 repack path → 5% SLOWER than stock — speed moves
opposite to bit-width, so the mechanism is the kernel table, not bandwidth or
precision. (D also overturns the DDR4 field report: iq4_nl has a repack path
and won.) The +6.9% arrived DESPITE acceptance falling 0.773→0.642 (weights
moved), so the raw CPU-layer gain is larger than the headline. NOVEL RESULT:
"choose a tensor's quant by which chip executes it" — publishable.

**DO NOT SHIP THESE ARTIFACTS**: HE-50 B 45/50, D 44/50 vs baseline 48/50
(p=0.375/0.125, n.s.) but 8/9 discordant tasks favor baseline and the
acceptance shift confirms real weight movement — cause is requant-from-quant
(blocks 0-15 quantized twice, no imatrix). Acceptance canary was the wrong
instrument here by design (weights change ⇒ acceptance must move; it moved
monotonically in fidelity stock>q6_K>iq4_nl>q4_K — consistency check passed).

**Shippable recipe (commissioned)**: INVERT — start from unsloth Q4_K_XL
(blocks 0-15 experts are ALREADY gate/up q4_K, down q5_K, imatrix-quantized
from source) and retype only GPU blocks 16-39 DOWN to the Q3_K_XL mix. CPU
layers never requantized; same VRAM profile. Caveats: q5_K has NO x86 repack
(NEON only) → down_exps may miss a third of the win — test arm with down→q4_K;
gate on FULL HE-164 + editing instrument (HE-50 cannot resolve weight changes).

Process/upstream finds: (1) ssm_alpha/ssm_beta are 2-D F32 tensors that the
naive recipe silently quantized (60 tensors) — caught by dry-run; pin ALL 753.
(2) llama-quantize BUG: pinning an IQ tensor to its own existing type trips
the imatrix pre-flight even though `cur_type == new_type` tensors are copied
verbatim and never quantized — blocks selective requant on any model with IQ
tensors; one-line fix applied in scratch; → public bug-report queue (human-owned).

## 2026-09-05 — Wave 2 SHIPPABLE ARTIFACT PASSES; flagship polyglot INVERTS the tiers (results/wave2_shippable/, results/flagship_polyglot/)

**Arm E — "merged" 35B artifact: 62.08 t/s (+10.7% vs stock 56.06), HE-164
151/164 (p=0.69 vs 153), aider-polyglot 9/34 first-attempt / 16/34 overall
(p=1.000 vs the chain baseline) → PASSES BOTH GATES.** Built by TENSOR MERGE
with ZERO requantization: blocks 0-15 experts taken natively from unsloth
Q4_K_XL (gate/up q4_K, down q5_K — imatrix-quantized from source), everything
else from Q3_K_XL; output.weight pinned q6_K (Q4_K_XL's q8_0 would add 117 MiB
and break the VRAM gate). The planned "inverted requant" was UNEXECUTABLE
(converting 46 tensors TO IQ3_XXS genuinely needs an imatrix; none exists for
the 35B) — the merge is strictly better than the recipe it replaced: the only
artifact in the program with no requantized tensor. Arm F (E + down_exps
q5_K→q4_K requant) = 63.76 t/s but weaker on every quality axis (149/164,
p=0.29; lost one edit-compliance) → NOT shipped for +1.68 t/s.
KEY INSIGHT: fidelity bought SPEED, not just quality — F and Wave-2 arm B have
identical CPU-layer types, yet F is 4.1 t/s faster because B's acceptance
collapsed to 0.642 while F holds 0.726: requantization noise costs forward
passes via the MTP head. Both axes move together.
VRAM: E peaks 11,912 MiB → 130 MiB usable headroom (12,042-12,044 ceiling) —
BELOW the ≥200 standard, though it survived the full HE-164 + 34 multi-file
edits (hundreds of requests) at c=16384 without OOM. revv's planner (measured
anchor + 150 margin) will serve E at c=12288 automatically; ship decision:
E@16384 with 130 MiB (battery-evidenced) vs E@12288 (planner default). Hosting:
E must be published (HF, under Mericanii) — revv's first ARTIFACT, not config.
COMPARATOR CORRECTION: the 13/34 ÷ 18/34 polyglot figure is the NO-SPECULATION
arm; the chain config's own baseline is 9/34 ÷ 16/34 (cert data) — scoring E
against 13/18 would show a phantom regression. Open question (n=34, p n.s.):
whether speculation itself costs editing solve rate on the 35B (nospec 13 vs
MTP-only 11 vs chain 9 first-attempt) — differences are within McNemar noise
at n=34; needs a larger task set before any claim.

**FLAGSHIP (27B) ON OUR EDITING INSTRUMENT: 4/34 first-attempt, 8/34 overall**
vs the 35B's 9/34 ÷ 16/34 at the same (chain) config — **p=0.039** (vs the
35B's best arm p=0.0064). The 27B wins only 1-3 tasks the 35B misses: a broad
capability gap, not a trade of strengths. Edit-format compliance is PERFECT
(34/34) — it produces well-formed but WRONG edits (the format instrument is
blind to this, as found earlier). **The "speed tier" is both faster AND ~2×
more capable on multi-file coding edits than the "flagship."** Naming/product
implication: the 35B-A3B should be revv's default coding model; the 27B's
remaining case (newer generation, dense, vendor GPQA/reasoning claims) is
UNMEASURED by us. Follow-up queued: 27B WITHOUT the chain (13 min) to separate
model capability from any speculation effect.
Box: GPU idle, 233 GB free, v11 + revv-home untouched. Seven RESULTS.md files.

## 2026-09-06/07 — revv QA, de-bloat, release packaged and published

- QA (2026-09-05/06): end-to-end Phase A+B on the box, 25+ checks; six defects fixed (`revv get speed` never worked, `up moe` rejected the new names, status computed free VRAM from nominal, compare STOCK OOM on MoE, bench could not fail on MoE, impossible test fixture at 12,287 free). Report: results/revv_qa_20260905.md.
- De-bloat (09-06): −160 shipped lines, three duplicated literals named, duplicated peak arithmetic folded; behaviour-snapshot harnesses (argv, CLI text, install.sh) plus an AST-equivalence check gate every refactor (tests/verify_refactor.sh). Eight follow-up items from the cleanup lead remain open (constants confirmed dead, `"{:,}".format` repeated 17×); none affect behaviour.
- Docs (09-06): README plain rewrite with neutral build names (moe/dense; speed/flagship deprecated after the editing instrument inverted them), EXPERIMENTS.md (what failed and why), BENCHMARKS.md §17b/§18 and the editing-instrument section. History rewritten to the founder's identity only (zero co-author trailers), force-pushed.
- Release (09-06 packaged, 09-08 published): multi-arch prebuilt v1.1.0-binaries (sm_75/80/86/89/90, bundled CUDA 12.0 runtime, launcher script, 723,421,067 bytes, sha256 c4af4f1f…). install.sh pins URL and hash. Published by the founder with `gh release create`; URL and hash verified against install.sh after publication.
- Paper (09-06): draft unpaused at local_inference_program/paper/cheaper-intelligence.md (~7,300 words; Fable 5 model per founder; no AI credited). Structure survey against comparable papers started 09-09.
- Isolation contract (09-08): install.sh no longer reuses a PATH llama-server (opt-in `--system-llama-server`); README states what revv touches (two directories) and `revv uninstall`.

## 2026-09-08/09 — Second machine: WSL2 3060, both builds, and what it changed

Machine: Windows 10 desktop, WSL2 Ubuntu 26.04 (glibc 2.43), driver 610.74, RTX 3060 12GB driving the display, Ryzen 5 3600, 32 GB RAM (WSL2 default cap showed 15 GB). First install of the v1.1.0 prebuilt on a machine that did not build it. Raw record: revv/TEST_WSL2_RESULTS.md; tables: revv/BENCHMARKS.md §19.

Results, unforced, desktop cleared: dense 8,192 ctx q8_0, bench 36.25 t/s (within 5% of 37.9), compare 21.9 → 34.6 t/s, 3.74× to done. MoE (after WSL RAM raised to 24 GB) 12,288 ctx q8_0, bench 47.2 t/s (84% of 55.9), compare 18.7 → 41.9 t/s, 4.49× to done. Dense within noise of the box because it is fully on the GPU; MoE loses ~16% on the CPU-side path (-t 6 vs -t 8, Hyper-V between process and RAM); the stock arm dropped by the same fraction, so it is the machine, not the config.

Defects found and fixed (revv main 3f9542d, 4d126fc, 73d1d7f):
1. Free-VRAM floor was a generic constant (11,528) and refused a card with 11,516 free that then ran at 4096 with 276 MiB spare. Measured whole-process rungs added (dense 4096/8192 = 11,307; MoE 4096/8192/12288 = 11,407/11,487/11,521; q8_0 KV, chain on, -ctxcp 0, two context-filling requests); floor now 11,457 derived from them.
2. Anchor arithmetic over-charged the MoE small rungs by 200–400 MiB (hybrid attention, tiny KV term) and planned 4096/q4_0 where 12,288/q8_0 fits; with measured rungs in play the estimator could price f16@6144 under measured q8_0@4096, so a monotonic guard now forbids any estimate undercutting a measured smaller configuration.
3. install.sh: download helper clobbered the caller's `dest` (archive left under its tmp name); ldd check ran without the bundled lib path (false "install cuda-cudart" warning).
4. `revv up` failure tail showed the previous session's log; doctor planned a generic file and disagreed with `up` (now plans each certified file).
5. WSL2 quirks now handled or documented: nvidia-smi only on login-shell PATH (fallback to /usr/lib/wsl/lib), half-of-RAM default (.wslconfig memory=), C: filling kills the VM with I/O errors (the vhdx grows on C:), closing the last window kills the VM (vmIdleTimeout=-1), Windows desktop share of the card moves 275–1,022 MiB with what is open.

Standing rule added: a planner floor must come from a measured configuration, never from an estimate plus margin; and a measured smaller configuration is a lower bound for every larger one.

Second-machine 16384 MoE rung measured 11,659 with 161 MiB spare, under the 200 MiB standard; the box's 11,832 certified figure stands for that rung.

## 2026-09-09 — Correction to the MMVQ→MMQ entry (2026-09-05, "~73 ms fixed entry cost")

The "~73 ms fixed entry cost at small ne11" was a per-forward-pass figure described as if it were a per-kernel cost. From results/mmvq_routing/b1_batch_probe.json: with routing ON, a whole forward costs a near-flat 77.7 / 77.7 / 78.6 / 80.2 / 82.2 / 86.5 ms at N = 2, 3, 4, 5, 6, 8 (fit 59.30 + 3.98·N), against OFF 55.9 / 62.9 / 71.5 / 76.4 / 86.4 / 103.0 (fit 39.57 + 7.81·N, r² 0.997). The tiled path is a flat ~78 ms per forward at small N, the per-column path is a line, and they cross at N≈6. The verdict (regression at ne11=3, do not ship) is unchanged; only the description of the mechanism is corrected. Paper v5 carries the corrected wording.

## 2026-09-09 — MoE 128K context profile (reader prompt) and the RAM-bandwidth question

A Reddit reader reports the 35B-A3B at Q4_K_S, 262K context, 55-60 t/s on a 3060 with 64 GB of fast DDR4 on bare metal. Consistent with our placement law, not a contradiction: context on this hybrid model is bought with expert blocks (463 MiB freed and 0.48 ms/token lost per block moved to the CPU on our box), and a faster memory bus pays less per block. Measured here: 128K at q8_0 KV needs 22 blocks on the CPU (20 OOMs at load); 46.8 t/s on the bench protocol (43.1 on a code prompt) at empty context, 17.0 t/s with 128,517 tokens in context, 456 t/s prefill, 432 MiB minimum free under two full-context requests. Shipped as `revv up moe --long`; default stays 16K/55.9. BENCHMARKS §20.

RAM bus: hypothesis that the box was single-channel is REJECTED — Proxmox `dmidecode` shows 2 × 32 GB DDR4-3200 on channels A and B. 25.7 GB/s STREAM Triad is normal for a Zen 2 3600 (write path half of read) in a VM. The reader's edge is faster memory on bare metal.

## 2026-09-12 — Local-model harness and abliterated builds

Founder's harnesses: OMP (oh-my-pi) kept for cloud models; local models get `~/mericanii/revv-chat` (editable stdlib chat) against a switcher endpoint on the box (192.168.1.81:8090, `/data/projects/revv-switch/`) that loads revv configs by model name. OMP's system prompt + tools ≈ 15K tokens, so only the 128K MoE profile works under OMP. Abliterated builds downloaded and registered: Huihui 27B UD-IQ3_XXS (runs under the dense planner, uncertified) and Huihui 35B-A3B plain Q3_K MTP (measured: 57.7 t/s short, 41 t/s at 15K, peak 11,434; registered as a compatible MoE build; BENCHMARKS §21). Quality of both unmeasured; for security research only.

## 2026-09-12 — KV cache format on the hybrid MoE inverts the dense finding

A/B on the abliterated Qwen3.6-35B-A3B (plain Q3_K, MTP), MoE flags, same 15.3K prompt: f16 KV 29.9/30.3 t/s vs q8_0 52.2/52.3 t/s at depth; 67.6 vs 67.4 at short context. The paper's "quantized KV is a capacity trade, slower than f16 at every depth" holds for the dense 27B (llama-bench tg64) and is WRONG for the hybrid MoE at depth. Planner fixed (revv main: MoE-line builds keep q8_0 even when f16 fits). Standing rule: KV-format conclusions are per architecture; re-measure on any new family. BENCHMARKS §21.

# Quality per Gigabyte: Measuring Qwen3.8-27B Under 4 Bits on a 12 GB Consumer GPU

**Mericanii — independent research lab**
Yugendren · version **v1.0-draft** · 2026-08-31
Corrections, questions, suggestions: **19thkingisreal@gmail.com**

> **STATUS: INTERNAL DRAFT — NOT FOR PUBLICATION.** Circulated for founder
> review. Repo and discussion links are placeholders.

---

## Abstract

How much coding ability do you lose squeezing a 27B model onto a $250 gaming
GPU? Nobody had published the answer, so we measured it.

These are, we believe, the first independent **end-to-end task-accuracy**
measurements of sub-4-bit GGUF builds of Qwen3.8-27B: one RTX 3060 12GB,
llama.cpp b10566, the full 164-task HumanEval. Independent quality-versus-size
signal already exists for this model in the form of KL-divergence and
perplexity — we are not first to measure degradation, only first (that we can
find) to measure it as executed, graded code (§1.1). Across four builds
spanning 6.76–10.17 GiB, pass@1 (problems solved on the first attempt) rises
**49.4% → 82.3%**, approximately **10 points per gigabyte across the measured
range, every adjacent step significant under paired testing** (§3). So the
rule for a 12 GB card is simple: run the largest build that still leaves room
for the context you need.

**The flagship configuration** — UD-IQ3_XXS with the model's built-in MTP
speculation head at n=2 and a q8_0 KV cache — delivers **82.3% HumanEval at
33.8 tok/s**, entirely in VRAM, in 11642 MiB of the card's 12288 MiB (§6).

We also report a pre-registered negative result: best-of-5 sampling at
temperature 0.8 **underperforms** a single greedy attempt here, oracle ceiling
75.0% against 82.3% (§5). Plus a correction to our own earlier headline claim,
which was measurement noise (§3.1), and five warnings for anyone benchmarking
quantized builds (§4.5).

One model, one card, one runtime, one benchmark, one seed; scoped in §7.

---

## 1. Motivation

The question a consumer GPU owner actually has is not *how much does
quantization hurt* in the abstract, but *how much ability do I get per gigabyte
of VRAM I own*. A 12 GB card fits roughly 10–11 GiB of weights alongside a
usable KV cache (the per-token memory attention keeps around), capping a 27B
dense model at about 3.2 bits per weight. Above that you are not resident, and
a single layer spilling to system RAM costs an order of magnitude: 21.75 tok/s
resident against 2.12 for a naive FFN-offload recipe at 15.4 GiB.

Almost nobody publishes in that regime. Quantization studies are dominated by
8-bit and 4-bit results on 70B and 405B models on datacenter hardware, where
compression looks close to free (*Give Me BF16 or Give Me Death?*, arXiv
2411.02355, >500k evaluations). Vendors publish benchmark tables without stating
precision at all: Qwen states none for any Qwen3.8-27B number, and the FP8 card
republishes the BF16 table verbatim. Community quantizers publish files, not
evaluations. So a 12 GB owner choosing between a 6.76 GiB and a 10.17 GiB build
of the same model has folklore and file sizes to go on.

We are a one-person lab and we own a 3060, so we measured the curve — harness
and exact flags published, and the results that went against us as prominent as
the ones that did not.

### 1.1 Prior art

We are not the first to measure quality loss in this size class, and we do not
want to read as claiming otherwise. What exists, and how it differs from this
paper:

- **`unsloth/Qwen3.8-27B-GGUF` HF discussion #49** — an independent
  KL-divergence benchmark of this exact model across quant sizes. Same model,
  different instrument: KLD measures distance from a reference output
  distribution, not whether generated code runs and passes.
- **turboderp's EXL3 KLD-vs-VRAM charts** — the same divergence methodology,
  applied across models, on the EXL3 trellis format rather than GGUF.
- **quanteval.ai's agentic GGUF ladder** — task-level scores for this model
  family (cited already in §7), on an aider-polyglot-style suite. Reports
  scores per build; to our knowledge does not report a deployable measured
  serving configuration (VRAM peak, decode speed, speculation) or pre-register
  decision gates before running.
- **Artefact2's canonical KLD gist** — the reference methodology most of the
  above build on; not specific to this model.
- **Unsloth's own "Divergence-300" finding** — a KLD/perplexity cliff between
  UD-Q2_K_XL (9.83 GB) and UD-IQ2_S (8.37 GB), read in the community as an
  "agentic breaks down below here" line. Also a distributional-distance
  signal, not an executed-task one.

**What is new here:** (a) end-to-end task accuracy — HumanEval pass@1 on code
that is actually run against hidden tests, not a proxy for how far a
distribution moved; (b) a deployable measured configuration — VRAM peak, decode
speed with speculation, and the exact serving flags, not a quant-quality number
in isolation; (c) decision gates for the best-of-n experiment (§5) written and
thresholded before any GPU time was spent. None of the sources above measure
all three. We read KLD/perplexity and executed-task accuracy as complementary,
not competing — KLD is cheap and catches divergence early; task accuracy is
what a user watching pass/fail actually experiences, and the two need not move
together.

---

## 2. Setup

### 2.1 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 3060 12GB (sm_86), driver 535.309.01, CUDA 12.0 |
| CPU | AMD Ryzen 5 3600 (6C/12T) |
| RAM | 47 GB DDR4 |
| OS | Ubuntu 24.04, **headless** (no display attached to the GPU) |
| Theoretical GPU bandwidth | 360 GB/s |

Headless matters: the flagship configuration peaks at 11642 MiB of 12288 MiB,
and a desktop session on the same card leaves no room for it.

### 2.2 Software and weights

- **Runtime:** llama.cpp build **10566** (commit `bb4caa754`), CUDA backend,
  built from source with `-DGGML_CUDA_FA_ALL_QUANTS=ON` and
  `CMAKE_CUDA_ARCHITECTURES=86`.
- **Weights:** `unsloth/Qwen3.8-27B-GGUF` (linked, not rehosted). Model
  released 2026-08-14; measurements ran 2026-08-23 to 2026-08-31.
- **Chat template:** `froggeric/Qwen-Fixed-Chat-Templates`, passed with
  `--chat-template-file`. The stock template in the GGUFs is broken for this
  family.
- **Comparison build:** `exnivo/Qwen3.8-20B-Minitron`, an independently
  produced structurally-pruned (64→44 layer), distillation-healed 20B at
  i1-IQ4_XS, 10.23 GiB — almost the same file size as our best quantization.

### 2.3 Correctness gate

Every measurement sits behind a 3-prompt coherence smoke test requiring 3/3
coherent output. Not ceremony: the Gated DeltaNet CUDA path in this family can
produce fast, well-formed *garbage* rather than crashing, and a speed benchmark
on a broken build looks excellent.

### 2.4 Evaluation protocol

- **Quality:** HumanEval, **all 164 tasks**, pass@1, temperature 0. Generation
  goes through `POST /v1/chat/completions` with the model's own chat template
  and a fixed system instruction ("Complete the following Python function.
  Respond with a single Python code block..."). Each completion is executed
  against the task's `check()` function in a subprocess, 10-second timeout.
- **Speed:** `llama-bench` pp512 / tg128, 3 repetitions, plus live-serving
  decode from `llama-server` timings. VRAM is the peak of a 500 ms
  `nvidia-smi` poll across the run.
- **GSM8K: dropped.** At n=50 it scored 0.94–1.00 on every build including the
  2.12 bpw one — at ceiling for this model class, discriminating nothing.

### 2.5 The missing reference point

Our standard is that every quality result carries three reference points: full
precision, the best existing quant, and ours. We cannot satisfy the first — a
BF16 27B misses a 3060 by more than 4×, and we do not rent GPUs. Our curve is
internally consistent (same harness, prompts, seed, day) but **not anchored to
an uncompressed reference**: read every absolute number here as "under this
harness", not as a leaderboard score.

That leg is missing from the public record generally: the one public GGUF ladder
for this model (quanteval.ai) has no BF16 anchor either, and the one study with
a BF16 anchor (WonderRico, SWE-bench Verified) has no GGUF rows.

---

## 3. The quality–size curve

All four builds fully resident, `-ngl 99`, HumanEval 164 tasks, pass@1,
temperature 0. "bpw" is bits per weight; the "MTP head" is a small built-in
extra layer that drafts several tokens at once for speculative decoding (§4.1).

| Build | bpw | File (GiB) | HumanEval (164) | Passed | 95% CI | Built-in MTP head? |
|---|---|---|---|---|---|---|
| UD-IQ2_XXS | 2.12 | 6.76 | **49.4%** | 81/164 | [.417, .570] | ❌ stripped |
| UD-IQ2_S | 2.45 | 7.79 | **62.2%** | 102/164 | [.548, .696] | ❌ stripped |
| UD-Q2_K_XL | 2.87 | 9.14 | **72.0%** | 118/164 | [.651, .788] | ✅ |
| UD-IQ3_XXS | 3.20 | 10.17 | **82.3%** | 135/164 | [.765, .882] | ✅ |

Steps between adjacent builds are **+12.8, +9.8, +10.3 points** per roughly
1.1 GiB. Because the same 164 tasks are scored on every build, adjacent builds
form matched pairs, so we also ran an exact McNemar test on the discordant
pairs (tasks that flip from fail→pass or pass→fail between the two builds)
for each step:

| Step | Discordant pairs (gain/loss) | McNemar exact p |
|---|---|---|
| IQ2_XXS → IQ2_S | +32 / −11 | **p = 0.0019** |
| IQ2_S → Q2_K_XL | +29 / −13 | **p = 0.0195** |
| Q2_K_XL → IQ3_XXS | +20 / −3 | **p = 0.0005** |

All three transitions are significant at p < 0.05. With only four points on
the curve we do not claim linearity, or the presence or absence of a knee —
what n=4 supports is narrower: **approximately 10 points per gigabyte across
the measured range, every adjacent step significant under paired testing.**
Two consequences:

1. **Weight bits dominate context.** A gigabyte spent on longer context is a
   gigabyte not spent on ten points of coding ability. Run the largest build
   that fits the context you need and quantize the KV cache instead (§4.3).
2. **Going small is penalized twice.** Builds below about 8.4 GiB in this family
   silently ship *without the MTP head*, costing a +68% lossless speedup on top
   of the quality (§4.1). Nothing in the filename or model card says so.

### 3.1 The noise correction (what we got wrong first)

Our first pass used n=20 HumanEval subsets because they are cheap. We measured
UD-IQ2_S at **0.70**, then, on an identical build with identical flags through
the identical harness, at **0.80**. At n=20 one task is 5 points, so a
±10-point swing is pure sampling noise.

That invalidated every fine distinction we had drawn, including our internal
headline at the time — "UD-Q2_K_XL 0.80 equals UD-Q4_K_S 0.80" — which was two
numbers in one noise band, not a demonstrated equality. We re-ran on the full
164 tasks; only full-set numbers appear above. The noise did not mis-rank the
builds, it *compressed* them.

**The generalizable warning: n=20 subsets carry ±10 points of noise and cannot
support any claim finer than that.** Many comparisons circulating in this
community rest on subsets that size, run once, and cannot distinguish the
builds they claim to.

### 3.2 Structural pruning loses to plain quantization

The pruned-and-healed 20B, at 10.23 GiB, is genuinely faster than the 9.14 GiB
quantization — 28.44 tok/s decode and 38.61 prefill against 21.09 and 25.63 —
and substantially worse: **HumanEval 0.50 against 0.80, GSM8K 0.90 against
0.96** on the matched n=20/n=50 protocol. Those are old n=20 numbers, ±10 points
each, but the gap is 30 and the ranking survives. Minitron was never re-run on
the full 164 tasks, so it is not comparable to the 72.0% above.

Somebody else already paid the pruning and distillation compute and gave the
result away, and it is *still* worse per byte than the original model run
through an off-the-shelf importance-matrix quantization. At this size class,
structural pruning is not a shortcut.

---

## 4. Speed

### 4.1 Speculation ablations

The dominant speed lever by a wide margin. Speculative decoding drafts several
tokens cheaply and verifies them in one pass; accepted drafts are free. All
rows: 16K context, q8_0 KV, flash attention on, `--parallel 1`. "MTP" is the
model's **built-in** `blk.64.nextn` head via `--spec-type draft-mtp`, *not* the
separate external MTP file.

| Build | Speculation config | VRAM (MiB) | Decode (tok/s) | Accept rate |
|---|---|---|---|---|
| UD-Q2_K_XL | none | 10212 | 21.09 | — |
| UD-Q2_K_XL | MTP n=2 | 10588 | 35.11 | 0.786 |
| UD-Q2_K_XL | MTP n=3 | 10740 | **35.39** | 0.707 |
| UD-Q2_K_XL | MTP n=4 | 10894 | 33.63 | 0.581 |
| UD-Q2_K_XL | ngram-mod ⚠ | 9762 | 86.12 | 0.754 |
| UD-IQ3_XXS | none | 11266 | 20.25 | — |
| UD-IQ3_XXS | MTP n=2 | 11642 | **33.83** | 0.759 |
| UD-IQ3_XXS | MTP n=3 | 11794 | 33.28 | 0.651 |
| UD-IQ3_XXS | ngram-mod ⚠ | 10816 | 134.40 | 0.958 |

- **MTP is worth +68% and is lossless** — speculation does not change the
  output distribution. Optimum n=2–3; at n=4 acceptance falls to 0.581 and it
  is *net slower* than n=3.
- **The head costs about 530 MiB**, not the ~1.3 GB we assumed. The external
  MTP file costs **+776 MiB for identical performance** — do not download it.
- **⚠ The ngram-mod rows are not a claim:** they were measured on a repeated
  prompt with a warm cache, which inflates n-gram speculation enormously. They
  must be re-measured on novel generation before anyone quotes them, and we
  print them only because omitting a 4–6× number we saw would be dishonest.
- **The DFlash2 drafter does not work on b10566:** `--spec-type draft-dflash`
  exists but the loader fails on all three quant variants with a tensor-count
  mismatch (`expected 81, got 58`). Untested, not disproven.

### 4.2 No decode cliff to 98K context

UD-Q2_K_XL, q4_0 KV, fully resident:

| Context | 8192 | 16384 | 32768 | 49152 | 65536 | 81920 | 98304 |
|---|---|---|---|---|---|---|---|
| VRAM (MiB) | 9782 | 9956 | 10324 | 10692 | 11060 | 11428 | 11796 |
| Decode (tok/s) | 21.05 | 20.98 | 21.00 | 21.00 | 20.99 | 21.00 | 21.02 |

Flat to 98K within 0.4%. The decode collapse in llama.cpp issue #27623
(35.6 → 1.4 tok/s past ~90K) **does not reproduce here**. KV grows ~368 MiB per
16K at q4_0, roughly 23 KiB per token: on a 12 GB card the binding constraint on
context is memory, not speed.

> **Caveat: measured against an allocated, not a filled, cache.** This sweep
> varies `-c` (the allocated context size) while the generated sequence itself
> stays short — VRAM grows because the cache is *allocated*, and decode stays
> flat because it stays largely *empty* across most of that range. It is not
> yet a measurement of decode speed with a cache genuinely filled to 98K real
> tokens, which is the condition a long session actually reaches. Issue #27623
> is measured at real occupancy (~91K KV position); "does not reproduce here"
> may mean the two are measuring different quantities rather than that the
> collapse is absent. **Filled-cache re-measurement at real occupancy is
> pending (internal gate S1).** Until it lands, read "flat to 98K" as a
> property of this allocation sweep, not a settled claim about long-session
> decode speed.

### 4.3 KV cache precision is a memory knob, not a speed knob

UD-IQ3_XXS at 16K context:

| KV type | VRAM (MiB) | Decode (tok/s) |
|---|---|---|
| f16 | 11718 | 20.28 |
| q8_0 | 11266 | 20.21 |
| q4_0 | 11010 | 20.18 |

Identical within noise. Quantize the KV cache purely to buy weight budget or
context length.

> **Caveat: same allocated-cache condition as §4.2.** This table is measured
> at 16K allocated context with a near-empty cache, not a filled one, and does
> not measure **prefill** — only decode. Whether KV precision stays
> speed-neutral once the cache is genuinely filled, and whether it costs
> anything on prefill, is untested here; a hybrid-architecture prefill
> collapse under sub-8-bit K has been reported elsewhere on this exact model
> family and has not been checked against our build. **Filled-cache and
> prefill re-measurement across f16/q8_0/q4_0 is pending (internal gate S1).**
> Until then, "a memory knob, not a speed knob" is provisional, not settled.

### 4.4 Where the bandwidth ceiling actually is

UD-Q2_K_XL, 9.82 GB of weights at 21.748 tok/s, implies **213.5 GB/s effective
read bandwidth = 59.3% of the 3060's 360 GB/s peak.** Flash attention and
batch/micro-batch sweeps moved decode by under 1%: an achievable ceiling, not
recoverable overhead.

**Correction to our own first-draft mechanism claim.** We originally wrote that
IQ-format quants reach only ~169 GB/s, roughly 21% below K-format, because they
dequantize more slowly on Ampere. That number reconstructs only from
UD-IQ2_XXS (7.26 GB × 23.33 tok/s ≈ 169 GB/s). The same arithmetic on
UD-IQ3_XXS — also an IQ-format build — gives ≈221 GB/s, *above*
UD-Q2_K_XL's 213 GB/s K-format number. Our two IQ-format builds disagree with
each other by more than IQ disagrees with K, so "IQ dequantizes slower than
K-format" is not a rule our own data supports — it was one build (IQ2_XXS)
mistaken for a format-wide mechanism.

**What we can say:** format-dependent decode efficiency does vary across these
builds, but it does not follow a simple IQ-slower-than-K rule — IQ2_XXS is a
slow outlier here, not IQ-format decode in general. Why IQ3_XXS and Q2_K_XL
land at nearly the same tok/s despite the size gap is genuinely unexplained by
anything in this data. We flag the mechanism as **open, pending a dedicated
format study** (`llama-bench` Q4_0 vs Q4_K_M vs IQ4_XS is the planned next
step); until it resolves, do not use file format (IQ vs K) as a proxy for
decode speed at a given size.

### 4.5 Five warnings for anyone benchmarking quantized builds

Each cost us a re-run:

1. **n=20 subsets carry ±10 points of noise** (§3.1).
2. **GSM8K is at ceiling for this model class** and discriminates nothing
   (§2.4).
3. **Builds below ~8.4 GiB silently ship without the MTP head** (§3, §4.1).
4. **Decode efficiency (effective GB/s) varies by build in ways that don't
   track a simple IQ-vs-K format rule** — our first-draft "IQ is 21% slower"
   claim was falsified by our own second IQ build (§4.4). Treat file size as
   a rough proxy for speed, not an exact one.
5. **§4.2/§4.3 are measured against an allocated, largely-empty KV cache.**
   Filled-cache behavior — especially past ~90K tokens, and on prefill — is
   unverified; re-measurement is pending (internal gate S1).

---

## 5. Test-time compute: a pre-registered negative result

### 5.1 The protocol, pre-registered

The standard playbook says sampling k candidates at non-zero temperature and
selecting among them beats a single greedy attempt, cheaply. On a card with no
per-token cost that can run overnight, spending 5× the compute to buy back
quality lost to quantization is exactly the trade a local user should want. We
wanted this to work.

Before any GPU time was spent we wrote decision gates with numeric thresholds,
plus a standing rule that no experiment starts before the previous gate's
verdict is recorded. Condensed; full text in the repo:

- **GATE 1** (~50 tasks, ~2h in): CONTINUE if pass@5 − pass@1 ≥ 5 points; KILL
  BOTH RUNS if < 3.
- **GATE 2** (after the IQ3_XXS run, ~6h): verified-selection ≥ 85% → proceed
  to a second build. Verified < 84% but pass@5 ≥ 88% → selection is the
  bottleneck, not the model. **pass@5 < 86% → direction dead; stop and publish
  the measurement findings as-is.**
- **GATE 3** (after the second build): verified ≥ 80% → "compute substitutes
  for VRAM" holds at two points on the curve.

Setup: UD-IQ3_XXS, k=5 completions per task, temperature 0.8, top_p 0.95,
max_tokens 768, 8192 context, all 164 tasks. Three scores per task: pass@1;
**pass@5**, the oracle ceiling (did *any* of the five pass the hidden test — the
best a perfect selector could do); and **verified-selection**, where candidates
are run against a visible test synthesised from the docstring's `>>>` examples
and the first to pass is chosen.

### 5.2 What happened

**GATE 1, at 51/164 tasks: CONTINUE.** pass@1 = 0.678, pass@5 = 0.882,
verified-selection = 0.882 — headroom +20.4 points against a +5 threshold, and
selection lossless (verified hitting the ceiling exactly). A clean win, so far.

**GATE 2, full 164 tasks: STOP THIS BRANCH.**

| Measurement | Score |
|---|---|
| Greedy baseline, temperature 0, single attempt | **82.3%** |
| Best-of-5, temp 0.8 — pass@1 (one sample) | 60.0% |
| Best-of-5, temp 0.8 — verified-selection | 67.1% |
| Best-of-5, temp 0.8 — **pass@5 (oracle ceiling)** | **75.0%** |

**The oracle ceiling of the sampled approach is 7.3 points below a single
greedy attempt.** Not the selection mechanism — the ceiling. Even a perfect
selector would lose.

The matched per-task analysis says why. Of the 29 tasks greedy failed, sampling
recovered **3**; of those greedy passed, sampling **lost 15**. Turning the
temperature up bought three problems and cost fifteen. The strongest variant we
could construct on paper — anchoring one candidate at temperature 0 — projects
to +1.8 points, not worth the GPU hours.

### 5.3 Mechanism and scope

Our reading: **quantization noise and sampling noise compound.** At 3.2 bpw the
model already generates from a distribution perturbed away from the one it was
trained to produce; greedy decoding takes the argmax of that perturbed
distribution, which is usually still right. Temperature 0.8 adds a second
deviation on top of the first, and below some bit-width the two together push
the *whole candidate set* off the correct answer more often than diversity
rescues it.

Every best-of-n result we know of was measured on full-precision or
lightly-quantized models. To our knowledge this is the first report that naive
test-time compute **fails** at around 3.2 bpw — a technique the field treats as
a free quality lever, inverting in the regime consumer hardware operates in.

Scope: one model, one bit-width, one temperature, one benchmark, k=5. The gate
file specified temp 1.0 only if GATE 1 failed, and it did not, so we never swept
temperature; we make no claim at 4-bit or 8-bit. We do claim that anyone about
to spend GPU hours on best-of-n at low bit-width should measure their own greedy
baseline first, because the sign of the effect is not what the literature would
lead them to expect.

The gates did their job — the second planned run was auto-killed by the
GATE 2 verdict, saving ~6 GPU-hours — and they are why we can tell you we did
not go looking for a positive result after the fact.

---

## 6. Practical recommendations

### The flagship configuration

**UD-IQ3_XXS + built-in MTP at n=2 + q8_0 KV, 16K context — 82.3% HumanEval at
33.8 tok/s, 11642 MiB peak.** The best quality that fits a headless 12 GB card,
and the configuration the rest of this paper refers back to.

```bash
llama-server \
  -m Qwen3.8-27B-UD-IQ3_XXS.gguf \
  -ngl 99 -fa on -c 16384 \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1 \
  --jinja --chat-template-file chat_template.jinja \
  --host 127.0.0.1 --port 8080
```

It leaves under 650 MiB of headroom and requires a headless GPU.

### Fallback: display attached, or more context wanted

**UD-Q2_K_XL + built-in MTP at n=3 + q8_0 KV, 16K context — 72.0% HumanEval at
35.4 tok/s, 10740 MiB peak.** Ten points cheaper, ~900 MiB roomier, marginally
faster.

```bash
llama-server \
  -m Qwen3.8-27B-UD-Q2_K_XL.gguf \
  -ngl 99 -fa on -c 16384 \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 --parallel 1 \
  --jinja --chat-template-file chat_template.jinja \
  --host 127.0.0.1 --port 8080
```

### Rules of thumb

1. **Buy weight bits before context** — ~10 HumanEval points per GiB. Quantize
   the KV cache to q8_0 or q4_0 instead; measured against an allocated
   (near-empty) cache it costs no decode speed (§4.3) — filled-cache and
   prefill re-measurement are pending (internal gate S1).
2. **Do not drop below about 8.4 GiB in this family**, or you lose the MTP head
   and its +68% on top of the quality.
3. **Never download the external MTP file:** +776 MiB for performance identical
   to the head already in your GGUF.
4. **n=2–3 for MTP, never 4.** Acceptance collapses and n=4 is net slower.
5. **Do not offload to system RAM to fit a bigger build** — 2.12 tok/s against
   21.75 resident. PCIe at 21–25 GB/s is slower than the DDR4 it reads from,
   and halving a tensor's bit-width is ~18× cheaper per token.
6. **Set `reasoning_effort` deliberately.** Left alone this model spends
   500–600 tokens reasoning about grade-school arithmetic.
7. **Do not use best-of-n at this bit-width without measuring your own greedy
   baseline** (§5).

---

## 7. Limitations

We would rather list these than have them listed for us.

- **HumanEval only, and HumanEval is a weak instrument** — single-turn,
  short-horizon, and substantially **quant-blind**. Independent data
  (quanteval.ai, 4156 rows) shows an aider-polyglot-style ladder for this exact
  model separating Q4_K_M (54.25), UD-Q3_K_XL (49.0) and UD-IQ2_XXS (32.5) far
  more sharply than HumanEval does. Read our curve as a *lower bound* on
  quantization damage.
- **Agentic performance is unmeasured, and the literature says that is exactly
  where damage concentrates.** RedHatAI's Gemma-4-31B-FP8 card reports MMLU-Pro,
  MATH-500, AIME, GPQA and LiveCodeBench all recovering 99.2–103.8% of baseline
  while **BFCLv4-Agentic falls 65.82 → 56.53 (85.9% recovery)**. Corroborating:
  on τ²-bench (arXiv 2607.27275) headline scores stay flat while tool-error rate
  doubles, 19.51 → 38.26; arXiv 2608.06564 measures decision margin falling to
  ×0.86 at 4-bit, ×0.33 at 3-bit, ×0.00 at 2-bit. **Our 82.3% licenses no claim
  about agentic coding at 3.2 bpw.**
- **No full-precision anchor** (§2.5), and **chat-template prompting**: we
  generate through `/v1/chat/completions` with the model's own template, not the
  original paper's few-shot completion format — the right protocol for measuring
  what a user will run, the wrong one for comparison against published numbers.
  Do not put our 82.3% next to a leaderboard row.
- **Single seed, one run per full-set point.** n=164 cuts the per-task quantum
  to 0.6 points; §3 now reports a 95% CI per point and an exact McNemar test
  per adjacent step, but that is still one seed and one run each — it bounds
  sampling uncertainty within a run, not seed-to-seed variance. Treat
  differences under ~3 points, or CIs that overlap, as unresolved.
- **One hardware configuration** (RTX 3060 12GB, Ampere, CUDA) and **one
  runtime** (llama.cpp b10566). The unresolved format-efficiency question in
  §4.4 is an Ampere result;
  the VRAM-fit conclusions are specific to 12288 MiB; EXL3 and other runtimes
  are unmeasured by us.
- **The n=20-era results embedded here** — the Minitron (§3.2) and CPU-offload
  comparisons — carry ±10 points and are labelled where they appear.
- **Contamination is unquantified.** HumanEval is old and widely mirrored. No
  absolute score here is unseen-problem ability; the *relative* curve across
  builds of one checkpoint is the load-bearing result.
- **Counter-evidence, engaged rather than ignored:** arXiv 2411.02355 finds
  quantization close to lossless across >500k evaluations — at 70B and 405B, at
  8-bit and W4A16. The scope condition appears to be that damage bites at
  ≤4 bits, hardest ≤3 bits, and worse for small models than large. Our regime is
  the far corner of that scope, which is why our results differ and why they
  were worth measuring.

---

## 8. Reproducibility

Everything is in the repo (link TBD): start at `harness/`, check the numbers
against `results/full164/`, verify your files against
`results/full164/model_hashes.txt`.

**Harness** — Python 3 standard library plus `requests`; needs `llama-server`,
`llama-bench` and `llama-cli` on `$PATH`.

| Script | What it does |
|---|---|
| `bootstrap_linux.sh` | Fresh-machine setup: GPU gating, CUDA llama.cpp build, deps, GGUF downloads, hardware profile, generated matrix config. Idempotent. |
| `profile_hw.py` | Hardware snapshot to JSON. |
| `bench_speed.py` | Wraps `llama-bench` (pp512/tg128, 3 reps); streamed SHA-256 of the model file, peak VRAM via 500 ms `nvidia-smi` polling. |
| `eval_quality.py` | HumanEval (and GSM8K) against a running `llama-server`. |
| `bestofn.py` | The §5 best-of-n / verified-selection eval. Resumable — writes after every task, which is what made mid-run gate checks possible. |
| `run_matrix.py` | Drives a config matrix: start server → eval → stop → bench → one JSON row. |
| `run_curve.sh`, `run_spec2.sh`, `run_bestofn.sh` | The exact drivers for §3, §4.1 and §5, published verbatim, so every flag in §4 and §6 can be checked against the code that produced the row. |

**Raw results** (`results/full164/`) — per-task pass/fail for the four full-164
runs, the per-candidate best-of-n records behind §5, and the speculation
ablation table. Every speed JSON carries `model_file`, `sha256_16`,
`file_size_gb`, `flags`, `pp_tok_s`, `tg_tok_s`, `peak_vram_mb`,
`llama_cpp_build` and `timestamp`.

**Model hashes** (`results/full164/model_hashes.txt`) — SHA-256 of each measured GGUF.
Community quantizers re-upload under unchanged filenames, so the hash is the
only thing tying this curve to specific files. Check yours before assuming our
numbers apply.

**Datasets.** HumanEval from `openai/human-eval`, cached to `harness/data/`;
GSM8K the same way, excluded here per §2.4.

**Safety note.** `eval_quality.py` executes untrusted model-generated code with
a 10-second timeout and no sandbox beyond it — the same trust model as the
standard HumanEval harness. Run it in a container.

---

## Acknowledgements, licensing, contact

Quantized builds by **Unsloth** (`unsloth/Qwen3.8-27B-GGUF`); fixed chat
template by **froggeric**; base model by **Qwen**, Apache-2.0; inference by
**llama.cpp**. We rehost no weights. Our harness and this paper are Apache-2.0.

**This is an open, ongoing effort.** Corrections and suggestions are welcome
and wanted — write to the contact address at the top of this paper, or open a
repo issue. Corrections are recorded in the repo, not quietly edited in. If a
number here is wrong, we want to be the second people to know.

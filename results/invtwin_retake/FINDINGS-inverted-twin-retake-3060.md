# FINDINGS — Inverted-twin RETAKE on the 3060: the old 3060 numbers were RIGHT (2026-09-02)

Run C (`FINDINGS.md`, "Run C verdict", correction 1) said:

> Inverted twin (axis B) is STRONGER than banked: on L40S+CPU it measures
> α_raw 0.745, **+77% at n=8 and +65% at n=16** (vs our 3060's 0.331/+42%/−14%).
> **The 3060 numbers likely suffered the weak-CPU verify + short battery;
> RE-TAKE on the 3060 before quoting either.**

This is that retake. **The verdict inverts the suspicion: the short battery was
not the problem.** The retaken 3060 numbers land almost exactly on the old ones
(α 0.320 vs 0.331; +37.6% vs +42%; −10.0% vs −14%). What the corrected battery
*does* overturn is something else — and bigger.

## 0. Protocol (the corrections that matter)

| | old `inverted_twin.sh` | this retake |
|---|---|---|
| prompts | **1**, run **twice** (warm repeat) | **6 distinct, novel**: 3 code, 2 prose, 1 structured |
| tokens measured | ~300, 1 prompt | **1920 per config** (6 x 320, all hit the cap) |
| acceptance | server-wide aggregate at exit | **per request**, `timings.draft_n / draft_n_accepted` |
| thinking | left ON | **`enable_thinking:false`** (see §5) |
| baseline | `llama-bench -n 32` | **same harness, same prompts, no `-md`** |
| cache | default | `--cache-ram 0 --no-cache-idle-slots` |

Config: Q8_0 27B target on **CPU** (`-ngl 0`), IQ3_XXS draft on **GPU**
(`-ngld 99`), greedy (`temperature 0, top_k 1, seed 42`), `-c 8192`,
`--parallel 1`. Box: RTX 3060 12 GB; Ryzen 5 3600 presented to the guest as
**10 vCPU / 1 thread per core** / 47 GB. Binary `9efa1595e (10718)`.
Raw: `retake.json`, `invtwin_retake.log`.

## 1. Headline

Pooled over 1920 generated tokens per config (`sum(tokens)/sum(decode_ms)`):

| config | decode t/s | acceptance | vs 10-thread base | vs its own base |
|---|---:|---:|---:|---:|
| `baseline_nospec` (no spec, 10 threads) | **1.158** | — | — | — |
| `baseline_nospec_t6` (no spec, 6 threads) | **1.291** | — | +11.5% | — |
| `spec_n8` | **1.594** | 0.3197 | **+37.6%** | +37.6% |
| `spec_n16` | **1.043** | 0.1744 | **−10.0%** | −10.0% |
| `spec_n8 -b 256 -ub 128` | 1.592 | 0.3197 | +37.5% | +37.5% |
| `spec_n8 -t 6` | **1.727** | 0.3197 | +49.1% | **+33.7%** |
| `spec_n8 -b 256 -ub 128 -t 6` | **1.734** | 0.3197 | **+49.7%** | +34.3% |

**Best absolute: 1.734 t/s** (n=8 + 6-thread pinning) — **+49.7%** on the
shipping-default no-spec config, of which **+11.5% is thread count alone** and
~34% is speculation.

## 2. Comparison against run C — and what it means

| quantity | run C, L40S+CPU | old 3060 | **this retake (3060)** |
|---|---:|---:|---:|
| α_raw (acceptance) | 0.745 | 0.331 | **0.3197** |
| gain at n=8 | +77% | +42% | **+37.6%** |
| gain at n=16 | +65% | −14% | **−10.0%** |

The retake **reproduces the old 3060 numbers within a few points on all three
quantities**, on a battery 6x larger and free of warm-repeat contamination.

1. **Run C's correction 1 is itself partly wrong.** The 3060's weak numbers were
   not a battery artifact. They are the real behaviour of this pair on this box.
   Amend `FINDINGS.md` correction 1: the re-take is done, the 3060 numbers stand.
2. **The α discrepancy cannot be placement.** Run C's own placement theorem
   states α is a property of the *model pair*, and the pair is identical
   (Q8_0 target, IQ3_XXS draft, greedy) in both runs. α 0.745 on L40S vs 0.320
   here therefore **cannot** be a hardware effect — one of the two is measuring
   something else. Given α varies **0.186 → 0.614 across content types** here
   (§3), and the L40S figure came from the same single-code-prompt battery this
   document retires, **the likely reading is that 0.745 is the warm-repeat /
   code-only artifact and ~0.32 is the honest value** — the same failure family
   as `FINDINGS.md` correction 2 (MTP 0.759/0.786 → 0.640). **Re-take the L40S
   lane on this battery before +77% or +65% is quoted again.**

## 3. The real finding: the speedup is a property of the CONTENT

Per-prompt at n=8 (all 320 generated tokens each):

| prompt | type | no-spec t/s | n=8 t/s | speedup | acceptance |
|---|---|---:|---:|---:|---:|
| `struct` (JSON weather network) | structured | 1.152 | **2.638** | **+129%** | **0.6140** |
| `code1` (Rust interval merge) | code | 1.152 | **2.424** | **+110%** | 0.5520 |
| `code3` (C++ SPSC ring buffer) | code | 1.151 | 1.683 | +46% | 0.3456 |
| `code2` (Go join-order DP) | code | 1.155 | 1.633 | +41% | 0.3333 |
| `prose1` (DRAM refresh explainer) | prose | 1.156 | 1.137 | **−1.6%** | 0.1938 |
| `prose2` (canal-vs-railway essay) | prose | 1.160 | 1.113 | **−4.1%** | 0.1855 |

**Speculation on this pair ranges from +129% to −4% depending purely on what is
generated.** Structured output and boilerplate-heavy code are highly predictable
to the IQ3 draft; discursive prose is not, and there speculation is a **net
loss**. "+37.6%" is an average over a mix nobody actually runs.

The old single-code-prompt battery structurally could not produce this, and it
is more actionable than the average: **gate speculation on observed acceptance
and disable below ~0.25**, which would convert both prose regressions into
no-ops and lift the pooled figure above 1.594 t/s.

## 4. The two CPU-side levers

### 4.1 `-b 256 -ub 128` on the verify side: **inert. Measured, not guessed.**

1.592 vs 1.594 t/s (10 threads) and 1.734 vs 1.727 (6 threads) — inside noise,
and per-request draft counts are **bit-identical** (`471/260`, `693/231`, ...),
confirming the flag changed nothing.

Structural reason: at `--parallel 1` with 8 draft tokens the verify batch is
**9 tokens**. Both 512 (default) and 256 are far above it, so the cap never
binds. **Batch-size tuning cannot help single-stream speculative verify**;
drop it from this lane.

### 4.2 6-thread pinning: real, but **not** a speculation win

`cpu_drafter_latency.md` finding 3 ("more threads is worse") reproduces on the
llama.cpp side, at smaller magnitude:

| | 10 threads | 6 threads | gain |
|---|---:|---:|---:|
| no-spec baseline | 1.158 | **1.291** | **+11.5%** |
| spec n=8 | 1.594 | **1.727** | +8.3% |

**The 6-thread baseline is the row that makes this honest.** Comparing
`spec_n8_t6` (1.727) to the *10-thread* baseline gives a flattering +49.1%, but
~11.5 points of that is pure thread count. Against its own 6-thread baseline the
speculation gain is **+33.7%** — slightly *less* than at 10 threads, because
pinning helps the un-speculated target more than the speculated one.

The guest exposes 10 vCPUs at 1 thread/core, so this is not the SMT-sibling
contention `cpu_drafter_latency.md` diagnosed on bare metal; more likely
memory-bandwidth saturation plus scheduling overhead. Either way: **set `-t 6`
for the CPU-hosted target regardless of speculation.**

## 5. Method note — thinking had to be disabled

The first launch ran with shipping template defaults. With `max_tokens 320` and
reasoning on, the model spends the entire budget inside `<think>`, which (a)
makes all six prompts generate the same *kind* of text, destroying the
code/prose/structured contrast that is the point of §3, and (b) measures
acceptance on reasoning tokens rather than answer text. Killed at 2 minutes and
restarted with `chat_template_kwargs: {"enable_thinking": false}` (the same trap
produces *empty* answers on the IQ3 flagship — see companion-prefill findings §5).

**Any α measured on this model family without stating its thinking mode is
uninterpretable.** Add it to battery metadata alongside generation length
(`FINDINGS.md` correction 3).

## 6. Verdict

The Q8-in-RAM verifier lane on the 3060 is **real but modest and highly
content-dependent**: **+37.6% at n=8** (α 0.320) against a like-for-like
baseline, **+49.7%** in the best config (n=8, `-t 6`, 1.158 → 1.734 t/s), and a
**net loss at n=16** (−10.0%). It does not reproduce run C's L40S +77%/+65%, and
the α gap points at the L40S battery, not the 3060's CPU. At 1.73 t/s the lane
is far from interactive — its use case is unattended batch work where Q8 quality
is worth ~10 min per 1000 tokens, not a chat path.

**Actions:** (1) amend `FINDINGS.md` correction 1 — the 3060 numbers stand;
(2) re-take the L40S lane on this 6-prompt battery before quoting +77%;
(3) drop `-b/-ub` tuning from this lane; (4) adopt `-t 6` for CPU-hosted
targets; (5) prototype acceptance-gated speculation (disable below α≈0.25).

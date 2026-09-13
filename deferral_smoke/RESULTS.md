# RESULTS — Expert-Deferral Quality Smoke Test (exp 4.1-smoke)

**VERDICT: RED at every tested depth, including d=1. Do not build the deferral scheduler
as specified.**

Run 2026-09-04/05 on the box. Build: `/data/scratch/deferral_smoke/llama.cpp` =
`daef7b687` + the two v11 patches + `deferral_sim.patch`, isolated tree, CUDA arch 86,
Release. The v11 tree and its deployed binaries were md5-verified untouched throughout.

All arms: `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 8192 -np 1 --n-cpu-moe 16 -t 8
--spec-type none`. Only `LLAMA_DEFER_DEPTH` varied. Per-arm engagement confirmed from the
server log line `defer_depth=N defer_layers=16`.

---

## The numbers

### Leg A — perplexity (20 chunks, `-c 2048`, deterministic)

| d | PPL | Δ vs d=0 |
|---|---|---|
| 0 | 6.8224 | — |
| 1 | 7.8123 | +14.5% |
| 2 | 9.0465 | +32.6% |
| 3 | 9.9576 | +46.0% |

### Leg B — HumanEval-164, paired McNemar vs d=0

| d | passed | rate | regressions b | improvements c | McNemar p | mean tok |
|---|---|---|---|---|---|---|
| 0 | 154/164 | 0.93902 | — | — | — | 264.4 |
| 1 | 153/164 | 0.93293 | 4 | 3 | 1.0000 | 257.2 |
| 2 | 151/164 | 0.92073 | 9 | 6 | 0.6072 | 249.5 |
| 3 | 139/164 | 0.84756 | 18 | 3 | **0.0015** | 248.7 |

### Leg C — aider-polyglot edit, 34 tasks, paired McNemar vs d=0

| d | solved 1st attempt | b | c | p | solved overall | b | c | p | format-ok | mean tok |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 13/34 | — | — | — | 18/34 | — | — | — | 34/34 | 1140.0 |
| 1 | **3/34** | 10 | **0** | **0.00195** | **5/34** | 14 | 1 | **0.00098** | 32/34 | 1442.6 |
| 2 | 1/34 | 12 | 0 | 0.00049 | 3/34 | 15 | 0 | 0.00006 | 32/34 | 1337.9 |
| 3 | 1/34 | 12 | 0 | 0.00049 | 2/34 | 16 | 0 | 0.00003 | 32/34 | 1042.6 |

---

## What this means

**Even d=1 is fatal.** On realistic multi-file code editing, deferring by a single block
takes first-attempt task success from 13/34 to 3/34: ten regressions, **zero**
improvements, p = 0.002. Overall success goes 18/34 → 5/34. This is not a noise band and
not churn — the complete absence of improvements is the tell. Since d=1 is the *minimum*
deferral that buys any overlap at all, there is no surviving configuration. The
optimisation is dead as specified.

**HumanEval-164 would have green-lit d=1, and it would have been wrong.** HE-164 gave
d=1 a McNemar p of exactly 1.0000 (4 regressions, 3 improvements — textbook balanced
churn). Had the smoke test stopped at the primary instrument, this would have shipped a
GREEN light for a change that destroys 77% of first-attempt edit success. The difference
is task shape: HumanEval is short, self-contained function completion (~265 tokens);
polyglot edit is long-horizon multi-file work (~1140 tokens). Degradation compounds over
generation length, so the short benchmark cannot see it.

**Perplexity ordered the depths correctly but mis-stated the magnitude.** It flagged
d=1 at +14.5% (real harm, correctly detected) while HE-164 said "noise". On this
evidence perplexity was the *better* early signal, not the worse one — the opposite of
this project's usual `BENCHMARK_LANDSCAPE.md` guidance, because here we are comparing one
model against itself with a single variable rather than ranking quants.

**The model floundered rather than failed fast.** Mean completion tokens rose 1140 → 1443
(+27%) at d=1 while success collapsed. It generated more and achieved less.

**Weight VRAM was unaffected**, as predicted: peak 10884 MiB (d=0) vs 10796–10898 MiB
across arms — within polling noise, no systematic increase. Expert placement was never
touched.

---

## Correction to the pre-registered decision rule

The rule in `RUNBOOK.md` §7.2 used **`edit_compliance_1`** as the Leg C gate. That was the
wrong field. `edit_compliance_1` (= `edit_ok_1`) measures whether the model emitted a
*well-formed* edit, not a *correct* one. It barely moved (34/34 → 32/34, p = 0.50 at every
depth) while actual task success collapsed. A gate built on it would have passed all three
deferral arms.

The correct gate is task success (`passed_1` / `passed`). Recorded here rather than quietly
fixed, because the near-miss is the point: a plausible-looking metric name concealed a
format check.

---

## Confidence and limits

- `d=0` reproduced the historical ncm16 nospec baseline **exactly** — 154/164, 0.93902,
  264.4 mean tokens, matching `results/q3kxl/reliability_Q3_K_XL_ncm16_nospec_he164_3060.json`.
  The rig is sound and the patch is a true no-op when off.
- Leg C arms were clean: 0 errors, 0 timeouts, correct engagement line on all four.
- n=34 is small, but b/c asymmetries of 10/0, 14/1, 15/0 and 16/0 are not marginal.
- All quality runs shared the box with a long-running `export_v2.sh` archive job. That job
  turned out to be **network-bound, not CPU-bound** (zstd at 0.7% CPU, load average 1.02),
  so contention was negligible. An earlier draft of this file claimed wall times were
  contaminated; that was wrong and came from comparing this session's **nospec** runs
  against the historical **spec-on** run. Like for like: historical nospec = 2222 s,
  this session's d=0 nospec = 1098 s, i.e. this session was *faster*. Timings here are usable.
- Decode throughput was **identical across all four arms** — 45.61 / 45.69 / 45.47 / 45.56
  tok/s at d = 0/1/2/3 (21.9 ms/token throughout). That is the patch behaving exactly as
  specified: scheduling-neutral, math-only. It also means **this experiment measured the
  cost of the optimisation without ever delivering its benefit** — by design, since
  delivering the benefit is the expensive build the experiment existed to de-risk.
- Not tested: interaction with speculative decoding (all arms ran `--spec-type none` by
  design), and `LLAMA_DEFER_LAYERS` other than 16.

## Recommendation

Do not build the overlap scheduler for this model at any depth in 1..3. If the idea is
pursued, it needs a fundamentally different formulation — one where the CPU experts'
contribution is not simply late, e.g. keeping a cheap on-GPU approximation of the routed
experts in the residual stream and correcting it when the CPU result lands. That is a
different, larger project and would need its own quality gate before any build.

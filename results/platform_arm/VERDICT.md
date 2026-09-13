# Platform arm: is sub-4-bit i-quant output WORSE on sm_86 than on sm_89?

**VERDICT: NO. Branch (b). The "open earthquake" is closed, negative.**

The RTX 3060 (sm_86) does **not** reproduce its old low HumanEval numbers. Run
under the L40S battery's own harness and flags, the same GGUF files on the same
3060 score within noise of the L40S. There is no sub-4-bit correctness defect on
sm_86, and there is no upstream bug to report on that basis.

The old 82.3% / 72.0% / 49.4% figures were a **harness defect, and we found and
reproduced the exact one**: the 2026-08 3060 harness never sent
`chat_template_kwargs.enable_thinking=false`, and the chat template it was
served with defaults thinking to **true**. Every one of those runs was scored on
a model that spent its 1024-token budget reasoning and got truncated before it
finished writing code.

_Run 2026-09-02 01:48-06:00 UTC on `ollama` (RTX 3060 12 GiB, sm_86).
All numbers are k/n with 95% Wilson intervals; McNemar is the exact two-sided
binomial test on discordant pairs._

---

## 1. Comparability: what was held fixed

The only intended free variable is the GPU architecture.

| | L40S arm (2026-09-02, AWS) | 3060 arm (this run) |
|---|---|---|
| harness file | `quality_battery/harness/reliability.py` | **byte-identical copy**, md5 `0ae0f60dcb53453274ed0ca00c10c39e` |
| `extract_code` | fixed parser (closed → unclosed → stripped) | same file, same parser |
| GGUF files | `unsloth-q27b/*.gguf` | the same files, same box of origin |
| HumanEval | canonical 164, md5 `88771747746422b8669027625578460c` | same |
| server flags | `-ngl 99 -fa on --jinja --no-warmup -c 8192 -np 1 -ctk q8_0 -ctv q8_0 --spec-type none` | identical (`-ngl 24` for the Q8 anchor only — 27.7 GiB will not fit on 12 GiB) |
| sampling | attempt 1 greedy: `temperature=0, top_p=1.0, seed=42`, `max_tokens=1024`, `enable_thinking=false`, `cache_prompt=false` | identical |
| llama.cpp | `b356fa2` (b10566 rebuild proved version is not a confound) | `daef7b6` = b10712, `/data/scratch/bin-unpatched` (unpatched, clean tree) |
| GPU | L40S, sm_89 | RTX 3060, sm_86 |

GPU guard `< 500 MiB` enforced before every server start; server stopped and
VRAM verified drained after every arm.

## 2. Anchor gate: Q8_0 — PASSED, and harder than required

Q8_0 at `-ngl 24` (partial offload, 1.72 t/s — 27.7 GiB does not fit on 12 GiB,
so the full 164 would take ~6 h) was run first as the gate, at 25 tasks, then
resumed to 54 tasks with the remaining wall clock.

Over the 54 paired tasks:

| | result |
|---|---|
| both pass | 52 |
| both fail | 2 — `HumanEval/38`, `HumanEval/50` (**the same two the L40S fails**) |
| discordant | **0** (b=0, c=0, p=1) |
| identical completion-token counts | **54/54 = 100.0%**, mean \|Δ\| = 0.0 tokens |

This is a two-sided anchor: the 3060 reproduces both the L40S's passes *and* its
specific failures. And greedy decoding is deterministic given identical
arithmetic, so an exact token-count match across two different GPUs on every
single task is far stronger than merely matching scores — the prompt, chat
template, tokenizer, sampling path and Q8 kernels are numerically equivalent
across the two machines. **The harness is sound on this box; any difference
found below is attributable to the quant kernels and not to the plumbing.**

## 3. Primary result: the same GGUFs, both GPUs

| build | L40S sm_89 | RTX 3060 sm_86 | 3060 95% CI | delta | **old 3060 number** |
|---|---|---|---|---|---|
| IQ3_XXS | 154/164 = 93.9% | **152/164 = 92.7%** | 87.6%-95.8% | −1.2 pt | 135/164 = 82.3% |
| Q2_K_XL | 154/164 = 93.9% | **153/164 = 93.3%** | 88.4%-96.2% | −0.6 pt | 118/164 = 72.0% |
| IQ2_XXS | 132/164 = 80.5% | **128/164 = 78.0%** | 71.1%-83.7% | −2.4 pt | 81/164 = 49.4% |
| Q8_0 (anchor) | 153/164 = 93.3% | 52/54 = 96.3% on the paired window | 87.5%-99.0% | 0 flips | — |

### Paired McNemar, same build across platforms (key = task_id, N=1 greedy)

| build | n paired | both pass | b (L40S pass / 3060 fail) | c (L40S fail / 3060 pass) | neither | p (exact) |
|---|---|---|---|---|---|---|
| IQ3_XXS | 164 | 152 | 2 | 0 | 10 | **0.5** |
| Q2_K_XL | 164 | 153 | 1 | 0 | 10 | **1.0** |
| IQ2_XXS | 164 | 122 | 10 | 6 | 26 | **0.4545** |
| Q8_0 (anchor) | 54 | 52 | 0 | 0 | 2 | 1.0 |

Not one build shows a significant platform difference. The three sub-4-bit
deltas (−1.2, −0.6, −2.4 pt) are all inside the confidence intervals, and the
IQ2_XXS discordance is close to symmetric (10 regressions against 6 recoveries).

### Flipped tasks

- **IQ3_XXS** — 2 regressions (`HumanEval/32`, `HumanEval/91`), 0 recoveries.
- **Q2_K_XL** — 1 regression (`HumanEval/83`), 0 recoveries.
- **IQ2_XXS** — 10 regressions (`11, 13, 40, 44, 71, 75, 94, 102, 126, 138`),
  6 recoveries (`1, 10, 46, 91, 124, 156`).

The divergences are ordinary sampling divergence, not corruption. On
`HumanEval/91` both GPUs write the same well-formed function; the L40S variant
additionally guards that `"I"` is a standalone word and the 3060 variant does
not, so the 3060 version fails the test. That is a *different correct-looking
answer*, which is exactly what a numerically-noisy-but-unbiased kernel produces
— not a broken one.

## 4. Sub-4-bit kernels ARE numerically divergent across architectures — but the divergence is unbiased

Fraction of paired tasks on which the two GPUs emitted the *same number* of
completion tokens under greedy decoding:

| build | identical token count | mean \|Δ\| tokens |
|---|---|---|
| Q8_0 | **54/54 = 100.0%** | 0.0 |
| IQ3_XXS | 150/164 = 91.5% | 7.2 |
| Q2_K_XL | 136/164 = 82.9% | 6.5 |
| IQ2_XXS | **110/164 = 67.1%** | 28.9 |

Same-GPU controls, from the L40S campaign's own stored runs, calibrate what
"no architectural difference" looks like:

| control (same L40S GPU) | identical token count |
|---|---|
| IQ3_XXS, MTP spec vs no-spec | 152/164 = 92.7% |
| IQ3_XXS, master vs b10566 build | 152/164 = 92.7% |
| IQ2_XXS, master vs b10566 build | **164/164 = 100.0%** |

So IQ3_XXS's cross-platform 91.5% is indistinguishable from ordinary
same-GPU config noise, but **IQ2_XXS is a genuine architectural signal**: swapping
llama.cpp versions on one GPU changes nothing (100%), while swapping sm_89 for
sm_86 changes a third of the greedy trajectories (67.1%). The sub-2-bit i-quant
path is *not* numerically equivalent across architectures.

**This is real, and it is also harmless.** It costs 2.4 pt with a McNemar p of
0.45 and flips tasks in both directions. It is a reproducibility caveat, not a
correctness defect: never expect bit-identical sub-2-bit output across GPU
architectures, and never A/B two quants on two different cards.

Caveat: the Q2_K_XL row mixes two variables — the L40S Q2_K_XL run used MTP
speculation while the 3060 arm is no-spec. The IQ3_XXS same-GPU control puts
the spec contribution at about 7 pp of token-count divergence, so Q2_K_XL's
82.9% is only mildly beyond that. Its *score* comparison is unaffected: the
L40S battery showed MTP vs no-spec to be per-task identical.

## 5. Attribution: what actually caused the old numbers

The 3060 clearing itself only tells us the platform is innocent. To convict the
real cause we reconstructed the 2026-08 protocol on this same GPU with today's
build, and flipped one knob.

The old protocol (`harness/run_eval.sh` + `harness/eval_quality.py`) differed
from the battery in six ways: it served
`--chat-template-file models/templates/chat_template.jinja` instead of `--jinja`;
it ran MTP speculation; its request body was only
`{model, messages, temperature:0, max_tokens}` — **no `chat_template_kwargs`**,
no seed, no top_p, no `cache_prompt:false`; it had no `reasoning_content`
fallback; and its `extract_code` returned raw text when a fence was left open.

Line 6 of that chat template (`qwen3.8-froggeric-v22.3`):

```jinja
{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}
```

**Thinking defaults to ON, and the old harness never turned it off.**

| arm (IQ3_XXS, RTX 3060, today's build) | solved | rate | 95% CI | mean tok/task |
|---|---|---|---|---|
| **original 2026-08 run** (archived) | 135/164 | 82.3% | — | — |
| **L1** — full legacy protocol reconstructed | **135/164** | **82.3%** | 75.8%-87.4% | **656** |
| **L3** — legacy, but `enable_thinking=false` | 153/164 | 93.3% | 88.4%-96.2% | 222 |
| battery protocol (primary arm above) | 152/164 | 92.7% | 87.6%-95.8% | 220 |
| L40S battery reference | 154/164 | 93.9% | 89.1%-96.7% | 223 |

L1 does not merely match the old score — it matches the old run
**164/164 task-for-task** (b=0, c=0, p=1). The old protocol is exactly
reproducible on today's build and today's GPU, which rules out both the GPU and
the llama.cpp version as the cause.

| pair | both | b (A-pass/B-fail) | c (A-fail/B-pass) | neither | p (exact) |
|---|---|---|---|---|---|
| L1 vs L3 (**only the thinking flag differs**) | 131 | 4 | 22 | 7 | **0.0005335** |
| L3 vs battery protocol (template + spec + parser differ) | 152 | 1 | 0 | 11 | **1.0** |
| L1 vs battery protocol | 131 | 4 | 21 | 8 | 0.0009105 |

**One request field is the entire effect.** Turning thinking off, changing
nothing else, recovers +18 net tasks (p=0.00053). Every *other* protocol
difference combined — custom chat-template file, MTP speculation, the
closed-fence-only `extract_code`, the missing seed/top_p/cache_prompt — is worth
exactly one task (p=1.0).

### Mechanism

| arm | fails | hit the 1024-token cap | failures with **no code fence at all** |
|---|---|---|---|
| L1 (thinking ON) | 29 | 26/164 overall, **26 of the 29 failures** | **15 of 29** |
| L3 (thinking OFF) | 11 | 2/164 | 0 |
| battery protocol | 12 | 2/164 | 0 |

With thinking on, mean output triples (656 vs 220 tokens) and 26 of 29 failures
are truncations at `max_tokens=1024`; over half never emit a code fence at all.
The old harness was not measuring code quality below 4 bits — it was measuring
how often a reasoning block overran a 1024-token budget. Cheaper quants ramble
longer, which is precisely why the apparent damage scaled with quantization
(IQ3 −11.6 pt, Q2_K_XL −21.9 pt, IQ2_XXS −31.1 pt) and looked like a
quantization cliff.

## 6. Consequences for the program

1. **No upstream bug report.** There is no sm_86 sub-4-bit correctness defect.
   Do not file one. (`kernel_profile.md`'s finding that sub-4-bit i-quants take
   a separate LUT code path stands — it is a *speed* result, and the +33% GEMV
   opportunity is untouched by this verdict.)
2. **The flagship is re-rated upward and the 11-point gap to anchor is gone.**
   IQ3_XXS on the 3060 is 92.7% (95% CI 87.6-95.8), not 82.3%. It is
   statistically indistinguishable from Q8_0 and from the L40S.
3. **The quality/size exchange rate collapses on the 3060 too, exactly as on the
   L40S.** Q2_K_XL 93.3%, IQ3_XXS 92.7%, Q8 anchor — mutually indistinguishable.
   The "9.7 pts/GiB HumanEval exchange rate" was an artifact of the same harness
   defect and must be retired. Real separation below ~3 bpw shows up in
   *format/adherence* instruments (edit-format compliance), not in HumanEval —
   which is the L40S battery's agentic-ladder finding, now confirmed as the only
   place the damage is visible.
4. **IQ2_XXS is the one build where quantization cost is still real**: 78.0% on
   the 3060 / 80.5% on the L40S vs ~93% for everything at or above ~3 bpw.
5. **Standing instrument rule (new).** Any harness that talks to a Qwen3-family
   model must send `chat_template_kwargs.enable_thinking=false` *and* assert on
   the realized token budget. A silent default flipped a whole program's
   headline conclusion. Add a canary: if mean tokens/task on HumanEval exceeds
   ~350, the thinking switch is not taking effect.
6. **Cross-architecture reproducibility caveat (new).** Sub-2-bit i-quants
   diverge on a third of greedy trajectories between sm_86 and sm_89. Never
   compare two quants measured on two different cards.

## 7. Limitations

- The Q8_0 anchor is paired over the first 54 of 164 tasks (partial offload runs
  at 1.72 t/s, ~6 h for the full set). Within that window it is perfect and
  two-sided — 0 discordant tasks, both of the L40S's in-window failures
  reproduced, 54/54 identical token counts — but it is not a full-length anchor.
- L40S Q2_K_XL was measured with MTP speculation, the 3060 arm without. The
  battery showed spec to be per-task neutral on this model, and the score
  comparison is unaffected, but the token-count divergence figure for Q2_K_XL
  mixes the two.
- Attribution arms were run on IQ3_XXS only. The mechanism (thinking-block
  truncation at a fixed token cap) predicts a *larger* effect on Q2_K_XL and
  IQ2_XXS, consistent with their larger old-vs-new gaps, but that was not
  measured directly.
- Single greedy run per build; no repeat-run variance estimate on the 3060.

## 8. Provenance / reproduce

```
box            ollama, NVIDIA GeForce RTX 3060 12044 MiB, sm_86
binary         /data/scratch/bin-unpatched/llama-server
               version 0.3.0-dev, commit daef7b6 (= b10712), clean tree
harness        /data/projects/q27b_on_12gb/platform_arm/harness/reliability.py
               md5 0ae0f60dcb53453274ed0ca00c10c39e (identical to the L40S battery)
probe harness  harness/reliability_probe.py   (adds --legacy-body / --legacy-extract)
drivers        platform_arm.sh  chain.sh  legacy_arm.sh  attribution.sh
results        /data/projects/q27b_on_12gb/results/platform_arm/
analysis       analyze_platform.py (paired/McNemar), legacy_cmp.py, attr_stats.py, mech.py

# primary arm
./platform_arm.sh he IQ3_XXS 99 164
./platform_arm.sh he Q2_K_XL 99 164
./platform_arm.sh he IQ2_XXS 99 164
./platform_arm.sh he Q8_0    24 164     # anchor, partial offload

# attribution
./legacy_arm.sh L1 IQ3_XXS 164          # full 2026-08 protocol
./legacy_arm.sh L3 IQ3_XXS 164          # ... but enable_thinking=false
```

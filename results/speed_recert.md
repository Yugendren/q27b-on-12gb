# CORRECTED official speed certification (thinking OFF)

Supersedes the speed cells of `results/mtp_profile.md` and the shipping-config
numbers quoted in `results/kernel_profile.md` §"shipping config impact".

**Why a re-take.** `results/platform_arm/VERDICT.md` (2026-09-02) found the legacy
harness never sent `chat_template_kwargs.enable_thinking=false` and the served
template defaults **thinking ON**. Every historical measurement therefore ran on
*reasoning-token* content. Speed on this rig is not content-neutral — acceptance,
and therefore the whole MTP speedup, is a property of the content being generated
(FINDINGS, inverted twin: +129% structured / +110% code / −2…−4% prose). So the
speed table had to be re-measured, not just re-labelled.

Rig: ollama box, RTX 3060 12 GB (sm_86) + Ryzen 5 3600.
Binary: `/data/projects/llama.cpp/build/bin/llama-server`, **build 10718 /
9efa1595e** — the same build that produced `mtp_profile.md` and
`kernel_profile.md`. Model files unchanged (`models/unsloth-q27b/*.gguf`).

## Protocol

```
server  -ngl 99 -fa on -ctk <kv> -ctv <kv> -c <ctx> --parallel 1 --no-warmup
        --chat-template-file models/templates/chat_template.jinja
        [--spec-type draft-mtp --spec-draft-n-max <n>]
client  temperature=0, top_k=1, seed=1234, max_tokens=400, cache_prompt=false,
        chat_template_kwargs={"enable_thinking": false}
prompt  the mtp_profile.md B-tree module prompt, verbatim (so the corrected
        cells are directly comparable to the thinking-ON cells)
cell    1 discarded warm-up + 4 measured requests; reported = mean of the 4
VRAM    peak of a 1 Hz nvidia-smi sample taken across the requests
guard   server refuses to start unless the card is < 500 MiB
```

Driver `speed_recert.py`, analysis `analyze_recert.py`, raw per-request records
in `results/speed_recert/{sweep,depth,content}.json`, per-cell server logs in
`results/speed_recert/*.serverlog`, flat table in `results/speed_recert.tsv`.

**Verification the switch actually took effect** (this is the whole point of the
re-take): every measured request records `reasoning_content` length and a
`<think>` scan. All 100+ measured requests: `reasoning_len = 0`, no `<think>`.
Pre-flight A/B on the shipping config, same prompt: thinking ON → 625 chars of
`reasoning_content`, 400 tokens, `finish=length`; thinking OFF → 0 chars,
216 tokens, `finish=stop`. The standing HumanEval tripwire (>350 mean completion
tokens/task) reads **158.8** — passes with 2.2× margin.

## 1. The official table

Decode t/s, 400-token code generation at shallow depth (~110-token prompt).

| # | config | decode t/s | vs no-spec | accept | mean len | VRAM peak MiB |
|---|---|---:|---:|---:|---:|---:|
| **1** | **IQ3_XXS + MTP n=2, q8_0 KV, c=16384  [SHIPPING]** | **34.39** | **1.719×** | 0.781 | 2.56 | **11958** |
| 2 | IQ3_XXS + MTP n=2, **f16** KV, c=16384 | **OOM** | — | — | — | 11960 (dies) |
| 3a | IQ3_XXS + MTP n=3, q8_0 KV, c=16384 | 32.76 | 1.638× | 0.638 | 2.91 | 11984 |
| 3b | IQ3_XXS + MTP n=4 / n=5, q8_0 KV, c=16384 | **OOM** | — | — | — | — |
| 4a | Q2_K_XL + MTP n=2, f16 KV, c=16384 | 34.73 | 1.735× | 0.771 | 2.54 | 11156 |
| 4b | Q2_K_XL + MTP n=2, q8_0 KV, c=16384 | 34.64 | 1.732× | 0.781 | 2.56 | **10704** |
| 5 | IQ3_XXS no-spec, f16 KV, c=16384 | 20.15 | 1.007× | — | — | 11440 |
| 5b | IQ3_XXS no-spec, q8_0 KV, c=16384 (**raw floor**) | **20.00** | 1.000× | — | — | 10986 |

Within-cell spread across the 4 timed requests is **0.1–0.8%**. End-of-session
drift controls (same cells re-run ~3 h later, card at its 170 W cap the whole
time): shipping −0.56%, Q2_K_XL −0.07%, c8k/n3 +0.26%, c8k/f16/n3 +1.09%.
**Treat ±1% as the noise floor; nothing under ~2% is a real difference.**

Build cross-check: the shipping cell on the quality-arm binary
(`/data/scratch/bin-unpatched`, build 169 / daef7b6) gives 34.08 t/s = −0.89%,
inside the drift band. No build confound between the speed and quality lineages.

### Lineage bridge — what the thinking fix cost on the speedometer

Identical cell (`c=8192`, q8_0 KV, MTP n=2), only the switch differs:

| protocol | decode t/s | acceptance |
|---|---:|---:|
| thinking ON (`mtp_profile.md`, 2026-09-01) | 35.26 | 0.818 |
| thinking OFF (this cert) | **34.06** | **0.781** |

Reasoning tokens are **easier to draft** than answer tokens: acceptance falls
0.818 → 0.781 and raw decode falls 3.4%. Every legacy speed number was therefore
optimistic by ~3% at n=2 — and much more at deeper n (§3). This is the *first*
reason raw t/s is the wrong metric here; §5 is the second and far larger one.

## 2. f16 KV: the +7% is REAL but UNREACHABLE in the shipping config

`kernel_profile.md` measured f16 19.33 vs q8_0 18.07 t/s (+7.0%) with
`llama-bench` at **depth 16384**, no speculation. Two corrections:

**(a) The gain is a depth effect and is ~zero at the depth the cert measures.**
At a ~110-token prompt, f16 vs q8_0 no-spec is **20.15 vs 20.00 = +0.72%**
(c=16384) and +0.58% at c=8192 — inside noise. Re-measured with a **13.5K-token
prompt** so decode runs at real depth, the effect reappears exactly as
`kernel_profile.md` describes:

| depth ~13,517 tokens, c=16384 | decode t/s |
|---|---:|
| IQ3_XXS no-spec, q8_0 KV | 18.13 |
| IQ3_XXS no-spec, **f16** KV | **19.11 (+5.42%)** |
| IQ3_XXS + MTP n=2, q8_0 KV | **36.00** (1.986× — acceptance 0.981) |
| IQ3_XXS + MTP n=2, f16 KV | **OOM** |

(+5.4% at 13.5K vs the published +7.0% at 16.4K is the expected depth scaling.)

**(b) f16 KV + MTP does not fit on 12 GB at c=16384.** The server *loads*
(11,960 MiB) and then dies with `CUDA error: out of memory` in
`ggml_cuda_pool_vmm::alloc` on the **first request** — the allocation failure is
in graph compute, not model load, so a load-time VRAM check will not catch it.
n=3/4/5 fail earlier, at load. Measured ceiling for IQ3_XXS + f16 KV + MTP n=2:
**c=12288 fits (33.92 t/s, 12,026 MiB peak — 262 MiB of headroom on the card)**,
c=16384 does not.

**Verdict: the "free +7%" is not available to the shipping config.** It is
available only by giving something up — dropping to c=12288, dropping
speculation (which costs 42%, not 7%), or dropping to Q2_K_XL (§4).

## 3. MTP depth sweep — the optimum moved from n=3 to **n=2**

At c=16384 the sweep is truncated by VRAM (n=4 and n=5 OOM in graph compute), so
the full curve was taken at c=8192, where `mtp_profile.md` measured its
thinking-ON curve. Same cells, same prompt, only the switch differs:

| n | thinking ON (legacy) t/s | accept | **thinking OFF (corrected)** t/s | **accept** |
|---:|---:|---:|---:|---:|
| 1 | 31.77 | 0.937 | not re-taken | — |
| 2 | 35.26 | 0.818 | **34.06** | **0.781** |
| 3 | **35.67** ← old optimum | 0.724 | 32.57 (−4.4%) | 0.637 |
| 4 | 34.65 | 0.619 | 31.61 (−7.2%) | 0.556 |
| 5 | 35.22 | 0.614 | 30.22 (−11.3%) | 0.508 |

**α shifted down at every depth and the curve is now monotonically decreasing:
the new optimal draft depth is n = 2.** Under the legacy protocol the curve was
flat-to-peaked at n=3–5 because reasoning text drafts well; on real answer
content the per-drafted-token cost (13.4 ms, `mtp_profile.md` finding 1) is no
longer repaid past n=2. Confirmed at c=16384 too: n=3 is 32.76 vs n=2's 34.39
(−4.7%). **The shipping config was already at the optimum; it is now at the
optimum for a measured reason.**

One exception worth flagging: at **f16** KV, n=3 does *not* collapse
(34.51 / 34.89 t/s in two independent runs, acceptance 0.700 vs q8_0's 0.637).
Quantized KV costs ~6 points of draft acceptance at n=3 while costing ~0 at n=2
(0.781 vs 0.784) — i.e. **KV quantization damages the MTP head's deeper drafts**,
a mechanism not previously recorded. It is not exploitable here (f16+MTP does not
fit at c=16384), but it is the reason the n-curve is KV-dependent.

**Reproducibility caveat, both f16/n=3 cells:** greedy output was *not*
byte-reproducible across repeats of the identical request (two distinct
completions, two distinct draft counts, 5.5% t/s spread), while every q8_0 cell
was bit-stable. Also, at c=8192/q8_0, n=4 and n=5 produce a *different* greedy
completion than no-spec/n=2/n=3 (which are mutually identical). **Deep MTP is not
output-neutral on this build** — one prompt, so an observation, not a verdict,
but it is a quality reason to stay at n=2 on top of the speed reason.

## 4. Q2_K_XL vs IQ3_XXS — the kernel profile's prediction was too generous

Q2_K_XL is 9,374 MiB of weights vs IQ3_XXS's 10,429 (10.1% fewer bytes) and now
scores 93.3% on HumanEval-164 vs IQ3_XXS's 92.7%. Matched cells:

| pair | Q2_K_XL | IQ3_XXS | delta |
|---|---:|---:|---:|
| MTP n=2, q8_0 KV, c=16384 | 34.64 | 34.39 | **+0.75%** |
| MTP n=2, f16 KV, c=16384 | 34.73 | OOM | — |
| no-spec floor (see §1) | not taken | 20.00 | — |

**Q2_K_XL is +0.75% — inside the ±1% drift band, i.e. NOT measurably faster.**
The kernel profile predicted +5–8% from 10–12% fewer bytes on the argument that
sub-4-bit GEMV is compute-bound; the measurement says the realised gain is
smaller still. Consistent with the profile's core finding (achieved bandwidth
*falls* as quants shrink: IQ2_XXS is 34% smaller for 12.8% faster), extrapolated
to this much smaller byte step. **Smaller-is-faster is not merely saturating
below 4 bits — at a 10% byte step it is fully exhausted.**

What Q2_K_XL *does* buy is **1,254 MiB of VRAM** (10,704 vs 11,958 peak), and
that is a real capability difference: it is the only flagship-class config that
runs **f16 KV at c=16384** (11,156 MiB) and it runs **MTP n=4** there (34.34 t/s,
11,462 MiB) where IQ3_XXS OOMs at n=4. Speed is not the reason to choose it.

Note the two quants' greedy completions on the code prompt are **different**
(sha1 `1d82fb3c` vs `d7b314ae`) despite identical draft counters (311/243) —
the identical acceptance is a coincidence, not identical output. Their speed
cells are therefore matched in token count but not in token content.

## 5. Tokens-per-TASK: the number that actually moved

Same server (shipping config), same 25 HumanEval tasks, greedy, `max_tokens=1024`
(the legacy budget), concurrency 1. **The only variable is the request body.**

| | corrected (thinking OFF) | legacy (thinking ON) | ratio |
|---|---:|---:|---:|
| mean completion tokens / task | **158.8** | 474.0 | **2.99× fewer** |
| mean wall-clock / task | **4.79 s** | 13.38 s | **2.79× faster** |
| median wall-clock / task | 4.67 s | 12.60 s | 2.70× |
| worst task | 9.5 s | 30.0 s | 3.16× |
| total wall for 25 tasks | **119.8 s** | 334.6 s | 2.79× |
| pass@1 | **25/25 (100%)** | 23/25 (92%) | +2 tasks |
| attempts hitting the 1024 cap | 0 | 1 | — |
| *effective* throughput (tokens/s over the run) | 33.1 | 35.4 | **0.94×** |

**The thinking fix is a 2.79× task-speed win — 4× larger than every hardware and
config lever in this document combined**, and it comes with +2 solved tasks.

The last row is the punchline and belongs in every future certification: under
the legacy protocol the machine posts a **higher raw t/s** (35.4 vs 33.1 —
reasoning tokens draft better, §1) while taking **2.8× longer to do the job**.
Decode t/s and effective speed moved in *opposite directions*. Raw t/s is a
kernel diagnostic; tokens-per-task is the product metric.

## 6. Content-typed speculation check — recalibrating the gating rule

Best config (= shipping) vs no-spec, same 5 novel prompts, 1 warm-up + 2 measured
each, 400 tokens, thinking off.

| content | spec t/s | no-spec t/s | speedup | acceptance | mean len |
|---|---:|---:|---:|---:|---:|
| struct (JSON registry) | 37.84 | 19.98 | **+89.3%** | 0.928 | 2.86 |
| code1 (LSM memtable) | 35.48 | 20.06 | **+76.9%** | 0.833 | 2.67 |
| code2 (Go scheduler) | 33.08 | 20.01 | **+65.3%** | 0.746 | 2.49 |
| prose2 (rail-gauge essay) | 29.66 | 20.00 | **+48.3%** | 0.616 | 2.23 |
| prose1 (SSD write amp) | 29.23 | 20.00 | **+46.1%** | 0.599 | 2.20 |
| *(long-context task, §2)* | 36.00 | 18.13 | +98.6% | 0.981 | 2.96 |

Two results:

1. **The content law survives the correction, with the same ordering**
   (struct > code > prose) and a wide acceptance spread (0.599–0.981, vs the
   inverted twin's 0.186–0.614 on its lane). No-spec is flat at 20.01 t/s across
   all five prompts (spread **0.48%**), so the entire variation is speculation,
   not prompt difficulty.

2. **The α≈0.25 threshold is exactly right — and it never fires on this lane.**
   Derived from this run's own constants rather than inherited:

   ```
   T_base            = 1000 / 20.001              = 50.00 ms/token
   T_round(n=2)      = mean_len * 1000 / 34.388   = 74.45 ms      (mean_len 2.560)
   break-even        mean_len* = T_round / T_base = 1.4890
   with mean_len = 1 + 2*alpha  ->  alpha*        = 0.2445
   ```

   The standing rule ("disable speculation below α≈0.25", imported from the
   inverted-twin lane) turns out to be within 2% of the MTP lane's true
   break-even. **But the lowest acceptance measured on any content type is
   0.599 — 2.45× above the threshold — and even that returns +46%.** The gate is
   correct and inert.

   The same two constants predict every content cell from acceptance alone,
   `t/s = (1 + 2α) / 74.45 ms`, to within **1.4%** (struct +1.4%, code1 +0.9%,
   code2 +1.2%, prose2 +1.1%, prose1 +1.1%). So the whole content-typed table
   collapses to one measured number per content type: α.

   **Recommended rule for the MTP lane: leave speculation on unconditionally and
   do not build content classification or runtime acceptance gating for it** —
   the gate would need α < 0.245, which no observed content produces (prose, the
   floor, sits at 0.599). Keep the gate on the CPU-target/twin lane, where prose
   genuinely goes negative and where it was measured.

## 7. Verdict

**Corrected official config: unchanged — IQ3_XXS + MTP n=2 + q8_0 KV + c=16384
at 34.39 t/s (1.719× over the 20.00 t/s no-spec floor), 11,958 MiB peak.**
The re-take moved the headline number from a thinking-ON 35.26 (measured at
c=8192) to **34.39**, and confirmed the config choice for reasons that are now
measured rather than inherited:

* **n=2 is the optimum** — the n=3 advantage was an artefact of drafting
  reasoning text; on answer content the curve falls monotonically.
* **f16 KV is not adoptable here** — its (real) +5.4% lives at depth, and
  f16 + MTP OOMs at c=16384. Available only at c≤12288, or on Q2_K_XL.
* **Q2_K_XL is not a speed upgrade** (+0.75%, inside noise). It is a
  1,254 MiB *VRAM* upgrade, and the flagship's justification remains the
  format/adherence gap (94.1% vs 67.6% edit-compliance), not speed.
* **The real speed win of this cycle was the thinking switch: 2.79× per task.**

### Standing rules added

1. **Report tokens-per-task alongside decode t/s.** This cycle produced a
   config change where the two metrics move in opposite directions; a cert
   quoting only t/s would have preferred the 2.8×-slower protocol.
2. **VRAM must be certified from a peak sample taken during requests, not at
   load.** Three configs in this run pass a load-time check and then OOM inside
   `ggml_cuda_pool_vmm::alloc` on the first request (f16+MTP at c=16384;
   q8_0+MTP n=4/n=5 at c=16384).
3. **Quote KV-type deltas with the depth they were measured at.** f16-vs-q8_0 is
   +0.7% at shallow depth and +5.4% at 13.5K on the same rig; a bare "+7%" is
   not a well-formed claim.
4. **Cell-level noise on this box is ±1%** (within-cell 0.1–0.8%, 3-hour drift
   ≤1.1% with the card pinned at its 170 W cap). Do not report differences
   below ~2% as effects.

### Not covered / open

* Depth arm run at 13.5K, not 16.4K (a 19.6K-token prompt overflows c=16384);
  the +5.4% is therefore a slight under-read of the published depth-16384 +7.0%.
* Q2_K_XL's no-spec floor and its full n-curve at q8_0 KV were not taken.
* The f16/n=3 non-reproducibility and the n≥4 output divergence are one-prompt
  observations, not adjudicated results.
* Everything here is batch 1, `--parallel 1`, single request at a time.

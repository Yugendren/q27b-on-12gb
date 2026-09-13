# Certification: Qwen3.6-35B-A3B as candidate revv v2 flagship

Campaign run 2026-09-03 on the ollama box (RTX 3060 12 GiB / sm_86, Ryzen 5
3600, 47 GiB DDR4, `/data` 787 GiB). Cost: $0 (own hardware).

Artifacts: `results/cert35b/` (placement.json, cells.json, depth.json, the two
reliability JSONL/summary pairs, edit_*.jsonl/json, ident_*.json, memcap_*.json,
per-cell server logs). Drivers: `cert35b_sweep.py`, `cert35b_stage2.sh`,
`/data/scratch/cert35b/{mtp_identity.py,memcap_test.sh,gguf_probe.py}`.

Reference arm throughout: the certified 27B flagship
(Qwen3.8-27B UD-IQ3_XXS + MTP n=2, q8_0 KV, c=16384) as recorded in
`results/speed_recert.md`, `results/platform_arm/` and
`quality_battery/results/edit_IQ3_XXS.*`.

---

## STAGE 0 — GATES

**(a) LICENSE — PASS, unencumbered.** `Qwen/Qwen3.6-35B-A3B` ships a verbatim
201-line Apache-2.0 (`Copyright 2026 Alibaba Cloud`), fetched and read in full.
No appended clauses, no acceptable-use rider, no field-of-use restriction; the
only "notwithstanding" in the file is stock Apache §4. The model card carries no
supplementary terms. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` is `license:apache-2.0`
and ungated.

This is the material difference from Qwen3.8-Flash-Next, which the MoE-tiering
campaign killed on licensing: Flash-Next's Qwen Community License requires a
separately negotiated licence for AI-coding products with no revenue floor.
**Qwen3.6-35B-A3B has no such clause. The Flash-Next trap does not apply.**

**(b) llama.cpp support + MTP head — PASS.** Our existing quality-arm binary
(`llama-server-v11`, commit `daef7b687`) already carries `--spec-type draft-mtp`,
`--spec-draft-n-max`, `--n-cpu-moe` / `-ncmoe`, and loads the model with no
rebuild.

GGUF header (`gguf_probe.py`) on `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`:

| key | value |
|---|---|
| `general.architecture` | `qwen35moe` |
| `qwen35moe.block_count` | 41 (40 model layers + 1 MTP layer) |
| `qwen35moe.nextn_predict_layers` | **1 — the MTP head is present** |
| MTP tensors | `blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` |
| `expert_count` / `expert_used_count` | 256 / 8 (+1 shared) |
| `expert_feed_forward_length` | 512 |
| `full_attention_interval` | 4 → only 10 of 40 layers hold a real KV cache |
| `context_length` | 262144 |
| imatrix | `unsloth_calibration`, 510 entries / 77 chunks |

Server confirms at load: `creating MTP draft context against the target model`.

**Quant selection, and a disk constraint worth recording.** `/data` was at 96 %
(33 GiB free) before the campaign and is at 99 % (≈12 GiB) after. UD-Q4_K_XL
(21.28 GiB) was taken as the primary rung. **The planned second rung (~16 GiB,
UD-Q3_K_XL or UD-IQ4_XS) was NOT downloaded — there is no room for it**, and
nothing on `/data` was identifiable as dead scratch with enough confidence to
delete (the largest items are a 97 GiB swapfile that the memory-cap test depends
on, and 349 GiB of live project trees). This is the one deliverable gap in the
campaign; see Open questions.

---

## STAGE 1 — PLACEMENT / SPEED

Protocol = the corrected speed-recert protocol of record, unchanged except for
the model: server `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 -np 1 --jinja
--no-warmup`; client greedy `temperature=0, top_k=1, seed=1234, max_tokens=400,
cache_prompt=false, chat_template_kwargs.enable_thinking=false`; the
`mtp_profile.md` B-tree prompt verbatim; **1 discarded warm-up + 4 measured
requests**; VRAM = peak of a 1 Hz sample taken *during the requests* (the
peak-VRAM-under-load rule); GPU guard < 500 MiB before every launch.

Thinking-off verified per request: every measured request recorded
`reasoning_content` length 0 and no `<think>` leak. Pre-flight A/B on this model:
thinking ON → 1473 chars of `reasoning_content`; OFF → 0. **Qwen3.6's template
defaults thinking ON**, exactly like the 27B — the trap is live on this model too.

### The placement curve

`--n-cpu-moe N` keeps the MoE weights of the first N of 41 layers on the CPU.
N = 41 is all-experts-on-CPU; descending N moves experts onto the card.

| ncm | no-spec t/s | MTP n=2 t/s | MTP n=4 t/s | prefill t/s | VRAM peak (n=2) | host RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 29.31 | 32.55 | 34.11 | 87–90 | 3 588 | 22.2 GiB |
| 36 | 31.17 | 37.40 | 32.64 | 98 | 5 886 | 19.5 GiB |
| 32 | 33.37 | 39.88 | 36.64 | 109–110 | 7 740 | 17.5 GiB |
| 28 | 35.42 | 44.32 | 39.54 | 120–123 | 9 596 | 15.5 GiB |
| **26** | 36.93 | **45.51** | 41.47 | 128–129 | **10 524** | 14.5 GiB |
| 24 | 38.78 | 43.96 | 45.07 | 133–139 | 11 452 | 13.5 GiB |
| 22 | 39.79 | **FAIL** | **OOM** | 148 | 11 586 (no-spec) | 12.3 GiB |
| 20 | OOM | OOM | OOM | — | — | — |

Drift controls (cells re-run later in the session): ncm26/n=2 45.51 → 45.65
(+0.29 %), ncm28/n=2 44.32 → 44.07 (−0.58 %), ncm24/n=4 45.07 → 44.45 (−1.38 %).
**±1 % remains the noise floor.** Draft acceptance reproduced to full double
precision on the repeat (0.8600682593856656 both times).

**BEST CONFIG: `--n-cpu-moe 26 --spec-type draft-mtp --spec-draft-n-max 2`,
q8_0 KV, c=16384 → 45.51 / 45.65 t/s, prefill 128 t/s, VRAM peak 10 524 MiB,
host RSS 14.5 GiB.** It is chosen over ncm24/n=4 (44.75 t/s mean of two, and
11 588 MiB leaves only 700 MiB of headroom) on both speed and safety.

### The curve is linear, and it explains the field prior

Least squares on the no-spec arm:

> **T_decode(N) = 14.447 ms + N × 0.4845 ms**  — fits all 7 cells within ±1.1 %,
> i.e. at the noise floor.

VRAM accounting corroborates it: every layer moved to the card costs a constant
**463.4 MiB** (measured 463.0–464.5 across five independent steps), so the full
expert mass is 18.4 GiB on top of a 2.86 GiB non-expert base ≈ the 21.28 GiB
file. The intercept is the GPU-side floor; the slope is DDR4 expert streaming.

Two consequences:
- **Extrapolating to N=0 gives 69.2 t/s no-spec.** The Reddit P40 report of
  50–70 t/s is therefore *consistent with our physics, not in tension with it*:
  a 24 GiB P40 holds all 18.4 GiB of experts, so those users are sitting at the
  intercept. On a 12 GiB card we can seat only 15 of 41 layers and the remaining
  26 layers of DDR4 streaming cost 12.6 ms/token.
- The binding constraint on this box is **VRAM capacity, not DDR4 bandwidth**.
  A 16 GiB card would seat ~9 more layers (≈ +4.4 ms saved) and a 24 GiB card
  would seat all of them. This is the single highest-leverage upgrade.

### Depth

One depth point at ~13–14 K filled context (ncm26, c=16384):

| config | decode t/s | prefill t/s | acceptance | VRAM peak |
|---|---:|---:|---:|---:|
| MTP n=2 @ depth | **44.07** | 364.6 | **0.928** | 10 384 |
| no-spec @ depth | 34.56 | 368.2 | — | 9 594 |

Decode holds (−3.2 % vs shallow), **acceptance rises 0.860 → 0.928**, and
prefill improves 2.9× as the batch gets large. As on the 27B, deep context is
speculation's best regime — and VRAM does *not* grow at depth (only 10 of 40
layers carry a KV cache), so the shallow peak is the binding one.

---

## STAGE 2 — QUALITY

Protocol of record (`platform_arm.sh` `he` phase): `-ngl 99 -fa on --jinja
--no-warmup -c 8192 -np 1 -ctk q8_0 -ctv q8_0 --spec-type none`, driven by the
**byte-identical** `reliability.py` at `--max-attempts 1`, greedy, thinking off.
The one forced deviation: the model cannot fit in 12 GiB, so quality is certified
**at the shipping placement** (`--n-cpu-moe 26`). That matters — see the MTP
finding below, which shows CPU/CUDA expert placement is not numerically neutral.

### HumanEval-164

| | 27B UD-IQ3_XXS (certified) | 35B-A3B UD-Q4_K_XL |
|---|---:|---:|
| pass@1 | 152/164 = **92.68 %** [87.65, 95.77] | 151/164 = **92.07 %** [86.91, 95.31] |
| mean completion tokens | 219.8 | 269.7 |
| thinking tripwire (>350) | ok | **ok** |

**McNemar exact, paired over all 164 tasks: b=8, c=7, p = 1.0 —
indistinguishable.** Both models sit on the same ~93 % saturation ceiling that
the EXL3 head-to-head already identified for this instrument.

### Edit-format compliance (aider-polyglot, 34 Python exercises)

Exact 34/34 task-set match with the reference run confirmed (set difference
empty both ways). Reference params reproduced: max_tokens 2048, attempts 2,
test_timeout 30, seed 42, sample_seed 1337.

| metric | 27B UD-IQ3_XXS | 35B-A3B UD-Q4_K_XL | McNemar |
|---|---:|---:|---|
| **edit-format compliance @1** | 32/34 = **94.12 %** | 32/34 = **94.12 %** | b=2 c=2, p=1.0 — identical |
| pass@1 | 4/34 = 11.76 % | 6/34 = 17.65 % | b=3 c=5, p=0.727 |
| pass@2 | 9/34 = 26.47 % | 14/34 = **41.18 %** | b=4 c=9, p=0.267 |

Edit-compliance — the instrument of record for this model family, and the thing
that justified the flagship quant in the first place — is **exactly at parity**.

Task success is **directionally better** for the 35B (+14.7 pts on pass@2, 9
discordant wins vs 4) but **does not reach significance at n=34** (p=0.267).
Consistent with the standing rule that this instrument cannot resolve small
differences at this n, **this is not quotable as a capability win** — it is a
hypothesis worth a larger run.

*Caveat:* the 27B edit reference was measured on an L40S (AWS quality battery),
not on this 3060. HumanEval is same-box for both arms; edit-compliance is not.
The platform arm established GPU-architecture equivalence for Q8 greedy
(54/54 identical token counts), and IQ3_XXS at ~3 bpw sits above the sub-2-bit
band where cross-architecture reproducibility breaks down, so the comparison is
believed sound — but it is a cross-platform comparison and is flagged as such.

---

## THE MTP LOSSLESSNESS FINDING (unplanned, and it matters)

FINDINGS carries a standing open flag: greedy divergence was observed on the 27B
at n≥3–4, and "deep-MTP losslessness needs a dedicated check before any n>2
config ever ships". We ran that check here, at n=2, because the entire speed case
for this model rests on MTP.

`mtp_identity.py`: 5 greedy probes (2 code, 1 structured JSON, 1 prose, 1 short
HumanEval-style), temperature 0 / top_k 1 / seed 1234, 512 max tokens, compared
by SHA-256 of the completion.

| comparison | identical | reading |
|---|---|---|
| no-spec A vs no-spec B (same server, back-to-back) | **5/5** | pipeline is deterministic |
| no-spec A vs no-spec C (**server restart**, same flags) | **5/5** | deterministic across restarts |
| no-spec vs **draft-mtp n=2** | **1/5** | **diverges** |
| no-spec vs **draft-mtp n=4** | **1/5** | **diverges** |

The control is what makes this conclusive. Greedy decoding on this stack is
bit-reproducible both within a server and across restarts, so the MTP divergence
is **not** ambient nondeterminism — it is a real property of the `draft-mtp`
path. Divergence appears *early* (first differing character at 99–338 of ~2 000),
and the single probe that agrees is the only one short enough to stop on its own
(201 tokens); all four 512-token probes differ.

Speculative decoding is supposed to be mathematically exact — the verify step
should reject any draft token the target would not itself have emitted. A
divergence therefore indicates the MTP verify path in this build is not exact
for this architecture (or that the MTP layer perturbs target state). **This is a
llama.cpp/architecture-level observation, not a quant artifact.**

Because the shipping config depends on MTP, quality was **re-certified with MTP
on** rather than assumed. Result below.

### Quality re-certified WITH MTP on

Same protocol, same placement, only `--spec-type` differs:

| arm | HumanEval-164 | tripwire | wall |
|---|---:|---:|---:|
| 35B-A3B, `--spec-type none` | 151/164 = 92.07 % | 269.7 tok ok | 1 428 s |
| 35B-A3B, **`draft-mtp` n=2 (shipping)** | **153/164 = 93.29 %** | 265.5 tok ok | **1 165 s** |

McNemar no-spec vs MTP: b=1, c=3, **p = 0.625 — indistinguishable**. Only 4 of
164 tasks change verdict.

**Reading: `draft-mtp` n=2 is NOT bit-exact, but it IS quality-neutral in
aggregate.** Those are different claims and only the second one is needed to
ship. What is lost is *bit-reproducibility*, which matters for byte-identity
claims, response caching and A/B forensics — not for capability. The shipping
config's certified HumanEval number is therefore 93.29 %, and it also does the
same work 1.23× faster in wall-clock.

**Program consequence:** the 27B flagship ships MTP n=2 and its n=2 losslessness
was never directly proven either (the kernel offensive proved byte-identity
across *builds*, not across *spec modes*). This check should be re-run against
the 27B before any byte-identity claim is made for it.

---

## STAGE 3 — THE MAINSTREAM-RIG QUESTION (32 GiB RAM)

Our box has 47 GiB, which is not the configuration the v2 flagship question is
about. The shipping cell was re-run inside a cgroup v2 scope with
`MemoryMax=<cap>` and **`MemorySwapMax=0`** (this box has a 96 GiB swapfile that
would otherwise silently absorb overflow and disguise a "does not fit" as a
"fits but crawls"). Page cache dropped before each cell. **The cap was verified
bound by reading `memory.max` out of the server's own cgroup** — a first attempt
that did not verify this was discarded.

| cap | `memory.max` verified | load | decode t/s | prefill t/s | cgroup reclaim events (`max`) | OOM kills |
|---|---|---:|---:|---:|---:|---:|
| 32 GiB | 34359738368 | 24 s | **46.58** | 129.5 | **0** | 0 |
| 24 GiB | 25769803776 | 23 s | 46.52 | 128.4 | **0** | 0 |
| 20 GiB | 21474836480 | 24 s | 44.05 | 123.2 | 2 618 | 0 |
| 16 GiB | 17179869184 | 25 s | 44.92 | 127.1 | 4 563 | 0 |

**Answer: yes, comfortably — a 12 GiB + 32 GiB rig runs the shipping config at
full speed with literally zero memory-pressure events.** The tighter caps are
the positive control that proves the instrument bites: reclaim events appear
only at 20 GiB and below.

The reason it is this easy is worth stating, because it is counter-intuitive
given a 21.3 GiB file: **at ncm26 the host-side working set is only ~14.7 GiB**
(26 CPU layers × 463 MiB + 2.9 GiB non-expert). The 15 GPU-resident layers are
read once at load and then never touched from host memory again, so their page
cache is freely evictable. Independently measured server RSS at this cell was
14.5 GiB — the prediction and the measurement agree.

---

## VERDICT TABLE — 35B-A3B vs the certified 27B

Both arms on the same RTX 3060, same harnesses, corrected protocol, thinking off.

| axis | 27B UD-IQ3_XXS (certified) | **35B-A3B UD-Q4_K_XL** | verdict |
|---|---:|---:|---|
| **HumanEval-164** | 92.68 % (152/164) | **93.29 %** (153/164) | tie (McNemar p=1.0) |
| **Edit-format compliance** | 94.12 % (32/34) | **94.12 %** (32/34) | tie (p=1.0) |
| polyglot pass@2 | 26.47 % (9/34) | 41.18 % (14/34) | +14.7 pts, **not significant** (p=0.267) |
| **decode t/s (best config)** | 34.39 | **45.58** | **+32.5 %** |
| decode t/s @ ~13-14 K depth | 36.00 | **44.07** | **+22.4 %** |
| **prefill t/s (shallow)** | 277 | **128** | **−54 %** |
| prefill t/s @ depth | 471 | 365 | −22 % |
| MTP speedup | 1.72× | 1.23× | 27B speculates far better |
| draft acceptance | 0.781 | 0.860 | 35B higher, but see below |
| VRAM peak under load | 11 958 MiB | **10 524 MiB** | 1.4 GiB more headroom |
| host RAM working set | n/a (fully GPU-resident) | 14.5 GiB | 27B needs no host RAM |
| fits 12 GiB + 32 GiB rig | yes (trivially) | **yes, 0 reclaim events** | both pass |
| model file on disk | 10.18 GiB | 21.28 GiB | 27B ~2.1x cheaper |
| licence | Apache-2.0 | **Apache-2.0, clean** | both fine |

### The honest shape of the result

**The 35B-A3B is ~32 % faster at decode and statistically identical in quality,
and it costs 54 % of the prefill throughput and 2× the disk.**

The decode win is real, reproduced, and mechanistically explained. The prefill
loss is equally real: 26 layers of expert GEMM run on 10 Zen2 cores, and prompt
processing is compute-bound where decode is bandwidth-bound. For an agentic
coding product — long files, long system prompts, repeated re-reads — prefill is
not a secondary axis. At depth the gap narrows to −22 %, which is the regime
that actually matters most, but it does not close.

Note also that **MTP pays much less on this model than on the 27B (1.23× vs
1.72×)**, and this is structural rather than a tuning miss: verifying k drafted
tokens through a 256-expert MoE touches up to 8k distinct experts, so
speculation's usual trick — amortising one weight-streaming pass over several
tokens — largely fails for sparse MoE. Most of the 35B's speed advantage comes
from having 3 B active parameters, not from its MTP head.

---

## RECOMMENDATION

**Adopt as the revv v2 SPEED TIER now; promote to v2 flagship only after the
prefill question is settled.**

Rationale:
1. It clears every gate that killed its predecessors: Apache-2.0 with no
   AI-coding-product clause (Flash-Next's killer), runs on our existing binary
   with no rebuild, MTP head present and working.
2. Quality is certified at parity with the current flagship on both instruments
   of record — including edit-format compliance, which is the instrument that
   justified the flagship quant in the first place.
3. 45.6 t/s is the fastest decode this program has ever certified on this card,
   and it is +32.5 % over a flagship whose config levers we had declared
   exhausted at 34–36. The drafter-v2 verdict closed the 55 t/s road at the
   model layer; this reopens it from a different direction — and the linear
   placement law says a 24 GiB card would reach ~69 t/s no-spec on this model.
4. It fits the mainstream rig with zero memory pressure.

Why not outright flagship yet:
- **Prefill is 54 % lower shallow / 22 % lower at depth.** No instrument in this
  campaign measures end-to-end agentic wall-clock, which is the quantity that
  actually decides this. That is the one experiment that should precede promotion.
- The pass@2 improvement that would justify "better model, not just faster" is
  underpowered at n=34 and must not be quoted as a win.
- Only one quant rung was certified (disk-bound), so we do not know the
  speed/quality curve for this model the way we know the 27B's.

### Caveats
- **Single box, DDR4-3200, 10 cores.** The slope of the placement law is DDR4
  expert streaming; a DDR5 rig will sit on a shallower slope and post higher
  numbers at every ncm. These figures are a floor for modern hardware, not a
  central estimate.
- Cross-platform caveat on the edit-compliance arm only (27B reference was
  L40S); HumanEval is same-box on both arms.
- `draft-mtp` is not bit-reproducible (see above). Quality-neutral, but no
  byte-identity claims may be made for any MTP config.
- The model is multimodal; `--mmproj` and `-np > 1` are unsupported with MTP
  upstream. We certified text-only, single-slot, which is our lane — but a
  multi-user deployment cannot use MTP today.

## Open questions / next steps
1. **End-to-end agentic wall-clock, 35B-A3B vs 27B** — the promotion gate.
2. Second quant rung (UD-Q3_K_XL ~16 GiB): would seat ~11 more layers on the
   card. The placement law predicts ≈ +5.3 ms/token saved → **~55-58 t/s**.
   Blocked on `/data` free space, not on time or money.
3. Re-run `mtp_identity.py` against the 27B — its n=2 losslessness is assumed,
   not proven.
4. A 16 GiB or 24 GiB card is the highest-leverage hardware change this program
   could make for this model: the curve is VRAM-capacity-bound, not
   bandwidth-bound, on the 3060.

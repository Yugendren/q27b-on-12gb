# 27B flagship: MTP losslessness re-check

Run 2026-09-03 on the ollama box (RTX 3060 12 GiB / sm_86, Ryzen 5 3600,
47 GiB DDR4). Cost: $0.

**Why this exists.** `results/cert_35ba3b.md` found that `--spec-type draft-mtp`
is *not* bit-exact on Qwen3.6-35B-A3B (4/5 greedy probes diverged from no-spec,
with a 5/5 no-spec restart control proving the pipeline itself is
deterministic), and closed with an explicit program consequence:

> The 27B flagship ships MTP n=2 and its n=2 losslessness was never directly
> proven either (the kernel offensive proved byte-identity across *builds*, not
> across *spec modes*). This check should be re-run against the 27B before any
> byte-identity claim is made for it.

This is that run. FINDINGS also carried a second, older flag — that q8_0 KV was
implicated in the deep-draft (n>=3) divergence seen on the 27B — so the f16 KV
arm is included to settle it.

Artifacts: `results/mtp27b/` (9 `ident_*.json` captures, 6 `cmp_*.txt`
comparisons, `tokdiv.txt`, the two `ab_he50_*.jsonl/json` HumanEval arms,
`ab_mcnemar_he50.txt`, per-arm server logs and VRAM readings).
Drivers: `/data/scratch/mtp27b/{ident.sh,ab.sh,tokdiv.py}` and
`/data/scratch/mtp27b/mtp_identity.py`.

---

## Protocol — identical to the cert's, by construction

The probe driver is a **byte-identical copy** of the cert agent's
`mtp_identity.py` (md5 `27ba5cfcc2b9bc70076a2072c975d6e0` on both files, verified
after copying). 5 greedy probes — 2 code (B-tree, LSM-tree), 1 structured JSON,
1 prose, 1 short HumanEval-style — at `temperature=0, top_k=1, seed=1234`,
512 max tokens, `enable_thinking=false`, compared by SHA-256 of the completion.

Server protocol = **the 27B's own certified shipping protocol**
(`results/speed_recert.md`): speed-lineage binary
`/data/projects/llama.cpp/build/bin/llama-server` build 10718 / `9efa1595e`,
`-ngl 99 -fa on --chat-template-file models/templates/chat_template.jinja
--parallel 1 --no-warmup`, model `Qwen3.8-27B-UD-IQ3_XXS.gguf`, q8_0 KV,
c=16384. GPU guard < 500 MiB enforced before all 9 server launches.

**Forced deviation, documented.** The f16 KV arm runs at **c=8192**, not 16384.
IQ3_XXS + MTP n=2 + f16 KV at c=16384 is a *certified OOM* on this card
(`speed_recert.tsv`, cell `c16k_f16kv_iq3_mtp2` = FAIL_RUN, "CUDA OOM in graph
compute"). c=8192/f16 is a certified-working cell. Its no-spec partner is run at
the same c=8192/f16, so the comparison stays paired.

**Thinking tripwire: clean.** All 45 probe responses across all 9 arms recorded
`reasoning_content` length **0** and no `<think>` leak.

Speeds observed match the certified table (no-spec 20.0–20.1 t/s vs certified
20.00; MTP n=2 c16k 35.2 t/s mean over the 5 probes vs 34.39 on the single
B-tree cell), confirming these are the certified configs and not some other cell.

---

## The controls — the 27B pipeline is bit-reproducible

| control | identical | reading |
|---|---|---|
| no-spec **A vs B** (same server, back-to-back) | **5/5** | pipeline is deterministic |
| no-spec **A vs C** (**server restart**, same flags) | **5/5** | deterministic across restarts |

Greedy no-spec decoding on the 27B is byte-reproducible both within a server and
across a full restart. Any divergence below is therefore a real property of the
code path, not ambient nondeterminism.

---

## RESULT — `draft-mtp` is NOT bit-exact on the 27B either

| comparison | KV | ctx | n | identical | verdict |
|---|---|---:|---:|---|---|
| no-spec vs **draft-mtp n=2** — **THE SHIPPING CONFIG** | q8_0 | 16384 | 2 | **1/5** | **diverges** |
| no-spec vs draft-mtp n=2 | q8_0 | 8192 | 2 | **1/5** | diverges |
| no-spec vs draft-mtp n=2 | **f16** | 8192 | 2 | **2/5** | diverges |
| no-spec vs draft-mtp n=3 | q8_0 | 8192 | 3 | **2/5** | diverges |

**The 27B's "certified lossless" MTP claim is FALSE and must be withdrawn.**

The failure pattern reproduces the 35B's almost exactly: 1/5 identical at n=2,
and the single probe that agrees (`humaneval_like`) is the only one short enough
to stop on its own (201 tokens, `finish=stop`); every 512-token probe that runs
to the length cap differs. On the 35B the same probe was the same lone survivor.

### It is worse than "not bit-exact" — it is *deterministically* different

| comparison | identical | reading |
|---|---|---|
| **no-spec**, c=16384 vs c=8192 (q8_0 KV) | **5/5** | no-spec is context-size-invariant |
| **draft-mtp n=2**, c=16384 vs c=8192 (q8_0 KV) | **5/5** | **MTP is equally deterministic** |
| no-spec, q8_0 vs f16 KV (c=8192) | 2/5 | KV quant *is* a real arithmetic change |
| draft-mtp n=2, q8_0 vs f16 KV (c=8192) | 1/5 | same |

The `draft-mtp` path is perfectly reproducible — it just reproduces a
**different answer** from the one the target model produces on its own, and it
does so identically every time, across context sizes and across restarts. This
rules out a race, a scheduling artifact or an uninitialised buffer, and points at
a systematic defect in the verify step (or at the MTP layer perturbing target
state). The q8_0-vs-f16 rows are the positive control that the instrument detects
genuine numerical changes when they exist.

### Where it diverges — token-level quantification

`tokdiv.py` re-tokenises both completions with the model's own tokenizer
(llama-server `/tokenize`) and reports the first differing **token** index, not
just the first differing character.

| comparison | probe | 1st div token | 1st div char | % into completion |
|---|---|---:|---:|---:|
| q8/c16k n=2 (**shipping**) | code_btree | 214 | 761 | 41.8 % |
| q8/c16k n=2 | code_lsm | **17** | 60 | 3.3 % |
| q8/c16k n=2 | struct_json | 436 | 1014 | 85.2 % |
| q8/c16k n=2 | prose_ssd | **18** | 96 | 3.5 % |
| q8/c16k n=2 | humaneval_like | — | — | identical |
| q8/c8k n=3 | code_btree | 458 | 1552 | 89.5 % |
| q8/c8k n=3 | code_lsm | 17 | 60 | 3.3 % |
| q8/c8k n=3 | prose_ssd | 18 | 96 | 3.5 % |
| f16/c8k n=2 | code_btree | 154 | 598 | 30.1 % |
| f16/c8k n=2 | code_lsm | 156 | 641 | 30.5 % |
| f16/c8k n=2 | prose_ssd | 18 | 96 | 3.5 % |

Across the 10 diverging probes: first-divergence token index **min 17, median
154, max 458** — i.e. **3.3 % to 89.5 %** of the way into the completion.
Divergence is not a rare late-sequence tail event; it can land on the 17th
generated token. Once it lands the two completions are different texts, not
different spellings: `prose_ssd` diverges at token 18 into "…due to the
fundamental **constraint that NAND flash memory can only be erased**…" (no-spec)
versus "…due to the fundamental **mismatch between the logical block size**…"
(MTP), and both continue coherently from there.

### The q8_0 KV flag is retired

The standing suspicion was that q8_0 KV was implicated in the deep-draft
divergence. **It is not.** f16 KV diverges too (3/5 probes at n=2), and the
`prose_ssd` probe diverges at exactly token 18 under q8_0 *and* f16, at n=2 *and*
n=3. The KV quant is not the mechanism; the `draft-mtp` path is. This closes
that recert flag as a wrong lead.

### n=3 is not worse than n=2

n=3 scored 2/5 identical vs n=2's 1/5 — inside the noise of a 5-probe
instrument, and in the *opposite* direction to the original "n>=3 is the
dangerous one" framing. The honest reading is that **n=2 was never safe**; the
original flag simply caught the effect first at higher n because deeper drafts
diverge more visibly, not because n=2 was exact.

---

## Quality: is it neutral, like it was on the 35B?

Bit-exactness and quality-neutrality are different claims, and only the second
one is needed to ship. Run per the **quality** protocol of record
(`platform_arm.sh` `he` phase): quality-arm binary build 169 / `daef7b6`,
`-ngl 99 -fa on --jinja --no-warmup -c 8192 -np 1 -ctk q8_0 -ctv q8_0`, the
byte-identical `reliability.py` at `--max-attempts 1`, greedy, thinking off.
The only thing that differs between the two arms is `--spec-type`.

| arm | HumanEval-50 pass@1 | 95 % CI | mean completion tokens | tripwire | wall |
|---|---:|---|---:|---|---:|
| 27B IQ3_XXS, `--spec-type none` | 48/50 = **96.00 %** | [86.5, 98.9] | 165.7 | ok | 446.5 s |
| 27B IQ3_XXS, **`draft-mtp` n=2 (shipping)** | 49/50 = **98.00 %** | [89.5, 99.7] | 169.2 | ok | **253.8 s** |

**McNemar exact, paired over all 50 tasks: b=0, c=1, p = 1.0 —
indistinguishable.** Exactly one task changes verdict (HumanEval/32, which MTP
passes and no-spec fails), and it changes in MTP's favour.

MTP also does the same work **1.76× faster in wall-clock** (446.5 s → 253.8 s),
which is the shipping config's whole reason for existing.

*Caveat on power:* n=50, and 48/50 vs 49/50 on a saturated instrument can only
exclude large effects. This run rules out a quality *collapse*, which is what the
non-exactness finding put at risk. It is not, and is not quoted as, proof of
exact quality equality.

---

## VERDICT

**The 27B flagship's `draft-mtp` n=2 is NOT bit-exact with no-spec greedy
decoding. It IS quality-neutral on HumanEval-50 (p = 1.0). The shipping config
stands; the "certified lossless" label does not.**

This is the same verdict, with the same shape and the same 1/5 signature, as
`cert_35ba3b.md` reached for the 35B-A3B. Two different models, two different
architectures (dense 27B vs sparse-MoE 35B-A3B), two different binaries
(build 10718 / `9efa1595e` here, `daef7b687` there) — and the same defect.
**This is a llama.cpp `draft-mtp` implementation property, not a model,
quant, KV or build artifact.**

### What must change

1. **Withdraw the "certified lossless" claim for the 27B flagship.** Speculative
   decoding is supposed to be mathematically exact; ours is not. No
   byte-identity claim may be made for any MTP config on either model.
2. **Response caching keyed on (prompt, greedy params) is unsafe across spec
   modes.** A cache populated by a no-spec run and read by an MTP run (or vice
   versa) will serve text the current config would not produce.
3. **A/B forensics must hold `--spec-type` fixed.** Any experiment that compares
   two builds, quants or prompts while spec mode varies is confounded — the
   divergence lands as early as token 17.
4. **Quality certification must be re-run with MTP on, not assumed from the
   no-spec run.** The cert did this for the 35B; this run does it for the 27B at
   n=50. A full 164 with MTP on is the remaining gap for the 27B.
5. The kernel offensive's byte-identity result stands but is **narrower than it
   reads**: it proved identity across *builds at fixed spec mode*, not across
   spec modes.

### Open questions
1. **Root-cause the verify step.** The determinism result (5/5 across context
   sizes, both arms) makes this tractable: it is a fixed, reproducible
   difference, so a single instrumented run comparing target logits at the
   accept/reject decision would localise it. Worth an upstream issue with the
   `prose_ssd` probe as a 18-token repro.
2. **Full HumanEval-164 with MTP n=2 on the 27B** — closes item 4 above.
3. Whether the defect is in `draft-mtp` specifically or in llama.cpp's generic
   speculative path: re-running this probe with a *draft-model* spec config
   (`--spec-type draft-model` with the separate `mtp-Qwen3.8-27B-Q4_0.gguf`)
   would separate the two. If the generic path is exact and only `draft-mtp`
   is not, that is a much sharper upstream bug report.

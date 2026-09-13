# STAGE 2 — certification re-audits under the corrected protocol

Two entries on `results/platform_arm/VERDICT.md`'s re-audit list, re-taken.

**Headline: the ASCII vocab prune's −4.9-point penalty was 100% truncation
artifact. Under the corrected protocol the prune is quality-neutral — and all
nine of its legacy losses come back.**

Raw records: `results/platform_arm/reliability_IQ3_XXS_ASCII_nospec_3060.jsonl`
(+ `_n1.json` summary), verdict `results/platform_arm/prune_reverdict.json`,
run logs `results/stage2/`.

---

## 2a. ASCII-prune re-verdict — HumanEval-164, thinking OFF, greedy, no-spec

### Protocol

Run through the **byte-identical code path** that produced the reference,
by adding one model alias to `platform_arm/platform_arm.sh` and changing
nothing else:

```
server  /data/scratch/bin-unpatched/llama-server -ngl 99 -fa on --jinja
        --no-warmup -c 8192 -np 1 -ctk q8_0 -ctv q8_0 --spec-type none
client  platform_arm/harness/reliability.py --max-attempts 1 --limit 164
        (greedy: temperature 0, top_p 1.0, seed 42;
         chat_template_kwargs.enable_thinking=false)
```

Reference arm: `results/platform_arm/reliability_IQ3_XXS_nospec_3060.jsonl`,
unchanged, per-task. Paired significance: exact two-sided McNemar, the same
implementation as `platform_arm/legacy_cmp.py`.

### Result

| protocol | flagship IQ3_XXS | ASCII-pruned | delta | McNemar (b / c) | p |
|---|---:|---:|---:|---:|---:|
| **legacy** (both arms thinking ON) | 135/164 = **82.3%** | 127/164 = **77.4%** | **−4.9 pts** | 9 / 1 | **0.0215** |
| **corrected** (thinking OFF) | 152/164 = **92.68%** | **153/164 = 93.29%** | **+0.61 pts** | **0 / 1** | **1.0** |

* Discordance under the corrected protocol: **b = 0, c = 1** — there is not a
  single task the flagship solves and the prune does not. The one-task edge to
  the prune is noise (p = 1.0), so the correct statement is
  **statistically indistinguishable**, not "the prune is better".
* **All 9 of the prune's legacy-protocol losses are recovered**:
  HumanEval/24, 37, 81, 87, 92, 117, 123, 126, 146 — every one of them passes
  under the corrected protocol. That is the whole −4.9 points, and it was
  reasoning-budget truncation, exactly the mechanism VERDICT.md identified.
* Format instrument: `fence_unclosed` 1 (prune) vs 2 (flagship) — the
  adherence channel, which is the only place quantization damage was still
  visible above ~2.9 bpw, also shows nothing.
* Tripwire: mean completion **217.8** tok/task (prune) and **219.8**
  (flagship), against the 350 limit. Both pass.

### Why the legacy comparison went wrong

The legacy protocol left thinking ON, so both arms spent their 1,024-token
budget reasoning and *both* were truncation-limited. That does not cancel
between arms: whichever build's reasoning happens to run slightly longer loses
tasks it can actually solve. Here the prune was the unlucky one by 9 tasks and
the difference cleared p<0.05. **A relative A/B is not protected against a
protocol defect that both arms share.** New standing rule implied: relative
build comparisons must be re-taken, not re-labelled, when the protocol changes.

### Verdict

**The −4.9-point ASCII-prune penalty is RETIRED.** The prune costs no
measurable HumanEval quality (Δ = +0.6 pts, p = 1.0, b = 0) while removing
120,373 of 248,320 vocab entries and **555.3 MB / 5.08%** of the file
(`Qwen3.8-27B-UD-IQ3_XXS-ASCII.gguf.report.json`). It is now an adoptable
capacity lever rather than a rejected one, and the retention table should be
updated accordingly.

Scope limit, stated plainly: this measures HumanEval only. The prune removes
all non-ASCII tokens, so any workload with non-ASCII input or output is
**out of scope by construction** and this verdict says nothing about it.

---

## 2b. canary2 quick pass — does the gauge still detect anything?

`harness/canary2.py` already sent `enable_thinking=false` (line 100), so unlike
`eval_quality.py` it was never exposed to the thinking bug — the re-audit
confirms that rather than fixing it. So the gauge was asked two sharper
questions instead.

Re-run: pruned and flagship, `--jinja` plus the canonical chat template,
against the existing `results/anchor_reference.json` (Q8_0 anchor).

### (1) Reproducibility — the gauge is exactly deterministic

| build | probes | responses identical to the earlier run | match verdicts identical |
|---|---:|---:|---:|
| pruned | 54 | **54 / 54** | 54 / 54 |
| flagship | 54 | **54 / 54** | 54 / 54 |

Every probe reproduced **bit-for-bit** against `results/canary2_div_*.json`.
The gauge has no run-to-run noise; whatever it reports is a property of the
build.

### (2) Resolution — the reported gap is two probes, and it is not significant

| build | anchor exact-match | divergence points | token flip rate | canary_score |
|---|---:|---:|---:|---:|
| flagship IQ3_XXS | 40 / 54 | 28.8 | 0.3168 | 0.6842 |
| ASCII-pruned | 38 / 54 | 30.8 | 0.3198 | 0.6842 |

Paired McNemar on the per-probe agreement vector: **b = 2, c = 0, p = 0.5**.
The two builds give **identical responses on 77.8%** of probes, and the entire
headline separation is two probes, `c2-06` and `c3-06`. The composite
`canary_score` is *identical* to 16 digits for both builds.

**Instrument verdict:** with 54 probes and exact McNemar, the smallest
one-sided effect the gauge can call at p<0.05 is **6 discordant probes = 11.1
percentage points**. It physically cannot resolve the 2-5 point differences
these certifications care about. Its agreement with 2a's "no damage" finding is
therefore **concordant but not confirmatory** — it agrees by being underpowered,
not by being sensitive.

Two further limits worth recording: the Q8 anchor itself scores 0.30 on the
C4 algorithmic-judgment battery, so a block of probes is "everyone fails" and
carries no information; and the anchor reference was taken at `-ngl 24`
(partial offload), which is fine for a quality reference but means the anchor
is not on the same execution path as the candidates.

### Verdict

**The canary2 divergence gauge is reproducible and honest but underpowered.**
It detects no residual damage in the ASCII prune, consistent with 2a — but its
resolution floor is ~11 points, so that null is weak evidence. **Do not quote
canary2 divergence-point differences under ~11 points as findings.** If the
gauge is to stay in the certification suite it needs its probe count raised by
roughly 5-10× (or its scoring changed to use the token-flip rate, which is
continuous, rather than the binary exact-match, which is what throws away the
power).

---

## Combined Stage 2 verdict

1. **Prune penalty: REAL? NO — truncation artifact.** −4.9 pts → +0.6 pts,
   p 0.0215 → 1.0, 9/9 losses recovered. The prune ships as a free 5.08%
   capacity saving on ASCII workloads.
2. **canary2: still deterministic, still blind below ~11 points.** It confirms
   nothing that 2a did not already show, and it should not be trusted to catch
   the small format/adherence damage that is now the certification instrument
   of record.

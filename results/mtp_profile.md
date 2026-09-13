# MTP round profiling — where the ~18 ms/round goes

Answers the diagnostic in `12-PREDICTIVE-KV-AND-SELFDRAFT.md` §8.2: theory at α = 0.786, γ = 2, **c = 0.03** predicts **2.27×**; we measure **1.665×**, leaving **~18 ms/round unexplained**. Suspected cause: hybrid-GDN recurrent-state snapshotting at every draft position (TreeWY 2608.20961).

Rig: ollama box, RTX 3060 12 GB + Ryzen 5 3600. llama.cpp build **10718 / 9efa1595e**. `Qwen3.8-27B-UD-IQ3_XXS.gguf`, `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 8192`, project chat template, `--parallel 1`, batch 1.

Workload: one fixed greedy code prompt (a persistent on-disk B-tree module), `temperature=0, top_k=1, seed=1234, max_tokens=400, cache_prompt=false`. Each cell = 1 discarded warm-up request + 4 measured requests, mean of the 4. All counters are llama-server's own (`draft acceptance = A (X accepted / Y generated), mean len = L`).

Derived quantities, both exact given those counters:

```
rounds     = n_accepted / (mean_len - 1)      # mean_len = 1 + n_acc/n_verify_steps
depth_eff  = n_generated / rounds             # drafted tokens actually issued per round
T_round_ms = mean_len * 1000 / decode_t_s     # decode_t_s counts ACCEPTED tokens
```

---

## 0. Verdicts

| # | Finding |
|---|---|
| 1 | **The ~18 ms/round is 100% a per-drafted-token charge, and it is ~9x the byte-ratio prediction.** `T_round(n) = 47.8 ms + 13.4 ms x n`, R^2 = 0.9987 over n = 1..5. The intercept equals the unspeculated token time (49.3 ms), so the measured `c` is **0.273, not 0.030**. |
| 2 | **Batch verification is free; drafting is not.** T0/T_base = 0.970 rules out candidate cause (ii) (wider verify batch). The residual is **11.95 ms per drafted token**, which reproduces the whole gap at every depth (102-116% explained). |
| 3 | **At most ~36% of the slope is the MTP head's own forward.** The draft-forward-free `ngram-mod` control carries a slope of 8.60 ms/token vs MTP's 13.43. So **>=64% (8.6 ms/token) is verify-batch + GDN recurrent-state snapshot/rollback + per-round graph work that NO drafter avoids** - consistent with the TreeWY/Bole diagnosis, and it means a better drafter cannot recover it. |
| 4 | **`--spec-draft-p-min` recovers ~1%, not more.** Best cell `n=4, p_min=0.50` -> 36.02 t/s vs 35.67 for plain `n=3` (+1.0%). It works exactly as the cost model predicts (depth_eff 4.00 -> 3.57, acceptance 0.619 -> 0.715) but the two effects nearly cancel. |
| 5 | **Stacking `ngram-mod` under `draft-mtp` recovers exactly nothing on novel text.** On 5 distinct prompts x 400 tokens, `ngram-mod` **never issued a single draft** (zero `draft acceptance` lines; 20.16 t/s vs a 20.22 t/s baseline), and `draft-mtp,ngram-mod` = 32.04 t/s vs `draft-mtp` alone 32.07 t/s. |
| 6 | **The 86-134 t/s ngram-mod figure in the program files is a repeated-prompt artefact.** Re-firing one prompt 5x gives 147.6 t/s at mean accepted length **56.7 tokens/round** - it is replaying its own previous answer. This closes the item `12-...` §11 called *the most important outstanding number in the program*, and the answer is **1.00x**. |
| 7 | **MTP draft depth is VRAM-capped at n<=5** on 12 GB at 8K ctx: each extra depth costs ~152 MiB, n=5 sits at 11,968 MiB, and **n=7 OOMs in CUDA graph allocation**. |

---

## 1. Raw sweep

| config | n_max | p_min | decode t/s | speedup | accept | mean len | depth_eff | **T_round ms** | VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nospec | - | - | 20.30 | 1.000× | — | 1.00 | 0 | **49.3** | 11132 |
| mtp_n1 | 1 | - | 31.77 | 1.565× | 0.937 | 1.94 | 1.00 | **61.1** | 11354 |
| mtp_n2 | 2 | - | 35.26 | 1.737× | 0.818 | 2.64 | 2.01 | **74.9** | 11510 |
| mtp_n3 | 3 | - | 35.67 | 1.757× | 0.724 | 3.17 | 3.00 | **88.7** | 11664 |
| mtp_n4 | 4 | - | 34.65 | 1.707× | 0.619 | 3.47 | 3.99 | **100.1** | 11816 |
| mtp_n5 | 5 | - | 35.22 | 1.735× | 0.614 | 4.06 | 4.99 | **115.3** | 11968 |
| mtp_n4_pmin030 | 4 | 0.30 | 33.05 | 1.628× | 0.592 | 3.31 | 3.89 | **100.0** | 11816 |
| mtp_n4_pmin050 | 4 | 0.50 | 36.02 | 1.775× | 0.715 | 3.55 | 3.57 | **98.5** | 11816 |
| mtp_n4_pmin075 | 4 | 0.75 | 35.32 | 1.740× | 0.891 | 3.57 | 2.89 | **101.1** | 11816 |
| mtp_n2_pmin050 | 2 | 0.50 | 32.95 | 1.623× | 0.792 | 2.52 | 1.92 | **76.5** | 11510 |
| ngram_only | 4 | - | 147.63 | 7.273× | 0.886 | 56.67 | 63.00 | **383.8** | 10722 |
| mtp_n2_ngram | 2 | - | 143.57 | 7.072× | 0.846 | 22.70 | 25.64 | **158.1** | 11538 |
| ngram_mtp_n2 | 2 | - | 143.75 | 7.082× | 0.846 | 22.70 | 25.64 | **157.9** | 11538 |
| ngram_n1 | 1 | - | 148.18 | 7.299× | 0.886 | 56.67 | 63.00 | **382.4** | 10722 |
| ngmod_d1 | 1 | - | 26.46 | 1.304× | 1.000 | 2.00 | 1.00 | **75.6** | - |
| ngmod_d2 | 2 | - | 36.91 | 1.818× | 1.000 | 3.00 | 2.00 | **81.3** | - |
| ngmod_d4 | 4 | - | 43.65 | 2.150× | 0.988 | 4.95 | 4.00 | **113.5** | - |
| ngmod_d8 | 8 | - | 66.97 | 3.299× | 0.994 | 8.95 | 7.98 | **133.6** | - |
| mtp_n7 | 7 | - | FAIL | | | | | | |
| novel_nospec | - | - | 20.22 | 0.996× | — | 1.00 | 0 | **49.5** | - |
| novel_mtp_n3 | - | - | 32.07 | 1.580× | 0.616 | 2.84 | 2.99 | **88.6** | - |
| novel_ngram | - | - | 20.16 | 0.993× | — | 1.00 | 0 | **49.6** | - |
| novel_mtp3_ngram | - | - | 32.04 | 1.579× | 0.616 | 2.84 | 2.99 | **88.6** | - |
| novel_mtp_n4_pmin050 | - | - | 33.08 | 1.630× | 0.677 | 3.30 | 3.40 | **99.8** | - |

Baseline **20.30 t/s = 49.26 ms/token**, one unspeculated target forward. Our best MTP config reproduces the program's 1.665× within noise.

## 2. The per-round cost model

A speculation round is one verify forward plus whatever the drafting machinery adds. Fit `T_round(n) = T0 + k·n` across the MTP depth sweep:

```
T_round(n) = 47.79 ms  +  13.43 ms × n        R² = 0.9987   (n = 1, 2, 3, 4, 5)
```

**The fit is essentially exact (R² = 0.9987) and the intercept lands on the baseline.** T0 = 47.79 ms vs a measured unspeculated token of 49.26 ms — a ratio of **0.970**.

Two conclusions fall straight out:

1. **The fixed part of a round is free.** Verifying a batch of n+1 tokens costs the same as decoding 1 — as it must on a bandwidth-bound rig where the batch dimension is free. Candidate cause (ii) in §8.2 (*'batch-3 verification costing more than one forward'*) is **ruled out**: it would show up as T0 > T_base and it does not.
2. **The whole gap is a per-drafted-token charge of k = 13.43 ms.**

Expressed as Leviathan's `c` (draft forward / target forward):

| quantity | value |
|---|---:|
| c predicted from bytes read (§8.2) | **0.030** |
| c measured = k / T_base = 13.43 / 49.26 | **0.273** |
| ratio | **9.1×** |

Each drafted token costs **13.43 ms** where the byte ratio says it should cost **1.48 ms**. The residual is **11.95 ms per drafted token**.

### Does that account for the gap?

| n | modelled T_round at c=0.03 | measured T_round | gap | (k − 0.03·T_base)·n | fraction explained |
|---:|---:|---:|---:|---:|---:|
| 1 | 50.7 | 61.1 | **10.3 ms** | 12.0 ms | **116%** |
| 2 | 52.2 | 74.9 | **22.7 ms** | 24.0 ms | **106%** |
| 3 | 53.7 | 88.7 | **35.0 ms** | 35.8 ms | **102%** |
| 4 | 55.2 | 100.1 | **45.0 ms** | 47.7 ms | **106%** |
| 5 | 56.6 | 115.3 | **58.6 ms** | 59.6 ms | **102%** |

> **At the shipping depth n=2 the unexplained overhead is 22.7 ms/round** — the program's estimate was ~18 ms/round against a slightly faster baseline (21.09 vs our 20.30 t/s). **The per-drafted-token charge explains ~100% of it, at every depth from n=1 to n=5.** There is no residual constant-per-round term left to find.

## 3. Splitting the slope — the draft-forward-free control

`ngram-mod` issues drafts from a lookup over the context and runs **no draft-model forward at all**, but goes through the identical verify batch, the identical GDN recurrent-state snapshot/rollback path (`llm_arch_supports_rs_rollback`, PR #22400) and the identical per-round graph build. Pinning its depth with `--spec-ngram-mod-n-min/-n-max` and fitting the same model isolates everything that is *not* the MTP head.

*(Note: `ngram-mod` **ignores** `--spec-draft-n-max`; its own defaults are n-min 48 / n-max 64. That is why the `ngram_only` / `ngram_n1` rows in §1 show depth_eff ≈ 63 regardless of the flag.)*

```
ngram-mod (no draft forward):  T_round(n) = 68.78 ms + 8.60 ms × n   R² = 0.9359
draft-mtp:                     T_round(n) = 47.79 ms + 13.43 ms × n   R² = 0.9987
```

| slope component | ms per drafted token | share of the MTP slope |
|---|---:|---:|
| **not** the MTP forward — GDN state snapshot/rollback + wider verify + graph | **8.60** | **64%** |
| the MTP head's own forward | **4.83** | **36%** |
| (implied MTP-forward-only c) | — | 0.098 |

Note the control's *intercept*: 68.8 ms vs the MTP fit's 47.8 ms and the 49.3 ms baseline. **`ngram-mod` carries ~20 ms of fixed per-round cost of its own** — the `--spec-ngram-mod-n-match 24` lookup scan over the context — which is why `ngmod_d1` (26.5 t/s) is *slower* than `mtp_n1` (31.8 t/s) at the same draft depth. That cost sits in the intercept, not the slope, so it does not contaminate the split above; but it is another reason not to treat the n-gram tier as free.

Caveat, stated plainly: the ngram control runs at high acceptance on a repeated prompt, so it performs proportionally **fewer rollbacks** than MTP does. Its slope is therefore a **lower bound** on the state-management cost, which makes the split above conservative in the direction that matters.

## 4. Mitigations measurable without code changes

### 4a. `--spec-draft-p-min` (early-exit drafting)

| config | p_min | depth_eff | accept | decode t/s | vs best plain MTP |
|---|---:|---:|---:|---:|---:|
| mtp_n2_pmin050 | 0.50 | 1.92 | 0.792 | **32.95** | -7.6% |
| mtp_n4_pmin030 | 0.30 | 3.89 | 0.592 | **33.05** | -7.4% |
| mtp_n4_pmin050 | 0.50 | 3.57 | 0.715 | **36.02** | +1.0% |
| mtp_n4_pmin075 | 0.75 | 2.89 | 0.891 | **35.32** | -1.0% |

Best plain MTP in this sweep: **mtp_n3 at 35.67 t/s**.

`p_min` does exactly what the cost model says it should: it truncates the draft (depth_eff falls below n_max) and raises acceptance, trading `k·n` against `mean_len`. The effect is real but small, because the two move together — you buy back slope by giving up accepted tokens.

### 4b. Stacking `ngram-mod` under `draft-mtp`

| config | decode t/s | accept | mean len | depth_eff |
|---|---:|---:|---:|---:|
| ngram_only | **147.63** | 0.886 | 56.67 | 63.00 |
| mtp_n2_ngram | **143.57** | 0.846 | 22.70 | 25.64 |
| ngram_mtp_n2 | **143.75** | 0.846 | 22.70 | 25.64 |
| ngram_n1 | **148.18** | 0.886 | 56.67 | 63.00 |

⚠️ **These numbers are an artefact and must not be quoted as speculation results.** Each config fired the *same* prompt 5×, so `ngram-mod` replays its own previous answer: mean accepted length 56.7 tokens per round at depth 63. This is the warm/repeated-prompt hazard the program already flagged (`12-…` §11). §5 re-measures it honestly.

Order does not matter: `draft-mtp,ngram-mod` and `ngram-mod,draft-mtp` land within 0.15% of each other — consistent with `common/speculative.h`'s documented chaining semantics (first successful drafter wins the round).

## 5. Novel-generation re-measure (5 distinct prompts, one request each)

Nothing here can be replayed from a previous answer. This is the honest number for the ngram tiers.

| config | decode t/s | speedup | accept | mean len | depth_eff | T_round ms |
|---|---:|---:|---:|---:|---:|---:|
| nospec | **20.22** | 1.000× | — | 1.00 | 0 | 49.5 |
| mtp_n3 | **32.07** | 1.586× | 0.616 | 2.84 | 2.99 | 88.6 |
| ngram | **20.16** | 0.997× | — | 1.00 | 0 | 49.6 |
| mtp3_ngram | **32.04** | 1.585× | 0.616 | 2.84 | 2.99 | 88.6 |
| mtp_n4_pmin050 | **33.08** | 1.636× | 0.677 | 3.30 | 3.40 | 99.8 |

## 6. What this means for the program

**The `c = 0.03` line in `12-PREDICTIVE-KV-AND-SELFDRAFT.md` §8.2 is a correct byte-count and a wrong cost model.** Bytes read are the right first-order model for the *target's* forward pass, which is a single large bandwidth-bound sweep. They are the wrong model for the *draft* step, which is a sequence of tiny dependent kernel launches over a 0.3 GB head plus per-position recurrent-state bookkeeping across 48 GDN layers. At batch 1 that work is latency-bound, not bandwidth-bound, so it does not shrink with the byte count.

Three consequences, in order of how much they should change behaviour:

1. **Re-price every speculation proposal in the program against c = 0.27, not c = 0.03.** The `c` sweep table in §8.1 already contains our answer: at alpha = 0.786 it gives 2.27x for c = 0.03 and 1.72x for c = 0.20. **Our measured 1.66-1.78x is the c ~ 0.2-0.3 row, not the c = 0.03 row.** The theory was never wrong; the constant fed into it was.

   §8.2's conclusion - *'driving c to zero would buy 6%'* - therefore inverts. Holding the measured acceptance at each depth fixed and only changing the per-drafted-token cost in the fitted model:

   | n | measured t/s | if k = 1.48 ms (bytes only) | if only the runtime term (8.60) were removed |
   |---:|---:|---:|---:|
   | 2 | 35.26 | **52.0** (+48%) | **46.0** (+30%) |
   | 3 | 35.67 | **60.6** (+70%) | **50.8** (+42%) |
   | 4 | 34.65 | **64.6** (+86%) | **51.7** (+49%) |

   §8.2 guessed *'if (i) is even half of it, that is 5-7 t/s sitting on the floor.'* The measured figure is **15-17 t/s** sitting on the floor for the runtime term alone.

2. **But §8.2's headline conclusion survives, for a different reason.** At least 64% of that overhead is charged by the *runtime*, not by the drafter, so it is invariant to which drafter you plug in. Swapping in a cheaper proposer cannot collect it; only removing the per-draft-position state work can. That is exactly what TreeWY (2608.20961) and Bole exist to do, and neither is in llama.cpp. **This is now a measured motivation for that port rather than a suspicion.**

3. **Retire the ngram-mod tier from the roadmap as a decode accelerator.** Finding 5/6 above is unambiguous: on novel generation it fires zero drafts and costs nothing and buys nothing. The §11 claim that we *'already ship the bottom rung of the cascade'* is not supported at batch 1 on novel text. It may still earn its keep on genuinely repetitive workloads (re-editing a file, regenerating with a small diff), which is a different and narrower claim than the one on file.

### Recommended shipping config (unchanged in shape, sharpened in value)

`--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.50` measured **36.02 t/s (1.775x)** on the repeated prompt and **33.08 t/s (1.636x)** on novel text, versus 35.67 / 32.07 for plain `n=3`. That is a **+1.0% / +3.2%** free win from two existing flags, at identical VRAM. Worth taking; not worth celebrating.

### Not done, and why

The optional instrumented-llama.cpp build (timers inside the `draft-mtp` loop) was **not** run. The `ngram-mod` depth control in §3 delivers the same draft-vs-not-draft split from stock binaries, and it does so without a 20-40 minute CUDA rebuild competing for the box. What the control does *not* give is the split *within* the 8.60 ms/token non-draft term - state-snapshot vs state-rollback vs graph rebuild. That specific three-way split still needs source timers, and §3's caveat (the control under-rolls-back) is the reason it is worth doing before committing to a TreeWY port.


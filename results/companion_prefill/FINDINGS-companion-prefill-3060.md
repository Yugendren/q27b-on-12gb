# FINDING — the flagship prefill constant is ~500 t/s, and prefill is very nearly LINEAR (2026-09-02)

Box: ollama, RTX 3060 12 GB (12044 MiB usable), AMD Ryzen 5 3600 host exposed as
10 vCPU / 47 GB to the guest. Model
`models/unsloth-q27b/Qwen3.8-27B-UD-IQ3_XXS.gguf` (10.17 GiB, 27.32 B params).
`llama-bench` build `daef7b687 (10712)`, `llama-server` build `9efa1595e (10718)`.
Raw: `results/3060_prefill_curve.md`.

## 1. The constant — the 225 / 400 / 519 conflict is settled

```
llama-bench -m Qwen3.8-27B-UD-IQ3_XXS.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
            -p 512,2048,4096,8192,16384,32768 -n 0 -r 3
```

| prefill length | t/s | s to prefill |
|---:|---:|---:|
| 512 | 525.83 ± 7.48 | 0.97 |
| 2048 | 523.96 ± 0.40 | 3.91 |
| 4096 | 514.48 ± 0.24 | 7.96 |
| 8192 | 503.17 ± 0.20 | 16.28 |
| 16384 | 482.34 ± 0.14 | 33.97 |
| 32768 | 445.16 ± 0.07 | 73.61 |

**Verdict: `519 t/s` (`12 §12`) is right; `225 t/s` (`04 §7.1`) and `25.63 t/s`
are wrong** — 25.63 t/s is a *decode* number that has been quoted as prefill
somewhere in the chain. `400 t/s ("maybe … resident")` is a decent guess for the
16–32K band. The honest single sentence is: **445–525 t/s across 512–32768
tokens, ~500 t/s at the 8K working point.**

Confirmed independently, server-side, during the admission replay (a different
binary, a different code path, real prompts):

| source | tokens | ms | t/s |
|---|---:|---:|---:|
| llama-bench pp8192 | 8192 | — | 503.2 |
| llama-server `timings.prompt_n/prompt_ms` | 7224 | 14318.8 | 504.5 |
| llama-bench pp16384 | 16384 | — | 482.3 |
| llama-server `timings` | 14224 | 29231.9 | 486.6 |

Two methods, two binaries, agreement to <1%. The constant is not in doubt any more.

## 2. The more important result: prefill on this model is essentially LINEAR

Throughput falls only **15.4%** from 512 → 32768 tokens (525.8 → 445.2 t/s), i.e.
a 64× increase in context costs 15% in rate. A dense transformer's quadratic
attention term would show far more decay by 32K.

This is the hybrid architecture showing up in the cost model: most of
Qwen3.8-Flash-Next's layers are gated-DeltaNet (linear attention), so prefill
cost is ~O(n) with a weak O(n²) correction from the few full-attention layers.

**This contradicts `DESIGN.md §2.3`,** which argues the quadratic term works in
admission's favour — i.e. that halving the context more-than-halves prefill
time. Measured, the payoff is only *slightly* super-linear:

| keep rate at 32K | tokens | prefill s | speedup vs full |
|---:|---:|---:|---:|
| 1.00 | 32768 | 73.61 | 1.00× |
| 0.50 | 16384 | 33.97 | 2.17× |
| 0.25 | 8192 | 16.28 | 4.52× |
| 0.10 | 3277 | ~6.3 | ~11.7× |

So a keep rate of *k* buys about **1/k × (a 5–15% bonus)**, not the
super-linear windfall the design doc assumed. Companion prefill has to win on
keep rate alone; the curvature is not going to rescue it. Update `DESIGN.md §2.3`.

## 3. Correction: `-c 40960` is NOT the max context with MTP on this build

`RUNBOOK_3060.md §2a` and our config notes state "-c 40960 is our measured max
context with MTP". On build `9efa1595e (10718)` that is false — it OOMs, and so
does `-c 32768`. Measured ceiling probe (`results/3060_ctx_ceiling.md`), each row
a fresh server on an otherwise-idle card:

| config | -c | result | VRAM at health |
|---|---:|---|---:|
| MTP (`--spec-type draft-mtp --spec-draft-n-max 2`) | 40960 | **OOM** (compute pp buffers) | - |
| MTP | 32768 | **OOM** (`failed to create MTP context`) | - |
| MTP | 24576 | OK | 11836 MiB |
| MTP | 20480 | OK | 11660 MiB |
| MTP | 16384 | OK | 11484 MiB |
| no MTP | 40960 | OK | 11600 MiB |
| no MTP | 32768 | OK | 11288 MiB |

**`-c 40960` is the max WITHOUT MTP. With MTP the ceiling is between 24576 and
32768** — MTP's own compute buffers cost roughly 8K of context. The shipping
config as written in the runbook does not start on a 12 GB card at this build.
The admission eval below therefore ran **without MTP at `-c 32768`**; MTP is a
decode-side optimisation and does not affect the prefill numbers that are the
subject of this measurement.

## 4. `c_s` — the admission ceiling. Gate PASSED, with room to spare

`c_s = P_target / P_scorer`. `RUNBOOK_3060.md §3` gate: proceed if `c_s < 0.3`.
`results/3060_c_s.txt`, `results/3060_c_s_embed-{gpu,cpu}.json`.

bge-small-en-v1.5 Q8_0 (33 M params, 36.7 MB), mean pooling, 120-word chunks:

| placement | 2001 tok | 6001 tok | 12001 tok | 24000 tok |
|---|---:|---:|---:|---:|
| **GPU (`-ngl 99`)** | 23,960 t/s | 53,940 t/s | 55,593 t/s | **56,823 t/s** |
| **CPU (`-ngl 0`, 10 threads)** | 8,470 t/s | 14,343 t/s | 14,761 t/s | **14,822 t/s** |

Against a target running 482 t/s at 16K and 445 t/s at 32K:

| scorer placement | `c_s` at 16K | `c_s` at 24K | ceiling | gate |
|---|---:|---:|---:|---|
| embed on GPU | 0.0087 | 0.0078 | ~128× | **PASS** (`< 0.3`) |
| embed on CPU | 0.033 | 0.030 | ~33× | **PASS** (`< 0.3`) |

Two things worth recording:

1. **The embedder is not the bottleneck and never will be.** Even on DDR4 CPU it
   is 30× cheaper than the prefill it replaces. The scorer-cost term in the net
   speedup is ~3% at worst.
2. **It costs zero VRAM to run it on CPU** (`c_s = 0.030` vs `0.008`), which
   matters enormously here because the flagship already holds 11.3 of 12.0 GB.
   The GPU placement is a 4× cheaper scorer that we cannot afford to house; the
   CPU placement is free and still passes the gate by 10×. **Ship the embedder on
   CPU.**

This closes `RUNBOOK_3060.md §8`'s open question about the CPU-placed scorer, at
least for the embedding backend: **the prediction that DDR4 would be as dead as
Apple Silicon's 94 t/s 0.5B is wrong for a 33 M embedder** — a 0.5B LM scorer
would indeed be dead, but the winning backends do not need one.

## 5. Trap re-confirmed: the flagship returns EMPTY content under its shipping template

`RUNBOOK_3060.md §4b` warns that a reasoning template plus a small
`--max-tokens` yields empty answers that look exactly like quantization damage.
Reproduced on the real 27B, `--max-tokens 48`, prompt "what is the secret access
code?" (`think_probe.py`):

| request `chat_template_kwargs` | finish | predicted_n | content | reasoning len |
|---|---|---:|---|---:|
| *(none — server default `reasoning_effort: medium`)* | `length` | 48 | `''` | 198 |
| `{"reasoning_effort": "low"}` | `length` | 48 | `''` | 216 |
| `{"enable_thinking": false}` | `stop` | 8 | `'BLUE-HERON-42'` | 0 |
| `{"reasoning_effort": "none"}` | `stop` | 8 | `'BLUE-HERON-42'` | 0 |

Run in the shipping configuration this eval would have scored **0/45 on every
backend including the full-context reference**, and the obvious (wrong)
conclusion would have been "IQ3_XXS is too damaged to do retrieval". Note that
`reasoning_effort: low` does **not** help — only `enable_thinking: false` or
`reasoning_effort: none` do. All admission results below use
`--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'`.

---

# 6. THE CURVE — admission is VIABLE, and the winning scorer is the free one

Battery: 45 tasks (5 families x 3 lengths x 3 depths x 1), backends
`last, random, lexical, embed, hybrid` x keep rates `0.5, 0.35, 0.25, 0.15, 0.10`,
plus a full-context `none@1.0` reference per task. **1170 rows, 0 errors, 0
cache-hit rows.** Greedy, `--max-tokens 48`,
`--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'`.
Two-phase path: `admit-only` (embedder on GPU, target down) then `replay`
(target on GPU, no scorer). Raw: `results_3060.jsonl`, `CURVE_3060.md`,
`curve_3060.csv`.

Deviations from `RUNBOOK_3060.md` §4, all forced and all recorded: depths
`0,50,100` not `0,10,25,50,75,90,100` and `--n-per-cell 1` not 3 (the full
150-task battery is ~12 h of GPU); `--repeat 1` (greedy decode is
deterministic — verified, `agree_with_ref` is stable); no MTP (see §3);
`lm_ppl`/`draft_expand` skipped as instructed (proven net-negative).

## 6.1 Headline (all families, n=45 per cell)

Reference `none@1.0`: **93.3% accuracy, 36.20 s prefill.**

| backend | keep | accuracy | needle_retained | scorer_ms | net TTFT ms | speedup |
|---|---:|---:|---:|---:|---:|---:|
| **lexical** | 0.10 | 91.1% | 97.8% | 689 | 6240 | **5.38x** |
| **lexical** | 0.15 | 95.6% | 97.8% | 691 | 7814 | **4.32x** |
| **lexical** | 0.25 | **97.8%** | 97.8% | 688 | 10996 | **3.11x** |
| **lexical** | 0.35 | **97.8%** | 97.8% | 688 | 14232 | 2.43x |
| lexical | 0.50 | 95.6% | 97.8% | 687 | 19209 | 1.83x |
| hybrid | 0.10 | 93.3% | 97.8% | 1013 | 6775 | 4.99x |
| hybrid | 0.25 | **97.8%** | 97.8% | 1013 | 11983 | 2.89x |
| hybrid | 0.50 | 93.3% | 97.8% | 1012 | 20684 | 1.71x |
| embed | 0.10 | 82.2% | 82.2% | 1009 | 6778 | 4.98x |
| embed | 0.25 | 93.3% | 91.1% | 1008 | 11992 | 2.89x |
| `last` (control) | 0.25 | **35.6%** | 37.8% | 684 | 10975 | 3.12x |
| `random` (control) | 0.25 | **57.8%** | 57.8% | 686 | 11758 | 2.94x |
| `none` (reference) | 1.00 | 93.3% | 97.8% | 0 | 36205 | 1.00x |

## 6.2 Against the decision rule (`DESIGN.md §4.3`, fixed before the numbers)

> Viable if some backend retains >=90% of full-context accuracy at an effective
> keep rate <=0.5 and a net speedup >=1.5x (scorer cost included), **and beats
> both the `lexical` and `last` controls**.

- Accuracy >=90% of reference (>=84.0%): **yes** — lexical and hybrid clear it at
  every keep rate down to 0.10, and at 0.25/0.35 they *exceed* full context
  (97.8% vs 93.3%).
- Effective keep rate <=0.5: **yes** — measured `kr_eff` 0.14–0.60.
- Net speedup >=1.5x with scorer cost included: **yes** — 1.71x to 5.38x.
- Beats `last`: **yes, overwhelmingly** — 97.8% vs 35.6% at keep 0.25.
- Beats `lexical`: **NO.**

**Verdict: the MECHANISM is strongly viable; the COMPANION is not.** Nothing
beats `lexical`, which is a free, model-free, CPU-only scorer. The ranking is
`lexical >= hybrid > embed >> random > last`, and `hybrid` is just
`lexical + embed` being dragged down by `embed`. The embedder costs **+325 ms
per request** (1013 vs 688 ms scorer time) and buys **negative** accuracy.

This is outcome (2) in `RUNBOOK_3060.md §7` — "scorer-limited" — but with an
inverted conclusion. The next experiment is not a better scorer. **The next
experiment is deleting the scorer**: ship `lexical`, drop the embedding server,
and reclaim the 325 ms and the whole 8073 service.

## 6.3 The result that actually matters: admission gets BETTER with context

Per context length, `lexical` vs the reference and the `last` control:

| ctx | reference acc | lexical best cell | speedup there | `last` at same keep |
|---:|---:|---|---:|---:|
| 8 000 | 86.7% | 93.3% @ keep 0.25 | 2.57x | 33.3% |
| 16 000 | 100.0% | 100.0% @ keep 0.15 | 4.33x | 40.0% |
| 32 000 | 93.3% | **100.0% @ keep 0.10** | **7.01x** | 46.7% |

At 32K, lexical admission keeping ~13.6% of the tokens answers **100%** of the
battery — better than the 93.3% the model manages with the whole prompt — at a
**7.0x** prefill speedup, scorer cost included. The payoff grows with context
because the scorer's ~690 ms fixed cost amortises while the prefill it removes
grows linearly (§2).

**Pruning improves accuracy.** At 8K and 32K the pruned runs beat the
full-context reference. The 27B is being *distracted* by material the scorer
correctly discards. Note this is the opposite of `04 §3.7`'s prediction that a
hybrid's concentrated retrieval heads make it MORE fragile to context surgery —
measured, this hybrid is more robust to context surgery than to context bloat.
`needle_retained` is 97.8% while `acc|needle_retained` is 100.0% at keep 0.25:
when the evidence survives, the answer is right. The system is scorer-recall
limited, not model limited.

## 6.4 What this does NOT show — read before quoting the 7x

1. **The battery is synthetic NIAH, which structurally flatters `lexical`.** The
   needle is planted with vocabulary drawn from the question, so term overlap is
   the strongest available signal *by construction*. That an embedder cannot beat
   BM25-style lexical overlap here is close to a tautology. It is evidence that
   the embedder is not needed on *this* battery; it is NOT evidence about real
   agentic traffic, where the question and the evidence often share no terms.
   **`RUNBOOK_3060.md §6`'s replay-real-logged-traffic step was not run** and is
   the single highest-value follow-up.
2. **No `--tail-tokens` ablation was run** (`RUNBOOK_3060.md §5`). The `last`
   control collapsing to 35.6% is strong indirect evidence that the effect is not
   "just keep the last N tokens", but the ablation is the direct test and it is
   still outstanding.
3. **`n-per-cell 1`, `repeat 1`.** Each cell is 45 tasks; a single task flip
   moves a cell by 2.2pp. Differences under ~5pp between lexical and hybrid are
   inside the noise — which is itself the point: they are indistinguishable.
4. **Effective keep rate != nominal.** Protected zones (system prompt, last user
   turn, tail) mean `keep_rate 0.10` actually bills 0.136–0.242 of the tokens.
   All speedups above are against `keep_rate_effective_measured`, i.e. what the
   target actually prefilled — not admission's own estimate.

## 6.5 Recommendation

Ship `--backend lexical --keep-rate 0.25` as the default: 97.8% accuracy (vs
93.3% full context), 3.11x mean net speedup, 7.0x at 32K, zero VRAM, zero extra
model. Keep `embed`/`hybrid` behind a flag pending a real-traffic replay, which
is the only measurement that can justify the embedding server's existence.

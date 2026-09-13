# CPU drafter block latency — is DFlash2 shape (b) viable on a Ryzen 5 3600?

`ARCHITECTURE.md` §3.2 budgets **380 ms** for one 8-token draft block and estimates config **b1** at **170–290 ms** from a FLOP/bandwidth model, tagged `[GUESS]` on the GFLOPS figure. §7 of that doc lists this as open item 6: *"Shape (b)'s timing budget is a FLOP model, not a measurement."* This is the measurement.

Box: AMD **Ryzen 5 3600** (Zen 2, 6 cores / 12 threads, AVX2 — **no AVX-512, no native bf16**), 47 GB DDR4. torch **2.13.0+cpu** CPU-only build, batch 1, random weights, `torch.set_grad_enabled(False)`, `set_num_interop_threads(1)`, greedy (temperature 0) selector. Each cell is 5 warm-up iterations then 20–50 timed ones (50 wherever a rep took under ~0.8 s); **median** reported, `reps` recorded in the JSON.

## 0. Verdicts

| # | Finding |
|---|---|
| 1 | **`ARCHITECTURE.md`'s 170–290 ms estimate for b1 is roughly right for the drafter body and only in the best case.** b1's body alone is **198 ms** (bf16, 6 threads) even with the context fully cached, and **2163 ms** if the context is recomputed each block. |
| 2 | **The doc's core call — "the vocabulary, not the drafter, makes or breaks the CPU Companion" — is confirmed, but it is not the *dominant* term it predicted.** The full 248k head costs **306 ms**, the ASCII-pruned 128k head **158 ms**, the 32k head **40 ms**. The doc predicted 200–350 ms of compute for the full head; measured is 306 ms, because it is **bandwidth-bound, not compute-bound** (7 rows x 5120 is a skinny GEMV, so it is a 2.5 GB streaming read). |
| 3 | **More threads is worse.** 12 threads is consistently ~35–40% *slower* than 6 on this part — SMT siblings contend for the same FP/load-store units and the workload is already bandwidth-limited. Set `torch.set_num_threads(6)`. |
| 4 | **bf16 on Zen 2 is a trap: it wins where you are bandwidth-bound and loses catastrophically where you are compute-bound.** Zen 2 has no bf16 instructions, so oneDNN up-converts. On the 248k head (pure streaming read) bf16 is 3.3x faster (997 → 306 ms); on the warm body 1.06x (211 → 198 ms); but on the cold-context `fc` GEMM it is **4.8x SLOWER** (447 → 2163 ms). Store weights bf16, compute the large GEMMs in fp32. |
| 5 | **Verdict on shape (b): b1 (662 M) is NOT viable; b2/b3 are, but only with a pruned head and only against the *unaided* deadline.** See §6. |
| 6 | **The deadline in `ARCHITECTURE.md` is the wrong one.** 380 ms is 8 tokens at the *unaided* GPU rate. The flagship already ships with `draft-mtp`, which this session measured at **35.67 t/s** — so the GPU produces 8 tokens in **224 ms**, not 380. A Companion that replaces MTP must beat 224 ms, and it must also out-draft MTP's acceptance. |

## 1. Config reproduction check

| config | shape | measured params | `ARCHITECTURE.md` §3 | layers | fc | selector | convs |
|---|---|---:|---:|---:|---:|---:|---:|
| **b1** | 3L / 8192 / 3 taps / rank-128 | **661.9 M** | 661.9 M | 534.8 M | 78.6 M | 9.0 M | 39.4 M |
| **b2** | 2L / 4096 / 2 taps / rank-64 | **314.0 M** | 314.0 M | 230.7 M | 52.4 M | 4.5 M | 26.3 M |
| **b3** | 1L / 4096 / 2 taps / rank-64 | **185.5 M** | 185.5 M | 115.4 M | 52.4 M | 4.5 M | 13.1 M |

Every bucket reproduces the doc's sizing table to the decimal, so what is timed below is what the doc costed. (hidden 5120 pinned to the target per §2.1(A); heads 32 q / 8 kv / head_dim 128; block_size 8 → **7 draft slots** per block; 32k-pruned selector codebooks.)

## 2. Drafter body — `block_forward`, median ms

Two context regimes, and the difference between them is the single biggest engineering lever in this table:

* **`ctx=512` (cold)** — the cache-free path `block_forward` actually implements: `fc(target_hidden)` over the whole context (a `[512, 15360] x [15360, 5120]` GEMM for 3 taps) plus every layer's context K/V, recomputed for every block.
* **`ctx=8` (warm)** — proxy for a properly KV-cached inference path, where `fc` and the per-layer `k_ctx`/`v_ctx` are retained and only the newly confirmed tokens are projected. The term this proxy drops is the SDPA scan over the cached context: at ctx=512 that is 34 MFLOP and ~13 MB of cache reads, i.e. well under a millisecond. **This is the number a real deployment would see.**

| config | dtype | threads | ctx=8 | ctx=512 | cold/warm |
|---|---|---:|---:|---:|---:|
| b1 | fp32 | 6 | 211.4 | 447.3 | 2.1x |
| b1 | fp32 | 12 | 291.2 | 540.2 | 1.9x |
| b1 | bf16 | 6 | 198.5 | 2162.6 | 10.9x |
| b1 | bf16 | 12 | 274.3 | 2380.7 | 8.7x |
| b2 | fp32 | 6 | 90.2 | 241.3 | 2.7x |
| b2 | fp32 | 12 | 128.7 | 292.9 | 2.3x |
| b2 | bf16 | 6 | 98.2 | 1365.0 | 13.9x |
| b2 | bf16 | 12 | 144.7 | 1529.6 | 10.6x |
| b3 | fp32 | 6 | 67.6 | 180.6 | 2.7x |
| b3 | fp32 | 12 | 73.3 | 212.5 | 2.9x |
| b3 | bf16 | 6 | 56.6 | 1147.5 | 20.3x |
| b3 | bf16 | 12 | 80.6 | 1265.9 | 15.7x |

> **Recomputing the context costs 2–3x the whole rest of the block.** Any real Companion must cache `fc(target_hidden)` and the context K/V incrementally; `config.py` already anticipates this with the `freeze_fc` flag (*"enables the fc-projected activation cache"*). Everything below uses the warm number.

## 3. Output head — the vocabulary question

`F.linear(draft_hidden[1, 7, 5120], W[V, 5120])` — the target's `lm_head` applied to the 7 draft slots.

| vocab | dtype | weight | 6 threads | 12 threads | implied GB/s @6t |
|---|---|---:|---:|---:|---:|
| 248,320 (full) | fp32 | 5.09 GB | **997.0** | 1064.5 | 5.1 |
| 248,320 (full) | bf16 | 2.54 GB | **306.3** | 338.7 | 8.3 |
| 128,000 (ASCII-pruned) | fp32 | 2.62 GB | **503.8** | 556.6 | 5.2 |
| 128,000 (ASCII-pruned) | bf16 | 1.31 GB | **157.6** | 174.8 | 8.3 |
| 32,768 (pruned) | fp32 | 0.67 GB | **129.7** | 135.9 | 5.2 |
| 32,768 (pruned) | bf16 | 0.34 GB | **40.4** | 45.4 | 8.3 |

The head is a pure streaming read of the weight matrix — 7 output rows is far too few to amortise anything — so its cost tracks bytes, not FLOPs, and scales linearly in vocab. **The ASCII prune (248,320 → 128,000 rows) halves it exactly, as designed.**

## 4. Candidate selector

`topk(logits, 16)` over V, then the 7-step sequential greedy bigram walk over the rank-r codebooks. The 7 steps are strictly sequential (each predecessor feeds the next), so this is latency, not throughput.

| config | dtype | vocab | 6 threads | 12 threads |
|---|---|---|---:|---:|
| b1 | fp32 | 248,320 (full) | **1.6** | 3.5 |
| b1 | fp32 | 32,768 (pruned) | **0.7** | 1.2 |
| b1 | bf16 | 248,320 (full) | **1.6** | 3.9 |
| b1 | bf16 | 32,768 (pruned) | **0.8** | 1.9 |
| b2 | fp32 | 248,320 (full) | **1.5** | 3.2 |
| b2 | fp32 | 32,768 (pruned) | **0.7** | 1.1 |
| b2 | bf16 | 248,320 (full) | **1.6** | 3.8 |
| b2 | bf16 | 32,768 (pruned) | **0.8** | 1.7 |
| b3 | fp32 | 248,320 (full) | **1.5** | 3.1 |
| b3 | fp32 | 32,768 (pruned) | **0.7** | 1.1 |
| b3 | bf16 | 248,320 (full) | **1.6** | 3.8 |
| b3 | bf16 | 32,768 (pruned) | **0.8** | 1.8 |

Negligible either way — the top-k dominates and it is a single pass over the logits. Not a design constraint.

## 5. End-to-end block latency

total = warm body + head + selector, all medians at **6 threads** (the faster setting everywhere).

| config | dtype | vocab | body | head | select | **TOTAL** | vs 380 ms (doc) | vs 224 ms (MTP-on) |
|---|---|---|---:|---:|---:|---:|---|---|
| b1 | fp32 | 248,320 (full) | 211 | 997 | 1.6 | **1210 ms** | **3.18x** MISSES | **5.39x** MISSES |
| b1 | fp32 | 128,000 (ASCII-pruned) | 211 | 504 | 1.6 | **717 ms** | **1.89x** MISSES | **3.20x** MISSES |
| b1 | fp32 | 32,768 (pruned) | 211 | 130 | 0.7 | **342 ms** | **0.90x** FITS | **1.52x** MISSES |
| b1 | bf16 | 248,320 (full) | 198 | 306 | 1.6 | **506 ms** | **1.33x** MISSES | **2.26x** MISSES |
| b1 | bf16 | 128,000 (ASCII-pruned) | 198 | 158 | 1.6 | **358 ms** | **0.94x** FITS | **1.59x** MISSES |
| b1 | bf16 | 32,768 (pruned) | 198 | 40 | 0.8 | **240 ms** | **0.63x** FITS | **1.07x** MISSES |
| b2 | fp32 | 248,320 (full) | 90 | 997 | 1.5 | **1089 ms** | **2.87x** MISSES | **4.85x** MISSES |
| b2 | fp32 | 128,000 (ASCII-pruned) | 90 | 504 | 1.5 | **596 ms** | **1.57x** MISSES | **2.65x** MISSES |
| b2 | fp32 | 32,768 (pruned) | 90 | 130 | 0.7 | **221 ms** | **0.58x** FITS | **0.98x** FITS |
| b2 | bf16 | 248,320 (full) | 98 | 306 | 1.6 | **406 ms** | **1.07x** MISSES | **1.81x** MISSES |
| b2 | bf16 | 128,000 (ASCII-pruned) | 98 | 158 | 1.6 | **257 ms** | **0.68x** FITS | **1.15x** MISSES |
| b2 | bf16 | 32,768 (pruned) | 98 | 40 | 0.8 | **139 ms** | **0.37x** FITS | **0.62x** FITS |
| b3 | fp32 | 248,320 (full) | 68 | 997 | 1.5 | **1066 ms** | **2.81x** MISSES | **4.75x** MISSES |
| b3 | fp32 | 128,000 (ASCII-pruned) | 68 | 504 | 1.5 | **573 ms** | **1.51x** MISSES | **2.55x** MISSES |
| b3 | fp32 | 32,768 (pruned) | 68 | 130 | 0.7 | **198 ms** | **0.52x** FITS | **0.88x** FITS |
| b3 | bf16 | 248,320 (full) | 57 | 306 | 1.6 | **364 ms** | **0.96x** FITS | **1.62x** MISSES |
| b3 | bf16 | 128,000 (ASCII-pruned) | 57 | 158 | 1.6 | **216 ms** | **0.57x** FITS | **0.96x** FITS |
| b3 | bf16 | 32,768 (pruned) | 57 | 40 | 0.8 | **98 ms** | **0.26x** FITS | **0.44x** FITS |

### Vocab-head share of the block

| config (bf16) | head share @248k | @128k ASCII | @32k |
|---|---:|---:|---:|
| b1 | 60% | 44% | 17% |
| b2 | 75% | 61% | 29% |
| b3 | 84% | 73% | 41% |

## 6. Verdict

### 6.1 The doc's estimate was accurate; its recommendation was not

`ARCHITECTURE.md` §3.2 predicted **"b1 with a pruned 32k head lands at roughly 170–290 ms per block"**. Measured: **240 ms** — inside the predicted range, near the middle. The FLOP model was good.

Two of its component assumptions were wrong in offsetting directions, and that matters for reusing the model:

| doc assumption | measured | note |
|---|---|---|
| lm_head full vocab: *"~65 ms traffic + 200–350 ms compute — the dominant term"* | **306 ms bf16**, essentially all traffic | right total, wrong mechanism. It is a GEMV: 7 output rows cannot amortise a 2.5 GB weight read |
| DDR4 *"~40 GB/s"* | **8.3 GB/s bf16 / 5.2 GB/s fp32** achieved | 5x optimistic. Use 8.3 GB/s for any future vocab arithmetic on this box |
| drafter forward *"~100–200 ms at 50–100 GFLOPS AVX2"* | **198 ms** warm (b1) | in range, at the pessimistic end (~46 GFLOPS effective) |

But §3.3's recommendation — *"Train b1 (3L / 8192 / 3 taps / rank 128) as the primary"* — does not survive. §3.2's own hedge does: *"b2/b3 (314 M / 185 M) are the safe variants and should be benchmarked first."* **They are, and they should be.**

### 6.2 The deadline that actually binds

The 380 ms figure is 8 tokens at the *unaided* GPU rate. The flagship does not run unaided — it ships `draft-mtp`. Using this session's companion measurement (`results/mtp_profile.md`):

| deadline | source | ms for 8 tokens |
|---|---|---:|
| `ARCHITECTURE.md` §3.2 | 21.1 t/s, UD-Q2_K_XL, no spec | 380 |
| measured, no spec | 20.30 t/s, UD-IQ3_XXS | 394 |
| **measured, as shipped** | **35.67 t/s, `draft-mtp n=3`** | **224** |

A CPU Companion for shape (b) *replaces* the MTP drafter (both occupy the drafting slot), so the bar it must clear is the third row.

### 6.3 The bar restated as an acceptance requirement

Latency alone is not the test — a pipelined Companion drafts block k+1 while the GPU verifies block k, so steady-state round time is `max(CPU_block, GPU_verify_round)`. From `mtp_profile.md` §3, a verify round at depth 7 with **no** draft-model forward on the GPU costs `47.79 + 7 × 8.60 = **108 ms**`. To beat the shipped 35.67 t/s the Companion must then average `L > 35.67 × max(CPU_block, 108) / 1000` accepted tokens per block, out of the **7 draft slots** a block_size-8 DFlash2 emits:

| config (bf16, 32k head, 6 threads) | CPU block | steady-state round | required mean accepted len | feasible? |
|---|---:|---:|---:|---|
| b1 | 240 ms | 240 ms | **8.55 / 7** | **NO** — exceeds the 7 available slots |
| b2 | 139 ms | 139 ms | **4.97 / 7** | **yes** — needs 71% of 7 slots |
| b3 | 98 ms | 108 ms | **3.85 / 7** | **yes** — needs 55% of 7 slots |

*(Assumes a CPU-drafted round pays the same 8.60 ms/token server-side state-management cost that `mtp_profile.md` §3 measured for the GPU path. llama.cpp's `-devd none` CPU-draft path goes through the same verify and recurrent-state machinery, so this transfers, but it is an assumption, not a measurement.)*

### 6.4 Bottom line

**Shape (b) is viable on a Ryzen 5 3600 — at b2 or b3, in bf16, with a pruned head, and only if the context is cached.** Concretely:

* **b1 (662 M) is dead.** Even at its best (bf16 + 32k head, cached context) it is 240 ms, which forces a 240 ms steady-state round and demands 8.6 accepted tokens from 7 slots. **There is no acceptance rate that makes b1 win.** Do not train it for shape (b).
* **b3 (185 M) is the comfortable one**: 98 ms/block, so the *GPU* is the bottleneck, not the CPU, and it needs 3.85/7 = 55% mean acceptance to beat MTP.
* **b2 (314 M) is the interesting one**: 139 ms/block, needs 4.96/7 = 71%. That is a demanding but not absurd target for a personalised drafter, and b2 has 70% more capacity than b3.
* **The 32k vocab prune is mandatory, not an optimisation.** At the full 248k head, the head alone (306 ms) exceeds the 224 ms deadline for every config. The ASCII prune to 128k is not enough either (158 ms head → b2 at 257 ms, b3 at 216 ms). Only the 32k head leaves room for the drafter.
* **Three free engineering wins**, all measured here: 6 threads not 12 (−35%), cache the context projection (−2.1x to −2.7x in fp32), and keep the large `fc` GEMM in fp32 while storing everything else bf16.

### 6.5 What this does not measure

Random weights, so **acceptance is not measured here — only latency**. §6.3 turns the latency result into the acceptance target that the training run has to hit; whether a personalised b2/b3 can hit 71% / 55% mean block acceptance against a 27B target is exactly the open bet `ARCHITECTURE.md` §3.3 flags as *"not yet tested"*. This measurement narrows the question from *"is shape (b) fast enough?"* to *"can a 314 M drafter accept 5 of 7?"*


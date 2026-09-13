# Turbo4/TurboQuant KV evaluation + headless memory-overclock probe (2026-09-04)

Card: RTX 3060 12GB (sm_86, 12,044 MiB usable), driver 535.309.01, CUDA 12.0,
headless, GPU guard <500 MiB before every launch, all servers killed after.

Drivers: `turbo4_eval.py`, `oc_probe.py`, `analyze_probes.py` (this repo).
Raw: `results/turbo4/{speed,probes,xcheck,moe}.json`, `results/oc_probe/oc_probe.json`.

---

# TASK 1 — Turbo4 / TurboQuant KV

## 1. The fork is real, and it is the one described

| claim | verdict |
|---|---|
| "Buun" fork exists | **YES** — `github.com/spiritbuun` (name field: "Buun"), repo `spiritbuun/buun-llama-cpp`, 818 stars, detached fork, 43 branches, actively pushed |
| adds `-ctk turbo4` | **YES** — `common/arg.cpp:390`; accepted KV strings: `f16 q8_0 turbo8 turbo4 turbo3 turbo3_tcq turbo2 turbo2_tcq turbo1_tcq vbr` |
| TCQ is real | **YES** — self-published paper "Closing the Gap: Trellis-Coded Quantization for KV Cache at 2-3 Bits", authors "buun and Claude (Anthropic)", on HF `datasets/spiritbuun/turboquant-tcq-kv-cache` with 23 codebook `.bin` files. **Not on arXiv.** |
| "Tom Turney" | **DIFFERENT PERSON** — `github.com/TheTom`, whose line is TurboQuant+/Walsh-Hadamard/PolarQuant (`TheTom/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`). Same origin discussion (ggml-org/llama.cpp#20969), different work. |
| upstream status | **NOTHING MERGED.** PR #21089 (CPU TBQ3_0/TBQ4_0) closed unmerged 2026-06-02 on opportunity-cost grounds; #24050, #25352 also closed. |
| academic ancestor | arXiv **2504.19874** "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (Zandieh/Daliri/Hadian/Mirrokni, Google, 2025-04-28). Ships **no code**. All llama.cpp implementations are third-party. |

**Built:** `master` @ `3823c9eb` (2026-09-03), `-DGGML_CUDA=ON -DGGML_CUDA_FA=ON
-DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release`.
Exit 0 in ~50 min (526 flash-attention template instances). Loads and runs our
27B IQ3_XXS and the 35B-A3B Q3_K_XL unmodified.

`turbo4` block layout, from `ggml/src/ggml-common.h:356`: `block_turbo4_0` = 66
bytes per 128 values = **4.125 bpv**, vs q4_0 4.5, q8_0 8.5, f16 16. The
"smaller than q4_0" claim is true by construction, not by measurement.

## 2. The mechanism, read from source before measuring

`ggml/include/ggml-vbr.h:39`, `ggml_vbr_kv_dequant_sides()`, comment verbatim:

> "Turbo tiers **always materialize**; q8_0/bf16 only next to a turbo partner
> (the mixed pair must become (F16,F16)); f16 never does."

So a turbo-typed KV cache is **decoded into an f16 scratch buffer and then fed
to the ordinary f16 flash-attention MMA kernel** (`fattn.cu:1654`,
`k_turbo4_dequant_f16_inv_fwht`, plus the inverse-FWHT rotation). Turbo4 is a
*storage* codec, not a fused-attention codec.

Two predictions follow, both confirmed below:
1. **Turbo4 can never beat f16 on speed** — it is f16 attention plus a decode pass.
2. **The VRAM saving is eroded** by the f16 scratch, so it under-delivers vs its bpv.

## 3. Speed: the matched triple (c=8192, all three fit)

27B IQ3_XXS, MTP n=2, `-fa on`, `-ctxcp 0`, all arms on the **fork binary** so the
build is a constant. 1 discarded warm-up + 3 measured requests. "depth" = 6.5K
filled context.

| KV | bpv | shallow t/s | **depth t/s** | VRAM peak shallow | VRAM peak depth | acceptance (depth) |
|---|---|---|---|---|---|---|
| **f16** | 16 | 35.087 | **39.567** | 11,922 | 11,772 | 0.9925 |
| q8_0 | 8.5 | 34.444 | 38.402 | 11,694 | 11,538 | 0.9925 |
| **turbo4** | 4.125 | 34.810 | **38.882** | 11,556 | 11,426 | 0.9925 |

vs f16 at depth: turbo4 **−1.73%**, q8_0 **−2.94%**.
vs q8_0 at depth: turbo4 **+1.25%** and **−112 MiB**.

## 4. Speed at the shipping context (c=16384, 13.5K depth)

f16 does not run at all here on this binary (see §5).

| KV | shallow t/s | depth t/s | VRAM peak shallow | VRAM peak depth |
|---|---|---|---|---|
| q8_0 | 34.422 | 36.738 | 12,012 | 11,856 |
| **turbo4** | **34.810** | **37.568** | **11,712** | **11,610** |
| q4_0 | 33.965 | — | 11,912 | — |

turbo4 vs q8_0: **+1.13% shallow, +2.26% at depth, −246 MiB**.
turbo4 vs q4_0 (shallow): **+2.49% faster and 0.375 bpv smaller**.

Note the VRAM saving under-delivers exactly as §2 predicts: at 4.125 vs 8.5 bpv
the q8_0→turbo4 KV saving at 16K should be ~350 MiB; measured 246 MiB. The
~100 MiB difference is the f16 dequant scratch.

## 5. Capacity — this is where turbo4 actually pays

| config | result |
|---|---|
| turbo4, **c=32768**, MTP n=2 | **RUNS: 35.001 t/s, 11,896 MiB peak** (marginal — 2 of 4 requests completed before an OOM kill) |
| q8_0, c=32768 | **FAILS at load: out of memory** |
| f16, c=16384 | fails (HTTP 500 on every request) |
| f16, c=12288 | loads at 11,666 MiB, dies on request 2 |
| f16, c=10240 | dies on request 2 |
| f16, c=8192 | **OK** — the f16 ceiling on this binary |

**turbo4 doubles the context ceiling of the shipping config on this card**: 32K
where q8_0 cannot even allocate. That is the real product of this fork.

**⚠ f16 ceiling regressed on the fork: 8192, vs 12288 on our v11 lineage.** The
fork's per-context overhead is higher. This is why the matched triple had to run
at c=8192.

## 6. Build cross-check — is the fork itself a regression?

Identical config (IQ3_XXS, q8_0 KV, c=16384, MTP n=2, `-ctxcp 0`), same protocol:

| binary | decode t/s | VRAM peak | acceptance |
|---|---|---|---|
| llama-server-v11 (speed lineage) | 35.003 | 11,830 | 0.7756 |
| fork 3823c9eb | 34.409 | 12,012 | 0.7707 |

**The fork costs −1.70% speed and +182 MiB before any KV choice is made.** All
§3–§5 comparisons are internally valid (same binary), but adopting turbo4 means
paying this too: turbo4-on-fork 34.810 vs q8_0-on-v11 35.003 is a **net −0.6%**
at the shipping context. The KV win is real; the fork tax cancels it.

## 7. Output neutrality — 10 greedy probes, speculation OFF

`--spec-type none` deliberately: FINDINGS 2026-09-03 established draft-mtp is
NOT bit-exact on this model, so with MTP on a divergence could not be attributed
to the KV codec. c=8192 (f16 reference must fit). Identical probes out of 10:

|  | f16 | q8_0 | turbo4 | q4_0 |
|---|---|---|---|---|
| **f16** | 10 | 6 | **4** | 5 |
| **q8_0** | 6 | 10 | 6 | 5 |
| **turbo4** | 4 | 6 | 10 | 4 |
| **q4_0** | 5 | 5 | 4 | 10 |

**Turbo4 is NOT output-neutral vs f16 — 6/10 probes diverge.** But neither is
q8_0 (4/10) or q4_0 (5/10). turbo4 sits in the same perturbation class as the KV
quants we already ship, marginally worse than q8_0 on this sample.

All divergences inspected are **cosmetic word choice, not errors** — e.g. p6,
f16 "heavier to create and **manage**" vs turbo4 "heavier to create and
**communicate with**". Every ground-truthable probe is correct in every arm
(first 12 primes ✓, OSI layers ✓, 17×23=391 ✓ in all four).

**⚠ n=10 cannot resolve a quality ranking.** 6/10 vs 4/10 is not significant at
this sample size. This probe set says "no pathology, same class as existing KV
quants" and nothing stronger. A quality claim needs the full-164 battery.

## 8. 35B-A3B cell (Q3_K_XL, `--n-cpu-moe 16`, MTP n=2, c=16384)

| KV | decode t/s | prefill t/s | VRAM peak | acceptance |
|---|---|---|---|---|
| q8_0 | 46.500 | 241.7 | 11,602 | 0.699 |
| **turbo4** | **47.257** | 241.8 | **11,518** | 0.744 |

turbo4 **+1.63%, −84 MiB** — same sign and magnitude as the 27B. Both sit below
the certified 48.54 t/s (v11 lineage), consistent with the §6 fork tax.

## 9. VERDICT on the key question

> *Does Turbo4 break our measured KV speed-vs-capacity tradeoff — smaller than
> q4_0 AND faster?*

**Partly. It breaks the quantized-KV branch of the law; it does not break the law.**

- **The law holds.** f16 is still the fastest KV that fits (39.567 t/s at depth,
  ahead of turbo4's 38.882 by 1.7%). Source explains why and it is structural,
  not tunable: turbo tiers *always* materialize to f16 before attention, so
  turbo4 = f16 attention + a decode pass. **f16 wins where it fits — unchanged.**
- **The quantized-KV frontier moved.** Within the KV types that fit at
  c≥12288, turbo4 is simultaneously the **smallest** (4.125 bpv, below q4_0's
  4.5) and the **fastest** (+2.26% over q8_0 and +2.49% over q4_0 at 16K). Our
  old rule "never q4_0 for speed reasons; q8_0 beyond 16K" is superseded:
  **turbo4 dominates both q4_0 and q8_0 on both axes simultaneously.** That is a
  genuine, previously-unavailable Pareto move.
- **The real prize is capacity, not speed.** c=32768 runs on turbo4 at 35 t/s
  where q8_0 OOMs at load. Doubling the context ceiling on a 12 GB card is worth
  far more than the ~2% decode delta.

**Adoption recommendation: NOT YET, and the blocker is the fork, not the codec.**
The fork costs −1.70% and +182 MiB globally (§6), which cancels the KV win at the
current 16K context; it regresses the f16 ceiling from 12288 to 8192; c=32768 was
unstable (an OOM kill mid-cell); and it required `-ctxcp 0` to survive at all.
Turbo4 becomes compelling **only if long context is the goal** — then it is the
only way to 32K on this card and the fork tax is worth paying. Prerequisites
before any ship: full-164 quality battery on turbo4, and a stability soak at
c=32768.

---

# TASK 2 — headless memory-overclock probe

## 1. What is and is not possible headless on driver 535.309.01

Confirmed by direct attempt (passwordless sudo available, no X server):

| lever | works headless? | notes |
|---|---|---|
| `nvidia-smi -pm 1` persistence | **YES** | |
| `nvidia-smi -pl <W>` power limit | **YES** | 170 W default → **190 W max**. Real lever. |
| `nvidia-smi -lgc <MHz>` lock GPU clock | **YES, but only within stock P-states** | max selectable 2145 MHz |
| `nvidia-smi -lmc <MHz>` lock memory clock | **YES, but only stock P-states** | selectable set is exactly {405, 810, 5001, 7301, 7501} |
| `-rgc` / `-rmc` reset | **YES**, verified | |
| **memory clock OFFSET (true VRAM OC)** | **NO** | requires `nvidia-settings` (needs X) — driver 535 exposes no NVML clock-offset API; `nvidia-smi --help` has no `-goc`/offset option on this version |
| GPU clock offset | **NO** | same reason |
| over-voltage | **NO** | |

**⚠ The above-spec silent-clamp trap.** `sudo nvidia-smi -lgc 2400` and
`-lmc 8000` both return `"GPU clocks set to (gpuClkMin 2400, gpuClkMax 2400)"`
and `"All done."` — they look like they worked. Arm E was run specifically to
test this: with lgc 2400 / lmc 8000 requested, measured clocks under load were
**sm 1876 MHz / mem 7318 MHz — identical to the lgc 2145 / lmc 7501 arm**, and
throughput matched to 4 decimal places. **nvidia-smi accepts above-spec values
and silently clamps.** Anything automating this must verify with
`--query-gpu=clocks.sm,clocks.mem` under load, never trust the exit status.

## 2. Measured effect (27B flagship, IQ3_XXS, q8_0 KV)

llama-bench tg64, `-r 3`; server arm = shipping config c=16384 MTP n=2, 4 reqs.

| arm | tg64 d=0 | tg64 d=13000 | server t/s (MTP) | acceptance | sm/power @d13k |
|---|---|---|---|---|---|
| **A stock** (pl 170, no locks) | 22.845 | 20.390 | **35.103** | 0.7756 | 1865 MHz / 161.5 W |
| B pl 190 | 22.959 (+0.50%) | 20.621 (**+1.13%**) | — | — | 1886 MHz / 179.8 W |
| C pl 190 + lgc 2145 | 22.879 (+0.15%) | 20.581 (+0.94%) | **35.586 (+1.38%)** | 0.7756 | 1882 MHz / 179.8 W |
| D C + lmc 7501 | 22.899 (+0.24%) | 20.584 (+0.95%) | 35.564 (+1.31%) | 0.7756 | 1882 MHz / 180.4 W |
| E lgc 2400 + lmc 8000 | 22.900 (+0.24%) | 20.582 (+0.95%) | — | — | 1876 MHz / 180.2 W |

llama-bench stddev was 0.013–0.051 t/s (≈0.1–0.25%), so the ~+1% at depth is
real but small. **Best achievable headless: +1.4% on the shipping config, bought
with +11.8% power limit and +7 °C (77 → 84 °C).**

## 3. Stability

**Acceptance was 0.7756 in every server arm — bit-identical across stock, C and
D. Zero drift, so the >5-point tripwire never came close to firing.** No
corruption, no OOM, no throttle event beyond the normal SW power cap. Locked
clocks on this card at stock P-states are simply not a stability risk, because
they are not an overclock.

## 4. Why the gain is so small

At depth the card sits at ~1880 MHz against a 2145 MHz max **with throttle reason
`0x4` (SW_POWER_CAP) active even at pl 190 W**. The workload is power-limited,
not clock-limited, and it is still power-limited after the 20 W raise — the extra
budget buys 17 MHz. Clock *locking* (`-lgc`) mostly removes ramp latency: at
d=0 the stock arm averaged 1716 MHz vs 1884 MHz locked, and produced **the same
throughput** (22.845 vs 22.879). Ramp is not the bottleneck either.

## 5. Revert — verified

`-rgc`, `-rmc`, `-pl 170` all applied and re-read: power limit 170.00 W, clocks
back to idle 210 MHz / 405 MHz, no locks active, GPU 1 MiB. Persistence mode was
`Disabled` before this run and was restored to `Disabled`.

## 6. What revv can and cannot automate

- **CAN:** `-pl` between 100 and 190 W; `-lgc`/`-lmc` selection among stock
  P-states; `-pm`; all of it reversible with `-rgc`/`-rmc`/`-pl 170`, no X, no reboot.
- **CANNOT:** any true overclock. No memory transfer-rate offset, no core offset,
  no voltage control on driver 535 headless. Reaching those needs either an X
  session (`nvidia-settings` coolbits) or a driver new enough to expose NVML
  clock offsets — neither is available here.
- **MUST:** verify any applied clock by reading `clocks.sm`/`clocks.mem` **under
  load**, because above-spec requests succeed silently and clamp.
- **Worth it?** +1.4% for +12% power and +7 °C is a poor trade on a card that is
  already power-capped. **Recommendation: leave clocks at stock.** This lane is
  closed — not because it is dangerous, but because there is nothing there.

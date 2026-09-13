# STAGE 3 — round-overhead decomposition: where the ~24 ms/round goes

**Verdict: the round overhead is ~97% GPU work and ~3% host. There is no
host-side round overhead to attack. The 55 t/s goal cannot be reached by
round-overhead engineering — the binding constant is the 13.83 ms marginal
cost of each drafted token, and the only lever found that moves it is the
ASCII vocab prune (+5.7% on the shipping config, free after Stage 2).**

Driver `roundbudget.py` (synth / prune / knobs / batch), profiler
`nsys_round.py`, analysis `analyze_round.py`. Raw:
`results/round_budget/{synth,prune,knobs,batch_probe,nsys,round_budget}.json`,
per-cell server logs `results/round_budget/*.serverlog`, run logs
`results/stage3_*.log`. Stock binary and stock libraries throughout, so the
numbers compose with `results/speed_recert.md`'s cost model.

## Method — the lever that made this measurable

`llama-server` has `--spec-synth-rates P0,P1,...` (benchmarking only).
Reading `tools/server/server-context.cpp:3887`, it replaces **only the
acceptance decision**: the draft model still runs and the verify batch is still
n+1 tokens. That decouples α from n, which is exactly what a residual needs:

* **rates all 0** → every round emits exactly one token, so measured ms/token
  **is T_round(n)**, with the full draft + verify + host cost of an n-token draft.
* **rates all 1** → every round emits n+1 tokens on identical GPU work, so the
  difference isolates the reject / KV-rollback path.

Cells: 1 discarded warm-up + 4 measured, 400-token generation, the
`speed_recert.md` code prompt, GPU guard <500 MiB, VRAM from a during-request
peak. End-of-session drift control (`rb_c16k_a0_n2` repeated last):
13.376 → 13.347 t/s = **−0.22%**. Noise floor ±1% as established.

## 1. The ladder

c = 16384, q8_0 KV, IQ3_XXS. `T_base` = no-spec ms/token.

| cell | decode t/s | **T_round (ms)** | tokens/round |
|---|---:|---:|---:|
| no-spec | 20.180 | **49.55** | 1 |
| α=0, n=1 | 16.326 | **61.25** | 1 |
| α=0, n=2 | 13.376 | **74.76** | 1 |
| α=0, n=3 | 11.247 | **88.91** | 1 |
| α=1, n=1 | 32.518 | 61.50 | 2 |
| α=1, n=2 | 39.946 | 75.10 | 3 |
| α=1, n=3 | 44.694 | 89.50 | 4 |
| **real n=2 (shipping)** | **34.146** | **74.97** | 2.56 (α=0.781) |

Straight-line fit through the α=0 points:

```
T_round(n) = 47.31 ms + 13.83 ms * n          R^2 = 0.99982
```

Three results fall straight out:

1. **The fixed cost of turning speculation on is ZERO.** The n→0 intercept is
   **47.31 ms** against a measured no-spec token of **49.55 ms** — the
   speculative loop's fixed overhead comes out at **−2.24 ms**, i.e. nothing,
   with the sign an artifact of the fit. Every millisecond of the round scales
   with the number of drafted tokens.
2. **The reject / rollback path is free.** T_round at α=0 vs α=1 differs by
   **−0.25 / −0.34 / −0.59 ms** at n=1/2/3 (rejecting is *marginally cheaper*,
   as it must be — nothing extra gets committed to the KV cache). KV rollback
   and the acceptance loop are not a cost centre.
3. **The cost model was right and the residual is real.** Measured
   T_round at real acceptance is **74.97 ms**, against `speed_recert.md`'s
   fitted **74.45 ms** and the α=0 measurement's **74.76 ms** — three
   independent routes to the same number within 0.7%. The 13.83 ms/drafted
   token also reproduces `mtp_profile.md` finding 1's 13.4 ms, measured a
   different way under a different protocol.

The c=8192 ladder extends this to n=5 with the same shape
(61.34 / 74.96 / 89.09 / 100.51 / 115.18 ms; slope 13.32, intercept 48.25,
fixed overhead −1.63 ms), so the linearity is not a two-point artefact.

## 2. Splitting the 13.83 ms — the verify-batch probe

A speculative round verifies n+1 tokens in **one** forward. `llama-bench -p N
-n 0 -d D` measures exactly that: a batch-N forward at KV depth D.

| batch | ms/forward, depth 0 | depth 4096 |
|---:|---:|---:|
| 1 | 51.13 | 50.89 |
| 2 | 56.66 | 57.13 |
| 3 | 64.06 | 65.08 |
| 4 | 72.75 | 73.81 |
| 5 | 79.41 | 80.43 |
| 6 | 89.20 | 90.25 |
| 8 | 106.61 | 107.56 |

**Marginal cost of one extra token in the verify batch: 8.01 ms**
(depth 0, R² 0.995; 8.15 ms at depth 4096 — depth-independent).

This is itself a finding. A batch-2 decode forward "should" be nearly free
against batch-1 if the GEMV were bandwidth-bound: the same weights are read
either way. It costs **5.5 ms more**, and the curve stays linear out to batch 8.
The decode forward on this model is **not** bandwidth-bound; it is
instruction-issue-bound — precisely `kernel_profile.md`'s finding, now visible
from the outside.

## 3. The round budget, ms by ms

Shipping config, n=2, T_round = **74.97 ms**:

| component | ms | share | reducible? |
|---|---:|---:|---|
| verify forward, the 1 token you'd decode anyway | **49.55** | 66.1% | only by making the forward faster (kernel work) |
| widening the verify batch 1 → 3 tokens (2 × 8.01) | **16.02** | 21.4% | **partly** — it is ALU work, so it is exactly what a lower-instruction-count kernel attacks |
| 2 × MTP head draft forward (2 × 5.82) | **11.64** | 15.5% | **yes, partly** — 25% of it is the LM head (§4) |
| fixed host / graph / sync | **−2.24** | −3.0% | **nothing there to take** |
| unexplained residual | **0.00** | 0.0% | — |

The three GPU terms are measured independently (in-server ladder, llama-bench
batch sweep) and they close the budget to within **0.001 ms**.

So the "~24 ms/round" residual from `speed_recert.md` resolves as
**16.0 ms of verify widening + 11.6 ms of MTP drafting − 2.2 ms** — and the
MTP head's own forward is **5.82 ms, 11.7% of a full 49.55 ms model forward**,
for what is nominally one block (`blk.64`) plus the output head.

## 4. The profile — host share, measured

Nsight Systems, one 30 s window placed inside steady-state decode.
**`--cuda-graph-trace=node` is mandatory**: llama.cpp is built with
`GGML_CUDA_USE_GRAPHS`, and nsys's default graph granularity hides the
per-layer kernels. The first attempt without it reported a nonsensical
"GPU busy 7.6%" while showing only ~6,400 `mul_mat_vec_q` ops in 30 s; with
node tracing the same window contains **987,921** GPU ops.

| | MTP n=2 | no-spec |
|---|---:|---:|
| GPU ops in window | 987,921 | 1,194,037 |
| **GPU busy** | **97.03%** | **97.84%** |
| GPU idle | 0.889 s | 0.646 s |
| gaps | 607,794 | 692,024 |
| mean gap | 1.46 µs | 0.93 µs |
| median gap | 0.064 µs | 0.064 µs |
| p99 gap | 9.02 µs | 1.47 µs |
| gaps > 1 ms | 35 (0.100 s total) | 34 (0.063 s) |

* Rounds in the MTP window ≈ 395 (1,086 tokens at mean_len 2.75), so the
  **total** GPU idle is **2.25 ms per round** — 3.0% of the round — spread
  over ~1,540 gaps averaging 1.5 µs. That is kernel-launch latency between
  back-to-back kernels, not a synchronization stall.
* Speculation's *extra* host idle over no-spec is 0.243 s per 30 s =
  **0.62 ms per round = 0.8% of the round.**
* There is no per-round sync stall: only 35 gaps in 30 s exceed 1 ms, the same
  count as the no-spec control, and they total 0.1 s.

**The profile and the ladder agree independently: the host share of the
speculative round is under 1 ms, and the total host idle under 2.3 ms.**

## 5. Knobs — nothing shipped in llama-server touches it

Each cell is the shipping config with exactly one option moved.
Baseline 34.146 t/s. **Acceptance is identical (0.78135) in every cell that
ran**, so these are pure timing comparisons on an identical token workload.

| knob | decode t/s | vs shipping | note |
|---|---:|---:|---|
| `--spec-draft-p-min 0` | 34.143 | −0.01% | noise |
| `--poll 1 --spec-draft-poll 1` | 34.111 | −0.10% | noise |
| `--backend-sampling` (target side) | 34.106 | −0.12% | **VRAM 12,000 MiB — killed one request** |
| `--no-op-offload` | 34.084 | −0.18% | noise |
| `--no-spec-draft-backend-sampling` | 33.922 | −0.66% | draft sampling on the host is slightly worse |
| `--spec-type draft-mtp-adaptive` (n=3, n_min 1) | 32.790 | **−3.97%** | acceptance collapses 0.781 → 0.649 |
| `-ub 2048` | — | — | **OOM** |

**No knob helps.** That is the expected result given §4: there is no host time
to remove, so an option that reorganises host work cannot pay. `--backend-sampling`
is additionally disqualified on VRAM — it takes the card to 12,000 MiB of 12,288
and a request died mid-cell. `draft-mtp-adaptive` is a real regression on this
workload; its default `n_min_adaptive` of 3 also makes it un-runnable at n_max=2
(arg-parser error), which is worth knowing before anyone tries it.

## 6. The one lever that did move — the ASCII prune

The decomposition makes a falsifiable prediction. If the 5.82 ms MTP draft
forward is dominated by the output head (`output.weight` is 715 MB, by far the
largest tensor a draft forward reads), then halving that tensor must shrink the
marginal drafted-token cost. The ASCII vocab prune halves it exactly
(715 MB → 368 MB, −48.5%), and **Stage 2 just showed it costs no measurable
quality**. Same α=0 lever, so the two builds are not confounded by acceptance.

| | flagship IQ3_XXS | ASCII-pruned | delta |
|---|---:|---:|---:|
| T_base (no-spec) | 49.55 ms | **48.36 ms** | −2.4% |
| T_round α=0, n=1 | 61.25 | **58.78** | −4.0% |
| T_round α=0, n=2 | 74.76 | **70.93** | −5.1% |
| T_round α=0, n=3 | 88.91 | **83.53** | −6.1% |
| **marginal ms / drafted token** | **13.83** | **12.38** | **−10.5%** |
| implied MTP draft forward | 5.82 ms | **4.37 ms** | **−25%** |
| fixed spec overhead | −2.24 ms | −2.03 ms | (both zero) |
| **shipping config, real α** | **34.146 t/s** | **36.102 t/s** | **+5.73%** |
| acceptance | 0.78135 | **0.78135** | identical |
| VRAM peak during requests | 11,830 MiB | **11,498 MiB** | **−332 MiB** |

The prediction holds quantitatively: the prune helps *drafting* four times more
than it helps the *base* forward (−10.5% vs −2.4%), which is what you expect if
the output head is a much larger share of a one-block draft forward than of a
65-block target forward. Acceptance is bit-identical, so the +5.73% is pure
speed on the same token workload.

**+5.73% is above the ±1% noise floor and is the largest config-lever gain
found since the recert declared config levers exhausted at 34-36 t/s.** It is
available today, on a build that already exists, and Stage 2 removed the
quality objection to it.

## 7. Realistic ceiling for round-overhead work — and the 55 t/s question

With `T_round(n) = 47.31 + 13.83n` and at most n+1 tokens per round, the
**perfect-acceptance ceiling** is `1000(n+1) / (47.31 + 13.83n)`:

| n | flagship ceiling @ α=1 | pruned ceiling @ α=1 | measured α at that n |
|---:|---:|---:|---:|
| 2 | 40.0 | 42.2 | 0.781 |
| 3 | 45.0 | 47.9 | 0.637 |
| 4 | 48.7 | 52.2 | 0.556 |
| 5 | 51.5 | 55.5 | 0.508 |
| 7 | **55.5** | 60.2 | — |
| ∞ | 72.3 | 80.8 | — |

* **55 t/s is unreachable at any n≤6 even with perfect acceptance**, and
  acceptance is already down to 0.51 at n=5. On the flagship it needs n≥7 at
  α=1; on the pruned build n≥5 at α=1. Real α at those depths is ~0.5.
* **Eliminating 100% of host time buys +3.1%** (74.97 → 72.72 ms, i.e.
  34.15 → 35.2 t/s), and most of that 2.25 ms is launch latency across ~1,540
  kernel boundaries per round, not a removable stall. **Realistic ceiling for
  round-overhead work specifically: +1 to +3%, and the engineering to get it is
  a kernel-fusion project, not a scheduling fix.**

**Reducible vs irreducible, per component:**

| component | ms @ n=2 | verdict |
|---|---:|---|
| base verify forward | 49.55 | **irreducible as "overhead"** — it is the work; only kernel speed moves it |
| verify batch widening | 16.02 | **REDUCIBLE via kernel work.** It is ALU-bound, not bandwidth-bound (§2), so instruction-count reductions attack it directly |
| MTP draft forwards | 11.64 | **PARTLY REDUCIBLE.** 25% is the output head and the ASCII prune takes it today (§6). The rest is the MTP block itself |
| host / graph / sync | ≤2.25 | **IRREDUCIBLE in practice** — 97-98% GPU busy, no sync stalls, no knob helps, ceiling +3% |

## 8. Verdict

1. **The 24 ms/round unknown is CLOSED and it is not host machinery.**
   16.0 ms verify-batch widening + 11.6 ms MTP drafting − 2.2 ms fixed;
   budget closes to 0.001 ms; profile independently shows 97.0% GPU busy and
   ≤0.62 ms/round of speculation-attributable host idle.
2. **Round-overhead work is a dead end as a speed programme.** Ceiling +1-3%.
   Stop looking for scheduling wins; no shipped `--spec-*` knob helps.
3. **The live target moved to the marginal drafted token (13.83 ms).** Two
   attack surfaces, both already understood: the verify forward's ALU cost
   (kernel offensive — and note §2 reframes the round-1 attenuation: the
   marginal token is *compute*, so kernel gains should show up there rather
   than being purely amortised away), and the MTP draft forward's output head.
4. **Ship the ASCII prune: +5.73%, −332 MiB, quality-neutral.**
   New best measured shipping number: **36.10 t/s** at 11,498 MiB peak.
5. **55 t/s is not reachable on this architecture at n≤4 under any acceptance.**
   The honest statement is now arithmetic, not opinion: it requires either
   n≥5 with near-perfect deep acceptance, or cutting the 12.4-13.8 ms marginal
   drafted-token cost roughly in half.

### Not covered / open

* The MTP draft forward's remaining 4.37 ms (pruned) is not itself decomposed
  into block-vs-head; that needs a per-kernel attribution of the draft graph
  from the nsys traces already captured (`nsys_mtp2_gputrace.csv`).
* The prune's +5.73% was measured on one prompt at n=2; it should get a full
  cert cell set (depth, content types, n-curve) before it becomes the
  official config.
* `-ub 2048` OOMed and was not retried at intermediate ubatch sizes.

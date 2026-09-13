# STAGE 4 — kernel offensive round 2: Q2_K / Q3_K vec_dot

**Verdict: a clean win exists and is CAPTURED — the Q3_K mmvq kernel is
+26.6% faster at batch 1, byte-identical, all four gates passed. It is worth
approximately NOTHING to us (−0.44% / −0.17% end-to-end, both inside noise)
because Q3_K is 1.1-6.6% of the bytes in the models we ship. It is an upstream
contribution, not a stack change. Q2_K: no win available, and the SASS says
why.**

Artifacts: `/data/scratch/kernel_offensive/verify_q3k.c` (proof),
`q3k_patch.py` (the patch), `sass_k/` (both SASS dumps + per-type histograms),
`/data/scratch/q3k_round2/` (bin-stock, bin-q3k, `gates/`, logs).

## 1. Target selection — SASS census of the untouched types

Round 1 never touched Q2_K or Q3_K. Extracting their `mul_mat_vec_q` kernels
from the existing stock sm_86 dump (`sass/mmvq_sm86.sass`, extractor extended
with `ggml_type` 10 and 11):

| type | SASS instr | int-ALU | dp4a | **ALU : dp4a** | **registers** |
|---|---:|---:|---:|---:|---:|
| Q4_K | 184 | 92 | 8 | 11.5 | 39 |
| **Q2_K** | 216 | 116 | 8 | **14.5** | 42 |
| IQ4_XS | 240 | 160 | 8 | 20.0 | 47 |
| IQ2_XXS | 264 | 182 | 8 | 22.8 | 47 |
| IQ2_S | 272 | 190 | 8 | 23.8 | 47 |
| IQ3_XXS | 288 | 202 | 8 | 25.3 | 53 |
| IQ3_S | 304 | 217 | 8 | 27.1 | 55 |
| **Q3_K** | **304** | **219** | **4** | **54.8** | **71** |

Two things jump out and both turned out to matter:

* **Q3_K is by far the most issue-saturated kernel in the file** — 54.8
  integer-ALU instructions per dot product, twice the worst i-quant.
* **Q3_K is the only quant kernel above 64 registers.** Everything else sits at
  39-55. On sm_86 (65,536 registers/SM) 71 registers is the wrong side of a
  hard occupancy cliff.

**Q2_K, by contrast, is the *least* saturated K-quant measured** (14.5:1, below
even Q4_K's i-quant-free path), uses no SIMD-video intrinsics at all, and sits
at 42 registers. Its only visibly redundant work is the
`m |= m<<8; m |= m<<16` scale broadcast, which nvcc has already folded into
2 IMADs; the best bit-exact replacement is a single PRMT, saving 1 instruction
per iteration = **4 of 216 (1.9%)** with no register or occupancy effect.
On a model where Q2_K is 7.2% of bytes that is a ~0.13% ceiling.
**Q2_K verdict: NO CLEAN WIN AVAILABLE, and not worth the gates.**

## 2. Root cause in Q3_K — the round-1 defect, again

`vec_dot_q3_K_q8_1_impl_mmvq` (`vecdotq.cuh:450`) folds the high bit in with a
**saturating SIMD-video subtract**:

```c
const int vil = (vl >> (2*i)) & 0x03030303;   // bytes in {0,1,2,3}
const int vih = ((vh >> i) << 2) & 0x04040404; // bytes in {0,4}
const int vi  = __vsubss4(vil, vih);
```

sm_75+ removed the SIMD-video instructions and nvcc emulates them — exactly
round 1's root cause. In the stock SASS the emulation is unmistakable:
**8 × `0x7f7f7f7f` masks, 8 × `0x80808080` masks, 8 × `PRMT ... 0xba98`**
(the sign-merge/saturate step) and 5 × `0x4040404`, clustered at instructions
164-228 of a 304-instruction kernel.

**The saturation is dead code.** The operands are constrained to `{0,1,2,3}`
and `{0,4}`, so every difference lies in `[-4, 3]` and the `[-128,127]` clamp
can never fire. And the subtract itself is avoidable:

```
h = 0  ->  vi_b = vil_b                        (0..3)
h = 1  ->  vi_b = vil_b - 4 = 252 + vil_b = 0xFC | vil_b
```

because `0xFC` has its low two bits clear and `vil_b < 4`. So:

```c
const int vih = (int)(((vh >> i) & 0x01010101u) * 0xFCu);  // carry-free: bytes are 0/1
const int vi  = vil | vih;
```

One shift, one mask, one multiply, one OR — replacing a mask, a shift and an
~11-instruction emulation.

## 3. The four gates

**Gate 1 — exhaustive host proof: PASS.**
`kernel_offensive/verify_q3k.c`, run on the box:

```
[1] post-mask domain: 4096 cases, 0 mismatches        <- COMPLETE enumeration
[2] real expressions, random 32-bit inputs: 16000000 cases, 0 mismatches
[3] OR==ADD guard on 0xFC: ok
[4] carry-free multiply guard: ok
RESULT: BIT-IDENTICAL (16004096 cases total)
```

Case [1] is a *complete* enumeration, not a sample: each result byte depends on
exactly 2 bits of `vl` and 1 bit of `vh`, so `4^4 × 2^4 = 4096` covers the
whole post-mask domain. Case [2] pushes random full-width inputs through the
**actual** masking expressions, so "the masks really do constrain the domain"
is checked rather than assumed. [3] and [4] guard the two identities the
rewrite rests on.

**Gate 2 — backend tests: PASS.** On the patched build,
`test-backend-ops -o MUL_MAT -b CUDA0` → **1193/1193 tests passed, OK**;
`-o MUL_MAT_ID` (the MoE path, which also has a Q3_K kernel) → **OK**.

**Gate 3 — byte-identical transcripts: PASS.** Both binaries come from the same
configure of the same source tree, differing only in that one line
(patched build → `bin-q3k`; header reverted, incremental rebuild → `bin-stock`),
so a transcript difference could only be the patch. 3 greedy prompts × 200
tokens, `temperature=0, top_k=1, seed=1234`, thinking off:

| model | stock sha1 | patched sha1 | |
|---|---|---|---|
| Q2_K_XL (`-ngl 99`) | `c260b101…` | `c260b101…` | **identical** |
| Q3_K_XL (`-ngl 52`) | `22034439…` | `22034439…` | **identical** |

**Gate 4 — SASS control of untouched types: PASS.** Recompiling `mmvq.cu` for
sm_86 and hashing every kernel body: of **294 kernels, exactly 12 changed, and
all 12 are `ggml_type11` (Q3_K)** — the 8 `mul_mat_vec_q` variants, the 3 fused
variants, and `mul_mat_vec_q_moe`. The other 282 are byte-for-byte unchanged.
Per-type instruction histograms confirm it: Q2_K 216, Q4_K 184, IQ2_XXS 264,
IQ3_XXS 288, IQ3_S 304, IQ2_S 272, IQ4_XS 240 — all identical to stock.

Instruction counts on the changed kernels:

| kernel variant | stock | patched | |
|---|---:|---:|---:|
| `type11, ncols=1` (the decode kernel) | 304 | **272** | **−10.5%** |
| `type11, ncols=1, fused` | 792 | 640 | −19.2% |
| `type11, ncols=1, fused+small_k` | 1920 | 1624 | −15.4% |
| `type11, ncols=2` | 608 | 536 | −11.8% |
| `type11, ncols=8` | 1144 | 1080 | −5.6% |
| `mul_mat_vec_q_moe, type11` | 496 | 416 | −16.1% |

Attribution is exact: the emulation signature is *gone* (`0x7f7f7f7f` 8→0,
`0x80808080` 8→0, `PRMT 0xba98` 8→0, `0x4040404` 5→0) and replaced by
**exactly 4 IMADs with the `0xFC` constant** — one per unrolled iteration, as
designed. No spills either way (0 bytes stack frame both builds).

## 4. The kernel measurement — and why it is 26.6%, not 10.5%

`test-backend-ops perf -o MUL_MAT`, the Q3_K shapes, CUDA0:

| batch n | stock µs/run | patched µs/run | **speedup** |
|---:|---:|---:|---:|
| **1** | 145.72 | **115.08** | **+26.6%** |
| 2 | 157.22 | 127.80 | +23.0% |
| 3 | 185.75 | 161.59 | +15.0% |
| 4 | 212.00 | 187.16 | +13.3% |
| 5 | 236.09 | 209.98 | +12.4% |
| 8 | 332.42 | 314.40 | +5.7% |
| **512** | **2097.33** | **2098.59** | **−0.06%** |

The n=512 row is a **clean internal control**: at that batch llama.cpp switches
to the MMQ path (`vec_dot_q3_K_q8_1_impl_mmq`), which the patch does not touch,
and the timing is identical to 0.06%.

**A −10.5% instruction count does not buy +26.6%. The occupancy does.**
ptxas register usage on the decode kernel drops **71 → 64**. On sm_86 with
65,536 registers per SM, 71 registers caps residency at
`floor(65536/71) = 923` threads, while 64 registers gives exactly **1,024** —
one more 256-thread block per SM, +33% occupancy. That, not the instruction
count, is the dominant term, and it is why the gain is largest at n=1 (the most
latency-bound shape) and decays to +5.7% by n=8.

The register census in §1 explains why this was available at all: **Q3_K was the
only quant kernel in the file above 64 registers**, and the excess was the
`__vsubss4` emulation's live temporaries.

## 5. End-to-end: null, exactly as the byte census predicted

Pre-registered prediction from the GGUF tensor census: `Q3_K` is
**1.1% of Q2_K_XL's bytes** and **6.6% of Q3_K_XL's**, so even a free kernel
could not move either model by more than ~1-7% of its Q3_K time share.

`llama-bench`, tg128, 5 reps, same binaries:

| model | stock t/s | patched t/s | delta |
|---|---:|---:|---:|
| Q2_K_XL (`-ngl 99`) | 21.371 ± 0.092 | 21.277 ± 0.027 | **−0.44%** |
| Q3_K_XL (`-ngl 52`) | 6.268 ± 0.032 | 6.258 ± 0.022 | **−0.17%** |

Both nulls, both inside the ±1% noise floor. (The Q3_K_XL cell is additionally
diluted: at 12.23 GiB it does not fit on the card, so `-ngl 52` makes it
host-bound at 6.3 t/s and any GPU-side gain is buried.) **The measurement
agrees with the arithmetic: there is nothing here for our models.**

For reference, on a model where Q3_K is the dominant tensor type — plain
`Q3_K_M` / `Q3_K_S`, among the most widely downloaded GGUF quants — the same
patch would be worth on the order of +15-25% raw decode. That is the population
this belongs to.

## 6. Verdict

1. **Q2_K: NO CLEAN WIN AVAILABLE.** It carries no emulated video
   instructions, has the lowest ALU:dp4a ratio of any K-quant measured (14.5:1),
   and sits comfortably at 42 registers. The one bit-exact idea (PRMT broadcast)
   is 1.9% of the kernel on 7.2% of the bytes — not worth the gates.
2. **Q3_K: CLEAN WIN, CAPTURED AND VERIFIED. +26.6% at batch 1, +5.7% at
   batch 8, byte-identical, all four gates passed.** Root cause is round 1's
   root cause (nvcc emulating a removed SIMD-video instruction) and the
   mechanism is a register/occupancy cliff, which is new.
3. **DO NOT SHIP IT IN OUR STACK.** Q3_K is 1.1-6.6% of our models' bytes and
   the end-to-end effect is −0.44% / −0.17%, i.e. nothing. Adding a third
   private patch to the build for an unmeasurable gain is a maintenance cost
   with no return.
4. **DO submit it upstream.** It is the cleanest of the three patches now in
   hand: a single-line replacement of a saturating intrinsic whose saturation is
   provably unreachable, with an exhaustive proof, 1193/1193 backend tests,
   byte-identical transcripts, an untouched-type SASS control and an untouched
   MMQ-path timing control. Per llama.cpp's `AGENTS.md`, the founder must own
   and submit it personally — the idea is two sentences, and the register-cliff
   observation is the part reviewers will care about.
5. **Round-2 conclusion for the kernel programme:** the remaining headroom is
   not in the types we ship. Stage 3 identified where our headroom actually is
   (the verify batch's ALU cost and the MTP draft's output head), and that is a
   better target than more vec_dot archaeology.

### Not covered / open

* The Q2_K PRMT-broadcast idea was analysed but not implemented or gated.
* No pure-Q3_K model exists on the box, so the +15-25% figure for `Q3_K_M`
  users is an extrapolation from the per-op numbers, not a measurement.
* The occupancy explanation is inferred from ptxas register counts and the
  shape of the n-curve; it was not confirmed with an `ncu` achieved-occupancy
  counter.
* `ncols>1` and fused variants were counted in SASS but not benchmarked.

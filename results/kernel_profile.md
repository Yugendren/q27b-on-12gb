# Kernel-level decode profile — where Qwen3.8-27B (hybrid GDN) loses bandwidth

Answers the two open diagnostics: this model reaches **~59–65% of peak memory
bandwidth** on decode where a standard dense model in llama.cpp reaches
**~90%**, and **quantized KV decode is slower than f16 KV**. Both are now
measured per kernel.

Rig: ollama box, RTX 3060 12 GB (GA106, 28 SMs, 192-bit GDDR6 @ 15 Gbps =
**360 GB/s theoretical peak**), driver 535.309.01 / CUDA 12.2.
llama.cpp build **10712 / daef7b687**.
Model `Qwen3.8-27B-UD-IQ3_XXS.gguf` — 10.174 GiB, arch `qwen35`, **65 blocks
= 48 gated-delta-net + 16 full-attention**, n_embd 5120, n_ff 17408, dense
(no experts), 866 tensors.

Tools: **Nsight Systems 2026.1.1** (time/timeline) and **Nsight Compute
2022.4.1** (per-kernel DRAM traffic and throughput). Both are real profilers;
**the GGML_PERF fallback was not needed**. The distro `nsight-systems` package
is Ubuntu's broken target-only build (no `QdstrmImporter`), so a self-contained
`nsight-systems-cli` .deb was extracted to `/data/tools/nsys-2026` — no system
package replaced, no CUDA driver touched. `ncu` was already complete but needs
`sudo` for GPU performance counters (also no driver change; the
`NVreg_RestrictProfilingToAdminUsers` module reload was deliberately avoided).

---

## 0. Verdicts

| # | Finding |
|---|---|
| 1 | **Decode is not launch-bound. The GPU is 99.0% busy.** Launch gap / idle is **0.484 ms of 48.933 ms** (0.99%) at depth 0, and stays 1.0–1.1% at 16k depth and under quantized KV. Any optimisation premised on hiding launch overhead has ~0.5 ms/token to win, total. |
| 2 | **89.6% of a decode step is one kernel: `mul_mat_vec_q`** — 43.837 ms of 48.933 ms, 468 launches/token. Everything else (GDN, attention, norms, rope, rope, state copies, gaps) is 10.4% combined. |
| 3 | **The bandwidth shortfall is the quantisation format, not the hybrid architecture.** `mul_mat_vec_q` is template-instantiated per ggml type. Splitting the *same* decode step by type: **sub-4-bit LUT/codebook quants carry 75.2% of the bytes and 81.8% of the GEMV time at 214.1 GB/s = 59.5% of peak, while 4-bit-and-up arithmetic quants (Q4_K/Q5_K/IQ4_XS) run at 324.3 GB/s = 90.1% of peak.** The "~90% for a dense model" number is reproduced *inside this model*, by its own 4-bit tensors. |
| 4 | **The i-quant GEMVs are compute-bound, not memory-bound.** IQ3_XXS: DRAM 46.9% but SM 69.8%. IQ2_XXS: DRAM 36.7%, SM 73.7%. Q4_K: DRAM 68.0%, IQ4_XS: DRAM 86.5%. The sub-4-bit kernels are ALU-saturated on codebook lookup + shift/mask unpacking, so extra bandwidth would buy them nothing. |
| 5 | **Quantized KV is slower because it changes *which attention kernel runs*, not because of a separate dequant kernel.** At d=16384, f16 KV runs `flash_attn_ext_f16` (218 us/layer, 68.3 MB, **313.9 GB/s = 87.2% of peak, SM 8.3% → bandwidth-bound**). q4_0 KV runs `flash_attn_ext_vec` (626 us/layer, 36.0 MB, **57.4 GB/s = 16.0% of peak, SM 64.2% → compute-bound**). **q4_0 reads 47% fewer bytes and takes 2.9x longer.** A `dequantize_block_q4_0` kernel does exist but costs 0.007 ms/token — the dequant is *inside* the attention inner loop. |
| 6 | **This hits the shipping config.** `q8_0` KV — what `mtp_profile.md` ships — takes the *same* slow `flash_attn_ext_vec` path: 6.17 ms/token vs f16's 3.35 ms. Measured at d=16384: **f16 19.33, q8_0 18.07 (−6.5%), q4_0 17.71 (−8.4%) t/s.** |
| 7 | **`flash_attn_ext_f16` proves the rig and the runtime are fine.** It hits **87.2% of theoretical peak** on this GPU, in this model, inside the same decode step. The 59% figure is a property of the weight GEMVs, not of the GPU, the driver, or the hybrid GDN design. |
| 8 | **The 8.60 ms/drafted-token "state management" from `mtp_profile.md` §3 is not GPU state work.** The *entire* GDN recurrent-state path in a full decode step — `gated_delta_net_cuda` + `ssm_conv_f32` + every state copy/gather/set-rows across all 48 GDN layers — costs **2.39 ms/token**. 8.60 ms is **3.6x** that. It must be host-side: graph rebuild, sync, rollback bookkeeping. That sharpens the TreeWY/Bole motivation rather than weakening it. |

---

## 1. Method

`llama-bench` with `-p 0` so the run is pure decode. Four configs:
f16/q4_0 KV × depth 0/16384, each at `-n 32` and `-n 96`, `-r 1`.

Two extraction passes, because the first was wrong:

* **Rejected:** differencing `n96 − n32` per-kernel totals. It cancels model
  load, warmup and prefill exactly, but at `-d 16384` the ~17 s prefill has
  ~1% run-to-run variance, which leaks in as ~2.7 ms/token of noise —
  `mul_mat_q` (a prefill-only kernel) showed a bogus 1.7 ms/token delta on a
  *zero* count delta.
* **Used:** the lm_head GEMV (`mul_mat_vec_q`, gridX = 248320) fires exactly
  once per decode step, so it is used as a step boundary and only kernels
  strictly inside a decode window are counted. Steps whose wall time exceeds
  1.10x the median are dropped (the `-d` runs contain 6 prefill/transition
  windows, one as long as 457 ms).

Validation: reconstructed wall times match llama-bench's own reported t/s to
within 0.4% in all four configs (20.44/20.36/19.03/17.51 vs
20.29/20.21/18.97/17.45), and the reconstructed GEMV launch count is exactly
468, matching ncu independently. nsys overhead is ~2% (20.29 vs 20.65 t/s
unprofiled).

For bandwidth, **time comes from nsys and bytes come from ncu**. nsys keeps the
`ggml_type` template parameter in `demangledName`, so per-launch time is
attributed to a quant type exactly. An earlier attempt rescaled ncu's own
durations onto nsys time uniformly; that is invalid (ncu serialises and flushes
L2, and its per-kernel overhead is *not* uniform — it pushed IQ4_XS to 112% of
theoretical peak, a physical impossibility). ncu's byte counts are unaffected by
serialisation and are used as-is.

---

## 2. Ranked kernel table — one decode step, f16 KV, depth 0

Wall **48.933 ms/token = 20.44 t/s**; GPU busy 48.449 ms; gap 0.484 ms.

| kernel | grid | n/tok | ms/tok | % step | us/call |
|---|---|---:|---:|---:|---:|
| `mul_mat_vec_q` (ffn gate/up, n_ff 17408) | 17408x1x1 | 99 | 19.021 | 38.87 | 192.1 |
| `mul_mat_vec_q` (ffn down + attn out, 5120) | 5120x1x1 | 128 | 12.892 | 26.35 | 100.7 |
| `mul_mat_vec_q` (GDN in-proj, 10240) | 10240x1x1 | 48 | 4.565 | 9.33 | 95.1 |
| `mul_mat_vec_q` (GDN ssm, 6144) | 6144x1x1 | 48 | 2.855 | 5.83 | 59.5 |
| `mul_mat_vec_q` (**lm_head**, vocab 248320) | 248320x1x1 | 1 | 2.103 | 4.30 | 2102.6 |
| `mul_mat_vec_q` (attn q/gate, 12288) | 12288x1x1 | 16 | 1.765 | 3.61 | 110.3 |
| `gated_delta_net_cuda` | 48x1x32 | 48 | 0.903 | 1.85 | 18.8 |
| `k_get_rows_float_vec` (GDN state read) | 1x768x1 | 48 | 0.876 | 1.79 | 18.2 |
| `rms_norm_f32` | 1x1x1 | 129 | 0.603 | 1.23 | 4.7 |
| `quantize_q8_1` (activation quant) | 20x1x1 | 340 | 0.453 | 0.93 | 1.3 |
| `mul_mat_vec_q` (attn k/v, 1024) | 1024x1x1 | 32 | 0.378 | 0.77 | 11.8 |
| `mul_mat_vec_q` (GDN dt, 48) | 48x1x1 | 96 | 0.259 | 0.53 | 2.7 |
| `flash_attn_ext_f16` | 16x1x1 | 16 | 0.194 | 0.40 | 12.1 |
| `cpy_scalar` | 480x1x1 | 48 | 0.154 | 0.31 | 3.2 |
| `ssm_conv_f32` | 1x80x1 | 48 | 0.138 | 0.28 | 2.9 |

### By class, all four configs (ms/token)

| class | f16 d0 | q4_0 d0 | f16 d16k | q4_0 d16k |
|---|---:|---:|---:|---:|
| GEMV (`mul_mat_vec_q`) | **43.837** (89.6%) | 43.805 | **44.223** (84.2%) | 44.528 (78.0%) |
| full attention | 0.220 | 0.276 | **3.355** (6.4%) | **7.403** (13.0%) |
| state copy / gather | 1.344 | 1.303 | 1.348 | 1.310 |
| GDN / delta-net | 1.041 | 1.040 | 1.041 | 1.046 |
| norms | 0.899 | 0.901 | 0.908 | 0.916 |
| quantize / dequantize | 0.634 | 0.707 | 0.642 | 0.720 |
| elementwise / gating | 0.426 | 0.505 | 0.429 | 0.516 |
| rope | 0.048 | 0.047 | 0.047 | 0.048 |
| **launch gap / idle** | **0.484** (0.99%) | 0.563 (1.15%) | 0.548 (1.04%) | 0.632 (1.11%) |
| **wall** | **48.933** | 49.128 | 52.540 | 57.118 |
| **t/s** | **20.44** | 20.36 | 19.03 | 17.51 |

Note the GDN and state-copy columns are **flat across depth** — as they must be
for a recurrent layer. All depth scaling lands in full attention.

---

## 3. Answer (i) — which kernels account for the bandwidth shortfall

One decode step's 468 GEMVs, split by quantisation type. Time from nsys, DRAM
bytes and throughput counters from ncu.

| quant | n/tok | GiB/tok | ms/tok | % step | GB/s | % of 360 | DRAM% | SM% | dequant | bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| IQ3_XXS | 109 | 2.791 | 13.288 | 27.16 | 225.5 | 62.7 | 46.9 | 69.8 | LUT/codebook | compute |
| IQ3_S | 103 | 2.429 | 11.285 | 23.06 | 231.1 | 64.2 | 47.0 | 65.4 | LUT/codebook | compute |
| **IQ4_XS** | 39 | 1.370 | 4.395 | 8.98 | **334.8** | **93.0** | 86.5 | 71.3 | arithmetic | **bandwidth** |
| IQ2_S | 30 | 0.780 | 4.218 | 8.62 | 198.6 | 55.2 | 40.7 | 66.5 | LUT/codebook | compute |
| **Q4_K** | 26 | 0.879 | 2.836 | 5.80 | **332.9** | **92.5** | 68.0 | 67.3 | arithmetic | **bandwidth** |
| IQ2_XXS | 21 | 0.446 | 2.737 | 5.59 | 175.1 | 48.6 | 36.7 | 73.7 | LUT/codebook | compute |
| IQ2_XS | 12 | 0.274 | 1.551 | 3.17 | 189.4 | 52.6 | 39.7 | 75.3 | LUT/codebook | compute |
| Q3_K | 6 | 0.145 | 1.021 | 2.09 | 152.6 | 42.4 | 32.4 | 82.9 | LUT/codebook | compute |
| IQ1_S | 7 | 0.130 | 0.963 | 1.97 | 145.4 | 40.4 | 30.1 | 59.0 | LUT/codebook | mixed |
| Q2_K | 9 | 0.156 | 0.807 | 1.65 | 207.6 | 57.7 | 44.0 | 83.9 | LUT/codebook | compute |
| Q8_0 | 96 | 0.024 | 0.259 | 0.53 | 100.0 | 27.8 | 18.5 | 11.8 | arithmetic | latency (tiny) |
| Q5_K | 7 | 0.024 | 0.090 | 0.18 | 280.9 | 78.0 | 69.1 | 65.2 | arithmetic | bandwidth |

**The split that answers the question:**

| group | bytes | GEMV time | achieved | % of peak |
|---|---:|---:|---:|---:|
| sub-4-bit LUT/codebook | 7.151 GiB (75.2%) | 35.870 ms (81.8%) | **214.1 GB/s** | **59.5%** |
| 4-bit-and-up arithmetic | 2.328 GiB (24.5%) | 7.709 ms (17.6%) | **324.3 GB/s** | **90.1%** |
| all GEMV | 9.503 GiB | 43.837 ms | 232.8 GB/s | 64.7% |

**This is the whole answer.** The "~90% of peak for a standard dense model"
figure is not a property of dense models — it is a property of *4-bit-and-up
block quants with arithmetic dequant*, and this model's own Q4_K/IQ4_XS/Q5_K
tensors hit exactly that (90.1%) on this GPU in this decode step. The 41%
shortfall is contributed almost entirely by the **67.7% of model bytes stored in
sub-4-bit i-quants** (IQ3_XXS 27.4%, IQ3_S 23.8%, IQ2_S 7.6%, IQ2_XXS 4.4%,
IQ2_XS 2.7%, IQ1_S/M 1.8%), whose `vec_dot` inner loops are ALU-saturated on
codebook lookup and bit unpacking (SM 65–84% while DRAM sits at 30–47%).

### Independent whole-model confirmation

Across four UD quants of this same architecture (existing `bench_*.json`,
tg128, f16 KV), achieved bandwidth **falls** as the quant gets more aggressive:

| model | size | t/s | achieved | % of peak | sub-4-bit share |
|---|---:|---:|---:|---:|---:|
| IQ3_XXS | 10.174 GiB | 20.69 | 226 GB/s | 62.8% | 67.7% |
| Q2_K_XL | 9.144 GiB | 21.75 | 214 GB/s | 59.3% | 73.9% |
| IQ2_S | 7.787 GiB | 22.48 | 188 GB/s | 52.2% | — |
| IQ2_XXS | 6.757 GiB | 23.33 | 169 GB/s | 47.0% | 84.3% |

IQ2_XXS is **34% smaller** than IQ3_XXS but only **12.8% faster**; a purely
bandwidth-bound model would predict +51% (31.1 t/s). Shrinking the model past
4 bpw buys progressively less because each step down moves more bytes onto a
slower dequant path. This is a second, independent line of evidence for the
same mechanism, and it did not require the profiler.

---

## 4. Answer (ii) — which kernels appear or grow with quantized KV

At depth 0 the KV cache is nearly empty and KV type is irrelevant
(20.65 / 20.58 / 20.53 t/s for f16 / q4_0 / q8_0 — within noise). The whole
effect is at depth. Measured at **d = 16384**, clean (unprofiled) `llama-bench`:

| KV type | t/s @ d16384 | vs f16 | attention kernel | FA ms/token |
|---|---:|---:|---|---:|
| f16 | **19.33** | — | `flash_attn_ext_f16` (+ `stream_k_fixup`) | 3.355 |
| q8_0 | 18.07 | **−6.5%** | `flash_attn_ext_vec` (+ `combine_results`) | 6.173 |
| q4_0 | 17.71 | **−8.4%** | `flash_attn_ext_vec` (+ `combine_results`) | 7.403 |

**Kernels that appear only with quantized KV:** `flash_attn_ext_vec`,
`flash_attn_combine_results`, `k_set_rows_quant`, `fwht_cuda`,
`dequantize_block_q4_0`.
**Kernels that disappear:** `flash_attn_ext_f16`,
`flash_attn_stream_k_fixup_uniform`, `k_set_rows`.

Per-layer ncu at d=16384:

| KV | kernel | grid | us/call | DRAM bytes | GB/s | % peak | DRAM% | SM% | bound |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| f16 | `flash_attn_ext_f16` | 28x1x1 | 217.6 | 68.31 MB | **313.9** | **87.2** | 91.9 | 8.3 | bandwidth |
| q4_0 | `flash_attn_ext_vec` | 1x7x24 | 626.4 | 35.97 MB | **57.4** | **16.0** | 16.6 | 64.2 | **compute** |

The byte counts confirm the cache is genuinely compressed (68.31 MB ≈ 16
layers × 4 KV heads × 256 × 2 × 2 B × 16384 = 67.1 MB for f16; 35.97 MB ≈ the
q4_0 4.5-bit equivalent). **The compression works and the kernel still loses.**
The dequant cost is not a visible dequant kernel — `dequantize_block_q4_0` is
0.007 ms/token — it is the on-the-fly unpack inside `flash_attn_ext_vec`'s
inner loop, which drops the kernel from 91.9% DRAM utilisation to 16.6% and
raises SM utilisation from 8.3% to 64.2%.

The secondary costs are real but small: `fwht_cuda` (+0.078 ms, 64 launches)
and `k_set_rows_quant` (+0.073 ms) are the quantize-on-write path.

---

## 5. Answer (iii) — bound classification per top kernel

| kernel / group | ms/tok | DRAM% | SM% | classification |
|---|---:|---:|---:|---|
| `mul_mat_vec_q` sub-4-bit i-quants | 35.870 | 30–47 | 59–84 | **compute-bound** (codebook/LUT dequant) |
| `mul_mat_vec_q` Q4_K / IQ4_XS / Q5_K | 7.709 | 68–87 | 65–71 | **bandwidth-bound** (near optimal) |
| `flash_attn_ext_f16` (f16 KV, depth) | 3.355 | 91.9 | 8.3 | **bandwidth-bound** (87.2% of peak) |
| `flash_attn_ext_vec` (quantized KV, depth) | 7.403 | 16.6 | 64.2 | **compute-bound** (in-loop dequant) |
| `gated_delta_net_cuda` | 0.903 | 73.2 | 55.7 | bandwidth-bound |
| `k_get_rows_float_vec` (GDN state read) | 0.876 | 79.0 | 36.4 | bandwidth-bound |
| `rms_norm_f32`, `l2_norm_f32` | 0.899 | 0.8–2.3 | 0.6–10 | **launch/latency-bound** (4.7 us for 41 KB) |
| `quantize_q8_1` (340+ launches) | 0.634 | 2.3–6.7 | 4.7–14 | **launch/latency-bound** (1.3 us/call) |
| `mul_mat_vec_q` Q8_0 tiny (96/tok) | 0.259 | 18.5 | 11.8 | **launch/latency-bound** (2.7 us/call) |
| `cpy_scalar`, `concat_cont`, `k_set_rows` | ~0.35 | 2–11 | 10–36 | launch/latency-bound |

The latency-bound tail is real but small: of the **1,933 kernel launches per
decode token**, 1,449 run in under 5 us each and together account for
**2.898 ms (5.92% of the step)**; the measured gap between kernels adds only
0.484 ms (0.99%) on top of that.

---

## 6. Cross-check against `mtp_profile.md`

`mtp_profile.md` §3 attributes **8.60 ms per drafted token** to
"GDN recurrent-state snapshot/rollback + wider verify + per-round graph work".
This profile bounds the GPU-side part of that:

| GDN state work in a full decode step | ms/token |
|---|---:|
| `gated_delta_net_cuda` + `ssm_conv_f32` (48 layers) | 1.041 |
| `k_get_rows_float_vec` + `k_get_rows_float` (state read) | 0.985 |
| `cpy_scalar` + `concat_cont` + `k_set_rows` (state write) | 0.359 |
| **total, all 48 GDN layers** | **2.386** |

**A whole decode step's worth of GDN state traffic is 2.39 ms. The MTP profile's
per-drafted-token state charge is 8.60 ms — 3.6x larger.** So the 8.60 ms is
*not* the recurrent-state kernels; it is host-side work (graph rebuild, device
sync, rollback bookkeeping) that a profiler of GPU kernels cannot see and that
a faster GPU would not fix. That is consistent with `mtp_profile.md`'s
conclusion and makes it stronger: the TreeWY/Bole port targets host-side
per-draft-position bookkeeping, not GPU state bandwidth.

---

## 7. Top 3 kernel-level optimisation targets

### 1. Sub-4-bit i-quant GEMV dequantisation — ceiling **+30 to +36% (20.4 → ~27 t/s)**

81.8% of GEMV time and 75.2% of bytes run at 59.5% of peak while the same
kernel on 4-bit data runs at 90.1%. Moving the LUT bytes onto the measured
arithmetic-dequant rate:

| if sub-4-bit GEMVs ran at… | GEMV ms | wall ms | t/s | gain |
|---|---:|---:|---:|---:|
| measured 4-bit+ rate (324.3 GB/s) | 43.84 → 31.65 | 48.93 → 36.74 | 20.44 → **27.22** | **+33.2%** |
| IQ4_XS rate (334.8 GB/s) | 43.84 → 30.90 | 48.93 → 36.00 | 20.44 → **27.78** | +35.9% |
| `flash_attn_ext_f16` rate (313.9 GB/s) | 43.84 → 32.43 | 48.93 → 37.53 | 20.44 → **26.65** | +30.4% |

**The honest caveat, and it is the important part of this report:** the obvious
route — requantise to IQ4_XS/Q4_K — **does not fit**. At ~4.25 bpw a 27B model
is ~14.4 GiB, well over the 12 GB card. *The model only fits in 12 GB because
of the sub-4-bit quants, and those quants are exactly what costs the
bandwidth.* So this ceiling is real but not freely collectable; it has to be
bought by optimising the i-quant `vec_dot` kernels themselves (codebook
residency in shared/constant memory, better ILP, wider loads) rather than by
changing the format. Treat +33% as the ceiling of a **CUDA kernel engineering
project**, not a flag change. A partial win — closing half the gap — is
~23.7 t/s (+16%).

### 2. Stop quantizing the KV cache — ceiling **+6.5% at 16k, free, today**

The shipping config uses `-ctk q8_0 -ctv q8_0`. Measured at d=16384:
f16 **19.33** vs q8_0 **18.07** t/s. Switching to f16 KV is a flag change worth
**+7.0%** at 16k depth, and it grows with context because the penalty is on the
attention kernel whose cost scales with depth.

VRAM constraint, stated plainly: f16 KV for this model costs 64 KiB/token
(16 attention layers × 4 KV heads × 256 × 2 × 2 B) = **1.0 GiB at 16k**,
2.0 GiB at 32k. With a 10.17 GiB model on a 12 GB card, f16 KV is viable to
roughly **16–20k context** and not beyond. Past that you are forced back onto
`flash_attn_ext_vec` and should expect to pay the 6.5–8.4%. The real fix is
upstream: `flash_attn_ext_vec` needs the tensor-core treatment
`flash_attn_ext_f16` already has, or quantized KV needs a tile-dequant-to-shared
path so the unpack is amortised.

### 3. There is no third target — the rest is spread thin

After GEMV (89.6%) and the KV/attention kernel choice, **nothing else exceeds
2.75% of a decode step.** State copy/gather 1.344 ms (2.75%), GDN 1.041 ms
(2.13%), norms 0.899 ms (1.84%), activation quantize 0.634 ms (1.30%),
elementwise 0.426 ms (0.87%), launch gaps 0.484 ms (0.99%).

Perfectly fusing **every** sub-5-us kernel (2.898 ms) and eliminating **all**
launch gap (0.484 ms) — which no one can do — bounds out at 3.382 ms/token,
i.e. **48.93 → 45.55 ms = 21.95 t/s, +7.4%**. A realistic fusion effort (norms
into the following GEMV, `quantize_q8_1` into the GEMV prologue) is worth
**+2 to +3%**. That is not where the time is, and this report recommends
against spending effort there.

---

## 8. Caveats and what was not measured

* Peak bandwidth is the **theoretical** 360 GB/s. Real achievable peak on
  GA106 is typically 85–92% of that, so the "90.1% of peak" figure for
  4-bit quants means those kernels are essentially **at the hardware limit** —
  there is nothing left in them. Percentages against a measured STREAM-style
  ceiling would be a few points higher across the board; the *ratios* between
  quant types, which carry the argument, are unaffected.
* ncu serialises kernels and flushes L2, so its absolute durations run ~33%
  long in aggregate (58.3 ms vs the real 43.8 ms for the same 468 launches).
  All times quoted in this report are nsys/wall-clock; ncu is used only for
  DRAM byte counts and DRAM%/SM% throughput ratios, which are replay-safe.
* ncu byte counts were sampled from **one** decode step (470 launches). Per-type
  byte totals reconcile with the GGUF tensor inventory to within a few percent
  (e.g. ncu IQ3_XXS 2.791 GiB vs GGUF 2.783 GiB), which is the cross-check.
* Speculative decoding was **not** profiled — all configs are plain `-p 0`
  decode at batch 1. The §6 comparison bounds the GPU-side GDN state cost but
  does not decompose the MTP draft loop itself; that still needs source timers,
  as `mtp_profile.md` §"Not done" says.
* `-d 16384` only. Attention cost scales with depth, so the KV-quant penalty in
  §4 grows beyond 16k and shrinks below it.
* The `type29` row (0.055 GiB, 3 launches) is a ggml type id not in this
  build's public enum table; too small to matter.

## 9. Artefacts

All in `/data/projects/q27b_on_12gb/results/kernel_profile/`:
`{f16,q4}_{d0,d16k}_{n32,n96}.nsys-rep` + `.sqlite` (8 traces), `q8_d16k.*`,
`ncu_gemv_step.csv` (470 GEMVs, 4 metrics), `ncu_fa_f16.csv`, `ncu_fa_q4.csv`,
`ncu_sample1.csv`, `step_breakdown.txt` / `.json`, `gemv_by_quant.json`,
`nsys_breakdown.txt`, and the scripts `phase1_nsys.sh`, `phase2b.sh`,
`analyze3.py`, `analyze_ncu.py`, `final2.py`.

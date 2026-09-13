# MMVQ→MMQ small-batch routing probe — RUNBOOK

Patch: mmvq_small_batch.patch (3 hunks, ggml/src/ggml-cuda/ggml-cuda.cu only).
Env-gated: `GGML_CUDA_MMQ_MIN_BATCH=N` (unset = stock behaviour). Default type set =
the compute-bound LUT group {Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S};
`GGML_CUDA_MMQ_ALL_TYPES=1` widens to all MMQ-supported types. ne11==1 path untouched.
Safety: only reroutes when MMQ is already eligible — never falls to the cuBLAS/f16 path.

## Why this exists
Verify batches (ne11 = 2..8 under MTP) route to MMVQ, which re-executes the sub-4-bit
codebook decode per column. Hypothesis: that is most of the 8.01 ms/drafted-token
"verify widening" cost. GDN is exonerated (≤10% share).
CAVEAT: the 8.01 baseline came from a NON-SWAR binary. Judge ONLY on the within-binary
A/B (w_off vs w_on from step b1), never against 8.01.

## 0. Preconditions (v11 source tree on box)
```
cd /data/projects/revv-home/src/llama.cpp
git status --porcelain     # expect: M mmvq.cu, M vecdotq.cuh, M server-context.cpp, M test_slot_save.py
cp ggml/src/ggml-cuda/ggml-cuda.cu /tmp/ggml-cuda.cu.pre-mmq
git apply --check mmvq_small_batch.patch && git apply mmvq_small_batch.patch
```
Do NOT commit/stash the existing dirty files.

## (a) Build to a SEPARATE dir — never overwrite v11 binaries
```
cmake -S . -B build-mmq -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-mmq -j"$(nproc)" --target llama-bench llama-server test-backend-ops
./build-mmq/bin/test-backend-ops -o MUL_MAT -b CUDA0 | tail -20                       # env unset: must pass
GGML_CUDA_MMQ_MIN_BATCH=2 ./build-mmq/bin/test-backend-ops -o MUL_MAT -b CUDA0 | tail -20
```
The env var is live iff this log line appears exactly once per process when set (never when unset):
`ggml_cuda_prefer_mmq_small_batch: GGML_CUDA_MMQ_MIN_BATCH=2 active (all_types=0): routing ...`
If absent → STOP, null experiment.

## (b) Settling runs
b1 — batch_probe (primary number, ~15 min). roundbudget.py hardcodes BENCH at line 49 to
/data/projects/llama.cpp/build/bin/llama-bench — override to build-mmq/bin/llama-bench for BOTH arms.
```
M=/data/projects/q27b_on_12gb/models/unsloth-q27b/Qwen3.8-27B-UD-IQ3_XXS.gguf
B=/data/projects/revv-home/src/llama.cpp/build-mmq/bin/llama-bench
for E in "" "2"; do GGML_CUDA_MMQ_MIN_BATCH=$E $B -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 1,2,3,4,5,6,8 -n 0 -d 0 -r 3 -o json; done
```
Fit ms_per_batch = c + w·N over N=1..8 (roundbudget.py:171-178). Report w_off and w_on.

b2 — nsys kernel diff (which kernel moved):
```
NS=/data/tools/nsys-2026/bin/nsys
for N in 1 3; do for E in "" "2"; do GGML_CUDA_MMQ_MIN_BATCH=$E $NS profile --cuda-graph-trace=node -o p${N}_e${E:-off} $B -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p $N -n 0 -d 0 -r 1; done; done
```
Expect OFF: mul_mat_vec_q grows ~w ms/token p1→p3, gated_delta_net_cuda grows ≤0.6 ms.
Expect ON at p3: mul_mat_vec_q largely replaced by mul_mat_q, total GEMV time down.
If gated_delta_net_cuda is what grows ~8 ms → the attribution is wrong; report and stop.

## (c) Main-loop A/B — flagship 27B
IQ3_XXS, `--spec-type draft-mtp --spec-draft-n-max 2 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 --parallel 1`.
Both settings, 2 repeats + 1 discarded warm-up, speed_recert.md prompt set. Record decode t/s,
acceptance, mean_len, VRAM peak, SM clock, temp.
ACCEPTANCE MUST NOT MOVE (±0.005): stage3.md §5 reference 0.7813504823151125. If it moves → bug.
Then 3×200-token greedy transcripts (temp 0, top_k 1, seed 1234) both arms: expect NOT byte-identical
(MMQ q8_1 activation layout, different accumulation order); requirement = sane (coherent, no loops).
GATE: full battery (HE-164 + aider-polyglot editing) only if decode t/s improves ≥5%.

## (d) Speed-tier arm — 35B (CORRECTED paths/cell)
Model: /data/models/coding/q36_35ba3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf (NOT models/q35b/).
Certified cell is NOW: --n-cpu-moe 16 -t 8, MTP n=2 (cpu_wave1/RESULTS.md) — not ncm26.
Affected tensors: Q3_K + IQ3_XXS (dense via hunk 2, routed experts via hunk 3); IQ4_XS excluded by default.
Only the 24/40 GPU-resident expert layers are affected — scale expectation by GPU share.
If flat, retry with GGML_CUDA_MMQ_ALL_TYPES=1 before concluding.

## Expected numbers (27B, n=2; model reproduces measured 34.15 t/s to 0.06%)
| w (ms/tok) | t/s | vs stock |
| 8.01 | 34.2 | — |
| 6.00 | 36.1 | +5.7% |
| 4.00 | 38.3 | +12.0% |
| 2.00 | 40.7 | +19.1% |
| 1.00 | 42.0 | +23.0% |
Inverse: 35→w7.1, 36→6.1, 37→5.2, 38→4.2, 39→3.4, 40→2.6, 41→1.8, 42→1.0.
CROSS-CHECK: predict (c) from b1's w_on BEFORE running (c); disagreement >1 t/s = broken model.
Thresholds: w_on ≤4 → STRONG PASS (battery + upstream PR); 4-6 → partial (try ALL_TYPES / MIN_BATCH=3);
≥6.5 → MMQ tile padding at ne11=3 ate the saving — report and stop.

## Risks
1. Baseline drift (non-SWAR 8.01) — only w_off is valid. 2. MMQ tile padding at ne11=3 (main failure
mode; b1 answers it in 15 min). 3. VRAM: MMQ adds a q8_1 staging buffer; shipping cell at 11,830-11,958
of 12,288 MiB — if OOM, retry at c=8192. 4. Numerics not bit-identical by design; acceptance is the
canary. 5. Not compiled — box build is the compile check (magic statics in a .cu TU, C++17).

## Revert
cp /tmp/ggml-cuda.cu.pre-mmq ggml/src/ggml-cuda/ggml-cuda.cu   (v11 binaries were never touched)

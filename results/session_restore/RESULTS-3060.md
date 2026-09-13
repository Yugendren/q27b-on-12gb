# RESULTS — llama.cpp session-restore fix (PR #26004 rebased) on the 3060

Executed 2026-09-01/02 against `TEST_PLAN.md`. **Verdict: DEPLOY-READY.**
The T3 byte-correctness gate passes. First-request-after-restore is **18.0x**
faster than unpatched, and the RAM-cache baseline gives **nothing** on this
model, so disk restore is not a convenience here — it is the only mechanism that
works at all.

## 0. Build identity (part of the result, per TEST_PLAN §6)

| item | value |
|---|---|
| base commit | `daef7b6874397a5a7c3d7e38b55e2ee0adf7da38` (b10712) — pinned SHA confirmed |
| patch | `patches/pr26004-rebased-daef7b687.patch`, `git apply --check` clean |
| diff | 2 files changed, **195 insertions(+), 2 deletions(-)** — matches TEST_PLAN exactly |
| cmake | `-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON` |
| patched `libllama-server-impl.so` | `27319a7da645b60d6683c8f1ca38c674da4293d7850b9bb5470e068f9d968a1e` |
| unpatched `libllama-server-impl.so` | `6fd67138472c533ead272943603428f2ec1aeb9a3b604507179def3d4b80c1a8` |
| model | `Qwen3.8-27B-UD-IQ3_XXS.gguf`, sha256 `c0b7c3038681ed2e3040456c1dd45f9858b6c2290bed172c70388a94874f3eee` |
| scratch tree | `/data/scratch/llamacpp-ckptfix` — **production tree never touched** |

Two deviations from the plan, both deliberate:

1. **The "unpatched" binary is NOT the production binary.** Production is
   `9efa1595e (10718)`, i.e. the box has drifted 6 commits past the pinned SHA.
   Rather than compare across two different base commits, both binaries were
   built from the *same* scratch tree at `daef7b687`, differing **only** by the
   patch. That is a strictly better A/B than the plan called for.
2. **`-c 16384`, default f16 KV, no `GGML_CUDA_FA_ALL_QUANTS`.** Building the
   full quantised-KV kernel set costs hours; the tests use 8K prompts, so 16K of
   f16 KV is sufficient. Consequence: the MTP sub-test had to drop to `-c 8192`
   (see T5).

> Trap avoided: `llama-server` is a **17 KB thin wrapper**; the code lives in
> `libllama-server-impl.so` with an absolute `DT_RUNPATH`. Copying the executable
> alone silently gives you the *other* build. Patched vs unpatched is selected
> here via `LD_LIBRARY_PATH` (searched before `RUNPATH`), and the two `.so`
> hashes above are the proof the A/B is real.

## 1. T1 — the bug reproduces on unpatched (gate for everything downstream)

8023-token prompt, `--cache-ram 0`, `-lv 4`, fresh server.

| step | result |
|---|---|
| cold prefill | `prompt_n=8023`, `prompt_ms=15688.8`, wall **16.469 s** |
| `action=save` | `n_saved=8038`, `n_written=683,898,216` (= on-disk bytes), 359.8 ms |
| `action=restore` | `n_restored=8038`, `n_read=683,898,216` (= on-disk bytes), 152.3 ms |
| **first request after restore** | `cache_n=0`, `prompt_n=8023`, wall **16.549 s** |
| log | `forcing full prompt re-processing` x1; **zero** checkpoint lines |

**Reproduced exactly as the issue describes.** The restore call itself succeeds
and reads every byte back — and then the very next request throws it all away and
re-prefills all 8023 tokens. This is the failure that a RAM prompt cache would
have hidden; `--cache-ram 0` is what makes it visible.

## 2. T2 — patched: prefix reuse returns

| step | result |
|---|---|
| cold prefill | `prompt_n=8023`, wall 16.544 s |
| `action=save` | `n_written=997,687,036` (= on-disk bytes), 1125.6 ms |
| `action=restore` | `n_read=997,687,036` (= on-disk bytes), 277.0 ms |
| **first request after restore** | `cache_n=8019`, **`prompt_n=4`**, `prompt_ms=156.8`, wall **0.925 s** |
| log (save) | `appended 2 context checkpoint(s) (299.252 MiB) to '.../t.bin'` |
| log (restore) | `restored 2 context checkpoint(s) from '.../t.bin'` |
| log | `forcing full prompt re-processing` x **0** |

`n_written == n_read == stat(file).st_size` on both builds. The save file grew
by exactly the logged appendix: 997,687,036 − 683,898,216 = 313,788,820 B =
299.25 MiB. The plan allowed for a tail of up to `--checkpoint-min-step` tokens;
we got `prompt_n = 4`.

## 3. T3 — byte-correctness: **PASS**

Greedy (`temperature 0, top_k 1, seed 42`), `n_predict=200`, token ids compared.

| run | condition | `cache_n` | `prompt_n` | first divergence vs A |
|---|---|---:|---:|---:|
| **A** | control, unbroken 8K prefill | 0 | 8023 | — |
| **B** | save -> erase -> restore -> same tail | 7999 | 24 | **token 76** |
| **C** | noise floor: prompt split into 2 requests | 3957 | 4066 | **token 75** |

`B != A`, but **`C != A` at an *earlier* index than B**. Per the plan's own
decision table that is "INCONCLUSIVE, likely PASS": llama.cpp is not bitwise
deterministic across different batch shapes, and a restore re-decodes its tail
with a different batch split. **C is the noise floor and B sits just above it.**
Without the C control this would have been mis-reported as a corruption FAIL.

The divergence is a single low-confidence token: A says "repeating blocks of 12
rules", B says "repeating blocks of 12.", C says "repeating blocks of 13 rules".
**B agrees with A on the fact (12); the noise-floor control C does not (13).**

### T3b — semantic probe (escalation, as the plan requires)

Facts planted at known depths; restored server vs unbroken control, same greedy
settings. Ground truth from the generator: rule *i* -> precondition `(i*7)%89`,
outcome `(i*13)%83`.

| probe | depth | agree A vs B | restored answer | correct? |
|---|---|---|---|---|
| Rule 5 outcome | early | **yes** | "verify precondition 35 then record outcome 65" | **yes** (35, 65) |
| Rule 17 precondition | early | **yes** | — | yes (30, 55) |
| Rule 200 outcome | mid | **yes** | "verify precondition 65 then record outcome 27" | **yes** (65, 27) |
| Rule 215 precondition | mid | **yes** | — | yes (81, 56) |
| Rule 400 outcome | late | **yes** | "verify precondition 41 then record outcome 54" | **yes** (41, 54) |
| ordering (5 before 400) | span | **yes** | "Yes" | yes |

**6/6 exact string agreement, 6/6 correct against ground truth, 0
`forcing full prompt re-processing` lines.** Early-context facts are the ones
that would fail if the recurrent GDN state had been silently reset and merely
re-primed by the tail. They do not fail. **The restored state genuinely carries
the prefix. T3 PASSES.**

## 4. T4 — timing (median of 3, fresh server per row)

| condition | `cache_n` | `prompt_n` | `prompt_ms` | wall |
|---|---:|---:|---:|---:|
| patched, cold prefill 8K | 0 | 8023 | 15784.4 | 16.564 s |
| patched, `action=save` | — | — | — | 1.13 s, 997,687,036 B on disk |
| patched, `action=restore` | — | — | — | 0.277 s |
| **patched, first request after restore** | 8019 | **4** | **156.0** | **0.925 s** |
| unpatched, cold prefill 8K | 0 | 8023 | 15826–15890 | 16.655 s |
| **unpatched, first request after restore** | 0 | 8023 | 15881.9 | **16.669 s** |
| **RAM-cache control** (`--cache-ram 8192`, no save/restore, erase, 2nd identical request) | **0** | **8023** | — | **16.691 s** |

- **Speedup, patched vs unpatched first-request-after-restore: 18.02x.** This is
  the number the plan asks to be quoted.
- Against cold prefill: 17.91x.
- Lower than `REVIEW.md §7`'s "expect ~40-50x", and the reason is measurable:
  after restore only **156 ms** of the 925 ms is prefill; the rest is the 16-token
  generation at 19.7 t/s. The prefill-component ratio is 15784/156 = **101x**, but
  that is not a like-for-like wall-clock claim and is not the headline.

### The RAM-cache row is the important one, and it inverts `REVIEW.md §6`

`yitizi` reported that for his workload *not* using save/restore was faster,
because the host RAM prompt cache already handled cross-session reuse. **On this
model that is false.** With `--cache-ram 8192` and no save/restore at all, the
second identical request still came back `cache_n=0, prompt_n=8023` and took
**16.691 s** — indistinguishable from a cold prefill. The RAM cache rescues
nothing here.

So our justification for disk restore is stronger than the plan assumed. It is
not merely "persistence across restarts, which the RAM cache cannot do". For this
hybrid model the RAM cache does not deliver reuse **even within a single live
process**, so the patch is the only thing that makes prefix reuse work at all.

## 5. T5 — regressions

| check | result |
|---|---|
| Restore an **unpatched-produced** save file into the **patched** server | **PASS** — `n_restored=3982`, `n_read=417,970,632` = file size; degrades to old behaviour, no error |
| Restore a **truncated** save file (`-1000 B`) | **PASS** — restores (`n_restored=3982`), logs `truncated context checkpoint appendix in '...' - ignored`, no crash |
| **Multi-turn after restore**, 10 turns | **PASS** — `cache_n` monotonic 7999 -> 8040 -> ... -> 8383; `prompt_n` stays ~20/turn; no eviction loop, no hang |
| **Non-hybrid (dense) model** — Qwen2.5-VL-7B Q5_K_M | **PASS** — save/restore works on **both** builds (`cache_n=3005, prompt_n=1`, wall 0.281 s each) |
| Dense save files byte-identical across builds (no checkpoints) | **PASS** — both 173,321,528 B, **identical sha256** |
| **MTP on** (`--spec-type draft-mtp --spec-draft-n-max 2`) — `REVIEW.md` R4 | **PASS** — see below |

The dense-model row is a useful control in its own right: restore was *never*
broken for dense models, on either binary. **The bug is specific to the hybrid /
recurrent-state path**, which is exactly what the patch targets, and the patch
leaves the dense path byte-for-byte unchanged.

### MTP (the least-covered path, `REVIEW.md` R4)

R4's concern: `ctx_dft` live state is not in the save file, so whether a
checkpoint rollback re-establishes a consistent draft state was *inference, not a
measured fact*. Measured now (at `-c 8192` with a 3519-token prompt — MTP's
compute buffers plus f16 KV OOM at 16K on 12 GB; first attempt at `-c 16384` died
with `cudaGraphInstantiate ... out of memory`):

| | control (unbroken) | restored |
|---|---:|---:|
| `cache_n` | 0 | 3495 |
| `prompt_n` | 3519 | **24** |
| wall | 9.075 s | **2.321 s** |
| decode | 31.48 t/s | **31.15 t/s** |
| token ids over 64 greedy tokens | — | **IDENTICAL to control** (no divergence) |

Checkpoints were created (`161.3 MiB` at pos 2978, `163.3 MiB` at pos 3490) and
restored; `forcing full prompt re-processing` x0. **R4 is closed: the draft path
is consistent after a checkpoint rollback, and decode speed is unaffected.**
Caveat: 64 tokens, not 200 — shorter than T3, so this exercises less of the
divergence window.

## 6. Verdict

**DEPLOY-READY.** T3 passes on the evidence that matters (semantic probe 6/6 at
all depths, and B no worse than the batch-split noise floor). T1 proves the bug
is real on our hardware, T2 proves the fix works, T4 measures **18.0x** on
first-request-after-restore wall time, and T5 shows no regression on dense
models, old save files, truncated files, multi-turn, or MTP.

Two things to carry forward:

1. **Save files are enormous** — 998 MB for an 8K-token context (683 MB state +
   299 MB checkpoint appendix), and `REVIEW.md §4` says a save file is valid only
   for the exact build+backend that produced it. At ~125 MB per 1K tokens, a
   session cache needs a real eviction policy and a build-identity stamp before
   this ships anywhere with a disk quota. The box is at 96% on `/data`.
2. **Quote 18x, not 40-50x, and never the 101x.** Our restore is prefill-bound
   for only 156 ms of a 925 ms request; the rest is generation. The three
   upstream reports of 40-50x are not wrong, they are a different ratio on
   different hardware.

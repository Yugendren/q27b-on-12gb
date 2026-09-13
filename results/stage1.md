# STAGE 1 — STACK TEST: stock vs full stack on identical work

**The demo number: 19.4× wall-clock and 5/12 → 12/12 solved, on the same
12-task workload, on the same RTX 3060 12 GB.**

Rig: ollama box, RTX 3060 12 GB (sm_86) + Ryzen 5 3600, 2026-09-02.
Driver `stack_test.py`, raw records `results/stack_test/stack_test.json`,
per-arm server logs `results/stack_test/srv_{A,B,Bplus}.log`,
run log `results/stack_test_A.log`.

## Protocol

Both arms run the **identical workload**: HumanEval tasks 0-9 (system prompt,
scoring and code-extraction imported verbatim from
`platform_arm/harness/reliability.py`, so pass rates are on the same footing as
`results/platform_arm/*`), plus two long-context tasks built as synthetic
repository dumps with machine-checkable needles planted in them:

| task | prompt tokens | question |
|---|---:|---|
| `LC/1_retrieve` | 7,715 | report `RETENTION_LIMIT` and `AUDIT_CODE` of a named module |
| `LC/2_multihop` | 12,782 | follow a module's `DEPENDS_ON` reference, then report the target's two constants |

`LC/1` is deliberately sized to **fit inside arm A's 8,192-token window** with
its 400-token answer budget, so arm A gets a genuine shot at it. `LC/2` does
not fit in 8,192 at all — that is a capability difference and it is reported as
one, not hidden.

Client: `temperature=0, top_k=1, seed=1234, cache_prompt=false`,
`max_tokens=1024` on HumanEval (the legacy budget, so the thinking-ON arm is
allowed to spend it rather than being artificially truncated) and 400 on the
long-context tasks. GPU guard <500 MiB before each launch; VRAM certified from
the peak of a 1 Hz sample taken **during requests**; every request records
`reasoning_content` length and a `<think>` scan.

| | **Arm A — "what a normal user gets"** | **Arm B — our stack** |
|---|---|---|
| model | `Qwen3.8-27B-UD-Q4_K_S.gguf` (15.4 GB) | `Qwen3.8-27B-UD-IQ3_XXS.gguf` (10.9 GB) |
| offload | **partial** — no `-ngl`, the runtime's own fit splits it host/device | `-ngl 99`, fully resident |
| context / KV | 8,192, f16 | 16,384, q8_0 |
| speculation | none | MTP `n=2` |
| libraries | stock | kernel-offensive `LD_LIBRARY_PATH` preload (verified: the binary's `RUNPATH` is a RUNPATH, so `ldd` confirms the patched `libggml-cuda.so.0` / `libllama.so.0` are the ones loaded) |
| thinking | **ON** (the served Qwen3 template's default) | **OFF** (`enable_thinking=false`) |
| other | — | `--cache-ram 0` |

## 1. The composed table

| | **Arm A (stock)** | **Arm B (our stack)** | ratio |
|---|---:|---:|---:|
| **wall-clock, all 12 tasks** | **1,713.6 s (28 min 34 s)** | **88.6 s** | **19.35×** |
| **tasks solved** | **5 / 12** | **12 / 12** | +7 |
| HumanEval 0-9 pass | 4 / 10 | **10 / 10** | +6 |
| long-context pass | 1 / 2 | **2 / 2** | +1 |
| decode t/s (pooled) | 4.71 | **41.41** | 8.79× |
| mean wall-clock / HumanEval task | 165.7 s | **4.47 s** | 37.1× |
| mean completion tokens / HumanEval task | 772.1 | **160.2** | 4.8× fewer |
| total completion tokens, whole run | 7,855 | 1,636 | 4.8× fewer |
| effective throughput (tokens / total wall) | 4.58 | 18.48 | — |
| VRAM **peak during requests** | 10,868 MiB | 11,858 MiB | — |
| max `reasoning_content` length | 4,705 chars | **0** | — |

Per-task detail:

| task | A: pass / tok / wall | B: pass / tok / wall |
|---|---|---|
| HumanEval/0 | ok · 435 · 90.3 s | ok · 173 · 5.1 s |
| HumanEval/1 | **fail** · 1024 (cap) · 214.4 s | ok · 218 · 5.8 s |
| HumanEval/2 | ok · 583 · 121.9 s | ok · 97 · 2.8 s |
| HumanEval/3 | ok · 536 · 112.5 s | ok · 157 · 4.3 s |
| HumanEval/4 | **fail** · 1024 (cap) · 254.4 s | ok · 169 · 4.7 s |
| HumanEval/5 | **fail** · 1024 (cap) · 214.6 s | ok · 153 · 4.3 s |
| HumanEval/6 | **fail** · 915 · 191.4 s | ok · 200 · 5.4 s |
| HumanEval/7 | ok · 934 · 195.5 s | ok · 112 · 3.2 s |
| HumanEval/8 | **fail** · 578 · 121.4 s | ok · 163 · 4.5 s |
| HumanEval/9 | **fail** · 668 · 140.1 s | ok · 160 · 4.5 s |
| LC/1 (7.7K) | ok · 55.7 s | ok · 16.4 s |
| LC/2 (12.8K) | **HTTP 400 — prompt exceeds the 8,192 window** | ok · 27.2 s |

## 2. What actually produced the 19×

The gap is **not** one lever, and it is not mostly the kernel:

1. **Partial offload is the single largest term.** Q4_K_S is 15.4 GB on a
   12 GB card, so the runtime leaves layers on the host and decode collapses to
   **4.71 t/s**. Arm B's model is 10.9 GB and is fully resident. This alone is
   most of the 8.8× decode ratio — the 1.72× from MTP speculation and the +2.5%
   from the kernel patch ride on top of a fully-resident model, not the other
   way round.
2. **Thinking ON costs 4.8× the tokens *and* the tasks.** Arm A spends
   772 tokens/task against arm B's 160, and **3 of its 10 HumanEval attempts
   die on the 1,024-token cap mid-reasoning** — the same truncation mechanism
   `results/platform_arm/VERDICT.md` identified as the cause of the historical
   82.3% figure. This is why Arm A's HumanEval score is 4/10 while the same
   family of model scores 92.7-93.3% under the corrected protocol.
3. **Context is a hard capability wall.** Arm A cannot attempt `LC/2` at all;
   the server returns HTTP 400. Note that arm A cannot simply be given a bigger
   window either — at Q4_K_S on 12 GB, more context means *fewer* GPU layers
   and an even lower decode rate.

**Multiplicative structure:** 4.8× fewer tokens × ~8.8× faster per token
≈ 42× on per-task work, tempered to 19.35× overall by the two long-context
tasks (whose cost is prefill-dominated and where arm A's failure is free).

Two honest notes on arm B's numbers:

* **41.41 t/s is above the certified 34.39 t/s** and that is expected, not a
  contradiction. `results/speed_recert.md` §6 established that speculation
  speedup is a property of the content; the pooled acceptance here is
  **0.9918** (HumanEval completions are extremely draftable), against 0.781 on
  the certification's prose-ish code prompt. The shipping cert number stands;
  this is the same config on easier-to-draft content.
* **11,858 MiB peak on a 12,288 MiB card** — 430 MiB of headroom, measured
  during requests, on a run that included a 12.8K-token prefill.

**Thinking tripwire: PASSED for arm B** — mean completion 160.2 tok/task
(limit 350, 2.2× margin), `reasoning_len` 0 on all 12 requests, no `<think>`
leakage. Arm A's `think_leak=True` is by construction: thinking ON is what the
arm is testing.

## 3. Arm B+ — session restore on the long prompts

The session-restore build (`/data/scratch/llamacpp-ckptfix/build/bin/llama-server`,
from the 2026-09-02 overnight triple) still exists, so this arm ran.
Sequence per task: cold request (full prefill) → `POST /slots/0?action=save`
→ evict the slot with an unrelated prompt → `POST /slots/0?action=restore` →
re-ask.

| task | cold wall | cold prefill | **warm wall** | **warm prefill** | save | restore | prefill × | wall × |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LC/1 (7,715 tok) | 16.38 s | 15,841 ms | **0.65 s** | **165 ms** | 0.34 s | 0.22 s | **95.8×** | 25.2× |
| LC/2 (12,782 tok) | 27.12 s | 26,690 ms | **0.61 s** | **184 ms** | 0.39 s | 0.27 s | **145.0×** | 44.5× |

Charging the restore call to the warm path, end-to-end speedups are
**18.8×** and **30.8×** — consistent with the 18.0× already certified, and
better at the deeper prompt, as the mechanism predicts.

**Correctness:** the warm completion is **byte-identical** to the cold one on
both tasks (`content_sha1` 40ff43ef078d3213 and the LC/2 pair match exactly),
and both answers are correct. Attribution is clean: `--cache-ram 0` disables
the RAM prompt cache, so the 165 ms warm prefill is the slot restore and
nothing else.

**Caveat that must travel with this row:** arm B+ runs the **session-restore
build's own libraries**, so it does **not** carry the kernel-offensive patch —
setting `LD_LIBRARY_PATH` to the kernel libs would replace the very `libllama.so`
that contains the restore fix. The two patch sets have never been merged into
one build. B+ is therefore "arm B minus the kernel patch (−2.5%), plus session
restore", not a strict superset of arm B.

## 4. Verdict

**STAGE 1 PASS. The product number is 19.4× wall-clock at 12/12 vs 5/12** on
identical work, on one 12 GB consumer card. With session restore on repeat
long-context traffic the long-prompt legs go a further 18.8-30.8×.

Ordering of the levers, by contribution, for anyone quoting this:
partial-offload elimination (model that fits) > thinking OFF (4.8× tokens,
+6 solved) > context capacity (one task is otherwise impossible) >
MTP speculation (1.72×) > kernel patch (+2.5%).

### Not covered / open

* 10 HumanEval tasks, not 164 — this is a wall-clock demo, not a quality cert;
  the quality certs live in `results/platform_arm/`.
* Arm A uses llama.cpp's own layer-fit rather than ollama's estimator; the
  split it chose (10,714 MiB at load) is representative but not byte-for-byte
  what ollama would pick.
* Single request at a time, `--parallel 1`, both arms.

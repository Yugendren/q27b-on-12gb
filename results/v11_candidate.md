# v1.1 candidate — merged build, certification pass

Date: 2026-09-02. Box: `ollama`, RTX 3060 12GB, driver 535.309.01, sm_86.

**Verdict: all three gates PASS. The v1.1 candidate is cleared for deployment.**
Copied to `/data/projects/q27b_on_12gb/llama-server-v11/`.

**New best measured shipping number: 40.10 t/s at 11,502 MiB peak, HumanEval
25/25.** That beats the ~38 t/s estimate, and the quality result is stronger than
"~93%": the candidate's completions were **byte-identical to the certified
baseline on 25/25 tasks**.

---

## 1. The build

Scratch clone of `ggml-org/llama.cpp` at `daef7b6874397a5a7c3d7e38b55e2ee0adf7da38`
(= b10712), both patch sets applied, CUDA arch 86, Release, `LLAMA_CURL=OFF`.

**Patch overlap: none.** The two patches touch disjoint files, so no conflict
resolution was needed — `git apply --check` passed for the second patch with the
first already applied.

| patch | files |
|---|---|
| `mmvq_iquant_decode.patch` | `ggml/src/ggml-cuda/mmvq.cu`, `ggml/src/ggml-cuda/vecdotq.cuh` |
| `pr26004-rebased-daef7b687.patch` | `tools/server/server-context.cpp`, `tools/server/tests/unit/test_slot_save.py` |

Combined diffstat: **4 files changed, 263 insertions(+), 26 deletions(-)**.

    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=86
    -- Using CMAKE_CUDA_ARCHITECTURES=86 CMAKE_CUDA_ARCHITECTURES_NATIVE=86-real

Build succeeded (exit 0). `llama-server --version` →
`0.3.0-dev (build 10712, commit daef7b687)` as first built.

*Provenance wrinkle:* after `revv install.sh` re-ran cmake configure over the same
tree, the reported build number changed to `build 1` (git-describe metadata is
absent in the seeded clone). The **commit is correct and unchanged**
(`daef7b687`), and the deployed binary reports `build 1`. Verify this build by
commit and by the sha256 table below, not by the build number.

**Packaging note:** this llama.cpp builds shared libraries — `llama-server` is an
18 KB wrapper and the server code lives in `libllama-server-impl.so`. The
`TEST_PLAN.md` §2 check `strings build/bin/llama-server | grep -c "context
checkpoint(s) from"` therefore returns **0**, which looks like a failed patch and
is not. The string is present (count 1) in `libllama-server-impl.so`. The
deployed directory ships the full runtime set plus a wrapper, because `RUNPATH`
is an absolute path into the build tree rather than `$ORIGIN`.

Provenance (deployed copies):

| file | sha256 |
|---|---|
| `llama-server` | `77250ee5f3fa8e26aff7971b482beb7fbca1e85f3a31bb6973d8d0ce4cd658d3` |
| `libllama-server-impl.so` | `3774e1ad8c2ebbd986e73677e9ce8bc198d928e067e310263f6df40fd3d3d70e` |
| `libggml-cuda.so.0.22.0` | `7aece5e0c1966855fecd849d97e58fd734d571dfc25c3b7fe81e99fe17f5bc9c` |

---

## 2. Gate (a) — kernel gate: byte-identical greedy output vs stock — **PASS**

Stock reference: `/data/projects/llama.cpp/build/bin/llama-server`, the clean
production tree at the same pinned SHA, also built arch 86 / Release.

Both servers run on the flagship `Qwen3.8-27B-UD-IQ3_XXS.gguf` (an IQ-quantized
model, so the patched `vec_dot_iq*` paths are actually exercised), `-c 4096`,
`-fa on`, `-ctk/-ctv q8_0`, greedy (`temperature=0, top_k=1, seed=42`),
`cache_prompt=false`, `return_tokens=true`, 128 tokens per prompt.

| prompt | token ids identical | text identical | length |
|---|---|---|---|
| `def fibonacci(n):` | **yes** | yes | 128/128 |
| `The capital of France is` | **yes** | yes | 128/128 |
| `Write a SQL query that selects all users older than 30…` | **yes** | yes | 128/128 |

**3/3 byte-identical. The kernel patch is output-neutral.**

## 3. Gate (b) — session restore T2 + T3 — **PASS**

Config: merged build, flagship model, `-c 16384 -ngl 99 --jinja -np 1 -fa on
-ctk/-ctv q8_0 --cache-ram 0 --ctx-checkpoints 8 --checkpoint-min-step 2048
--slot-prompt-similarity 0.1 -lv 4`. Fresh server per condition.

The `--cache-ram 0` and `-lv 4` traps from `REVIEW.md` §6 were both honoured.

*Correction to the plan:* `TEST_PLAN.md`'s 900-rule probe prompt assumes ~8K
tokens; on this tokenizer it is **16,798 tokens** and is rejected outright at
`c=16384`. Reduced to 420 rules → 7,790 tokens.

### T2 — prefix reuse returns: PASS

    created context checkpoint 1 of 8 (pos_min = 7273, n_tokens = 7274, 149.626 MiB)
    created context checkpoint 2 of 8 (pos_min = 7785, n_tokens = 7786, 149.626 MiB)
    appended 2 context checkpoint(s) (299.252 MiB) to '.../t.bin'
    restored 2 context checkpoint(s) from '.../t.bin'
    restored context checkpoint (pos_min = 7785, n_past = 7786, 149.626 MiB)

| condition | cache_n | prompt_n | prompt_ms |
|---|---:|---:|---:|
| A — cold prefill (control) | 0 | 7,790 | 15,264 |
| B — cold prefill | 0 | 7,790 | 15,366 |
| **B — first request after restore** | **7,786** | **4** | **149** |
| C — split prefill (noise floor) | 3,844 | 3,946 | 8,089 |

`n_written == n_read == on-disk size == 749,052,320 bytes` — exact, all three.

Prefill on the first request after restore: **15,366 ms → 149 ms (103x)**. Note
`prompt_n = 4`, better than the "tail of up to `--checkpoint-min-step`" the plan
allowed for.

### T3 — byte-correctness (THE GATE): PASS, unambiguous

Greedy everywhere (`temperature=0, top_k=1, seed=42, n_predict=200`), comparing
**token ids**, not text.

| pair | first divergence |
|---|---|
| **B vs A (restored vs unbroken)** | **none — identical, 200/200 token ids** |
| C vs A (noise floor) | index 26 |

This is the top row of the plan's interpretation table: `B == A` exactly.
Crucially C is *not* identical to A — a different batch split does change the
output at token 26 — which proves the comparison is not vacuous and that B's
exactness is a real result rather than an artifact of a deterministic setup.

T3b was not needed.

---

## 4. The v1.1 candidate measurement

Config: `Qwen3.8-27B-UD-IQ3_XXS-ASCII.gguf` (10,379,561,824 bytes, vocab 127,947,
MTP head present) + merged build + `--spec-type draft-mtp --spec-draft-n-max 2` +
`-ctk/-ctv q8_0` + `-c 16384` + `-fa on` + `--parallel 1`, thinking off.

Harness: `revv bench` protocol — 4 requests, 400 new tokens, greedy
(`temperature=0, top_k=1, seed=1234`), `cache_prompt=false`, warmup excluded,
decode rate taken from llama-server's own `predicted_per_second`.

### Speed and VRAM

    request 1    40.32 t/s    400 tokens   10.22 s wall
    request 2    40.16 t/s    400 tokens   10.26 s wall
    request 3    40.01 t/s    400 tokens   10.30 s wall
    request 4    39.91 t/s    400 tokens   10.33 s wall

    decode      40.10 t/s mean, 40.09 median      spread 1.0%

| | decode t/s | peak VRAM |
|---|---:|---:|
| **v1.1 candidate (ASCII + merged)** | **40.10** | **11,502 MiB** |
| ASCII + stock kernel | 38.92 | 11,500 MiB |
| flagship + merged kernel | 37.86 | 11,830 MiB |

All three measured back-to-back on the same box with the same harness, so the
deltas are like-for-like:

| effect | measured here | prior art | agreement |
|---|---:|---:|---|
| kernel patch (on ASCII) | **+3.03%** | +2.5% (`FINDINGS.md:658`) | consistent |
| ASCII prune (on merged) | **+5.92%**, −328 MiB | +5.73%, −332 MiB (`stage3.md` §6) | excellent |

**40.10 t/s beats the ~38 t/s estimate.** The estimate compounded the two effects
off a 34.4 base; both effects reproduced almost exactly, but the base under this
harness is higher (the flagship reads 37.86 t/s here, not 34.4 — a harness
difference, see caveat below). Peak VRAM 11,502 MiB reproduces `stage3.md`'s
11,498 MiB to within 4 MiB.

Headroom: 11,502 of 12,044 usable → **542 MiB free**, versus 214 MiB for the
flagship. The prune is what makes this config comfortable rather than knife-edge.

### Quality — 25-task HumanEval spot-check

`reliability_probe.py --max-attempts 1 --limit 25 --concurrency 1 --max-tokens
1024 --temp-retry 0.8 --top-p 0.95` (corrected protocol: thinking off,
fixed fence extraction), against the running candidate.

| metric | v1.1 candidate | certified baseline (`speed_recert/tpt_thinkoff.json`) |
|---|---:|---:|
| pass@1 | **25/25 = 100%** (CI95 86.7–100%) | 25/25 = 100% |
| mean completion tokens/task | **158.76** | 158.76 |
| total completion tokens | **3,969** | 3,969 |
| wall time | **106.95 s** | 119.80 s |

**The completions are byte-identical on 25/25 tasks.** Verified by diffing the
per-task JSONL against `results/speed_recert/tpt_thinkoff.jsonl`, not merely by
comparing summary statistics.

This is a stronger result than the ~93% that was expected, and it is worth being
precise about why. It means the ASCII prune, the merged kernel, and revv's
substitution of the GGUF's embedded chat template for the froggeric template are
**jointly output-neutral** on this workload — the candidate is not "as good as"
the baseline, it is emitting the same tokens, 10.7% faster in wall time.

The 158.76 mean is far below the 350 tok/task tripwire, independently confirming
thinking was off.

*Scope:* 25 tasks bounds pass@1 only to ~[86.7%, 100%]. The byte-identity result
is the load-bearing evidence here, not the 100% itself. A full 164-task run is
still required before quoting a headline accuracy number for v1.1; the standing
164-task references are 92.68% (flagship) and 93.29% (ASCII), statistically
indistinguishable (`stage2.md` §2a).

---

## 5. Caveats

1. **40.10 vs 34.4 are different harnesses.** The `revv bench` protocol used here
   reads the flagship at 37.86 t/s where `speed_recert` reads 34.39 t/s — a ~10%
   harness gap (different prompt, hence different MTP acceptance). **Do not
   compare 40.10 against 34.39.** The internally consistent comparison is the
   three-row table in §4, all taken with one harness in one session.
2. **Noise floor on this box is ±1%** (`speed_recert.md` standing rule 4). The
   +3.03% kernel delta clears it; treat anything under ~2% as no difference.
3. The ASCII prune is **ASCII/English+code only** by construction. Any non-ASCII
   workload is out of scope for this config.
4. `--chat-template-kwargs` for `enable_thinking` is **deprecated upstream** at
   this commit (warns on every start, recommends `--reasoning off`). It works
   correctly now; migrate before un-pinning llama.cpp.
5. T5 regression items from `TEST_PLAN.md` (unpatched→patched save files,
   truncated appendix, 10-turn multi-turn, restore with MTP enabled) were **not
   run** — this was the quick T2+T3 gate only. R4 (MTP + checkpoint interaction)
   remains the least-covered path and is still inference, not measurement.

---

## 6. Deployment

    /data/projects/q27b_on_12gb/llama-server-v11/
      llama-server-v11      <- wrapper; sets LD_LIBRARY_PATH, use this
      llama-server          <- 18 KB wrapper binary
      lib*.so*              <- full runtime set (222 MB total)

Verified self-contained by running `--version` from `/`. Patched marker present
in the deployed `libllama-server-impl.so`.

Suggested launch line for the v1.1 candidate:

    /data/projects/q27b_on_12gb/llama-server-v11/llama-server-v11 \
      -m /data/projects/q27b_on_12gb/models/unsloth-q27b/Qwen3.8-27B-UD-IQ3_XXS-ASCII.gguf \
      -ngl 99 -fa on -c 16384 -ctk q8_0 -ctv q8_0 \
      --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1 \
      --jinja --chat-template-kwargs '{"enable_thinking":false}' \
      --host 127.0.0.1 --port 8080

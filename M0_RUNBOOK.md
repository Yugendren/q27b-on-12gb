# M0 Runbook — Qwen3.8-27B Inference Baselines

**Compiled:** 2026-08-23 · **Status:** execution-ready · **Scope:** M0 baseline truth only (measure what exists today; no search, no tuning)

Targets:
- **Rig A** — RTX 3060 12GB (Ampere, `sm_86`), Linux, 32 GB system RAM
- **Rig B** — RTX 5080 16GB (Blackwell, `sm_120`), Linux

Runtimes: llama.cpp (recent build) and exllamav3 / TabbyAPI.

Frozen gate from `PROJECT_CONTRACT.md`: **peak VRAM ≤ 11.5 GB incl. KV cache**, ≥15 raw tok/s, ≥25 code-effective tok/s on Rig A.

---

## READ THIS FIRST — six premises in the M0 brief that did not survive verification

Every one of these was checked against primary sources. Do not skip.

| # | Premise as briefed | Verified status |
|---|---|---|
| P1 | "Gated DeltaNet CUDA bug fixed around build ~10450" | **Disproven as a *fix* point.** `b10448...b10450` contains exactly two commits, both UI-only. b10450 is where one reporter happened to land, not where anything was fixed. No PR is attributable. Treat **b10450+ as an empirical floor only**. [compare](https://github.com/ggml-org/llama.cpp/compare/b10448...b10450) |
| P2 | "the `--jinja` flag requirement" | **Outdated.** `--jinja` is **default-enabled** on current master. The load-bearing flag is `--chat-template-file`. [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) |
| P3 | "MTP flags `--spec-type draft-mtp`, `--draft-n`" | `--spec-type draft-mtp` is **real**; **`--draft-n` and `--mtp-model` do not exist.** Correct flag is `--spec-draft-n-max` (default 3). [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) |
| P4 | "load the 3.00bpw EXL3 revision on the 3060" | **Does not fit.** 3.00bpw weights alone are **12.87 GiB** — larger than the whole 12 GiB card and above the 11.5 GB M0 gate. See §4. |
| P5 | "`--target-bpw` … from discussion #18531 / **merged PR**" | **Not merged.** [PR #15550](https://github.com/ggml-org/llama.cpp/pull/15550) is still **open and in draft** after ~12 months. `--target-bpw` **does not exist in llama.cpp master.** You must build a fork to use it. See §6. |
| P6 | "MTP head is probably not usable in exllamav3" | **It is usable.** turboderp confirmed MTP support directly; the EXL3 quant carries `mtp_bits: 4`. Enable with `draft_mode: mtp`. See §4. |

---

## 1. ARTIFACT TABLE

All sizes are **exact bytes** read from the Hugging Face model API (`?blobs=true`) on 2026-08-23, cross-checked against the rendered file tree. GiB = bytes / 2^30.

### 1a. unsloth/Qwen3.8-27B-GGUF
Repo: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF> · tree: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main>

| File | Bytes | GiB | Fits 12 GiB card? |
|---|---:|---:|---|
| `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 9,828,981,664 | 9.15 | yes (weights only) |
| `Qwen3.8-27B-UD-IQ3_XXS.gguf` | 10,934,860,704 | 10.18 | yes (weights only) |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13,146,393,504 | 12.24 | no — needs offload |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,559,178,144 | 16.35 | no — needs offload |
| `Qwen3.8-27B-UD-Q4_K_S.gguf` | 15,358,213,024 | 14.30 | no — offload recipe §3 |
| `Qwen3.8-27B-Q8_0.gguf` | 29,047,086,048 | 27.05 | quality-reference candidate |
| `Qwen3.8-27B-UD-Q8_K_XL.gguf` | 31,457,991,680 | 29.30 | quality-reference candidate |
| `imatrix_unsloth.gguf` | 13,642,656 | 0.0127 | **imatrix — see §6** |
| `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 1,369,590,656 | 1.28 | MTP draft head |
| `BF16/Qwen3.8-27B-BF16-0000{1,2}-of-00002.gguf` | 49,986,159,616 + 4,671,576,000 | 50.90 | reference logits, §5 |
| `mmproj-F16.gguf` | 927,607,488 | 0.864 | **EXCLUDE — see §8 G1** |
| `mmproj-BF16.gguf` | 931,146,432 | 0.867 | **EXCLUDE — see §8 G1** |

> ⚠️ **Discrepancy to be aware of.** The 3060 recipe in §3 points `-m` at `Qwen3.8-27B-Q4_K_S.gguf` inside an `unsloth/` folder, but **this repo publishes no plain `Q4_K_S`** — its only non-`UD` quants are `Q4_0`, `Q4_1`, `Q8_0`. The poster's file is almost certainly `Qwen3.8-27B-UD-Q4_K_S.gguf` (14.30 GiB) renamed locally. `bartowski` does publish a true `Q4_K_S` (16,713,148,000 B / 15.57 GiB) if you want the literal file. **Record which one you actually ran.**

### 1b. bartowski/Qwen3.8-27B-GGUF
Repo: <https://huggingface.co/bartowski/Qwen3.8-27B-GGUF>

| File | Bytes | GiB |
|---|---:|---:|
| `Qwen3.8-27B-IQ2_S.gguf` | 10,295,330,400 | 9.59 |
| `Qwen3.8-27B-IQ3_XXS.gguf` | 12,626,773,600 | 11.76 |
| `Qwen3.8-27B-Q8_0.gguf` | 29,116,388,960 | 27.12 |
| `Qwen3.8-27B-imatrix.gguf` | 13,642,688 | 0.0127 |
| `Qwen3.8-27B-calibration-v6.txt` | 1,258,850 | — |
| `mmproj-Qwen3.8-27B-f16.gguf` | 927,607,008 | 0.864 |
| `mmproj-Qwen3.8-27B-bf16.gguf` | 931,145,952 | 0.867 |

Note: bartowski's `IQ2_S` (9.59 GiB) is **larger** than unsloth's `UD-IQ2_S` (8,371,970,048 B / 7.80 GiB) — different recipes under the same nominal name. Never compare across publishers by quant name; compare by measured bytes and measured KL.

### 1c. ggml-org/Qwen3.8-27B-GGUF (official auto-conversion)
Repo: <https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF>

| File | Bytes | GiB |
|---|---:|---:|
| `Qwen3.8-27B-Q4_K_M.gguf` | 18,973,870,432 | 17.67 |
| `Qwen3.8-27B-Q8_0.gguf` | 28,595,763,552 | 26.63 |
| `Qwen3.8-27B-BF16.gguf` | 53,808,281,952 | 50.11 |
| `mtp-Qwen3.8-27B-BF16.gguf` | 5,946,009,888 | 5.54 |
| `mtp-Qwen3.8-27B-Q8_0.gguf` | 3,164,006,688 | 2.95 |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1,680,271,648 | 1.56 |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 931,145,888 | 0.867 |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 629,247,008 | 0.586 |

Note the official `Q4_K_M` is **17.67 GiB vs unsloth's `UD-Q4_K_M` at 16,464,440,224 B (15.33 GiB)** — a 2.3 GiB spread at the same nominal quant. This is exactly the "human hand-tuning" baseline spread M0 exists to measure.

### 1d. turboderp/Qwen3.8-27B-exl3
Repo: <https://huggingface.co/turboderp/Qwen3.8-27B-exl3>

**EXL3 bpw variants are git BRANCHES, not files.** `main` holds only README + calibration traces. Verified branches: `2.00bpw`, `2.50bpw`, `3.00bpw`, `3.50bpw`, `4.00bpw`, `5.00bpw`, `6.00bpw`, plus `SC_*` variants (`SC_1.40/1.60/1.80/2.00/2.20bpw_H3`, `SC_3.00bpw_H4`, `SC_4.00bpw_H5`).

| Revision | safetensors bytes (shard1 + shard2) | Total GiB | Rig A (12 GiB) | Rig B (16 GiB) |
|---|---:|---:|---|---|
| `2.50bpw` | 8,582,745,470 + 3,714,792,495 | **11.45** | ✗ at M0 gate (see §4) | ok |
| `3.00bpw` | 8,575,532,487 + 5,244,406,822 | **12.87** | **✗ exceeds card** | ok |
| `3.50bpw` | 8,530,774,298 + 6,807,634,163 | **14.29** | ✗ | tight, Q4 KV only |

Internal consistency check: the deltas are 1.522 GB and 1.518 GB per 0.5 bpw — implying ~24.3 B quantized params and ~4.7 GB of fixed-precision overhead (embeddings, lm_head, vision tower). Coherent; sizes are trustworthy.

### 1e. Download commands

`huggingface-cli` was **removed in `huggingface_hub` v1.0**; the current binary is `hf`. Use `hf download`. ([migration guide](https://huggingface.co/docs/huggingface_hub/en/concepts/migration), [CLI docs](https://huggingface.co/docs/huggingface_hub/en/guides/cli))

```bash
pip install -U "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
export MODELS=/data/models          # adjust

# --- unsloth GGUFs (note: --include patterns, NOT whole-repo pulls) ---
hf download unsloth/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-UD-Q4_K_XL.gguf" \
  --local-dir "$MODELS/unsloth-q27b"

hf download unsloth/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-UD-IQ3_XXS.gguf" "Qwen3.8-27B-UD-Q2_K_XL.gguf" "Qwen3.8-27B-UD-Q3_K_XL.gguf" \
  --local-dir "$MODELS/unsloth-q27b"

# the Q4_K_S used by the §3 recipe + the MTP draft head + the imatrix
hf download unsloth/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-UD-Q4_K_S.gguf" "MTP/mtp-Qwen3.8-27B-Q4_0.gguf" "imatrix_unsloth.gguf" \
  --local-dir "$MODELS/unsloth-q27b"

# --- bartowski ---
hf download bartowski/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-IQ2_S.gguf" "Qwen3.8-27B-IQ3_XXS.gguf" \
            "Qwen3.8-27B-imatrix.gguf" "Qwen3.8-27B-calibration-v6.txt" \
  --local-dir "$MODELS/bartowski-q27b"

# --- official ggml-org Q4_K_M + Q8_0 reference candidate ---
hf download ggml-org/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-Q4_K_M.gguf" \
  --local-dir "$MODELS/ggml-q27b"

hf download ggml-org/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-Q8_0.gguf" \
  --local-dir "$MODELS/ggml-q27b"

# --- EXL3: one branch per bpw, via --revision ---
hf download turboderp/Qwen3.8-27B-exl3 --revision 3.00bpw --local-dir "$MODELS/exl3-3.00bpw"
hf download turboderp/Qwen3.8-27B-exl3 --revision 2.50bpw --local-dir "$MODELS/exl3-2.50bpw"
hf download turboderp/Qwen3.8-27B-exl3 --revision 3.50bpw --local-dir "$MODELS/exl3-3.50bpw"
```

**Do not download mmproj.** Every `--include` above is explicit precisely so the ~0.87 GiB `mmproj-*` never lands on disk or in VRAM. See §8 G1.

Disk budget: the GGUF set above is ≈ 95 GiB; adding one Q8_0 reference ≈ 122 GiB; adding a BF16 reference ≈ 172 GiB.

---

## 2. LLAMA.CPP REQUIREMENTS

### 2a. Minimum build — empirical floor, no verified fix commit

The brief's "fixed around build ~10450" **does not hold as stated.** Verified facts:

- The bug report is [discussion #27164](https://github.com/ggml-org/llama.cpp/discussions/27164) (author `infinitelayerworks`, 2026-08-16). Symptom: **model loads fine, runs at full GPU speed, emits garbage** — e.g. `/Q i`, `ance iurnesNSE){''),('-ract`. Same GGUF on the **Vulkan** backend produced correct text, isolating the fault to the CUDA path for Qwen3.8's Gated DeltaNet layers.
- Reporter's broken checkout `221f0f6` → working checkout `ece963f` ("approximately build 10450").
- **But** `221f0f6` is *"metal : add SILU_BACK (#25982)"* — a Metal commit, i.e. just an arbitrary checkout point. And [`b10448...b10450`](https://github.com/ggml-org/llama.cpp/compare/b10448...b10450) contains only `0d9ceae` (MCP UI) and `ece963f` (settings UI masking) — **zero CUDA/ggml changes**.

**Conclusion:** the real fix is somewhere in the unattributed range between those checkouts. Use **b10450 or newer** as a floor, and **verify empirically on each rig** with the smoke test in §2f rather than trusting a build number.

⚠️ **UNVERIFIED:** the PR that actually fixed the Gated DeltaNet CUDA corruption, and any publisher-stated minimum llama.cpp version for Qwen3.8. Neither the unsloth model card nor the MTP hub states one.

### 2b. Build commands

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
# If you have an existing shallow clone, this is mandatory — see §8 G2:
git fetch --unshallow 2>/dev/null; git pull origin master
git log -1 --format='%H %cd'   # RECORD THIS in results/

# Rig A — RTX 3060, sm_86
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"

# Rig B — RTX 5080, sm_120 (Blackwell needs CUDA >= 12.8)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
```

`-DGGML_CUDA_FA_ALL_QUANTS=ON` is **not optional for this project** — see §8 G5.

CUDA 12.8 is the first release with `sm_120` Blackwell targets. If cmake picks up an older nvcc from `PATH`, pin it with `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`. Known Blackwell build hazard: MXFP4 template instances can fail ptxas on `sm_120` ([issue #19662](https://github.com/ggml-org/llama.cpp/issues/19662)); a `sm_120` workaround thread is [issue #22696](https://github.com/ggml-org/llama.cpp/issues/22696). Also open and unresolved: [issue #27329](https://github.com/ggml-org/llama.cpp/issues/27329), Qwen3.8-27B **NVFP4 decode hangs on Blackwell** — avoid NVFP4 on Rig B at M0.

### 2c. `--jinja` — default-enabled, so this is not the real requirement

Verbatim from [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md):

```
--jinja, --no-jinja    whether to use jinja template engine for chat (default: enabled)
```

Pass `--jinja` explicitly anyway — it is free, self-documenting, and protects you if you ever run an older binary where it was opt-in. The flag that actually changes behaviour is `--chat-template-file`.

### 2d. Chat-template bug and the community-fixed template

**Two distinct verified bugs.**

**Bug A — mid-conversation system message → HTTP 500.** The official template raises `"System message must be at the beginning."`, crashing agentic clients that inject a developer/system turn mid-conversation.
Source: [unsloth/Qwen3.8-27B-GGUF discussion #42](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/42) ("Chat template issue and ways to solve it", author `Snol-std`).

**Bug B — think-tag poisoning, KV invalidation, fatal stall.** The official 3.8 template injects duplicate blank `<think></think>` into history, throws a fatal exception when `enable_thinking=false`, and defaults reasoning effort to `xhigh`.
Source: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/blob/main/README.md). Corroborated by [Qwen/Qwen3.8-27B discussion #68](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/68).

**Fixed file location:** `chat_template.jinja` at the **repository root** of `froggeric/Qwen-Fixed-Chat-Templates`. Per its README, that single file covers all Qwen 3.5 / 3.6 / 3.8 sizes.

```bash
hf download froggeric/Qwen-Fixed-Chat-Templates \
  --include "chat_template.jinja" --local-dir "$MODELS/templates"

# apply it:
--jinja --chat-template-file "$MODELS/templates/chat_template.jinja" \
--reasoning-format deepseek --reasoning-preserve
```

**M0 discipline:** the template changes token counts and therefore tok/s and quality. Use the **same template for every baseline** and record its SHA256 in `results/`.

### 2e. MTP speculative decoding — verified flags

[PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) *"llama + spec: MTP Support"* by `am17an`, **merged 2026-05-16**. (The MTP hub README says "July 2026" — that date is wrong; trust the PR page.) The PR adds the `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` enum value; `--spec-type` itself pre-existed. Reported: ~1.85–1.90x decode at ~75% acceptance with 3 draft tokens on Qwen3.6, with a **prompt-processing regression** from device-to-host embedding transfers.

Verbatim flag documentation from the server README:

```
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,
             ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
                              comma-separated list of types of speculative decoding to use (default: none)
--spec-draft-model, -md       draft model for speculative decoding (default: unused)
--spec-draft-n-max N          number of tokens to draft for speculative decoding (default: 3)
--spec-draft-n-min N          minimum number of draft tokens to use for speculative decoding (default: 0)
--spec-draft-p-min P          minimum speculative decoding probability (greedy) (default: 0.00)
--spec-draft-p-split P        speculative decoding split probability (default: 0.10)
--spec-draft-ngl, -ngld       max. number of draft model layers to store in VRAM
```

**`--draft-n` and `--mtp-model` do not exist. Do not use them.**

### 2f. Recommended draft-n (community MTP hub)

[github.com/sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) — *"One llama.cpp flag unlocks +33-39% decode speed for Qwen3.8-27B on consumer GPUs."* **Default branch is `master`, not `main`** (`/blob/main/` 404s).

Its documented launch command:

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf -c 131072 -ngl 999 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
```

Recommended `--spec-draft-n-max`:

| Class | n-max |
|---|---|
| 24 GB (3090/4090) | **2** |
| 32 GB+ / faster cards | **3–4** |
| bandwidth-starved APUs | 2, with `--spec-draft-p-min 0.60–0.75` |
| multi-GPU tensor-parallel | re-sweep from scratch |

The hub explicitly warns *"the n-max sweet spot is card- and topology-dependent"* and that a config which paid on a bandwidth-starved APU **inverted below baseline** on a 960 GB/s card. **Neither rig here is in the table** — Rig A (12 GB, ~360 GB/s) is below the smallest listed class. **Sweep `--spec-draft-n-max` ∈ {1,2,3,4} on both rigs and record acceptance rate.** Do not inherit n-max=2 as a default.

⚠️ A "n-max up to 12 for APUs" figure circulates from this README; it is **one benchmark table row** (GMK EVO-X2 / Ryzen AI Max+ 395, ROCm, paired with `--spec-type draft-mtp,ngram-mod --spec-ngram-mod-n-min 24`), **not** general guidance.

### 2g. Correctness smoke test — run before ANY measurement

The Gated DeltaNet bug produces *fast, well-formed, garbage* output; it will not announce itself. On each rig, before benchmarking:

```bash
./build/bin/llama-cli -m "$MODELS/unsloth-q27b/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
  -ngl 99 -c 4096 --temp 0 -n 128 \
  -p "Write a Python function that returns the nth Fibonacci number." \
  --jinja --chat-template-file "$MODELS/templates/chat_template.jinja"
```

Then repeat with `-DGGML_VULKAN=ON` build or `--device none` (CPU). **If CUDA output differs qualitatively from CPU/Vulkan output, your CUDA build is bad — stop and rebuild.** This is the only reliable detector.

---

## 3. THE 3060 COMMUNITY RECIPE — verbatim

Source: [unsloth/Qwen3.8-27B-GGUF discussion #61](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/61) — *"Running Qwen3.8-27B Q4_K_S on an RTX 3060 12GB with 96K Context + MTP (~10 t/s)"*, author `Hjx2`.

Reported results: **~225 tok/s prompt processing, ~9.7 tok/s generation, ~88% MTP acceptance, ~11.04 GB VRAM, 96K context.**

**Reproduced verbatim from the thread (original is a Windows `.bat`):**

```bat
@echo off
title Llama Server - QWEN 3.8
:: Set the model path

set MODEL_PATH=D:\llm\model\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_S.gguf

llama-server.exe ^
-m "%MODEL_PATH%" ^
--jinja ^
--reasoning-preserve ^
--chat-template-kwargs "{\"reasoning_effort\":\"medium\"}" ^
-fa on ^
-fit off ^
-ngl 99 ^
--override-tensor "blk\.([0-9]|[1-3][0-9]|4[0-5])\.ffn_.*=CPU" ^
-ctk q4_0 ^
-ctv q4_0 ^
--gpu-layers-draft all ^
--spec-type draft-mtp ^
--spec-draft-n-max 2 ^
-lv 4 ^
--no-mmproj ^
-np 1 ^
--temp 1.0 ^
--top-p 0.95 ^
--top-k 20 ^
--min-p 0.0 ^
--presence-penalty 0.0 ^
--repeat-penalty 1.0 ^
--load-mode none ^
--no-warmup ^
-b 256 ^
-ub 128 ^
-c 98304 ^
--host 0.0.0.0 ^
--port 8033

if %ERRORLEVEL% NEQ 0 (
echo.
echo [ERROR] Llama server exited with error code %ERRORLEVEL%
pause
)
```

### 3a. Linux translation for Rig A

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL="$MODELS/unsloth-q27b/Qwen3.8-27B-UD-Q4_K_S.gguf"   # see §1a discrepancy note
TPL="$MODELS/templates/chat_template.jinja"

./build/bin/llama-server \
  -m "$MODEL" \
  --jinja --chat-template-file "$TPL" \
  --reasoning-preserve \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  -fa on \
  -ngl 99 \
  --override-tensor "blk\.([0-9]|[1-3][0-9]|4[0-5])\.ffn_.*=CPU" \
  -ctk q4_0 -ctv q4_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -lv 4 \
  --no-mmproj \
  -np 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --load-mode none --no-warmup \
  -b 256 -ub 128 -c 98304 \
  --host 127.0.0.1 --port 8033
```

**Notes on the translation (do not skip):**

- `--override-tensor "blk\.([0-9]|[1-3][0-9]|4[0-5])\.ffn_.*=CPU"` pushes the FFN tensors of **blocks 0–45** to CPU while attention stays on GPU. This is the entire trick — it trades PCIe/CPU bandwidth for VRAM. On Linux, single-quote the regex to stop the shell touching the backslashes.
- ⚠️ **`-fit off` and `--gpu-layers-draft all` are not in the current master server README.** `-fit` does not appear at all; the documented draft-layers flag is `--spec-draft-ngl` / `-ngld`. They may be aliases, newer than the docs, or fork-specific. **Run `./build/bin/llama-server --help | grep -E 'fit|draft'` on your build and adapt.** Omitted above rather than guessed.
- The recipe does **not** pass an explicit MTP draft model. If your build requires one, supply `-md "$MODELS/unsloth-q27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"` (1.28 GiB) — budget that against the 11.5 GB gate.
- `--no-mmproj` is load-bearing (§8 G1). `--chat-template-kwargs '{"reasoning_effort":"medium"}'` is load-bearing (§8 G3).
- On Linux drop the PowerShell quote-escaping: `'{"reasoning_effort":"medium"}'`, not `"{\"...\"}"`.

Community replicaton in-thread: `Tamal777` on a Ryzen 5 5600 got 6.64–8.45 tok/s. A 4070S user got 2 tok/s via KoboldCpp — **use upstream llama.cpp, not wrappers**, for M0 numbers.

---

## 4. EXL3 ON THE 3060 (exllamav3 + TabbyAPI)

### 4a. ⚠️ Sizing reality — the briefed plan does not fit

| Branch | Weights (GB) | GiB | Rig A 12 GiB / 11.5 GB gate |
|---|---:|---:|---|
| `3.50bpw` | 15.34 | 14.29 | ✗ |
| `3.00bpw` | 13.82 | 12.87 | ✗ **larger than the entire card** |
| `SC_3.00bpw_H4` | 13.45 | 12.53 | ✗ |
| `2.50bpw` | 12.30 | 11.45 | ✗ — weights alone ≈ the whole gate, no room for KV |
| `SC_2.20bpw_H3` | 10.80 | 10.06 | ✓ ~1.4 GiB left for KV + activations + MTP head |
| `2.00bpw` | 10.78 | 10.04 | ✓ ~1.4 GiB left |

exllamav3 has **no CPU-offload path** comparable to llama.cpp's `-ot` — weights must be resident. So on Rig A:

> **Load `2.00bpw` or `SC_2.20bpw_H3`, not `3.00bpw`.** Q4 KV cache is mandatory, and `cache_size` must be set far below the 262,144 native context.

`3.00bpw` (12.87 GiB) is the right target on **Rig B (16 GiB)**, leaving ~3 GiB for cache. Still download 3.00bpw — measure it on Rig B, and record "does not load" as the honest Rig A result (that is a legitimate M0 datapoint per the contract's honesty metric).

Why the inflation vs a naive 27B×3bpw≈10.4 GB: the model is multimodal (vision + video encoders) with a **248,320-token vocab** and 5,120 hidden size, so the non-quantized portion is large. Confirmed by the linear fit in §1d (~4.7 GB fixed overhead).

⚠️ **UNVERIFIED:** no published measurement of actual runtime VRAM for this model on a 3060 exists. The table is weights-only; measure the real peak per §7.

### 4b. Install (headless Linux, CUDA, Ampere sm_86)

**exllamav3 latest release: `v1.4.2` (2026-08-11).** The EXL3 quants declare `quantization_config.version: "1.4.2"` — **run 1.4.2 or newer.**
Sources: [releases](https://github.com/turboderp-org/exllamav3/releases), [README](https://github.com/turboderp-org/exllamav3/blob/master/README.md), [setup.py](https://github.com/turboderp-org/exllamav3/blob/master/setup.py)

Requirements: PyTorch **CUDA 12.4 or later**; `setup.py` pins `torch>=2.6.0`, plus `tokenizers>=0.21.1`, `numpy>=2.1.0`, `safetensors`, `llguidance>=1.7.0`, `flash-linear-attention>=0.5.0`. v1.4.2 ships wheels for **cp310–cp314**, Linux + Windows, `cu128` (torch 2.7.0–2.11.0) and `cu132` (torch 2.11.0).

```bash
python3.12 -m venv ~/venvs/exl3 && source ~/venvs/exl3/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Prebuilt wheel (recommended) — match cp/torch/cu to your venv exactly.
# Example asset name for py3.12 + torch2.9.0 + cu128:
#   exllamav3-1.4.2+cu128.torch2.9.0-cp312-cp312-linux_x86_64.whl
# Get the exact URL from the release assets list, do not construct it by hand:
#   https://github.com/turboderp-org/exllamav3/releases/latest
pip install <wheel-url-copied-from-releases-page>

python -c "import exllamav3, torch; print(exllamav3.__version__, torch.cuda.get_device_capability())"
# expect: 1.4.2  (8, 6)   on Rig A
```

**Ampere performance caveat.** The README still warns *"The Marlin-inspired GEMM kernel still needs work to achieve the same efficiency on Ampere GPUs and to remain memory-bound at lower bitrates."* However [issue #144](https://github.com/turboderp-org/exllamav3/issues/144) shows turboderp stating on **2026-08-11: "EXL3 much improved on Ampere now."** The README wording predates that comment — **treat the README caveat as stale and measure**. Historical community reports there: prompt processing well optimized on Ampere, token generation lagging ik_llama/koboldcpp by ~5–8 tok/s.

⚠️ **UNVERIFIED:** the exact compiled CUDA arch list in the prebuilt wheels. No explicit `TORCH_CUDA_ARCH_LIST`/`-gencode` sm_86 entry appears in `setup.py`; coverage appears inherited from torch defaults. If the wheel misbehaves on sm_86, build from source.

### 4c. TabbyAPI install

Source: [Getting Started wiki](https://github.com/theroyallab/tabbyAPI/wiki/01.-Getting-Started), [repo](https://github.com/theroyallab/tabbyAPI)

```bash
git clone https://github.com/theroyallab/tabbyAPI && cd tabbyAPI
./start.sh          # creates venv, installs deps
# or manual: pip install -U .[cu12]      (CUDA 12.x)
#            pip install -U .[cu13]      (CUDA 13.x, Python 3.12+)
```
Python 3.10–3.14, "preferably 3.12". Requirements state **CUDA 12.8+** and "NVIDIA GPU (supports Ampere generation and higher for parallel batching)". Docker: `ghcr.io/theroyallab/tabbyapi:latest`.

### 4d. Loading a specific bpw revision

**The revision is a *download-time* flag, not a config key.** Verified in [`common/args.py`](https://github.com/theroyallab/tabbyAPI/blob/main/common/args.py): the `download` subcommand takes `repo_id`, `--revision` ("Branch name in HuggingFace repo"), `--folder-name`, `--token`, `--include`, `--exclude`. `config_sample.yml` has **no** revision key.

```bash
# --folder-name is REQUIRED in practice: without it the folder defaults to the
# repo_id's last component, so two bpw branches would collide on disk.
./start.sh download turboderp/Qwen3.8-27B-exl3 \
  --revision SC_2.20bpw_H3 --folder-name Qwen3.8-27B-exl3-SC2.20

./start.sh download turboderp/Qwen3.8-27B-exl3 \
  --revision 3.00bpw --folder-name Qwen3.8-27B-exl3-3.00     # for Rig B
```
Then point Tabby at the local directory via `model.model_dir` + `model.model_name`. (Equivalently, the `hf download --revision` commands in §1e produce the same layout.)

Runtime alternative: `POST /v1/model/load` — but note the wiki's warning that *"load requests are ephemeral and `config.yml` options will not apply"*, so prefer `config.yml` for reproducible M0 runs.

### 4e. Q4 KV cache

Config key is **`model.cache_mode`**. Verbatim from [`config_sample.yml`](https://github.com/theroyallab/tabbyAPI/blob/main/config_sample.yml):

```yaml
  # Enable different cache modes for VRAM savings (default: FP16).
  # Specify the pair k_bits,v_bits where k_bits and v_bits are integers
  # from 2-8 (i.e. 8,8). The legacy values 'FP16', 'Q8', 'Q6', 'Q4' are also accepted.
  cache_mode: FP16
```

So both `cache_mode: Q4` and `cache_mode: 4,4` are valid. **`cache_size` must be a multiple of 256.** A separate `draft_cache_mode` governs the draft path. CLI equivalents are auto-generated (`--cache-mode`).

### 4f. MTP in exllamav3 — **YES, supported** (contradicts the brief)

- **turboderp, 2026-08-21, [issue #300](https://github.com/turboderp-org/exllamav3/issues/300):** *"It is already supported. The MTP layer is embedded in the original model's .safetensors files, and quantized versions retain the same overall structure."*
- Corroborated inside the quant itself: [`3.00bpw/config.json`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/raw/3.00bpw/config.json) contains `"mtp_num_hidden_layers": 1`, `"mtp_bits": 4`, and `quantization_config: {quant_method: "exl3", version: "1.4.2", bits: 3.0, head_bits: 6, mtp_bits: 4}`. The MTP head is explicitly quantized at 4 bits and shipped in the EXL3 repo.
- It works because Qwen3.8-27B declares `architectures: ["Qwen3_5ForConditionalGeneration"]` / `model_type: "qwen3_5"`, which exllamav3's README support table lists under "Qwen 3.5". **"Qwen3.8" never appears by that name in the table** — don't conclude it's unsupported from that absence.

Enable via the `draft_model` block in `config.yml`:
```yaml
draft_model:
  # Drafting mode for exllamav3 (default: model).
  # Options: model, disabled, mtp, ngram.
  draft_mode: mtp
  draft_num_tokens:        # tokens to draft per iteration — SWEEP THIS
  dynamic_draft:           # adjust draft tokens by observed acceptance rate
  draft_cache_mode: Q4
```
All four modes exist: `model` (separate draft model; `draft_model_name` applies only here), `mtp`, `ngram` (`ngram_match_min`, default 2), `disabled`. If MTP underperforms, `ngram` is the fallback — **you do not need a separate draft model.**

In flight, worth knowing: [PR #303](https://github.com/turboderp-org/exllamav3/pulls) "Speed up Qwen MTP decoding by 22% with a selected vocabulary head" (open, 2026-08-22). [Issue #260](https://github.com/turboderp-org/exllamav3/issues/260) reports illegal memory access with MTP on **multi-GPU layer splits** — not applicable to either single-GPU rig here. DFlash2 is a **feature request only** ([issue #294](https://github.com/turboderp-org/exllamav3/issues/294)), not implemented, despite some blog claims otherwise.

⚠️ Inconclusive: an attempt to confirm MTP tensor names inside `model.safetensors.index.json` on the 3.00bpw branch was truncated by fetch size. Treat as inconclusive, **not** contradictory — turboderp's statement plus `mtp_bits: 4` are the reliable evidence.

---

## 5. QUALITY MEASUREMENT PLAN (perplexity + KL divergence)

### 5a. Get the eval corpus

llama.cpp's own [`scripts/get-wikitext-2.sh`](https://github.com/ggml-org/llama.cpp/blob/master/scripts/get-wikitext-2.sh) pulls:

```bash
curl -L -o wikitext-2-raw-v1.zip \
  https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip     # -> wikitext-2-raw/wiki.test.raw
```
Verified live: HTTP 200, **4,721,645 bytes**. The old `s3.amazonaws.com/research.metamind.io/wikitext/...` URL 301-redirects — do not use it. There is **no** `get-wikitext-103.sh` on master.

### 5b. The two-step KL workflow

Flag definitions verbatim from [`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp):
```cpp
{"--kl-divergence"}   "computes KL-divergence to logits provided via --kl-divergence-base"
{"--save-all-logits", "--kl-divergence-base"}, "FNAME"   "set logits file"
```
`--save-all-logits` is an **alias** for `--kl-divergence-base`; both are perplexity-tool-only.

```bash
# Step 1 — generate REFERENCE logits (once, on a big GPU; see 5c)
./build/bin/llama-perplexity -m "$REF_MODEL" \
  -f wikitext-2-raw/wiki.test.raw \
  --kl-divergence-base /data/kld/q27b-ref.kld

# Step 2 — score each candidate against them (on any rig)
./build/bin/llama-perplexity -m "$CANDIDATE" \
  -f wikitext-2-raw/wiki.test.raw \
  --kl-divergence-base /data/kld/q27b-ref.kld --kl-divergence
```

Reported metrics: mean KLD ± uncertainty, PPL ratio and its `ln`, ΔPPL, mean Δp, Pearson correlation of correct-token probabilities, Δp percentiles, RMS Δp, and "Same top p". **KLD 0 = identical distributions.** Source: [tools/perplexity/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md).

### 5c. ⚠️ The reference-logits problem — plan disk and GPU first

**Disk.** The README states verbatim: *"The logit file will be very large, 11 GiB for LLaMA 2 or 37 GiB for LLaMA 3 when using the Wikitext-2 test set."* Size scales with **vocabulary**. Qwen3.8-27B has a **248,320-token vocab — roughly 1.9x LLaMA 3's 128k**, so budget on the order of **~70 GiB** for the `.kld` file. Confirm empirically on a short slice before committing the full run.

**GPU for BF16 reference.** The BF16 GGUF is **50.11 GiB** (ggml-org) / 50.90 GiB (unsloth, 2 shards). Neither rig can hold it. Options, in order of preference:

1. **Rented GPU (recommended for the frozen reference).** One **H100 80GB** or **A100 80GB** holds BF16 weights (~50 GiB) plus activations. Generate `q27b-ref.kld` once there, then transfer. Given ~70 GiB of logits, run the perplexity pass on the rented box and **copy down only the `.kld`** — or, better, run *both* steps there for the reference and copy down just the numbers. Budget the transfer, not just the GPU-hours.
2. **Q8_0 as proxy reference (fallback).** Use `ggml-org/Qwen3.8-27B-Q8_0.gguf` (26.63 GiB) or `unsloth/Qwen3.8-27B-Q8_0.gguf` (27.05 GiB) — fits a single 32 GB or 48 GB card, and CPU-generatable on a 32 GB-RAM box overnight if desperate.
   **Caveats, state them in every result that uses it:**
   - All KLD numbers become *"divergence from Q8_0"*, **not** from the true model. They are **not comparable** to published KLD-vs-fp16 figures.
   - Q8_0's own divergence from BF16 is small but **nonzero and unmeasured** — it silently floors your entire ladder.
   - It compresses apparent differences between good quants near the top of the range.
   - The contract's quality metric is defined *"vs. the fp16 reference"* — a Q8_0 proxy does **not** satisfy it. Use it for fast iteration; get the real BF16 reference before anything is frozen or published.
3. **Do not** use a 4-bit model as reference under any circumstances.

**Precision caveat that applies to all references.** The README notes the logit file stores values cast to **16-bit unsigned integers with scaling**, not exact FP32 — which is why llama.cpp's own scoreboard shows a nonzero KLD (0.000551) for the `f16` row against itself. **That value is your noise floor**; differences smaller than it are meaningless.

### 5d. M0 discipline

Run every candidate with the **same** corpus file, same `-c`, same template, same `reasoning_effort`. Perplexity/KLD is measured on raw text completion, so `reasoning_effort` does not apply to §5 — but it absolutely applies to any benchmark-subset scoring you add. Record: model SHA256, exact bytes, llama.cpp commit, KLD mean ± uncertainty, ΔPPL, Same-top-p.

---

## 6. `--target-bpw` — ⚠️ NOT IN LLAMA.CPP MASTER

**This is the biggest correction in the runbook.** `GOAL_STATE.md` names `--target-bpw` as one of the three head-to-head baselines, so this materially affects M0 scope.

| Claim | Status |
|---|---|
| [Discussion #18531](https://github.com/ggml-org/llama.cpp/discussions/18531) "Auto-Adaptive Mixed-Precision Quantization (Target File Size/BPW)", by **EAddario**, 2026-01-01 | **VERIFIED, exists** |
| Implementing PR is [**#15550**](https://github.com/ggml-org/llama.cpp/pull/15550) | **VERIFIED** |
| That PR is **merged** | **FALSE** — API reports `state=open`, `draft=true`, `merged=false`. Opened 2025-08-24, still active 2026-08-22. |
| Any *other* merged PR adds `--target-bpw` | **FALSE** — search `repo:ggml-org/llama.cpp "target-bpw" type:pr is:merged` → **0 results** |
| `--target-bpw` exists in master | **FALSE** — `grep "target.bpw"` on master's `tools/quantize/quantize.cpp` → **0 hits**. Master's `usage()` lists only `--allow-requantize --leave-output-tensor --pure --imatrix --include-weights --exclude-weights --output-tensor-type --token-embedding-type --tensor-type --tensor-type-file --prune-layers --keep-split --override-kv --dry-run` |
| Dependency [PR #14891](https://github.com/ggml-org/llama.cpp/pull/14891) (activation-based imatrix statistics) | **VERIFIED open, not merged** |

**To use it you must build EAddario's fork**, and you likely also want unmerged #14891 for the activation statistics it consumes.

```bash
git clone https://github.com/EAddario/llama.cpp eaddario-llama.cpp
cd eaddario-llama.cpp
git checkout a50ddb660834b626befdb5b9aeda9bd6a2dd5666   # PR #15550 head, verified
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"
```

Verbatim help text from that branch:
```
  --target-bpw n
                          target a bits per weight (bpw); must be a positive number between 0.0 and 16.0
  --target-size n
                          target a file size; must be a positive number
                          allowed units: b, k|kib, m|mib, g|gib, t|tib; defaults to b (bytes) if none is provided
  --state-file [filename]
                          file name to use/save; if none is provided, the default name will be used
note: --target-bpw and --target-size cannot be used together
```

Real examples from the branch's `tools/quantize/README.md`:
```bash
./llama-quantize --target-bpw 4.5678 --state-file --imatrix imatrix.gguf input-model-f32.gguf 8
./llama-quantize --target-size 1.5g --state-file my-state-file.dat --imatrix imatrix.gguf input-model-f32.gguf 8
```
Default state filename: `<model name>-<model hash>.bpw_state`.

**Two copy-paste traps:**
- Discussion #18531 shows `--save-state --state-file …`. **`--save-state` does not exist** in the PR head's `usage()` (0 grep hits) — the discussion text is stale relative to the branch.
- The branch README's second example contains a literal typo `---state-file` (three dashes). Don't copy either verbatim.

### 6a. imatrix — flag, requirement, and where to get one

Flag is **`--imatrix file_name`** (present in master and unchanged on the branch).

⚠️ **Partially UNVERIFIED:** the PR and discussion say an imatrix with activation data is *"required for best results"* — worded as strongly recommended. No hard "refuses to run without it" assertion was found. **Assume you need it.**

**Two ready-made imatrices for this exact model — no generation needed:**
- `unsloth/Qwen3.8-27B-GGUF` → `imatrix_unsloth.gguf` (13,642,656 B)
- `bartowski/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-imatrix.gguf` (13,642,688 B)

(Both ≈13.0 MiB and within 32 bytes of each other — likely the same shape/tensor set, different data. Use both and compare; that is a free extra datapoint.)

**To make your own** ([tools/imatrix/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)):
```bash
./build/bin/llama-imatrix -m "$MODELS/ggml-q27b/Qwen3.8-27B-BF16.gguf" \
  -f calibration_datav3.txt -ngl 99 -o q27b-imatrix.gguf
# output is GGUF by default; --output-format dat gives the legacy binary format
./build/bin/llama-quantize --imatrix q27b-imatrix.gguf in-f32.gguf out-q4_k_m.gguf q4_k_m
```
Note this needs the BF16 model resident — same rented-GPU consideration as §5c.

**Calibration corpora** (all URLs live-verified HTTP 200; index gist: <https://gist.github.com/Jaid/36739dd673f71388a58e20db841fd42c>):

| File | Author | Bytes | URL |
|---|---|---:|---|
| `groups_merged.txt` | kalomaze | 201,119 | <https://github.com/ggerganov/llama.cpp/files/14194570/groups_merged.txt> |
| `calibration_datav3.txt` | bartowski1182 | 279,515 | <https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt> |
| `calibration_data_v5_rc.txt` | tristandruyen | 437,053 | <https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt> |
| `Qwen3.8-27B-calibration-v6.txt` | bartowski (model-specific) | 1,258,850 | in `bartowski/Qwen3.8-27B-GGUF` — **prefer this one, it is tuned for this model** |

`groups_merged.txt` originates in [discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263), described by kalomaze as *"a decent general purpose imatrix calibration dataset… more diverse than wikitext"* at ~30k tokens including code.
⚠️ The widely-cited `gist.github.com/kalomaze/2f95f9d0f2f81ac00fd6b39e6d1f52fd` is **404/dead** — use the `files/14194570` attachment. Also note `calibration_datav3.txt` is byte-identical in size to Dampfinchen's `groups_merged-enhancedV3.txt` — a rehost, not an independent corpus. **Never calibrate on `wiki.test.raw` and then evaluate on it** — that contaminates §5.

---

## 7. VRAM AND SPEED MEASUREMENT

### 7a. llama-bench

pp512/tg128 are the **defaults** (`-p` 512, `-n` 128), so the canonical run is short. Source: [tools/llama-bench/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)

```bash
./build/bin/llama-bench -m "$MODEL" -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -r 5 -o json \
  > results/bench_$(basename "$MODEL" .gguf).json
```

Verified flags and defaults:

| Flag | Default | Note |
|---|---|---|
| `-p, --n-prompt` | **512** | produces the `pp512` row |
| `-n, --n-gen` | **128** | produces the `tg128` row |
| `-pg <pp,tg>` | — | combined prompt+gen test |
| `-ngl, --n-gpu-layers` | -1 | |
| `-fa, --flash-attn <on\|off\|auto>` | **auto** | tri-state now — **not** the old `0\|1` |
| `-ctk` / `-ctv` | f16 | |
| `-b` / `-ub` | 2048 / 512 | |
| `-d, --n-depth` | 0 | prefills KV to benchmark **at depth** — use this, see below |
| `-r, --repetitions` | 5 | |
| `-o, --output <csv\|json\|jsonl\|md\|sql>` | md | JSON/JSONL include per-repetition samples; md/csv give only mean ± stddev |
| `-ot, --override-tensor` | — | so the §3 offload recipe is benchable |
| `-fitt/--fit-target <MiB>`, `-fitc/--fit-ctx <n>` | — | fit model to device memory with a margin — better than manual `-ngl` bisection |

Sweeps: comma-separate, repeat the flag, or use ranges `first-last`, `first-last+step`, `first-last*mult`. Everything except `-r`, `-o`, `-v` is sweepable; pp×tg run as a cross product.

**Two M0-critical caveats:**
- llama-bench **excludes tokenization and sampling time**. Your contract's "code-effective tok/s" is an end-to-end number — measure that against `llama-server`, not here.
- `pp512/tg128` at depth 0 flatters a long-context config. The contract cares about 96K-context behaviour, so **also sweep `-d`** (e.g. `-d 0,4096,16384,65536`) — decode speed degrades with KV depth and that is where the 12 GB story actually lives.
- llama-bench does **not** exercise MTP. MTP speedups must be measured via `llama-server` (§3) with acceptance rate logged.

### 7b. Peak VRAM on Linux

**Two methods that answer different questions — record both.**

**(1) llama.cpp's own buffer accounting.** It prints its allocations to stderr at load. Verified format strings on master:
```
%s: %12s model buffer size = %8.2f MiB      # src/llama-model.cpp
%s: %10s    KV buffer size = %8.2f MiB      # src/llama-kv-cache.cpp
%s: %10s compute buffer size = %8.2f MiB    # src/llama-context.cpp
%s: %10s  output buffer size = %8.2f MiB    # src/llama-context.cpp
```
Maintainer guidance, @slaren in [discussion #9784](https://github.com/ggml-org/llama.cpp/discussions/9784): *"Look at the messages printed while loading the model, llama.cpp will tell you the size of (almost) every backend buffer it allocates."*

**(2) `nvidia-smi` polling — this is the number the 11.5 GB gate is defined against.** The same maintainer adds the operative caveat: *"The CUDA runtime also needs some memory that may not be accounted elsewhere."* Note the hedge **"(almost) every"** — summed buffers **under-report** true device occupancy by the CUDA context (typically a few hundred MB). So:

> **Use `nvidia-smi` peak for the contract gate. Use the printed buffers to attribute *where* the memory went.**

```bash
# poll at 100 ms into a log, run the workload, take the max
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -lms 100 > results/vram_poll.csv &
POLL=$!
# ... run llama-bench / llama-server workload here ...
kill $POLL
awk -F', ' 'NR>1 && $2>m {m=$2} END {print "PEAK MiB:", m}' results/vram_poll.csv
```

⚠️ **Sampling misses short peaks.** The transient **prompt-processing compute buffer at large `-ub`** is the classic one — it can spike and vanish between 100 ms samples. Mitigations: poll at `-lms 50`, and cross-check against the printed `compute buffer size`. If the two disagree by more than the CUDA-context allowance, trust the larger.

Also run on an **otherwise idle, headless** GPU (`nvidia-smi` should read near 0 MiB before launch); a desktop compositor can hold 300–800 MiB and silently eat your margin.

⚠️ **UNVERIFIED:** there is no llama.cpp-official document on measuring peak VRAM, and no maintainer-recommended polling command or interval. The polling incantation above is standard practice, not upstream guidance. (Negative result from browsing READMEs/docs — GitHub code search needs auth — so treat as "not found", not "proven absent".)

---

## 8. GOTCHAS — all verified

**G1 · mmproj auto-load VRAM trap.** Qwen3.8-27B is a **vision-language** model (`Qwen3_5ForConditionalGeneration`, pipeline `image-text-to-text`). llama.cpp documents `--mmproj-auto` / `--no-mmproj` as *"whether to use multimodal projector file (if available), useful when using `-hf` (default: **enabled**)"*, and the projector is **offloaded to GPU by default**. On a `-hf` pull this silently eats **~0.87 GiB of VRAM** you did not budget — 7.5% of the 12 GiB card, against an 11.5 GB gate. **Always pass `--no-mmproj` for text-only baselines**, and never download the mmproj files (§1e). Lazy-loading is requested but not implemented ([discussion #20855](https://github.com/ggml-org/llama.cpp/discussions/20855)). Related: [issue #23371](https://github.com/ggml-org/llama.cpp/issues/23371) — MTP + vision can OOM during mmproj restore. Source: [multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md), [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

**G2 · Stale CUDA build → confident gibberish.** Three compounding traps from [discussion #27164](https://github.com/ggml-org/llama.cpp/discussions/27164): (a) the old CUDA Gated DeltaNet path corrupts output while loading cleanly and running at full speed; (b) a `--depth 1` clone makes `git pull` report *"Already up to date"* while hundreds of commits behind — fix with `git fetch --unshallow`; (c) even after rebuilding, a stale `libggml-cuda.so` earlier in the library path gets loaded instead of the new one. Verify with `ldd ./build/bin/llama-server` and `echo $LD_LIBRARY_PATH`. Run §2g before trusting any number.

**G3 · `reasoning_effort` defaults to `xhigh` — burns your token budget.** Qwen3.8's template exposes `xhigh` (default) / `medium` / `low` / `none`. `xhigh` makes the model overthink trivial prompts, inflating latency and destroying tok/s comparability. Set it explicitly:
```
--chat-template-kwargs '{"reasoning_effort":"medium"}'
```
Sources: [Unsloth Qwen3.8 docs](https://unsloth.ai/docs/models/qwen3.8), [Simon Willison, 2026-08-16](https://simonwillison.net/2026/Aug/16/qwen-38-27b/) ("defaults to wildly overthinking things"), [Qwen/Qwen3.8-27B discussion #113](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/113) ("This model cannot stop thinking"). **Freeze one value across all M0 baselines and record it** — this single knob can swing effective tok/s several-fold.

**G4 · `-fa` now defaults to `auto`, not off.** Verbatim: `-fa, --flash-attn [on|off|auto]` — *"set Flash Attention use ('on', 'off', or 'auto', default: 'auto')"*. In `auto`, llama.cpp **silently disables FA** when it detects an unsupported combination (observed log: *"Flash Attention was auto, set to disabled"*). Silent disable changes both VRAM and speed. **Always pass `-fa on` explicitly for M0 and grep the startup log to confirm it stayed on.** Source: [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

**G5 · Quantized KV cache silently falls back off the fused FA path.** `-ctk`/`-ctv` accept `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`, but a **default CUDA build compiles only a subset of FA kernel combinations**. Ask for a KV type without a compiled kernel — `q5_1` among them — and attention drops off the fused path with **no warning**, reportedly 25–45x slower prefill. Fixes: build with **`-DGGML_CUDA_FA_ALL_QUANTS=ON`** (already in §2b) or use matched, well-supported K/V types (`q4_0`/`q4_0`, or `q8_0`/`q8_0`). Detection: send a short prompt and check prompt-eval; **>~50 ms/token on a modern GPU means you fell back to CPU.** Sources: [issue #24485](https://github.com/ggml-org/llama.cpp/issues/24485), [discussion #22411](https://github.com/ggml-org/llama.cpp/discussions/22411) (AMD HIP: asymmetric K/V types drop to a slower non-fused path, also silently).
⚠️ Partially verified: the *mechanism* and the `q5_1`-not-in-default-kernel-set consequence are confirmed; I found no issue naming `q5_1` **specifically** as a filed bug. Treat "q5_1 CPU fallback" as an instance of G5, not a separate defect.

**G6 · Never compare quants by name across publishers.** `Q4_K_M` is 15.33 GiB (unsloth UD) vs 17.67 GiB (ggml-org) — a 2.3 GiB spread. `IQ2_S` is 7.80 GiB (unsloth UD) vs 9.59 GiB (bartowski). Compare by **measured bytes and measured KL divergence** only (§5).

**G7 · Blackwell (Rig B) specifics.** CUDA **12.8+** required for `sm_120`. MXFP4 template instances can fail ptxas on `sm_120` ([issue #19662](https://github.com/ggml-org/llama.cpp/issues/19662)). **Qwen3.8-27B NVFP4 decode hangs on Blackwell — [issue #27329](https://github.com/ggml-org/llama.cpp/issues/27329), still OPEN.** Avoid NVFP4 on Rig B at M0. If cmake picks a stale nvcc from `PATH`, pin `-DCMAKE_CUDA_COMPILER`.

**G8 · `hf`, not `huggingface-cli`.** `huggingface-cli` was **removed** in `huggingface_hub` v1.0. Scripts copied from older guides will fail outright. Use `hf download` (§1e).

---

## 9. EXECUTION ORDER

Do these in order; each gates the next.

1. **Build** llama.cpp on both rigs per §2b (`FA_ALL_QUANTS=ON`, correct arch). Record the commit hash.
2. **Smoke-test correctness** per §2g on both rigs. CUDA output must match CPU/Vulkan qualitatively. **Do not proceed on a bad build** — G2 output looks fine and is not.
3. **Download** the §1e set. Verify byte sizes against §1 tables. Confirm no `mmproj-*` landed.
4. **Fetch and pin** the fixed chat template (§2d); record its SHA256. Freeze `reasoning_effort` (§8 G3) — one value, all runs.
5. **llama-bench sweep** (§7a) across the GGUF ladder on both rigs, with `-d` depth sweep. Poll VRAM throughout (§7b).
6. **llama-server** runs for end-to-end + MTP: reproduce §3 on Rig A; sweep `--spec-draft-n-max` ∈ {1,2,3,4} and log acceptance rate (§2f). This is where "code-effective tok/s" is measured.
7. **EXL3** (§4): Rig A gets `2.00bpw` / `SC_2.20bpw_H3` with `cache_mode: Q4` and `draft_mode: mtp`. Rig B gets `3.00bpw`. Record "3.00bpw does not load on 12 GB" as a real result.
8. **Quality** (§5): decide reference strategy *before* running. Rent the big GPU for the BF16 `.kld` if the numbers are going anywhere near a paper; use Q8_0 proxy only for fast iteration, always labelled.
9. **`--target-bpw`** (§6): only if you build EAddario's fork. Otherwise **record it as unavailable** and adjust the `GOAL_STATE.md` three-way comparison accordingly.

**Record for every run:** model file SHA256 + exact bytes, llama.cpp/exllamav3 commit or version, full command line, template SHA256, `reasoning_effort`, peak `nvidia-smi` MiB, llama.cpp's printed buffer breakdown, pp/tg tok/s, MTP acceptance rate, and KLD ± uncertainty.

---

## 10. CONSOLIDATED ⚠️ UNVERIFIED / CORRECTED ITEMS

**Premises from the M0 brief that were wrong (details in the header table):**

1. **`--target-bpw` is not merged and not in master** (PR #15550 open+draft). Requires building a fork. *Affects `GOAL_STATE.md`'s three-way thesis comparison — needs a scope decision.*
2. **Build ~b10450 is not a verified fix point** — that range is UI-only commits. Empirical floor only.
3. **`--draft-n` / `--mtp-model` do not exist.** Use `--spec-draft-n-max` / `--spec-type draft-mtp`.
4. **`--jinja` is default-enabled**, not a requirement. `--chat-template-file` is the operative flag.
5. **EXL3 3.00bpw does not fit a 3060** (12.87 GiB weights vs 12 GiB card). Use 2.00bpw / SC_2.20bpw_H3.
6. **MTP *is* supported in exllamav3** (`draft_mode: mtp`), contrary to the brief's doubt.

**Genuinely unverified — could not confirm from any primary source:**

7. The **PR that actually fixed** the Gated DeltaNet CUDA corruption. Unattributed; only a broken→working checkout pair is documented.
8. Any **publisher-stated minimum llama.cpp version** for Qwen3.8. Neither unsloth's card nor the MTP hub states one.
9. **`-fit off` and `--gpu-layers-draft all`** from the §3 recipe — absent from current master's server README (documented equivalent is `--spec-draft-ngl`/`-ngld`). May be aliases, newer than docs, or fork-specific. **Check `llama-server --help` on your build.**
10. Whether an **imatrix is a hard requirement** for `--target-bpw` or only "required for best results".
11. The **exact compiled CUDA arch list** in exllamav3 prebuilt wheels (no explicit sm_86 entry in `setup.py`).
12. **Runtime VRAM for EXL3 Qwen3.8-27B on a 3060** — no published measurement exists. §4a is weights-only.
13. **`q5_1` specifically** as a filed CPU-fallback bug. The G5 mechanism is confirmed; no issue names `q5_1` by itself. Treat as an instance of G5.
14. Any **official llama.cpp guidance** on peak-VRAM measurement or a recommended polling interval. §7b is standard practice, not upstream doctrine.
15. The **`Qwen3.8-27B-Q4_K_S.gguf`** filename in the §3 recipe does not exist in the unsloth repo (see §1a). Assumed to be `UD-Q4_K_S` renamed.

**Reliability note:** items 7, 9 and the §2f n-max guidance rest on **single community sources** (one GitHub discussion, one personal repo) with no maintainer confirmation. This model postdates the compiler's training data, so everything here was checked against live pages only, not against independent prior knowledge. Re-verify before publishing any number that depends on them.

---
*End of runbook.*

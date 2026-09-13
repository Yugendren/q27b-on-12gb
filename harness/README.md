# Measurement harness

Cross-platform (macOS dev now, Linux/CUDA target machines later) tooling for
measuring llama.cpp inference speed and quality. Stdlib + `requests` only —
no PyYAML, no numpy/pandas. Requires `llama-server`, `llama-bench`,
`llama-cli`, `llama-perplexity` on `$PATH` (llama.cpp build 10566 tested).

Built for the q27b_on_12gb M0 baseline sweep; the same scripts run unchanged
on the reference 3060/4070 rigs — only `configs/matrix.yaml` changes.

## Setup (Linux GPU machine)

```bash
python3 -m pip install --user requests   # only third-party dep
# if blocked by PEP 668 (externally-managed env):
python3 -m pip install --user --break-system-packages requests
```

`nvidia-smi` must be on `$PATH` for VRAM polling (bench_speed.py) and GPU
detection (profile_hw.py). Both degrade gracefully (null/empty) if it's
missing.

## Scripts

### `profile_hw.py`
One-shot hardware snapshot: platform, CPU model, logical cores, RAM,
GPU(s) (name + VRAM total; unified-memory flag on Apple Silicon), disk free.

```bash
python3 profile_hw.py                    # print JSON to stdout
python3 profile_hw.py --out hw.json       # write to file
```

### `bench_speed.py`
Wraps `llama-bench` for one model: pp512 (prompt processing, 512 tok) and
tg128 (text gen, 128 tok), 3 reps each by default, parsed from
`llama-bench -o json`. Hashes the model (streamed SHA-256, first 16 hex
chars — safe for 17 GB files) and records file size. On Linux, polls
`nvidia-smi --query-gpu=memory.used` every 500 ms in a background thread for
the duration of the llama-bench run and reports the peak (summed across GPUs
if there's more than one). On macOS `peak_vram_mb` is always `null` — there
is no VRAM counter distinct from unified system memory on Apple Silicon.

```bash
python3 bench_speed.py --model /path/to/model.gguf --flags "-ngl 999 -c 4096"
python3 bench_speed.py --model /path/to/model.gguf --flags "-ngl 999" --out results/speed.jsonl
```

Key output fields: `model_file`, `sha256_16`, `file_size_gb`, `flags`,
`pp_tok_s`, `tg_tok_s`, `peak_vram_mb`, `llama_cpp_build`, `timestamp`.

### `eval_quality.py`
Runs a fixed small quality suite against an **already-running**
`llama-server` started with `--jinja` (needed for chat-template rendering).

- **GSM8K**: first 50 problems (deterministic, in dataset order) from the
  official test set, downloaded once to `harness/data/gsm8k_test.jsonl` and
  cached. Reference answer parsed from the `#### N` suffix; model answer is
  the last number found in the completion. `temperature=0`,
  `max_tokens=1024`. Score = fraction correct.
- **HumanEval**: first 20 tasks, downloaded once
  (`github.com/openai/human-eval` raw `HumanEval.jsonl.gz`, decompressed and
  cached to `harness/data/HumanEval.jsonl`). Completion is executed against
  the task's `check()` function in a subprocess with a 10s timeout.
  Score = pass@1 (single sample per task).

Both tasks go through `POST {endpoint}/v1/chat/completions`.

```bash
# in one terminal:
llama-server -m model.gguf --jinja -c 4096 --port 8080 -ngl 999
# in another:
python3 eval_quality.py --endpoint http://127.0.0.1:8080 \
  --limit-gsm8k 50 --limit-humaneval 20 --out results/eval.json
```

### `run_matrix.py`
Reads `configs/matrix.yaml` (see `configs/matrix.example.yaml`; parsed by
`harness/yaml_lite.py`, a **minimal stdlib-only YAML subset** — flat list of
string-valued mappings only, no nesting). For each entry:

1. start `llama-server` on a free localhost port with `server_flags`
2. poll `/health` (up to 120s)
3. run `eval_quality.py` against it
4. stop the server
5. run `bench_speed.py` (`llama-bench`) on the same gguf with `bench_flags`
6. append one merged JSON row to `results/results.jsonl`

Failures at any stage (bad model path, server won't start, server crashes
mid-eval, bench fails) are recorded as a row with `status != "ok"` and an
`error` field; the matrix continues to the next entry rather than aborting.
Per-entry server stdout/stderr is captured to `results/logs/<name>.server.log`.

```bash
cp configs/matrix.example.yaml configs/matrix.yaml   # edit paths/flags
python3 harness/run_matrix.py \
  --config configs/matrix.yaml \
  --out results/results.jsonl \
  --limit-gsm8k 50 --limit-humaneval 20
```

## Config format (`configs/matrix.yaml`)

```yaml
- name: qwen3-0.6b-q4_k_m
  gguf_path: models/Qwen3-0.6B-Q4_K_M.gguf
  server_flags: "--jinja -c 4096 -ngl 999"
  bench_flags: "-ngl 999"
```

On CUDA machines, `server_flags`/`bench_flags` is where `-ngl 999` (full GPU
offload), `-fa`/`--flash-attn`, `-ctk`/`-ctv` (KV cache quant), and similar
go. `run_matrix.py` always adds `-m <gguf_path>` and, for the server,
`--port <free port>` — don't duplicate those in the flag strings.

## Bootstrap (Linux CUDA rigs)

`harness/bootstrap_linux.sh` prepares a fresh Rig A (RTX 3060 12GB) or Rig B
(RTX 5080 16GB) machine end-to-end: hardware gating, a CUDA llama.cpp build,
the harness's python deps, the Qwen3.8-27B GGUF artifacts for the first M0
pass, a hardware profile snapshot, and a ready-to-run `configs/matrix.yaml`.
It is idempotent — safe to re-run after a partial failure or to pick up a
newer llama.cpp release.

```bash
# first run on a fresh box (downloads ~25 GiB: the two baseline GGUFs)
harness/bootstrap_linux.sh

# pull the whole M0_RUNBOOK.md §1e GGUF ladder instead (~130 GiB; prompts
# for confirmation unless --yes is passed)
harness/bootstrap_linux.sh --tier full --yes

# override where large artifacts land
harness/bootstrap_linux.sh --models-dir /data/models --llama-cpp-dir /data/llama.cpp
```

What it does, in order: (1) checks for `nvidia-smi`, a supported GPU, driver/
CUDA version, RAM, and ≥60 GB free disk, aborting with a clear message if
unfit; (2) resolves the latest `ggml-org/llama.cpp` GitHub release and either
downloads a prebuilt Linux CUDA binary (if one exists) or clones/builds from
source with `-DGGML_CUDA_FA_ALL_QUANTS=ON` and the right
`CMAKE_CUDA_ARCHITECTURES` (86 for the 3060, 120 for the 5080), then symlinks
the binaries into `~/.local/bin`; (3) installs `requests` and the `hf` CLI
(`huggingface_hub[hf_transfer]`), handling PEP 668 with
`--break-system-packages` same as above; (4) downloads the GGUF/template
artifacts for `--tier minimal|full`, explicitly excluding `mmproj-*`; (5)
runs `profile_hw.py` to `results/hw_profile_$(hostname).json`; (6) generates
`configs/matrix.$(hostname).yaml` — a plain full-GPU `-ngl 99` baseline entry
plus the M0_RUNBOOK.md §3a community FFN-offload + MTP recipe entry; (7)
prints the exact `run_matrix.py` command to run next, after the §2g
correctness smoke test (the Gated DeltaNet CUDA bug produces fast,
well-formed *garbage*, not a crash — never skip that check).

Does not touch exllamav3/TabbyAPI (M0_RUNBOOK.md §4) — that's a separate
runtime this harness's scripts don't drive; EXL3 setup stays a manual step
per the runbook.

## Known limitations / not yet verified

- Tested end-to-end on macOS (Apple Silicon, Metal backend) only. The Linux
  + `nvidia-smi` VRAM-polling path in `bench_speed.py` and the GPU-detection
  path in `profile_hw.py` are implemented per the `nvidia-smi` CSV output
  format but **not exercised on real CUDA hardware yet** — verify on first
  3060/4070 run.
- `yaml_lite.py` intentionally supports only the flat shape shown above. If
  `matrix.yaml` ever needs nesting or multi-line values, switch to PyYAML.
- `eval_quality.py` HumanEval execution runs untrusted model-generated code
  directly via `subprocess.run(["python3", path], timeout=10)` with no
  sandboxing beyond the timeout — same trust model as the standard
  HumanEval harness, not hardened for adversarial models.
- GSM8K/HumanEval prompts use a fixed system-prompt instruction (not the
  original papers' few-shot prompt format) since generation goes through
  `/v1/chat/completions` with the model's own chat template. Scores are
  therefore not directly comparable to leaderboard numbers using raw
  completion-mode few-shot prompting — they're comparable across configs
  measured with this harness, which is what M0 needs.

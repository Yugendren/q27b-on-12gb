#!/usr/bin/env bash
# bootstrap_linux.sh — prepare a Linux CUDA machine to run the M0 baseline
# matrix for the q27b_on_12gb project.
#
# Targets (per M0_RUNBOOK.md):
#   Rig A — RTX 3060 12GB (Ampere, sm_86), 32GB system RAM
#   Rig B — RTX 5080 16GB (Blackwell, sm_120)
#
# Idempotent: safe to re-run. Each stage checks whether its work is already
# done and skips ahead when possible.
#
# Usage:
#   harness/bootstrap_linux.sh [--tier minimal|full] [--yes]
#                               [--models-dir DIR] [--llama-cpp-dir DIR]
#                               [--help]
#
# See harness/README.md "Bootstrap" section for the full walkthrough.
set -euo pipefail

# ------------------------------------------------------------------------
# Globals / defaults
# ------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TIER="minimal"
ASSUME_YES=0
MODELS_DIR="$REPO_ROOT/models"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MIN_FREE_DISK_GB=60
LLAMA_CPP_REPO_URL="https://github.com/ggml-org/llama.cpp"
CHAT_TEMPLATE_REPO="froggeric/Qwen-Fixed-Chat-Templates"

HOST="$(hostname -s 2>/dev/null || hostname)"

# Filled in as the script runs; printed in the final summary.
DETECTED_GPU_NAME=""
CUDA_ARCH=""
LLAMA_BIN_DIR=""
RESOLVED_BUILD=""
RESOLVED_BUILD_SOURCE=""
TEMPLATE_PATH=""
Q2KXL_PATH=""
Q4KS_PATH=""
MTP_PATH=""

# ------------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------------
log() { printf '[bootstrap] %s\n' "$*"; }
warn() { printf '[bootstrap][WARN] %s\n' "$*" >&2; }
die() {
    printf '[bootstrap][FATAL] %s\n' "$*" >&2
    exit 1
}

# ------------------------------------------------------------------------
# Arg parsing
# ------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [OPTIONS]

Prepares a Linux CUDA machine to run the M0 baseline matrix:
  1. Detect GPU / driver / CUDA / RAM / disk; abort if unfit.
  2. Ensure a recent llama.cpp CUDA build (prebuilt release, else source build).
  3. Install python deps the harness needs (requests).
  4. Download the Qwen3.8-27B GGUF artifacts needed for the first M0 pass
     (mmproj explicitly excluded).
  5. Run harness/profile_hw.py -> results/hw_profile_\$(hostname).json
  6. Generate configs/matrix.\$(hostname).yaml from the downloaded artifacts.
  7. Print the next command to run the matrix.

Options:
  --tier minimal|full   Artifact set to download (default: minimal).
                         minimal = top-2 baselines used by the two matrix
                                   entries this script generates (~25 GiB).
                         full    = the whole GGUF ladder from
                                   M0_RUNBOOK.md §1e, minus mmproj (~130 GiB).
  --yes                 Don't prompt for confirmation before a >30 GiB
                         download.
  --models-dir DIR      Where model artifacts land (default: $REPO_ROOT/models).
  --llama-cpp-dir DIR   Where to clone/build llama.cpp (default: \$HOME/llama.cpp).
  -h, --help            Show this help and exit.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            TIER="${2:?--tier requires an argument}"
            shift 2
            ;;
        --tier=*)
            TIER="${1#*=}"
            shift
            ;;
        --yes)
            ASSUME_YES=1
            shift
            ;;
        --models-dir)
            MODELS_DIR="${2:?--models-dir requires an argument}"
            shift 2
            ;;
        --models-dir=*)
            MODELS_DIR="${1#*=}"
            shift
            ;;
        --llama-cpp-dir)
            LLAMA_CPP_DIR="${2:?--llama-cpp-dir requires an argument}"
            shift 2
            ;;
        --llama-cpp-dir=*)
            LLAMA_CPP_DIR="${1#*=}"
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1 (see --help)"
            ;;
    esac
done

if [[ "$TIER" != "minimal" && "$TIER" != "full" ]]; then
    die "--tier must be 'minimal' or 'full', got: $TIER"
fi

# ------------------------------------------------------------------------
# Step 1 — hardware detection / gating
# ------------------------------------------------------------------------
detect_hardware() {
    log "=== Step 1/7: hardware detection ==="

    if [[ "$(uname -s)" != "Linux" ]]; then
        warn "this script targets Linux; running on $(uname -s) — continuing anyway."
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found on \$PATH — no CUDA GPU detected. Install the NVIDIA driver first."
    fi

    DETECTED_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/^ *//;s/ *$//')"
    if [[ -z "$DETECTED_GPU_NAME" ]]; then
        die "nvidia-smi ran but reported no GPU. Aborting."
    fi

    local driver_version vram_mib ram_gb cuda_version_str disk_free_kb disk_free_gb
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"
    vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)"
    cuda_version_str="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | awk '{print $3}')"

    if [[ -r /proc/meminfo ]]; then
        ram_gb="$(awk '/^MemTotal:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
    else
        ram_gb="unknown"
    fi

    disk_free_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -z "$disk_free_kb" ]]; then
        die "could not determine free disk space for $REPO_ROOT"
    fi
    disk_free_gb="$((disk_free_kb / 1024 / 1024))"

    log "GPU:            $DETECTED_GPU_NAME"
    log "Driver version: ${driver_version:-unknown}"
    log "CUDA version:   ${cuda_version_str:-unknown}"
    log "VRAM total:     ${vram_mib:-unknown} MiB"
    log "System RAM:     ${ram_gb} GB"
    log "Free disk ($REPO_ROOT): ${disk_free_gb} GB"

    if [[ "$disk_free_gb" -lt "$MIN_FREE_DISK_GB" ]]; then
        die "only ${disk_free_gb} GB free at $REPO_ROOT, need >= ${MIN_FREE_DISK_GB} GB. Free up space or pass --models-dir/--llama-cpp-dir pointing at a larger volume."
    fi

    case "$DETECTED_GPU_NAME" in
        *3060*)
            CUDA_ARCH=86
            log "Recognized as Rig A (RTX 3060) -> CMAKE_CUDA_ARCHITECTURES=86"
            ;;
        *5080*)
            CUDA_ARCH=120
            log "Recognized as Rig B (RTX 5080) -> CMAKE_CUDA_ARCHITECTURES=120"
            if [[ -n "${cuda_version_str:-}" ]]; then
                local major minor
                major="${cuda_version_str%%.*}"
                minor="${cuda_version_str#*.}"
                if [[ "$major" -lt 12 || ("$major" -eq 12 && "$minor" -lt 8) ]]; then
                    warn "sm_120 (Blackwell) needs CUDA >= 12.8; detected ${cuda_version_str}. Build will likely fail — update the driver/toolkit."
                fi
            fi
            warn "Blackwell gotchas (M0_RUNBOOK.md G7): MXFP4 template instances can fail ptxas on sm_120 (llama.cpp#19662); Qwen3.8-27B NVFP4 decode hangs on Blackwell and is still open (llama.cpp#27329) — avoid NVFP4 on this rig at M0."
            ;;
        *)
            CUDA_ARCH="native"
            warn "GPU '$DETECTED_GPU_NAME' doesn't match a known rig (3060/5080). Falling back to CMAKE_CUDA_ARCHITECTURES=native (requires CMake >= 3.24). Verify the build picks the right arch."
            ;;
    esac
}

# ------------------------------------------------------------------------
# Step 2 — llama.cpp CUDA build
# ------------------------------------------------------------------------
ensure_llama_cpp() {
    log "=== Step 2/7: llama.cpp CUDA build ==="

    local candidate_bin="$LLAMA_CPP_DIR/build/bin"
    if [[ -x "$candidate_bin/llama-server" && -x "$candidate_bin/llama-bench" \
        && -x "$candidate_bin/llama-cli" && -x "$candidate_bin/llama-perplexity" ]]; then
        LLAMA_BIN_DIR="$candidate_bin"
        RESOLVED_BUILD_SOURCE="existing (skipped rebuild)"
        RESOLVED_BUILD="$(build_number_from_binary "$LLAMA_BIN_DIR/llama-cli")"
        log "Found existing build at $LLAMA_BIN_DIR (build ${RESOLVED_BUILD:-unknown}) — skipping. Delete this dir to force a rebuild."
        link_binaries
        return
    fi
    if [[ -n "${LLAMA_PREBUILT_BIN_DIR:-}" && -x "${LLAMA_PREBUILT_BIN_DIR}/llama-server" ]]; then
        LLAMA_BIN_DIR="$LLAMA_PREBUILT_BIN_DIR"
        RESOLVED_BUILD_SOURCE="existing prebuilt (skipped rebuild)"
        RESOLVED_BUILD="$(build_number_from_binary "$LLAMA_BIN_DIR/llama-cli")"
        log "Found existing prebuilt at $LLAMA_BIN_DIR — skipping."
        link_binaries
        return
    fi

    local api_json latest_tag
    api_json="$(curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" 2>/dev/null || true)"
    if [[ -z "$api_json" ]]; then
        warn "could not reach GitHub releases API; falling back to 'master' for the source build."
        latest_tag="master"
    else
        latest_tag="$(printf '%s' "$api_json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tag_name",""))' 2>/dev/null || true)"
        if [[ -z "$latest_tag" ]]; then
            warn "could not parse latest release tag; falling back to 'master'."
            latest_tag="master"
        fi
    fi
    log "Latest llama.cpp release tag: $latest_tag"
    log "M0_RUNBOOK.md P1: no verified fix commit is attributable for the Gated DeltaNet CUDA bug -- 'build ~10450' is an empirical floor only, not a confirmed fix point. Using the latest release rather than pinning that unverified number. Run the §2g correctness smoke test on this build before trusting any numbers."

    local prebuilt_url="" api_json_file
    if [[ -n "$api_json" && "$latest_tag" != "master" ]]; then
        api_json_file="$(mktemp)"
        printf '%s' "$api_json" >"$api_json_file"
        prebuilt_url="$(python3 - "$api_json_file" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
best = ""
for a in d.get("assets", []):
    name = a.get("name", "").lower()
    if "cuda" not in name:
        continue
    if "win" in name or "macos" in name or "osx" in name:
        continue
    if "ubuntu" in name or "linux" in name:
        best = a.get("browser_download_url", "")
        break
print(best)
PYEOF
)"
        rm -f "$api_json_file"
    fi

    if [[ -n "$prebuilt_url" ]]; then
        log "Found a prebuilt Linux CUDA release asset: $prebuilt_url"
        install_prebuilt "$prebuilt_url" "$latest_tag"
    else
        log "No prebuilt Linux CUDA release asset found for $latest_tag (ggml-org typically ships CUDA prebuilts for Windows only, and CPU-only zips for Linux). Falling back to a source build."
        build_from_source "$latest_tag"
    fi

    RESOLVED_BUILD="${latest_tag#b}"
    link_binaries
}

build_number_from_binary() {
    local bin="$1"
    [[ -x "$bin" ]] || return 0
    "$bin" --version 2>&1 | grep -oE 'build[: ]+[0-9]+' | head -n1 | grep -oE '[0-9]+' || true
}

install_prebuilt() {
    local url="$1" tag="$2"
    local dest_dir="$LLAMA_CPP_DIR/prebuilt-$tag"
    local zip_path="$dest_dir.zip"

    if [[ -x "$dest_dir/bin/llama-server" ]]; then
        log "Prebuilt $tag already extracted at $dest_dir — skipping download."
        LLAMA_BIN_DIR="$dest_dir/bin"
        RESOLVED_BUILD_SOURCE="prebuilt release ($tag)"
        return
    fi

    mkdir -p "$LLAMA_CPP_DIR"
    log "Downloading prebuilt release: $url"
    curl -fsSL -o "$zip_path" "$url"
    mkdir -p "$dest_dir"
    if ! command -v unzip >/dev/null 2>&1; then
        die "unzip not found on \$PATH — needed to extract the prebuilt release. Install unzip or omit this step by pointing --llama-cpp-dir at a pre-built tree."
    fi
    unzip -q -o "$zip_path" -d "$dest_dir"

    local server_bin
    server_bin="$(find "$dest_dir" -maxdepth 3 -type f -name 'llama-server' | head -n1)"
    if [[ -z "$server_bin" ]]; then
        warn "prebuilt archive extracted but no llama-server binary found inside — falling back to source build."
        build_from_source "$tag"
        return
    fi
    LLAMA_BIN_DIR="$(dirname "$server_bin")"
    RESOLVED_BUILD_SOURCE="prebuilt release ($tag)"
    log "Prebuilt binaries in $LLAMA_BIN_DIR"
}

build_from_source() {
    local tag="$1"

    if ! command -v cmake >/dev/null 2>&1; then
        die "cmake not found on \$PATH — required to build llama.cpp from source. Install cmake (and a CUDA-capable compiler) first."
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
        warn "nvcc not found on \$PATH — the CUDA toolkit may not be installed or not on PATH. cmake -DGGML_CUDA=ON will fail without it."
    fi

    if [[ -d "$LLAMA_CPP_DIR/.git" ]]; then
        log "Existing clone at $LLAMA_CPP_DIR — updating."
        (
            cd "$LLAMA_CPP_DIR"
            git fetch --unshallow 2>/dev/null || true
            git fetch --tags --force origin
        )
    else
        log "Cloning $LLAMA_CPP_REPO_URL to $LLAMA_CPP_DIR"
        git clone "$LLAMA_CPP_REPO_URL" "$LLAMA_CPP_DIR"
    fi

    (
        cd "$LLAMA_CPP_DIR"
        if [[ "$tag" != "master" ]] && git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
            git checkout -B "bootstrap-$tag" "tags/$tag"
        else
            warn "tag '$tag' not found locally after fetch — building origin/master instead."
            git checkout -B "bootstrap-master" origin/master
        fi
        git log -1 --format='[bootstrap] llama.cpp commit: %H %cd'
    )

    local cmake_args=(-B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release)
    log "Configuring: cmake ${cmake_args[*]}  (in $LLAMA_CPP_DIR)"
    log "Note: -DGGML_CUDA_FA_ALL_QUANTS=ON is load-bearing (M0_RUNBOOK.md G5) — without it, some -ctk/-ctv quant combos silently fall off the fused flash-attention path (25-45x slower prefill, no warning)."

    local nproc_n
    nproc_n="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"

    (
        cd "$LLAMA_CPP_DIR"
        cmake "${cmake_args[@]}"
        cmake --build build --config Release -j"$nproc_n"
    )

    LLAMA_BIN_DIR="$LLAMA_CPP_DIR/build/bin"
    RESOLVED_BUILD_SOURCE="source build ($tag, arch=$CUDA_ARCH)"
}

link_binaries() {
    local target_dir="$HOME/.local/bin"
    mkdir -p "$target_dir"
    local bin
    for bin in llama-server llama-bench llama-cli llama-perplexity llama-imatrix llama-quantize; do
        if [[ -x "$LLAMA_BIN_DIR/$bin" ]]; then
            ln -sf "$LLAMA_BIN_DIR/$bin" "$target_dir/$bin"
        fi
    done
    log "Symlinked llama.cpp binaries into $target_dir"
    case ":$PATH:" in
        *":$target_dir:"*) ;;
        *) warn "$target_dir is not on \$PATH. Add: export PATH=\"$target_dir:\$PATH\"" ;;
    esac

    if [[ -x "$LLAMA_BIN_DIR/llama-cli" ]]; then
        log "ldd check (G2 stale-libggml-cuda.so trap):"
        ldd "$LLAMA_BIN_DIR/llama-server" 2>/dev/null | grep -i cuda || true
    fi
}

# ------------------------------------------------------------------------
# Step 3 — python deps
# ------------------------------------------------------------------------
pip_install() {
    # $1: pip package spec, e.g. "requests" or "huggingface_hub[hf_transfer]"
    local spec="$1"
    if python3 -m pip install --user "$spec" >/tmp/bootstrap_pip.log 2>&1; then
        return 0
    fi
    if grep -qi "externally-managed-environment" /tmp/bootstrap_pip.log; then
        log "PEP 668 externally-managed environment detected for '$spec' — retrying with --break-system-packages (per harness/README.md)."
        python3 -m pip install --user --break-system-packages "$spec"
        return 0
    fi
    cat /tmp/bootstrap_pip.log >&2
    return 1
}

ensure_python_deps() {
    log "=== Step 3/7: python dependencies ==="

    if python3 -c "import requests" >/dev/null 2>&1; then
        log "requests already importable — skipping."
    else
        log "Installing requests (harness's only third-party dependency)."
        pip_install "requests"
    fi

    if command -v hf >/dev/null 2>&1; then
        log "hf CLI already on \$PATH ($(command -v hf))."
    else
        log "Installing huggingface_hub[hf_transfer] (provides the 'hf' CLI; huggingface-cli was removed in v1.0 — M0_RUNBOOK.md G8)."
        pip_install "huggingface_hub[hf_transfer]"
        if ! command -v hf >/dev/null 2>&1; then
            local user_base
            user_base="$(python3 -m site --user-base 2>/dev/null || true)"
            if [[ -n "$user_base" && -x "$user_base/bin/hf" ]]; then
                export PATH="$user_base/bin:$PATH"
                log "Added $user_base/bin to PATH for this run."
            fi
        fi
        command -v hf >/dev/null 2>&1 || die "installed huggingface_hub but 'hf' is still not on \$PATH — add its install dir to \$PATH and re-run."
    fi
    export HF_HUB_ENABLE_HF_TRANSFER=1
}

# ------------------------------------------------------------------------
# Step 4 — model downloads
# ------------------------------------------------------------------------
# Byte counts below are copied verbatim from M0_RUNBOOK.md §1 (exact bytes
# from the HF API, 2026-08-23). Used only to print a size estimate before
# downloading — never to validate downloaded files (hf download/HF hub
# already does integrity checking).
UNSLOTH_UD_Q2_K_XL_BYTES=9828981664
UNSLOTH_UD_IQ3_XXS_BYTES=10934860704
UNSLOTH_UD_Q3_K_XL_BYTES=13146393504
UNSLOTH_UD_Q4_K_XL_BYTES=17559178144
UNSLOTH_UD_Q4_K_S_BYTES=15358213024
UNSLOTH_IMATRIX_BYTES=13642656
UNSLOTH_MTP_Q4_0_BYTES=1369590656
BARTOWSKI_IQ2_S_BYTES=10295330400
BARTOWSKI_IQ3_XXS_BYTES=12626773600
BARTOWSKI_IMATRIX_BYTES=13642688
BARTOWSKI_CALIB_BYTES=1258850
GGML_Q4_K_M_BYTES=18973870432
GGML_Q8_0_BYTES=28595763552

bytes_to_gib_str() {
    awk -v b="$1" 'BEGIN { printf "%.2f", b / (1024*1024*1024) }'
}

hf_dl() {
    # hf_dl <repo> <local_dir> <include...>
    local repo="$1" local_dir="$2"
    shift 2
    mkdir -p "$local_dir"
    log "hf download $repo --include $* --local-dir $local_dir"
    hf download "$repo" --include "$@" --local-dir "$local_dir"
}

download_models() {
    log "=== Step 4/7: model artifact download (tier=$TIER) ==="

    local unsloth_dir="$MODELS_DIR/unsloth-q27b"
    local bartowski_dir="$MODELS_DIR/bartowski-q27b"
    local ggml_dir="$MODELS_DIR/ggml-q27b"
    local templates_dir="$MODELS_DIR/templates"

    Q2KXL_PATH="$unsloth_dir/Qwen3.8-27B-UD-Q2_K_XL.gguf"
    Q4KS_PATH="$unsloth_dir/Qwen3.8-27B-UD-Q4_K_S.gguf"
    MTP_PATH="$unsloth_dir/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
    TEMPLATE_PATH="$templates_dir/chat_template.jinja"

    local total_bytes=$((UNSLOTH_UD_Q2_K_XL_BYTES + UNSLOTH_UD_Q4_K_S_BYTES + UNSLOTH_MTP_Q4_0_BYTES + UNSLOTH_IMATRIX_BYTES))
    if [[ "$TIER" == "full" ]]; then
        total_bytes=$((total_bytes + UNSLOTH_UD_IQ3_XXS_BYTES + UNSLOTH_UD_Q3_K_XL_BYTES + UNSLOTH_UD_Q4_K_XL_BYTES))
        total_bytes=$((total_bytes + BARTOWSKI_IQ2_S_BYTES + BARTOWSKI_IQ3_XXS_BYTES + BARTOWSKI_IMATRIX_BYTES + BARTOWSKI_CALIB_BYTES))
        total_bytes=$((total_bytes + GGML_Q4_K_M_BYTES + GGML_Q8_0_BYTES))
    fi

    local total_gib
    total_gib="$(bytes_to_gib_str "$total_bytes")"
    log "Estimated download size for tier '$TIER': ${total_gib} GiB (mmproj files explicitly excluded, per M0_RUNBOOK.md §8 G1)."

    if awk -v g="$total_gib" 'BEGIN{exit !(g>30)}'; then
        if [[ "$ASSUME_YES" -ne 1 ]]; then
            read -r -p "[bootstrap] This will download ~${total_gib} GiB. Continue? [y/N] " reply
            case "$reply" in
                [yY] | [yY][eE][sS]) ;;
                *) die "aborted by operator (re-run with --yes to skip this prompt)." ;;
            esac
        else
            log "Download >30 GiB but --yes was passed — continuing without prompting."
        fi
    fi

    # Chat template (both tiers) — froggeric's fix for the two verified
    # chat-template bugs in M0_RUNBOOK.md §2d. Load-bearing for every entry.
    hf_dl "$CHAT_TEMPLATE_REPO" "$templates_dir" "chat_template.jinja"

    # Minimal tier: the two baseline artifacts the generated matrix uses --
    # UD-Q2_K_XL (plain full-GPU -ngl 99 baseline) and UD-Q4_K_S + its MTP
    # draft head + imatrix (the §3 community FFN-offload + MTP recipe).
    hf_dl "unsloth/Qwen3.8-27B-GGUF" "$unsloth_dir" "Qwen3.8-27B-UD-Q2_K_XL.gguf"
    hf_dl "unsloth/Qwen3.8-27B-GGUF" "$unsloth_dir" \
        "Qwen3.8-27B-UD-Q4_K_S.gguf" "MTP/mtp-Qwen3.8-27B-Q4_0.gguf" "imatrix_unsloth.gguf"

    if [[ "$TIER" == "full" ]]; then
        log "Full tier: downloading the remainder of the M0_RUNBOOK.md §1e GGUF ladder."
        hf_dl "unsloth/Qwen3.8-27B-GGUF" "$unsloth_dir" "Qwen3.8-27B-UD-Q4_K_XL.gguf"
        hf_dl "unsloth/Qwen3.8-27B-GGUF" "$unsloth_dir" \
            "Qwen3.8-27B-UD-IQ3_XXS.gguf" "Qwen3.8-27B-UD-Q3_K_XL.gguf"
        hf_dl "bartowski/Qwen3.8-27B-GGUF" "$bartowski_dir" \
            "Qwen3.8-27B-IQ2_S.gguf" "Qwen3.8-27B-IQ3_XXS.gguf" \
            "Qwen3.8-27B-imatrix.gguf" "Qwen3.8-27B-calibration-v6.txt"
        hf_dl "ggml-org/Qwen3.8-27B-GGUF" "$ggml_dir" "Qwen3.8-27B-Q4_K_M.gguf"
        hf_dl "ggml-org/Qwen3.8-27B-GGUF" "$ggml_dir" "Qwen3.8-27B-Q8_0.gguf"
    fi

    log "Verifying no mmproj-* landed on disk under $MODELS_DIR ..."
    local stray_mmproj
    stray_mmproj="$(find "$MODELS_DIR" -iname 'mmproj-*' 2>/dev/null || true)"
    if [[ -n "$stray_mmproj" ]]; then
        warn "found stray mmproj file(s) — these were NOT requested by this script and eat ~0.87 GiB of VRAM if auto-loaded (M0_RUNBOOK.md G1). Investigate/remove:"
        printf '%s\n' "$stray_mmproj" >&2
    else
        log "Confirmed: no mmproj-* files present."
    fi

    [[ -f "$Q2KXL_PATH" ]] || die "expected artifact missing after download: $Q2KXL_PATH"
    [[ -f "$Q4KS_PATH" ]] || die "expected artifact missing after download: $Q4KS_PATH"
    [[ -f "$MTP_PATH" ]] || die "expected artifact missing after download: $MTP_PATH"
    [[ -f "$TEMPLATE_PATH" ]] || die "expected artifact missing after download: $TEMPLATE_PATH"
}

# ------------------------------------------------------------------------
# Step 5 — hardware profile snapshot
# ------------------------------------------------------------------------
run_hw_profile() {
    log "=== Step 5/7: hardware profile snapshot ==="
    mkdir -p "$REPO_ROOT/results"
    local out_path="$REPO_ROOT/results/hw_profile_${HOST}.json"
    python3 "$SCRIPT_DIR/profile_hw.py" --out "$out_path"
    log "Wrote $out_path"
}

# ------------------------------------------------------------------------
# Step 6 — generate configs/matrix.$(hostname).yaml
# ------------------------------------------------------------------------
generate_matrix_config() {
    log "=== Step 6/7: generating matrix config ==="
    mkdir -p "$REPO_ROOT/configs"
    local out_path="$REPO_ROOT/configs/matrix.${HOST}.yaml"

    # NOTE on shape: harness/yaml_lite.py is a minimal stdlib parser — flat
    # list of string-valued mappings only, no nesting, no multi-line values.
    # It splits each line on the FIRST ':' only and strips a matching pair
    # of outer quotes if-and-only-if the value's first and last characters
    # match; it does NOT do YAML escape processing. That means embedded
    # single/double quotes (e.g. the JSON in --chat-template-kwargs, or the
    # double-quoted --override-tensor regex) are safe here as long as we
    # do NOT wrap the whole flags value in an outer quote pair -- so the
    # entries below deliberately leave server_flags/bench_flags unquoted at
    # the top level. Verified against yaml_lite.load_matrix_yaml() +
    # shlex.split() (the same path run_matrix.py uses) before shipping this
    # script.
    cat >"$out_path" <<EOF
# Auto-generated by harness/bootstrap_linux.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ) for host ${HOST}.
# Tier: ${TIER}. Parsed by harness/yaml_lite.py -- keep entries flat, no nesting.
#
# Entry 1: plain full-GPU baseline (spec item: "a plain -ngl 99 entry").
# Entry 2: the M0_RUNBOOK.md §3a community FFN-offload + MTP recipe,
#          translated to Linux and to run_matrix.py's -m/--port convention
#          (run_matrix.py adds -m <gguf_path> and --port <free port> itself;
#          don't duplicate those here). -fit off and --gpu-layers-draft all
#          from the original .bat are omitted -- they're absent from current
#          master's server --help (M0_RUNBOOK.md §3a note); verify with
#          './build/bin/llama-server --help | grep -E "fit|draft"' on your
#          build and adapt if needed.
#
# reasoning_effort is frozen at "medium" across both entries per
# M0_RUNBOOK.md §8 G3 (default xhigh burns the token budget). -fa on is
# explicit per G4 (auto can silently disable FA). --no-mmproj is explicit
# per G1 (mmproj is auto-loaded and auto-offloaded to GPU otherwise).

- name: qwen27b-unsloth-ud-q2kxl-ngl99
  gguf_path: ${Q2KXL_PATH}
  server_flags: --jinja --chat-template-file ${TEMPLATE_PATH} --chat-template-kwargs '{"reasoning_effort":"medium"}' -fa on -ngl 99 -ctk q4_0 -ctv q4_0 --no-mmproj -c 4096 --host 127.0.0.1
  bench_flags: -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -d 0,4096

- name: qwen27b-unsloth-ud-q4ks-ffnoffload-mtp
  gguf_path: ${Q4KS_PATH}
  server_flags: --jinja --chat-template-file ${TEMPLATE_PATH} --reasoning-preserve --chat-template-kwargs '{"reasoning_effort":"medium"}' -fa on -ngl 99 --override-tensor "blk\.([0-9]|[1-3][0-9]|4[0-5])\.ffn_.*=CPU" -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-model ${MTP_PATH} -lv 4 --no-mmproj -np 1 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --load-mode none --no-warmup -b 256 -ub 128 -c 98304 --host 127.0.0.1
  bench_flags: -ngl 99 -fa on --override-tensor "blk\.([0-9]|[1-3][0-9]|4[0-5])\.ffn_.*=CPU" -ctk q4_0 -ctv q4_0 -d 0,4096,16384,65536
EOF

    log "Wrote $out_path"
    log "Sanity-parsing it with harness/yaml_lite.py ..."
    python3 - "$out_path" "$SCRIPT_DIR" <<'PYEOF'
import sys
out_path, script_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, script_dir)
import yaml_lite, shlex  # noqa: E402
entries = yaml_lite.load_matrix_yaml(out_path)
assert len(entries) == 2, f"expected 2 entries, got {len(entries)}"
for e in entries:
    shlex.split(e["server_flags"])
    shlex.split(e["bench_flags"])
print(f"[bootstrap] OK: {len(entries)} entries parse cleanly and their flags shlex.split() without error.")
PYEOF
}

# ------------------------------------------------------------------------
# Step 7 — next command
# ------------------------------------------------------------------------
print_next_steps() {
    log "=== Step 7/7: summary ==="
    echo
    echo "============================================================"
    echo " BOOTSTRAP SUMMARY"
    echo "============================================================"
    echo " GPU:                 $DETECTED_GPU_NAME"
    echo " CUDA arch used:      $CUDA_ARCH"
    echo " llama.cpp binaries:  $LLAMA_BIN_DIR"
    echo " llama.cpp build:     ${RESOLVED_BUILD:-unknown} (${RESOLVED_BUILD_SOURCE:-unknown})"
    echo " Models dir:          $MODELS_DIR"
    echo " Tier downloaded:     $TIER"
    echo " HW profile:          $REPO_ROOT/results/hw_profile_${HOST}.json"
    echo " Matrix config:       $REPO_ROOT/configs/matrix.${HOST}.yaml"
    echo "============================================================"
    echo
    echo "Before trusting any numbers, run the correctness smoke test"
    echo "(M0_RUNBOOK.md §2g -- the Gated DeltaNet CUDA bug produces fast,"
    echo "well-formed GARBAGE, not an error):"
    echo
    echo "  $LLAMA_BIN_DIR/llama-cli -m \"$Q2KXL_PATH\" \\"
    echo "    -ngl 99 -c 4096 --temp 0 -n 128 \\"
    echo "    -p \"Write a Python function that returns the nth Fibonacci number.\" \\"
    echo "    --jinja --chat-template-file \"$TEMPLATE_PATH\""
    echo
    echo "Then compare against a CPU/Vulkan run of the same prompt. If the"
    echo "CUDA output differs qualitatively, stop -- do not benchmark on a"
    echo "bad build."
    echo
    echo "NEXT COMMAND (run the M0 matrix once the smoke test passes):"
    echo
    echo "  cd \"$REPO_ROOT\""
    echo "  python3 harness/run_matrix.py \\"
    echo "    --config configs/matrix.${HOST}.yaml \\"
    echo "    --out results/results.${HOST}.jsonl \\"
    echo "    --limit-gsm8k 50 --limit-humaneval 20"
    echo
    if [[ "$TIER" == "minimal" ]]; then
        echo "(This tier covers only the 2 baseline entries above. Re-run this"
        echo "script with --tier full to pull the rest of the M0_RUNBOOK.md §1e"
        echo "GGUF ladder for the full matrix -- ~130 GiB, requires confirmation.)"
        echo
    fi
    echo "Resolved llama.cpp build: ${RESOLVED_BUILD:-unknown}"
}

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
main() {
    log "q27b_on_12gb M0 bootstrap starting on host '$HOST' (tier=$TIER)"
    detect_hardware
    ensure_llama_cpp
    ensure_python_deps
    download_models
    run_hw_profile
    generate_matrix_config
    print_next_steps
}

main "$@"

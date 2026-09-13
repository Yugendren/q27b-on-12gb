#!/usr/bin/env bash
# S1 — occupancy-honest measurement bundle (07-SYNTHESIS.md S1, ~1 GPU-day).
# Answers, in one unattended run: does the #27623 decode cliff exist at real
# occupancy (A); does prefill collapse at sub-8-bit K (B); does the format
# ceiling track format not size (C); does KV precision buy speed at a filled
# 16K context on the flagship quant (D).
#
# Unlike the old allocated-only context sweep (run_ablations.sh EXP2), this
# script uses `llama-bench -p <N>` to actually FILL the KV cache with N
# prompt tokens before the tg (decode) phase is measured — the pp phase and
# the tg phase share the same llama-bench invocation and the same filled
# context, so a "decode cliff" measured here reflects real occupancy, not
# an empty-cache best case.
#
# Parts A, B, D need quantized K/V (q4_0 etc.) and so use the explicit
# FA_ALL_QUANTS build at /data/projects/llama.cpp/build/bin/llama-bench
# (build 10712). Part C has no KV-quant requirement and uses the PATH build
# (~/.local/bin, also 10712) for the format-bandwidth table.
#
# Appends TSV rows to results/s1_occupancy.tsv (schema modeled on
# ablations.tsv but widened for llama-bench's pp/tg-per-point + format
# metrics). Resumable: each row is skipped if already present for the same
# exp/build/config/param key. Costs zero API credits, no server required
# (llama-bench only) except none — all measurements are llama-bench, not
# llama-server.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /data/projects/q27b_on_12gb || exit 1

LB_FA=/data/projects/llama.cpp/build/bin/llama-bench   # FA_ALL_QUANTS build, required for q4_0/q8_0 KV combos
LB_STD=llama-bench                                      # PATH build (~/.local/bin)
M=models/unsloth-q27b
OUT=results/s1_occupancy.tsv
mkdir -p results
[ -f "$OUT" ] || printf 'exp\tbuild\tconfig\tparam\tvram_mib\tprefill_ts\tdecode_ts\tfilesize_gb\teff_gbs\tnote\n' > "$OUT"

log() { echo "[s1 $(date -u +%H:%M:%S)] $*"; }

sanitize() { printf '%s' "$1" | tr '\t\n\r' '   '; }

have_bin() {
  local b="$1"
  if [[ "$b" == */* ]]; then
    [ -x "$b" ]
  else
    command -v "$b" >/dev/null 2>&1
  fi
}

# --- resumability: has this exact (exp,build,config,param) row already landed? ---
row_exists() {
  local exp="$1" build="$2" cfg="$3" param="$4"
  [ -f "$OUT" ] || return 1
  awk -F'\t' -v e="$exp" -v b="$build" -v c="$cfg" -v p="$param" \
    '$1==e && $2==b && $3==c && $4==p {found=1} END{exit !found}' "$OUT"
}

append_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" >> "$OUT"
}

# --- GPU-guard: block until VRAM is free (<500 MiB used), before every GPU op ---
wait_gpu_free() {
  local tag="${1:-gpu}"
  log "[$tag] waiting for GPU to be free (<500 MiB used)..."
  local i used
  for i in $(seq 1 900); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 500 ]; then
      log "[$tag] GPU free (${used} MiB used) after ${i}s"
      return 0
    fi
    if [ "$i" -eq 900 ]; then
      log "[$tag] WARNING: GPU still busy (${used:-?} MiB used) after 900s, proceeding anyway"
    fi
    sleep 1
  done
}

# --- run a llama-bench invocation in the background, polling nvidia-smi for
#     peak VRAM used until it exits. Sets $LAST_VRAM. ---
LAST_VRAM="-"
run_llama_bench_vram() {
  local out_json="$1" out_log="$2"; shift 2
  : > "$out_json"; : > "$out_log"
  ( "$@" > "$out_json" 2>"$out_log" ) &
  local pid=$!
  local max=0 used
  while kill -0 "$pid" 2>/dev/null; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -gt "$max" ]; then max=$used; fi
    sleep 1
  done
  wait "$pid"
  local rc=$?
  LAST_VRAM=$max
  return $rc
}

# --- parse a llama-bench -o json array for the pp entry (n_prompt==$2, n_gen==0)
#     and the tg entry (n_prompt==0, n_gen==$3). Echoes "pp tg" (each may be "-"). ---
parse_pp_tg() {
  python3 - "$1" "$2" "$3" <<'PYEOF'
import json, sys
path, np_expect, ng_expect = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    data = json.load(open(path))
except Exception:
    print("- -")
    sys.exit(0)
pp = tg = None
for row in data:
    n_prompt = row.get("n_prompt", 0)
    n_gen = row.get("n_gen", 0)
    if n_prompt == np_expect and n_gen == 0 and pp is None:
        pp = row.get("avg_ts")
    if n_gen == ng_expect and n_prompt == 0 and tg is None:
        tg = row.get("avg_ts")
print(f"{pp if pp is not None else '-'} {tg if tg is not None else '-'}")
PYEOF
}

filesize_gb() {
  local bytes
  bytes=$(stat --format=%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null)
  if [ -z "$bytes" ]; then echo "-"; return; fi
  awk -v b="$bytes" 'BEGIN{printf "%.4f", b/1000000000}'
}

############################################################################
# PART A — filled-context decode sweep (the #27623 decode-cliff check, real
# occupancy). Q2_K_XL, ctk/ctv q4_0, -p actually fills the cache first.
############################################################################
log "=== A: filled-context decode sweep (Q2_K_XL, ctk/ctv q4_0) ==="
MODEL_A="$M/Qwen3.8-27B-UD-Q2_K_XL.gguf"
BUILD_A="Qwen3.8-27B-UD-Q2_K_XL"
CFG_A="ngl99_fa1_ctkq4_ctvq4"
for p in 8192 32768 65536 90112; do
  if row_exists s1a "$BUILD_A" "$CFG_A" "$p"; then
    log "SKIP s1a p=$p (already have)"; continue
  fi
  if [ ! -f "$MODEL_A" ]; then
    append_row s1a "$BUILD_A" "$CFG_A" "$p" - - - - - "FAIL:model_missing:$MODEL_A"
    log "FAIL s1a p=$p model missing"; continue
  fi
  if ! have_bin "$LB_FA"; then
    append_row s1a "$BUILD_A" "$CFG_A" "$p" - - - - - "FAIL:binary_missing:$LB_FA"
    log "FAIL s1a p=$p binary missing"; continue
  fi
  wait_gpu_free "s1a_p$p"
  ctxsize=$(( p + 128 + 512 ))
  json="results/s1a_p${p}.json"; logf="results/s1a_p${p}.log"
  run_llama_bench_vram "$json" "$logf" "$LB_FA" -m "$MODEL_A" -ngl 99 -fa 1 \
    -ctk q4_0 -ctv q4_0 -c "$ctxsize" -p "$p" -n 128 -r 2 -o json
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$json" ]; then
    note=$(sanitize "FAIL:rc=$rc:$(tail -c 300 "$logf")")
    append_row s1a "$BUILD_A" "$CFG_A" "$p" "$LAST_VRAM" - - - - "$note"
    log "FAIL s1a p=$p"; continue
  fi
  read -r pp tg <<< "$(parse_pp_tg "$json" "$p" 128)"
  append_row s1a "$BUILD_A" "$CFG_A" "$p" "$LAST_VRAM" "${pp:--}" "${tg:--}" - - ""
  log "OK s1a p=$p pp=$pp tg=$tg vram=$LAST_VRAM"
done

############################################################################
# PART B — sub-8-bit-K prefill hazard (checking the reported 20x prefill
# collapse, #27109). Same Q2_K_XL model, -p 16384 -n 32, 4 KV combos.
############################################################################
log "=== B: sub-8-bit-K prefill hazard, p=16384 ==="
MODEL_B="$MODEL_A"
BUILD_B="$BUILD_A"
P_B=16384
for combo in "f16_f16:f16:f16" "q8_q8:q8_0:q8_0" "q4_q4:q4_0:q4_0" "q4k_q8v:q4_0:q8_0"; do
  name=${combo%%:*}; rest=${combo#*:}; ctk=${rest%%:*}; ctv=${rest#*:}
  cfg="ngl99_fa1_ctk${ctk}_ctv${ctv}"
  if row_exists s1b "$BUILD_B" "$cfg" "$P_B"; then
    log "SKIP s1b $name (already have)"; continue
  fi
  if [ ! -f "$MODEL_B" ]; then
    append_row s1b "$BUILD_B" "$cfg" "$P_B" - - - - - "FAIL:model_missing:$MODEL_B"; continue
  fi
  if ! have_bin "$LB_FA"; then
    append_row s1b "$BUILD_B" "$cfg" "$P_B" - - - - - "FAIL:binary_missing:$LB_FA"; continue
  fi
  wait_gpu_free "s1b_$name"
  json="results/s1b_${name}.json"; logf="results/s1b_${name}.log"
  run_llama_bench_vram "$json" "$logf" "$LB_FA" -m "$MODEL_B" -ngl 99 -fa 1 \
    -ctk "$ctk" -ctv "$ctv" -c 17408 -p "$P_B" -n 32 -r 2 -o json
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$json" ]; then
    note=$(sanitize "FAIL:rc=$rc:$(tail -c 300 "$logf")")
    append_row s1b "$BUILD_B" "$cfg" "$P_B" "$LAST_VRAM" - - - - "$note"
    log "FAIL s1b $name"; continue
  fi
  read -r pp tg <<< "$(parse_pp_tg "$json" "$P_B" 32)"
  append_row s1b "$BUILD_B" "$cfg" "$P_B" "$LAST_VRAM" "${pp:--}" "${tg:--}" - - ""
  log "OK s1b $name pp=$pp tg=$tg vram=$LAST_VRAM"
done

############################################################################
# PART C — format bandwidth ceiling (Ampere format-speed table). Downloads
# small Qwen3.5-9B-GGUF variants (Q4_0, Q4_K_M, IQ4_XS) and computes
# effective GB/s = filesize_GB * decode_tok_s for each.
############################################################################
log "=== C: format bandwidth ceiling (Qwen3.5-9B-GGUF variants) ==="
export HF_HOME=/data/hf-cache
FMT_DIR=models/bench/qwen3.5-9b-gguf
mkdir -p "$FMT_DIR"
for fmt in Q4_0 Q4_K_M IQ4_XS; do
  cfg="fmt_${fmt}"
  if row_exists s1c Qwen3.5-9B "$cfg" pp512; then
    log "SKIP s1c $fmt (already have)"; continue
  fi
  gguf=$(ls "$FMT_DIR"/*"${fmt}"*.gguf 2>/dev/null | head -1)
  if [ -z "$gguf" ]; then
    log "downloading $fmt via hf CLI ..."
    if ! hf download unsloth/Qwen3.5-9B-GGUF --include "*${fmt}*.gguf" \
        --local-dir "$FMT_DIR" > "results/s1c_dl_${fmt}.log" 2>&1; then
      note=$(sanitize "FAIL:download:$(tail -c 300 "results/s1c_dl_${fmt}.log")")
      append_row s1c Qwen3.5-9B "$cfg" pp512 - - - - - "$note"
      log "FAIL s1c $fmt download"; continue
    fi
    gguf=$(ls "$FMT_DIR"/*"${fmt}"*.gguf 2>/dev/null | head -1)
  fi
  if [ -z "$gguf" ]; then
    append_row s1c Qwen3.5-9B "$cfg" pp512 - - - - - "FAIL:file_not_found_after_download"
    log "FAIL s1c $fmt: no gguf matching *${fmt}*.gguf in $FMT_DIR"; continue
  fi
  if ! have_bin "$LB_STD"; then
    append_row s1c Qwen3.5-9B "$cfg" pp512 - - - - - "FAIL:binary_missing:$LB_STD"; continue
  fi
  wait_gpu_free "s1c_$fmt"
  json="results/s1c_${fmt}.json"; logf="results/s1c_${fmt}.log"
  run_llama_bench_vram "$json" "$logf" "$LB_STD" -m "$gguf" -ngl 99 -fa 1 -p 512 -n 128 -r 3 -o json
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$json" ]; then
    note=$(sanitize "FAIL:rc=$rc:$(tail -c 300 "$logf")")
    append_row s1c Qwen3.5-9B "$cfg" pp512 "$LAST_VRAM" - - - - "$note"
    log "FAIL s1c $fmt"; continue
  fi
  read -r pp tg <<< "$(parse_pp_tg "$json" 512 128)"
  fsize=$(filesize_gb "$gguf")
  eff_gbs="-"
  if [[ "$tg" =~ ^[0-9.]+$ && "$fsize" =~ ^[0-9.]+$ ]]; then
    eff_gbs=$(awk -v fs="$fsize" -v tg="$tg" 'BEGIN{printf "%.3f", fs*tg}')
  fi
  append_row s1c Qwen3.5-9B "$cfg" pp512 "$LAST_VRAM" "${pp:--}" "${tg:--}" "$fsize" "$eff_gbs" ""
  log "OK s1c $fmt pp=$pp tg=$tg size=${fsize}GB eff=${eff_gbs}GB/s"
done

############################################################################
# PART D — KV precision at a FILLED 16K context, flagship IQ3_XXS quant.
############################################################################
log "=== D: KV precision sweep at filled 16K (IQ3_XXS) ==="
MODEL_D="$M/Qwen3.8-27B-UD-IQ3_XXS.gguf"
BUILD_D="Qwen3.8-27B-UD-IQ3_XXS"
P_D=16384
for kv in "f16:f16:f16" "q8:q8_0:q8_0" "q4:q4_0:q4_0"; do
  name=${kv%%:*}; rest=${kv#*:}; ctk=${rest%%:*}; ctv=${rest#*:}
  cfg="kv_${name}"
  if row_exists s1d "$BUILD_D" "$cfg" "$P_D"; then
    log "SKIP s1d $name (already have)"; continue
  fi
  if [ ! -f "$MODEL_D" ]; then
    append_row s1d "$BUILD_D" "$cfg" "$P_D" - - - - - "FAIL:model_missing:$MODEL_D"; continue
  fi
  if ! have_bin "$LB_FA"; then
    append_row s1d "$BUILD_D" "$cfg" "$P_D" - - - - - "FAIL:binary_missing:$LB_FA"; continue
  fi
  wait_gpu_free "s1d_$name"
  json="results/s1d_${name}.json"; logf="results/s1d_${name}.log"
  run_llama_bench_vram "$json" "$logf" "$LB_FA" -m "$MODEL_D" -ngl 99 -fa 1 \
    -ctk "$ctk" -ctv "$ctv" -c 17408 -p "$P_D" -n 128 -r 2 -o json
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$json" ]; then
    note=$(sanitize "FAIL:rc=$rc:$(tail -c 300 "$logf")")
    append_row s1d "$BUILD_D" "$cfg" "$P_D" "$LAST_VRAM" - - - - "$note"
    log "FAIL s1d $name"; continue
  fi
  read -r pp tg <<< "$(parse_pp_tg "$json" "$P_D" 128)"
  append_row s1d "$BUILD_D" "$cfg" "$P_D" "$LAST_VRAM" "${pp:--}" "${tg:--}" - - ""
  log "OK s1d $name pp=$pp tg=$tg vram=$LAST_VRAM"
done

echo S1DONE > results/s1_occupancy_done.txt
log "ALL S1 OCCUPANCY MEASUREMENTS COMPLETE"

#!/usr/bin/env bash
# Platform arm: re-run the L40S quality battery's HumanEval-164 leg on the
# RTX 3060 (sm_86) with THE SAME harness file (byte-identical reliability.py),
# the same server flags, the same greedy no-spec protocol and the same GGUF
# files. The only variable left is the GPU architecture.
#
# usage: platform_arm.sh he <BUILD> <NGL> [LIMIT]
set -uo pipefail
W=/data/projects/q27b_on_12gb
SRV=${SRV_BIN:-/data/scratch/bin-unpatched/llama-server}
M=$W/models/unsloth-q27b
R=$W/results/platform_arm
H=$W/platform_arm/harness
PORT=18099
mkdir -p "$R" "$W/platform_arm/logs"

log(){ echo "[parm $(date -u +%H:%M:%S)] $*" | tee -a "$W/platform_arm/logs/platform_arm.log"; }

MODEL_IQ3=$M/Qwen3.8-27B-UD-IQ3_XXS.gguf
MODEL_Q2K=$M/Qwen3.8-27B-UD-Q2_K_XL.gguf
MODEL_Q8=$M/Qwen3.8-27B-Q8_0.gguf
MODEL_IQ2X=$M/Qwen3.8-27B-UD-IQ2_XXS.gguf

SRVPID=""
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# GPU guard: refuse to start a server unless the card is essentially empty,
# so nothing else on this box can perturb the measurement.
guard(){
  for i in $(seq 1 120); do
    u=$(gpu_used); [ "${u:-9999}" -lt 500 ] && { log "GPU guard OK (${u} MiB)"; return 0; }
    sleep 2
  done
  log "GPU GUARD FAILED: $(gpu_used) MiB still resident"; return 1
}

stop_srv(){
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRVPID" 2>/dev/null
  SRVPID=""
  for i in $(seq 1 120); do
    u=$(gpu_used); [ "${u:-9999}" -lt 500 ] && return 0; sleep 1
  done
}
trap 'stop_srv' EXIT

start_srv(){
  local tag="$1" model="$2"; shift 2
  local lg="$R/srv_${tag}.log"
  [ -f "$model" ] || { log "MISSING MODEL $model"; return 1; }
  guard || return 1
  local t0=$SECONDS
  dd if="$model" of=/dev/null bs=64M status=none 2>/dev/null
  log "prewarm $(basename "$model") in $((SECONDS-t0))s"
  log "START $tag :: $(basename "$model") :: $*"
  "$SRV" -m "$model" -fa on --host 127.0.0.1 --port $PORT --jinja --no-warmup "$@" > "$lg" 2>&1 &
  SRVPID=$!
  for i in $(seq 1 2400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && {
      local v; v=$(gpu_used); log "UP $tag in ${i}s, VRAM ${v} MiB"
      echo "$v" > "$R/vram_${tag}.txt"; return 0; }
    kill -0 $SRVPID 2>/dev/null || { log "SERVER DIED $tag"; tail -25 "$lg" | tee -a "$W/platform_arm/logs/platform_arm.log"; return 1; }
    sleep 1
  done
  log "TIMEOUT $tag"; return 1
}

case "${1:-}" in
he)
  B="${2:?build}"; NGL="${3:-99}"; LIMIT="${4:-164}"
  case "$B" in
    IQ2_XXS) MP=$MODEL_IQ2X ;; Q2_K_XL) MP=$MODEL_Q2K ;;
    IQ3_XXS) MP=$MODEL_IQ3 ;; Q8_0) MP=$MODEL_Q8 ;;
    *) echo "unknown build $B"; exit 2 ;;
  esac
  # EXACT nospec profile from the L40S battery (battery.sh, `he <B> nospec`):
  #   -ngl 99 -fa on --jinja --no-warmup -c 8192 -np 1 -ctk q8_0 -ctv q8_0 --spec-type none
  # Only -ngl varies, and only for the Q8_0 anchor which cannot fit on 12 GiB.
  PROF=(-ngl "$NGL" -c 8192 -np 1 -ctk q8_0 -ctv q8_0 --spec-type none)
  TAG="${B}_nospec_3060"
  start_srv "$TAG" "$MP" "${PROF[@]}" || exit 1
  python3 $H/reliability.py --endpoint http://127.0.0.1:$PORT --tag "${TAG}_n1" \
    --max-attempts 1 --limit "$LIMIT" --concurrency 1 --data-dir $H/data \
    --out $R/reliability_${TAG}.jsonl --summary $R/reliability_${TAG}_n1.json
  rc=$?
  stop_srv
  log "DONE $TAG rc=$rc"
  exit $rc
  ;;
*) echo "unknown phase: ${1:-}"; exit 2 ;;
esac

#!/usr/bin/env bash
# STAGE 2 -- quality certification of Qwen3.6-35B-A3B UD-Q4_K_XL on the 3060.
#
# Protocol of record: results/platform_arm/harness/platform_arm.sh, whose
# `he` phase serves
#     -ngl 99 -fa on --jinja --no-warmup -c 8192 -np 1 -ctk q8_0 -ctv q8_0
#     --spec-type none
# and drives it with the byte-identical reliability.py at --max-attempts 1,
# greedy, thinking OFF (reliability.py sends chat_template_kwargs.
# enable_thinking=false itself).
#
# The ONLY deviation forced by this model: it does not fit in 12 GiB, so the
# expert tensors of the first N layers live on the CPU (--n-cpu-moe). Quality
# is therefore certified AT THE SHIPPING PLACEMENT, which matters because the
# placement sweep showed draft acceptance drifting with placement (0.72 -> 0.86),
# i.e. CPU and CUDA expert kernels are not bit-identical.
#
# Speculation is OFF for the quality run (protocol of record). Shipping with
# MTP on is justified separately by mtp_identity.py, which must show
# byte-identical greedy output between --spec-type none and draft-mtp n=2.
set -uo pipefail

W=/data/projects/q27b_on_12gb
SRV=$W/llama-server-v11/llama-server
MODEL=/data/models/coding/q36_35ba3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
H=$W/platform_arm/harness
PH=$W/cert35b_harness
R=$W/results/cert35b
PORT=18095
NCM=${NCM:-26}
TAG=Q4_K_XL_ncm${NCM}_nospec_3060

mkdir -p "$R"
log(){ echo "[stage2 $(date -u +%H:%M:%S)] $*"; }

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

SRVPID=""
stop_srv(){
  [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null
  for i in $(seq 1 60); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRVPID" 2>/dev/null
  SRVPID=""
  for i in $(seq 1 120); do u=$(gpu_used); [ "${u:-9999}" -lt 500 ] && return 0; sleep 1; done
}
trap 'stop_srv' EXIT

# GPU guard -- refuse to measure on a dirty card.
for i in $(seq 1 120); do
  u=$(gpu_used); [ "${u:-9999}" -lt 500 ] && { log "GPU guard OK (${u} MiB)"; break; }
  sleep 2
done
u=$(gpu_used); [ "${u:-9999}" -lt 500 ] || { log "GPU GUARD FAILED (${u} MiB)"; exit 1; }

log "prewarming model into page cache"
dd if="$MODEL" of=/dev/null bs=64M status=none

log "starting server: ncm=$NCM, c=8192, q8_0 KV, spec=none"
"$SRV" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port $PORT --jinja --no-warmup \
  -c 8192 -np 1 -ctk q8_0 -ctv q8_0 --n-cpu-moe "$NCM" --spec-type none \
  > "$R/srv_${TAG}.log" 2>&1 &
SRVPID=$!
for i in $(seq 1 1200); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { log "UP in ${i}s, VRAM $(gpu_used) MiB"; break; }
  kill -0 $SRVPID 2>/dev/null || { log "SERVER DIED"; tail -30 "$R/srv_${TAG}.log"; exit 1; }
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { log "TIMEOUT"; exit 1; }
gpu_used > "$R/vram_${TAG}.txt"

# ---- leg 1: HumanEval-164, byte-identical harness, greedy, 1 attempt ----
log "HumanEval-164 starting"
python3 $H/reliability.py --endpoint http://127.0.0.1:$PORT --tag "${TAG}_n1" \
  --max-attempts 1 --limit 164 --concurrency 1 --data-dir $H/data \
  --out $R/reliability_${TAG}.jsonl --summary $R/reliability_${TAG}_n1.json
log "HumanEval rc=$?"

# ---- leg 2: polyglot edit-compliance, 34 tasks, reference params ----
# Reference run params (results/edit_IQ3_XXS.json): max_tokens 2048, attempts 2,
# test_timeout 30, seed 42, sample_seed 1337. Concurrency was 8 there; the server
# here has one slot (-np 1) so concurrency only affects queueing, not results --
# lowered to 2 with a long request timeout so nothing trips the 600 s default.
log "polyglot edit-compliance starting"
python3 $PH/polyglot_edit.py --endpoint http://127.0.0.1:$PORT \
  --src $PH/polyglot_src --tag "${TAG}" --n-tasks 34 \
  --max-tokens 2048 --attempts 2 --concurrency 2 --request-timeout 1800 \
  --out $R/edit_${TAG}.jsonl --summary $R/edit_${TAG}.json
log "polyglot rc=$?"

stop_srv
log "STAGE2_DONE"

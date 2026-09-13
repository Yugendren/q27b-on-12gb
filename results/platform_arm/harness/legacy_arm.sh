#!/usr/bin/env bash
# Attribution arm. If the 3060 reproduces the L40S numbers under the battery
# protocol, then the project's old 82.3%/72.0%/49.4% figures were caused by
# something other than the GPU. This arm reconstructs the OLD protocol on the
# SAME box and the SAME build and flips one knob at a time.
#
# Old protocol (q27b_on_12gb/harness/run_eval.sh + eval_quality.py):
#   server: -c 8192 --chat-template-file models/templates/chat_template.jinja
#           -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --spec-type draft-mtp
#           --spec-draft-n-max 2 --parallel 1        (NO --jinja, NO --no-warmup)
#   client: {model, messages, temperature:0, max_tokens}  -- no
#           chat_template_kwargs, so the froggeric-v22.3 template's
#           `enable_thinking ... else true` default leaves THINKING ON;
#           no reasoning_content fallback; closed-fence-only extract_code.
#
# arms:
#   L1  everything legacy                       -> should reproduce ~82-83%
#   L3  legacy EXCEPT enable_thinking=false     -> isolates the thinking switch
#
# usage: legacy_arm.sh <L1|L3> <BUILD> [LIMIT]
set -uo pipefail
W=/data/projects/q27b_on_12gb
SRV=${SRV_BIN:-/data/scratch/bin-unpatched/llama-server}
M=$W/models/unsloth-q27b
R=$W/results/platform_arm
H=$W/platform_arm/harness
PORT=18099
log(){ echo "[legacy $(date -u +%H:%M:%S)] $*" | tee -a "$W/platform_arm/logs/legacy.log"; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

ARM="${1:?arm}"; B="${2:?build}"; LIMIT="${3:-164}"
case "$B" in
  IQ2_XXS) MP=$M/Qwen3.8-27B-UD-IQ2_XXS.gguf ;;
  Q2_K_XL) MP=$M/Qwen3.8-27B-UD-Q2_K_XL.gguf ;;
  IQ3_XXS) MP=$M/Qwen3.8-27B-UD-IQ3_XXS.gguf ;;
  *) echo "unknown build $B"; exit 2 ;;
esac

case "$ARM" in
  L1) CLIENT_FLAGS=(--legacy-body --legacy-extract) ;;
  L3) CLIENT_FLAGS=(--legacy-extract) ;;   # new body => enable_thinking=false
  *)  echo "unknown arm $ARM"; exit 2 ;;
esac

TAG="${B}_${ARM}_3060"
for i in $(seq 1 120); do u=$(gpu); [ "${u:-9999}" -lt 500 ] && break; sleep 2; done
[ "$(gpu)" -lt 500 ] || { log "GPU GUARD FAILED ($(gpu) MiB)"; exit 1; }
log "GPU guard OK ($(gpu) MiB)"

log "START $TAG"
"$SRV" -m "$MP" -c 8192 --port $PORT --host 127.0.0.1 \
  --chat-template-file $W/models/templates/chat_template.jinja \
  -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --spec-type draft-mtp --spec-draft-n-max 2 \
  --parallel 1 > "$R/srv_${TAG}.log" 2>&1 &
SRVPID=$!
trap 'kill $SRVPID 2>/dev/null' EXIT
for i in $(seq 1 1200); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { log "UP $TAG in ${i}s, VRAM $(gpu) MiB"; break; }
  kill -0 $SRVPID 2>/dev/null || { log "SERVER DIED $TAG"; tail -25 "$R/srv_${TAG}.log"; exit 1; }
  sleep 1
done

python3 $H/reliability_probe.py --endpoint http://127.0.0.1:$PORT --tag "${TAG}_n1" \
  --max-attempts 1 --limit "$LIMIT" --concurrency 1 --data-dir $H/data \
  "${CLIENT_FLAGS[@]}" \
  --out $R/reliability_${TAG}.jsonl --summary $R/reliability_${TAG}_n1.json
rc=$?
kill $SRVPID 2>/dev/null
for i in $(seq 1 120); do u=$(gpu); [ "${u:-9999}" -lt 500 ] && break; sleep 1; done
log "DONE $TAG rc=$rc"

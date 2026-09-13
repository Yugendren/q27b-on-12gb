#!/usr/bin/env bash
# STAGE 2 — certification re-audits under the corrected protocol.
#
# (a) ASCII-prune re-verdict: full HumanEval-164 on the pruned flagship
#     through the EXACT code path that produced
#     results/platform_arm/reliability_IQ3_XXS_nospec_3060.jsonl (same
#     reliability.py, same greedy no-spec server profile), so the -4.9pt
#     legacy verdict can be re-taken as a paired comparison.
# (b) canary2 divergence, pruned + flagship, thinking off, against the
#     existing results/anchor_reference.json.
#
# One stage at a time, one server at a time, GPU guard before each.
set -uo pipefail
W=/data/projects/q27b_on_12gb
cd "$W"
R=$W/results
mkdir -p "$R/stage2"

log(){ echo "[stage2 $(date -u +%H:%M:%S)] $*" | tee -a "$R/stage2/stage2.log"; }

guard(){
  for i in $(seq 1 240); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${u:-9999}" -lt 500 ] && { log "GPU guard OK (${u} MiB)"; return 0; }
    sleep 2
  done
  log "GPU GUARD FAILED"; return 1
}

# ---------------------------------------------------------------- 2a
log "=== 2a ASCII-prune HumanEval-164, corrected protocol ==="
guard || exit 1
SRV_BIN=/data/scratch/bin-unpatched/llama-server \
  bash "$W/platform_arm/platform_arm.sh" he IQ3_XXS_ASCII 99 164 \
  > "$R/stage2/ascii_he164.log" 2>&1
log "2a rc=$? (see results/stage2/ascii_he164.log)"

guard || exit 1
python3 "$W/platform_arm/prune_cmp.py" > "$R/stage2/prune_reverdict.txt" 2>&1
log "2a verdict written"

# ---------------------------------------------------------------- 2b
# Same launcher shape as anchor_then_divergence.sh (which produced
# anchor_reference.json), plus --jinja so the enable_thinking switch is
# unambiguously live, and thinking is verified from the responses afterwards.
B=/data/projects/llama.cpp/build/bin/llama-server
PORT=18092
SRV=""
runsrv(){
  local m="$1"
  guard || return 1
  $B -m "$m" -ngl 99 -fa on -c 8192 --port $PORT --host 127.0.0.1 --jinja \
     --chat-template-file models/templates/chat_template.jinja \
     -ctk q8_0 -ctv q8_0 --no-warmup --parallel 1 > "$R/stage2/srv_$(basename $m .gguf).log" 2>&1 &
  SRV=$!
  for i in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $SRV 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
stopsrv(){ [ -n "$SRV" ] && kill $SRV 2>/dev/null; sleep 10; SRV=""; }
trap 'stopsrv' EXIT

log "=== 2b canary2 divergence, thinking off ==="
for pair in "Qwen3.8-27B-UD-IQ3_XXS-ASCII:pruned" "Qwen3.8-27B-UD-IQ3_XXS:flagship"; do
  f=${pair%%:*}; tag=${pair#*:}
  if runsrv "models/unsloth-q27b/${f}.gguf"; then
    python3 harness/canary2.py --endpoint http://127.0.0.1:$PORT \
      --reference $R/anchor_reference.json \
      --out "$R/stage2/canary2_re_${tag}.json" > "$R/stage2/canary2_re_${tag}.log" 2>&1
    log "2b $tag rc=$?"
  else
    log "2b $tag SERVER FAILED"
  fi
  stopsrv
done

python3 "$W/harness/canary_cmp.py" \
  "$R/stage2/canary2_re_pruned.json" "$R/stage2/canary2_re_flagship.json" \
  "$R/canary2_div_pruned.json" "$R/canary2_div_flagship.json" \
  > "$R/stage2/canary2_reaudit.txt" 2>&1
log "2b verdict written"
echo STAGE2_DONE > "$R/stage2/done.txt"

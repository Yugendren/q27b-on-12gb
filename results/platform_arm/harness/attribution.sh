#!/usr/bin/env bash
# Runs after chain.sh's sub-4-bit phase. Takes over the GPU, runs the two
# attribution arms (L1 = full legacy protocol, L3 = legacy but with thinking
# switched off), then hands the rest of the wall clock back to the Q8 anchor.
set -uo pipefail
W=/data/projects/q27b_on_12gb/platform_arm
R=/data/projects/q27b_on_12gb/results/platform_arm
log(){ echo "[attr $(date -u +%H:%M:%S)] $*" | tee -a "$W/logs/attribution.log"; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

log "waiting for SUB4 PHASE COMPLETE"
for i in $(seq 1 3600); do
  grep -q "SUB4 PHASE COMPLETE" "$W/logs/chain.log" 2>/dev/null && { log "sub4 done"; break; }
  sleep 10
done

# chain.sh moves straight on to the Q8 resume; stop it so the attribution arms
# get the GPU first. The Q8 anchor is restarted at the end of this script and
# resumes from its own JSONL, so nothing is lost.
pkill -f "bash chain.sh" && log "stopped chain.sh"
sleep 3
pkill -f "reliability.py --endpoint" 2>/dev/null
sleep 3
pkill -f "bin-unpatched/llama-server" 2>/dev/null
for i in $(seq 1 120); do u=$(gpu); [ "${u:-9999}" -lt 500 ] && break; sleep 2; done
log "GPU now $(gpu) MiB"

log "=== ARM L1: full legacy protocol, IQ3_XXS ==="
bash "$W/legacy_arm.sh" L1 IQ3_XXS 164 > "$W/logs/run_L1_IQ3_XXS.log" 2>&1
log "L1 rc=$?"

log "=== ARM L3: legacy but enable_thinking=false, IQ3_XXS ==="
bash "$W/legacy_arm.sh" L3 IQ3_XXS 164 > "$W/logs/run_L3_IQ3_XXS.log" 2>&1
log "L3 rc=$?"

log "=== resuming Q8_0 anchor toward 164 ==="
"$W/platform_arm.sh" he Q8_0 24 164 > "$W/logs/run_Q8_0_resume.log" 2>&1
log "ATTRIBUTION COMPLETE"

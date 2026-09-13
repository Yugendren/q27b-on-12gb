#!/usr/bin/env bash
# Chain driver for the platform arm.
#
# The Q8_0 anchor runs partial-offload (-ngl 24) at 1.72 t/s -- the full 164
# would take ~6 h and starve the actual experiment. We take it to GATE_N tasks
# (per the spec's "25 tasks is enough for the gate if time-pressed"), stop it,
# and immediately run the three sub-4-bit builds fully offloaded, which is the
# measurement the arm exists for. Q8 is resumed afterwards from its own JSONL.
set -uo pipefail
W=/data/projects/q27b_on_12gb/platform_arm
R=/data/projects/q27b_on_12gb/results/platform_arm
GATE_N=${GATE_N:-25}
log(){ echo "[chain $(date -u +%H:%M:%S)] $*" | tee -a "$W/logs/chain.log"; }

gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
solved(){ python3 -c "
import json,sys
try:
    rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    print(sum(1 for r in rs if r.get('passed_at')==1),'/',len(rs))
except Exception as e: print('?',e)
" "$1"; }

# --- 1. wait for the Q8 gate to reach GATE_N tasks, then stop it -------------
log "waiting for Q8_0 to reach $GATE_N tasks"
for i in $(seq 1 5400); do
  n=$(grep -cE "^\[reliability\] [0-9]+/164 HumanEval" "$W/logs/run_Q8_0.log" 2>/dev/null)
  n=${n:-0}
  [ "$n" -ge "$GATE_N" ] && { log "Q8_0 reached $n tasks"; break; }
  pgrep -f "platform_arm.sh he Q8_0" >/dev/null || { log "Q8_0 run exited early at $n tasks"; break; }
  sleep 10
done
if pgrep -f "platform_arm.sh he Q8_0" >/dev/null; then
  # kill the harness only; platform_arm.sh then runs its own stop_srv and
  # shuts the server down cleanly, leaving the flushed JSONL intact.
  pkill -f "reliability.py --endpoint .*Q8_0_nospec_3060" && log "stopped Q8_0 harness"
  for i in $(seq 1 180); do pgrep -f "platform_arm.sh he Q8_0" >/dev/null || break; sleep 2; done
fi
pkill -f "bin-unpatched/llama-server" 2>/dev/null
for i in $(seq 1 120); do u=$(gpu); [ "${u:-9999}" -lt 500 ] && break; sleep 2; done
log "GPU now $(gpu) MiB; Q8 gate = $(solved "$R/reliability_Q8_0_nospec_3060.jsonl")"

# --- 2. the actual experiment: three sub-4-bit builds, full offload ----------
for B in IQ3_XXS Q2_K_XL IQ2_XXS; do
  log "START $B"
  "$W/platform_arm.sh" he "$B" 99 164 > "$W/logs/run_${B}.log" 2>&1
  rc=$?
  log "END $B rc=$rc solved=$(solved "$R/reliability_${B}_nospec_3060.jsonl")"
done
log "SUB4 PHASE COMPLETE"

# --- 3. resume the Q8 anchor toward the full 164 with whatever time is left --
log "resuming Q8_0 anchor to 164"
"$W/platform_arm.sh" he Q8_0 24 164 > "$W/logs/run_Q8_0_resume.log" 2>&1
log "Q8 anchor final = $(solved "$R/reliability_Q8_0_nospec_3060.jsonl")"
log "CHAIN COMPLETE"

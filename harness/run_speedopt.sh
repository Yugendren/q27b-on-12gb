#!/usr/bin/env bash
# After the curve finishes: measure speed-optimisation levers on the
# best-quality resident build (Q2_K_XL). MTP speculation is the big untested one.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /data/projects/q27b_on_12gb

while [ ! -f results/curve_done.txt ]; do sleep 60; done
echo "[speedopt] starting $(date -u)"

M=models/unsloth-q27b/Qwen3.8-27B-UD-Q2_K_XL.gguf
MTP=models/unsloth-q27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf

# 1. Baseline re-run with flash attention explicitly on + quantised KV
llama-bench -m "$M" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 3 -o json \
  > results/bench_q2kxl_fa_kvq8.json 2> results/bench_q2kxl_fa_kvq8.err

# 2. Batch/ubatch sweep (kernel efficiency probe)
llama-bench -m "$M" -ngl 99 -fa 1 -b 2048 -ub 512 -p 512 -n 128 -r 3 -o json \
  > results/bench_q2kxl_ub512.json 2> results/bench_q2kxl_ub512.err
llama-bench -m "$M" -ngl 99 -fa 1 -b 4096 -ub 1024 -p 512 -n 128 -r 3 -o json \
  > results/bench_q2kxl_ub1024.json 2> results/bench_q2kxl_ub1024.err

# 3. MTP speculative decoding via llama-server (llama-bench can't drive spec).
#    Measure real generation throughput on a fixed coding prompt, n=1,2,3 drafts.
for N in 0 2 3; do
  if [ "$N" = "0" ]; then SPEC=""; TAG="nospec"; else
    SPEC="--spec-type draft-mtp --spec-draft-n-max $N -md $MTP"; TAG="mtp$N"; fi
  echo "[speedopt] === $TAG ==="
  # shellcheck disable=SC2086
  llama-server -m "$M" -ngl 99 -fa on -c 8192 --port 18082 --host 127.0.0.1 \
    --chat-template-file models/templates/chat_template.jinja $SPEC \
    > "results/spec_server_${TAG}.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 300); do
    curl -sf http://127.0.0.1:18082/health >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || break
    sleep 1
  done
  if ! kill -0 $SRV 2>/dev/null; then
    echo "[speedopt] $TAG server failed to start:"; tail -15 "results/spec_server_${TAG}.log"; continue
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/[speedopt] $TAG VRAM: /"
  for rep in 1 2; do
    curl -sf http://127.0.0.1:18082/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a complete Python class implementing a thread-safe LRU cache with get, put, and stats methods. Include docstrings."}],"temperature":0,"max_tokens":600}' \
      > /dev/null 2>&1
  done
  grep "eval time" "results/spec_server_${TAG}.log" | tail -2 | sed "s/^/[speedopt] $TAG /"
  grep -iE "draft acceptance|accept" "results/spec_server_${TAG}.log" | tail -2 | sed "s/^/[speedopt] $TAG /"
  kill $SRV 2>/dev/null; sleep 5
done

echo SPEEDOPTDONE > results/speedopt_done.txt
echo "[speedopt] done $(date -u)"

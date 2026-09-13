#!/usr/bin/env bash
# CORRECTED speculation tests. Prior run failed because it used the EXTERNAL
# MTP file (+776 MiB for nothing). The built-in blk.64.nextn head is already
# inside every quant >= UD-Q2_K_XL and costs only ~668 MiB.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /data/projects/q27b_on_12gb

M="${MODEL:-models/unsloth-q27b/Qwen3.8-27B-UD-Q2_K_XL.gguf}"
PORT=18083
PROMPT='{"messages":[{"role":"user","content":"Refactor this Python function to be thread-safe and add type hints:\n\ndef add_item(cache, key, value):\n    if key in cache:\n        cache[key] += value\n    else:\n        cache[key] = value\n    return cache\n\nGive the full rewritten function."}],"temperature":0,"max_tokens":500}'

try () {
  local tag="$1"; shift
  echo "[spec2] ===== $tag ====="
  llama-server -m "$M" -ngl 99 -fa on -c 16384 --port $PORT --host 127.0.0.1 \
    --chat-template-file models/templates/chat_template.jinja "$@" \
    > "results/spec2_${tag}.log" 2>&1 &
  local SRV=$!
  local ok=0
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 1
  done
  if [ $ok -eq 0 ]; then
    echo "[spec2] $tag FAILED:"; grep -iE "error|oom|out of memory|unknown|invalid" "results/spec2_${tag}.log" | head -4
    kill $SRV 2>/dev/null; sleep 3; return
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/[spec2] $tag VRAM /"
  for r in 1 2 3; do
    curl -sf http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$PROMPT" >/dev/null 2>&1
  done
  grep -E "eval time =" "results/spec2_${tag}.log" | grep -v prompt | tail -3 | sed "s/^.*| */[spec2] $tag /"
  grep -iE "accept|draft" "results/spec2_${tag}.log" | tail -3 | sed "s/^.*| */[spec2] $tag /"
  kill $SRV 2>/dev/null; sleep 5
}

# what speculation types does this build actually expose?
echo "[spec2] available spec types:"; llama-server --help 2>&1 | grep -A6 -iE "spec-type|speculative" | head -20

try baseline
try mtp_builtin_n2 --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
try mtp_builtin_n3 --spec-type draft-mtp --spec-draft-n-max 3 --parallel 1
try mtp_builtin_n4 --spec-type draft-mtp --spec-draft-n-max 4 --parallel 1
try ngram          --spec-type ngram-mod --spec-draft-n-max 4 --parallel 1
try mtp_kvq4  --spec-type draft-mtp --spec-draft-n-max 3 --parallel 1 -ctk q4_0 -ctv q4_0

echo SPEC2DONE > results/spec2_done.txt
echo "[spec2] done $(date -u)"

#!/usr/bin/env bash
# Waits for the GPU to be free, starts llama-server for a model, runs the
# best-of-n verified-selection HumanEval eval (harness/bestofn.py) against
# it, saves logs + resumable results JSON, then stops the server.
#
# Usage:
#   bash harness/run_bestofn.sh MODEL TAG [K] [LIMIT] [TEMP] [TOP_P] [MAX_TOKENS] [EXTRA_SERVER_FLAGS]
#
# Example:
#   bash harness/run_bestofn.sh models/unsloth-q27b/Qwen3.8-27B-UD-IQ2_S.gguf \
#       iq2s 5 164 0.8 0.95 768 "-ngl 99 -fa on"
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

MODEL="$1"; TAG="$2"
K="${3:-5}"; LIMIT="${4:-164}"; TEMP="${5:-0.8}"; TOP_P="${6:-0.95}"
MAX_TOKENS="${7:-768}"; EXTRA="${8:-}"
PORT=18091

echo "[bestofn] $TAG :: $MODEL k=$K limit=$LIMIT temp=$TEMP top_p=$TOP_P max_tokens=$MAX_TOKENS extra='$EXTRA'"

echo "[bestofn] waiting for GPU to be free (<500 MiB used)..."
for i in $(seq 1 900); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ -n "$used" ] && [ "$used" -lt 500 ]; then
    echo "[bestofn] GPU free (${used} MiB used) after ${i}s"
    break
  fi
  if [ "$i" -eq 900 ]; then
    echo "[bestofn] WARNING: GPU still busy (${used:-?} MiB used) after 900s, proceeding anyway"
  fi
  sleep 1
done

# shellcheck disable=SC2086
llama-server -m "$MODEL" -c 8192 --port $PORT --host 127.0.0.1 \
  --chat-template-file models/templates/chat_template.jinja \
  $EXTRA > "results/server_bestofn_${TAG}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "[bestofn] up in ${i}s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "[bestofn] FATAL server died"; tail -20 "results/server_bestofn_${TAG}.log"; exit 1; }
  sleep 1
done

nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/[bestofn] VRAM: /'

python3 harness/bestofn.py --endpoint "http://127.0.0.1:$PORT" \
  --k "$K" --limit "$LIMIT" --temp "$TEMP" --top-p "$TOP_P" --max-tokens "$MAX_TOKENS" \
  --out "results/bestofn_${TAG}.json" \
  > "results/bestofn_${TAG}.log" 2> "results/bestofn_${TAG}.err"
RC=$?
echo "[bestofn] exit=$RC"
tail -5 "results/bestofn_${TAG}.log"
tail -5 "results/bestofn_${TAG}.err"

python3 -c "
import json
try:
    d = json.load(open('results/bestofn_${TAG}.json'))
    a = d.get('aggregate', {})
    print('pass@1               :', a.get('pass_at_1'))
    print('pass@k               :', a.get('pass_at_k'))
    print('verified-selection   :', a.get('verified_selection_score'))
    print('frac visible-usable  :', a.get('frac_tasks_with_visible_tests'))
    print('wall_time_s          :', a.get('wall_time_s'))
except Exception as e:
    print('parse fail:', e)
"

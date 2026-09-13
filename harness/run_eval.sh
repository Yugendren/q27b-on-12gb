#!/usr/bin/env bash
# Start llama-server for a model, run the quality suite against it, stop server.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

MODEL="$1"; TAG="$2"; NGSM="${3:-50}"; NHE="${4:-20}"; EXTRA="${5:-}"
PORT=18081

echo "[eval] $TAG :: $MODEL (gsm8k=$NGSM humaneval=$NHE) extra='$EXTRA'"
# shellcheck disable=SC2086
llama-server -m "$MODEL" -c 8192 --port $PORT --host 127.0.0.1 \
  --chat-template-file models/templates/chat_template.jinja \
  $EXTRA > "results/server_${TAG}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "[eval] up in ${i}s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "[eval] FATAL server died"; tail -20 "results/server_${TAG}.log"; exit 1; }
  sleep 1
done

nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/[eval] VRAM: /'

python3 harness/eval_quality.py --endpoint "http://127.0.0.1:$PORT" \
  --limit-gsm8k "$NGSM" --limit-humaneval "$NHE" > "results/quality_${TAG}.json" 2>"results/quality_${TAG}.err"
RC=$?
echo "[eval] exit=$RC"
tail -5 "results/quality_${TAG}.err"
python3 -c "
import json,sys
try:
    d=json.load(open('results/quality_${TAG}.json'))
    print('GSM8K :', d.get('gsm8k',{}).get('score'), '  HumanEval:', d.get('humaneval',{}).get('score'))
    print('wall  :', round(d.get('wall_time_s',0),1),'s')
except Exception as e:
    print('parse fail:',e)
"

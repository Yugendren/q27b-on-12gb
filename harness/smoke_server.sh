#!/usr/bin/env bash
# M0 correctness smoke test: start llama-server, ask 3 factual/reasoning
# questions, print answers. Detects the stale-CUDA "coherent gibberish" bug.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:-models/unsloth-q27b/Qwen3.8-27B-UD-Q2_K_XL.gguf}"
NGL="${2:-99}"
PORT="${3:-18080}"
EXTRA="${4:-}"

echo "[smoke] model=$MODEL ngl=$NGL port=$PORT extra=$EXTRA"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# shellcheck disable=SC2086
llama-server -m "$MODEL" -ngl "$NGL" -c 4096 --port "$PORT" --host 127.0.0.1 \
  --chat-template-file models/templates/chat_template.jinja \
  $EXTRA > "results/smoke_server_$(basename "$MODEL" .gguf).log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

echo "[smoke] waiting for health..."
for i in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[smoke] server up after ${i}s"; break
  fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "[smoke] FATAL: server died. Tail:"
    tail -25 "results/smoke_server_$(basename "$MODEL" .gguf).log"
    exit 1
  fi
  sleep 1
done

echo "[smoke] VRAM after load:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

ask() {
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$1}],\"temperature\":0,\"max_tokens\":300}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' 2>&1
}

echo "=============== Q1 factual ==============="
ask '"What is the capital of France, and which river runs through it? Answer in one sentence."'
echo "=============== Q2 arithmetic ==============="
ask '"A shop sells pens at 3 for $5. How much do 12 pens cost? Show your working briefly."'
echo "=============== Q3 code ==============="
ask '"Write a Python function that returns the nth Fibonacci number iteratively. Code only."'
echo "=============== done ==============="

#!/usr/bin/env bash
# Waits for the stage-1/2 queue to finish, then measures the full
# quality-vs-size curve for every build that fits resident on 12 GB.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /data/projects/q27b_on_12gb

echo "[curve] waiting for prior queue to finish..."
while [ ! -f results/queue_done.txt ]; do sleep 60; done
echo "[curve] prior queue done. waiting for downloads..."
while [ ! -f results/dl_curve_done.txt ]; do sleep 60; done
echo "[curve] starting at $(date -u)"

M=models/unsloth-q27b

# name : file : extra server flags
run_one () {
  local tag="$1" file="$2" flags="$3"
  if [ ! -f "$file" ]; then echo "[curve] SKIP $tag (missing $file)"; return; fi
  echo "[curve] ===== $tag ====="
  # speed benchmark first (cheap, GPU exclusive)
  llama-bench -m "$file" -ngl 99 -p 512 -n 128 -r 3 -o json \
    > "results/bench_${tag}.json" 2> "results/bench_${tag}.err" || echo "[curve] bench failed $tag"
  # quality suite
  bash harness/run_eval.sh "$file" "$tag" 50 20 "$flags" \
    > "results/evalrun_${tag}.log" 2>&1 || echo "[curve] eval failed $tag"
  echo "[curve] $tag done at $(date -u)"
}

run_one iq2xxs "$M/Qwen3.8-27B-UD-IQ2_XXS.gguf" "-ngl 99"
run_one iq2s   "$M/Qwen3.8-27B-UD-IQ2_S.gguf"   "-ngl 99"
run_one iq3xxs "$M/Qwen3.8-27B-UD-IQ3_XXS.gguf" "-ngl 99 -ctk q4_0 -ctv q4_0 -fa on"

echo CURVEDONE > results/curve_done.txt
echo "[curve] ALL DONE $(date -u)"

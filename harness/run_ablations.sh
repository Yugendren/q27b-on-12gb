#!/usr/bin/env bash
# Comprehensive unattended ablation suite. Runs for many hours on the 3060.
# Costs zero API credits. Appends one line per result to results/ablations.tsv
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /data/projects/q27b_on_12gb
M=models/unsloth-q27b
OUT=results/ablations.tsv
[ -f "$OUT" ] || echo -e "exp\tbuild\tconfig\tctx\tvram_mib\tprefill_ts\tdecode_ts\taccept\tnote" > "$OUT"

log() { echo "[abl $(date -u +%H:%M:%S)] $*"; }

# --- generic server benchmark: returns prefill/decode/vram/acceptance ---
bench_server () {
  local exp="$1" build="$2" cfgname="$3" ctx="$4" model="$5"; shift 5
  local port=18090 tag="${exp}_${cfgname}_${ctx}"
  [ -f "$model" ] || { log "SKIP $tag (missing $model)"; return; }
  llama-server -m "$model" -ngl 99 -fa on -c "$ctx" --port $port --host 127.0.0.1 \
    --chat-template-file models/templates/chat_template.jinja "$@" \
    > "results/abl_${tag}.log" 2>&1 &
  local srv=$! ok=0
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $srv 2>/dev/null || break; sleep 1
  done
  if [ $ok -eq 0 ]; then
    local why; why=$(grep -ioE "out of memory|unknown argument[^ ]*|failed to load|unsupported" "results/abl_${tag}.log" | head -1)
    echo -e "$exp\t$build\t$cfgname\t$ctx\t-\t-\t-\t-\tFAIL:${why:-unknown}" >> "$OUT"
    log "FAIL $tag ${why:-unknown}"; kill $srv 2>/dev/null; sleep 4; return
  fi
  local vram; vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  # warm + 3 timed generations
  for r in 1 2 3; do
    curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Implement a thread-safe LRU cache in Python with get/put/stats, full type hints and docstrings."}],"temperature":0,"max_tokens":400}' >/dev/null 2>&1
  done
  local dec pre acc
  dec=$(grep -E "eval time =" "results/abl_${tag}.log" | grep -v prompt | tail -1 | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
  pre=$(grep -E "prompt eval time =" "results/abl_${tag}.log" | tail -1 | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
  acc=$(grep -oE "draft acceptance = [0-9.]+" "results/abl_${tag}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo -e "$exp\t$build\t$cfgname\t$ctx\t${vram:--}\t${pre:--}\t${dec:--}\t${acc:--}\t" >> "$OUT"
  log "OK $tag decode=${dec:-?} vram=${vram:-?}"
  kill $srv 2>/dev/null; sleep 5
}

############ EXP 1: DFlash2 drafter vs built-in MTP ############
log "=== EXP1 drafter comparison ==="
DF=$(ls models/dflash2/*.gguf 2>/dev/null | grep -iE "q4_k_m|q4" | head -1)
for build in Qwen3.8-27B-UD-Q2_K_XL Qwen3.8-27B-UD-IQ3_XXS; do
  MODEL="$M/${build}.gguf"
  bench_server exp1 "$build" nospec 16384 "$MODEL" -ctk q8_0 -ctv q8_0
  for n in 2 3 4; do
    bench_server exp1 "$build" "mtp_n$n" 16384 "$MODEL" -ctk q8_0 -ctv q8_0 \
      --spec-type draft-mtp --spec-draft-n-max $n --parallel 1
  done
  if [ -n "$DF" ]; then
    for n in 3 4 5; do
      bench_server exp1 "$build" "dflash_n$n" 16384 "$MODEL" -ctk q8_0 -ctv q8_0 \
        --spec-type draft-dflash -md "$DF" --spec-draft-n-max $n --parallel 1
    done
  else
    log "DFlash2 model not found, skipping"
  fi
  bench_server exp1 "$build" ngram 16384 "$MODEL" -ctk q8_0 -ctv q8_0 \
    --spec-type ngram-mod --spec-draft-n-max 4 --parallel 1
done

############ EXP 2: context scaling — verify the reported decode cliff ############
log "=== EXP2 context scaling (llama.cpp #27623 decode cliff) ==="
for ctx in 8192 16384 32768 49152 65536 81920 98304; do
  bench_server exp2 Qwen3.8-27B-UD-Q2_K_XL "kvq4" "$ctx" "$M/Qwen3.8-27B-UD-Q2_K_XL.gguf" \
    -ctk q4_0 -ctv q4_0
done

############ EXP 3: KV precision sweep ############
log "=== EXP3 KV precision ==="
for kv in "f16:-ctk f16 -ctv f16" "q8:-ctk q8_0 -ctv q8_0" "q4:-ctk q4_0 -ctv q4_0"; do
  name=${kv%%:*}; flags=${kv#*:}
  # shellcheck disable=SC2086
  bench_server exp3 Qwen3.8-27B-UD-IQ3_XXS "kv_$name" 16384 "$M/Qwen3.8-27B-UD-IQ3_XXS.gguf" $flags
done

############ EXP 4: Minitron (structurally pruned 20B) head-to-head ############
log "=== EXP4 Minitron ==="
MIN=$(ls models/minitron/*.gguf 2>/dev/null | head -1)
if [ -n "$MIN" ]; then
  bench_server exp4 Minitron-20B nospec 16384 "$MIN" -ctk q8_0 -ctv q8_0
  bench_server exp4 Minitron-20B mtp_n3 16384 "$MIN" -ctk q8_0 -ctv q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --parallel 1
else
  log "Minitron not downloaded, skipping"
fi

############ EXP 5: quality of the new builds (long, runs last) ############
log "=== EXP5 quality suites ==="
[ -n "$MIN" ] && bash harness/run_eval.sh "$MIN" minitron20b 50 20 "-ngl 99 -ctk q8_0 -ctv q8_0 -fa on" > results/evalrun_minitron.log 2>&1
bash harness/run_eval.sh "$M/Qwen3.8-27B-UD-IQ2_S.gguf" iq2s_recheck 50 20 "-ngl 99 -ctk q8_0 -ctv q8_0 -fa on" > results/evalrun_iq2s_recheck.log 2>&1

echo ABLDONE > results/ablations_done.txt
log "ALL ABLATIONS COMPLETE"

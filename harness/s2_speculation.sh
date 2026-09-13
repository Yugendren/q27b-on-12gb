#!/usr/bin/env bash
# S2 — the speculation bake-off + correctness bundle (07-SYNTHESIS.md S2,
# ~1 GPU-day). One harness for MTP vs DFlash2 vs ngram-mod vs the
# comma-stack combos, a CPU-draft-device probe, and an adversarial
# losslessness gate, all on build 10712 against the flagship config
# (UD-IQ3_XXS, q8_0 KV, 16K ctx, temp 0).
#
# CAVEATS (read before trusting the lossless gate):
#  - The losslessness gate below does byte-identity on greedy (temp=0)
#    output, per the spec that requested this script. 07-SYNTHESIS.md G9
#    warns this is an insufficient MTP-losslessness gate on its own — bf16
#    non-associativity can cause *legitimate* divergence at temp 0, and the
#    real gate they recommend is full HumanEval-164 spec-ON vs spec-OFF +
#    monotonicity across depths (~4 GPU-h). Treat a FAIL here as "needs the
#    full HumanEval gate to adjudicate," not as proof of a bug; treat a
#    PASS as "necessary, not sufficient."
#  - Comma-separated --spec-type stacking (e.g. "draft-mtp,ngram-mod") is
#    UNVERIFIED for build 10712 — 07-SYNTHESIS.md cites it as a documented
#    combo but we have not confirmed this exact build parses it. The
#    bench_server function degrades gracefully (logs FAIL + verbatim
#    reason) if the flag is rejected.
#  - The CPU-draft device flag name is UNVERIFIED — llama-server --help is
#    grepped at runtime for one of -devd / --device-draft /
#    --spec-draft-device, the one found (or NONE_FOUND) is logged to
#    results/s2_draft_device_flag.txt, and the CPU-draft test is skipped
#    with a note if none match.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /data/projects/q27b_on_12gb || exit 1

SRV=/data/projects/llama.cpp/build/bin/llama-server   # build 10712, FA_ALL_QUANTS
M=models/unsloth-q27b/Qwen3.8-27B-UD-IQ3_XXS.gguf
BUILD=Qwen3.8-27B-UD-IQ3_XXS
CTX=16384
PORT=18095
OUT=results/s2_speculation.tsv
mkdir -p results
[ -f "$OUT" ] || printf 'exp\tbuild\tconfig\tctx\tvram_mib\tprefill_ts\tdecode_ts\taccept\tnote\n' > "$OUT"

DF="models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
if [ ! -f "$DF" ]; then
  ALT=$(ls models/dflash2/*.gguf 2>/dev/null | grep -iE "q4_k_m|q4" | head -1)
  if [ -n "$ALT" ]; then
    echo "[s2] WARNING: expected DFlash2 file not found at $DF, falling back to $ALT"
    DF="$ALT"
  fi
fi

PROMPT_JSON='{"messages":[{"role":"user","content":"Implement a thread-safe LRU cache in Python with get/put/stats, full type hints and docstrings. Include a small __main__ demonstration."}],"temperature":0,"max_tokens":500}'

log() { echo "[s2 $(date -u +%H:%M:%S)] $*"; }

sanitize() { printf '%s' "$1" | tr '\t\n\r' '   '; }

have_bin() {
  local b="$1"
  if [[ "$b" == */* ]]; then
    [ -x "$b" ]
  else
    command -v "$b" >/dev/null 2>&1
  fi
}

row_exists() {
  local exp="$1" build="$2" cfg="$3" ctx="$4"
  [ -f "$OUT" ] || return 1
  awk -F'\t' -v e="$exp" -v b="$build" -v c="$cfg" -v x="$ctx" \
    '$1==e && $2==b && $3==c && $4==x {found=1} END{exit !found}' "$OUT"
}

append_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" >> "$OUT"
}

# --- GPU-guard: block until VRAM is free (<500 MiB used), before every GPU op ---
wait_gpu_free() {
  local tag="${1:-gpu}"
  log "[$tag] waiting for GPU to be free (<500 MiB used)..."
  local i used
  for i in $(seq 1 900); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 500 ]; then
      log "[$tag] GPU free (${used} MiB used) after ${i}s"
      return 0
    fi
    if [ "$i" -eq 900 ]; then
      log "[$tag] WARNING: GPU still busy (${used:-?} MiB used) after 900s, proceeding anyway"
    fi
    sleep 1
  done
}

############################################################################
# generic server benchmark: starts llama-server with the given extra flags,
# fires the fixed coding prompt 3x, records peak VRAM / prefill / decode /
# acceptance, appends one TSV row. Any load failure is captured verbatim.
############################################################################
bench_server() {
  local exp="$1" cfgname="$2"; shift 2
  local tag="${exp}_${cfgname}"
  if row_exists "$exp" "$BUILD" "$cfgname" "$CTX"; then
    log "SKIP $tag (already have)"; return
  fi
  if [ ! -f "$M" ]; then
    append_row "$exp" "$BUILD" "$cfgname" "$CTX" - - - - "FAIL:model_missing:$M"
    log "FAIL $tag model missing"; return
  fi
  if ! have_bin "$SRV"; then
    append_row "$exp" "$BUILD" "$cfgname" "$CTX" - - - - "FAIL:binary_missing:$SRV"
    log "FAIL $tag binary missing"; return
  fi
  wait_gpu_free "$tag"
  "$SRV" -m "$M" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c "$CTX" --port "$PORT" --host 127.0.0.1 \
    --chat-template-file models/templates/chat_template.jinja "$@" \
    > "results/s2_${tag}.log" 2>&1 &
  local srv_pid=$! ok=0
  local i
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$srv_pid" 2>/dev/null || break
    sleep 1
  done
  if [ "$ok" -eq 0 ]; then
    local why
    why=$(grep -ioE "out of memory|unknown argument[^ ]*|failed to load|unsupported|error[^,]*" \
      "results/s2_${tag}.log" | head -1)
    local note
    note=$(sanitize "FAIL:${why:-unknown}:$(tail -c 300 "results/s2_${tag}.log")")
    append_row "$exp" "$BUILD" "$cfgname" "$CTX" - - - - "$note"
    log "FAIL $tag ${why:-unknown}"
    kill "$srv_pid" 2>/dev/null; sleep 4; return
  fi
  # background peak-VRAM monitor for the life of the benchmark reps
  : > "results/s2_${tag}.vram"
  ( while kill -0 "$srv_pid" 2>/dev/null; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [[ "$u" =~ ^[0-9]+$ ]] && echo "$u"
      sleep 1
    done ) > "results/s2_${tag}.vram" &
  local mon_pid=$!
  local r
  for r in 1 2 3; do
    curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "$PROMPT_JSON" >/dev/null 2>&1
  done
  kill "$mon_pid" 2>/dev/null; wait "$mon_pid" 2>/dev/null
  local vram
  vram=$(sort -n "results/s2_${tag}.vram" 2>/dev/null | tail -1)
  local dec pre acc
  dec=$(grep -E "eval time =" "results/s2_${tag}.log" | grep -v prompt | tail -1 \
    | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
  pre=$(grep -E "prompt eval time =" "results/s2_${tag}.log" | tail -1 \
    | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
  acc=$(grep -ioE "(draft acceptance|acceptance rate)[ =:]+[0-9.]+%?" "results/s2_${tag}.log" \
    | tail -1 | grep -oE "[0-9.]+")
  append_row "$exp" "$BUILD" "$cfgname" "$CTX" "${vram:--}" "${pre:--}" "${dec:--}" "${acc:--}" ""
  log "OK $tag decode=${dec:-?} accept=${acc:-?} vram=${vram:-?}"
  kill "$srv_pid" 2>/dev/null; sleep 5
}

############################################################################
# FLAGSHIP BAKE-OFF
############################################################################
log "=== flagship spec bake-off: $BUILD, q8 KV, ctx=$CTX ==="
bench_server exp_s2 baseline
bench_server exp_s2 mtp_n2 --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
bench_server exp_s2 mtp_n3 --spec-type draft-mtp --spec-draft-n-max 3 --parallel 1

if [ -f "$DF" ]; then
  for n in 4 7; do
    bench_server exp_s2 "dflash_n$n" --spec-type draft-dflash -md "$DF" --spec-draft-n-max "$n" --parallel 1
  done
else
  log "DFlash2 model not found (looked for $DF), skipping dflash configs"
fi

bench_server exp_s2 ngram --spec-type ngram-mod --spec-draft-n-max 4 --parallel 1

# UNVERIFIED: comma-separated --spec-type stacking on build 10712.
bench_server exp_s2 mtp_n2_ngram --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --parallel 1
if [ -f "$DF" ]; then
  bench_server exp_s2 dflash_n7_ngram --spec-type draft-dflash,ngram-mod -md "$DF" --spec-draft-n-max 7 --parallel 1
fi

############################################################################
# CPU-DRAFT DEVICE TEST
############################################################################
log "=== detecting draft-device flag ==="
HELP_TXT=$("$SRV" --help 2>&1 || true)
DEVICE_FLAG=""
for cand in "--spec-draft-device" "-devd" "--device-draft"; do
  if printf '%s' "$HELP_TXT" | grep -qF -- "$cand"; then
    DEVICE_FLAG="$cand"
    break
  fi
done
if [ -n "$DEVICE_FLAG" ]; then
  log "draft-device flag detected: $DEVICE_FLAG (UNVERIFIED value syntax, guessing CPU)"
  echo "$DEVICE_FLAG" > results/s2_draft_device_flag.txt
  if [ -f "$DF" ]; then
    bench_server exp_s2 dflash_n7_cpu_draft --spec-type draft-dflash -md "$DF" \
      --spec-draft-n-max 7 --parallel 1 "$DEVICE_FLAG" CPU
  else
    log "DFlash2 model not found, skipping CPU-draft test"
  fi
else
  log "WARNING: no known draft-device flag found in --help output; UNVERIFIED, skipping CPU-draft test"
  echo "NONE_FOUND" > results/s2_draft_device_flag.txt
  append_row exp_s2 "$BUILD" dflash_n7_cpu_draft "$CTX" - - - - "SKIP:no_draft_device_flag_detected_in_help"
fi

log "=== CPU-draft VRAM/speed delta (vs GPU-draft dflash_n7) ==="
python3 - "$OUT" <<'PYEOF'
import csv, sys
rows = {}
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        rows[(row["exp"], row["config"])] = row
a = rows.get(("exp_s2", "dflash_n7"))
b = rows.get(("exp_s2", "dflash_n7_cpu_draft"))
if a and b:
    try:
        dv = int(a["vram_mib"]) - int(b["vram_mib"])
        print(f"[s2] VRAM saved by CPU draft: {dv} MiB")
    except (ValueError, KeyError):
        print("[s2] vram delta: n/a (missing numeric data)")
    try:
        dd = float(b["decode_ts"]) - float(a["decode_ts"])
        print(f"[s2] decode t/s delta (cpu-draft - gpu-draft): {dd:+.2f}")
    except (ValueError, KeyError):
        print("[s2] decode delta: n/a (missing numeric data)")
else:
    print("[s2] cannot compute delta: dflash_n7 and/or dflash_n7_cpu_draft rows missing")
PYEOF

############################################################################
# LOSSLESSNESS GATE — 10 diverse prompts x {no-spec, mtp, dflash}, greedy
# (temp=0), byte-identity vs the no-spec baseline. See caveat block at top
# re: G9 / bf16 non-associativity.
############################################################################
log "=== losslessness gate: 10 prompts x {no-spec, mtp, dflash} ==="
LOSSLESS_OUT=results/s2_lossless.txt

PROMPTS=(
  "Write a Python function to compute the nth Fibonacci number iteratively."
  "Explain the CAP theorem in three sentences."
  "Write a haiku about a GPU running out of memory."
  "Implement binary search in C, iterative, with bounds checking."
  "List the first 8 prime numbers."
  "Write a SQL query that finds the second-highest salary per department."
  "Summarize the plot of a heist movie in two sentences."
  "Write a regex that matches a valid IPv4 address."
  "Explain the difference between TCP and UDP in one paragraph."
  "Write a bash one-liner that finds the 5 largest files under the current directory."
)

capture_outputs() {
  local tag="$1"; shift
  local dir="results/s2_lossless_${tag}"
  mkdir -p "$dir"
  if [ -f "$dir/.complete" ]; then
    log "SKIP capture $tag (already complete)"
    return 0
  fi
  if ! have_bin "$SRV"; then
    log "SKIP capture $tag (missing $SRV)"; return 1
  fi
  if [ ! -f "$M" ]; then
    log "SKIP capture $tag (missing model $M)"; return 1
  fi
  wait_gpu_free "lossless_$tag"
  "$SRV" -m "$M" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -c "$CTX" --port "$PORT" --host 127.0.0.1 \
    --chat-template-file models/templates/chat_template.jinja "$@" \
    > "results/s2_lossless_${tag}.log" 2>&1 &
  local srv_pid=$! ok=0
  local i
  for i in $(seq 1 400); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$srv_pid" 2>/dev/null || break
    sleep 1
  done
  if [ "$ok" -eq 0 ]; then
    log "FAIL capture $tag: server did not start"
    grep -ioE "error|out of memory|unknown|invalid" "results/s2_lossless_${tag}.log" | head -3
    kill "$srv_pid" 2>/dev/null; sleep 4
    return 1
  fi
  local idx=0 p outfile
  for p in "${PROMPTS[@]}"; do
    idx=$((idx + 1))
    outfile="$dir/p${idx}.txt"
    if [ -f "$outfile" ]; then continue; fi
    python3 - "$PORT" "$p" "$outfile" <<'PYEOF'
import json, sys, urllib.request
port, prompt, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": 300}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                              headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    text = d["choices"][0]["message"]["content"]
except Exception as e:
    text = f"__REQUEST_FAILED__:{e}"
open(outfile, "w").write(text)
PYEOF
  done
  touch "$dir/.complete"
  kill "$srv_pid" 2>/dev/null; sleep 5
  return 0
}

capture_outputs nospec
capture_outputs mtp --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
if [ -f "$DF" ]; then
  capture_outputs dflash --spec-type draft-dflash -md "$DF" --spec-draft-n-max 7 --parallel 1
else
  log "DFlash2 model missing, skipping dflash lossless capture"
fi

log "=== comparing outputs byte-for-byte ==="
: > "$LOSSLESS_OUT"   # recomputed fresh each run (cheap, no GPU) so reruns don't accumulate stale rows
{
  echo "# losslessness gate: greedy (temp=0) byte-identity check"
  echo "# generated $(date -u)"
  echo "# NOTE: byte-identity at temp=0 is necessary but not sufficient (see"
  echo "# G9 caveat in this script's header) -- a FAIL here should be"
  echo "# followed up with the full HumanEval-164 spec-ON-vs-OFF gate."
  n_total=0
  n_pass=0
  for idx in $(seq 1 "${#PROMPTS[@]}"); do
    base="results/s2_lossless_nospec/p${idx}.txt"
    for cfg in mtp dflash; do
      cand="results/s2_lossless_${cfg}/p${idx}.txt"
      if [ ! -f "$base" ] || [ ! -f "$cand" ]; then
        echo "p${idx} nospec_vs_${cfg} SKIP (missing output)"
        continue
      fi
      n_total=$((n_total + 1))
      if cmp -s "$base" "$cand"; then
        echo "p${idx} nospec_vs_${cfg} PASS"
        n_pass=$((n_pass + 1))
      else
        echo "p${idx} nospec_vs_${cfg} FAIL"
      fi
    done
  done
  echo "# summary: ${n_pass}/${n_total} pairs byte-identical"
} >> "$LOSSLESS_OUT"
cat "$LOSSLESS_OUT"

echo S2DONE > results/s2_speculation_done.txt
log "ALL S2 SPECULATION MEASUREMENTS COMPLETE"

#!/usr/bin/env bash
# MAINSTREAM-RIG TEST -- does Qwen3.6-35B-A3B still run at speed on a box with
# only 32 GiB of system RAM?
#
# Our measurement box has 47 GiB, which is NOT the mainstream configuration the
# v2 flagship question is about (12 GiB VRAM + 32 GiB RAM). So we cap the server
# into a cgroup v2 scope with MemoryMax=<CAP> and MemorySwapMax=0 and re-measure
# the shipping cell. Swap is zeroed deliberately: this box has a 96 GiB swapfile
# that would otherwise silently absorb the overflow and turn a "does not fit"
# into a "fits, but crawls" without telling us which.
#
# The model is mmap'd, so its 21.3 GiB of file pages are reclaimable page cache
# charged to the cgroup. Under a tight cap the kernel evicts them and re-reads
# from disk on the next token -- which is exactly the degradation a 32 GiB owner
# would feel. No prewarm here, on purpose: we want the honest cold behaviour.
#
# usage: memcap_test.sh <CAP e.g. 32G> <NCM> <SPEC none|mtp2> [TAG]
set -uo pipefail

CAP="${1:?cap e.g. 32G}"
NCM="${2:-26}"
SPEC="${3:-mtp2}"
TAG="${4:-memcap_${CAP}_ncm${NCM}_${SPEC}}"

W=/data/projects/q27b_on_12gb
SRV=$W/llama-server-v11/llama-server
MODEL=/data/models/coding/q36_35ba3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
R=$W/results/cert35b
PORT=18094
mkdir -p "$R"

case "$SPEC" in
  none) SPECFLAGS=(--spec-type none) ;;
  mtp2) SPECFLAGS=(--spec-type draft-mtp --spec-draft-n-max 2) ;;
  mtp4) SPECFLAGS=(--spec-type draft-mtp --spec-draft-n-max 4) ;;
  *) echo "bad spec $SPEC"; exit 2 ;;
esac

log(){ echo "[memcap $(date -u +%H:%M:%S)] $*"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# GPU guard
for i in $(seq 1 120); do u=$(gpu_used); [ "${u:-9999}" -lt 500 ] && break; sleep 2; done
u=$(gpu_used); [ "${u:-9999}" -lt 500 ] || { log "GPU GUARD FAILED (${u} MiB)"; exit 1; }
log "GPU guard OK (${u} MiB)"

# Drop the page cache so the cap is tested from a cold start rather than
# inheriting 21 GiB of already-resident model pages from a previous cell.
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && log "page cache dropped" || log "WARN: could not drop caches (no sudo) -- results may be optimistic"

UNIT="memcap$$"
log "launching server under MemoryMax=$CAP MemorySwapMax=0, ncm=$NCM spec=$SPEC"
systemd-run --user --scope --unit="$UNIT" \
  -p MemoryMax="$CAP" -p MemorySwapMax=0 -p MemoryAccounting=yes \
  "$SRV" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port $PORT --jinja --no-warmup \
  -c 16384 -np 1 -ctk q8_0 -ctv q8_0 --n-cpu-moe "$NCM" "${SPECFLAGS[@]}" \
  > "$R/srv_${TAG}.log" 2>&1 &
RUNPID=$!

t0=$SECONDS
UP=0
for i in $(seq 1 1800); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { UP=1; break; }
  kill -0 $RUNPID 2>/dev/null || break
  sleep 1
done
LOAD_S=$((SECONDS-t0))

if [ "$UP" != "1" ]; then
  log "SERVER FAILED TO COME UP under $CAP (load ${LOAD_S}s) -- likely OOM-killed"
  log "--- cgroup events ---"; cat "$CG/memory.events" 2>/dev/null
  log "--- server log tail ---"; tail -25 "$R/srv_${TAG}.log"
  dmesg 2>/dev/null | tail -5 | grep -i "oom\|killed" || true
  echo "{\"tag\":\"$TAG\",\"cap\":\"$CAP\",\"status\":\"FAIL_NO_START\",\"load_s\":$LOAD_S}" > "$R/${TAG}.json"
  systemctl --user stop "${UNIT}.scope" 2>/dev/null
  exit 1
fi
log "UP in ${LOAD_S}s, VRAM $(gpu_used) MiB"

# Locate the scope's cgroup by asking the SERVER PROCESS where it lives, rather
# than guessing the systemd path. Then PROVE the cap actually bound -- an
# unverified memory cap makes the whole test worthless.
SPID=$(pgrep -f "llama-serve[r]-v11/llama-server -m $MODEL" | head -1)
CGREL=$(awk -F: '{print $3}' /proc/$SPID/cgroup 2>/dev/null | head -1)
CG="/sys/fs/cgroup${CGREL}"
log "server pid=$SPID cgroup=$CG"
MAXV=$(cat "$CG/memory.max" 2>/dev/null || echo missing)
SWAPV=$(cat "$CG/memory.swap.max" 2>/dev/null || echo missing)
log "CAP VERIFY: memory.max=$MAXV memory.swap.max=$SWAPV (requested $CAP / swap 0)"
if [ "$MAXV" = "missing" ] || [ "$MAXV" = "max" ]; then
  log "*** CAP DID NOT BIND -- refusing to report this cell as a 32G result ***"
  CAPBOUND=false
else
  CAPBOUND=true
fi

python3 /data/scratch/cert35b/smoke_req.py $PORT 2>&1 | tee "$R/${TAG}.reqlog"
RC=$?

log "--- cgroup memory ---"
PEAK=$(cat "$CG/memory.peak" 2>/dev/null || echo -1)
CUR=$(cat "$CG/memory.current" 2>/dev/null || echo -1)
EV=$(cat "$CG/memory.events" 2>/dev/null | tr '\n' ' ')
log "memory.peak=$PEAK memory.current=$CUR"
log "memory.events: $EV"
PSI=$(cat "$CG/memory.pressure" 2>/dev/null | tr '\n' ' ')
log "memory.pressure: $PSI"

python3 - "$R/${TAG}.json" "$TAG" "$CAP" "$NCM" "$SPEC" "$LOAD_S" "$PEAK" "$CUR" "$EV" "$PSI" "$MAXV" "$CAPBOUND" "$R/${TAG}.reqlog" <<'PY'
import json,re,sys
out,tag,cap,ncm,spec,load_s,peak,cur,ev,psi,maxv,bound,reqlog = sys.argv[1:14]
tps=[float(m) for m in re.findall(r"decode=([\d.]+)", open(reqlog).read())]
pf =[float(m) for m in re.findall(r"prefill=([\d.]+)", open(reqlog).read())]
meas=tps[1:] if len(tps)>1 else tps          # drop the warm-up
json.dump({"tag":tag,"cap":cap,"ncm":int(ncm),"spec":spec,
           "status":"OK" if bound=="true" else "CAP_NOT_BOUND",
           "cap_bound":bound=="true","memory_max_bytes":maxv,
           "load_s":int(load_s),"cgroup_memory_peak":int(peak),
           "cgroup_memory_current":int(cur),"memory_events":ev,
           "memory_pressure":psi,
           "decode_tps":meas,"decode_tps_mean":sum(meas)/len(meas) if meas else None,
           "prefill_tps_mean":sum(pf[1:])/len(pf[1:]) if len(pf)>1 else None},
          open(out,"w"), indent=1)
print("wrote",out)
PY

systemctl --user stop "${UNIT}.scope" 2>/dev/null
sleep 5
log "MEMCAP_DONE $TAG"

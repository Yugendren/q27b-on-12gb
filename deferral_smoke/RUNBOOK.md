# RUNBOOK — Expert-Deferral Quality Smoke Test (exp 4.1-smoke)

For the measurement agent. Everything below runs **on the box** (`ssh ollama`).
Read §0 before touching anything. Read `NOTES.md` for why the patch does what it does.

**One-line summary of what you are measuring:** the routed experts of blocks 0-15 (exactly the
tensors `--n-cpu-moe 16` puts on CPU) have their output joined into the residual stream `d` blocks
late. Nothing else changes — no scheduling change, same serial graph, same cost.
Question: at what `d` does quality break?

**What `d` means, precisely:** block `i`'s routed-expert output is **absent from the input of blocks
`i+1 .. i+d`** and present from block `i+d+1` onward. So `d` = the number of blocks of GPU work a
real scheduler would get to overlap the CPU expert compute with. `d=0` is off and is a structurally
identical code path to stock.

---

## 0. PRECONDITIONS AND SAFETY — read first

### 0.1 DO NOT rebuild in the v11 source tree. This is a live hazard.

`/data/projects/revv-home/src/llama.cpp` is the tree that produced `llama-server-v11`
(proven by md5 — see NOTES.md §0). It is tempting to patch it in place for a fast incremental
build. **Do not.** The deployed binary resolves its shared libraries by RUNPATH:

```
$ objdump -x /data/projects/q27b_on_12gb/llama-server-v11/llama-server | grep RUNPATH
  RUNPATH   /data/projects/revv-home/src/llama.cpp/build/bin:
```

The harness scripts invoke that **raw binary**, not the `llama-server-v11` LD_LIBRARY_PATH wrapper.
So every server the *existing campaign* starts from now on loads `libllama.so` out of the v11 build
tree. Rebuilding there would silently put the deferral patch inside unrelated certification runs.
Build in an isolated tree (§1). It costs ~30-45 min of CUDA compile and is worth it.

### 0.2 Wait for the GPU

At the time this runbook was written a campaign was live:

```
$ nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
3347929, 11734 MiB, /data/projects/q27b_on_12gb/llama-server-v11/llama-server
```

`cert_chain4.sh`'s edit leg was mid-run at that moment. **By the end of this dev session the GPU was
free and `cert_chain4` had completed** (`results/cpu_wave1/RESULTS.md` updated, no compute apps).
Re-check anyway — time will have passed, and the harness scripts' built-in GPU guard (waits up to
20 min for VRAM < 500 MiB) is a backstop, not a licence to race:

```bash
ssh ollama 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv; \
            ls -la /data/projects/q27b_on_12gb/results/cpu_wave1/edit_*edit34_ctxcp0_3060.json 2>/dev/null; \
            pgrep -af "cert_chain|cpu_wave1"'
```

Proceed only when there is no compute app and no `cert_chain*` process.

### 0.3 Disk

`/data` was at **98% (19 GB free)**. The isolated build needs ~1.3 GB. That is fine, but check
before you start (`df -h /data`) and do not download the 17 GB model again — reuse the one in place.

### 0.4 One binary serves all four arms

`LLAMA_DEFER_DEPTH` is read once per process via `getenv`. When it is `0` or unset, the patched code
takes a structurally identical path to stock (`moe_out_defer == nullptr`, no extra tensors, no extra
adds). So **build once, and select the arm with an environment variable.** Do not build four times.

---

## 1. REBUILD (isolated tree)

The exact cmake configuration of the v11 build, recovered from its `CMakeCache.txt`:

```
CMAKE_BUILD_TYPE=Release   GGML_CUDA=ON   CMAKE_CUDA_ARCHITECTURES=86
GGML_CUDA_FA=ON            GGML_CUDA_FA_ALL_QUANTS=OFF
GGML_NATIVE=ON             GGML_BLAS=OFF  BUILD_SHARED_LIBS=ON
LLAMA_CURL=OFF             GGML_CCACHE=ON GGML_SCHED_MAX_COPIES=4
```

Reproduce it in a scratch copy:

```bash
ssh ollama
set -e
SRC=/data/scratch/deferral_smoke/llama.cpp
mkdir -p /data/scratch/deferral_smoke

# 1a. copy the v11 source tree (source only; the build dir is NOT copied)
rsync -a --exclude 'build/' --exclude 'build-*/' \
      /data/projects/revv-home/src/llama.cpp/ $SRC/

# 1b. confirm you copied the right thing
cd $SRC && git log --oneline -1        # must print: daef7b687 vulkan: top_k radix select ...
git status --porcelain                  # must list exactly the 4 v11 patch files:
                                        #   M ggml/src/ggml-cuda/mmvq.cu
                                        #   M ggml/src/ggml-cuda/vecdotq.cuh
                                        #   M tools/server/server-context.cpp
                                        #   M tools/server/tests/unit/test_slot_save.py
```

Copy `deferral_sim.patch` up from the dev workspace and apply it:

```bash
# from the Mac:
scp /Users/yugendren/experiments/q27b_on_12gb/deferral_smoke/deferral_sim.patch \
    ollama:/data/scratch/deferral_smoke/

# on the box:
cd /data/scratch/deferral_smoke/llama.cpp
patch -p1 --dry-run < ../deferral_sim.patch     # must report 2 files, no fuzz, no reject
patch -p1          < ../deferral_sim.patch
git status --porcelain                          # now 6 modified files (4 v11 + our 2)
```

The patch touches only `src/models/qwen35moe.cpp` and `src/models/models.h`. It does **not** touch
any of the four v11-patch files, so there is no interaction.

Build:

```bash
cd /data/scratch/deferral_smoke/llama.cpp
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DGGML_NATIVE=ON -DGGML_BLAS=OFF \
  -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF \
  -DGGML_CCACHE=ON -DGGML_SCHED_MAX_COPIES=4 \
  2>&1 | tail -20

nohup cmake --build build -j10 > /data/scratch/deferral_smoke/build.log 2>&1 &
# ~30-45 min (CUDA). Watch: tail -f /data/scratch/deferral_smoke/build.log
# NOTE: -j10 is the full core count. If the campaign is somehow still running, use -j6.
```

Verify the build and that the RUNPATH points at the **new** tree (not the v11 one):

```bash
SRVD=/data/scratch/deferral_smoke/llama.cpp/build/bin/llama-server
ls -la $SRVD
objdump -x $SRVD | grep RUNPATH
# MUST print /data/scratch/deferral_smoke/llama.cpp/build/bin
# If it prints the revv-home path, STOP — you built in the wrong place.
```

Record `$SRVD` — that is the server for every arm below.

---

## 2. SANITY CHECKS (do these before burning 3 hours of eval)

### T1 — d=0 must be identical to stock

```bash
MODEL=/data/models/coding/q36_35ba3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
SRVD=/data/scratch/deferral_smoke/llama.cpp/build/bin/llama-server
SRVV=/data/projects/q27b_on_12gb/llama-server-v11/llama-server

# same prompt, greedy, both binaries, no deferral
for BIN in "$SRVV" "$SRVD"; do
  $BIN -m $MODEL -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 2048 --no-warmup --jinja \
       -np 1 --n-cpu-moe 16 -t 8 --spec-type none --host 127.0.0.1 --port 18099 \
       > /tmp/t1_$(basename $(dirname $(dirname $BIN))).log 2>&1 &
  sleep 90
  curl -s http://127.0.0.1:18099/completion -d \
    '{"prompt":"def fib(n):","n_predict":64,"temperature":0,"seed":42,"cache_prompt":false}' \
    | python3 -c 'import sys,json; print(repr(json.load(sys.stdin)["content"]))'
  pkill -f "port 18099"; sleep 20
done
```

**REQUIRED: byte-identical output.** If it differs, the patch changed the d=0 path — stop and report.

### T2 — the schedule log, and the tail-flush path

With `LLAMA_DEFER_DEPTH > 0` the server prints the complete deferral schedule once, at graph-build
time, to the server log. Audit it.

```bash
# production-shaped schedule
LLAMA_DEFER_DEPTH=3 LLAMA_DEFER_LAYERS=16 $SRVD -m $MODEL -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 -c 2048 --no-warmup --jinja -np 1 --n-cpu-moe 16 -t 8 \
  --spec-type none --host 127.0.0.1 --port 18099 2>&1 | grep -A20 "expert-deferral"
```

Expect exactly this shape (n_layer=40, so nothing spills past the end):

```
***** EXPERT-DEFERRAL SIMULATION ACTIVE *****
  LLAMA_DEFER_DEPTH=3 LLAMA_DEFER_LAYERS=16
  routed-expert output of blocks 0..15 joins the residual 3 block(s) late
  (block i's routed experts are absent from the input of blocks i+1..i+3)
  expert-deferral schedule (n_layer=40, depth=3, layers=16):
    blk  0 routed-expert out -> top of blk 4                (blk 1..3 run stale)
    blk  1 routed-expert out -> top of blk 5                (blk 2..4 run stale)
    ...
    blk 15 routed-expert out -> top of blk 19               (blk 16..18 run stale)
```

**Audit: exactly 16 lines; blocks 0..15 each appearing once; target = i+d+1; stale range = i+1..i+d.**

Note the `+1` in the target — it is deliberate and load-bearing. `d` is defined as *the number of
blocks that run without the contribution*, which is the number of blocks of GPU work a real
scheduler gets to overlap CPU expert latency with. Joining at `i+d` instead of `i+d+1` would be a
numerical no-op at `d=1` (verified: bit-identical logits to stock). See NOTES.md §2 — this was a
real bug caught in local verification, so if you see targets of `i+d` you are running an old build.

Then force the end-of-stack spill path, which the production config never reaches:

```bash
LLAMA_DEFER_DEPTH=3 LLAMA_DEFER_LAYERS=40 $SRVD ... --port 18099 2>&1 | grep -A45 "expert-deferral"
```

Expect blocks 36..39 to read `-> tail flush (pre output_norm)` and the server to serve a normal,
non-garbage completion. This exercises the tail-flush code. It is a correctness check only — do not
evaluate this configuration.

---

## 3. HARNESS PREPARATION

The two protocol-of-record scripts hardcode four things that must change:

| Line | Problem |
|---|---|
| `SRV=$W/llama-server-v11/llama-server` | points at the stock binary |
| `R=$W/results/cpu_wave1` | would write into the live campaign's results dir |
| `TAG=...` | has no `d` in it — **all four arms would overwrite each other** |
| `PORT=18095` | may collide with the campaign |

Make parameterised copies. Do **not** edit the originals.

```bash
cd /data/projects/q27b_on_12gb
for f in cpu_wave1_quality cpu_wave1_edit; do
  sed -e 's|^SRV=$W/llama-server-v11/llama-server|SRV=${SRV:-$W/llama-server-v11/llama-server}|' \
      -e 's|^R=$W/results/cpu_wave1|R=${R:-$W/results/cpu_wave1}|' \
      -e 's|^PORT=18095|PORT=${PORT:-18095}|' \
      -e 's|^TAG=\(.*\)$|TAG=\1${DEFERTAG:-}|' \
      $f.sh > defer_${f}.sh
  chmod +x defer_${f}.sh
done
diff cpu_wave1_quality.sh defer_cpu_wave1_quality.sh   # must show exactly those 4 line changes
diff cpu_wave1_edit.sh    defer_cpu_wave1_edit.sh
```

Everything else — server flags, `reliability.py --max-attempts 1 --concurrency 1`,
`polyglot_edit.py --n-tasks 34 --max-tokens 2048 --attempts 2 --concurrency 2`, greedy, thinking
off — stays byte-identical to the protocol of record. That is the point: the *only* variable across
arms is `LLAMA_DEFER_DEPTH`.

Results go to a fresh directory:

```bash
export RDEF=/data/projects/q27b_on_12gb/results/deferral_smoke
mkdir -p $RDEF
```

---

## 4. LEG A — perplexity tripwire (fast, run this first)

Purpose: a cheap, fully deterministic numeric signal, ~5 min/arm, so a catastrophic `d` is caught
before spending 3 hours on HumanEval.

### 4.1 Two caveats that change how you run this

1. **`-c 512` returns garbage on this architecture.** Recorded in `FINDINGS.md:351` — suspected
   upstream bug, `llama-perplexity` at `-c 512` on hybrid gated-delta-net models returns uniform
   logits. The 35B is hybrid (30 GDN blocks + 10 full-attention). **Use `-c 2048`.**
2. **Perplexity is a weak instrument for *ranking* configs here** — `BENCHMARK_LANDSCAPE.md:53`
   records Spearman +0.24 vs agentic quality. That warning is about comparing *different quants*.
   Here every arm is the same model, same quant, same corpus, same everything except `d`, and
   `llama-perplexity` is deterministic (no sampling). So it is a perfectly good **tripwire and
   curve-shape indicator**. It is *not* the decision instrument — HumanEval and edit-compliance are.

### 4.2 Corpus

No wikitext corpus exists on the box and disk is tight. You do not need a standard corpus: the
absolute PPL number is meaningless here, only the *delta across arms* matters, so any fixed English
text works as long as it is byte-identical across all four arms. Build one on-box, no download:

```bash
cd /data/projects/q27b_on_12gb
python3 - <<'EOF'
import json, pathlib
# deterministic, on-box, ~200k chars of natural-ish English + code prose
srcs = [
  "/data/projects/q27b_on_12gb/ARCHITECTURE.md",
  "/data/projects/q27b_on_12gb/FINDINGS.md",
  "/data/projects/q27b_on_12gb/STRATEGY.md",
]
out = []
for s in srcs:
    p = pathlib.Path(s)
    if p.exists():
        out.append(p.read_text(errors="replace"))
he = pathlib.Path("/data/projects/q27b_on_12gb/platform_arm/harness/data/HumanEval.jsonl")
if he.exists():
    for line in he.read_text().splitlines():
        d = json.loads(line)
        out.append(d.get("prompt","") + d.get("canonical_solution",""))
txt = "\n\n".join(out)
pathlib.Path("/data/projects/q27b_on_12gb/results/deferral_smoke/ppl_corpus.txt").write_text(txt)
print("chars:", len(txt))
EOF
```

Freeze it: `md5sum $RDEF/ppl_corpus.txt` and quote that hash in your report. All four arms must use
the same file.

*(Optional, if you prefer a standard corpus and disk allows ~15 MB:
`bash /data/scratch/deferral_smoke/llama.cpp/scripts/get-wikitext-2.sh` and point `-f` at
`wikitext-2-raw/wiki.test.raw`. Either is acceptable; just be consistent and say which you used.)*

### 4.3 Run

ncm16-equivalent settings. Note perplexity has no speculative path at all, so no spec flags apply.

```bash
MODEL=/data/models/coding/q36_35ba3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
PPL=/data/scratch/deferral_smoke/llama.cpp/build/bin/llama-perplexity
RDEF=/data/projects/q27b_on_12gb/results/deferral_smoke

for D in 0 1 2 3; do
  echo "=== ppl d=$D ==="
  LLAMA_DEFER_DEPTH=$D LLAMA_DEFER_LAYERS=16 \
  $PPL -m $MODEL -f $RDEF/ppl_corpus.txt \
       -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
       -c 2048 --n-cpu-moe 16 -t 8 --chunks 20 --seed 1234 \
       > $RDEF/ppl_d${D}.log 2>&1
  grep -E "^Final estimate|expert-deferral" $RDEF/ppl_d${D}.log
done

grep -H "Final estimate" $RDEF/ppl_d*.log
```

### 4.4 Reading Leg A

`llama-perplexity` is deterministic, so there is **no run-to-run noise** — any delta is real signal,
and the question is only magnitude.

| ΔPPL vs d=0 | Meaning | Action |
|---|---|---|
| exactly 0.000 at d=0 vs stock binary | build correct | required (see T1) |
| < 1% at d=3 | staleness is numerically tiny | proceed to Leg B, expect GREEN |
| 1-5% at d=3 | real but modest | proceed to Leg B — this is exactly the case Leg B exists to adjudicate |
| 5-20% at d=3 | large | run Leg B at d=0 and d=1 only; report the curve |
| > 20% at d=1 | broken | **STOP.** Report RED, do not spend 3 hours. Include the ppl curve. |

Also plot the shape: if ΔPPL is roughly linear in `d`, the effect is a smooth degradation and a
smaller `d` is likely shippable. If it is flat then jumps, something structural breaks at that depth.

---

## 5. LEG B — HumanEval-164 at production flags, `--spec-type none`

### 5.1 Why `--spec-type none` (this is deliberate, not a shortcut)

Production runs `--spec-type ngram-simple,draft-mtp --spec-draft-n-max 2`. We turn it off. Three
reasons, in order of importance:

1. **Single clean forward path = one variable.** With speculation on, generated bytes depend on the
   draft/verify accept-reject trajectory. Deferral perturbs the target model's logits slightly,
   which perturbs acceptance, which perturbs batch shapes, which perturbs FP rounding, which flips
   greedy argmaxes. The project has already measured this: `cpu_wave1_quality.sh`'s own header notes
   *"changing the spec settings changes generated bytes (batch-shape FP rounding flipping a greedy
   argmax)"*. A quality delta measured with spec on could not be attributed to staleness.
2. **`draft-mtp` runs a second graph we also modified the input to.** The MTP head is seeded by
   `res->t_h_nextn`, which is downstream of the deferral (we flush before `output_norm` precisely so
   the seed is correct — see NOTES.md §2). Enabling `draft-mtp` would fold the deferral's effect on
   the *draft* head into the same number as its effect on the *target* model. Two coupled effects,
   one measurement.
3. **A nospec ncm16 baseline already exists** for cross-checking:
   `results/q3kxl/reliability_Q3_K_XL_ncm16_nospec_he164_3060.json` = **154/164 (0.93902)**. Useful
   as a sanity anchor — but **do not use it as your control.** Measure `d=0` yourself with the
   patched binary, in this campaign. Historical runs differ in binary and in campaign conditions.

Follow-up owed: deferral × speculative-decoding interaction is **not** covered by this experiment. If
Leg B is GREEN, a separate spec-on confirmation is required before shipping.

### 5.2 Run

Each arm is ~800 s of eval plus ~2 min model load plus the `dd` page-cache prewarm. Budget ~20 min
per arm, ~1.5 h for four.

```bash
cd /data/projects/q27b_on_12gb
export RDEF=/data/projects/q27b_on_12gb/results/deferral_smoke
export SRV=/data/scratch/deferral_smoke/llama.cpp/build/bin/llama-server

for D in 0 1 2 3; do
  echo "########## HE-164  d=$D ##########"
  SRV=$SRV R=$RDEF PORT=18095 DEFERTAG=_d${D} \
  LLAMA_DEFER_DEPTH=$D LLAMA_DEFER_LAYERS=16 \
  NCM=16 THREADS=8 SPEC=none LIMIT=164 CTXCP=none \
    bash defer_cpu_wave1_quality.sh 2>&1 | tee $RDEF/he164_d${D}.runlog
done
```

Produces, per arm:
`$RDEF/reliability_Q3_K_XL_ncm16_t8_nospec_he164_3060_d${D}.{jsonl,json}`,
`srv_..._d${D}.log`, `vrampeak_..._d${D}.txt`.

**After each arm, confirm the deferral actually engaged** (or, for `d=0`, that it did not):

```bash
grep -c "EXPERT-DEFERRAL SIMULATION ACTIVE" $RDEF/srv_*_d0.log   # must be 0
grep -c "EXPERT-DEFERRAL SIMULATION ACTIVE" $RDEF/srv_*_d3.log   # must be 1
grep "routed-expert out ->" $RDEF/srv_*_d3.log | wc -l           # must be 16
```

An arm where this check fails is void. This is the single most likely failure mode of the whole
exercise (env var not reaching the server process) — check it every time.

### 5.3 Expected VRAM — unchanged

Expert **placement is untouched** by this patch: `--n-cpu-moe 16` still puts exactly
`blk.{0..15}.ffn_{gate,up,gate_up,down}_exps` on the CPU and nothing else moves. Model buffer sizes
must be identical across all four arms.

The only expected change is the **compute buffer**, because up to `d` pending
`[n_embd=2048, n_tokens]` fp32 tensors are held live across block boundaries:

```
extra bytes  <=  d * 2048 * n_ubatch * 4
d=3, ub=512  ->  ~12.6 MiB
```

**So: model buffer identical; CUDA compute buffer up to ~13 MiB larger at d=3; peak VRAM within
~13 MiB of the d=0 arm.** Compare `vrampeak_*_d0.txt` against `vrampeak_*_d3.txt`. If the difference
exceeds ~20 MiB, something other than the intended change is happening — investigate before
believing the quality numbers. (Headroom is comfortable at `-c 8192`; the 456 MiB-headroom figure in
RESULTS.md was for the `-c 16384` speed argv.)

---

## 6. LEG C — edit-compliance (polyglot 34), conditional

Run this only if Leg B does not already produce a clear RED. ~15 min per arm.

```bash
cd /data/projects/q27b_on_12gb
for D in 0 1 2 3; do
  echo "########## EDIT-34  d=$D ##########"
  SRV=/data/scratch/deferral_smoke/llama.cpp/build/bin/llama-server R=$RDEF PORT=18095 DEFERTAG=_d${D} \
  LLAMA_DEFER_DEPTH=$D LLAMA_DEFER_LAYERS=16 \
  NCM=16 THREADS=8 SPEC=none ARM=defer_d${D} CTXCP=none \
    bash defer_cpu_wave1_edit.sh 2>&1 | tee $RDEF/edit34_d${D}.runlog
done
```

Produces `$RDEF/edit_Q3_K_XL_ncm16_t8_nospec_edit34_3060_d${D}.{jsonl,json}` with
`edit_compliance_1 {n, correct, rate, ci95}`, `pass_1`, `pass_2`, `mean_completion_tokens`.

If time is short, run `d=0` and `d=3` only — those are the two the decision rule needs.

---

## 7. DECISION RULE

### 7.1 Reference points (existing measurements, ncm16, for orientation only)

| Metric | Value | Source |
|---|---|---|
| HE-164 pass@1, ncm16 nospec | 154/164 = 0.93902 | `results/q3kxl/reliability_Q3_K_XL_ncm16_nospec_he164_3060.json` |
| HE-164 pass@1, ncm16 recommended (spec on) | 153/164 = 0.93293 | `results/cpu_wave1/reliability_..._ngrammtp_m256_he164_3060.json` |
| observed spread across 4 ncm16 HE-164 runs | 152-154 / 164 | ibid. |
| edit_compliance_1, ncm16 recommended | 34/34 = 1.000 | `results/cpu_wave1/edit_..._edit34_3060.json` |
| mean completion tokens/task (thinking tripwire) | 265.5, must stay ≪ 350 | ibid. |

**Your control is your own `d=0` arm, not these.** These exist to catch a broken setup: if your
`d=0` lands outside 152-154/164, something is wrong with the rig — fix that before interpreting
anything.

### 7.2 The rule

The project's instrument of record for a quality decision is **paired McNemar exact, per task** —
not a delta threshold. Precedents: `cert_35ba3b.md:237` (b=1,c=3,p=0.625),
`RESULTS.md:539` (1 discordant, p=1.0). Use it. The per-task JSONL is on disk, so pairing is free.

```bash
python3 - <<'EOF'
import json, sys, itertools
from math import comb
R="/data/projects/q27b_on_12gb/results/deferral_smoke/"
def load(d):
    # reliability.py schema: keys are task_id, passed_at, attempts, index,
    # total_completion_tokens, total_wall_s. passed_at is 1 on success, None on failure.
    p=f"{R}reliability_Q3_K_XL_ncm16_t8_nospec_he164_3060_d{d}.jsonl"
    out={}
    for line in open(p):
        r=json.loads(line)
        out[r["task_id"]] = r["passed_at"] is not None
    return out
base=load(0)
for d in (1,2,3):
    try: arm=load(d)
    except FileNotFoundError: continue
    keys=sorted(set(base)&set(arm))
    b=sum(1 for k in keys if base[k] and not arm[k])   # regressions
    c=sum(1 for k in keys if arm[k] and not base[k])   # improvements
    n=b+c
    # exact two-sided binomial, p=0.5
    p = 1.0 if n==0 else min(1.0, 2*sum(comb(n,i) for i in range(0,min(b,c)+1))/2**n)
    print(f"d={d}: passed {sum(arm.values())}/{len(keys)} vs d0 {sum(base.values())}/{len(keys)} "
          f"| regressions b={b} improvements c={c} | McNemar exact p={p:.4f}")
EOF
```

**GREEN-LIGHT the build** if *all* of the following hold for **d=3**:

1. **HE-164, paired:** McNemar exact `p > 0.05` **and** net task delta `|passed(d3) - passed(d0)| <= 3`
   (≈ ±1.8 pt; matches the project's own tolerance and the 152-154 empirical spread).
2. **Edit-compliance:** `edit_compliance_1(d=3) >= edit_compliance_1(d=0) - 1 task` (i.e. ≥ 33/34 if
   your `d=0` is 34/34).
3. **Thinking tripwire:** `mean_completion_tokens_per_task(d=3) < 350` and within ~15% of `d=0`.
   A blow-up here means degeneration/rambling, which pass@1 can mask.
4. **VRAM:** peak within ~20 MiB of `d=0` (§5.3).
5. **Leg A:** ΔPPL(d3 vs d0) is not catastrophic (say < 10%). If PPL moved a lot but HumanEval did
   not, say so explicitly rather than quietly discarding it — that combination is interesting.

**Report a DEGRADATION CURVE** (not just pass/fail) in every case: for each of `d = 0,1,2,3` give
PPL, HE-164 passed/164, McNemar p vs d=0, edit_compliance_1, and mean tokens. Then state:

> **The largest `d` that is still within noise of `d=0` is `d = ___`.**

That number is the deliverable. If `d=3` is GREEN, the scheduler can be built for depth 3. If only
`d=1` is GREEN, the overlap window is one block and the architect needs to know that *before*
committing to the build, because a 1-block window may not be enough to hide CPU expert latency —
which would make the whole optimisation not worth building.

If `d=1` already fails: **RED**. Report it immediately; the optimisation is dead as specified and no
further legs are needed.

### 7.3 What a GREEN light does and does not authorise

GREEN means **the numerics of staleness are acceptable**. It does **not** mean the optimisation
works. Read `NOTES.md` §4 before reporting — in particular these are untested and remain live risks:
speculative-decoding rollback of a pending deferred contribution; recurrent-state (gated delta net)
hazards under real concurrency; CPU→GPU transfer numerics; and whether the overlap actually buys any
wall-clock time at all (this patch is deliberately scheduling-neutral and measures nothing about
speed).

---

## 8. REPORTING TEMPLATE

```
Build:      /data/scratch/deferral_smoke/llama.cpp @ daef7b687 + v11 patches + deferral_sim.patch
            RUNPATH verified: ____
T1 d=0 == stock byte-identical:  PASS / FAIL
T2 schedule log 16 lines, targets i+3:  PASS / FAIL
T2 tail-flush (LAYERS=40):  PASS / FAIL
ppl corpus md5: ____   chunks: 20   -c 2048

  d  |  PPL    | ΔPPL%  | HE-164 | McNemar p | edit_compl_1 | mean tok | peak VRAM
-----+---------+--------+--------+-----------+--------------+----------+-----------
  0  |         |   —    |   /164 |     —     |        /34   |          |
  1  |         |        |   /164 |           |        /34   |          |
  2  |         |        |   /164 |           |        /34   |          |
  3  |         |        |   /164 |           |        /34   |          |

Per-arm deferral engagement check (grep EXPERT-DEFERRAL in srv log): d0=0, d1=1, d2=1, d3=1  ✔/✘

VERDICT:  GREEN (build for d=3)  /  PARTIAL (largest safe d = __)  /  RED
```

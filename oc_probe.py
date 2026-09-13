#!/usr/bin/env python3
"""Headless clock/power probe on the 27B flagship — adapted from
q3kxl_sweep.py (same guard(), VramSampler, start_server()/stop_server(),
fire() and JSON output shape), retargeted at nvidia-smi clock/power state
rather than a model config knob.

Every arm is a *clock state*, not a model config: the model config is frozen
at the shipping flagship (IQ3_XXS, q8_0 KV, c=16384, MTP n=2) and only the
nvidia-smi state changes between arms.

Instruments, per arm:
  llama-bench tg64 at d=0 and d=13000, -r 3   -> raw decode, low variance
  server cell, MTP n=2, 4 measured requests   -> shipping decode + acceptance

Tripwire (the field insight): MTP acceptance is the first instability
symptom. If pooled acceptance falls more than ACC_TRIPWIRE below the stock
arm's, the arm is marked UNSTABLE and all clock state is reverted at once.

ALL clock state is reverted in a finally: block, and the revert is verified
by re-reading nvidia-smi.
"""
import argparse, hashlib, json, os, re, signal, subprocess, sys, threading, time, urllib.request

REPO = "/data/projects/q27b_on_12gb"
BIN_DIR = f"{REPO}/llama-server-v11"
BIN = f"{BIN_DIR}/llama-server"
BENCH = f"{BIN_DIR}/llama-bench"
MODEL = f"{REPO}/models/unsloth-q27b/Qwen3.8-27B-UD-IQ3_XXS.gguf"
TMPL = f"{REPO}/models/templates/chat_template.jinja"
OUTDIR = f"{REPO}/results/oc_probe"
PORT = 18097
URL = f"http://127.0.0.1:{PORT}"

ACC_TRIPWIRE = 0.05          # 5 points, per the brief
CTX = 16384
KV = "q8_0"

CODE_PROMPT = (
    "Write a complete Python module implementing a persistent B-tree index on disk. "
    "Include: a Node class with serialization to fixed-size pages, insert with node "
    "splitting, search, an in-order range scan generator, a free-page list, and a "
    "crash-safe write path using a write-ahead log. Full type hints, docstrings, and "
    "a __main__ demo that inserts 1000 keys and verifies them. Do not abbreviate."
)
MAXTOK = 400


# ---------------------------------------------------------------- nvidia-smi
def smi(*args, check=True):
    o = subprocess.run(["sudo", "nvidia-smi"] + list(args),
                       capture_output=True, text=True, timeout=120)
    if check and o.returncode != 0:
        raise RuntimeError(f"nvidia-smi {args} failed: {o.stderr.strip()}")
    return o.stdout.strip()


def query(fields):
    o = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60)
    return [x.strip() for x in o.stdout.strip().split(",")]


def clock_state():
    """Everything about the current clock/power state that we can read."""
    f = ("persistence_mode,power.limit,power.draw,clocks.sm,clocks.mem,"
         "temperature.gpu,clocks_throttle_reasons.active")
    v = query(f)
    return {"persistence": v[0], "power_limit_w": v[1], "power_draw_w": v[2],
            "sm_mhz": v[3], "mem_mhz": v[4], "temp_c": v[5],
            "throttle_active": v[6]}


def apply_arm(arm):
    """Apply one clock state. Returns the observed state after applying."""
    smi("-pm", "1")
    smi("-rgc"); smi("-rmc")
    smi("-pl", str(arm["pl"]))
    if arm.get("lgc"):
        smi("-lgc", str(arm["lgc"]))
    if arm.get("lmc"):
        smi("-lmc", str(arm["lmc"]))
    time.sleep(3)
    return clock_state()


def revert_all():
    out = {}
    for cmd in (["-rgc"], ["-rmc"], ["-pl", "170"]):
        try:
            out[" ".join(cmd)] = smi(*cmd)
        except Exception as e:
            out[" ".join(cmd)] = f"ERROR: {e}"
    time.sleep(2)
    out["verified_state"] = clock_state()
    return out


# ---------------------------------------------------------------- guard/util
def gpu_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def guard(limit=500, tries=240):
    for _ in range(tries):
        u = gpu_used()
        if 0 <= u < limit:
            return u
        time.sleep(2)
    sys.exit(f"GPU GUARD FAILED: {gpu_used()} MiB still resident")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def read_rss_mib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


class VramSampler(threading.Thread):
    """1 Hz nvidia-smi sampler: VRAM peak, SM clock, mem clock, power, temp,
    throttle reasons -- the clock columns are the point of this run."""
    def __init__(self, pid=None):
        super().__init__(daemon=True)
        self.pid = pid
        self.peak = 0
        self.rss_peak = 0.0
        self.clocks, self.mclocks, self.power, self.temps = [], [], [], []
        self.throttles = set()
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=memory.used,clocks.sm,clocks.mem,power.draw,"
                     "temperature.gpu,clocks_throttle_reasons.active",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20)
                p = [x.strip() for x in o.stdout.strip().splitlines()[0].split(",")]
                mem = int(p[0])
                if mem > self.peak:
                    self.peak = mem
                self.clocks.append(int(p[1]))
                self.mclocks.append(int(p[2]))
                self.power.append(float(p[3]))
                self.temps.append(int(p[4]))
                self.throttles.add(p[5])
            except Exception:
                pass
            if self.pid is not None:
                rss = read_rss_mib(self.pid)
                if rss is not None and rss > self.rss_peak:
                    self.rss_peak = rss
            self.stop_flag.wait(1.0)

    def summary(self):
        return {
            "sm_clock_mean": round(mean(self.clocks), 1) if self.clocks else None,
            "sm_clock_max": max(self.clocks) if self.clocks else None,
            "sm_clock_min": min(self.clocks) if self.clocks else None,
            "mem_clock_mean": round(mean(self.mclocks), 1) if self.mclocks else None,
            "power_mean_w": round(mean(self.power), 1) if self.power else None,
            "power_max_w": max(self.power) if self.power else None,
            "temp_max_c": max(self.temps) if self.temps else None,
            "throttle_reasons": sorted(self.throttles),
        }


# ---------------------------------------------------------------- llama-bench
def run_bench(arm_name, depth, reps=3):
    """llama-bench tg64 at a filled depth. Sampler runs alongside so the arm's
    real clocks under load are recorded, not just the requested ones."""
    guard()
    cmd = [BENCH, "-m", MODEL, "-ngl", "99", "-fa", "on",
           "-ctk", KV, "-ctv", KV, "-p", "0", "-n", "64",
           "-d", str(depth), "-r", str(reps), "-o", "json"]
    print(f"  bench d={depth}: $ {' '.join(cmd)}", flush=True)
    s = VramSampler(); s.start()
    t0 = time.time()
    try:
        o = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    finally:
        s.stop_flag.set(); s.join(timeout=5)
    wall = time.time() - t0
    row = {"depth": depth, "reps": reps, "wall_s": round(wall, 1),
           "cmd": cmd, **s.summary(), "vram_peak_mib": s.peak}
    try:
        js = json.loads(o.stdout[o.stdout.index("["):])
        r = js[-1]
        row["tg_tps"] = r["avg_ts"]
        row["tg_tps_stddev"] = r["stddev_ts"]
        row["samples_ts"] = r.get("samples_ts")
        row["status"] = "OK"
    except Exception as e:
        row["status"] = f"FAIL:{e}"
        row["stderr_tail"] = o.stderr[-800:]
    print(f"    -> {row.get('tg_tps')} t/s (sd {row.get('tg_tps_stddev')}) "
          f"sm={row['sm_clock_mean']} mem={row['mem_clock_mean']} "
          f"P={row['power_mean_w']}W T={row['temp_max_c']}C "
          f"throttle={row['throttle_reasons']}", flush=True)
    for _ in range(120):
        if gpu_used() < 500:
            break
        time.sleep(1)
    return row


# ---------------------------------------------------------------- server cell
def post(body, timeout=1200):
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start_server(name, cmd, logpath):
    guard()
    print(f"  $ {' '.join(cmd)}", flush=True)
    lf = open(logpath, "w")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if p.poll() is not None:
            lf.close()
            txt = open(logpath, errors="replace").read()
            m = re.search(r"(out of memory|failed to allocate|unknown argument\S*|"
                          r"invalid|failed to load|unsupported)[^\n]*", txt, re.I)
            return None, lf, f"FAIL:{m.group(0)[:120] if m else 'unknown'}", time.time() - t0
        try:
            urllib.request.urlopen(URL + "/health", timeout=5)
            return p, lf, None, time.time() - t0
        except Exception:
            time.sleep(2)
    return p, lf, "FAIL:health_timeout", time.time() - t0


def stop_server(p, lf):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=120)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    lf.close()
    for _ in range(120):
        if gpu_used() < 500:
            break
        time.sleep(1)
    time.sleep(3)


def fire(prompt, tag):
    body = {"messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": MAXTOK,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    w0 = time.time()
    r = post(body)
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    t = r.get("timings") or {}
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    return {"tag": tag, "prompt_n": t.get("prompt_n"),
            "prompt_tps": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"),
            "decode_tps": t.get("predicted_per_second"),
            "draft_n": dn, "draft_n_accepted": da,
            "acceptance": (da / dn) if (dn and da is not None) else None,
            "wall_s": round(time.time() - w0, 3),
            "finish": ch.get("finish_reason"),
            "content_sha1": hashlib.sha1(content.encode()).hexdigest()[:16],
            "content_head": content[:80]}


def run_server_cell(name, reps=4):
    logpath = f"{OUTDIR}/{name}.serverlog"
    cmd = [BIN, "-m", MODEL, "-ngl", "99", "-fa", "on",
           "-ctk", KV, "-ctv", KV, "-c", str(CTX),
           "--host", "127.0.0.1", "--port", str(PORT), "--no-warmup",
           # -ctxcp 0: context checkpoints are ~150 MiB each on this model and
           # this config peaks near 12,000 of 12,044 MiB. Off in every arm so
           # a checkpoint allocation can never be mistaken for clock instability.
           "-ctxcp", "0",
           "--chat-template-file", TMPL, "--parallel", "1",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    p, lf, err, load_s = start_server(name, cmd, logpath)
    if err:
        print(f"  {err}", flush=True)
        if p:
            stop_server(p, lf)
        return {"name": name, "status": "FAIL", "error": err}
    s = VramSampler(pid=p.pid); s.start()
    rows = []
    try:
        for i in range(reps + 1):
            try:
                row = fire(CODE_PROMPT, f"r{i}")
            except Exception as e:
                rows.append({"tag": f"r{i}", "error": str(e), "warmup": i == 0})
                continue
            row["warmup"] = (i == 0)
            rows.append(row)
            print(f"    r{i} {row['decode_tps']:.3f} t/s acc="
                  f"{row['acceptance']:.4f} n={row['predicted_n']} "
                  f"sha={row['content_sha1']}", flush=True)
    finally:
        s.stop_flag.set(); s.join(timeout=5)
        stop_server(p, lf)
    ok = [r for r in rows if not r.get("warmup") and r.get("decode_tps")]
    dn = sum((r["draft_n"] or 0) for r in ok)
    da = sum((r["draft_n_accepted"] or 0) for r in ok)
    agg = {"name": name, "status": "OK" if ok else "REQ_FAIL",
           "load_time_s": round(load_s, 2), "argv": cmd,
           "vram_peak_mib": s.peak, "rss_peak_mib": round(s.rss_peak, 1),
           "decode_tps": [r["decode_tps"] for r in ok],
           "decode_tps_mean": mean([r["decode_tps"] for r in ok]),
           "prefill_tps_mean": mean([r["prompt_tps"] for r in ok]),
           "acceptance_pooled": (da / dn) if dn else None,
           "rows": rows, **s.summary()}
    print(f"  -> server {agg['decode_tps_mean']:.3f} t/s, "
          f"acc={agg['acceptance_pooled']:.4f}, vram={s.peak} MiB, "
          f"sm={agg['sm_clock_mean']}, P={agg['power_mean_w']}W", flush=True)
    return agg


# ---------------------------------------------------------------- arms
ARMS = [
    {"name": "A_stock",           "pl": 170, "lgc": None, "lmc": None,
     "note": "factory default: pl 170 W, no locks (pm on for sampling stability)"},
    {"name": "B_pl190",           "pl": 190, "lgc": None, "lmc": None,
     "note": "power limit to the card's max (190 W)"},
    {"name": "C_pl190_lgc2145",   "pl": 190, "lgc": 2145, "lmc": None,
     "note": "pl max + GPU clock locked to max supported (2145 MHz)"},
    {"name": "D_pl190_lgc2145_lmc7501", "pl": 190, "lgc": 2145, "lmc": 7501,
     "note": "pl max + gpu clock max + mem clock pinned to max P-state (7501 MHz)"},
    {"name": "E_lgc2400_probe",   "pl": 190, "lgc": 2400, "lmc": 8000,
     "note": "ABOVE-SPEC REQUEST: does nvidia-smi silently clamp, or actually OC?"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all", help="comma list of arm names, or 'all'")
    ap.add_argument("--server-arms", default="A_stock,D_pl190_lgc2145_lmc7501",
                    help="arms that also get the MTP server cell")
    ap.add_argument("--depths", default="0,13000")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    outf = f"{OUTDIR}/oc_probe.json"
    results = json.load(open(outf)) if os.path.exists(outf) else {}

    want = [a for a in ARMS if args.arms == "all" or a["name"] in args.arms.split(",")]
    server_arms = set(args.server_arms.split(","))
    depths = [int(d) for d in args.depths.split(",")]

    results.setdefault("_meta", {})["baseline_clock_state"] = clock_state()
    results["_meta"]["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results["_meta"]["model"] = MODEL
    results["_meta"]["bin"] = BIN

    stock_acc = None
    try:
        for arm in want:
            print(f"\n=== ARM {arm['name']} — {arm['note']} ===", flush=True)
            applied = apply_arm(arm)
            print(f"  applied, idle state: {applied}", flush=True)
            rec = {"arm": arm, "applied_state_idle": applied, "benches": [], }
            for d in depths:
                rec["benches"].append(run_bench(arm["name"], d))
            if arm["name"] in server_arms:
                rec["server"] = run_server_cell(f"{arm['name']}_mtp2")
                acc = rec["server"].get("acceptance_pooled")
                if arm["name"] == "A_stock":
                    stock_acc = acc
                elif stock_acc is not None and acc is not None:
                    drop = stock_acc - acc
                    rec["acceptance_drop_vs_stock"] = round(drop, 4)
                    if drop > ACC_TRIPWIRE:
                        rec["TRIPWIRE"] = "UNSTABLE: acceptance dropped >5pts"
                        print(f"  !! TRIPWIRE: acceptance {acc:.4f} vs stock "
                              f"{stock_acc:.4f} — reverting NOW", flush=True)
                        results[arm["name"]] = rec
                        json.dump(results, open(outf, "w"), indent=1)
                        break
            results[arm["name"]] = rec
            json.dump(results, open(outf, "w"), indent=1)
    finally:
        print("\n=== REVERTING ALL CLOCK STATE ===", flush=True)
        results["_meta"]["revert"] = revert_all()
        print(json.dumps(results["_meta"]["revert"], indent=1), flush=True)
        json.dump(results, open(outf, "w"), indent=1)
    print("\nOC_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()

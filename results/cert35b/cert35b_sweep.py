#!/usr/bin/env python3
"""Qwen3.6-35B-A3B placement certification — adapted from
q27b_on_12gb/speed_recert.py (same measurement protocol, VramSampler, guard(),
parse_server_log(), start_server()/stop logic, and JSON output shape), retargeted
at a MoE model where the speed-relevant knob is CPU/GPU expert placement
(--n-cpu-moe) rather than quant choice, and served with the GGUF's own embedded
chat template (--jinja) instead of the 27B repo's --chat-template-file.

Differences from the reference driver, all deliberate:
  - single model (Qwen3.6-35B-A3B, Q4_K_XL), no BIN_NEW / IQ3 / Q2KXL table.
  - server launched with --jinja, NOT --chat-template-file (the 27B template
    must never be applied to this model).
  - every launch is preceded by a `dd` prewarm of the gguf into page cache
    (prewarm_s recorded), on top of the existing GPU guard.
  - VramSampler additionally samples the server process RSS (/proc/<pid>/status
    VmRSS) and system-wide used memory (/proc/meminfo MemTotal-MemAvailable)
    at 1 Hz, and records the peak of each per cell.
  - modes are `placement` (the --n-cpu-moe x spec-type grid), `depth` (one
    named config decoded at ~14k context), and `cell` (one ad-hoc config) —
    there is no `content` mode here.
  - a cell that fails to start, or a request that raises, does not abort the
    run: it's recorded with status FAIL / REQ_FAIL and the sweep continues.

Protocol (unchanged from the reference):
  server  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c <ctx> --host 127.0.0.1
          --port <PORT> --no-warmup --jinja --parallel 1 <extra>
  client  temperature=0, top_k=1, seed=1234, max_tokens=400, cache_prompt=false,
          chat_template_kwargs={"enable_thinking": false}
  cell    1 discarded warm-up request + 4 measured requests, mean of the 4
  VRAM    max of a 1 Hz nvidia-smi sample taken while the requests run
  guard   refuse to launch unless the card is under 500 MiB
"""
import argparse, hashlib, json, os, re, signal, subprocess, sys, threading, time, urllib.request

BIN = os.environ.get(
    "CERT_BIN", "/data/projects/q27b_on_12gb/llama-server-v11/llama-server")
MODEL = os.environ.get(
    "CERT_MODEL",
    "/data/models/coding/q36_35ba3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
OUTDIR = "/data/projects/q27b_on_12gb/results/cert35b"
PORT = 18096
URL = f"http://127.0.0.1:{PORT}"

# The mtp_profile.md / speed_recert.py prompt, verbatim, so cells here are
# directly comparable to the 27B lineage.
CODE_PROMPT = (
    "Write a complete Python module implementing a persistent B-tree index on disk. "
    "Include: a Node class with serialization to fixed-size pages, insert with node "
    "splitting, search, an in-order range scan generator, a free-page list, and a "
    "crash-safe write path using a write-ahead log. Full type hints, docstrings, and "
    "a __main__ demo that inserts 1000 keys and verifies them. Do not abbreviate."
)

MAXTOK = 400


def depth_prompt(n_blocks=62):     # 90 blocks = 19593 tok, overflows c=16384
    """~14K tokens of novel-ish code context, so decode runs at DEPTH ~14-15K
    inside a c=16384 window. kernel_profile.md's f16-vs-q8 KV gap is a property
    of attention at depth (it is zero at depth 0), so the KV question cannot be
    answered by a short prompt."""
    blocks = []
    for i in range(n_blocks):
        blocks.append(
            f"# --- module_{i:03d}.py ---\n"
            f"def transform_{i:03d}(rows: list[dict], threshold_{i:03d}: float = {i}.5) -> list[dict]:\n"
            f"    \"\"\"Filter rows whose 'score_{i:03d}' exceeds threshold_{i:03d} and\n"
            f"    normalise the 'weight_{i:03d}' column against the surviving maximum.\"\"\"\n"
            f"    kept = [r for r in rows if r.get('score_{i:03d}', 0.0) > threshold_{i:03d}]\n"
            f"    if not kept:\n"
            f"        return []\n"
            f"    peak = max(r['weight_{i:03d}'] for r in kept) or 1.0\n"
            f"    for r in kept:\n"
            f"        r['weight_{i:03d}'] = r['weight_{i:03d}'] / peak\n"
            f"        r['bucket_{i:03d}'] = int(r['weight_{i:03d}'] * {i % 7 + 3})\n"
            f"    return sorted(kept, key=lambda r: -r['weight_{i:03d}'])\n")
    return ("Below is a dump of a small Python repository.\n\n" + "\n".join(blocks) +
            "\n\nNow write a single dispatcher module that imports every transform_NNN "
            "function above, registers them in a dict keyed by index, and exposes "
            "run_pipeline(rows, order) applying them in the given order with error "
            "handling and type hints. Do not abbreviate.")


def gpu_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def guard(limit=500, tries=240):
    """Refuse to start a server unless the card is essentially empty."""
    for _ in range(tries):
        u = gpu_used()
        if 0 <= u < limit:
            return u
        time.sleep(2)
    sys.exit(f"GPU GUARD FAILED: {gpu_used()} MiB still resident")


def read_rss_mib(pid):
    """Server process RSS from /proc/<pid>/status, in MiB."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        return None
    return None


def read_sys_used_mib():
    """System-wide used memory = MemTotal - MemAvailable, from /proc/meminfo, in MiB."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0])  # kB
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if total is None or avail is None:
            return None
        return (total - avail) / 1024.0
    except Exception:
        return None


def post(body, timeout=1800):
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def parse_server_log(path, keep=4):
    """llama-server's own 'draft acceptance = A (X accepted / Y generated),
    mean len = L' print_timing lines; average the last `keep`."""
    try:
        txt = open(path, errors="replace").read()
    except OSError:
        return {}
    rows = re.findall(
        r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), "
        r"mean len =\s*([\d.]+)", txt)
    if not rows:
        return {}
    rows = rows[-keep:]
    return {
        "log_accept": mean([float(r[0]) for r in rows]),
        "log_n_acc": mean([float(r[1]) for r in rows]),
        "log_n_gen": mean([float(r[2]) for r in rows]),
        "log_mean_len": mean([float(r[3]) for r in rows]),
    }


def tail_log(path, n=25):
    try:
        return open(path, errors="replace").read().splitlines()[-n:]
    except OSError:
        return []


class VramSampler(threading.Thread):
    """VRAM peak + SM clock + temperature, PLUS host-RAM peaks: server process
    RSS (/proc/<pid>/status VmRSS) and system-wide used memory (/proc/meminfo
    MemTotal-MemAvailable). --n-cpu-moe puts expert tensors on the CPU, so host
    RAM is now a real confound between placement cells alongside VRAM."""

    def __init__(self, pid=None):
        super().__init__(daemon=True)
        self.pid = pid
        self.stop_flag = threading.Event()
        self.peak = 0
        self.clocks = []
        self.temps = []
        self.rss_peak = 0.0
        self.sys_used_peak = 0.0

    def run(self):
        while not self.stop_flag.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,clocks.sm,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
                mem, clk, tmp = [int(x.strip()) for x in o.split(",")]
                if mem > self.peak:
                    self.peak = mem
                self.clocks.append(clk)
                self.temps.append(tmp)
            except Exception:
                pass
            if self.pid is not None:
                rss = read_rss_mib(self.pid)
                if rss is not None and rss > self.rss_peak:
                    self.rss_peak = rss
            sysu = read_sys_used_mib()
            if sysu is not None and sysu > self.sys_used_peak:
                self.sys_used_peak = sysu
            self.stop_flag.wait(1.0)


def prewarm_model():
    """dd the gguf into page cache before every launch, so cold-cache disk
    reads never masquerade as a load-time or first-token regression."""
    t0 = time.time()
    try:
        subprocess.run(["dd", f"if={MODEL}", "of=/dev/null", "bs=64M"],
                       capture_output=True, timeout=1800)
    except Exception as e:
        print(f"  prewarm failed: {e}", flush=True)
    return time.time() - t0


def build_cmd(ctx, kv, extra):
    return [BIN, "-m", MODEL, "-ngl", "99", "-fa", "on",
            "-ctk", kv, "-ctv", kv, "-c", str(ctx),
            "--host", "127.0.0.1", "--port", str(PORT), "--no-warmup",
            "--jinja", "--parallel", "1"] + list(extra)


def spec_extra(spec):
    if spec == "none":
        return ["--spec-type", "none"]
    if spec == "mtp2":
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    if spec == "mtp4":
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
    raise ValueError(f"unknown spec {spec}")


def start_server(name, cmd, logpath):
    guard()
    print(f"\n=== {name} ===\n$ {' '.join(cmd)}", flush=True)
    lf = open(logpath, "w")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if p.poll() is not None:
            lf.close()
            why = ""
            try:
                txt = open(logpath, errors="replace").read()
                m = re.search(r"(out of memory|failed to allocate|unknown argument\S*|"
                              r"invalid|failed to load|unsupported)[^\n]*", txt, re.I)
                why = m.group(0)[:120] if m else "unknown"
            except OSError:
                why = "unknown"
            return None, lf, f"FAIL:{why}", time.time() - t0
        try:
            urllib.request.urlopen(URL + "/health", timeout=5)
            load_s = time.time() - t0
            print(f"  healthy in {load_s:.0f}s, vram={gpu_used()} MiB", flush=True)
            return p, lf, None, load_s
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
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": MAXTOK,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    w0 = time.time()
    r = post(body)
    wall = time.time() - w0
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    t = r.get("timings") or {}
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    row = {
        "tag": tag,
        "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": t.get("prompt_per_second"),          # prefill t/s
        "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
        "decode_tps": t.get("predicted_per_second"),       # decode t/s
        "draft_n": dn, "draft_n_accepted": da,
        "acceptance": (da / dn) if (dn and da is not None) else None,
        "wall_s": round(wall, 3),
        "finish": ch.get("finish_reason"),
        # thinking tripwire: with the switch working these must be 0 / False
        "reasoning_len": len(reasoning),
        "think_tag_in_content": "<think>" in content,
        "content_sha1": hashlib.sha1(content.encode()).hexdigest()[:16],
        "content_head": content[:80],
    }
    return row


def run_cell(name, ctx, kv, extra, prompt, reps=4, memcap_note=None):
    """1 discarded warm-up + `reps` measured requests against a single prompt.
    Never raises: a server that won't start, or a request that raises, is
    recorded on the returned dict (status FAIL / REQ_FAIL) so the caller can
    move on to the next cell."""
    logpath = f"{OUTDIR}/{name}.serverlog"
    cmd = build_cmd(ctx, kv, extra)
    prewarm_s = prewarm_model()
    p, lf, err, load_s = start_server(name, cmd, logpath)
    if err:
        print(f"  {err}", flush=True)
        log_tail = tail_log(logpath, 25)
        if p:
            stop_server(p, lf)
        else:
            for _ in range(60):
                if gpu_used() < 500:
                    break
                time.sleep(1)
        return {
            "name": name, "status": "FAIL", "argv": cmd,
            "ctx": ctx, "kv": kv, "extra": extra,
            "prewarm_s": round(prewarm_s, 2), "load_time_s": round(load_s, 2),
            "memcap_note": memcap_note,
            "error": err, "log_tail": log_tail,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    vram_at_load = gpu_used()
    sampler = VramSampler(pid=p.pid)
    sampler.start()
    rows = []
    req_error = False
    try:
        for i in range(reps + 1):          # rep 0 = warm-up, discarded
            tag = f"r{i}"
            try:
                row = fire(prompt, tag)
            except Exception as e:
                print(f"  {tag}: REQUEST FAILED {e}", flush=True)
                rows.append({"tag": tag, "error": str(e), "warmup": i == 0})
                req_error = True
                continue
            row["warmup"] = (i == 0)
            rows.append(row)
            print("  {tag:<6} n={predicted_n:<4} {decode_tps:>7.3f} t/s "
                  "prefill={prompt_tps:>8.2f} t/s draft {draft_n}/{draft_n_accepted} "
                  "acc={acc} finish={finish} reason_len={reasoning_len}".format(
                      acc=("%.4f" % row["acceptance"]) if row["acceptance"] is not None else "-",
                      **{k: (v if v is not None else 0) for k, v in row.items()}),
                  flush=True)
    finally:
        sampler.stop_flag.set()
        sampler.join(timeout=5)
        stop_server(p, lf)

    ok = [r for r in rows if not r.get("warmup") and r.get("decode_tps")]
    agg = {
        "name": name,
        "status": "REQ_FAIL" if req_error else "OK",
        "argv": cmd,
        "ctx": ctx, "kv": kv, "extra": extra,
        "prewarm_s": round(prewarm_s, 2),
        "load_time_s": round(load_s, 2),
        "vram_at_load_mib": vram_at_load,
        "vram_peak_mib": sampler.peak,
        "rss_peak_mib": round(sampler.rss_peak, 1) if sampler.rss_peak else None,
        "sys_used_peak_mib": round(sampler.sys_used_peak, 1) if sampler.sys_used_peak else None,
        "sm_clock_mean": round(mean(sampler.clocks), 1) if sampler.clocks else None,
        "sm_clock_min": min(sampler.clocks) if sampler.clocks else None,
        "temp_max_c": max(sampler.temps) if sampler.temps else None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_measured": len(ok),
        "memcap_note": memcap_note,
        "rows": rows,
    }
    if ok:
        agg["decode_tps"] = [r["decode_tps"] for r in ok]
        agg["decode_tps_mean"] = mean(agg["decode_tps"])
        agg["prefill_tps"] = [r["prompt_tps"] for r in ok]
        agg["prefill_tps_mean"] = mean(agg["prefill_tps"])
        agg["predicted_n_mean"] = mean([r["predicted_n"] for r in ok])
        agg["prompt_n_mean"] = mean([r["prompt_n"] for r in ok])
        dn = sum((r["draft_n"] or 0) for r in ok)
        da = sum((r["draft_n_accepted"] or 0) for r in ok)
        agg["acceptance_pooled"] = (da / dn) if dn else None
        agg["reasoning_len_max"] = max(r["reasoning_len"] for r in ok)
        agg["think_leak"] = any(r["think_tag_in_content"] for r in ok) or \
                            agg["reasoning_len_max"] > 0
        agg.update(parse_server_log(logpath, keep=reps))
        print(f"  -> {agg['decode_tps_mean']:.3f} t/s decode (mean of {len(ok)}), "
              f"prefill {agg['prefill_tps_mean']:.2f} t/s, "
              f"acc={agg['acceptance_pooled']}, vram_peak={sampler.peak} MiB, "
              f"rss_peak={agg['rss_peak_mib']} MiB, sys_used_peak={agg['sys_used_peak_mib']} MiB, "
              f"think_leak={agg['think_leak']}", flush=True)
    return agg


# ---------------------------------------------------------------- modes
def run_placement():
    outf = f"{OUTDIR}/placement.json"
    results = json.load(open(outf)) if os.path.exists(outf) else {}
    # block_count = 41 (40 model layers + the blk.40 nextn/MTP layer), so 41 is
    # the true all-experts-on-CPU baseline; descend until the card OOMs.
    ncms = [41, 36, 32, 28, 26, 24, 22, 20]
    specs = ["nospec", "mtp2", "mtp4"]
    spec_arg = {"nospec": "none", "mtp2": "mtp2", "mtp4": "mtp4"}
    for ncm in ncms:
        for suffix in specs:
            name = f"ncm{ncm}_{suffix}"
            if results.get(name, {}).get("status") == "OK":
                print(f"skip {name} (done)", flush=True)
                continue
            extra = ["--n-cpu-moe", str(ncm)] + spec_extra(spec_arg[suffix])
            agg = run_cell(name, 16384, "q8_0", extra, CODE_PROMPT)
            results[name] = agg
            json.dump(results, open(outf, "w"), indent=1)
    print("\nPLACEMENT_DONE", flush=True)


def run_depth(args):
    outf = f"{OUTDIR}/depth.json"
    results = json.load(open(outf)) if os.path.exists(outf) else {}
    name = f"depth_ncm{args.ncm}_{args.spec}"
    extra = ["--n-cpu-moe", str(args.ncm)] + spec_extra(args.spec)
    agg = run_cell(name, 16384, "q8_0", extra, depth_prompt())
    results[name] = agg
    json.dump(results, open(outf, "w"), indent=1)
    print("\nDEPTH_DONE", flush=True)


def run_cell_mode(args):
    outf = f"{OUTDIR}/cells.json"
    results = json.load(open(outf)) if os.path.exists(outf) else []
    extra = ["--n-cpu-moe", str(args.ncm)] + spec_extra(args.spec)
    agg = run_cell(args.name, args.ctx, args.kv, extra, CODE_PROMPT,
                   memcap_note=args.memcap_note)
    results.append(agg)
    json.dump(results, open(outf, "w"), indent=1)
    print("\nCELL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    sub.add_parser("placement", help="the --n-cpu-moe x spec-type grid, ctx=16384, q8_0 KV")

    p_depth = sub.add_parser("depth", help="one named config decoded at ~14k depth")
    p_depth.add_argument("--ncm", type=int, required=True, help="--n-cpu-moe value")
    p_depth.add_argument("--spec", choices=["none", "mtp2", "mtp4"], required=True)

    p_cell = sub.add_parser("cell", help="one ad-hoc config")
    p_cell.add_argument("--name", required=True)
    p_cell.add_argument("--ncm", type=int, required=True, help="--n-cpu-moe value")
    p_cell.add_argument("--spec", choices=["none", "mtp2", "mtp4"], required=True)
    p_cell.add_argument("--ctx", type=int, default=16384)
    p_cell.add_argument("--kv", choices=["q8_0", "f16"], default="q8_0")
    p_cell.add_argument("--memcap-note", default=None)

    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.mode == "placement":
        run_placement()
    elif args.mode == "depth":
        run_depth(args)
    elif args.mode == "cell":
        run_cell_mode(args)
    else:
        sys.exit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""STAGE 3 (ii) — host-vs-GPU split of the speculative round, by profile.

The residual we are chasing (~24 ms of every 74.45 ms round) is either GPU
work we have not attributed (the MTP head's own forwards, the widened verify)
or it is host time: graph building, launch latency, sync points, the
acceptance loop, KV rollback. Only a profile can tell those apart.

Method: run llama-server under Nsight Systems with a 30 s capture window
placed entirely inside a long steady-state MTP decode, then reconstruct the
GPU timeline. Everything in the window that is NOT covered by a CUDA kernel or
memcpy is time the GPU spent idle waiting on the host, and the round rate is
known from the server's own timings, so:

    host share per round = (window - GPU busy) / rounds in window

The same trace also gives the gap histogram, which says whether the host time
is one big stall per round (a sync) or thousands of small launch gaps.

usage: nsys_round.py [nospec|mtp2|both]
"""
import json, os, re, signal, subprocess, sys, time, threading, urllib.request

REPO = "/data/projects/q27b_on_12gb"
sys.path.insert(0, REPO)
import speed_recert as SR  # noqa: E402

NSYS = ("/data/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.1.1/"
        "target-linux-x64/nsys")
OUTDIR = f"{REPO}/results/round_budget"
PORT = 18094
URL = f"http://127.0.0.1:{PORT}"

DELAY_S = 150      # capture starts this long after the server process launches
DUR_S = 30
GEN_TOK = 2600     # ~75 s of decode at 34 t/s, so the window sits inside it


def gpu_used():
    return SR.gpu_used()


def launch(tag, spec_extra):
    os.makedirs(OUTDIR, exist_ok=True)
    SR.guard()
    rep = f"{OUTDIR}/nsys_{tag}"
    srv = [SR.BIN, "-m", SR.IQ3, "-ngl", "99", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "-c", "16384",
           "--host", "127.0.0.1", "--port", str(PORT), "--no-warmup",
           "--chat-template-file", SR.TMPL, "--parallel", "1"] + spec_extra
    # --cuda-graph-trace=node is MANDATORY here, not a nicety: llama.cpp is built
    # with GGML_CUDA_USE_GRAPHS, and nsys's default graph granularity reports a
    # whole captured graph as one entity, so the per-layer GEMVs never appear in
    # the trace. The first attempt without it reported "GPU busy 7.6%" while
    # showing only ~6,400 mul_mat_vec_q ops in 30 s of decode -- an obvious
    # capture artifact, since the timing ladder puts the GPU near saturation.
    cmd = [NSYS, "profile", "-t", "cuda", "--cuda-graph-trace", "node",
           "-o", rep, "--force-overwrite", "true",
           "--delay", str(DELAY_S), "--duration", str(DUR_S),
           "--sample", "none", "--cpuctxsw", "none", "--trace-fork-before-exec", "true",
           "--kill", "none"] + srv
    print("$ " + " ".join(cmd), flush=True)
    lf = open(f"{OUTDIR}/nsys_{tag}.serverlog", "w")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    return p, lf, rep


def wait_health(p, t_limit=1200):
    t0 = time.time()
    while time.time() - t0 < t_limit:
        if p.poll() is not None:
            return False
        try:
            urllib.request.urlopen(URL + "/health", timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


def fire(maxtok):
    body = {"messages": [{"role": "user", "content": SR.CODE_PROMPT}],
            "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": maxtok,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    d["_wall"] = time.time() - t0
    return d


def run(tag, spec_extra):
    t_launch = time.time()
    p, lf, rep = launch(tag, spec_extra)
    if not wait_health(p):
        lf.close()
        return {"tag": tag, "error": "server did not come up under nsys"}
    print(f"  healthy at t+{time.time()-t_launch:.0f}s, vram={gpu_used()} MiB", flush=True)
    fire(64)   # warm-up: builds the graphs so the window sees steady state
    # start the long generation ~12 s before the capture window opens
    sleep_to = DELAY_S - 12 - (time.time() - t_launch)
    if sleep_to > 0:
        print(f"  idling {sleep_to:.0f}s so the window lands mid-decode", flush=True)
        time.sleep(sleep_to)
    print(f"  firing {GEN_TOK}-token generation at t+{time.time()-t_launch:.0f}s", flush=True)
    d = fire(GEN_TOK)
    t = d.get("timings") or {}
    info = {"tag": tag,
            "decode_tps": t.get("predicted_per_second"),
            "predicted_n": t.get("predicted_n"),
            "predicted_ms": t.get("predicted_ms"),
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
            "wall_s": d["_wall"]}
    print(f"  {info}", flush=True)
    # nsys stops the app at the end of --duration only with --kill; we used
    # --kill none, so shut the server down ourselves and let nsys finalise.
    time.sleep(5)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=300)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    lf.close()
    # nsys re-parents the server under nsys-launcher, so killing our process
    # group is not enough -- the first run left 11.8 GiB resident and tripped the
    # next cell's GPU guard. Sweep by port, then wait for the card to drain.
    subprocess.run(["pkill", "-f", f"--port {PORT}"], capture_output=True)
    subprocess.run(["pkill", "-f", str(PORT)], capture_output=True)
    for _ in range(180):
        if gpu_used() < 500:
            break
        time.sleep(1)
    info["report"] = rep + ".nsys-rep"
    return info


def analyse(rep):
    """GPU busy vs idle inside the capture window, from the kernel/memcpy trace."""
    csv = rep.replace(".nsys-rep", "_gputrace.csv")
    r = subprocess.run([NSYS, "stats", "--report", "cuda_gpu_trace",
                        "--format", "csv", "--force-export", "true",
                        "-o", "-", rep],
                       capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        return {"error": r.stderr[-2000:]}
    open(csv, "w").write(r.stdout)
    hdr = None
    ivals = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        cols = line.split(",")
        if hdr is None:
            if cols[0].strip().strip('"').lower().startswith("start"):
                hdr = [c.strip().strip('"') for c in cols]
            continue
        try:
            st = float(cols[0])
            du = float(cols[1])
        except (ValueError, IndexError):
            continue
        ivals.append((st, st + du))
    if not ivals:
        return {"error": "no gpu trace rows"}
    ivals.sort()
    # merge overlapping intervals (streams can overlap) -> true GPU-busy time
    merged = []
    cs, ce = ivals[0]
    for s, e in ivals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    t0, t1 = merged[0][0], merged[-1][1]
    busy = sum(e - s for s, e in merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    gaps.sort()
    window_ns = t1 - t0
    idle = window_ns - busy

    def pct(p):
        return gaps[int(p * (len(gaps) - 1))] if gaps else 0.0

    return {
        "n_gpu_ops": len(ivals),
        "window_s": window_ns / 1e9,
        "gpu_busy_s": busy / 1e9,
        "gpu_idle_s": idle / 1e9,
        "gpu_busy_frac": busy / window_ns,
        "n_gaps": len(gaps),
        "total_gap_s": sum(gaps) / 1e9,
        "gap_us_mean": (sum(gaps) / len(gaps)) / 1e3 if gaps else None,
        "gap_us_p50": pct(0.50) / 1e3, "gap_us_p90": pct(0.90) / 1e3,
        "gap_us_p99": pct(0.99) / 1e3, "gap_us_max": gaps[-1] / 1e3 if gaps else None,
        # gaps big enough to be a sync/host stall rather than launch latency
        "n_gaps_over_100us": sum(1 for g in gaps if g > 100e3),
        "s_in_gaps_over_100us": sum(g for g in gaps if g > 100e3) / 1e9,
        "n_gaps_over_1ms": sum(1 for g in gaps if g > 1e6),
        "s_in_gaps_over_1ms": sum(g for g in gaps if g > 1e6) / 1e9,
        "csv": csv,
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.makedirs(OUTDIR, exist_ok=True)
    outf = f"{OUTDIR}/nsys.json"
    res = json.load(open(outf)) if os.path.exists(outf) else {}
    cells = {"mtp2": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
             "nospec": []}
    todo = list(cells) if which == "both" else [which]
    for tag in todo:
        if tag in res and not res[tag].get("error"):
            print(f"skip {tag}", flush=True)
            continue
        info = run(tag, cells[tag])
        if "report" in info and os.path.exists(info["report"]):
            info["trace"] = analyse(info["report"])
            t = info["trace"]
            if "gpu_busy_frac" in t:
                print(f"  {tag}: GPU busy {100*t['gpu_busy_frac']:.1f}% of "
                      f"{t['window_s']:.1f}s, idle {t['gpu_idle_s']:.2f}s in "
                      f"{t['n_gaps']} gaps (mean {t['gap_us_mean']:.1f} us)", flush=True)
        res[tag] = info
        json.dump(res, open(outf, "w"), indent=1)
    print("NSYS_DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turbo4 / TurboQuant KV cache A/B on the 27B flagship — adapted from
q3kxl_sweep.py (same guard(), VramSampler, start_server()/stop_server(),
fire(), parse_server_log() and JSON output shape), retargeted at the KV
cache type.

The question this answers: our KV law says quantized KV is SLOWER than f16
on this stack (dequant lands on a compute-bound attention kernel), so KV
quantization is a pure capacity trade bought at a speed cost. Turbo4
(spiritbuun/buun-llama-cpp, 4.125 bpv, PolarQuant+QJL) claims to be smaller
than q4_0. Does it also break the law by being FASTER than f16?

Three things are held fixed and must stay fixed: the model (IQ3_XXS 27B),
the speculation config (MTP n=2, except the probe mode which is nospec so
divergence is attributable to KV alone), and the binary — every speed arm
runs on the FORK binary, including the f16/q8_0 controls, so a build
difference cannot masquerade as a KV effect. `xcheck` mode measures the
fork-vs-v11 build delta separately.

Matched triple: c=12288 is the f16 ceiling on 12 GB (FINDINGS 2026-09-02:
f16 at c=16384 loads then OOMs on first request), so the only honest
3-way comparison runs all arms at c=12288. c=16384 arms are a separate
2-way (q8_0 vs turbo4) at the shipping context.
"""
import argparse, hashlib, json, os, re, signal, subprocess, sys, threading, time, urllib.request

REPO = "/data/projects/q27b_on_12gb"
FORK = "/home/ollama/turbo4/buun-llama-cpp/build/bin"
V11 = f"{REPO}/llama-server-v11"
MODEL_27B = f"{REPO}/models/unsloth-q27b/Qwen3.8-27B-UD-IQ3_XXS.gguf"
MODEL_35B = "/data/models/coding/q36_35ba3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
TMPL = f"{REPO}/models/templates/chat_template.jinja"
OUTDIR = f"{REPO}/results/turbo4"
PORT = 18098
URL = f"http://127.0.0.1:{PORT}"

CODE_PROMPT = (
    "Write a complete Python module implementing a persistent B-tree index on disk. "
    "Include: a Node class with serialization to fixed-size pages, insert with node "
    "splitting, search, an in-order range scan generator, a free-page list, and a "
    "crash-safe write path using a write-ahead log. Full type hints, docstrings, and "
    "a __main__ demo that inserts 1000 keys and verifies them. Do not abbreviate."
)
MAXTOK = 400


def depth_prompt(n_blocks):
    """q3kxl_sweep.depth_prompt, verbatim, with the block count exposed so the
    same generator can fill a c=12288 window (n=42, ~9.5K tok) and a c=16384
    window (n=62, ~14K tok)."""
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


# 10 greedy probes for the output-neutrality check. Short, deterministic,
# and varied enough that a KV codec's error would show up somewhere.
PROBES = [
    "Write a Python function that merges two sorted lists into one sorted list. Code only.",
    "Explain in exactly three sentences why binary search requires a sorted array.",
    "What is 17 * 23 + 456? Show the arithmetic step by step.",
    "Write a SQL query that returns the second-highest salary from a table `emp(id, salary)`.",
    "List the first 12 prime numbers, comma separated, nothing else.",
    "Write a Rust function `fn rev(s: &str) -> String` that reverses a string by grapheme-free chars.",
    "Summarise the difference between a process and a thread in one paragraph.",
    "Write a regex that matches an RFC-5322-ish email address and explain each part.",
    "Implement quicksort in C with an in-place Lomuto partition. Code only.",
    "Name the seven layers of the OSI model in order, one per line, nothing else.",
]


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
    def __init__(self, pid=None):
        super().__init__(daemon=True)
        self.pid, self.peak, self.rss_peak = pid, 0, 0.0
        self.clocks, self.temps = [], []
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,clocks.sm,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20)
                p = [x.strip() for x in o.stdout.strip().splitlines()[0].split(",")]
                mem = int(p[0])
                if mem > self.peak:
                    self.peak = mem
                self.clocks.append(int(p[1]))
                self.temps.append(int(p[2]))
            except Exception:
                pass
            if self.pid is not None:
                r = read_rss_mib(self.pid)
                if r is not None and r > self.rss_peak:
                    self.rss_peak = r
            self.stop_flag.wait(1.0)


def prewarm(model):
    t0 = time.time()
    try:
        subprocess.run(["dd", f"if={model}", "of=/dev/null", "bs=64M"],
                       capture_output=True, timeout=1800)
    except Exception as e:
        print(f"  prewarm failed: {e}", flush=True)
    return time.time() - t0


def post(body, timeout=1800):
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tail_log(path, n):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return ""


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
            txt = open(logpath, errors="replace").read()
            m = re.search(r"(out of memory|failed to allocate|unknown argument\S*|"
                          r"invalid|failed to load|unsupported|not supported)[^\n]*",
                          txt, re.I)
            return None, lf, f"FAIL:{m.group(0)[:140] if m else 'unknown'}", time.time() - t0
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


def fire(prompt, tag, maxtok=MAXTOK):
    body = {"messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": maxtok,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    w0 = time.time()
    r = post(body)
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    t = r.get("timings") or {}
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    return {"tag": tag, "prompt_n": t.get("prompt_n"),
            "prompt_tps": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"),
            "decode_tps": t.get("predicted_per_second"),
            "draft_n": dn, "draft_n_accepted": da,
            "acceptance": (da / dn) if (dn and da is not None) else None,
            "wall_s": round(time.time() - w0, 3), "finish": ch.get("finish_reason"),
            "reasoning_len": len(reasoning),
            "think_tag_in_content": "<think>" in content,
            "content_sha1": hashlib.sha1(content.encode()).hexdigest()[:16],
            "content_head": content[:80], "content": content}


def build_cmd(bin_dir, model, ctx, kv, extra, jinja=False):
    # -ctxcp 0: the fork's base carries upstream PR#15293 context checkpoints,
    # default 32 per slot at ~150 MiB each on this model. On a 12 GB card
    # already at ~11.3-11.7 GiB the FIRST checkpoint OOMs cudaGraphInstantiate
    # on request #2 (observed: every arm died after r0). Off for every arm,
    # including the f16/q8_0 controls, so it is a constant. It is also why
    # fork VRAM numbers are not directly comparable to the v11 lineage --
    # see the xcheck cell.
    cmd = [f"{bin_dir}/llama-server", "-m", model, "-ngl", "99", "-fa", "on",
           "-ctk", kv, "-ctv", kv, "-c", str(ctx),
           "--host", "127.0.0.1", "--port", str(PORT), "--no-warmup",
           "-ctxcp", "0", "--parallel", "1"]
    cmd += ["--jinja"] if jinja else ["--chat-template-file", TMPL]
    return cmd + list(extra)


def spec_extra(spec):
    return {"none": ["--spec-type", "none"],
            "mtp2": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
            "mtp4": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]}[spec]


def run_cell(name, bin_dir, model, ctx, kv, extra, prompt, reps=3,
             keep_content=False, jinja=False, maxtok=MAXTOK):
    """1 discarded warm-up + `reps` measured requests. Never raises."""
    logpath = f"{OUTDIR}/{name}.serverlog"
    cmd = build_cmd(bin_dir, model, ctx, kv, extra, jinja)
    pw = prewarm(model)
    p, lf, err, load_s = start_server(name, cmd, logpath)
    if err:
        print(f"  {err}", flush=True)
        lt = tail_log(logpath, 30)
        if p:
            stop_server(p, lf)
        else:
            for _ in range(60):
                if gpu_used() < 500:
                    break
                time.sleep(1)
        return {"name": name, "status": "FAIL", "argv": cmd, "ctx": ctx, "kv": kv,
                "bin": bin_dir, "error": err, "log_tail": lt,
                "prewarm_s": round(pw, 2), "load_time_s": round(load_s, 2)}

    vram_at_load = gpu_used()
    s = VramSampler(pid=p.pid); s.start()
    rows, req_error = [], False
    try:
        for i in range(reps + 1):
            try:
                row = fire(prompt, f"r{i}", maxtok)
            except Exception as e:
                print(f"  r{i}: REQUEST FAILED {e}", flush=True)
                rows.append({"tag": f"r{i}", "error": str(e), "warmup": i == 0})
                req_error = True
                continue
            row["warmup"] = (i == 0)
            if not keep_content:
                row.pop("content", None)
            rows.append(row)
            print(f"  r{i} n={row['predicted_n']} {row['decode_tps']:.3f} t/s "
                  f"prefill={row['prompt_tps']:.2f} acc="
                  f"{('%.4f' % row['acceptance']) if row['acceptance'] is not None else '-'} "
                  f"sha={row['content_sha1']}", flush=True)
    finally:
        s.stop_flag.set(); s.join(timeout=5)
        stop_server(p, lf)

    ok = [r for r in rows if not r.get("warmup") and r.get("decode_tps")]
    dn = sum((r["draft_n"] or 0) for r in ok)
    da = sum((r["draft_n_accepted"] or 0) for r in ok)
    agg = {"name": name, "status": "REQ_FAIL" if req_error else "OK", "argv": cmd,
           "bin": bin_dir, "model": model, "ctx": ctx, "kv": kv, "extra": extra,
           "prewarm_s": round(pw, 2), "load_time_s": round(load_s, 2),
           "vram_at_load_mib": vram_at_load, "vram_peak_mib": s.peak,
           "rss_peak_mib": round(s.rss_peak, 1),
           "sm_clock_mean": round(mean(s.clocks), 1) if s.clocks else None,
           "temp_max_c": max(s.temps) if s.temps else None,
           "n_measured": len(ok), "rows": rows,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if ok:
        agg["decode_tps"] = [r["decode_tps"] for r in ok]
        agg["decode_tps_mean"] = mean(agg["decode_tps"])
        agg["prefill_tps_mean"] = mean([r["prompt_tps"] for r in ok])
        agg["prompt_n_mean"] = mean([r["prompt_n"] for r in ok])
        agg["predicted_n_mean"] = mean([r["predicted_n"] for r in ok])
        agg["acceptance_pooled"] = (da / dn) if dn else None
        agg["think_leak"] = any(r["think_tag_in_content"] for r in ok) or \
                            max(r["reasoning_len"] for r in ok) > 0
        print(f"  -> {agg['decode_tps_mean']:.3f} t/s decode, "
              f"prefill {agg['prefill_tps_mean']:.2f}, "
              f"acc={agg['acceptance_pooled']}, vram_peak={s.peak} MiB", flush=True)
    return agg


# ---------------------------------------------------------------- modes
def mode_speed(args):
    """The A/B grid. Every arm on the FORK binary so the build is a constant."""
    outf = f"{OUTDIR}/speed.json"
    res = json.load(open(outf)) if os.path.exists(outf) else {}
    # (arm, ctx, kv, prompt_blocks) -- 42 blocks ~9.5K tok fits c=12288 with
    # 400 output tokens; 62 blocks ~14K fits c=16384.
    grid = [
        # matched triple at the f16 ceiling
        ("m12_f16",    12288, "f16",    None), ("m12_f16_d",    12288, "f16",    42),
        ("m12_q8_0",   12288, "q8_0",   None), ("m12_q8_0_d",   12288, "q8_0",   42),
        ("m12_turbo4", 12288, "turbo4", None), ("m12_turbo4_d", 12288, "turbo4", 42),
        # shipping context, 2-way
        ("s16_q8_0",   16384, "q8_0",   None), ("s16_q8_0_d",   16384, "q8_0",   62),
        ("s16_turbo4", 16384, "turbo4", None), ("s16_turbo4_d", 16384, "turbo4", 62),
        # the capacity claim: turbo4 is smaller than q4_0 -- does the 12 GB card
        # take a context that q8_0 cannot?
        ("s16_q4_0",   16384, "q4_0",   None),
        ("cap_f16_16k",   16384, "f16",    None),
        ("cap_turbo4_32k", 32768, "turbo4", None),
        ("cap_q8_0_32k",   32768, "q8_0",   None),
        # f16 does not survive c=12288 on THIS binary (it did on v11 -- the
        # fork's per-context overhead is higher), so the honest 3-way runs at
        # c=8192 where all three sit clear of the 12,044 MiB wall. 30 blocks
        # ~= 6.5K tok, leaving room for 400 output tokens.
        ("m8_f16",     8192, "f16",    None), ("m8_f16_d",     8192, "f16",    30),
        ("m8_q8_0",    8192, "q8_0",   None), ("m8_q8_0_d",    8192, "q8_0",   30),
        ("m8_turbo4",  8192, "turbo4", None), ("m8_turbo4_d",  8192, "turbo4", 30),
        # where does f16 actually die on this binary?
        ("cap_f16_10k", 10240, "f16", None),
    ]
    for name, ctx, kv, blocks in grid:
        if args.only and name not in args.only.split(","):
            continue
        if res.get(name, {}).get("status") == "OK":
            print(f"skip {name} (done)", flush=True)
            continue
        prompt = CODE_PROMPT if blocks is None else depth_prompt(blocks)
        res[name] = run_cell(name, FORK, MODEL_27B, ctx, kv,
                             spec_extra("mtp2"), prompt, reps=args.reps)
        json.dump(res, open(outf, "w"), indent=1)
    print("\nSPEED_DONE", flush=True)


def mode_xcheck(args):
    """Fork binary vs the v11 speed-lineage binary at the identical shipping
    config, so any fork-wide build delta is separated from the KV effect."""
    outf = f"{OUTDIR}/xcheck.json"
    res = json.load(open(outf)) if os.path.exists(outf) else {}
    for tag, bd in (("v11", V11), ("fork", FORK)):
        name = f"xcheck_{tag}_q8_16k"
        if res.get(name, {}).get("status") == "OK":
            continue
        res[name] = run_cell(name, bd, MODEL_27B, 16384, "q8_0",
                             spec_extra("mtp2"), CODE_PROMPT, reps=args.reps)
        json.dump(res, open(outf, "w"), indent=1)
    print("\nXCHECK_DONE", flush=True)


def mode_probes(args):
    """10 greedy probes per KV type, speculation OFF.

    nospec is deliberate: FINDINGS 2026-09-03 established that draft-mtp is
    NOT bit-exact on this model (batch-shape FP order), so with MTP on a
    divergence could not be attributed to the KV codec. With --spec-type none
    the control is 5/5 reproducible, so any divergence here IS the KV type.
    """
    outf = f"{OUTDIR}/probes.json"
    res = json.load(open(outf)) if os.path.exists(outf) else {}
    for kv in ["f16", "q8_0", "turbo4", "q4_0"]:
        if kv in res:
            print(f"skip probes {kv} (done)", flush=True)
            continue
        name = f"probe_{kv}"
        logpath = f"{OUTDIR}/{name}.serverlog"
        # c=8192: the f16 arm is the reference, and f16 does not survive
        # c>8192 on this binary (see the cap_f16_* cells).
        cmd = build_cmd(FORK, MODEL_27B, 8192, kv, spec_extra("none"))
        prewarm(MODEL_27B)
        p, lf, err, load_s = start_server(name, cmd, logpath)
        if err:
            res[kv] = {"status": "FAIL", "error": err, "log_tail": tail_log(logpath, 30)}
            if p:
                stop_server(p, lf)
            json.dump(res, open(outf, "w"), indent=1)
            continue
        outs = []
        try:
            for i, pr in enumerate(PROBES):
                try:
                    r = fire(pr, f"p{i}", maxtok=256)
                    outs.append({"i": i, "prompt": pr, "sha1": r["content_sha1"],
                                 "n": r["predicted_n"], "content": r["content"]})
                    print(f"  p{i} sha={r['content_sha1']} n={r['predicted_n']}", flush=True)
                except Exception as e:
                    outs.append({"i": i, "prompt": pr, "error": str(e)})
        finally:
            stop_server(p, lf)
        res[kv] = {"status": "OK", "kv": kv, "argv": cmd, "probes": outs}
        json.dump(res, open(outf, "w"), indent=1)
    print("\nPROBES_DONE", flush=True)


def mode_moe(args):
    """One 35B-A3B cell: the current best ship (ncm16, mtp2, Q3_K_XL)."""
    outf = f"{OUTDIR}/moe.json"
    res = json.load(open(outf)) if os.path.exists(outf) else {}
    for kv in args.moe_kv.split(","):
        name = f"moe_ncm16_mtp2_{kv}"
        if res.get(name, {}).get("status") == "OK":
            continue
        res[name] = run_cell(name, FORK, MODEL_35B, 16384, kv,
                             ["--n-cpu-moe", "16"] + spec_extra("mtp2"),
                             CODE_PROMPT, reps=args.reps, jinja=True)
        json.dump(res, open(outf, "w"), indent=1)
    print("\nMOE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["speed", "xcheck", "probes", "moe"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma list of cell names (speed mode)")
    ap.add_argument("--moe-kv", default="q8_0,turbo4")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    {"speed": mode_speed, "xcheck": mode_xcheck,
     "probes": mode_probes, "moe": mode_moe}[args.mode](args)


if __name__ == "__main__":
    main()

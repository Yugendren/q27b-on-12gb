#!/usr/bin/env python3
"""STAGE 1 — STACK TEST: stock-vs-full-stack on identical work.

The product demo number. Two servers, the SAME 12-task workload, measured
end to end.

  Arm A  "what a normal user gets" — ollama-style defaults: the Q4_K_S GGUF
         (15.4 GB, so it cannot fit on a 12 GB card and the runtime splits it
         host/device), c=8192, f16 KV, NO speculation, thinking left ON (the
         served Qwen3 template's default, which is what a user who does not
         know about chat_template_kwargs gets).
  Arm B  "our stack" — IQ3_XXS + the kernel-offensive libs (LD_LIBRARY_PATH
         preload of the patched ggml/llama .so under the stock server binary)
         + MTP n=2 + q8_0 KV + c=16384 + thinking OFF + --cache-ram 0.
  Arm B+ Arm B, plus the session save/restore patch: the second run of the
         same long prompt restores the slot instead of re-prefilling.

Protocol rules inherited from results/speed_recert.md and
results/platform_arm/VERDICT.md:
  * GPU guard: refuse to launch unless the card is under 500 MiB.
  * VRAM is certified from the PEAK OF A 1 Hz SAMPLE TAKEN DURING REQUESTS,
    never from load (three configs in the recert passed a load check and then
    OOMed inside ggml_cuda_pool_vmm::alloc on the first request).
  * Thinking tripwire: every request records reasoning_content length and a
    <think> scan; a HumanEval-style mean completion above 350 tok on an arm
    that is supposed to have thinking OFF means the switch is not taking
    effect -> stop.
  * Report tokens-per-TASK and wall-clock alongside decode t/s: the recert
    produced a protocol where raw t/s and effective speed move in opposite
    directions.

HumanEval scoring (extract_code / build_program / run_program) is imported
verbatim from the battery harness so the pass rates are on the same footing
as results/platform_arm/*.

usage: stack_test.py A|B|Bplus|all
"""
import hashlib, json, os, re, signal, subprocess, sys, threading, time, urllib.request

REPO = "/data/projects/q27b_on_12gb"
sys.path.insert(0, f"{REPO}/platform_arm/harness")
from reliability import (extract_code, build_program, run_program,   # noqa: E402
                         fence_unclosed, load_humaneval, HUMANEVAL_SYSTEM)

M       = f"{REPO}/models/unsloth-q27b"
Q4KS    = f"{M}/Qwen3.8-27B-UD-Q4_K_S.gguf"
IQ3     = f"{M}/Qwen3.8-27B-UD-IQ3_XXS.gguf"
BIN     = "/data/projects/llama.cpp/build/bin/llama-server"
BIN_SR  = "/data/scratch/llamacpp-ckptfix/build/bin/llama-server"   # session-restore patch
PATCHLIB = "/data/scratch/kernel_offensive/proto/llama.cpp/build/bin"
OUTDIR  = f"{REPO}/results/stack_test"
DATADIR = f"{REPO}/platform_arm/harness/data"
PORT    = 18088
URL     = f"http://127.0.0.1:{PORT}"

N_HE      = 10       # HumanEval/0 .. HumanEval/9
HE_MAXTOK = 1024     # the legacy budget, so a thinking-ON arm is not
                     # artificially truncated -- it is allowed to spend it
LC_MAXTOK = 400


# ------------------------------------------------------------------ workload
def _repo_dump(n_blocks, seed_tag):
    """A synthetic repository dump in the speed_recert.depth_prompt style, with
    machine-checkable needles planted in it.

    Each module carries a RETENTION_LIMIT, an AUDIT_CODE and a DEPENDS_ON, so a
    single prompt supports both a direct-retrieval question and a two-hop one.
    Values are derived from the index, so the checker never has to trust a
    stored copy of the answer."""
    out = []
    for i in range(n_blocks):
        out.append(
            f"# --- {seed_tag}/module_{i:03d}.py ---\n"
            f"RETENTION_LIMIT = {7000 + 13 * i}\n"
            f"AUDIT_CODE = \"AC-{seed_tag.upper()}-{(i * 37 + 11) % 997:03d}\"\n"
            f"DEPENDS_ON = \"module_{(i * 17 + 5) % n_blocks:03d}\"\n"
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
    return "Below is a dump of a Python repository.\n\n" + "\n".join(out) + "\n\n"


def lc_tasks():
    """Two long-context tasks.

    lc1 is sized (measured with llama-tokenize: 7,705 tok) so that it FITS
    inside Arm A's 8192-token window with the 400-token answer budget -- i.e.
    Arm A gets a genuine shot at it. lc2 (12,810 tok) does not fit in 8192 at
    all; that is a capability difference and it is reported as one, not hidden."""
    n1, n2 = 30, 50
    d1 = _repo_dump(n1, "alpha")
    d2 = _repo_dump(n2, "beta")

    q1_i = 21
    q1 = (f"Read the repository dump above. Report two facts about "
          f"module_{q1_i:03d}: its RETENTION_LIMIT and its AUDIT_CODE. "
          f"Answer with exactly one line, in the form\n"
          f"ANSWER: <retention_limit>|<audit_code>\n"
          f"and nothing else.")
    a1 = f"{7000 + 13 * q1_i}|AC-ALPHA-{(q1_i * 37 + 11) % 997:03d}"

    q2_i = 40
    dep = (q2_i * 17 + 5) % n2
    q2 = (f"Read the repository dump above. module_{q2_i:03d} names another "
          f"module in its DEPENDS_ON field. Follow that reference and report "
          f"the AUDIT_CODE and the RETENTION_LIMIT of the module it points to. "
          f"Answer with exactly one line, in the form\n"
          f"ANSWER: <audit_code>|<retention_limit>\n"
          f"and nothing else.")
    a2 = f"AC-BETA-{(dep * 37 + 11) % 997:03d}|{7000 + 13 * dep}"

    return [
        {"task_id": "LC/1_retrieve", "prompt": d1 + q1, "answer": a1, "maxtok": LC_MAXTOK},
        {"task_id": "LC/2_multihop", "prompt": d2 + q2, "answer": a2, "maxtok": LC_MAXTOK},
    ]


ANSWER_RE = re.compile(r"ANSWER:\s*([^\n]+)")


def score_lc(text, expected):
    m = ANSWER_RE.search(text or "")
    if not m:
        return False, "no ANSWER line"
    got = m.group(1).strip().strip("`").strip()
    return (got == expected), got


# ------------------------------------------------------------------- plumbing
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


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.peak, self.clocks, self.temps = 0, [], []

    def run(self):
        while not self.stop_flag.is_set():
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,clocks.sm,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
                mem, clk, tmp = [int(x.strip()) for x in o.split(",")]
                self.peak = max(self.peak, mem)
                self.clocks.append(clk)
                self.temps.append(tmp)
            except Exception:
                pass
            self.stop_flag.wait(1.0)


def start_server(name, cmd, env_extra, logpath):
    guard()
    env = dict(os.environ)
    env.update(env_extra or {})
    print(f"\n=== {name} ===\n$ {' '.join(cmd)}\n  env {env_extra}", flush=True)
    lf = open(logpath, "w")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         env=env, cwd=REPO)
    t0 = time.time()
    while time.time() - t0 < 2400:
        if p.poll() is not None:
            lf.close()
            txt = open(logpath, errors="replace").read()
            m = re.search(r"(out of memory|failed to allocate|unknown argument\S*|"
                          r"invalid|failed to load|unsupported)[^\n]*", txt, re.I)
            return None, lf, "FAIL:" + (m.group(0)[:150] if m else "unknown")
        try:
            urllib.request.urlopen(URL + "/health", timeout=5)
            print(f"  healthy in {time.time()-t0:.0f}s, vram_at_load={gpu_used()} MiB", flush=True)
            return p, lf, None
        except Exception:
            time.sleep(2)
    return p, lf, "FAIL:health_timeout"


def stop_server(p, lf):
    if p is not None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=180)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
    if lf:
        lf.close()
    for _ in range(180):
        if gpu_used() < 500:
            break
        time.sleep(1)
    time.sleep(3)


def post(body, timeout=3600):
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fire(messages, maxtok, thinking, cache_prompt=False):
    body = {"messages": messages, "temperature": 0, "top_k": 1, "seed": 1234,
            "max_tokens": maxtok, "cache_prompt": cache_prompt}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    w0 = time.time()
    err = None
    try:
        r = post(body)
    except Exception as e:
        return {"wall_s": round(time.time() - w0, 3), "error": str(e)[:300]}, "", err
    wall = time.time() - w0
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    t = r.get("timings") or {}
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    rec = {
        "wall_s": round(wall, 3),
        "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
        "decode_tps": t.get("predicted_per_second"),
        "draft_n": dn, "draft_n_accepted": da,
        "acceptance": (da / dn) if (dn and da is not None) else None,
        "finish": ch.get("finish_reason"),
        "reasoning_len": len(reasoning),
        "think_tag_in_content": "<think>" in content,
        "content_sha1": hashlib.sha1(content.encode()).hexdigest()[:16],
    }
    return rec, content, reasoning


# ---------------------------------------------------------------- arm runners
def run_workload(arm, thinking, he_items, lcs):
    rows = []
    for it in he_items:
        rec, content, _ = fire(
            [{"role": "system", "content": HUMANEVAL_SYSTEM},
             {"role": "user", "content": it["prompt"]}],
            HE_MAXTOK, thinking)
        if "error" in rec:
            rec.update({"task_id": it["task_id"], "kind": "humaneval", "passed": False})
            rows.append(rec)
            print(f"  {it['task_id']:<16} REQUEST FAILED {rec['error']}", flush=True)
            continue
        code = extract_code(content)
        prog = build_program(it["prompt"], code, it["entry_point"], it["test"])
        ok, errtxt = run_program(prog)
        rec.update({"task_id": it["task_id"], "kind": "humaneval", "passed": ok,
                    "fence_unclosed": fence_unclosed(content),
                    "error": (errtxt or "")[:200] if not ok else None,
                    "response": content[:4000]})
        rows.append(rec)
        print(f"  {it['task_id']:<16} pass={ok} n={rec['predicted_n']} "
              f"{rec['wall_s']:.1f}s {(rec['decode_tps'] or 0):.2f} t/s "
              f"reason_len={rec['reasoning_len']}", flush=True)

    for lc in lcs:
        rec, content, _ = fire([{"role": "user", "content": lc["prompt"]}],
                               lc["maxtok"], thinking)
        if "error" in rec:
            rec.update({"task_id": lc["task_id"], "kind": "longctx", "passed": False})
            rows.append(rec)
            print(f"  {lc['task_id']:<16} REQUEST FAILED {rec['error']}", flush=True)
            continue
        ok, got = score_lc(content, lc["answer"])
        rec.update({"task_id": lc["task_id"], "kind": "longctx", "passed": ok,
                    "expected": lc["answer"], "got": got, "response": content[:2000]})
        rows.append(rec)
        print(f"  {lc['task_id']:<16} pass={ok} got={got!r} exp={lc['answer']!r} "
              f"{rec['wall_s']:.1f}s prompt_n={rec['prompt_n']} "
              f"reason_len={rec['reasoning_len']}", flush=True)
    return rows


def summarise(arm, rows, sampler, vram_at_load, wall_total, extra=None):
    he = [r for r in rows if r.get("kind") == "humaneval"]
    lc = [r for r in rows if r.get("kind") == "longctx"]
    ok_he = [r for r in he if r.get("predicted_n")]
    tot_n = sum((r.get("predicted_n") or 0) for r in rows)
    tot_ms = sum((r.get("predicted_ms") or 0) for r in rows)
    s = {
        "arm": arm,
        "wall_total_s": round(wall_total, 2),
        "n_tasks": len(rows),
        "he_pass": sum(1 for r in he if r.get("passed")),
        "he_n": len(he),
        "lc_pass": sum(1 for r in lc if r.get("passed")),
        "lc_n": len(lc),
        "pass_total": sum(1 for r in rows if r.get("passed")),
        "he_wall_s": round(sum(r["wall_s"] for r in he), 2),
        "lc_wall_s": round(sum(r["wall_s"] for r in lc), 2),
        "he_mean_wall_s": round(sum(r["wall_s"] for r in he) / len(he), 2) if he else None,
        "he_mean_completion_tokens": (sum((r.get("predicted_n") or 0) for r in he) / len(he)
                                      if he else None),
        "decode_tps_pooled": (1000.0 * tot_n / tot_ms) if tot_ms else None,
        "decode_tps_he_pooled": ((1000.0 * sum(r["predicted_n"] for r in ok_he) /
                                  sum(r["predicted_ms"] for r in ok_he)) if ok_he else None),
        "effective_tps": (tot_n / wall_total) if wall_total else None,
        "total_completion_tokens": tot_n,
        "vram_at_load_mib": vram_at_load,
        "vram_peak_mib": sampler.peak if sampler else None,
        "sm_clock_mean": (round(sum(sampler.clocks) / len(sampler.clocks), 1)
                          if sampler and sampler.clocks else None),
        "temp_max_c": max(sampler.temps) if sampler and sampler.temps else None,
        "reasoning_len_max": max((r.get("reasoning_len") or 0) for r in rows) if rows else 0,
        "acceptance_pooled": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
    }
    dn = sum((r.get("draft_n") or 0) for r in rows)
    da = sum((r.get("draft_n_accepted") or 0) for r in rows)
    if dn:
        s["acceptance_pooled"] = da / dn
    s["think_leak"] = s["reasoning_len_max"] > 0 or any(r.get("think_tag_in_content") for r in rows)
    if extra:
        s.update(extra)
    return s


def cmd_A():
    # ollama-style: no -ngl, so the runtime's own fit_params splits the 15.4 GB
    # Q4_K_S across host/device the way a stock local-inference stack does;
    # default f16 KV; no speculation; --jinja so the served template is the
    # model's own (whose thinking default is ON -- that is the point).
    return [BIN, "-m", Q4KS, "-fa", "on", "-c", "8192",
            "--host", "127.0.0.1", "--port", str(PORT), "--jinja",
            "--no-warmup", "--parallel", "1"], {}


def cmd_B(binary=BIN, cache_ram="0"):
    # The kernel-offensive libs are preloaded by LD_LIBRARY_PATH against the
    # stock binary (its RUNPATH is a RUNPATH, not an RPATH, so LD_LIBRARY_PATH
    # wins -- verified with ldd). The session-restore build must NOT get them:
    # its fix lives in ITS libllama.so, so overriding the library path would
    # silently un-patch the very thing arm B+ is measuring. The two patch sets
    # have never been merged into one build, and B+ is reported as such.
    env = {"LD_LIBRARY_PATH": PATCHLIB} if binary == BIN else {}
    return ([binary, "-m", IQ3, "-ngl", "99", "-fa", "on",
             "-ctk", "q8_0", "-ctv", "q8_0", "-c", "16384",
             "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
             "--cache-ram", cache_ram,
             "--host", "127.0.0.1", "--port", str(PORT), "--jinja",
             "--no-warmup", "--parallel", "1"], env)


def run_arm(arm):
    os.makedirs(OUTDIR, exist_ok=True)
    he_items = load_humaneval(DATADIR, N_HE)
    lcs = lc_tasks()

    if arm == "A":
        cmd, env = cmd_A()
        thinking = True
    elif arm == "B":
        cmd, env = cmd_B()
        thinking = False
    else:
        raise ValueError(arm)

    log = f"{OUTDIR}/srv_{arm}.log"
    p, lf, err = start_server(arm, cmd, env, log)
    if err:
        print(f"  {err}", flush=True)
        stop_server(p, lf)
        return {"arm": arm, "error": err}
    vram_at_load = gpu_used()
    sampler = VramSampler(); sampler.start()
    t0 = time.time()
    try:
        rows = run_workload(arm, thinking, he_items, lcs)
    finally:
        wall = time.time() - t0
        sampler.stop_flag.set(); sampler.join(timeout=5)
        stop_server(p, lf)
    s = summarise(arm, rows, sampler, vram_at_load, wall)
    print(f"\n  ARM {arm}: {s['wall_total_s']}s total, pass {s['pass_total']}/{s['n_tasks']}, "
          f"decode {s['decode_tps_pooled']:.2f} t/s pooled, peak {s['vram_peak_mib']} MiB, "
          f"think_leak={s['think_leak']}", flush=True)
    return s


def run_bplus():
    """Arm B+ : session save/restore on the long prompts.

    Uses the session-restore patched server (/data/scratch/llamacpp-ckptfix).
    Sequence per long task: cold run (full prefill) -> /slots save -> restart
    is not needed; we clear the slot with a different prompt, then /slots
    restore and re-ask. The measured quantity is the second run's TTFT/prefill
    against the first's."""
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(BIN_SR):
        return {"arm": "Bplus", "skipped": "session-restore build missing"}
    lcs = lc_tasks()
    cmd, env = cmd_B(binary=BIN_SR)
    slotdir = "/data/scratch/slots_stage1"
    os.makedirs(slotdir, exist_ok=True)
    cmd = cmd + ["--slot-save-path", slotdir]
    log = f"{OUTDIR}/srv_Bplus.log"
    p, lf, err = start_server("Bplus", cmd, env, log)
    if err:
        stop_server(p, lf)
        return {"arm": "Bplus", "error": err}
    sampler = VramSampler(); sampler.start()
    out = []
    try:
        for lc in lcs:
            tag = lc["task_id"].replace("/", "_")
            msgs = [{"role": "user", "content": lc["prompt"]}]
            # 1. cold: full prefill, prompt cached into the slot
            cold, c1, _ = fire(msgs, lc["maxtok"], False, cache_prompt=True)
            # 2. persist the slot
            t0 = time.time()
            urllib.request.urlopen(urllib.request.Request(
                f"{URL}/slots/0?action=save",
                data=json.dumps({"filename": f"{tag}.bin"}).encode(),
                headers={"Content-Type": "application/json"}), timeout=600).read()
            save_s = time.time() - t0
            # 3. evict: a different prompt takes the slot over
            fire([{"role": "user", "content": "Say OK."}], 8, False, cache_prompt=True)
            # 4. restore and re-ask -- this is the run that must be cheap
            t0 = time.time()
            urllib.request.urlopen(urllib.request.Request(
                f"{URL}/slots/0?action=restore",
                data=json.dumps({"filename": f"{tag}.bin"}).encode(),
                headers={"Content-Type": "application/json"}), timeout=600).read()
            restore_s = time.time() - t0
            warm, c2, _ = fire(msgs, lc["maxtok"], False, cache_prompt=True)
            ok_c, got_c = score_lc(c1, lc["answer"])
            ok_w, got_w = score_lc(c2, lc["answer"])
            out.append({
                "task_id": lc["task_id"],
                "cold": cold, "warm": warm,
                "slot_save_s": round(save_s, 3), "slot_restore_s": round(restore_s, 3),
                "cold_pass": ok_c, "warm_pass": ok_w,
                "identical_answer": (got_c == got_w),
                "cold_sha1": cold.get("content_sha1"), "warm_sha1": warm.get("content_sha1"),
                "prefill_speedup": ((cold.get("prompt_ms") or 0) /
                                    (warm.get("prompt_ms") or 1)) if warm.get("prompt_ms") else None,
                "wall_speedup": (cold["wall_s"] / warm["wall_s"]) if warm.get("wall_s") else None,
            })
            print(f"  {lc['task_id']}: cold {cold['wall_s']:.2f}s "
                  f"(prefill {cold.get('prompt_ms', 0):.0f}ms) -> warm {warm['wall_s']:.2f}s "
                  f"(prefill {warm.get('prompt_ms', 0):.0f}ms, restore {restore_s:.2f}s) "
                  f"same_answer={got_c == got_w}", flush=True)
    finally:
        sampler.stop_flag.set(); sampler.join(timeout=5)
        stop_server(p, lf)
    return {"arm": "Bplus", "vram_peak_mib": sampler.peak, "tasks": out,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUTDIR, exist_ok=True)
    outf = f"{OUTDIR}/stack_test.json"
    res = json.load(open(outf)) if os.path.exists(outf) else {}
    todo = ["A", "B", "Bplus"] if what == "all" else [what]
    for arm in todo:
        if arm in res and not res[arm].get("error"):
            print(f"skip {arm} (done)", flush=True)
            continue
        r = run_bplus() if arm == "Bplus" else run_arm(arm)
        res[arm] = r
        json.dump(res, open(outf, "w"), indent=1)
        # standing tripwire
        if arm == "B" and r.get("think_leak"):
            sys.exit("THINKING LEAK on arm B — switch not taking effect; stopping")
        if arm == "B" and (r.get("he_mean_completion_tokens") or 0) > 350:
            sys.exit(f"TRIPWIRE: arm B mean completion "
                     f"{r['he_mean_completion_tokens']:.0f} tok > 350 — stopping")
    print("\nSTACK_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""STAGE 3 — round-overhead decomposition: where does the ~24 ms/round go?

The corrected cost model (results/speed_recert.md §6) gives
    T_base       = 50.00 ms/token          (no-spec, 20.001 t/s)
    T_round(n=2) = 74.45 ms/round          (34.388 t/s at mean_len 2.560)
and one verify forward is ~45-50 ms, so roughly 24 ms/round is draft +
host machinery. That number is currently a residual, not a measurement.
This decomposes it.

THE LEVER: llama-server has `--spec-synth-rates P0,P1,...` (benchmarking
only). Reading tools/server/server-context.cpp:3887, it replaces ONLY the
acceptance DECISION -- the draft model still runs and the verify batch is
still n+1 tokens. So it decouples alpha from n, which is exactly what the
residual needs:

    rates all 0  -> every round emits exactly 1 token, so measured
                    ms/token IS T_round(n) with the full draft+verify+host
                    cost of an n-token draft
    rates all 1  -> every round emits n+1 tokens; identical GPU work, but
                    no reject/rollback path -> isolates rollback cost

Regressing T_round(n)|alpha=0 on n then splits the residual:
    intercept - T_base = FIXED per-round host/graph/sync overhead
    slope              = MARGINAL cost of one more drafted token
                         (= one MTP head forward + the verify batch widening)

and the verify-batch widening is measured separately with llama-bench
(`-p 1..5` at the same depth = a forward at batch 1..5).

modes:
  synth   the alpha-controlled n ladder (the decomposition)
  knobs   every --spec-* / scheduling option that could shrink the overhead
  batch   llama-bench batch-width probe (verify forward vs batch size)
"""
import json, os, re, subprocess, sys, time

REPO = "/data/projects/q27b_on_12gb"
sys.path.insert(0, REPO)
import speed_recert as SR   # noqa: E402  -- reuse the certified cell runner

OUTDIR = f"{REPO}/results/round_budget"
SR.OUTDIR = OUTDIR
SR.PORT = 18093
SR.URL = f"http://127.0.0.1:{SR.PORT}"

IQ3 = SR.IQ3
ASCII = f"{SR.M}/Qwen3.8-27B-UD-IQ3_XXS-ASCII.gguf"
BENCH = "/data/projects/llama.cpp/build/bin/llama-bench"


def mtp(n, rates=None, extra=None):
    e = ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]
    if rates is not None:
        e += ["--spec-synth-rates", ",".join(str(r) for r in rates)]
    return e + (extra or [])


def synth_configs():
    """alpha-controlled ladder.

    c=16384 is the shipping window (and the one the 74.45 ms model was fitted
    at) but IQ3+MTP OOMs there past n=3, so the long lever arm for the
    regression is taken at c=8192, where the recert's n-curve also lives."""
    cfgs = [
        # --- reference cells (no synth): the numbers the model was fitted to
        ("rb_c16k_nospec",     dict(model=IQ3, kv="q8_0", ctx=16384, extra=[])),
        ("rb_c16k_real_n2",    dict(model=IQ3, kv="q8_0", ctx=16384, extra=mtp(2))),
        # --- alpha = 0 : one token per round, so ms/token == T_round(n)
        ("rb_c16k_a0_n1",      dict(model=IQ3, kv="q8_0", ctx=16384, extra=mtp(1, [0]))),
        ("rb_c16k_a0_n2",      dict(model=IQ3, kv="q8_0", ctx=16384, extra=mtp(2, [0, 0]))),
        ("rb_c16k_a0_n3",      dict(model=IQ3, kv="q8_0", ctx=16384, extra=mtp(3, [0, 0, 0]))),
        # --- alpha = 1 : same GPU work, no reject/rollback path
        ("rb_c16k_a1_n1",      dict(model=IQ3, kv="q8_0", ctx=16384, extra=mtp(1, [1]))),
        ("rb_c16k_a1_n2",      dict(model=IQ3, kv="q8_0", ctx=16384, extra=mtp(2, [1, 1]))),
        ("rb_c16k_a1_n3",      dict(model=IQ3, kv="q8_0", ctx=16384, extra=mtp(3, [1, 1, 1]))),
        # --- long lever arm at c=8192 (n=4,5 fit here)
        ("rb_c8k_nospec",      dict(model=IQ3, kv="q8_0", ctx=8192, extra=[])),
        ("rb_c8k_a0_n1",       dict(model=IQ3, kv="q8_0", ctx=8192, extra=mtp(1, [0]))),
        ("rb_c8k_a0_n2",       dict(model=IQ3, kv="q8_0", ctx=8192, extra=mtp(2, [0, 0]))),
        ("rb_c8k_a0_n3",       dict(model=IQ3, kv="q8_0", ctx=8192, extra=mtp(3, [0, 0, 0]))),
        ("rb_c8k_a0_n4",       dict(model=IQ3, kv="q8_0", ctx=8192, extra=mtp(4, [0] * 4))),
        ("rb_c8k_a0_n5",       dict(model=IQ3, kv="q8_0", ctx=8192, extra=mtp(5, [0] * 5))),
        ("rb_c8k_a1_n5",       dict(model=IQ3, kv="q8_0", ctx=8192, extra=mtp(5, [1] * 5))),
        # drift control, taken last
        ("rb_z_repeat_c16k_a0_n2", dict(model=IQ3, kv="q8_0", ctx=16384, extra=mtp(2, [0, 0]))),
    ]
    return cfgs


def knob_configs():
    """(iii) does any shipped option shrink the per-round machinery?

    Each cell is the SHIPPING config with exactly one knob moved, so the
    comparison is against rb_c16k_real_n2 and nothing else changes."""
    return [
        # draft-side sampling is offloaded to the backend by default; forcing it
        # onto the host changes the sync structure of the round
        ("rb_k_nodraftbackendsamp", dict(model=IQ3, kv="q8_0", ctx=16384,
                                         extra=mtp(2, None, ["--no-spec-draft-backend-sampling"]))),
        # target-side backend sampling is OFF by default -- turning it on removes
        # a device->host logits copy per round
        ("rb_k_backendsampling",    dict(model=IQ3, kv="q8_0", ctx=16384,
                                         extra=mtp(2, None, ["--backend-sampling"]))),
        # busy-wait instead of blocking on the draft threadpool
        ("rb_k_poll",               dict(model=IQ3, kv="q8_0", ctx=16384,
                                         extra=mtp(2, None, ["--poll", "1",
                                                             "--spec-draft-poll", "1"]))),
        # host<->device op offload heuristics
        ("rb_k_noopoffload",        dict(model=IQ3, kv="q8_0", ctx=16384,
                                         extra=mtp(2, None, ["--no-op-offload"]))),
        # the adaptive MTP implementation (different round structure). n_max=2
        # is rejected by the arg parser (its n_min_adaptive default is 3), so
        # the cell is taken at the shallowest depth it will accept.
        ("rb_k_mtpadaptive",        dict(model=IQ3, kv="q8_0", ctx=16384,
                                         extra=["--spec-type", "draft-mtp-adaptive",
                                                "--spec-draft-n-max", "3",
                                                "--spec-draft-n-min-adaptive", "1"])),
        # never abandon a draft early (p_split/p_min branches are host-side work)
        ("rb_k_pmin0",              dict(model=IQ3, kv="q8_0", ctx=16384,
                                         extra=mtp(2, None, ["--spec-draft-p-min", "0"]))),
        # graph reuse / batch shape: does a wider physical batch help the verify?
        ("rb_k_ub2048",             dict(model=IQ3, kv="q8_0", ctx=16384,
                                         extra=mtp(2, None, ["-ub", "2048"]))),
    ]


def prune_configs():
    """Directed test of the decomposition's one actionable prediction.

    If the MTP draft forward's cost is dominated by the LM head (output.weight
    is 715 MB of the IQ3_XXS file -- by far the largest single tensor a draft
    forward must read), then halving that tensor must show up as a smaller
    marginal cost per drafted token. The ASCII vocab prune does exactly that:
    output.weight 715 MB -> 368 MB (-48.5%), and stage 2 just showed it costs
    no measurable quality. Same alpha=0 lever, so the comparison is not
    confounded by the two builds' different acceptance."""
    return [
        ("rb_p_nospec", dict(model=ASCII, kv="q8_0", ctx=16384, extra=[])),
        ("rb_p_a0_n1",  dict(model=ASCII, kv="q8_0", ctx=16384, extra=mtp(1, [0]))),
        ("rb_p_a0_n2",  dict(model=ASCII, kv="q8_0", ctx=16384, extra=mtp(2, [0, 0]))),
        ("rb_p_a0_n3",  dict(model=ASCII, kv="q8_0", ctx=16384, extra=mtp(3, [0, 0, 0]))),
        ("rb_p_real_n2", dict(model=ASCII, kv="q8_0", ctx=16384, extra=mtp(2))),
    ]


def run_batch_probe():
    """Verify-forward cost as a function of batch width.

    A speculative round verifies n+1 tokens in ONE forward. llama-bench's
    `-p N -d D` measures exactly that: a batch-N forward at KV depth D. The
    n=1..6 curve gives the marginal cost of widening the verify batch, which is
    the part of the regression slope that is NOT the MTP head."""
    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/batch_probe.json"
    SR.guard()
    cmd = [BENCH, "-m", IQ3, "-ngl", "99", "-fa", "1",
           "-ctk", "q8_0", "-ctv", "q8_0",
           "-p", "1,2,3,4,5,6,8", "-n", "0", "-d", "0,4096",
           "-r", "5", "-o", "json"]
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    open(f"{OUTDIR}/batch_probe.raw", "w").write(r.stdout + "\n---STDERR---\n" + r.stderr)
    try:
        rows = json.loads(r.stdout)
    except Exception:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit("llama-bench did not return JSON")
    tab = []
    for x in rows:
        tab.append({"n_prompt": x.get("n_prompt"), "n_gen": x.get("n_gen"),
                    "n_depth": x.get("n_depth"),
                    "avg_ts": x.get("avg_ts"), "stddev_ts": x.get("stddev_ts"),
                    "ms_per_batch": (1000.0 * (x.get("n_prompt") or 0) / x["avg_ts"])
                    if x.get("avg_ts") else None})
    json.dump(tab, open(out, "w"), indent=1)
    for t in tab:
        print(f"  batch {t['n_prompt']:>2} depth {t['n_depth']:>5}: "
              f"{t['ms_per_batch']:.2f} ms/forward  ({t['avg_ts']:.1f} t/s)", flush=True)
    return tab


def run_cells(configs, outname):
    os.makedirs(OUTDIR, exist_ok=True)
    outf = f"{OUTDIR}/{outname}.json"
    results = json.load(open(outf)) if os.path.exists(outf) else []
    done = {r["config"] for r in results if not r.get("error")}
    prompts = [("code", SR.CODE_PROMPT)]
    for name, cfg in configs:
        if name in done:
            print(f"skip {name} (done)", flush=True)
            continue
        agg = SR.run_cell(name, cfg, prompts)
        results = [r for r in results if r["config"] != name] + [agg]
        json.dump(results, open(outf, "w"), indent=1)
        # synth cells produce garbage text by design, so the thinking tripwire is
        # the only correctness check that still applies to them
        if agg.get("think_leak"):
            sys.exit(f"THINKING LEAK on {name} — stopping")
    print(f"\n{outname.upper()}_DONE", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "synth"
    if mode == "synth":
        run_cells(synth_configs(), "synth")
    elif mode == "knobs":
        run_cells(knob_configs(), "knobs")
    elif mode == "prune":
        run_cells(prune_configs(), "prune")
    elif mode == "batch":
        run_batch_probe()
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

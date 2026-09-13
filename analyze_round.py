#!/usr/bin/env python3
"""STAGE 3 analysis — turn the alpha-controlled ladder into a ms-by-ms round
budget and a reducible/irreducible verdict.

Inputs:
  results/round_budget/synth.json        the alpha=0 / alpha=1 n-ladder
  results/round_budget/knobs.json        one-knob-moved cells
  results/round_budget/batch_probe.json  llama-bench forward cost vs batch width
  results/round_budget/nsys.json         GPU-busy vs host-idle inside a
                                         30 s window of steady MTP decode

Identities used:
  alpha = 0 -> exactly one accepted token per round, so the measured
               ms/token IS T_round(n).
  T_round(n) = T_verify(n+1) + n * T_draft + T_host
  no-spec    = T_verify(1)   + T_host_nospec
so a straight line through T_round(n)|alpha=0 gives
  slope     = d/dn [ T_verify(n+1) + n*T_draft ]  (marginal drafted token)
  intercept = T_verify(1) + T_host   (the n->0 limit of the speculative loop)
and intercept - T_base isolates the FIXED per-round overhead that speculation
adds even when it drafts nothing.
The batch probe measures d/dn T_verify(n+1) independently, so
  T_draft = slope - dT_verify/dn.
"""
import json, os, sys

RB = "/data/projects/q27b_on_12gb/results/round_budget"


def load(name):
    p = f"{RB}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def cell(rows, name):
    for r in rows or []:
        if r["config"] == name:
            return r
    return None


def ms_per_token(c):
    return 1000.0 / c["decode_tps_mean"] if c and c.get("decode_tps_mean") else None


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    m = sxy / sxx
    b = my - m * mx
    ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return m, b, (1 - ss_res / ss_tot if ss_tot else 1.0)


def main():
    synth = load("synth")
    knobs = load("knobs")
    batch = load("batch_probe")
    nsys = load("nsys")
    out = {}

    for ctx, pref in (("16384", "rb_c16k"), ("8192", "rb_c8k")):
        base = cell(synth, f"{pref}_nospec")
        if not base:
            continue
        tb = ms_per_token(base)
        ns, ts = [], []
        for n in range(1, 6):
            c = cell(synth, f"{pref}_a0_n{n}")
            if c and c.get("decode_tps_mean"):
                ns.append(n)
                ts.append(ms_per_token(c))
        if len(ns) < 2:
            continue
        slope, icept, r2 = linfit(ns, ts)
        blk = {
            "T_base_ms": tb,
            "T_round_alpha0_ms": dict(zip(map(str, ns), ts)),
            "fit_slope_ms_per_drafted_token": slope,
            "fit_intercept_ms": icept,
            "fit_r2": r2,
            "fixed_spec_overhead_ms": icept - tb,
        }
        # alpha = 1 cells: same GPU work, no reject/rollback
        a1 = {}
        for n in range(1, 6):
            c = cell(synth, f"{pref}_a1_n{n}")
            if c and c.get("decode_tps_mean"):
                # alpha=1 -> n+1 tokens per round
                a1[str(n)] = {"ms_per_token": ms_per_token(c),
                              "T_round_ms": ms_per_token(c) * (n + 1),
                              "mean_len": c.get("log_mean_len")}
        blk["alpha1"] = a1
        # rollback / reject cost = T_round(alpha=0) - T_round(alpha=1) at same n
        blk["reject_path_cost_ms"] = {
            k: blk["T_round_alpha0_ms"][k] - a1[k]["T_round_ms"]
            for k in a1 if k in blk["T_round_alpha0_ms"]}
        real = cell(synth, f"{pref}_real_n2")
        if real:
            blk["real_n2"] = {"decode_tps": real["decode_tps_mean"],
                              "acceptance": real.get("acceptance_pooled"),
                              "mean_len": real.get("log_mean_len"),
                              "T_round_ms": (real.get("log_mean_len") or 0) * 1000.0
                              / real["decode_tps_mean"] if real.get("log_mean_len") else None}
        out[f"ctx{ctx}"] = blk

    if batch:
        # marginal cost of widening the verify batch by one token
        for depth in sorted({b["n_depth"] for b in batch}):
            rows = sorted([b for b in batch if b["n_depth"] == depth],
                          key=lambda b: b["n_prompt"])
            xs = [b["n_prompt"] for b in rows]
            ys = [b["ms_per_batch"] for b in rows]
            m, b0, r2 = linfit(xs, ys)
            out.setdefault("verify_batch", {})[f"depth{depth}"] = {
                "ms_per_forward": dict(zip(map(str, xs), ys)),
                "marginal_ms_per_extra_token": m,
                "fixed_forward_ms": b0, "r2": r2}

    # decomposition at the shipping context
    c16 = out.get("ctx16384")
    vb = (out.get("verify_batch") or {}).get("depth0")
    if c16 and vb:
        dverify = vb["marginal_ms_per_extra_token"]
        t_draft = c16["fit_slope_ms_per_drafted_token"] - dverify
        c16["decomposition"] = {
            "d_verify_per_extra_token_ms": dverify,
            "mtp_draft_forward_ms": t_draft,
            "fixed_host_overhead_ms": c16["fixed_spec_overhead_ms"],
        }
        if c16.get("real_n2", {}).get("T_round_ms"):
            tr = c16["real_n2"]["T_round_ms"]
            c16["decomposition"]["budget_at_n2"] = {
                "T_round_measured_ms": tr,
                "verify_forward_ms": c16["T_base_ms"],
                "verify_widening_2extra_ms": 2 * dverify,
                "mtp_draft_2_forwards_ms": 2 * t_draft,
                "fixed_host_ms": c16["fixed_spec_overhead_ms"],
                "unexplained_ms": tr - (c16["T_base_ms"] + 2 * dverify + 2 * t_draft
                                        + c16["fixed_spec_overhead_ms"]),
            }

    if nsys:
        out["nsys"] = {k: {kk: vv for kk, vv in v.items() if kk != "csv"}
                       for k, v in nsys.items()}
        for tag, v in nsys.items():
            t = v.get("trace") or {}
            if "gpu_busy_frac" in t and v.get("decode_tps"):
                # rounds inside the window, from the server's own counters
                mean_len = ((v.get("draft_n_accepted") or 0) /
                            (v.get("predicted_n") or 1)) if v.get("draft_n") else 0
                tok_in_window = v["decode_tps"] * t["window_s"]
                out["nsys"][tag]["tokens_in_window"] = tok_in_window
                out["nsys"][tag]["host_idle_ms_per_token"] = \
                    1000.0 * t["gpu_idle_s"] / tok_in_window if tok_in_window else None

    if knobs:
        ref = cell(synth, "rb_c16k_real_n2")
        rt = ref["decode_tps_mean"] if ref else None
        out["knobs"] = {}
        for r in knobs:
            if r.get("error") or not r.get("decode_tps_mean"):
                out["knobs"][r["config"]] = {"error": r.get("error", "no result")}
                continue
            out["knobs"][r["config"]] = {
                "decode_tps": r["decode_tps_mean"],
                "vs_shipping_pct": (100.0 * (r["decode_tps_mean"] / rt - 1)) if rt else None,
                "acceptance": r.get("acceptance_pooled"),
                "vram_peak_mib": r.get("vram_peak_mib"),
            }

    print(json.dumps(out, indent=2))
    json.dump(out, open(f"{RB}/round_budget.json", "w"), indent=2)


if __name__ == "__main__":
    main()

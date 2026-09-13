#!/usr/bin/env python3
"""STAGE 2a — ASCII-prune re-verdict under the CORRECTED protocol.

The -4.9pt prune penalty (77.4% vs 82.3%, results/quality_pruned_full.json vs
results/quality_full_iq3xxs.json) was measured with the legacy harness, which
never sent chat_template_kwargs.enable_thinking=false, so BOTH arms were
burning their 1024-token budget on reasoning and 26/29 of the flagship's
failures were truncations (results/platform_arm/VERDICT.md). A relative
comparison between two builds is not protected against that: a build whose
reasoning runs a little longer loses tasks it can actually solve.

This re-runs the pruned build through the byte-identical code path that
produced results/platform_arm/reliability_IQ3_XXS_nospec_3060.jsonl (same
reliability.py, same greedy no-spec server profile, same -ngl 99, same
c=8192/q8_0 KV) and pairs the two per-task, so the only difference is the
vocab prune.

Paired significance = exact McNemar (binomial, two-sided), same implementation
as platform_arm/legacy_cmp.py.
"""
import json, math, sys
from collections import Counter

R = "/data/projects/q27b_on_12gb/results/platform_arm"
LEG = "/data/projects/q27b_on_12gb/results"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def load_jsonl(tag):
    d = {}
    for line in open(f"{R}/reliability_{tag}.jsonl"):
        if line.strip():
            r = json.loads(line)
            d[r["task_id"]] = r["attempts"][0]
    return d


def load_legacy(name):
    d = json.load(open(f"{LEG}/{name}.json"))["humaneval"]["results"]
    return {r["task_id"]: r for r in d}


def rate(d):
    k = sum(1 for a in d.values() if a["passed"])
    return k, len(d), 100.0 * k / len(d)


def main():
    ref = load_jsonl("IQ3_XXS_nospec_3060")            # flagship, corrected protocol
    new = load_jsonl("IQ3_XXS_ASCII_nospec_3060")      # pruned,   corrected protocol
    common = sorted(set(ref) & set(new), key=lambda t: int(t.split("/")[1]))

    kr, nr, pr = rate(ref)
    kn, nn, pn = rate(new)
    b = sum(1 for t in common if ref[t]["passed"] and not new[t]["passed"])
    c = sum(1 for t in common if not ref[t]["passed"] and new[t]["passed"])
    both = sum(1 for t in common if ref[t]["passed"] and new[t]["passed"])
    neither = len(common) - both - b - c
    p = mcnemar(b, c)

    tok_r = sum(a.get("completion_tokens") or 0 for a in ref.values()) / nr
    tok_n = sum(a.get("completion_tokens") or 0 for a in new.values()) / nn
    unc_r = sum(1 for a in ref.values() if a.get("fence_unclosed"))
    unc_n = sum(1 for a in new.values() if a.get("fence_unclosed"))

    # legacy pair, for the lineage bridge
    lref = load_legacy("quality_full_iq3xxs")
    lnew = load_legacy("quality_pruned_full")
    lcommon = sorted(set(lref) & set(lnew))
    lb = sum(1 for t in lcommon if lref[t]["passed"] and not lnew[t]["passed"])
    lc_ = sum(1 for t in lcommon if not lref[t]["passed"] and lnew[t]["passed"])
    lp = mcnemar(lb, lc_)
    lkr = sum(1 for r in lref.values() if r["passed"])
    lkn = sum(1 for r in lnew.values() if r["passed"])

    out = {
        "corrected": {
            "flagship": {"passed": kr, "n": nr, "pct": pr,
                         "mean_completion_tokens": tok_r, "fence_unclosed": unc_r},
            "pruned": {"passed": kn, "n": nn, "pct": pn,
                       "mean_completion_tokens": tok_n, "fence_unclosed": unc_n},
            "delta_pts": pn - pr,
            "mcnemar": {"both": both, "neither": neither,
                        "flagship_only_b": b, "pruned_only_c": c, "p": p},
            "n_paired": len(common),
        },
        "legacy": {
            "flagship": {"passed": lkr, "n": len(lref), "pct": 100.0 * lkr / len(lref)},
            "pruned": {"passed": lkn, "n": len(lnew), "pct": 100.0 * lkn / len(lnew)},
            "delta_pts": 100.0 * lkn / len(lnew) - 100.0 * lkr / len(lref),
            "mcnemar": {"flagship_only_b": lb, "pruned_only_c": lc_, "p": lp},
        },
    }
    # did the prune's legacy losses come back once thinking was off?
    lost_legacy = {t for t in lcommon if lref[t]["passed"] and not lnew[t]["passed"]}
    recovered = sorted(t for t in lost_legacy if t in new and new[t]["passed"])
    out["legacy_prune_losses"] = sorted(lost_legacy)
    out["legacy_losses_recovered_under_corrected"] = recovered
    out["n_legacy_losses_recovered"] = len(recovered)

    # tripwire
    out["tripwire_mean_tokens_over_350"] = {"flagship": tok_r > 350, "pruned": tok_n > 350}

    print(json.dumps(out, indent=2))
    json.dump(out, open(f"{R}/prune_reverdict.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

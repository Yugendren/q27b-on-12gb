#!/usr/bin/env python3
"""STAGE 2b — is the canary2 divergence gauge still sensitive?

canary2.py already sends chat_template_kwargs.enable_thinking=false (line
100), so unlike eval_quality.py it was never exposed to the thinking bug --
but the re-audit list in FINDINGS names the canary2 anchor references anyway,
so the runs are re-taken and the gauge is asked a sharper question than
"what number does it print":

  1. REPRODUCIBILITY. Every canary2 probe is greedy (temperature 0), so a
     re-run of the same build against the same anchor must reproduce
     bit-for-bit. If it does not, the gauge's resolution is bounded by its own
     noise and no small build-to-build gap it reports means anything.
  2. RESOLUTION. The gauge's headline separation between the pruned build and
     the flagship was 30.8 vs 28.8 divergence points. This recomputes the
     underlying per-probe exact-match vector and runs paired McNemar on it, so
     that gap is expressed in probes rather than in points.

usage: canary_cmp.py <pruned.json> <flagship.json> [old_pruned.json old_flagship.json]
"""
import json, math, sys

sys.path.insert(0, "/data/projects/q27b_on_12gb/harness")
from canary2 import _flatten_battery_items, token_flip_rate   # noqa: E402

REF = "/data/projects/q27b_on_12gb/results/anchor_reference.json"
BATTERIES = ["c1_tool_composition", "c2_error_recovery", "c3_tool_choice_ambiguity",
             "c4_algorithmic_judgment", "c5_constraint_conflict"]
BAD = (None, "NO_VALID_CALL", "NO_ANSWER", "ERROR")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def probe_vector(path, ref):
    """Per-probe {id: (matches_anchor, response)} — the vector the headline
    divergence_points is an average of."""
    d = json.load(open(path))
    out = {}
    for key in BATTERIES:
        if key not in d or key not in ref:
            continue
        rmap = {i: (r, c) for i, r, c in _flatten_battery_items(key, ref[key])}
        for iid, resp, canon in _flatten_battery_items(key, d[key]):
            if iid not in rmap:
                continue
            rresp, rcanon = rmap[iid]
            out[iid] = ((canon == rcanon) and canon not in BAD, resp or "",
                        token_flip_rate(resp or "", rresp or ""))
    return d, out


def main():
    ref = json.load(open(REF))
    pruned_p, flag_p = sys.argv[1], sys.argv[2]
    dp, vp = probe_vector(pruned_p, ref)
    df, vf = probe_vector(flag_p, ref)
    common = sorted(set(vp) & set(vf))

    b = sum(1 for i in common if vf[i][0] and not vp[i][0])   # flagship matches anchor, prune does not
    c = sum(1 for i in common if not vf[i][0] and vp[i][0])
    both = sum(1 for i in common if vf[i][0] and vp[i][0])
    res = {
        "n_probes": len(common),
        "pruned": {"exact_match": sum(1 for i in common if vp[i][0]),
                   "divergence_points": dp["reference_divergence"]["divergence_points"],
                   "token_flip_rate": dp["reference_divergence"]["overall_token_flip_rate"],
                   "canary_score": dp["aggregate"]["canary_score"]},
        "flagship": {"exact_match": sum(1 for i in common if vf[i][0]),
                     "divergence_points": df["reference_divergence"]["divergence_points"],
                     "token_flip_rate": df["reference_divergence"]["overall_token_flip_rate"],
                     "canary_score": df["aggregate"]["canary_score"]},
        "paired_mcnemar": {"both": both, "flagship_only_b": b, "pruned_only_c": c,
                           "p": mcnemar(b, c)},
        "probes_where_they_differ": [i for i in common if vp[i][0] != vf[i][0]],
        "response_identical_rate": round(
            sum(1 for i in common if vp[i][1] == vf[i][1]) / len(common), 4) if common else None,
    }

    # reproducibility control against the earlier greedy runs, if given
    if len(sys.argv) > 4:
        for label, new_path, old_path in (("pruned", pruned_p, sys.argv[3]),
                                          ("flagship", flag_p, sys.argv[4])):
            _, vnew = probe_vector(new_path, ref)
            _, vold = probe_vector(old_path, ref)
            com = sorted(set(vnew) & set(vold))
            same_txt = sum(1 for i in com if vnew[i][1] == vold[i][1])
            same_match = sum(1 for i in com if vnew[i][0] == vold[i][0])
            res.setdefault("reproducibility", {})[label] = {
                "n": len(com),
                "identical_response_text": same_txt,
                "identical_match_verdict": same_match,
                "bit_reproducible": same_txt == len(com),
            }

    print(json.dumps(res, indent=2))
    json.dump(res, open("/data/projects/q27b_on_12gb/results/canary2_reaudit.json", "w"), indent=2)


if __name__ == "__main__":
    main()

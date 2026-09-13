#!/usr/bin/env python3
"""Paired per-task comparison of the SAME GGUF build run on two GPU
architectures (L40S sm_89 vs RTX 3060 sm_86), same harness, same flags.

Pass = attempt 1 passed (greedy). McNemar is the exact two-sided binomial
test on the discordant pairs, matching analyze.py's convention in the
quality battery.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

L40S_DIR = "/Users/yugendren/experiments/local_inference_program/quality_battery/results"

# build -> (L40S jsonl, 3060 jsonl)
PAIRS = {
    "IQ3_XXS":  ("reliability_IQ3_XXS_nospec.jsonl",  "reliability_IQ3_XXS_nospec_3060.jsonl"),
    "Q2_K_XL":  ("reliability_Q2_K_XL.jsonl",         "reliability_Q2_K_XL_nospec_3060.jsonl"),
    "IQ2_XXS":  ("reliability_IQ2_XXS_nospec.jsonl",  "reliability_IQ2_XXS_nospec_3060.jsonl"),
    "Q8_0":     ("reliability_Q8_0.jsonl",            "reliability_Q8_0_nospec_3060.jsonl"),
}


# The `response` field was added to reliability.py part-way through the L40S
# campaign, so the primary result files for some builds carry no generations.
# For the side-by-side evidence we fall back to the b10566 "oldbuild" re-run of
# the SAME build on the SAME L40S, which does carry them. That substitution is
# safe for IQ3_XXS (oldbuild vs master: McNemar b=0, c=0 -- per-task identical)
# and near-safe for IQ2_XXS (b=0, c=1 -- one task differs).
RESP_SRC = {
    "IQ3_XXS": "reliability_IQ3_XXS_oldbuild.jsonl",
    "IQ2_XXS": "reliability_IQ2_XXS_nospec_oldbuild.jsonl",
}


def load(path):
    """task_id -> attempt-1 record."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            a1 = next((a for a in r["attempts"] if a["attempt"] == 1), None)
            if a1 is None:
                continue
            out[r["task_id"]] = a1
    return out


def wilson_ci95(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


ERR_RE = re.compile(r"(\w*Error|timeout)")


def errkind(a):
    if a.get("passed"):
        return None
    e = (a.get("error") or "").strip()
    m = ERR_RE.findall(e)
    return m[-1] if m else (e[:30] or "EMPTY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", required=True, help="dir holding the 3060 jsonl files")
    ap.add_argument("--l40s-dir", default=L40S_DIR)
    ap.add_argument("--builds", default=",".join(PAIRS))
    ap.add_argument("--dump-diffs", type=int, default=0,
                    help="print N side-by-side generations per build for flipped tasks")
    args = ap.parse_args()

    print("| build | L40S sm_89 (k/164) | RTX 3060 sm_86 (k/n) | 3060 95% CI | delta |")
    print("|---|---|---|---|---|")
    rows = {}
    for b in args.builds.split(","):
        lf, nf = PAIRS[b]
        A = load(os.path.join(args.l40s_dir, lf))     # L40S
        B = load(os.path.join(args.new_dir, nf))      # 3060
        if not B:
            print(f"| {b} | - | NOT RUN | - | - |")
            continue
        common = sorted(set(A) & set(B), key=lambda t: int(t.split("/")[1]))
        ka = sum(1 for t in A.values() if t["passed"])
        kb = sum(1 for t in B.values() if t["passed"])
        lo, hi = wilson_ci95(kb, len(B))
        print(f"| {b} | {ka}/{len(A)} = {100*ka/len(A):.1f}% | {kb}/{len(B)} = {100*kb/len(B):.1f}% | "
              f"{100*lo:.1f}%-{100*hi:.1f}% | {100*kb/len(B) - 100*ka/len(A):+.1f} pt |")
        rows[b] = (A, B, common)

    print()
    print("### Greedy determinism across platforms")
    print()
    print("Fraction of paired tasks where the two GPUs emitted the SAME number of "
          "completion tokens. Greedy decoding is deterministic given identical "
          "arithmetic, so this is a direct proxy for kernel-level numerical "
          "agreement -- independent of whether the answer was right.")
    print()
    print("| build | identical token count | mean |L40S - 3060| tokens |")
    print("|---|---|---|---|")
    for b, (A, B, common) in rows.items():
        same = sum(1 for t in common
                   if A[t].get("completion_tokens") == B[t].get("completion_tokens"))
        d = [abs((A[t].get("completion_tokens") or 0) - (B[t].get("completion_tokens") or 0))
             for t in common]
        print(f"| {b} | {same}/{len(common)} = {100*same/len(common):.1f}% | "
              f"{sum(d)/len(d):.1f} |")

    print()
    print("### Paired McNemar, SAME build across platforms (key = task_id, N=1 greedy)")
    print()
    print("| build | n paired | both pass | b (L40S pass / 3060 fail) | c (L40S fail / 3060 pass) | neither | p (exact) |")
    print("|---|---|---|---|---|---|---|")
    for b, (A, B, common) in rows.items():
        both = sum(1 for t in common if A[t]["passed"] and B[t]["passed"])
        bb = sum(1 for t in common if A[t]["passed"] and not B[t]["passed"])
        cc = sum(1 for t in common if not A[t]["passed"] and B[t]["passed"])
        nn = sum(1 for t in common if not A[t]["passed"] and not B[t]["passed"])
        p = mcnemar_exact(bb, cc)
        print(f"| {b} | {len(common)} | {both} | {bb} | {cc} | {nn} | {p:.4g} |")

    print()
    print("### Flipped tasks (L40S pass -> 3060 fail), with the 3060 failure mode")
    print()
    for b, (A, B, common) in rows.items():
        flips = [t for t in common if A[t]["passed"] and not B[t]["passed"]]
        back = [t for t in common if not A[t]["passed"] and B[t]["passed"]]
        print(f"**{b}** — {len(flips)} regressions, {len(back)} recoveries")
        if flips:
            print()
            print("| task | 3060 error | 3060 completion_tokens | L40S completion_tokens |")
            print("|---|---|---|---|")
            for t in flips:
                print(f"| {t} | `{errkind(B[t])}` | {B[t].get('completion_tokens')} | {A[t].get('completion_tokens')} |")
        if back:
            print()
            print(f"recoveries: {', '.join(back)}")
        print()

    if args.dump_diffs:
        print("### Side-by-side generations on flipped tasks (bug-report evidence)")
        print()
        print("Same GGUF, same prompt, same greedy settings, same harness file; "
              "only the GPU differs.")
        print()
        for b, (A, B, common) in rows.items():
            src = A
            note = ""
            if not any((a.get("response") or "").strip() for a in A.values()) and b in RESP_SRC:
                src = load(os.path.join(args.l40s_dir, RESP_SRC[b]))
                note = f" (L40S text from {RESP_SRC[b]})"
            flips = [t for t in common if A[t]["passed"] and not B[t]["passed"]
                     and (src.get(t, {}).get("response") or "").strip()]
            for t in flips[:args.dump_diffs]:
                print(f"#### {b} — {t}{note}")
                print()
                print(f"L40S sm_89 (PASS, {src[t].get('completion_tokens')} tok):")
                print()
                print("```")
                print((src[t].get("response") or "")[:1600])
                print("```")
                print()
                print(f"RTX 3060 sm_86 (FAIL: {errkind(B[t])}, {B[t].get('completion_tokens')} tok):")
                print()
                print("```")
                print((B[t].get("response") or "")[:1600])
                print("```")
                print()

    print("### Failure-mode census (3060 vs L40S), attempt 1")
    print()
    print("| build | platform | fails | error-type counts | fence_unclosed |")
    print("|---|---|---|---|---|")
    for b, (A, B, common) in rows.items():
        for name, D in (("L40S sm_89", A), ("RTX 3060 sm_86", B)):
            fails = [a for a in D.values() if not a["passed"]]
            cnt = Counter(errkind(a) for a in fails)
            unc = sum(1 for a in D.values() if a.get("fence_unclosed"))
            print(f"| {b} | {name} | {len(fails)} | {dict(cnt)} | {unc} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

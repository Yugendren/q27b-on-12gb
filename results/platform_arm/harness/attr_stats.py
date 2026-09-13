#!/usr/bin/env python3
import json, math
R = "/data/projects/q27b_on_12gb/results/platform_arm"
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
def load(tag):
    d = {}
    for l in open(f"{R}/reliability_{tag}.jsonl"):
        if l.strip():
            r = json.loads(l); d[r["task_id"]] = r["attempts"][0]
    return d
def wil(k, n):
    z = 1.959963984540054; p = k/n; d = 1+z*z/n
    c = p+z*z/(2*n); m = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)
    return (c-m)/d, (c+m)/d

arms = {"L1 (all-legacy)": "IQ3_XXS_L1_3060",
        "L3 (legacy, thinking OFF)": "IQ3_XXS_L3_3060",
        "battery protocol": "IQ3_XXS_nospec_3060"}
D = {k: load(v) for k, v in arms.items()}
print("| arm | solved | rate | 95% CI | mean tok/task |")
print("|---|---|---|---|---|")
for k, d in D.items():
    s = sum(1 for a in d.values() if a["passed"]); n = len(d)
    lo, hi = wil(s, n)
    print(f"| {k} | {s}/{n} | {100*s/n:.1f}% | {100*lo:.1f}%-{100*hi:.1f}% | "
          f"{sum(a.get('completion_tokens') or 0 for a in d.values())/n:.0f} |")
print()
print("| pair | both | b (A-pass/B-fail) | c (A-fail/B-pass) | neither | p |")
print("|---|---|---|---|---|---|")
for a, b in [("L1 (all-legacy)", "L3 (legacy, thinking OFF)"),
             ("L3 (legacy, thinking OFF)", "battery protocol"),
             ("L1 (all-legacy)", "battery protocol")]:
    A, B = D[a], D[b]; com = sorted(set(A) & set(B))
    bo = sum(1 for t in com if A[t]["passed"] and B[t]["passed"])
    bb = sum(1 for t in com if A[t]["passed"] and not B[t]["passed"])
    cc = sum(1 for t in com if not A[t]["passed"] and B[t]["passed"])
    nn = len(com) - bo - bb - cc
    print(f"| {a} vs {b} | {bo} | {bb} | {cc} | {nn} | {mcnemar(bb,cc):.4g} |")

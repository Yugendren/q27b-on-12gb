#!/usr/bin/env python3
"""Compare the L1 legacy-protocol reconstruction against the ORIGINAL
2026-08 3060 run stored in results/quality_full_iq3xxs.json."""
import json, math, sys
from collections import Counter

OLD = "/data/projects/q27b_on_12gb/results/quality_full_iq3xxs.json"
R = "/data/projects/q27b_on_12gb/results/platform_arm"

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)

old = {r["task_id"]: bool(r["passed"]) for r in json.load(open(OLD))["humaneval"]["results"]}

def load(tag):
    d = {}
    try:
        for l in open(f"{R}/reliability_{tag}.jsonl"):
            if l.strip():
                r = json.loads(l); d[r["task_id"]] = r["attempts"][0]
    except FileNotFoundError:
        pass
    return d

print(f"ORIGINAL 2026-08 3060 IQ3_XXS (old harness, old protocol): "
      f"{sum(old.values())}/{len(old)} = {100*sum(old.values())/len(old):.1f}%")
for tag, label in [("IQ3_XXS_L1_3060", "L1  full legacy protocol, today"),
                   ("IQ3_XXS_L3_3060", "L3  legacy but thinking OFF"),
                   ("IQ3_XXS_nospec_3060", "battery protocol (primary arm)")]:
    n = load(tag)
    if not n: 
        print(f"{label:34s} NOT RUN"); continue
    common = sorted(set(old) & set(n), key=lambda t: int(t.split("/")[1]))
    k = sum(1 for a in n.values() if a["passed"])
    b = sum(1 for t in common if old[t] and not n[t]["passed"])
    c = sum(1 for t in common if not old[t] and n[t]["passed"])
    both = sum(1 for t in common if old[t] and n[t]["passed"])
    tok = sum(a.get("completion_tokens") or 0 for a in n.values()) / len(n)
    print(f"{label:34s} {k}/{len(n)} = {100*k/len(n):5.1f}%  "
          f"vs-original: both={both} b={b} c={c} p={mcnemar(b,c):.4g}  mean_tok={tok:.0f}")
    if tag == "IQ3_XXS_L1_3060":
        print(f"    per-task agreement with the original run: "
              f"{sum(1 for t in common if old[t]==n[t]['passed'])}/{len(common)}")

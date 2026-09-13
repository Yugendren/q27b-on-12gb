#!/usr/bin/env python3
"""Mechanism census for the legacy protocol: how the thinking block destroys
the answer at max_tokens=1024."""
import json, re
from collections import Counter
R = "/data/projects/q27b_on_12gb/results/platform_arm"

def load(tag):
    d = {}
    for l in open(f"{R}/reliability_{tag}.jsonl"):
        if l.strip():
            r = json.loads(l); d[r["task_id"]] = r["attempts"][0]
    return d

ERR = re.compile(r"(\w*Error|timeout)")
for tag in ["IQ3_XXS_L1_3060", "IQ3_XXS_L3_3060", "IQ3_XXS_nospec_3060"]:
    try: d = load(tag)
    except FileNotFoundError: continue
    fails = {t: a for t, a in d.items() if not a["passed"]}
    cap = sum(1 for a in d.values() if (a.get("completion_tokens") or 0) >= 1024)
    capf = sum(1 for a in fails.values() if (a.get("completion_tokens") or 0) >= 1024)
    nofence = sum(1 for a in fails.values()
                  if "```" not in (a.get("response") or ""))
    kinds = Counter((ERR.findall(a.get("error") or "") or ["EMPTY"])[-1] for a in fails.values())
    print(f"{tag}:  n={len(d)}  fails={len(fails)}  mean_tok={sum(a.get('completion_tokens') or 0 for a in d.values())/len(d):.0f}")
    print(f"   hit the 1024-token cap: {cap}/{len(d)} overall, {capf}/{len(fails)} of the failures")
    print(f"   failures whose completion contains NO code fence at all: {nofence}/{len(fails)}")
    print(f"   error kinds: {dict(kinds)}")

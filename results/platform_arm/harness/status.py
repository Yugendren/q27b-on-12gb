#!/usr/bin/env python3
"""Progress + per-task token/wall census for the platform arm."""
import glob, json, os, sys
R = "/data/projects/q27b_on_12gb/results/platform_arm"
for p in sorted(glob.glob(os.path.join(R, "reliability_*_3060.jsonl"))):
    rs = []
    for l in open(p):
        l = l.strip()
        if l:
            try: rs.append(json.loads(l))
            except json.JSONDecodeError: pass
    if not rs: continue
    k = sum(1 for r in rs if r.get("passed_at") == 1)
    tok = sum(r["attempts"][0].get("completion_tokens") or 0 for r in rs)
    wall = sum(r["attempts"][0].get("wall_seconds") or 0 for r in rs)
    print(f"{os.path.basename(p):48s} n={len(rs):3d} pass={k:3d} "
          f"mean_tok={tok/len(rs):6.1f} mean_wall={wall/len(rs):7.1f}s "
          f"eta164={(wall/len(rs))*164/60:6.1f}min")

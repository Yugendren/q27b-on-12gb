#!/usr/bin/env python3
"""Greedy-probe divergence analysis for the turbo4 KV A/B.

Reference arm is f16 (the highest-precision KV that fits). Every arm ran with
--spec-type none, so the control is deterministic (FINDINGS 2026-09-03: MTP is
NOT bit-exact on this model, so speculation would have masked the KV effect).
"""
import json

P = "/data/projects/q27b_on_12gb/results/turbo4/probes.json"
d = json.load(open(P))
ref = d["f16"]["probes"]

print("divergence vs f16 (sha1 of full completion, greedy, spec OFF, n=10):")
for kv in ["q8_0", "turbo4", "q4_0"]:
    ps = d[kv]["probes"]
    ident, det = 0, []
    for a, b in zip(ref, ps):
        if a["sha1"] == b["sha1"]:
            ident += 1
        else:
            ca, cb = a["content"], b["content"]
            i = next((j for j, (x, y) in enumerate(zip(ca, cb)) if x != y),
                     min(len(ca), len(cb)))
            det.append("p%d@char%d/%d" % (a["i"], i, len(ca)))
    print("  %-7s identical %d/10   diverged: %s" % (kv, ident, ", ".join(det)))

print()
print("cross-arm agreement matrix (identical probes out of 10):")
arms = ["f16", "q8_0", "turbo4", "q4_0"]
print("        " + "".join("%9s" % a for a in arms))
for a in arms:
    row = []
    for b in arms:
        n = sum(1 for x, y in zip(d[a]["probes"], d[b]["probes"])
                if x["sha1"] == y["sha1"])
        row.append("%9d" % n)
    print("%-8s" % a + "".join(row))

# The probes with a checkable ground truth -- divergence only matters if it
# changes the answer.
CHECK = {
    2: ("847", "17*23+456"),
    4: ("2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37", "first 12 primes"),
    9: ("Physical", "OSI layers"),
}
print()
print("substantive check on ground-truthable probes:")
for i, (needle, what) in CHECK.items():
    line = "  p%d (%s): " % (i, what)
    for kv in arms:
        c = d[kv]["probes"][i]["content"]
        line += "%s=%s  " % (kv, "OK" if needle.split(",")[0] in c else "MISS")
    print(line)

print()
print("=== p2 arithmetic, last 200 chars of each arm ===")
for kv in arms:
    c = d[kv]["probes"][2]["content"]
    print("--- %s: ...%s" % (kv, c[-200:].strip().replace("\n", " | ")))

print()
print("=== p6 (process vs thread, prose) first divergence context, turbo4 vs f16 ===")
a = d["f16"]["probes"][6]["content"]
b = d["turbo4"]["probes"][6]["content"]
i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), 0)
print("  f16   : ...%s" % a[max(0, i - 90):i + 90].replace("\n", " | "))
print("  turbo4: ...%s" % b[max(0, i - 90):i + 90].replace("\n", " | "))

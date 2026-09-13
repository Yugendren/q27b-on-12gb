#!/usr/bin/env python3
"""cmp_paired.py -- paired per-task comparison of two model runs on the same
task set, for a certification report.

Standard library only. Two subcommands:

  humaneval  -- reliability-style JSONL (task_id, passed_at, attempts[...])
                as emitted by the q27b_on_12gb reliability harness.
  edit       -- polyglot-edit-style JSONL (task_id, passed_1, passed_2,
                passed, edit_ok_1, edit_ok_2, completion_tokens, ...) as
                emitted by quality_battery/harness/polyglot_edit.py.

wilson_ci95() and mcnemar_exact() are copied VERBATIM from
  /Users/yugendren/experiments/q27b_on_12gb/results/platform_arm/harness/analyze_platform.py

Convention for "pass at attempt 1" on the humaneval side: a record counts as
passed iff its top-level `passed_at` field equals 1 (this is the field the
reliability harness itself uses to mean "passed on attempt 1"; it agrees
exactly with attempts[0]["passed"] in every file inspected while building
this tool).

Convention for the edit side (derived from reading polyglot_edit.py's
process_task()/summarize()):
  - task_id           str, exercise name (record["task_id"])
  - passed_1          bool, tests passed on attempt 1 (record["passed_1"])
  - passed_2          bool, tests passed on attempt 2 ONLY (record["passed_2"])
  - passed            bool, tests passed on attempt 1 OR 2 (record["passed"])
  - edit_ok_1         bool, attempt-1 output was a well-formed fenced
                       ```python block that parsed with ast.parse
                       (record["edit_ok_1"])
  - edit_ok_2         bool, same for attempt 2 (record["edit_ok_2"])
  - completion_tokens int, SUM of completion tokens across all attempts made
                       (record["completion_tokens"]) -- there is no per-attempt
                       token list in this format, unlike the humaneval format.
  IMPORTANT: polyglot_edit.py's own summarize() computes its "pass_2" summary
  metric as sum(r["passed"] for r in rows) -- i.e. the "pass_2" label in the
  harness's summary JSON actually means "passed within 2 attempts" (the
  overall `passed` field), NOT the raw per-attempt `passed_2` field. This
  tool's --metric pass_2 reproduces that exact convention: it reads
  record["passed"], not record["passed_2"], to stay consistent with the
  reference harness's own summary semantics.
"""
import argparse
import json
import math
import os
import sys


# ---------------------------------------------------------------------------
# Copied VERBATIM from analyze_platform.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """task_id -> full record."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def humaneval_pass1(r):
    """Convention: passed at attempt 1 iff top-level passed_at == 1."""
    return r.get("passed_at") == 1


def humaneval_attempt1(r):
    attempts = r.get("attempts") or []
    return attempts[0] if attempts else {}


EDIT_METRICS = {
    "edit_ok_1": lambda r: bool(r.get("edit_ok_1")),
    "pass_1": lambda r: bool(r.get("passed_1")),
    # NB: "pass_2" reproduces polyglot_edit.py's own summarize(), which
    # aggregates the overall `passed` field (attempt 1 OR 2) under the
    # "pass_2" label in its summary JSON -- not the raw `passed_2` field.
    "pass_2": lambda r: bool(r.get("passed")),
}


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------

def report_pairing(a, b, label_a, label_b):
    a_ids = set(a)
    b_ids = set(b)
    common = sorted(a_ids & b_ids)
    only_a = sorted(a_ids - b_ids)
    only_b = sorted(b_ids - a_ids)

    print("=== Pairing ===")
    print(f"  n_{label_a:<12} = {len(a_ids)}")
    print(f"  n_{label_b:<12} = {len(b_ids)}")
    print(f"  n_common      = {len(common)}")
    if only_a:
        print(f"  only in {label_a} ({len(only_a)}): {', '.join(only_a)}")
    else:
        print(f"  only in {label_a}: (none)")
    if only_b:
        print(f"  only in {label_b} ({len(only_b)}): {', '.join(only_b)}")
    else:
        print(f"  only in {label_b}: (none)")
    print()
    return common


def rate_line(label, k, n):
    lo, hi = wilson_ci95(k, n)
    rate = 100.0 * k / n if n else 0.0
    print(f"  {label:<24} {k:>4}/{n:<4} = {rate:6.2f}%   95% CI [{100*lo:5.2f}%, {100*hi:5.2f}%]")


def report_rates(a, b, label_a, label_b, metric_fn, metric_name):
    print(f"=== Rates: {metric_name} (each side's own full task set) ===")
    ka = sum(1 for r in a.values() if metric_fn(r))
    kb = sum(1 for r in b.values() if metric_fn(r))
    rate_line(label_a, ka, len(a))
    rate_line(label_b, kb, len(b))
    print()


def report_mcnemar(a, b, common, label_a, label_b, metric_fn):
    b_wins = [t for t in common if metric_fn(a[t]) and not metric_fn(b[t])]
    c_wins = [t for t in common if not metric_fn(a[t]) and metric_fn(b[t])]
    p = mcnemar_exact(len(b_wins), len(c_wins))
    verdict = "SIGNIFICANT (p<0.05)" if p < 0.05 else "indistinguishable"
    print("=== McNemar (paired, over n_common) ===")
    print(f"  b ({label_a}-only wins) = {len(b_wins)}")
    print(f"  c ({label_b}-only wins) = {len(c_wins)}")
    print(f"  p (exact, two-sided)  = {p:.4g}")
    print(f"  verdict: {verdict}")
    print()

    print("=== Discordant task_ids ===")
    print(f"  {label_a} passed, {label_b} failed ({len(b_wins)}): "
          f"{', '.join(b_wins) if b_wins else '(none)'}")
    print(f"  {label_b} passed, {label_a} failed ({len(c_wins)}): "
          f"{', '.join(c_wins) if c_wins else '(none)'}")
    print()
    return b_wins, c_wins


def report_tripwire(a, b, label_a, label_b):
    print("=== Thinking tripwire (humaneval, attempt 1, each side's own full task set) ===")
    for label, d in ((label_a, a), (label_b, b)):
        a1s = [humaneval_attempt1(r) for r in d.values()]
        toks = [a1.get("completion_tokens") or 0 for a1 in a1s]
        n = len(toks)
        mean_tok = sum(toks) / n if n else 0.0
        max_tok = max(toks) if toks else 0
        n_error = sum(1 for a1 in a1s if a1.get("error") is not None)
        n_fence = sum(1 for a1 in a1s if a1.get("fence_unclosed"))
        flag = "TRIPWIRE FAIL (>350)" if mean_tok > 350 else "ok"
        print(f"  {label:<12} n={n:<4} mean_completion_tokens={mean_tok:8.2f}  {flag}")
        print(f"  {'':<12} max_completion_tokens={max_tok}  "
              f"n_with_error={n_error}  n_fence_unclosed={n_fence}")
    print()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_humaneval(args):
    label_a = args.label_a
    label_b = args.label_b
    a = load_jsonl(args.a)
    b = load_jsonl(args.b)

    print(f"# humaneval paired comparison: {label_a} ({os.path.basename(args.a)}) "
          f"vs {label_b} ({os.path.basename(args.b)})")
    print()

    common = report_pairing(a, b, label_a, label_b)
    report_rates(a, b, label_a, label_b, humaneval_pass1, "passed_at==1 (attempt 1)")
    report_mcnemar(a, b, common, label_a, label_b, humaneval_pass1)
    report_tripwire(a, b, label_a, label_b)
    return 0


def cmd_edit(args):
    label_a = args.label_a
    label_b = args.label_b
    a = load_jsonl(args.a)
    b = load_jsonl(args.b)
    metric_fn = EDIT_METRICS[args.metric]

    print(f"# edit paired comparison: {label_a} ({os.path.basename(args.a)}) "
          f"vs {label_b} ({os.path.basename(args.b)})  [metric={args.metric}]")
    print()

    common = report_pairing(a, b, label_a, label_b)
    report_rates(a, b, label_a, label_b, metric_fn, args.metric)
    report_mcnemar(a, b, common, label_a, label_b, metric_fn)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_he = sub.add_parser("humaneval", help="reliability-style HumanEval JSONL comparison")
    p_he.add_argument("--a", required=True, help="path to side A JSONL")
    p_he.add_argument("--b", required=True, help="path to side B JSONL")
    p_he.add_argument("--label-a", default="A")
    p_he.add_argument("--label-b", default="B")
    p_he.set_defaults(func=cmd_humaneval)

    p_ed = sub.add_parser("edit", help="polyglot-edit-style JSONL comparison")
    p_ed.add_argument("--a", required=True, help="path to side A JSONL")
    p_ed.add_argument("--b", required=True, help="path to side B JSONL")
    p_ed.add_argument("--label-a", default="A")
    p_ed.add_argument("--label-b", default="B")
    p_ed.add_argument("--metric", choices=sorted(EDIT_METRICS), default="pass_1")
    p_ed.set_defaults(func=cmd_edit)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    main()
    sys.exit(0)

#!/usr/bin/env python3
"""Best-of-n verified-selection eval for HumanEval against a running llama-server.

Requests k independent completions per task (k sequential /v1/chat/completions
calls at the configured temperature/top_p -- NOT the `n=` batch param, since
llama-server support for it is unreliable). Reuses eval_quality.py's HumanEval
download/parsing, code-extraction, and subprocess-execution machinery.

Three scores per task:
  a) pass@1 / pass@k  -- each of the k candidates run against the task's
     HIDDEN test (the dataset `check()` function).
  b) verified-selection score -- candidates are additionally run against a
     VISIBLE test built from the '>>>' doctest-style examples parsed out of
     the task's docstring. The first candidate that passes the visible test
     is selected; its HIDDEN-test result is what counts. Tasks with no
     parseable visible examples fall back to selecting candidate #1 (i.e.
     behave like pass@1 for that task), and are flagged as such.

Resumable: partial results are written to --out after every task; a re-run
with the same --out skips task_ids that already have a full result.

Usage:
    python3 bestofn.py --endpoint http://127.0.0.1:8080 \\
        --k 5 --temp 0.8 --top-p 0.95 --limit 164 --max-tokens 768 \\
        --out results/bestofn.json
"""
import argparse
import ast
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

# reuse HumanEval download/parsing/execution machinery from eval_quality.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_quality as eq  # noqa: E402

DEFAULT_DATA_DIR = eq.DEFAULT_DATA_DIR

EXAMPLE_RE = re.compile(r"^\s*>>>\s*(.+?)\s*$")


# --------------------------------------------------------------------------
# visible-example (docstring '>>>') extraction
# --------------------------------------------------------------------------

def extract_raw_examples(prompt):
    """Pull (call_text, expected_text) pairs out of '>>>' doctest lines.

    The expected value is taken as the next non-blank line that isn't itself
    a '>>>' line. Anything that doesn't fit this shape (no following output
    line, e.g. two '>>>' in a row) is skipped.
    """
    lines = prompt.split("\n")
    n = len(lines)
    pairs = []
    i = 0
    while i < n:
        m = EXAMPLE_RE.match(lines[i])
        if m:
            call = m.group(1)
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and not EXAMPLE_RE.match(lines[j]):
                expected = lines[j].strip()
                # strip a trailing docstring close-quote fragment if present
                for q in ('"""', "'''"):
                    if expected.endswith(q) and not expected.startswith(q):
                        expected = expected[: -len(q)].strip()
                if expected:
                    pairs.append((call, expected))
                i = j + 1
                continue
        i += 1
    return pairs


def parse_examples(prompt):
    """Return the subset of extracted examples that are safely executable:
    the call must parse as a Python expression and the expected output must
    be a Python literal (so we never eval arbitrary model/docstring text).
    """
    usable = []
    for call, expected in extract_raw_examples(prompt):
        try:
            ast.parse(call, mode="eval")
            ast.literal_eval(expected)
        except Exception:
            continue
        usable.append({"call": call, "expected": expected})
    return usable


def build_visible_program(prompt, code, entry_point, examples):
    has_def = re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code) is not None
    body = code if has_def else (prompt + code)
    asserts = []
    for idx, ex in enumerate(examples):
        # build the failure message host-side and inject it via repr() so it's
        # always a syntactically valid (properly escaped) string literal,
        # regardless of what quote characters appear in the example text.
        msg = repr(f"visible example {idx} failed: {ex['call']} expected {ex['expected']}")
        # no wrapping parens around call: it may legally end in a trailing
        # '# comment' (seen in the wild in HumanEval docstrings), which would
        # swallow a following close-paren on the same line.
        asserts.append(
            f"__r{idx} = {ex['call']}\n"
            f"assert __r{idx} == ({ex['expected']}), {msg}"
        )
    return body + "\n\n" + "\n".join(asserts) + "\n"


# --------------------------------------------------------------------------
# server interaction (k independent sampled calls, not n=)
# --------------------------------------------------------------------------

def chat_complete_sample(endpoint, model_id, system, user, temperature, top_p, max_tokens, timeout=600):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body = {
        "model": model_id or "default",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    r = eq.requests.post(
        endpoint.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# per-task processing
# --------------------------------------------------------------------------

def process_task(item, endpoint, model_id, k, temperature, top_p, max_tokens):
    task_id = item["task_id"]
    prompt = item["prompt"]
    entry_point = item["entry_point"]
    test = item["test"]
    examples = parse_examples(prompt)
    has_visible = bool(examples)

    candidates = []
    for idx in range(k):
        try:
            completion = chat_complete_sample(
                endpoint, model_id, eq.HUMANEVAL_SYSTEM, prompt,
                temperature, top_p, max_tokens,
            )
            code = eq.extract_code(completion)
            request_error = None
        except Exception as e:  # noqa: BLE001
            code = ""
            request_error = f"request_error: {e}"

        if request_error is None:
            hidden_program = eq.build_program(prompt, code, entry_point, test)
            hidden_pass, hidden_error = eq.run_program(hidden_program)
        else:
            hidden_pass, hidden_error = False, request_error

        if has_visible:
            if request_error is None:
                visible_program = build_visible_program(prompt, code, entry_point, examples)
                visible_pass, visible_error = eq.run_program(visible_program)
            else:
                visible_pass, visible_error = False, request_error
        else:
            visible_pass, visible_error = None, None

        candidates.append({
            "index": idx,
            "request_error": request_error,
            "hidden_pass": hidden_pass,
            "hidden_error": hidden_error,
            "visible_pass": visible_pass,
            "visible_error": visible_error,
        })

    hidden_pass_count = sum(1 for c in candidates if c["hidden_pass"])
    any_hidden_pass = hidden_pass_count > 0

    if has_visible:
        selected_index = next((c["index"] for c in candidates if c["visible_pass"]), None)
        if selected_index is None:
            selected_index = 0
            selection_mode = "fallback_no_visible_pass"
        else:
            selection_mode = "visible_pass"
    else:
        selected_index = 0
        selection_mode = "fallback_no_examples"

    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "num_visible_examples": len(examples),
        "has_visible_tests": has_visible,
        "candidates": candidates,
        "hidden_pass_count": hidden_pass_count,
        "k": k,
        "any_hidden_pass": any_hidden_pass,
        "selected_index": selected_index,
        "selection_mode": selection_mode,
        "selected_hidden_pass": candidates[selected_index]["hidden_pass"],
    }


# --------------------------------------------------------------------------
# aggregation / resumable driver
# --------------------------------------------------------------------------

def compute_aggregate(tasks, wall_time_s):
    n = len(tasks)
    if n == 0:
        return {
            "n_tasks": 0, "pass_at_1": None, "pass_at_k": None,
            "verified_selection_score": None, "frac_tasks_with_visible_tests": None,
            "wall_time_s": round(wall_time_s, 2),
        }
    total_k = sum(len(t["candidates"]) for t in tasks)
    total_hidden_pass = sum(t["hidden_pass_count"] for t in tasks)
    return {
        "n_tasks": n,
        "pass_at_1": total_hidden_pass / total_k if total_k else None,
        "pass_at_k": sum(1 for t in tasks if t["any_hidden_pass"]) / n,
        "verified_selection_score": sum(1 for t in tasks if t["selected_hidden_pass"]) / n,
        "frac_tasks_with_visible_tests": sum(1 for t in tasks if t["has_visible_tests"]) / n,
        "wall_time_s": round(wall_time_s, 2),
    }


def load_existing(out_path):
    if out_path and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        try:
            with open(out_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def write_out(out_path, payload):
    text = json.dumps(payload, indent=2)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(text + "\n")
    os.replace(tmp_path, out_path)


def run(args):
    t0 = time.time()
    model_info = eq.get_model_info(args.endpoint)
    items = eq.load_humaneval(DEFAULT_DATA_DIR, args.limit)

    existing = load_existing(args.out)
    tasks_by_id = {}
    prior_wall_time = 0.0
    if existing and existing.get("tasks"):
        for t in existing["tasks"]:
            tasks_by_id[t["task_id"]] = t
        prior_wall_time = (existing.get("aggregate") or {}).get("wall_time_s") or 0.0
        print(
            f"[bestofn] resuming from {args.out}: {len(tasks_by_id)} tasks already done",
            file=sys.stderr,
        )

    params = {
        "k": args.k, "temperature": args.temp, "top_p": args.top_p,
        "max_tokens": args.max_tokens, "limit": args.limit,
    }

    for i, item in enumerate(items):
        task_id = item["task_id"]
        if task_id in tasks_by_id:
            print(f"[bestofn] {i + 1}/{len(items)} ({task_id}) skip (cached)", file=sys.stderr)
            continue

        result = process_task(
            item, args.endpoint, model_info["model_id"],
            args.k, args.temp, args.top_p, args.max_tokens,
        )
        tasks_by_id[task_id] = result
        print(
            f"[bestofn] {i + 1}/{len(items)} ({task_id}) "
            f"hidden={result['hidden_pass_count']}/{args.k} "
            f"visible_avail={result['has_visible_tests']} "
            f"selected_hidden_pass={result['selected_hidden_pass']}",
            file=sys.stderr,
        )

        ordered_tasks = [tasks_by_id[it["task_id"]] for it in items if it["task_id"] in tasks_by_id]
        elapsed = time.time() - t0
        write_out(args.out, {
            "endpoint": args.endpoint,
            "model_info": model_info,
            "params": params,
            "tasks": ordered_tasks,
            "aggregate": compute_aggregate(ordered_tasks, prior_wall_time + elapsed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    ordered_tasks = [tasks_by_id[it["task_id"]] for it in items if it["task_id"] in tasks_by_id]
    elapsed = time.time() - t0
    out = {
        "endpoint": args.endpoint,
        "model_info": model_info,
        "params": params,
        "tasks": ordered_tasks,
        "aggregate": compute_aggregate(ordered_tasks, prior_wall_time + elapsed),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if args.out:
        write_out(args.out, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="e.g. http://127.0.0.1:8080")
    ap.add_argument("--k", type=int, default=5, help="candidates sampled per task")
    ap.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95, help="sampling top_p")
    ap.add_argument("--limit", type=int, default=164, help="first N HumanEval tasks")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out", required=True, help="results JSON path (also used for resume)")
    args = ap.parse_args()

    out = run(args)
    agg = out["aggregate"]

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"

    print(f"[bestofn] tasks={agg['n_tasks']} k={args.k} temp={args.temp} top_p={args.top_p}")
    print(f"[bestofn] pass@1 (mean over all candidates) = {fmt(agg['pass_at_1'])}")
    print(f"[bestofn] pass@k (any candidate passes)     = {fmt(agg['pass_at_k'])}")
    print(f"[bestofn] verified-selection score          = {fmt(agg['verified_selection_score'])}")
    print(
        f"[bestofn] visible-tests usable on "
        f"{fmt(agg['frac_tasks_with_visible_tests'])} of tasks; "
        f"wall_time={agg['wall_time_s']}s -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

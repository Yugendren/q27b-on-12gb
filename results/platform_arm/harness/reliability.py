#!/usr/bin/env python3
"""Test-time reliability curve for HumanEval-164 against a llama-server.

For each task we issue up to --max-attempts SEQUENTIAL sampling attempts
against a running llama.cpp llama-server (OpenAI-compatible API), stopping
as soon as one attempt's completion passes the task's unit test. This
measures "verified retries": how much of the model's raw capability is
recoverable by re-sampling under a pass/fail verifier, with NO error
feedback given between attempts. Every attempt sees the exact same prompt
(system + user) regardless of what happened on prior attempts -- this is
pure verifier-gated resampling, not iterative self-repair. Attempt 1 is
greedy (temperature=0, top_p=1.0, seed=42); attempts 2..N use
--temp-retry / --top-p with seed=42+k so each retry is reproducible but
distinct.

Code patterns (load_humaneval, extract_code, build_program, run_program,
chat_complete body shape incl. chat_template_kwargs.enable_thinking=False
and the reasoning_content fallback) are lifted near-verbatim from:
    /Users/yugendren/experiments/q27b_on_12gb/harness/eval_quality.py

Usage:
    python3 reliability.py --endpoint http://127.0.0.1:8080 \\
        --out results/reliability.jsonl --summary results/reliability.json \\
        [--max-attempts 5] [--limit 164] [--data-dir DIR] \\
        [--max-tokens 1024] [--temp-retry 0.8] [--top-p 0.95] \\
        [--concurrency 1] [--tag LABEL]
"""
import argparse
import gzip
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

HUMANEVAL_SYSTEM = (
    "Complete the following Python function. Respond with a single Python "
    "code block containing the full function definition (including the "
    "signature and docstring exactly as given), with no additional "
    "explanation."
)


# --------------------------------------------------------------------------
# data acquisition
# --------------------------------------------------------------------------

def ensure_file(url, dest_path, binary=True):
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[reliability] downloading {url} -> {dest_path}", file=sys.stderr)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    mode = "wb" if binary else "w"
    with open(dest_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dest_path


def load_humaneval(data_dir, limit):
    gz_path = ensure_file(HUMANEVAL_URL, os.path.join(data_dir, "HumanEval.jsonl.gz"))
    jsonl_path = os.path.join(data_dir, "HumanEval.jsonl")
    if not os.path.isfile(jsonl_path) or os.path.getsize(jsonl_path) == 0:
        with gzip.open(gz_path, "rt") as fin, open(jsonl_path, "w") as fout:
            fout.write(fin.read())
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


# --------------------------------------------------------------------------
# server interaction
# --------------------------------------------------------------------------

def get_model_info(endpoint):
    r = requests.get(endpoint.rstrip("/") + "/v1/models", timeout=30)
    r.raise_for_status()
    data = r.json()
    models = data.get("data", [])
    return {
        "raw": data,
        "model_id": models[0]["id"] if models else None,
    }


def chat_complete(endpoint, model_id, system, user, max_tokens, temperature, top_p, seed, timeout=600):
    """POST one chat turn and return (content, usage_dict).

    Qwen3-family models default to an internal "thinking" mode that, in a
    llama.cpp/OpenAI-style response, streams into `reasoning_content`
    rather than `content` -- at these token caps the whole budget can be
    consumed by the think block, leaving `content` empty even though the
    model is otherwise healthy. We pass the standard Qwen3
    chat_template_kwargs.enable_thinking=false switch to disable it (a
    no-op / ignored field on non-Qwen3 templates). As a defensive
    fallback in case a server build ignores that switch, if `content`
    comes back empty but `reasoning_content` doesn't, we use
    `reasoning_content` instead of silently failing.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body = {
        "model": model_id or "default",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": False,
    }
    r = requests.post(
        endpoint.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout
    )
    r.raise_for_status()
    data = r.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    if not content.strip():
        content = message.get("reasoning_content") or content
    usage = data.get("usage") or {}
    return content, usage


# --------------------------------------------------------------------------
# HumanEval execution
# --------------------------------------------------------------------------

OPEN_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n", re.IGNORECASE)


def extract_code(text):
    """Pull the Python body out of a chat completion.

    The original implementation only matched a fully CLOSED ```...``` block and
    otherwise returned the raw text. When a model opens a fence and never closes
    it, that raw text still begins with "```python", which is a guaranteed
    SyntaxError -- so a formatting slip was scored as a total code failure.
    Measured blast radius: 85 of UD-IQ2_S's 96 HumanEval failures were this
    artifact (vs 1-3 for every other build in the ladder).

    Order: closed fence -> unclosed fence (take everything after the opener,
    dropping any trailing stray fence) -> raw text with any stray fence lines
    stripped.
    """
    m = CODE_FENCE_RE.search(text)
    if m:
        return m.group(1)
    m = OPEN_FENCE_RE.search(text)
    if m:
        tail = text[m.end():]
        idx = tail.find("```")
        return tail[:idx] if idx != -1 else tail
    # No fence opener at all, but a stray ``` can still poison the program.
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))


def fence_unclosed(text):
    """True if the model opened a code fence but never closed it.

    Reported as its own metric: this is a real instruction-following /
    format-compliance signal, and it must not be laundered into the code score.
    """
    if CODE_FENCE_RE.search(text):
        return False
    return OPEN_FENCE_RE.search(text) is not None


def build_program(prompt, code, entry_point, test):
    has_def = re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code) is not None
    body = code if has_def else (prompt + code)
    return body + "\n\n" + test + f"\n\ncheck({entry_point})\n"


def run_program(program_text, timeout_s=10):
    fd, path = tempfile.mkstemp(suffix=".py", prefix="humaneval_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(program_text)
        try:
            proc = subprocess.run(
                ["python3", path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return proc.returncode == 0, proc.stderr[-2000:] if proc.returncode != 0 else None
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s}s"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# attempt / task-level driver
# --------------------------------------------------------------------------

def attempt_params(k, temp_retry, top_p):
    """Sampling params for attempt index k (1-based).

    Attempt 1 is greedy (temperature=0, top_p=1.0, seed=42). Attempts
    k>=2 use --temp-retry / --top-p with seed=42+k.
    """
    if k == 1:
        return {"temperature": 0, "top_p": 1.0, "seed": 42}
    return {"temperature": temp_retry, "top_p": top_p, "seed": 42 + k}


def run_task(endpoint, model_id, item, max_attempts, max_tokens, temp_retry, top_p,
             prior=None):
    """Run verifier-gated retries for one task.

    `prior` is an existing record for this task from a previous, shallower run
    (e.g. --max-attempts 3). Its attempts are kept verbatim and we resume at
    attempt len(prior_attempts)+1 with the same seed schedule, so extending
    N=3 -> N=5 costs only the two extra attempts on still-unsolved tasks and
    leaves the N=3 numbers bit-identical.
    """
    task_id = item["task_id"]
    attempts = list(prior["attempts"]) if prior else []
    passed_at = prior.get("passed_at") if prior else None
    if passed_at is not None:
        return dict(prior)
    start_k = len(attempts) + 1
    for k in range(start_k, max_attempts + 1):
        params = attempt_params(k, temp_retry, top_p)
        t0 = time.time()
        prompt_tokens = None
        completion_tokens = None
        err_snip = None
        raw_text = ""
        unclosed = False
        try:
            completion, usage = chat_complete(
                endpoint,
                model_id,
                HUMANEVAL_SYSTEM,
                item["prompt"],
                max_tokens,
                params["temperature"],
                params["top_p"],
                params["seed"],
            )
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            unclosed = fence_unclosed(completion)
            raw_text = completion[:6000]
            code = extract_code(completion)
            program = build_program(item["prompt"], code, item["entry_point"], item["test"])
            ok, err = run_program(program)
            if not ok and err:
                err_snip = err[:400]
        except Exception as e:  # noqa: BLE001 - record and continue
            ok = False
            err_snip = str(e)[:400]
        wall_s = time.time() - t0
        attempts.append(
            {
                "attempt": k,
                "passed": ok,
                # stored so any scoring change can be re-applied OFFLINE
                # instead of costing another GPU run
                "response": raw_text,
                "fence_unclosed": unclosed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "wall_seconds": round(wall_s, 3),
                "temperature": params["temperature"],
                "seed": params["seed"],
                "error": err_snip if not ok else None,
            }
        )
        if ok:
            passed_at = k
            break

    total_completion_tokens = sum(a["completion_tokens"] or 0 for a in attempts)
    total_wall_s = sum(a["wall_seconds"] for a in attempts)
    return {
        "task_id": task_id,
        "index": item.get("_index"),
        "attempts": attempts,
        "n_attempts": len(attempts),
        "passed_at": passed_at,
        "total_completion_tokens": total_completion_tokens,
        "total_wall_s": round(total_wall_s, 3),
    }


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def wilson_ci95(passed, n):
    """Wilson score interval at 95% confidence, implemented inline (no scipy)."""
    if n == 0:
        return [0.0, 0.0]
    z = 1.959963984540054  # 97.5th percentile of standard normal
    p = passed / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return [max(0.0, lo), min(1.0, hi)]


def build_summary(results, max_attempts, tag, endpoint, model_id, wall_time_s, params):
    n_tasks = len(results)
    pass_at_n = {}
    for n in range(1, max_attempts + 1):
        passed = sum(1 for r in results if r["passed_at"] is not None and r["passed_at"] <= n)
        rate = passed / n_tasks if n_tasks else 0.0
        pass_at_n[str(n)] = {
            "passed": passed,
            "rate": rate,
            "ci95": wilson_ci95(passed, n_tasks),
        }

    solved = [r for r in results if r["passed_at"] is not None]
    mean_attempts_to_pass_among_solved = (
        sum(r["passed_at"] for r in solved) / len(solved) if solved else 0.0
    )
    # censored at max_attempts: unsolved tasks contribute the full attempt
    # budget spent (n_attempts == max_attempts), not an unbounded count.
    mean_attempts_all_tasks = (
        sum(r["n_attempts"] for r in results) / n_tasks if n_tasks else 0.0
    )

    total_completion_tokens = sum(r["total_completion_tokens"] for r in results)
    mean_completion_tokens_per_task = (
        total_completion_tokens / n_tasks if n_tasks else 0.0
    )

    # tokens spent on attempts >= 2, and solves gained after N=1
    tokens_attempts_2plus = 0
    for r in results:
        for a in r["attempts"]:
            if a["attempt"] >= 2:
                tokens_attempts_2plus += a["completion_tokens"] or 0
    solves_at_1 = pass_at_n.get("1", {}).get("passed", 0)
    solves_at_max = pass_at_n.get(str(max_attempts), {}).get("passed", 0)
    solves_gained = solves_at_max - solves_at_1
    tokens_per_additional_solve = (
        tokens_attempts_2plus / solves_gained if solves_gained > 0 else None
    )

    return {
        "tag": tag,
        "n_tasks": n_tasks,
        "max_attempts": max_attempts,
        "pass_at_N": pass_at_n,
        "mean_attempts_to_pass_among_solved": mean_attempts_to_pass_among_solved,
        "mean_attempts_all_tasks": mean_attempts_all_tasks,  # censored at max_attempts
        "mean_completion_tokens_per_task": mean_completion_tokens_per_task,
        "total_completion_tokens": total_completion_tokens,
        "tokens_per_additional_solve": tokens_per_additional_solve,
        "wall_time_s": round(wall_time_s, 2),
        "endpoint": endpoint,
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": params,
    }


# --------------------------------------------------------------------------
# resume / io
# --------------------------------------------------------------------------

def load_existing(out_path):
    """Load already-completed task results from --out (for resume)."""
    existing = {}
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing[rec["task_id"]] = rec
    return existing


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--endpoint", required=True, help="e.g. http://127.0.0.1:8080")
    ap.add_argument("--out", required=True, help="JSONL output path (per-task results)")
    ap.add_argument("--summary", required=True, help="JSON output path (aggregate summary)")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--limit", type=int, default=164)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temp-retry", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    t0 = time.time()

    model_info = get_model_info(args.endpoint)
    model_id = model_info["model_id"]

    items = load_humaneval(args.data_dir, args.limit)
    for i, item in enumerate(items):
        item["_index"] = i

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    summary_dir = os.path.dirname(os.path.abspath(args.summary)) or "."
    os.makedirs(summary_dir, exist_ok=True)

    existing = load_existing(args.out)

    def is_settled(rec):
        # A task needs no more work if it already passed, or if it already
        # burned the full attempt budget we are being asked for. Anything else
        # (e.g. 3 failed attempts on disk, --max-attempts 5 now) gets EXTENDED.
        return rec.get("passed_at") is not None or rec.get("n_attempts", 0) >= args.max_attempts

    todo = [item for item in items if not is_settled(existing.get(item["task_id"], {}))]
    n_extend = sum(1 for it in todo if it["task_id"] in existing)
    print(
        f"[reliability] {len(existing)} on disk, {len(todo)} to run "
        f"({n_extend} of them resumed/extended from a shallower run)",
        file=sys.stderr,
    )

    results_by_id = dict(existing)
    n_total = len(items)
    n_done = len(existing)

    # append mode for resumability; existing lines are already on disk
    out_f = open(args.out, "a")

    def process(item):
        return run_task(
            args.endpoint,
            model_id,
            item,
            args.max_attempts,
            args.max_tokens,
            args.temp_retry,
            args.top_p,
            prior=existing.get(item["task_id"]),
        )

    try:
        if args.concurrency > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(process, item): item for item in todo}
                for fut in as_completed(futures):
                    rec = fut.result()
                    results_by_id[rec["task_id"]] = rec
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    n_done += 1
                    print(
                        f"[reliability] {n_done}/{n_total} {rec['task_id']} "
                        f"passed_at={rec['passed_at']} attempts={rec['n_attempts']}",
                        file=sys.stderr,
                    )
        else:
            for item in todo:
                rec = process(item)
                results_by_id[rec["task_id"]] = rec
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                n_done += 1
                print(
                    f"[reliability] {n_done}/{n_total} {rec['task_id']} "
                    f"passed_at={rec['passed_at']} attempts={rec['n_attempts']}",
                    file=sys.stderr,
                )
    finally:
        out_f.close()

    # sort the JSONL file by original task index for stable ordering
    ordered = sorted(
        results_by_id.values(),
        key=lambda r: r.get("index") if r.get("index") is not None else 1 << 30,
    )
    with open(args.out, "w") as f:
        for rec in ordered:
            f.write(json.dumps(rec) + "\n")

    wall_time_s = time.time() - t0

    params = {
        "endpoint": args.endpoint,
        "out": args.out,
        "summary": args.summary,
        "max_attempts": args.max_attempts,
        "limit": args.limit,
        "data_dir": args.data_dir,
        "max_tokens": args.max_tokens,
        "temp_retry": args.temp_retry,
        "top_p": args.top_p,
        "concurrency": args.concurrency,
        "tag": args.tag,
    }

    summary = build_summary(
        ordered, args.max_attempts, args.tag, args.endpoint, model_id, wall_time_s, params
    )

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fixed small-suite quality eval against a running llama-server.

Requires the server to be started with --jinja (chat template support) and
reachable at --endpoint. Two tasks:

  a) GSM8K  — first N (default 50) problems from the official test set.
     Reference answer parsed from the "#### N" suffix; model answer is the
     last number found in the completion. temperature=0, max_tokens=1024.

  b) HumanEval — first N (default 20) tasks. Model completion executed
     against the task's `check()` function in a subprocess with a 10s
     timeout. Score = pass@1 (single sample per task).

Both tasks are driven through POST {endpoint}/v1/chat/completions.

Usage:
    python3 eval_quality.py --endpoint http://127.0.0.1:8080 \\
        [--limit-gsm8k 50] [--limit-humaneval 20] [--out results/eval.json]
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import requests

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
GSM8K_REF_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------
# data acquisition
# --------------------------------------------------------------------------

def ensure_file(url, dest_path, binary=True):
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[eval_quality] downloading {url} -> {dest_path}", file=sys.stderr)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    mode = "wb" if binary else "w"
    with open(dest_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dest_path


def load_gsm8k(data_dir, limit):
    path = ensure_file(GSM8K_URL, os.path.join(data_dir, "gsm8k_test.jsonl"))
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


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


def chat_complete(endpoint, model_id, system, user, max_tokens, timeout=600):
    """POST one chat turn, temperature=0, and return the answer text.

    Qwen3-family models default to an internal "thinking" mode that, in a
    llama.cpp/OpenAI-style response, streams into `reasoning_content`
    rather than `content` — at these token caps the whole budget can be
    consumed by the think block, leaving `content` empty even though the
    model is otherwise healthy. We pass the standard Qwen3
    chat_template_kwargs.enable_thinking=false switch to disable it (a
    no-op / ignored field on non-Qwen3 templates), matching the pattern in
    canary/canary.py. As a defensive fallback in case a server build
    ignores that switch, if `content` comes back empty but
    `reasoning_content` doesn't, we use `reasoning_content` instead of
    silently failing.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body = {
        "model": model_id or "default",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
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
    return content


# --------------------------------------------------------------------------
# GSM8K
# --------------------------------------------------------------------------

GSM8K_SYSTEM = (
    "Solve the math word problem step by step. "
    "On the final line, output only the final numeric answer, nothing else."
)


def gsm8k_reference(answer_field):
    m = GSM8K_REF_RE.search(answer_field)
    if not m:
        return None
    return m.group(1).replace(",", "")


def last_number(text):
    matches = NUM_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def numeric_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return False


def run_gsm8k(endpoint, model_id, items, max_tokens=1024):
    results = []
    correct = 0
    for i, item in enumerate(items):
        ref = gsm8k_reference(item["answer"])
        try:
            completion = chat_complete(
                endpoint, model_id, GSM8K_SYSTEM, item["question"], max_tokens
            )
            pred = last_number(completion)
            ok = numeric_eq(pred, ref) if pred is not None and ref is not None else False
            err = None
        except Exception as e:  # noqa: BLE001 - record and continue
            completion = None
            pred = None
            ok = False
            err = str(e)
        if ok:
            correct += 1
        results.append(
            {
                "index": i,
                "reference": ref,
                "predicted": pred,
                "correct": ok,
                "error": err,
            }
        )
        print(
            f"[eval_quality] gsm8k {i + 1}/{len(items)} correct={ok}", file=sys.stderr
        )
    score = correct / len(items) if items else None
    return {"n": len(items), "correct": correct, "score": score, "results": results}


# --------------------------------------------------------------------------
# HumanEval
# --------------------------------------------------------------------------

HUMANEVAL_SYSTEM = (
    "Complete the following Python function. Respond with a single Python "
    "code block containing the full function definition (including the "
    "signature and docstring exactly as given), with no additional "
    "explanation."
)


def extract_code(text):
    m = CODE_FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


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


def run_humaneval(endpoint, model_id, items, max_tokens=1024):
    results = []
    passed = 0
    for i, item in enumerate(items):
        task_id = item["task_id"]
        try:
            completion = chat_complete(
                endpoint, model_id, HUMANEVAL_SYSTEM, item["prompt"], max_tokens
            )
            code = extract_code(completion)
            program = build_program(
                item["prompt"], code, item["entry_point"], item["test"]
            )
            ok, err = run_program(program)
        except Exception as e:  # noqa: BLE001
            ok = False
            err = str(e)
        if ok:
            passed += 1
        results.append({"index": i, "task_id": task_id, "passed": ok, "error": err})
        print(
            f"[eval_quality] humaneval {i + 1}/{len(items)} ({task_id}) passed={ok}",
            file=sys.stderr,
        )
    score = passed / len(items) if items else None
    return {"n": len(items), "passed": passed, "score": score, "results": results}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(endpoint, limit_gsm8k=50, limit_humaneval=20, data_dir=DEFAULT_DATA_DIR, max_tokens=1024):
    """Run the full eval suite against a live server; return a result dict.

    Importable by run_matrix.py so it doesn't have to shell out and re-parse
    stdout.
    """
    t0 = time.time()

    model_info = get_model_info(endpoint)

    gsm8k_items = load_gsm8k(data_dir, limit_gsm8k)
    gsm8k_result = run_gsm8k(endpoint, model_info["model_id"], gsm8k_items, max_tokens)

    humaneval_items = load_humaneval(data_dir, limit_humaneval)
    humaneval_result = run_humaneval(endpoint, model_info["model_id"], humaneval_items, max_tokens)

    wall_time = time.time() - t0

    return {
        "endpoint": endpoint,
        "model_info": model_info,
        "gsm8k": gsm8k_result,
        "humaneval": humaneval_result,
        "aggregate": {
            "gsm8k_score": gsm8k_result["score"],
            "humaneval_score": humaneval_result["score"],
        },
        "wall_time_s": round(wall_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="e.g. http://127.0.0.1:8080")
    ap.add_argument("--limit-gsm8k", type=int, default=50)
    ap.add_argument("--limit-humaneval", type=int, default=20)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", help="write JSON result to this path (also printed to stdout)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    out = run(
        args.endpoint,
        limit_gsm8k=args.limit_gsm8k,
        limit_humaneval=args.limit_humaneval,
        data_dir=args.data_dir,
        max_tokens=args.max_tokens,
    )

    text = json.dumps(out, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

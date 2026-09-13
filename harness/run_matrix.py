#!/usr/bin/env python3
"""Drive the full measurement matrix: eval_quality + bench_speed per config
entry, appending one merged JSON line per entry to results/results.jsonl.

For each entry in configs/matrix.yaml:
  1. start llama-server on a free port with the entry's server_flags
  2. wait for /health
  3. run eval_quality.py against it
  4. stop the server
  5. run bench_speed.py (llama-bench) on the same gguf with bench_flags
  6. append one merged JSON row to the results file

Robust to server crashes/timeouts: each stage is wrapped so a failure is
recorded as a row with status != "ok" and the matrix continues with the
next entry, instead of aborting the whole run.

Usage:
    python3 run_matrix.py --config ../configs/matrix.yaml \\
        --out ../results/results.jsonl [--limit-gsm8k 50] [--limit-humaneval 20]
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

import bench_speed
import eval_quality
import yaml_lite

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HARNESS_DIR)
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "matrix.yaml")
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "results.jsonl")


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def resolve_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(REPO_ROOT, p)


def start_server(gguf_path, port, extra_flags, log_path):
    cmd = ["llama-server", "-m", gguf_path, "--port", str(port)] + shlex.split(extra_flags)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return cmd, proc, log_f


def wait_healthy(base_url, proc, timeout_s=120, interval_s=1.0):
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            return False, f"process exited during startup (code {proc.returncode})"
        try:
            r = requests.get(base_url + "/health", timeout=3)
            if r.status_code == 200:
                return True, None
        except requests.RequestException:
            pass
        time.sleep(interval_s)
    return False, f"health check timed out after {timeout_s}s"


def stop_server(proc, log_f, timeout_s=15):
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        try:
            log_f.close()
        except Exception:
            pass


def tail_file(path, n_bytes=2000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def run_entry(entry, args, log_dir):
    name = entry.get("name", "unnamed")
    gguf_path = resolve_path(entry["gguf_path"])
    server_flags = entry.get("server_flags", "")
    bench_flags = entry.get("bench_flags", "")

    row = {
        "name": name,
        "gguf_path": gguf_path,
        "server_flags": server_flags,
        "bench_flags": bench_flags,
        "status": "ok",
        "eval": None,
        "bench": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not os.path.isfile(gguf_path):
        row["status"] = "failed"
        row["error"] = f"gguf file not found: {gguf_path}"
        return row

    port = free_port()
    log_path = os.path.join(log_dir, f"{name}.server.log")
    print(f"[run_matrix] === {name} === starting server on port {port}", file=sys.stderr)

    cmd, proc, log_f = start_server(gguf_path, port, server_flags, log_path)
    try:
        healthy, err = wait_healthy(f"http://127.0.0.1:{port}", proc)
        if not healthy:
            row["status"] = "server_failed_to_start"
            row["error"] = err
            row["server_log_tail"] = tail_file(log_path)
            return row

        endpoint = f"http://127.0.0.1:{port}"
        try:
            print(f"[run_matrix] {name}: running eval_quality", file=sys.stderr)
            row["eval"] = eval_quality.run(
                endpoint,
                limit_gsm8k=args.limit_gsm8k,
                limit_humaneval=args.limit_humaneval,
                data_dir=args.data_dir,
                max_tokens=args.max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            row["status"] = "eval_failed"
            row["error"] = str(e)
            if proc.poll() is not None:
                row["server_log_tail"] = tail_file(log_path)
    finally:
        print(f"[run_matrix] {name}: stopping server", file=sys.stderr)
        stop_server(proc, log_f)

    try:
        print(f"[run_matrix] {name}: running bench_speed", file=sys.stderr)
        row["bench"] = bench_speed.run(
            gguf_path,
            flags=bench_flags,
            reps=args.bench_reps,
            pp_n=args.pp_n,
            tg_n=args.tg_n,
        )
        if not row["bench"].get("ok", False) and row["status"] == "ok":
            row["status"] = "bench_failed"
    except Exception as e:  # noqa: BLE001
        if row["status"] == "ok":
            row["status"] = "bench_failed"
        row["bench_error"] = str(e)

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="matrix.yaml path")
    ap.add_argument("--out", default=DEFAULT_OUT, help="results.jsonl path (appended)")
    ap.add_argument("--limit-gsm8k", type=int, default=50)
    ap.add_argument("--limit-humaneval", type=int, default=20)
    ap.add_argument("--data-dir", default=eval_quality.DEFAULT_DATA_DIR)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--bench-reps", type=int, default=3)
    ap.add_argument("--pp-n", type=int, default=512)
    ap.add_argument("--tg-n", type=int, default=128)
    ap.add_argument("--log-dir", default=None, help="dir for per-entry server logs (default: alongside --out)")
    args = ap.parse_args()

    entries = yaml_lite.load_matrix_yaml(args.config)
    print(f"[run_matrix] loaded {len(entries)} entries from {args.config}", file=sys.stderr)

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    log_dir = args.log_dir or os.path.join(os.path.dirname(os.path.abspath(out_path)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    n_ok = 0
    for entry in entries:
        row = run_entry(entry, args, log_dir)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        status = row["status"]
        n_ok += 1 if status == "ok" else 0
        print(f"[run_matrix] {row['name']}: status={status}", file=sys.stderr)

    print(f"[run_matrix] done: {n_ok}/{len(entries)} entries ok. results -> {out_path}", file=sys.stderr)
    return 0 if n_ok == len(entries) else 1


if __name__ == "__main__":
    sys.exit(main())

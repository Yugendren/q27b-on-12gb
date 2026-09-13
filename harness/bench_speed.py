#!/usr/bin/env python3
"""Speed benchmark wrapper around llama-bench.

Runs pp512 (prompt processing, 512 tok) and tg128 (text generation, 128 tok),
3 repetitions each, via `llama-bench -o json`. On Linux, polls
`nvidia-smi --query-gpu=memory.used` every 500ms in a background thread for
the duration of the run and records the peak (summed across GPUs if more
than one is present). On macOS there is no VRAM counter distinct from
unified system memory, so peak_vram_mb is reported as null.

Usage:
    python3 bench_speed.py --model /path/to/model.gguf --flags "-ngl 999 -c 4096"
"""
import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


def sha256_16(path, chunk_size=1 << 20):
    """Stream sha256 of a (possibly huge) file; return first 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


class VramMonitor:
    """Background poller of nvidia-smi memory.used. Linux/CUDA only."""

    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.peak_mb = None
        self._samples = 0

    def _poll_once(self):
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return
        if out.returncode != 0:
            return
        total = 0.0
        found = False
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                total += float(line)
                found = True
            except ValueError:
                continue
        if not found:
            return
        self._samples += 1
        if self.peak_mb is None or total > self.peak_mb:
            self.peak_mb = total

    def _run(self):
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def run_llama_bench(model_path, extra_flags, reps, pp_n, tg_n):
    cmd = [
        "llama-bench",
        "-m",
        model_path,
        "-p",
        str(pp_n),
        "-n",
        str(tg_n),
        "-r",
        str(reps),
        "-o",
        "json",
    ] + shlex.split(extra_flags)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return cmd, proc


def parse_bench_json(stdout_text):
    records = json.loads(stdout_text)
    pp_ts = None
    tg_ts = None
    build_number = None
    build_commit = None
    for rec in records:
        build_number = rec.get("build_number", build_number)
        build_commit = rec.get("build_commit", build_commit)
        if rec.get("n_prompt", 0) > 0 and rec.get("n_gen", 0) == 0:
            pp_ts = rec.get("avg_ts")
        elif rec.get("n_gen", 0) > 0 and rec.get("n_prompt", 0) == 0:
            tg_ts = rec.get("avg_ts")
    return pp_ts, tg_ts, build_number, build_commit


def run(model_path, flags="", reps=3, pp_n=512, tg_n=128):
    """Run the speed benchmark; return a result dict.

    Importable by run_matrix.py so it doesn't have to shell out and re-parse
    stdout.
    """
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        return {"ok": False, "error": f"model file not found: {model_path}"}

    system = platform.system()
    file_size_gb = round(os.path.getsize(model_path) / (1024 ** 3), 3)

    print(f"[bench_speed] hashing {model_path} ...", file=sys.stderr)
    sha16 = sha256_16(model_path)

    monitor = None
    if system == "Linux":
        monitor = VramMonitor(interval_s=0.5)
        monitor.start()

    print(f"[bench_speed] running llama-bench (reps={reps}) ...", file=sys.stderr)
    t0 = time.time()
    cmd, proc = run_llama_bench(model_path, flags, reps, pp_n, tg_n)
    elapsed = time.time() - t0

    peak_vram_mb = None
    if monitor is not None:
        monitor.stop()
        peak_vram_mb = monitor.peak_mb

    result = {
        "model_file": model_path,
        "sha256_16": sha16,
        "file_size_gb": file_size_gb,
        "flags": flags,
        "reps": reps,
        "pp_n": pp_n,
        "tg_n": tg_n,
        "pp_tok_s": None,
        "tg_tok_s": None,
        "peak_vram_mb": peak_vram_mb,
        "llama_cpp_build": None,
        "llama_cpp_commit": None,
        "wall_time_s": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": proc.returncode == 0,
        "cmd": cmd,
    }

    if proc.returncode != 0:
        result["error"] = "llama-bench exited non-zero"
        result["stderr_tail"] = proc.stderr[-2000:]
        return result

    try:
        pp_ts, tg_ts, build_number, build_commit = parse_bench_json(proc.stdout)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        result["ok"] = False
        result["error"] = f"failed to parse llama-bench json output: {e}"
        result["stdout_tail"] = proc.stdout[-2000:]
        return result

    result["pp_tok_s"] = pp_ts
    result["tg_tok_s"] = tg_ts
    result["llama_cpp_build"] = build_number
    result["llama_cpp_commit"] = build_commit
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to .gguf model file")
    ap.add_argument(
        "--flags",
        default="",
        help="extra flags passed through to llama-bench, e.g. '-ngl 999 -c 4096'",
    )
    ap.add_argument("--reps", type=int, default=3, help="repetitions per test (default 3)")
    ap.add_argument("--pp-n", type=int, default=512, help="prompt-processing tokens (default 512)")
    ap.add_argument("--tg-n", type=int, default=128, help="text-gen tokens (default 128)")
    ap.add_argument("--out", help="append result JSON line to this path instead of stdout")
    args = ap.parse_args()

    result = run(args.model, flags=args.flags, reps=args.reps, pp_n=args.pp_n, tg_n=args.tg_n)

    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(text, file=sys.stderr)
    else:
        print(text)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

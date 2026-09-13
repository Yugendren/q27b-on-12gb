#!/usr/bin/env python3
"""Emit a JSON hardware profile of the current machine.

Cross-platform: macOS (Apple Silicon) now, Linux/CUDA later.
Stdlib only.

Usage:
    python3 profile_hw.py [--out path.json]
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def _run(cmd):
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _bytes_to_gb(n):
    return round(n / (1024 ** 3), 2)


def cpu_model(system):
    if system == "Darwin":
        v = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if v:
            return v
        # Apple Silicon chip name fallback via hw.model (e.g. "Mac16,10")
        return _run(["sysctl", "-n", "hw.model"]) or "unknown"
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or "unknown"
    return platform.processor() or "unknown"


def ram_gb(system):
    if system == "Darwin":
        v = _run(["sysctl", "-n", "hw.memsize"])
        if v and v.isdigit():
            return _bytes_to_gb(int(v))
        return None
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 2)
        except OSError:
            pass
        return None
    return None


def apple_chip_name():
    out = _run(["system_profiler", "SPHardwareDataType"])
    if out:
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Chip:"):
                return line.split(":", 1)[1].strip()
    # fall back to brand string (already "Apple M4" style)
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"])


def gpus_darwin():
    chip = apple_chip_name() or "Apple Silicon (unknown model)"
    unified_mem = ram_gb("Darwin")
    return [
        {
            "name": chip,
            "vram_total_gb": unified_mem,
            "unified_memory": True,
        }
    ]


def gpus_linux():
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if out is None:
        return []
    gpus = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        name, mem_mib = parts
        try:
            mem_gb = round(float(mem_mib) / 1024, 2)
        except ValueError:
            mem_gb = None
        gpus.append({"name": name, "vram_total_gb": mem_gb, "unified_memory": False})
    return gpus


def disk_free_gb(path="/"):
    try:
        total, used, free = shutil.disk_usage(path)
        return _bytes_to_gb(free)
    except OSError:
        return None


def build_profile():
    system = platform.system()
    profile = {
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "model": cpu_model(system),
            "cores_logical": None,
        },
        "ram_gb": ram_gb(system),
        "gpus": [],
        "disk_free_gb": disk_free_gb("/"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        import os

        profile["cpu"]["cores_logical"] = os.cpu_count()
    except Exception:
        pass

    if system == "Darwin":
        profile["gpus"] = gpus_darwin()
    elif system == "Linux":
        profile["gpus"] = gpus_linux()
    else:
        profile["gpus"] = []

    return profile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write JSON to this path instead of stdout")
    args = ap.parse_args()

    profile = build_profile()
    text = json.dumps(profile, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())

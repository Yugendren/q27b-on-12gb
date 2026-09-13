#!/usr/bin/env python3
"""MTP losslessness check for Qwen3.6-35B-A3B.

Why this exists: FINDINGS (2026-09-02, speed re-certification) raised an open
flag -- at n>=3 the 27B showed greedy non-reproducibility under MTP, and the
standing rule is that deep-MTP losslessness needs a dedicated check before any
n>2 config ships. We certify QUALITY with --spec-type none (the protocol of
record), so shipping with MTP on is only justified if MTP is bit-identical to
no-spec under greedy.

Method: for each probe prompt, generate greedily (temperature=0, top_k=1,
seed=1234) against a no-spec server and against an MTP server, and compare the
completions byte-for-byte. Speculative decoding is *supposed* to be exactly
lossless (the verify step rejects any draft token the target would not have
produced), so any mismatch is a real defect, not noise.

Usage:
  python3 mtp_identity.py capture --port 18096 --tag nospec --out /path/a.json
  python3 mtp_identity.py capture --port 18096 --tag mtp2   --out /path/b.json
  python3 mtp_identity.py compare /path/a.json /path/b.json
"""
import argparse, hashlib, json, sys, urllib.request

PROBES = [
    ("code_btree",
     "Write a complete Python module implementing a persistent B-tree index on disk. "
     "Include: a Node class with serialization to fixed-size pages, insert with node "
     "splitting, search, an in-order range scan generator, a free-page list, and a "
     "crash-safe write path using a write-ahead log. Full type hints, docstrings, and "
     "a __main__ demo that inserts 1000 keys and verifies them. Do not abbreviate."),
    ("code_lsm",
     "Write a Python module implementing a disk-backed LSM-tree memtable with a skip "
     "list, a WAL append path, and an SSTable flush that writes a sparse index. Full "
     "type hints, docstrings, no abbreviation."),
    ("struct_json",
     "Output ONLY a JSON object describing a fictional container registry: a metadata "
     "block with schema_version, registry_url and generated_at, and a repositories "
     "array of 6 entries each with name, visibility, pull_count, and a tags array "
     "whose entries have tag, digest, size_bytes and pushed_at."),
    ("prose_ssd",
     "Explain to a systems engineer why write amplification in flash SSDs rises as the "
     "drive fills, and what over-provisioning and TRIM each do about it. Cover garbage "
     "collection, block erase granularity, and steady-state behaviour. Four paragraphs, "
     "no bullet lists."),
    ("humaneval_like",
     "Complete this Python function. Return only the full function.\n\n"
     "def is_prime_sum_pair(n: int) -> tuple[int, int] | None:\n"
     "    \"\"\"Return a pair of primes summing to the even integer n, or None.\"\"\"\n"),
]

MAXTOK = 512


def capture(port, tag, out):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    recs = {}
    for pid, prompt in PROBES:
        body = {"messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": MAXTOK,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        t = d.get("timings", {})
        recs[pid] = {
            "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "n_pred": t.get("predicted_n"),
            "decode_tps": t.get("predicted_per_second"),
            "finish": d["choices"][0].get("finish_reason"),
            "reasoning_len": len(msg.get("reasoning_content") or ""),
        }
        print(f"  {pid}: n={recs[pid]['n_pred']} sha={recs[pid]['sha256'][:16]} "
              f"{recs[pid]['decode_tps']:.2f} t/s finish={recs[pid]['finish']}", flush=True)
    json.dump({"tag": tag, "port": port, "records": recs}, open(out, "w"), indent=1)
    print(f"wrote {out}")


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    ids = [p for p, _ in PROBES]
    ident = 0
    print(f"\n{'probe':16s} {'n_a':>5s} {'n_b':>5s} {a['tag']:>12s} {b['tag']:>12s}  identical")
    for pid in ids:
        ra, rb = a["records"].get(pid), b["records"].get(pid)
        if not ra or not rb:
            print(f"{pid:16s}  MISSING")
            continue
        same = ra["sha256"] == rb["sha256"]
        ident += same
        print(f"{pid:16s} {ra['n_pred']:>5} {rb['n_pred']:>5} "
              f"{ra['sha256'][:12]:>12s} {rb['sha256'][:12]:>12s}  {same}")
        if not same:
            ca, cb = ra["content"], rb["content"]
            k = next((i for i in range(min(len(ca), len(cb))) if ca[i] != cb[i]),
                     min(len(ca), len(cb)))
            print(f"    first divergence at char {k}:")
            print(f"      {a['tag']}: ...{ca[max(0,k-60):k+60]!r}")
            print(f"      {b['tag']}: ...{cb[max(0,k-60):k+60]!r}")
    print(f"\nIDENTICAL {ident}/{len(ids)} probes")
    print("VERDICT:", "MTP IS LOSSLESS (byte-identical greedy)" if ident == len(ids)
          else "*** MTP DIVERGES FROM NO-SPEC — do not ship MTP on quality claims ***")
    return 0 if ident == len(ids) else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--port", default="18096")
    c.add_argument("--tag", required=True)
    c.add_argument("--out", required=True)
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    args = ap.parse_args()
    if args.mode == "capture":
        capture(args.port, args.tag, args.out)
    else:
        sys.exit(compare(args.a, args.b))


if __name__ == "__main__":
    main()

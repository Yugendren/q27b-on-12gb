#!/usr/bin/env python3
"""Minimal dependency-free GGUF header reader.

Prints: architecture, all KV metadata (truncated), and a tensor-name census.
Used here to answer one gate question: does this GGUF carry the MTP / nextn
prediction head, and does it declare an arch our llama.cpp build knows?
"""
import struct, sys, json

# GGUF value type enum
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
_FMT = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i",
        T_F32: "<f", T_BOOL: "<?", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
_SZ = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4,
       T_F32: 4, T_BOOL: 1, T_U64: 8, T_I64: 8, T_F64: 8}


class R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def s(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")

    def val(self, t):
        if t == T_STR:
            return self.s()
        if t == T_ARR:
            et = self.u32()
            n = self.u64()
            if et == T_STR:
                # avoid materialising 250k-token vocabs
                if n > 32:
                    for _ in range(n):
                        self.s()
                    return f"<array[str] n={n} skipped>"
                return [self.s() for _ in range(n)]
            sz = _SZ[et]
            buf = self.raw(sz * n)
            if n > 32:
                return f"<array[t{et}] n={n} skipped>"
            return list(struct.unpack("<" + _FMT[et][1] * n, buf))
        return struct.unpack(_FMT[t], self.raw(_SZ[t]))[0]


def main(path):
    with open(path, "rb") as f:
        r = R(f)
        magic = r.raw(4)
        if magic != b"GGUF":
            sys.exit(f"not a GGUF file (magic={magic!r})")
        ver = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        print(f"== {path}")
        print(f"gguf_version={ver} n_tensors={n_tensors} n_kv={n_kv}\n")

        kv = {}
        for _ in range(n_kv):
            k = r.s()
            t = r.u32()
            kv[k] = r.val(t)

        print("---- METADATA ----")
        for k, v in kv.items():
            sv = str(v)
            if len(sv) > 160:
                sv = sv[:160] + f"...<{len(sv)} chars>"
            print(f"  {k} = {sv}")

        # tensor names
        names = []
        for _ in range(n_tensors):
            nm = r.s()
            nd = r.u32()
            for _ in range(nd):
                r.u64()
            r.u32()   # ggml type
            r.u64()   # offset
            names.append(nm)

    print("\n---- MTP / NEXTN GATE ----")
    hits = [n for n in names if any(p in n.lower() for p in ("nextn", "mtp", "eh_proj", "enorm", "hnorm", "shared_head"))]
    print(f"  tensors matching nextn/mtp/eh_proj/enorm/hnorm/shared_head: {len(hits)}")
    for n in sorted(set(hits)):
        print(f"    {n}")
    kvhits = {k: v for k, v in kv.items() if any(p in k.lower() for p in ("nextn", "mtp", "predict"))}
    print(f"  metadata keys matching nextn/mtp/predict: {kvhits}")

    print("\n---- TENSOR NAME CENSUS (prefix families) ----")
    import re
    fam = {}
    for n in names:
        g = re.sub(r"\d+", "N", n)
        fam[g] = fam.get(g, 0) + 1
    for g, c in sorted(fam.items(), key=lambda x: -x[1]):
        print(f"  {c:5d}  {g}")


if __name__ == "__main__":
    main(sys.argv[1])

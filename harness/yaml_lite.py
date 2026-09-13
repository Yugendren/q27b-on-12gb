"""Minimal YAML subset parser — stdlib only, no PyYAML dependency.

Supports exactly the shape run_matrix.py needs: a top-level list of flat
string-valued mappings, e.g.

    - name: qwen3-0.6b-q4
      gguf_path: models/Qwen3-0.6B-Q4_K_M.gguf
      server_flags: "--jinja -c 4096"
      bench_flags: "-c 4096"

    - name: another-entry
      gguf_path: models/other.gguf
      server_flags: ""
      bench_flags: ""

Rules:
  - blank lines and lines starting with '#' (after stripping) are ignored
  - a new list item starts with '- key: value'
  - subsequent 'key: value' lines (indented) belong to the current item
    until the next '- ' line
  - values may optionally be wrapped in single or double quotes
  - no nested lists/mappings, no multi-line scalars, no anchors/tags

This is intentionally not a general YAML parser. If matrix.yaml needs more
than this shape, switch to PyYAML.
"""


def _parse_kv(s):
    key, sep, val = s.partition(":")
    if not sep:
        raise ValueError(f"expected 'key: value', got: {s!r}")
    key = key.strip()
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        val = val[1:-1]
    return key, val


def load_matrix_yaml(path):
    items = []
    current = None
    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- "):
                if current is not None:
                    items.append(current)
                current = {}
                k, v = _parse_kv(stripped[2:])
                current[k] = v
            elif ":" in stripped and current is not None:
                k, v = _parse_kv(stripped)
                current[k] = v
            else:
                raise ValueError(f"yaml_lite: cannot parse line: {line!r}")
    if current is not None:
        items.append(current)
    return items

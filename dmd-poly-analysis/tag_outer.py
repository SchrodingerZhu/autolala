#!/usr/bin/env python3
"""Tag the outermost analyzable affine.for in an MLIR file with {dmd.extract}.

Prefers the `loop_once` wrapper loop if present (matches how the AutoLALA
polybench examples are framed); otherwise tags the first top-level affine.for.
Idempotent: if a dmd.extract tag already exists, leaves the file unchanged.
"""
import re, sys


def tag(src: str) -> str:
    if "dmd.extract" in src:
        return src
    # locate the loop_once header, else the first affine.for
    m = re.search(r"affine\.for\s+%loop_once\s*=\s*0\s*to\s*1\s*\{", src)
    if not m:
        m = re.search(r"affine\.for\b[^\{]*\{", src)
    if not m:
        raise SystemExit("no affine.for found to tag")
    # walk braces from the opening { of that loop to its matching close
    i = m.end() - 1  # index of the '{'
    depth = 0
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    # insert the attribute right after the matching close brace
    return src[: i + 1] + " {dmd.extract}" + src[i + 1 :]


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        out = tag(f.read())
    with open(path, "w") as f:
        f.write(out)
    print(f"tagged {path}")

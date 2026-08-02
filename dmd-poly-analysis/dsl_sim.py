#!/usr/bin/env python3
"""Tiny reference interpreter for the AutoLALA DSL: exact trace, block-8
reuse-distance histogram, warm/cold counts. Ground truth for self-checks."""
import re
from collections import Counter

_FOR = re.compile(r"^for\s+(\w+)\s+in\s+(.+?)\s*\.\.\s*(.+?)\s*\{$")
_ACC = re.compile(r"^(read|write)\s+(\w+)\s*\[(.*)\]\s*;$")


def parse(dsl_text):
    params, arrays, body = [], {}, []
    lines = [l.strip() for l in dsl_text.splitlines() if l.strip()]
    stack = [body]
    for line in lines:
        if line.startswith("params"):
            params = [p.strip() for p in line[6:].rstrip(";").split(",")]
        elif line.startswith("array"):
            m = re.match(r"array\s+(\w+)\s*\[(.*)\]\s*;", line)
            arrays[m.group(1)] = [d.strip() for d in m.group(2).split(",")]
        elif _FOR.match(line):
            m = _FOR.match(line)
            node = ("for", m.group(1), m.group(2), m.group(3), [])
            stack[-1].append(node)
            stack.append(node[4])
        elif line == "}":
            stack.pop()
        elif _ACC.match(line):
            m = _ACC.match(line)
            stack[-1].append(("acc", m.group(1), m.group(2),
                              [s.strip() for s in m.group(3).split(",")]))
        else:
            raise ValueError(f"bad line {line!r}")
    return params, arrays, body


def trace(dsl_text, binds, block=8):
    """Yield block ids of the access trace under `binds` (param -> int)."""
    params, arrays, body = parse(dsl_text)
    dims = {a: [eval(d, {}, binds) for d in ds] for a, ds in arrays.items()}
    out = []

    def sub_eval(expr, env):
        return eval(expr.replace("/", "//"), {}, env)

    def run(block_stmts, env):
        for node in block_stmts:
            if node[0] == "for":
                _, var, lo, hi, sub = node
                lo_v = sub_eval(lo, env)
                hi_v = sub_eval(hi, env)
                for v in range(lo_v, hi_v):
                    env[var] = v
                    run(sub, env)
                env.pop(var, None)
            else:
                _, kind, arr, subs = node
                idx = [sub_eval(s, env) for s in subs]
                # The analyzer blocks only the innermost dimension: every
                # array row starts on a fresh cache line (the paper's padded
                # layout, innermost dim padded to a line multiple).
                out.append((arr, tuple(idx[:-1]), idx[-1] // block))

    run(body, dict(binds))
    return out


def stats(dsl_text, binds, block=8, repeat=1):
    tr = []
    for _ in range(repeat):
        tr.extend(trace(dsl_text, binds, block))
    last, hist, warm = {}, Counter(), 0
    period = len(tr) // repeat
    for t, b in enumerate(tr):
        if b in last:
            rd = len(set(tr[last[b] + 1: t + 1]))
            if repeat == 1 or t >= period:  # steady state for repeat=2
                hist[rd] += 1
                warm += 1
        last[b] = t
    total = period
    if repeat == 1:
        cold = total - warm
    else:
        cold = 0
    return dict(total=total, warm=warm, cold=cold, hist=dict(sorted(hist.items())))

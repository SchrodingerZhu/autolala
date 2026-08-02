#!/usr/bin/env python3
"""Domain-driven enumeration of a bin's exposed-iterator points.

The rectangular sweep over [0, lim)^k is hopeless for k = 3 (and slow for
triangular k = 2). isl-printed region guards are conjunctions of linear
comparisons and residue pins, so we extract, per iterator,
  * lower/upper bound expressions (may reference parameters and *earlier*
    iterators),
  * a single `(a·v + b) mod m = r` residue when present,
and enumerate nested over exactly the implied ranges. Every emitted point is
re-checked against the full guard, so an unparsed constraint only costs
speed, never correctness; an unparsed *bound* falls back to [0, lim).
"""
import re
from qpeval import compile_expr, call, free_names

_MOD = re.compile(r"^\(?\s*(-?\d+)?\s*\+?\s*(-?)\s*(\w+)\s*\)?\s*mod\s*(\d+)$")
_CMP = re.compile(r"(<=|>=|<|>|=)")


def _mod_residue(lhs, rhs, var):
    """`lhs mod m = r` with lhs linear in var (coeff ±1) -> (m, residue)."""
    m = _MOD.match(lhs.strip())
    if not m:
        return None
    c, sign, v, mod = m.groups()
    if v != var:
        return None
    c = int(c or 0)
    mod = int(mod)
    r = int(rhs)
    # (c ± v) ≡ r  (mod mod)  ->  v ≡ ±(r - c)
    res = (r - c) if sign != "-" else (c - r)
    return mod, res % mod


class IterBounds:
    def __init__(self, var):
        self.var = var
        self.los = []   # compiled exprs, v >= e
        self.his = []   # compiled exprs, v <= e
        self.mod = None  # (m, r)


def split_top_conjuncts(src):
    """Split on ' and ' at parenthesis depth 0 only — constraints inside a
    parenthesized disjunct must stay with it (they hold in one branch only,
    so harvesting them as global bounds would drop the other branch)."""
    out, depth, start, i = [], 0, 0, 0
    while i < len(src):
        ch = src[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and src.startswith(" and ", i):
            out.append(src[start:i])
            i += 5
            start = i
            continue
        i += 1
    out.append(src[start:])
    return out


def parse_bounds(guard_src, iters):
    """Best-effort per-iterator bounds from a conjunction. Returns
    {var: IterBounds}; anything unparsed is simply absent."""
    bounds = {v: IterBounds(v) for v in iters}
    for conj in split_top_conjuncts(guard_src):
        conj = conj.strip()
        if " or " in conj or conj.count("(") != conj.count(")"):
            continue
        if " mod " in conj:
            parts = conj.split("=")
            if len(parts) == 2:
                for v in iters:
                    mr = _mod_residue(parts[0], parts[1].strip(), v)
                    if mr:
                        bounds[v].mod = mr
            continue
        # comparison chain: e0 op e1 op e2 ...
        pieces = _CMP.split(conj)
        if len(pieces) < 3:
            continue
        exprs = [p.strip() for p in pieces[0::2]]
        ops = [p for p in pieces[1::2]]
        for (a, op, b) in zip(exprs, ops, exprs[1:]):
            for v in iters:
                if a == v:      # v op b
                    if op in ("<=",):
                        bounds[v].his.append(b)
                    elif op == "<":
                        bounds[v].his.append(f"({b}) - 1")
                    elif op in (">=",):
                        bounds[v].los.append(b)
                    elif op == ">":
                        bounds[v].los.append(f"({b}) + 1")
                    elif op == "=":
                        bounds[v].los.append(b)
                        bounds[v].his.append(b)
                if b == v:      # a op v
                    if op in ("<=",):
                        bounds[v].los.append(a)
                    elif op == "<":
                        bounds[v].los.append(f"({a}) + 1")
                    elif op in (">=",):
                        bounds[v].his.append(a)
                    elif op == ">":
                        bounds[v].his.append(f"({a}) - 1")
                    elif op == "=":
                        bounds[v].los.append(a)
                        bounds[v].his.append(a)
    return bounds


def order_iters(iters, bounds):
    """Topological-ish order: an iterator whose bounds mention another comes
    after it. Falls back to given order on cycles."""
    deps = {}
    for v in iters:
        b = bounds.get(v)
        names = set()
        if b:
            for e in b.los + b.his:
                names |= free_names(e)
        deps[v] = names & set(iters) - {v}
    out, left = [], list(iters)
    while left:
        progressed = False
        for v in list(left):
            if deps[v] <= set(out):
                out.append(v)
                left.remove(v)
                progressed = True
        if not progressed:
            out.extend(left)
            break
    return out


def enum_region(domain_src, count_src, iters, env, lim, mode="exact",
                conv=None):
    """Yield (point_env, count) for every integer point of one region.
    `env` maps parameters (and will be extended per iterator); `lim` is the
    fallback range. Soundness: full guard re-checked at every point."""
    guard = compile_expr(domain_src, "guard", mode)
    count = compile_expr(count_src, "val", mode)
    bounds = parse_bounds(domain_src, iters)
    order = order_iters(iters, bounds)
    conv = conv or (lambda x: x)
    lo_fns = {v: [compile_expr(e, "val", mode) for e in bounds[v].los]
              for v in order}
    hi_fns = {v: [compile_expr(e, "val", mode) for e in bounds[v].his]
              for v in order}

    import math as _math

    def rec(k, env):
        if k == len(order):
            if call(guard, env):
                c = call(count, env)
                if c:
                    yield dict(env), c
            return
        v = order[k]
        lo = 0
        for f in lo_fns[v]:
            if set(f[1]) <= env.keys():
                lo = max(lo, _math.ceil(call(f, env)))
        hi = lim
        for f in hi_fns[v]:
            if set(f[1]) <= env.keys():
                hi = min(hi, _math.floor(call(f, env)))
        step = 1
        start = lo
        if bounds[v].mod:
            m, r = bounds[v].mod
            start = lo + ((r - lo) % m)
            step = m
        for iv in range(start, hi + 1, step):
            env[v] = conv(iv)
            yield from rec(k + 1, env)
        env.pop(v, None)

    yield from rec(0, dict(env))

#!/usr/bin/env python3
"""Exact evaluation of dmd-cli plain formulas.

Grammar handled (the analyzer's `*_plain` rendering):
  expr      := head? segment*            (top-level additive pieces)
  segment   := ('+'|'-') '[' guard ']' ' => ' value
  head      := value                     (unguarded leading polynomial)
  value     := rational polynomial with floor(...), ^ powers
  guard     := comparisons chained with and/or/not, '=' equality, 'mod',
               floor(...)

Everything evaluates over exact `Fraction`s. Expressions are compiled once
into Python lambdas keyed by their free variables, so per-point enumeration
(hundreds of thousands of guard/value evaluations) stays fast.
"""
import math
import re
from fractions import Fraction

FUNCS = {"floor", "sqrt", "min", "max", "mod", "and", "or", "not"}

_COMPILED = {}


def free_names(expr):
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)) - FUNCS


def _floor(x):
    return math.floor(x)


def _sqrt(x):
    return math.sqrt(x)


def _pythonize(expr, wrap_ints):
    """Plain syntax -> python: '^'->'**', ' mod '->' % ', '='->'==' (guards),
    isl's implicit products '2p1' -> '2*p1', integer literals wrapped into
    Fraction unless directly an exponent."""
    expr = expr.replace("^", "**").replace(" mod ", " % ")
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    expr = re.sub(r"(\d)([A-Za-z_])", r"\1*\2", expr)
    if wrap_ints:
        expr = re.sub(r"(?<![\w.*])(\d+)", r"F(\1)", expr)
    return expr


def is_dyadic(expr):
    """True when every rational literal has a power-of-two denominator, so
    IEEE floats evaluate the expression exactly (integers < 2^53)."""
    for d in re.findall(r"/\s*(\d+)", expr):
        k = int(d)
        if k & (k - 1):
            return False
    return True


def compile_expr(expr, kind, mode="exact"):
    """Compile a value ('val') or guard ('guard') expression into a lambda
    taking keyword arguments for every free variable. mode='fast' uses float
    arithmetic — exact for dyadic formulas (checked by the caller via
    is_dyadic) and ~10x faster for enumeration sweeps."""
    key = (expr, kind, mode)
    if key in _COMPILED:
        return _COMPILED[key]
    names = sorted(free_names(expr))
    if mode == "fast":
        body = _pythonize(expr, wrap_ints=False)
        env = {"floor": math.floor, "sqrt": math.sqrt,
               "min": min, "max": max, "__builtins__": {}}
    else:
        body = _pythonize(expr, wrap_ints=True)
        env = {"F": Fraction, "floor": _floor, "sqrt": _sqrt,
               "min": min, "max": max, "__builtins__": {}}
    src = f"lambda {', '.join(names)}: ({body})" if names else f"lambda: ({body})"
    fn = eval(src, env)
    compiled = (fn, names)
    _COMPILED[key] = compiled
    return compiled


def call(compiled, env):
    fn, names = compiled
    return fn(**{n: env[n] for n in names})


_SEGMENT = re.compile(
    r"(?:^|(?P<sign>[+-])\s*)\[(?P<guard>[^\]]*)\]\s*=>\s*"
    r"(?P<value>.*?)(?=\s*[+-]\s*\[|$)",
    re.S,
)


class Piecewise:
    """A parsed plain formula: optional unguarded head + guarded segments."""

    def __init__(self, expr):
        self.src = expr
        expr = expr.strip()
        i = expr.find("[")
        head_src = expr if i < 0 else expr[:i]
        head_src = head_src.strip().rstrip("+-").strip()
        self.head = compile_expr(head_src, "val") if head_src else None
        self.segments = []
        if i >= 0:
            for m in _SEGMENT.finditer(expr[i:]):
                sign = -1 if m.group("sign") == "-" else 1
                guard = compile_expr(m.group("guard"), "guard")
                value = compile_expr(m.group("value").strip(), "val")
                self.segments.append((sign, guard, value))

    def __call__(self, env):
        total = call(self.head, env) if self.head else Fraction(0)
        for sign, guard, value in self.segments:
            if call(guard, env):
                total += sign * call(value, env)
        return total

    def free_names(self):
        names = set()
        if self.head:
            names |= set(self.head[1])
        for _, guard, value in self.segments:
            names |= set(guard[1]) | set(value[1])
        return names


def anchor(f, base, h, max_deg=8):
    """Exact degree and leading coefficient of the polynomial N -> f(N) on the
    residue class N = base + k*h, via finite differences over Fractions.
    Returns (degree, leading: Fraction, exact: bool). degree -1 means
    identically zero; exact=False flags a non-polynomial residual (the
    difference table did not terminate), in which case degree/leading describe
    the highest stable difference and must not be trusted blindly."""
    vals = [Fraction(f(base + k * h)) for k in range(max_deg + 2)]
    table = [vals]
    while len(table[-1]) > 1:
        prev = table[-1]
        table.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    deg = -1
    for d in range(len(table) - 1, -1, -1):
        if any(x != 0 for x in table[d]):
            deg = d
            break
    if deg < 0:
        return -1, Fraction(0), True
    exact = all(x == table[deg][0] for x in table[deg])
    lead = table[deg][0] / (math.factorial(deg) * Fraction(h) ** deg)
    return deg, lead, exact


_FLOOR_ARG = re.compile(r"floor\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


def detect_modulus(*exprs):
    """Least common multiple of the residue structure: `mod k` moduli and
    denominators appearing *inside floor arguments* (with 8, the cache line,
    always included). Free-standing coefficient denominators (e.g. the /3 of a
    triangular count) do not create residue classes and are ignored."""
    h = 8
    for expr in exprs:
        for k in re.findall(r"mod\s+(\d+)", expr):
            h = math.lcm(h, int(k))
        for arg in _FLOOR_ARG.findall(expr):
            for k in re.findall(r"/\s*(\d+)", arg):
                h = math.lcm(h, int(k))
    return h

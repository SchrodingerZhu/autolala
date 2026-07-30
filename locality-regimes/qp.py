"""Parse and evaluate the piecewise quasi-polynomial strings printed by dmd-cli.

The plain-text format is a sum of pieces:

    [cond] => expr + [cond] => expr + ... + expr

where each guarded piece contributes expr when cond holds and 0 otherwise
(regions are disjoint, so summing pieces equals selecting the active piece),
and a bare expr contributes unconditionally.  Conditions use  and/or,
comparisons (=, <, <=, >, >=), and  (e) mod k = r.  Expressions use
+ - * / ^ with integer literals and parameter names.

Everything is evaluated exactly over Fractions.  This module deliberately
refuses strings where a piece is multiplied by a coefficient (e.g. the
"warm" field can contain `2 * [cond] => ...` after printer-side merging);
callers should derive such quantities from the distributions instead.
"""
import math
import re
from fractions import Fraction

_INT = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_IMPLICIT_MUL = re.compile(r"(?<![\w.])(\d+)(?=[A-Za-z_])")
_RESERVED = {"and", "or", "mod", "F", "floor", "ceil"}


def _floor(x):
    return Fraction(math.floor(x))


def _ceil(x):
    return Fraction(math.ceil(x))


class QPError(ValueError):
    pass


def _split_pieces(s):
    """Split a piecewise string into signed segments at top-level '+ [' / '- ['.

    Returns a list of (sign, segment) where segment is either
    '[cond] => expr' or a bare 'expr'.
    """
    s = s.strip()
    if not s:
        raise QPError("empty expression")
    # Find split points: '+' or '-' at paren-depth 0 followed by '['.
    depth = 0
    splits = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "[" and depth == 0 and i > 0:
            # walk back over whitespace to the joining operator
            j = i - 1
            while j >= 0 and s[j] == " ":
                j -= 1
            if j >= 0 and s[j] in "+-":
                # make sure the operator is a joiner, not part of an expr
                # (it is a joiner exactly when it sits at depth 0, which it
                # does here), and not something like '2 * [' which we refuse.
                splits.append((j, s[j]))
            elif j >= 0 and s[j] == "*":
                raise QPError("coefficient-multiplied piece (printer merge); "
                              "derive this quantity from the distributions")
        i += 1
    segments = []
    start = 0
    sign = 1
    for pos, op in splits:
        segments.append((sign, s[start:pos].strip()))
        sign = 1 if op == "+" else -1
        start = pos + 1
    segments.append((sign, s[start:].strip()))
    return [(sg, seg) for sg, seg in segments if seg]


def _pythonize_cond(cond):
    cond = _IMPLICIT_MUL.sub(r"\1*", cond)
    cond = cond.replace(" mod ", " % ")
    # '=' -> '==' but keep >=, <=, ==, !=
    cond = re.sub(r"(?<![<>=!])=(?!=)", "==", cond)
    return cond


def _pythonize_expr(expr):
    expr = _IMPLICIT_MUL.sub(r"\1*", expr)
    expr = expr.replace(" mod ", " % ")
    return expr.replace("^", "**")


def _wrap_ints(src):
    return _INT.sub(r"F(\1)", src)


class Piecewise:
    """A compiled piecewise quasi-polynomial; call with a param->value dict."""

    def __init__(self, text):
        self.text = text
        self.pieces = []          # (sign, cond_code_or_None, expr_code)
        self.params = set()
        for sign, seg in _split_pieces(text):
            if seg.startswith("["):
                close = seg.index("]")
                cond = seg[1:close]
                rest = seg[close + 1:].strip()
                if not rest.startswith("=>"):
                    raise QPError(f"malformed piece: {seg[:80]}")
                expr = rest[2:].strip()
                cond_src = _wrap_ints(_pythonize_cond(cond))
                cond_code = compile(cond_src, "<cond>", "eval")
            else:
                cond_code, expr = None, seg
            expr_src = _wrap_ints(_pythonize_expr(expr))
            expr_code = compile(expr_src, "<expr>", "eval")
            self.pieces.append((sign, cond_code, expr_code))
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr + " " +
                                   (cond if cond_code else "")):
                if name not in _RESERVED:
                    self.params.add(name)

    def __call__(self, env):
        genv = {"__builtins__": {}, "F": Fraction,
                "floor": _floor, "ceil": _ceil}
        genv.update({k: Fraction(v) for k, v in env.items()})
        val = Fraction(0)
        for sign, cond_code, expr_code in self.pieces:
            if cond_code is not None and not eval(cond_code, genv):
                continue
            val += sign * eval(expr_code, genv)
        return val


def fit_polynomial(f, max_degree=8, base=1024, step=1024, verify=3):
    """Recover exact polynomial coefficients of a callable f(n)->Fraction.

    Samples f at n = base, base+step, ... on the caller-chosen residue class
    (choose base/step as multiples of any moduli in the QP so a single
    polynomial branch is active).  Tries degrees 0..max_degree, solving an
    exact Vandermonde system and verifying on `verify` extra points.
    Returns coefficient list c[0..d] (c[i] multiplies n^i) or None.
    """
    pts = [base + i * step for i in range(max_degree + 1 + verify)]
    vals = [f(n) for n in pts]
    for d in range(0, max_degree + 1):
        xs, ys = pts[:d + 1], vals[:d + 1]
        coeffs = _solve_vandermonde(xs, ys)
        if coeffs is None:
            continue
        ok = all(_poly_eval(coeffs, x) == y
                 for x, y in zip(pts[d + 1:], vals[d + 1:]))
        if ok:
            while len(coeffs) > 1 and coeffs[-1] == 0:
                coeffs.pop()
            return coeffs
    return None


def _poly_eval(coeffs, x):
    v = Fraction(0)
    for c in reversed(coeffs):
        v = v * x + c
    return v


def _solve_vandermonde(xs, ys):
    n = len(xs)
    A = [[Fraction(x) ** j for j in range(n)] for x in xs]
    b = list(ys)
    # Gaussian elimination, exact.
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col] != 0), None)
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        inv = 1 / A[col][col]
        A[col] = [a * inv for a in A[col]]
        b[col] *= inv
        for r in range(n):
            if r != col and A[r][col] != 0:
                factor = A[r][col]
                A[r] = [a - factor * p for a, p in zip(A[r], A[col])]
                b[r] -= factor * b[col]
    return b


def poly_str(coeffs, var="n"):
    """Human-readable exact polynomial string from Fraction coefficients."""
    if coeffs is None:
        return "<non-polynomial>"
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if c == 0:
            continue
        if i == 0:
            mag = f"{abs(c)}"
        else:
            v = var if i == 1 else f"{var}^{i}"
            mag = v if abs(c) == 1 else f"{abs(c)}{v}"
        terms.append(("- " if c < 0 else ("+ " if terms else "")) + mag)
    if not terms:
        return "0"
    return " ".join(terms).replace("+ -", "- ")


# --------------------------------------------------------------------------
# Region-domain satisfiability.
#
# Each rd/ri entry carries `regions` whose domain_plain constrains the
# symbolic parameters and possibly re-parameterized timestamp coordinates
# (free variables).  Piece guards in value/mass are printed gisted against
# this domain, so an entry is valid only where some region domain is
# satisfiable.  We decide satisfiability after substituting parameter
# values, existentially quantifying the free variables by Fourier-Motzkin
# elimination over the rational relaxation.  Modulo constraints involving
# free variables are treated as satisfiable (loose gate); on parameters
# they are evaluated exactly.

class _Lin:
    """Linear expression c0 + sum coeff[v]*v over free variables."""

    def __init__(self, const=Fraction(0), terms=None):
        self.const = Fraction(const)
        self.terms = dict(terms or {})

    def __add__(self, o):
        o = _lin(o)
        t = dict(self.terms)
        for v, c in o.terms.items():
            t[v] = t.get(v, Fraction(0)) + c
        return _Lin(self.const + o.const, t)

    __radd__ = __add__

    def __neg__(self):
        return _Lin(-self.const, {v: -c for v, c in self.terms.items()})

    def __sub__(self, o):
        return self + (-_lin(o))

    def __rsub__(self, o):
        return _lin(o) + (-self)

    def __mul__(self, o):
        o = _lin(o)
        if o.terms and self.terms:
            raise QPError("nonlinear domain")
        if o.terms:
            self, o = o, self
        return _Lin(self.const * o.const,
                    {v: c * o.const for v, c in self.terms.items()})

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _lin(o)
        if o.terms:
            raise QPError("division by variable")
        return self * _Lin(1 / o.const)


def _lin(x):
    return x if isinstance(x, _Lin) else _Lin(Fraction(x))


_CHAIN = re.compile(r"(<=|>=|<|>|=)")


def _split_top(s, sep):
    """Split s at occurrences of sep lying at parenthesis depth 0."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and s.startswith(sep, i):
            parts.append(s[start:i])
            start = i + len(sep)
            i += len(sep) - 1
        i += 1
    parts.append(s[start:])
    return parts


def _outer_parens(s):
    """True if s is fully wrapped by one matching pair of parentheses."""
    if not (s.startswith("(") and s.endswith(")")):
        return False
    depth = 0
    for j, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            return j == len(s) - 1
    return False


def _bool_dnf(s):
    """Parse a domain condition into DNF: a list of atom-string lists."""
    s = s.strip()
    parts = _split_top(s, " or ")
    if len(parts) > 1:
        out = []
        for p in parts:
            out += _bool_dnf(p)
        return out
    parts = _split_top(s, " and ")
    if len(parts) > 1:
        prod = [[]]
        for p in parts:
            sub = _bool_dnf(p)
            prod = [a + b for a in prod for b in sub]
        return prod
    if _outer_parens(s):
        return _bool_dnf(s[1:-1])
    return [[s]] if s else [[]]


def _atoms_sat(atoms, env):
    """Fourier-Motzkin feasibility of a flat conjunction of atoms."""
    ineqs = []          # list of _Lin e meaning e >= 0
    for atom in atoms:
        atom = atom.strip()
        if not atom:
            continue
        if " mod " in atom:
            # exact on parameters; loose (assume satisfiable) on free vars
            try:
                probe = _wrap_ints(_pythonize_cond(atom))
                genv = {"__builtins__": {}, "F": Fraction,
                        "floor": _floor, "ceil": _ceil}
                genv.update({k: Fraction(v) for k, v in env.items()})
                if not eval(probe, genv):
                    return False
            except NameError:
                pass
            continue
        parts = _CHAIN.split(atom)
        if len(parts) < 3:
            continue
        exprs, rels = parts[0::2], parts[1::2]
        vals = []
        for e in exprs:
            src = _wrap_ints(_pythonize_expr(e))
            genv = {"__builtins__": {}, "F": Fraction,
                    "floor": _floor, "ceil": _ceil}
            genv.update({k: Fraction(v) for k, v in env.items()})
            genv.update({name: _Lin(0, {name: Fraction(1)})
                         for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", e)
                         if name not in _RESERVED and name not in env})
            vals.append(_lin(eval(src, genv)))
        for a, rel, b in zip(vals, rels, vals[1:]):
            if rel == "<=":
                ineqs.append(b - a)
            elif rel == "<":
                ineqs.append(b - a - _Lin(1))
            elif rel == ">=":
                ineqs.append(a - b)
            elif rel == ">":
                ineqs.append(a - b - _Lin(1))
            else:  # equality
                ineqs.append(a - b)
                ineqs.append(b - a)
    # Fourier-Motzkin elimination
    while True:
        for e in ineqs:
            if not e.terms and e.const < 0:
                return False
        free = {v for e in ineqs for v in e.terms}
        if not free:
            return True
        v = next(iter(free))
        lows, highs, rest = [], [], []
        for e in ineqs:
            c = e.terms.get(v)
            if c is None:
                rest.append(e)
                continue
            r = _Lin(e.const, {u: k for u, k in e.terms.items() if u != v})
            if c > 0:
                lows.append(r * _Lin(Fraction(-1) / c))   # v >= r/(-c)
            else:
                highs.append(r * _Lin(Fraction(1) / -c))  # v <= -r/c... sign
        # e = c*v + r >= 0: c>0 -> v >= -r/c ; c<0 -> v <= -r/c = r/(-c)
        new = rest
        for lo in lows:
            for hi in highs:
                new.append(hi - lo)
        ineqs = new


def domain_satisfiable(domain, env):
    """True if the domain string is satisfiable at the given parameter env,
    existentially quantifying non-parameter variables (rational relaxation).
    Any construct the checker cannot handle makes the gate loose (True)."""
    domain = domain.strip()
    if not domain:
        return True
    try:
        for conj in _bool_dnf(domain):
            if _atoms_sat(conj, env):
                return True
        return False
    except Exception:
        return True  # unsupported construct: loose gate

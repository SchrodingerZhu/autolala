#!/usr/bin/env python3
"""Exact-trace DMD simulator, generated from the AutoLALA DSL.

For each kernel we parse dsl/sym_<k>.dsl (the exact program the symbolic
analyzer saw), emit its loop nest as C, and compute EXACT reuse distances over
8-element cache lines with a Fenwick tree (classic stack-distance algorithm).
This gives ground-truth  A(N), warm(N), DMD(N) = sum over warm accesses of
sqrt(RD)  with no scale approximation and no formula interpretation -- the
instrument we use to anchor leading terms that the symbolic route struggles
with on triangular iteration spaces.

RD convention: number of DISTINCT lines touched strictly between an access and
its previous access to the same line (immediate reuse -> RD 0), matching the
analyzer's reuse-distance definition.

Usage: python3 gen_sim.py            # all kernels, default sweeps
       python3 gen_sim.py syrk 128   # one kernel, chosen sizes
Writes exact/<kernel>.json with one record per N.
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DSL = os.path.join(HERE, "..", "dsl")
BLOCK = 8

# ---------------------------------------------------------------- DSL parsing

def tokenize(src):
    src = re.sub(r"//[^\n]*", "", src)
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\.\.|[{}\[\](),;+*-]|/", src)


class Parser:
    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek(); self.i += 1; return t

    def expect(self, t):
        got = self.next()
        assert got == t, f"expected {t!r}, got {got!r} near {self.toks[self.i-4:self.i+3]}"

    def parse(self):
        params, arrays, body = [], {}, []
        while self.peek():
            t = self.peek()
            if t == "params":
                self.next()
                while True:
                    params.append(self.next())
                    if self.next() == ";":
                        break
            elif t == "array":
                self.next()
                name = self.next()
                self.expect("[")
                dims = []
                while True:
                    dims.append(self.parse_expr())
                    if self.next() == "]":
                        break
                self.expect(";")
                arrays[name] = dims
            else:
                body.append(self.parse_stmt())
        return params, arrays, body

    def parse_stmt(self):
        t = self.next()
        if t == "for":
            var = self.next()
            self.expect("in")
            lo = self.parse_expr()
            self.expect("..")
            hi = self.parse_expr()
            self.expect("{")
            body = []
            while self.peek() != "}":
                body.append(self.parse_stmt())
            self.expect("}")
            return ("for", var, lo, hi, body)
        if t in ("read", "write"):
            arr = self.next()
            self.expect("[")
            idx = []
            while True:
                idx.append(self.parse_expr())
                if self.next() == "]":
                    break
            self.expect(";")
            return ("acc", arr, idx)
        raise AssertionError(f"unknown statement {t!r}")

    # affine expression: (NUM | NAME | NUM*NAME | NAME*NUM) joined by +/-
    def parse_expr(self):
        out = [self.parse_term()]
        while self.peek() in ("+", "-"):
            out.append(self.next())
            out.append(self.parse_term())
        return " ".join(out)

    def parse_term(self):
        neg = ""
        if self.peek() == "-":
            self.next(); neg = "-"
        t = self.next()
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|\d+", t), f"bad term {t!r}"
        if self.peek() == "*":
            self.next()
            return f"{neg}{t} * {self.next()}"
        return neg + t


# ------------------------------------------------------------- C generation

C_TEMPLATE = r"""#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

static int32_t *bit; static int64_t nbit;
static int64_t *last;              /* per line: 1-based position of previous access */
static int64_t t = 0, total = 0, warm = 0;
static double dmd = 0.0;
static int64_t hist[64];           /* hist[k]: warm accesses with floor(log2(RD+1)) = k-ish */
static int64_t max_rd = 0;
/* capacity misses at a fixed fully-associative cache measured in LINES: an
 * access misses when its reuse distance >= the cache's line capacity. 2MB and
 * 8MB last-level caches over 64-byte lines. Cold (first-touch) always misses. */
static int64_t miss_2mb = 0, miss_8mb = 0, cold = 0;
#define LINES_2MB (2097152 / 64)
#define LINES_8MB (8388608 / 64)

static inline void bit_add(int64_t pos, int v) {
    for (; pos <= nbit; pos += pos & -pos) bit[pos] += v;
}
static inline int64_t bit_query(int64_t pos) {
    int64_t s = 0;
    for (; pos > 0; pos -= pos & -pos) s += bit[pos];
    return s;
}
static inline void emit(int64_t line) {
    t++; total++;
    int64_t prev = last[line];
    if (prev) {
        int64_t rd = bit_query(t - 1) - bit_query(prev);
        dmd += sqrt((double)rd);
        warm++;
        if (rd > max_rd) max_rd = rd;
        int k = 0; int64_t v = rd; while (v) { v >>= 1; k++; }
        hist[k]++;
        if (rd >= LINES_2MB) miss_2mb++;
        if (rd >= LINES_8MB) miss_8mb++;
        bit_add(prev, -1);
    } else {
        cold++; miss_2mb++; miss_8mb++;
    }
    bit_add(t, 1);
    last[line] = t;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s N\n", argv[0]); return 2; }
    int64_t N = atoll(argv[1]);
BASES
    /* dry pass: count the trace and find the largest line index actually
     * touched (stencils reach past the declared footprint via halo accesses),
     * so the tables are sized to what the trace really uses. */
    int64_t TRACE = 0, MAXLINE = 0;
COUNT_NEST
    nbit = TRACE + 2;
    int64_t NLINES = MAXLINE + 2;
    bit = calloc(nbit + 1, sizeof(int32_t));
    last = calloc(NLINES + 2, sizeof(int64_t));
    if (!bit || !last) { fprintf(stderr, "alloc failed\n"); return 1; }
EMIT_NEST
    printf("{\"N\": %lld, \"total\": %lld, \"warm\": %lld, \"dmd\": %.10e, \"max_rd\": %lld, "
           "\"miss_2mb\": %lld, \"miss_8mb\": %lld, \"cold\": %lld, \"hist\": [",
           (long long)N, (long long)total, (long long)warm, dmd, (long long)max_rd,
           (long long)miss_2mb, (long long)miss_8mb, (long long)cold);
    int first = 1;
    for (int k = 0; k < 64; k++) if (hist[k]) {
        printf("%s[%d, %lld]", first ? "" : ", ", k, (long long)hist[k]);
        first = 0;
    }
    printf("]}\n");
    return 0;
}
"""


def build_kernel(name):
    src = open(os.path.join(DSL, f"sym_{name}.dsl")).read()
    params, arrays, body = Parser(tokenize(src)).parse()
    subst = {p: "N" for p in params}

    def cexpr(e):
        return re.sub(r"[A-Za-z_][A-Za-z0-9_]*",
                      lambda m: subst.get(m.group(0), m.group(0)), e)

    # arrays laid out consecutively, each starting on a line boundary
    bases, off_terms = [], []
    for aname, dims in arrays.items():
        size = " * ".join(f"({cexpr(d)})" for d in dims)
        prev = " + ".join(off_terms) if off_terms else "0"
        bases.append(f"    int64_t base_{aname} = {prev};")
        off_terms.append(f"((({size}) + {BLOCK - 1}) / {BLOCK}) * {BLOCK}")
    total_elems = "(" + " + ".join(off_terms) + ")"

    def nest(stmts, depth, stmt_fmt):
        out = []
        ind = "    " * (depth + 1)
        for s in stmts:
            if s[0] == "for":
                _, var, lo, hi, inner = s
                out.append(f"{ind}for (int64_t {var} = {cexpr(lo)}; "
                           f"{var} < {cexpr(hi)}; {var}++) {{")
                out += nest(inner, depth + 1, stmt_fmt)
                out.append(f"{ind}}}")
            else:
                _, arr, idx = s
                dims = arrays[arr]
                flat = f"({cexpr(idx[0])})"
                for d, ix in zip(dims[1:], idx[1:]):
                    flat = f"({flat} * ({cexpr(d)}) + ({cexpr(ix)}))"
                out.append(ind + stmt_fmt.format(line=f"(base_{arr} + {flat}) / {BLOCK}"))
        return out

    count_stmt = "{{ int64_t _l = {line}; if (_l > MAXLINE) MAXLINE = _l; TRACE++; }}"
    csrc = (C_TEMPLATE
            .replace("BASES", "\n".join(bases))
            .replace("COUNT_NEST", "\n".join(nest(body, 0, count_stmt)))
            .replace("EMIT_NEST", "\n".join(nest(body, 0, "emit({line});")))
            .replace("TOTAL_ELEMS", total_elems)
            .replace("BLOCK", str(BLOCK)))
    cpath = os.path.join(HERE, f"sim_{name}.c")
    open(cpath, "w").write(csrc)
    exe = os.path.join(HERE, f"sim_{name}")
    subprocess.run(["cc", "-O2", "-o", exe, cpath, "-lm"], check=True)
    return exe


KERNELS = {
    # class C (triangular) + two anchors from known classes A and B
    "trisolve": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "syrk":     [32, 48, 64, 96, 128, 192, 256, 384],
    "syr2k":    [32, 48, 64, 96, 128, 192, 256],
    "trmm":     [32, 48, 64, 96, 128, 192, 256, 384],
    "symm":     [32, 48, 64, 96, 128, 192, 256, 384],
    "cholesky": [32, 48, 64, 96, 128, 192, 256, 384, 512],
    "lu":       [32, 48, 64, 96, 128, 192, 256, 384],
    "gemm":     [32, 48, 64, 96, 128, 192, 256, 384],
    "mvt":      [64, 128, 256, 512, 1024, 2048, 4096],
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        todo = {name: [int(x) for x in sys.argv[2:]] or KERNELS[name]}
    else:
        todo = KERNELS
    for name, sweep in todo.items():
        exe = build_kernel(name)
        recs = []
        for N in sweep:
            out = subprocess.run([exe, str(N)], capture_output=True, text=True)
            if out.returncode != 0:
                print(f"{name} N={N}: FAILED {out.stderr.strip()[:120]}")
                break
            rec = json.loads(out.stdout)
            recs.append(rec)
            print(f"{name:10s} N={N:5d} total={rec['total']:.3e} warm={rec['warm']:.3e} "
                  f"dmd={rec['dmd']:.4e} max_rd={rec['max_rd']}", flush=True)
        json.dump(recs, open(os.path.join(HERE, f"{name}.json"), "w"), indent=1)

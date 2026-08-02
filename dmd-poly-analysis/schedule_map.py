#!/usr/bin/env python3
"""Map analyzer source signatures back to DSL accesses.

The lowering encodes a statement's schedule as alternating dimensions
  [t0(block pos), var, t1(block pos), var, ..., tk(block pos)]
where a block at dim index d names its position dimension t{d - #loops} and
each child statement i is pinned by t{..} = i; a for-loop contributes its
variable as the next dimension. Walking the DSL with the same rule gives every
access a position assignment {t0: .., t1: .., ...}. An analyzer `sources`
signature ("t0=0,i1=3,t2=1") matches an access iff it agrees with the
access's position assignment on every t-dimension both mention.
"""
import re


class Access:
    def __init__(self, kind, array, subscript, positions, loops):
        self.kind = kind          # "read" | "write"
        self.array = array
        self.subscript = subscript
        self.positions = positions  # {t-name: int}
        self.loops = loops          # [var, ...] enclosing, outer->inner

    @property
    def text(self):
        return f"{self.kind} {self.array}[{self.subscript}]"

    def __repr__(self):
        return f"<{self.text} @ {self.positions}>"


_FOR = re.compile(r"^for\s+(\w+)\s+in\s+(.+?)\s*\.\.\s*(.+?)\s*\{$")
_ACC = re.compile(r"^(read|write)\s+(\w+)\s*\[(.*)\]\s*;$")


def parse_accesses(dsl_text):
    """All accesses of a DSL program with their schedule positions."""
    lines = [l.strip() for l in dsl_text.splitlines()
             if l.strip() and not l.strip().startswith(("params", "array"))]
    accesses = []

    def walk(index, depth, nloops, positions, loops):
        """Parse one block body starting at `index` (after the opening line).
        Returns the index just past the block's closing '}'."""
        pos_name = f"t{depth - nloops}"
        stmt_idx = 0
        i = index
        while i < len(lines):
            line = lines[i]
            if line == "}":
                return i + 1
            m = _FOR.match(line)
            if m:
                var = m.group(1)
                i = walk(i + 1, depth + 2, nloops + 1,
                         {**positions, pos_name: stmt_idx}, loops + [var])
                stmt_idx += 1
                continue
            m = _ACC.match(line)
            if m:
                accesses.append(Access(m.group(1), m.group(2), m.group(3),
                                       {**positions, pos_name: stmt_idx}, loops))
                stmt_idx += 1
                i += 1
                continue
            raise ValueError(f"unrecognized DSL line: {line!r}")
        return i

    walk(0, 0, 0, {}, [])
    return accesses


def parse_signature(sig):
    """'t0=0,i1=3,t2=1' -> {'t0': 0, 'i1': 3, 't2': 1}.

    Under --infinite-repeat the analyzer wraps the program in a `__repeat`
    loop, which prepends one block level: its signatures read
    't0=0,__repeat=1,t1=...,t2=...' where t{k} corresponds to the unwrapped
    program's t{k-1}. Normalize those back so they match the DSL walk."""
    out = {}
    for part in sig.split(","):
        name, _, val = part.partition("=")
        out[name.strip()] = int(val)
    if "__repeat" in out:
        norm = {}
        for name, val in out.items():
            if name == "__repeat":
                continue
            if name.startswith("t") and name[1:].isdigit():
                k = int(name[1:])
                if k == 0:
                    continue  # the wrapper's own block position
                norm[f"t{k - 1}"] = val
            else:
                norm[name] = val
        out = norm
    return out


def match_accesses(sig, accesses):
    """Accesses compatible with one signature (t-dims must agree)."""
    fixed = parse_signature(sig)
    matched = []
    for acc in accesses:
        ok = True
        for name, val in acc.positions.items():
            if name in fixed and fixed[name] != val:
                ok = False
                break
        if ok:
            # signature t-dims not present in the access mean a deeper/other
            # nesting level: reject if the signature fixes a t-dim the access
            # does not have at all AND some access does have it (handled by
            # caller preferring exact matches); keep the simple containment.
            matched.append(acc)
    # prefer accesses whose position dims are exactly the signature's t-dims
    exact = [a for a in matched
             if all(t in a.positions for t in fixed if t.startswith("t"))]
    return exact or matched


def sources_to_text(sources, accesses, iterators=()):
    """Human summary of a bin's sources: distinct access texts, annotated with
    fixed loop values from the signatures where present."""
    seen = []
    for sig in sources:
        fixed = parse_signature(sig)
        for acc in match_accesses(sig, accesses):
            extra = ", ".join(f"{k}={v}" for k, v in sorted(fixed.items())
                              if not k.startswith("t") and k in
                              set(acc.loops) | set(iterators))
            label = acc.text + (f" ({extra})" if extra else "")
            if label not in seen:
                seen.append(label)
    return seen

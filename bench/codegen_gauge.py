#!/usr/bin/env -S uv run
"""Fingerprint a kernel's machine code, so cross-build A/B can be verified.

The build is `-flto -fwhole-program`, and GCC's inline budgets are UNIT-level
(inline-unit-growth, large-unit-insns, large-function-growth). Adding an
unrelated file therefore changes the absolute inlining budget and can silently
alter the codegen of functions that did not change -- measured on this repo:
set_residual's inlined body went 9,177 -> 10,759 instructions and 50 -> 70
scalar divides purely because benchmark arms were added to _fortran/.

That matters because comparing an arm measured in build B against a `prod`
baseline from build A is only valid if `prod` is the same code in both. Timing
cannot establish that -- the launch-to-launch noise floor is ~0.4%, which is
the same order as the differences being chased. Machine code can: if the
fingerprint matches, the comparison is exact.

Fingerprint = the recursive closure of a symbol and everything it calls,
normalised so that layout-only differences (load addresses, jump displacements)
do not register, then hashed. Also reported: instruction count and the
arithmetic mix, so a mismatch is diagnosable rather than merely alarming.

Usage:
    uv run python bench/codegen_gauge.py set_residual_ [--so PATH] [--json OUT]
    uv run python bench/codegen_gauge.py --compare a.json b.json
"""

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys

INSN = re.compile(r"^\s+[0-9a-f]+:\t(?:[0-9a-f]{2} )+\s*\t(.*)$")
SYMHDR = re.compile(r"^([0-9a-f]+) <(.+)>:")
CALLTGT = re.compile(r"\b(?:call|jmp)q?\s+([0-9a-f]+) <([^>+]+)(?:\+0x[0-9a-f]+)?>")

# Addresses and displacements move with layout without the code differing.
NORM = (
    (re.compile(r"#.*$"), ""),  # trailing comments
    (re.compile(r"\b[0-9a-f]{4,}\s+<([^>+]+)(?:\+0x[0-9a-f]+)?>"), r"<\1>"),
    (re.compile(r"0x[0-9a-f]{5,}"), "ADDR"),
    (re.compile(r"-?0x[0-9a-f]+\(%rip\)"), "RIP"),
    (re.compile(r"\s+"), " "),
)


def disassemble(so):
    out = subprocess.run(
        ["objdump", "-d", so], capture_output=True, text=True, check=True
    ).stdout
    funcs, cur = {}, None
    for line in out.split("\n"):
        m = SYMHDR.match(line)
        if m:
            cur = m.group(2)
            funcs[cur] = []
        elif cur is not None:
            m = INSN.match(line)
            if m:
                funcs[cur].append(m.group(1))
    return funcs


def normalise(text):
    for pat, rep in NORM:
        text = pat.sub(rep, text)
    return text.strip()


def closure(funcs, root):
    """Symbol plus everything it calls, depth-first, deterministic order."""
    seen, order = set(), []

    def walk(name):
        if name in seen or name not in funcs:
            return
        seen.add(name)
        order.append(name)
        for line in funcs[name]:
            for m in CALLTGT.finditer(line):
                walk(m.group(2))

    walk(root)
    return order


def fingerprint(so, root):
    funcs = disassemble(so)
    if root not in funcs:
        raise SystemExit(f"symbol {root!r} not found in {so}")
    order = closure(funcs, root)
    h = hashlib.sha256()
    n = 0
    mix = collections.Counter()
    for name in order:
        h.update(name.encode())
        for line in funcs[name]:
            t = normalise(line)
            h.update(t.encode())
            n += 1
            op = t.split(" ")[0]
            if op in (
                "vdivps",
                "vdivss",
                "vrcpps",
                "vgatherdps",
                "vfmadd213ps",
                "vmulps",
                "vaddps",
            ):
                mix[op] += 1
            if "%ymm" in t:
                mix["_ymm_ops"] += 1
    return dict(
        symbol=root,
        so=so,
        sha256=h.hexdigest()[:16],
        insns=n,
        nfuncs=len(order),
        funcs=order,
        mix=dict(mix),
        thunk=thunk_target(funcs[root]),
    )


# A body of a couple of instructions ending in an unconditional jump is not a
# kernel, it is a tail-call thunk. GCC's identical-code folding produces these
# whenever two functions compile to the same machine code -- which is exactly
# what happens when a bench arm is source-identical to the production kernel it
# was cloned from. The symbol then fingerprints as ~13 instructions and the
# Rule 9 gate silently passes on a stub while the real body lives under the
# other name, AND the thunked caller pays a PLT hop the other does not: that
# cost 8% on byte-identical source in the IRS study (docs/dev/plan_irs_traffic.md).
# Detect it and say so rather than reporting a hash of three instructions.
_JMP = re.compile(r"^\s*jmp\s+[0-9a-f]+ <([^>+]+)")


def thunk_target(body):
    real = [l for l in body if l.strip() and "nop" not in l]
    if len(real) > 3:
        return None
    for line in real:
        m = _JMP.search(line.split("\t")[-1] if "\t" in line else line)
        if m:
            return m.group(1).replace("@plt", "")
    return None


def show(fp):
    print(f"  symbol      {fp['symbol']}")
    print(f"  fingerprint {fp['sha256']}")
    print(f"  insns       {fp['insns']}  over {fp['nfuncs']} function(s)")
    print(f"  mix         {fp['mix']}")
    if fp["thunk"]:
        print(
            f"  *** WARNING: this symbol is a THUNK jumping to "
            f"{fp['thunk']} -- not a kernel body."
        )
        print(
            "      Identical-code folding has merged it with another function, so this"
        )
        print(
            "      fingerprint gates nothing and the extra PLT hop biases "
            "any timing of it."
        )
        print("      Rebuild without the arm whose source is identical to this kernel.")


def compare(a, b):
    fa, fb = json.load(open(a)), json.load(open(b))
    same = fa["sha256"] == fb["sha256"]
    print(f"A: {fa['sha256']}  insns={fa['insns']:<7} {a}")
    print(f"B: {fb['sha256']}  insns={fb['insns']:<7} {b}")
    if same:
        print("\n  IDENTICAL -- cross-build comparison of this symbol is exact.")
        return 0
    print("\n  DIFFERENT -- this symbol was recompiled differently.")
    print(
        f"    insns   {fa['insns']} -> {fb['insns']} "
        f"({(fb['insns'] / fa['insns'] - 1) * 100:+.1f}%)"
    )
    keys = sorted(set(fa["mix"]) | set(fb["mix"]))
    for k in keys:
        x, y = fa["mix"].get(k, 0), fb["mix"].get(k, 0)
        if x != y:
            print(f"    {k:<14} {x} -> {y}")
    only_a = [f for f in fa["funcs"] if f not in fb["funcs"]]
    only_b = [f for f in fb["funcs"] if f not in fa["funcs"]]
    if only_a:
        print(f"    only in A: {only_a}")
    if only_b:
        print(f"    only in B: {only_b}")
    print(
        "\n  A timing comparison across these two builds is NOT valid for this symbol."
    )
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("symbol", nargs="?", default="set_residual_")
    ap.add_argument("--so", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)

    so = args.so
    if so is None:
        import ember.fortran as F

        so = F.__file__
    fp = fingerprint(so, args.symbol)
    show(fp)
    if args.json:
        json.dump(fp, open(args.json, "w"), indent=1)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

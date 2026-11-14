#!/usr/bin/env python3
"""Inline the generated number macros of a LaTeX file, for latexdiff and for co-authors who edit by hand.

    python scripts/flatten_macros.py main.tex generated/macros.tex -o main_flat.tex [--inline-bbl main.bbl]

Every \\res<Name> defined in the macro file by \\newcommand{\\res<Name>}{<value>} is replaced by <value>. Two
table helpers of the author package are expanded as well:
  \\resOrd{<model>}{<cell>}{k}                prints \\resPpl<model><cell>{Joint,PQ,QP}[k], bold when it is the lowest
  \\resRow{<model>}{<cell>}{<label>}{<TCr>}{<prune>}{<quant>}   one row of the order table
The block between the lines '%% BEGIN res helpers' and '%% END res helpers' (their TeX definitions) and the
\\input of the macro file are dropped. The output therefore compiles without generated/ and prints the same text.

A macro written as '\\resName word' loses its space in TeX (a control word swallows the spaces after it); such
places are reported as warnings, because they are almost always a typo in the source ('\\resName{} word').
The output is generated: edit the source, not the flattened file.
"""
import argparse
import re
import sys

NAME = re.compile(r"\\(res[A-Za-z]+)")


def balanced(s, i):
    """s[i] == '{': return (content, index after the closing brace)."""
    assert s[i] == "{", s[i:i + 20]
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"unbalanced braces from: {s[i:i + 60]!r}")


def read_macros(path):
    text = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"\\(?:newcommand|providecommand|renewcommand)\{\\(res[A-Za-z]+)\}(?:\[\d\])?", text):
        k = m.end()
        if k < len(text) and text[k] == "{":
            val, _ = balanced(text, k)
            out[m.group(1)] = val
    return out


def as_float(v):
    v = re.sub(r"\\ensuremath\{(.*)\}", r"\1", v).replace("$", "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def ordinal(macros, model, cell, k):
    keys = [f"resPpl{model}{cell}{o}" for o in ("Joint", "PQ", "QP")]
    vals = [macros.get(x, "--") for x in keys]
    nums = [as_float(v) for v in vals]
    v = vals[k - 1]
    if all(n is not None for n in nums) and nums[k - 1] == min(nums):
        return f"\\textbf{{{v}}}"
    return v


def args(s, i, n):
    """Parse n brace arguments starting at s[i] (spaces allowed between them)."""
    out = []
    for _ in range(n):
        while s[i] in " \t\n":
            i += 1
        a, i = balanced(s, i)
        out.append(a)
    return out, i


def flatten(src, macros, warn):
    # drop helper definitions and the \input of the macro file
    src = re.sub(r"%% BEGIN res helpers.*?%% END res helpers[^\n]*\n", "", src, flags=re.S)
    src = re.sub(r"\\input\{generated/macros(\.tex)?\}[^\n]*\n", "% (number macros inlined by scripts/flatten_macros.py)\n", src)
    out, i = [], 0
    while True:
        m = NAME.search(src, i)
        if not m:
            out.append(src[i:])
            break
        out.append(src[i:m.start()])
        name, j = m.group(1), m.end()
        if name == "resRow":
            (model, cell, label, tcr, prune, quant), j = args(src, j, 6)
            row = [label, tcr, macros.get(f"resPpl{model}{prune}", "--"), macros.get(f"resPpl{model}{quant}", "--")]
            row += [ordinal(macros, model, cell, k) for k in (1, 2, 3)]
            out.append(" & ".join(row) + " \\\\")
        elif name == "resOrd":
            (model, cell, k), j = args(src, j, 3)
            out.append(ordinal(macros, model, cell, int(k)))
        elif name in macros:
            out.append(macros[name])
            if src.startswith("{}", j):
                j += 2
            elif j < len(src) and src[j] in " \n":
                line = src.count("\n", 0, m.start()) + 1
                warn.append(f"line {line}: \\{name} is followed by a space that TeX swallows (write \\{name}{{}})")
        else:
            warn.append(f"\\{name} is not defined in the macro file; left as is")
            out.append(m.group(0))
        i = j
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("macros")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--inline-bbl", help="replace \\bibliography{...} by the content of this .bbl file")
    a = ap.parse_args()
    macros = read_macros(a.macros)
    warn = []
    flat = flatten(open(a.tex, encoding="utf-8").read(), macros, warn)
    if a.inline_bbl:
        bbl = open(a.inline_bbl, encoding="utf-8").read()
        flat = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\n?", "", flat)
        flat = re.sub(r"\\bibliography\{[^}]*\}", lambda _: bbl, flat)  # a function's result is taken literally
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("%% GENERATED by scripts/flatten_macros.py from " + a.tex + " -- edit the source, not this file\n" + flat)
    for w in warn:
        print("WARNING", w, file=sys.stderr)
    print(f"{a.out}: {len(macros)} macros available, {len(warn)} warnings")
    return 1 if any("not defined" in w for w in warn) else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fill a Markdown template with the generated number macros, so a report quotes no number by hand.

    python scripts/render_template.py NOTE.template.md paper/generated/macros.tex -o NOTE.md

In the template, {{resName}} is replaced by the value of \\resName in the macro file, converted to plain text
(\\ensuremath{+0.016} -> +0.016, 7.69\\times 10^{-8} -> 7.69e-8, 50\\%+3 -> 50%+3). A value '--' (not available
yet) is printed as 'n/a'. An unknown name is an error, so a typo cannot pass silently. {{file:NAME.md}} includes a
generated snippet from --parts (e.g. the tables of scripts/note_tables.py). {{=expr}} is not supported: derived
numbers belong in the analysis code, not in templates.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flatten_macros import read_macros  # noqa: E402

FIELD = re.compile(r"\{\{\s*(res[A-Za-z]+)\s*\}\}")


def plain(v):
    v = re.sub(r"\\ensuremath\{(.*)\}", r"\1", v.strip())
    v = re.sub(r"\s*\\times\s*10\^\{(-?\d+)\}", r"e\1", v)
    v = v.replace("\\%", "%").replace("$", "").replace("\\,", " ").replace("~", " ")
    v = v.replace("\\textbf{", "").replace("}", "") if v.startswith("\\textbf{") else v
    return "n/a" if v == "--" else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("macros")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--parts", nargs="+", default=[],
                    help="folders of Markdown snippets, included with {{file:NAME.md}} (first folder that has it)")
    a = ap.parse_args()
    macros = read_macros(a.macros)
    text = open(a.template, encoding="utf-8").read()

    def include(m):
        for d in a.parts:
            p = os.path.join(d, m.group(1))
            if os.path.exists(p):
                return open(p, encoding="utf-8").read().rstrip("\n")
        return m.group(0)

    for _ in range(3):  # snippets may include snippets
        text = re.sub(r"\{\{file:([\w.\-]+)\}\}", include, text)
    if "{{file:" in text:
        print("unresolved {{file:...}} include (pass --parts)", file=sys.stderr)
        return 1
    missing = sorted({m for m in FIELD.findall(text) if m not in macros})
    if missing:
        print("unknown macros: " + ", ".join(missing), file=sys.stderr)
        return 1
    used = set(FIELD.findall(text))
    na = sorted(m for m in used if macros[m].strip() == "--")
    out = FIELD.sub(lambda m: plain(macros[m.group(1)]), text)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{a.out}: {len(used)} macros quoted" + (f", {len(na)} not available yet: {', '.join(na)}" if na else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Admit a new Colab image only if a known state recomputes bit for bit (DECISIONS.md #26, #39).

Compares an IMG (or BATCH) result line pulled from a runtime with the E1 line of the same state_id: WikiText-2 NLL
sum and perplexity, every layer's rel_err, and (when both have them) C4 and the per-item benchmark correctness.
Prints one verdict line: IDENTICAL or DIFFERENT, with the first differences.

    python scripts/img_check.py results/v2/IMG/SmolLM2-135M__esann-a6.jsonl
"""
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def e1_line(state_id):
    model = state_id.split("__")[0]
    for fam in ("E1", "E4", "E3", "E5", "E6"):
        p = os.path.join(ROOT, "results", "v2", fam, f"{model}.jsonl")
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    if r.get("state_id") == state_id and r.get("status") == "ok":
                        return r
    return None


def items(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def compare(new, old, root=ROOT):
    diffs = []
    for split in ("wikitext2", "c4"):
        a, b = new.get(split) or {}, old.get(split) or {}
        for k in ("nll_sum", "ppl"):
            if k in a and k in b and a[k] != b[k]:
                diffs.append(f"{split}.{k}: {a[k]!r} vs {b[k]!r}")
    la = {(d["layer"], d["name"]): d["rel_err"] for d in new.get("diags") or []}
    lb = {(d["layer"], d["name"]): d["rel_err"] for d in old.get("diags") or []}
    if set(la) != set(lb):
        diffs.append(f"layer sets differ ({len(la)} vs {len(lb)})")
    nd = sum(la[k] != lb[k] for k in la if k in lb)
    if nd:
        diffs.append(f"{nd} of {len(la)} layer rel_err differ")
    n_items = None
    if new.get("items_file") and old.get("items_file"):
        pa, pb = os.path.join(root, new["items_file"]), os.path.join(root, old["items_file"])
        if os.path.exists(pa) and os.path.exists(pb):
            ia, ib = items(pa), items(pb)
            n_items, nflip = 0, 0
            for t in ib:
                if t not in ia:
                    diffs.append(f"task {t} missing in the new items")
                    continue
                pos = {d: i for i, d in enumerate(ia[t]["doc_id"])}
                for m in ("acc", "acc_norm"):
                    if m in ib[t] and m in ia[t]:
                        for d, v in zip(ib[t]["doc_id"], ib[t][m]):
                            n_items += 1
                            if d in pos and round(ia[t][m][pos[d]]) != round(v):
                                nflip += 1
            if nflip:
                diffs.append(f"{nflip} of {n_items} item results differ")
    return diffs, len(la), n_items


def main():
    path = sys.argv[1]
    for line in open(path):
        if not line.strip():
            continue
        new = json.loads(line)
        old = e1_line(new["state_id"])
        if old is None:
            print(f"NO REFERENCE {new['state_id']}")
            continue
        diffs, nl, ni = compare(new, old)
        img = (new.get("metadata") or {}).get("colab_image")
        verdict = "IDENTICAL" if not diffs else "DIFFERENT"
        print(f"{verdict} {new['state_id']} on {img}: {nl} layers" + (f", {ni} item results" if ni else "")
              + ("" if not diffs else " | " + "; ".join(diffs[:5])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

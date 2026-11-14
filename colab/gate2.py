#!/usr/bin/env python3
"""Reproduction investigation after the phase-0 gate failed on SmolLM2-135M (fp32 vs published 16-bit).

Part 1 (--part fp16): SmolLM2-135M in 16-bit on the L4 vs the published 16-bit (T4) runs.
Part 2 (--part fp32): SmolLM2-360M in fp32 on the L4 vs the paper's own fp32 re-runs (results_fix.jsonl).
Prints every comparison and writes logs/gate2_<part>.json; exits 1 if a difference exceeds the tolerance.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("ESANN_OUT", ROOT)


def rows(p, base):
    p = os.path.join(base, p)
    return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []


def key(r):
    return (r["order"], round(r["sparsity"], 4), r["bits"])


ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["fp16", "fp32"], required=True)
ap.add_argument("--tol", type=float, default=0.005)
a = ap.parse_args()
if a.part == "fp16":
    ours = {key(r): r for r in rows("results/v2/REPRO16/SmolLM2-135M.jsonl", OUT)}
    pub = {key(r): r for r in rows("results/results_full.jsonl", ROOT)
           if r["model"].endswith("SmolLM2-135M") and r.get("seed", 0) == 0}
else:
    ours = {key(r): r for r in rows("results/v2/REPRO32/SmolLM2-360M.jsonl", OUT)}
    pub = {key(r): r for r in rows("results/results_fix.jsonl", ROOT) if r["model"].endswith("SmolLM2-360M")}
out, ok = {"part": a.part, "tol": a.tol, "checks": []}, True
for k, r in sorted(ours.items()):
    if k not in pub:
        continue
    rel = (r["wikitext2"]["ppl"] - pub[k]["ppl"]) / pub[k]["ppl"]
    good = abs(rel) <= a.tol
    ok &= good
    out["checks"].append({"cell": k, "ours": r["wikitext2"]["ppl"], "published": pub[k]["ppl"], "rel_diff": rel,
                          "dtype": r.get("dtype"), "status": "PASS" if good else "FAIL"})
if not out["checks"]:
    ok = False
out["result"] = "PASS" if ok else "FAIL"
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
with open(os.path.join(OUT, "logs", f"gate2_{a.part}.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out))
sys.exit(0 if ok else 1)

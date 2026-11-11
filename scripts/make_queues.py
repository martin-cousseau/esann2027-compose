#!/usr/bin/env python3
"""Write the unit queues of every experiment family to colab/queues/<family>__<model>.json.

A unit is a model state (see scripts/run_grid.py). States already present in any results/v2/*/*.jsonl
are left out, so families reuse the dense baseline and single-method runs instead of re-evaluating them.

Usage: python scripts/make_queues.py E1 [E3 E4 E5 E6 ...]   (or: all)
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CORE = ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B"]
SPARS = [0.25, 1 / 3, 0.5]
ORDERS = ["joint", "p_then_q", "q_then_p"]
ARMS = ["q_then_p_refit", "q_then_p_pmask", "q_then_p_pmask_refit", "p_then_q_stale"]


def u(order, s, b, seed=0, **kw):
    d = {"order": order, "sparsity": s, "bits": b, "seed": seed}
    d.update(kw)
    return d


def e1():
    q = [u("dense", 0.0, 16), u("quant", 0.0, 4), u("p_then_q", 0.5, 4), u("joint", 0.5, 4)]  # pilot units first
    q += [u("prune", s, 16) for s in SPARS] + [u("quant", 0.0, b) for b in (8, 3)]
    q += [u(o, s, b) for b in (4, 3) for s in SPARS for o in ORDERS]
    return {m: dedupe(q) for m in CORE}


def e3():
    # perplexity, C4 and KL only (tasks cut for budget, DECISIONS.md); the four review arms in four cells
    cells = [(0.25, 3), (1 / 3, 3), (0.5, 4), (0.5, 3)]
    return {m: [u(a, s, b, tasks=[]) for s, b in cells for a in ARMS] for m in CORE}


def e4():
    # frontier points with full metrics: 2-bit reference, 60% pruning alone and 60%+4-bit in every order
    return {m: [u("quant", 0.0, 2), u("prune", 0.6, 16), u("joint", 0.6, 4), u("p_then_q", 0.6, 4),
                u("q_then_p", 0.6, 4)] for m in CORE}


def e5_full(bits=(4,)):
    """The mission's E5 design (both models, prune-only seeds too); cut for budget, see e5()."""
    models = ["HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B"]
    out = {}
    for m in models:
        q = []
        for seed in (1, 2):
            q += [u("quant", 0.0, b, seed) for b in bits] + [u("prune", s, 16, seed) for s in SPARS]
            q += [u(o, s, b, seed) for b in bits for s in SPARS for o in ORDERS]
        out[m] = dedupe(q)
    return out


def e5(bits=(4,)):
    """E5 as run (DECISIONS.md #29): SmolLM2-360M, calibration seeds 1 and 2, 4-bit alone and every 4-bit cell
    in the three orders; perplexity, C4 and KL only."""
    q = []
    for seed in (1, 2):
        q += [u("quant", 0.0, b, seed, tasks=[]) for b in bits]
        q += [u(o, s, b, seed, tasks=[]) for b in bits for s in SPARS for o in ORDERS]
    return {"HuggingFaceTB/SmolLM2-360M": dedupe(q)}


def e6_full(models=("HuggingFaceTB/SmolLM2-1.7B",)):
    """The mission's E6 design (every cell in the three orders, MMLU on dense and 4-bit); cut for budget, see e6()."""
    out = {}
    for m in models:
        q = [u("dense", 0.0, 16, fewshot={"mmlu": 5}), u("quant", 0.0, 4, fewshot={"mmlu": 5}), u("quant", 0.0, 3)]
        q += [u("prune", s, 16) for s in (0.25, 0.5)]
        q += [u(o, s, b) for b in (4, 3) for s in (0.25, 0.5) for o in ORDERS]
        out[m] = q
    return out


def e6(models=("HuggingFaceTB/SmolLM2-1.7B",)):
    """E6 as run (DECISIONS.md #29, #31): perplexity, C4 and KL only. First the 7 frontier states (dense, 4-bit,
    3-bit, 50%+4 and 25%+4 in P->Q and joint), then Q->P at both 4-bit cells (the order spread at this scale)
    and pruning alone at 50% and 25% (the single-method baselines of the interaction and of the order rules)."""
    out = {}
    for m in models:
        q = [u("dense", 0.0, 16, tasks=[]), u("quant", 0.0, 4, tasks=[]), u("quant", 0.0, 3, tasks=[])]
        q += [u(o, s, 4, tasks=[]) for s in (0.5, 0.25) for o in ("p_then_q", "joint")]
        q += [u("q_then_p", s, 4, tasks=[]) for s in (0.5, 0.25)]
        q += [u("prune", s, 16, tasks=[]) for s in (0.5, 0.25)]
        out[m] = dedupe(q)
    return out


def dedupe(q):
    seen, out = set(), []
    for x in q:
        k = (x["order"], round(x["sparsity"], 4), x["bits"], x["seed"])
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def done_states():
    done = set()
    for f in glob.glob(os.path.join(ROOT, "results", "v2", "*", "*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("state_id"):
                    done.add(r["state_id"])
    return done


FAMILIES = {"E1": e1, "E3": e3, "E4": e4, "E5": e5, "E6": e6}


def main(argv):
    from scripts.run_grid import state_id
    fams = list(FAMILIES) if argv == ["all"] else argv
    done = done_states()
    os.makedirs(os.path.join(ROOT, "colab", "queues"), exist_ok=True)
    for fam in fams:
        for model, q in FAMILIES[fam]().items():
            todo = [x for x in q if state_id(model, x, "float32") not in done]
            path = os.path.join(ROOT, "colab", "queues", f"{fam}__{model.split('/')[-1]}.json")
            with open(path, "w") as f:
                json.dump(todo, f, indent=0)
            print(f"{fam} {model}: {len(q)} units, {len(q) - len(todo)} already done -> {len(todo)} queued ({path})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["E1"])

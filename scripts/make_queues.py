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
    # Qwen2.5-0.5B: only the two cells with a real Q->P penalty in E1 (50%+3 and 50%+4); 33%+3 (penalty under 1%)
    # and 25%+3 (Q->P beats P->Q) were cut for budget (DECISIONS.md #34, #35)
    order = {"Qwen/Qwen2.5-0.5B": [(0.5, 3), (0.5, 4)]}
    return {m: [u(a, s, b, tasks=[]) for s, b in order.get(m, cells) for a in ARMS] for m in CORE}


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


def e5(models=("HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-135M")):
    """E5 as run (DECISIONS.md #29, #38, #40): calibration seeds 1 and 2, the quantization-only state and every cell in
    the three orders; perplexity, C4 and KL only. 4-bit cells first (SmolLM2-360M on 2026-11-11, Qwen2.5-0.5B on
    2026-11-12), then the 3-bit cells (2026-11-13), then SmolLM2-135M at both bit-widths."""
    out = {}
    for m in models:
        q = []
        for b in (4, 3):
            for seed in (1, 2):
                q += [u("quant", 0.0, b, seed, tasks=[])]
                q += [u(o, s, b, seed, tasks=[]) for s in SPARS for o in ORDERS]
        out[m] = dedupe(q)
    return out


def e6_full(models=("HuggingFaceTB/SmolLM2-1.7B",)):
    """The mission's E6 design (every cell in the three orders, MMLU on dense and 4-bit); cut for budget, see e6()."""
    out = {}
    for m in models:
        q = [u("dense", 0.0, 16, fewshot={"mmlu": 5}), u("quant", 0.0, 4, fewshot={"mmlu": 5}), u("quant", 0.0, 3)]
        q += [u("prune", s, 16) for s in (0.25, 0.5)]
        q += [u(o, s, b) for b in (4, 3) for s in (0.25, 0.5) for o in ORDERS]
        out[m] = q
    return out


E6_MODELS = ("HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "allenai/OLMo-2-0425-1B")
E6_NO_KL = {"Qwen/Qwen2.5-3B"}        # a second fp32 copy for KL does not fit on the L4: perplexity only
E6_GRID = {"HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-1.5B", "allenai/OLMo-2-0425-1B"}  # the full E1 grid, not 8-bit


def e6(models=E6_MODELS):
    """E6, scale and family (DECISIONS.md #29, #31, #38, #40): perplexity, C4 and KL only (perplexity only where
    E6_NO_KL). Base design first: dense, 4- and 3-bit, 50%+4 and 25%+4 in P->Q and joint, Q->P at both 4-bit cells,
    pruning alone at 50% and 25%. For the models in E6_GRID it is then completed to the E1 grid without 8-bit:
    pruning alone at 33%, the 33%+4 cell and the three 3-bit cells in the three orders."""
    out = {}
    for m in models:
        kw = {"tasks": []}
        if m in E6_NO_KL:
            kw["kl"] = False
        q = [u("dense", 0.0, 16, **kw), u("quant", 0.0, 4, **kw), u("quant", 0.0, 3, **kw)]
        q += [u(o, s, 4, **kw) for s in (0.5, 0.25) for o in ("p_then_q", "joint")]
        q += [u("q_then_p", s, 4, **kw) for s in (0.5, 0.25)]
        q += [u("prune", s, 16, **kw) for s in (0.5, 0.25)]
        if m in E6_GRID:
            q += [u("prune", 1 / 3, 16, **kw)] + [u(o, 1 / 3, 4, **kw) for o in ORDERS]
            q += [u(o, s, 3, **kw) for s in SPARS for o in ORDERS]
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


def batch():
    """BATCH check (audit finding 17, DECISIONS.md #38): the four states whose benchmarks ran with one batch size
    (both dense references among them), re-evaluated with the batch groups used for every later state; compared
    item by item with E1. Not a study family: its per-item files carry the family name (scripts/run_grid.py)."""
    return {"HuggingFaceTB/SmolLM2-135M": [u("dense", 0.0, 16), u("quant", 0.0, 4), u("p_then_q", 0.5, 4)],
            "Qwen/Qwen2.5-0.5B": [u("dense", 0.0, 16)]}


FAMILIES = {"E1": e1, "E3": e3, "E4": e4, "E5": e5, "E6": e6}
CHECKS = {"BATCH": batch}  # queues for re-checks of existing states; not part of the analysis plan


def main(argv):
    from scripts.run_grid import state_id
    fams = list(FAMILIES) if argv == ["all"] else argv
    done = done_states()
    for fam in [f for f in fams if f in CHECKS]:  # re-checks run even though the state exists
        for model, q in CHECKS[fam]().items():
            path = os.path.join(ROOT, "colab", "queues", f"{fam}__{model.split('/')[-1]}.json")
            with open(path, "w") as f:
                json.dump(q, f, indent=0)
            print(f"{fam} {model}: {len(q)} units queued ({path})")
    fams = [f for f in fams if f not in CHECKS]
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

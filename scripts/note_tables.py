#!/usr/bin/env python3
"""Markdown tables for the note that accompanies the first author's revised draft (CPU only, no hand-typed numbers).

    python scripts/note_tables.py OUT_DIR

Reads the analysis outputs of scripts/make_tables.py (reports/analysis/*.csv) and the draft's published 16-bit
results (results/merged_results.json, as used by analysis/reproduce_paper.py), and writes one Markdown snippet per
table into OUT_DIR, for scripts/render_template.py ({{file:NAME.md}}):
  inventory.md        model states per family and model (results/v2)
  table1_compare.md   Table 1 of the draft (16-bit, first 500 items) against fp32 (seed 0), with the best order
  best_changes.md     cells whose best or worst order differs between the two
  rankings.md         order ranking per cell by six metrics (Q->P worst? best order?)
  kim_cells.md        the cells where 'weaker first' recommends quantizing first, by metric
  isr_compare.md      I_Sr of P->Q at 25%+4 and 33%+3 (draft method on draft data, fp32 with 95% intervals)
  spreads.md          best-to-worst order spread per model and cell (WikiText-2 ppl), 4-bit cells 25/33/50%
  seeds.md            seed checks (E5): per cell, mean gap between orders and paired t-test
  hardware.md         16-bit runs repeated on another GPU type
"""
import csv
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "reports", "analysis")
ORD = {"joint": "joint", "p_then_q": "P→Q", "q_then_p": "Q→P"}
CELLS = [(0.25, 4), (0.3333, 4), (0.5, 4), (0.25, 3), (0.3333, 3), (0.5, 3)]
CELL_LABEL = {(0.25, 4): "25%+4", (0.3333, 4): "33%+4", (0.5, 4): "50%+4",
              (0.25, 3): "25%+3", (0.3333, 3): "33%+3", (0.5, 3): "50%+3"}
CORE = ["SmolLM2-135M", "SmolLM2-360M", "Qwen2.5-0.5B"]
ALL_MODELS = CORE + ["SmolLM2-1.7B", "Qwen2.5-1.5B", "Qwen2.5-3B", "OLMo-2-0425-1B"]


def rows(name):
    p = os.path.join(A, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}"


def pct(x, nd=1, sign=True):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:+.{nd}f}%" if sign else f"{v:.{nd}f}%"


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out) + "\n"


def inventory():
    c = Counter()
    for p in glob.glob(os.path.join(ROOT, "results", "v2", "*", "*.jsonl")):
        fam = os.path.basename(os.path.dirname(p))
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok" and "state_id" in r:
                    c[(fam, r["state_id"].split("__")[0], (r.get("metadata") or {}).get("dtype") or r.get("dtype"))] += 1
    fams = sorted({k[0] for k in c})
    models = [m for m in ALL_MODELS if any(k[1] == m for k in c)]
    body = []
    for fam in fams:
        cnt = [sum(v for k, v in c.items() if k[0] == fam and k[1] == m) for m in models]
        dt = sorted({k[2] for k in c if k[0] == fam})
        body.append([fam] + [x or "" for x in cnt] + [sum(cnt), "/".join(str(d) for d in dt)])
    body.append(["**all**"] + [sum(v for k, v in c.items() if k[1] == m) for m in models]
                + [sum(c.values()), ""])
    return table(["family"] + models + ["total", "precision"], body)


def published():
    d = json.load(open(os.path.join(ROOT, "results", "merged_results.json")))
    out = {}
    for k, r in d.items():
        m, order, s, b, seed = k.split("|")
        out[(m, order, round(float(s), 4), int(b), int(seed))] = r
    return out


def v2_states():
    st = {}
    for r in rows("e1_states.csv"):
        st[r["state_id"]] = r
    return st


def ppl_v2(model, order, s, b, seed=0):
    sid = f"{model}__{order}__s{s:.4f}__b{b}__seed{seed}__float32"
    for fam in ("E1", "E4", "E5", "E6"):
        p = os.path.join(ROOT, "results", "v2", fam, f"{model}.jsonl")
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    if r.get("state_id") == sid and r.get("status") == "ok":
                        return (r.get("wikitext2") or {}).get("ppl")
    return None


def best(d):
    d = {k: v for k, v in d.items() if v is not None}
    return min(d, key=d.get) if len(d) == 3 else None


def worst(d):
    d = {k: v for k, v in d.items() if v is not None}
    return max(d, key=d.get) if len(d) == 3 else None


def table1_compare():
    pub = published()
    body, changes = [], []
    for m in CORE:
        for (s, b) in CELLS:
            old = {o: (pub.get((m, o, s, b, 0)) or {}).get("ppl") for o in ORD}
            new = {o: ppl_v2(m, o, s, b) for o in ORD}
            for o in ORD:
                star = "**" if best(new) == o else ""
                ostar = "**" if best(old) == o else ""
                d = 100 * (new[o] / old[o] - 1) if new[o] and old[o] else None
                body.append([m, CELL_LABEL[(s, b)], ORD[o], f"{ostar}{f(old[o])}{ostar}", f"{star}{f(new[o])}{star}",
                             pct(d) if d is not None else "n/a"])
            if best(old) != best(new) or worst(old) != worst(new):
                changes.append([m, CELL_LABEL[(s, b)], ORD.get(best(old), "n/a"), ORD.get(best(new), "n/a"),
                                ORD.get(worst(old), "n/a"), ORD.get(worst(new), "n/a"),
                                " / ".join(f(new[o]) for o in ORD)])
        for label, (o, s, b) in [("dense", ("dense", 0.0, 16)), ("4-bit", ("quant", 0.0, 4)), ("3-bit", ("quant", 0.0, 3)),
                                 ("25% pruned", ("prune", 0.25, 16)), ("33% pruned", ("prune", 0.3333, 16)),
                                 ("50% pruned", ("prune", 0.5, 16))]:
            old = (pub.get((m, o, s, b, 0)) or {}).get("ppl")
            new = ppl_v2(m, o, s, b)
            d = 100 * (new / old - 1) if new and old else None
            body.append([m, label, "single", f(old), f(new), pct(d) if d is not None else "n/a"])
    t1 = table(["model", "cell", "order", "draft (16-bit)", "fp32", "change"], body)
    t2 = table(["model", "cell", "best: draft", "best: fp32", "worst: draft", "worst: fp32", "fp32 joint / P→Q / Q→P"],
               changes) if changes else "No cell changes its best or worst order.\n"
    return t1, t2


def rankings():
    body = []
    metrics = [("wt_ppl", "WikiText-2 ppl"), ("c4_ppl", "C4 ppl"), ("wt_kl", "KL"), ("sr3_paper", "acc. 3 tasks"),
               ("sr9_v2", "acc. 9 tasks"), ("flips9", "flips")]
    for r in rows("q2_order_ranking.csv"):
        if r["seed"] != "0" or not r.get("wt_ppl_best"):
            continue  # other seeds, or a cell whose states are not all in yet
        row = [r["model"], r["cell"]]
        for k, _ in metrics:
            b, w = r.get(f"{k}_best"), r.get(f"{k}_worst")
            row.append(f"{ORD.get(b, b or 'n/a')} / {ORD.get(w, w or 'n/a')}")
        body.append(row)
    return table(["model", "cell"] + [f"{lab}: best / worst" for _, lab in metrics], body)


def kim_cells():
    body = []
    for r in rows("e7_kim.csv"):
        if r["family"] not in ("E1", "E6") or r["prediction"] != "q_then_p":
            continue
        ci = ""
        if r.get("observed_ci_lo"):
            ci = f" [{float(r['observed_ci_lo']):+.3f}, {float(r['observed_ci_hi']):+.3f}]"
        body.append([r["model"], r["cell"], r["metric_label"], f(r["prune_only"], 3), f(r["quant_only"], 3),
                     f(r["p_then_q"], 3), f(r["q_then_p"], 3), ORD.get(r["observed"], r["observed"]) + ci,
                     "yes" if r["correct"] == "True" else "no"])
    return table(["model", "cell", "metric", "P only", "Q only", "P→Q", "Q→P",
                  "better of P→Q, Q→P (Sr: 95% CI of P→Q − Q→P)", "rule right?"], body)


def draft_isr(pub, m, s, b):
    D = sum(pub[(m, "dense", 0.0, 16, 0)]["main"].values())
    sp = sum(pub[(m, "prune", s, 16, 0)]["main"].values()) / D
    sq = sum(pub[(m, "quant", 0.0, b, 0)]["main"].values()) / D
    return sum(pub[(m, "p_then_q", s, b, 0)]["main"].values()) / D - sp * sq


def isr_compare():
    pub = published()
    v = {}
    for r in rows("e1_isr.csv"):
        if r["set"] == "sr3_paper" and "__p_then_q__" in r["state_id"]:
            v[r["state_id"]] = r
    body = []
    for m in ALL_MODELS:
        row = [m]
        for (s, b) in [(0.25, 4), (0.3333, 3)]:
            try:
                old = f"{draft_isr(pub, m, s, b):+.3f}"
            except KeyError:
                old = "—"
            sid = f"{m}__p_then_q__s{s:.4f}__b{b}__seed0__float32"
            r = v.get(sid)
            new = f"{float(r['value']):+.3f} [{float(r['lo']):+.3f}, {float(r['hi']):+.3f}]" if r else "n/a"
            row += [old, new]
        body.append(row)
    return table(["model", "25%+4: draft", "25%+4: fp32 [95% CI]", "33%+3: draft", "33%+3: fp32 [95% CI]"], body)


def spreads():
    body = []
    by = defaultdict(dict)
    for r in rows("q2_order_ranking.csv"):
        if r["seed"] == "0" and r.get("wt_ppl_spread_pct"):
            by[r["model"]][r["cell"]] = float(r["wt_ppl_spread_pct"])
    cells = ["25%+4", "33%+4", "50%+4", "25%+3", "33%+3", "50%+3"]
    for m in ALL_MODELS:
        if m in by:
            body.append([m] + [pct(by[m].get(c), 1, False) if c in by[m] else "n/a" for c in cells])
    return table(["model"] + cells, body)


def seeds():
    body = []
    for r in rows("seed_order_pairs.csv"):
        if r["complete"] != "True":
            continue
        body.append([r["model"], r["cell"], f"{ORD[r['order_a']]} vs {ORD[r['order_b']]}", r["n_seeds"],
                     pct(100 * (math.exp(float(r["mean_log_ratio_b_over_a"])) - 1), 1),
                     "yes" if r["a_better_all_seeds"] == "True" else ("reversed" if r["b_better_all_seeds"] == "True" else "no"),
                     f(r["p_paired_t"], 3)])
    return table(["model", "cell", "pair (a vs b)", "seeds", "mean ppl(b)/ppl(a) − 1", "a better in every seed",
                  "paired t-test p"], body) if body else "No complete seed set yet.\n"


def hardware():
    body = [[r["family"], r["model"], r["state_id"].split("__")[1], r["sparsity"], r["bits"], r["dtype"], r["gpu"],
             f(r["ppl"]), r["reference"], f(r["ref_ppl"]), pct(r["rel_delta_pct"], 2)] for r in rows("repro_hardware.csv")]
    return table(["family", "model", "order", "s", "b", "dtype", "GPU", "ppl", "reference", "ref. ppl", "change"], body)


def pending():
    """Queues of the v3 runs (E5, E6, BATCH): states done (ok result line) out of planned."""
    done = defaultdict(set)
    for p in glob.glob(os.path.join(ROOT, "results", "v2", "*", "*.jsonl")):
        fam = os.path.basename(os.path.dirname(p))
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok" and "state_id" in r:
                    done[fam].add(r["state_id"])
    body = []
    for q in sorted(glob.glob(os.path.join(ROOT, "colab", "queues", "*.json"))):
        name = os.path.basename(q)[:-5]
        fam, model = name.split("__")[:2]
        if fam not in ("E5", "E6", "BATCH"):
            continue
        units = json.load(open(q))
        units = units.get("units", units) if isinstance(units, dict) else units
        sids = {f"{model}__{u['order']}__s{float(u['sparsity']):.4f}__b{int(u['bits'])}__seed{int(u.get('seed', 0))}__float32"
                for u in units}
        n = len(sids & done[fam])
        what = sorted({f"{int(u['bits'])}-bit" for u in units if u["order"] not in ("dense", "prune")})
        body.append([fam, model, ", ".join(what) or "dense", f"{n} of {len(sids)}", "done" if n == len(sids) else "running or queued"])
    return table(["family", "model", "bit-widths", "states with a result", "status"], body)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t1, t2 = table1_compare()
    parts = {"inventory.md": inventory(), "table1_compare.md": t1, "best_changes.md": t2, "rankings.md": rankings(),
             "kim_cells.md": kim_cells(), "isr_compare.md": isr_compare(), "spreads.md": spreads(), "seeds.md": seeds(),
             "hardware.md": hardware(), "pending.md": pending()}
    for k, v in parts.items():
        with open(os.path.join(out, k), "w") as fh:
            fh.write(v)
    print(f"{out}: {', '.join(parts)}")


if __name__ == "__main__":
    main()

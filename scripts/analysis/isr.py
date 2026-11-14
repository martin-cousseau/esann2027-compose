"""Model-level interaction I_Sr = Sr_PQ - Sr_P Sr_Q as the first author's Table 2 defines it, with intervals.

Definition (Table 2 of the draft and of v1): Sr is "the ratio of compressed to original performance summed over
tasks" over ARC-Easy (acc), PIQA (acc) and HellaSwag (acc_norm): Sr = sum_t acc(state, t) / sum_t acc(dense, t), a
ratio of sums. stats.sr with stats.SR_SETS["sr3_paper"] computes exactly this (paper metric of each task,
P.PAPER_MAIN_METRIC), and e1.py's I_Sr takes P and Q as the prune-only and quant-only states at the same s and b and
seed, with a paired bootstrap over items (the four states share each resample). So the v2 values with intervals are
e1's sr3_paper I_Sr; no other variant is needed.
Check: the published v1 runs (results/results_full.jsonl merged with results_fix.jsonl as analysis/reproduce_paper.py
merges them; 500-item subsets, field `main` = the paper's metric) are recomputed with this ratio of sums and with the
other reading of the sentence (the mean over tasks of per-task ratios); the ratio of sums reproduces Table 2.
IJCNN 7-8B rows: the published scores are the PUB dict literal of analysis/reproduce_paper.py, read with ast (never
retyped), and I_Sr is computed as that script does (Sr = sum of scores / sum of dense scores; 'all' = MMLU-Pro, BBH,
MATH; 'no MATH' = the first two).
"""
import ast
import os

from .common import MQ, P, PAPER_ORDERS, cell_label, finite, load_v1, short, write_csv

TABLE2_CELLS = ((0.25, 4), (round(1 / 3, 4), 3))           # the cells of Table 2 (P->Q)
IJCNN_CELLS = (("STwentyFiveBFour", "25%+4", ("j25_4", "p25", "q4")),   # the keys reproduce_paper.py combines
               ("SThirtyThreeBThree", "33%+3", ("j33_3", "p33", "q3")))
IJCNN_TASK_SETS = (("All", "MMLU-Pro, BBH, MATH", (0, 1, 2)), ("NoMath", "MMLU-Pro, BBH", (0, 1)))


def v2_rows(store, e1):
    """I_Sr (sr3_paper) with its paired-bootstrap interval for every combined state (three paper orders, seed 0) of
    the core models, and the sr9_v2 value next to it."""
    rows = []
    for model in [short(m) for m in MQ.CORE]:
        for s, b in store.cells(model, 0, PAPER_ORDERS):
            for o in PAPER_ORDERS:
                sid = store.sid(model, o, s, b, 0)
                isr = (e1["by_sid"].get(sid, {}).get("isr") or {}) if sid else {}
                v = isr.get("sr3_paper")
                rows.append({"model": model, "cell": cell_label(s, b), "sparsity": s, "bits": b, "order": o, "state_id": sid,
                             "table2_cell": (round(s, 4), b) in TABLE2_CELLS and o == "p_then_q",
                             "isr": v[0] if v else None, "lo": v[1] if v else None, "hi": v[2] if v else None,
                             "excludes_zero": (v[1] > 0 or v[2] < 0) if v and finite(v[1]) and finite(v[2]) else None,
                             "isr_sr9_v2": (isr.get("sr9_v2") or [None])[0]})
    return rows


def v1_check(paths):
    """Table 2 recomputed from the v1 runs: ratio of sums (the definition) and mean of per-task ratios."""
    v1 = load_v1(paths)
    if v1 is None:
        return []
    M = v1["merged"]

    def g(m, o, s, b):
        return M.get((m, o, round(float(s), 4), b, 0))

    rows = []
    for model in [short(m) for m in MQ.CORE]:
        d = g(model, "dense", 0.0, 16)
        for s, b in TABLE2_CELLS:
            p, q, c = g(model, "prune", s, 16), g(model, "quant", 0.0, b), g(model, "p_then_q", s, b)
            row = {"model": model, "cell": cell_label(s, b), "order": "p_then_q", "source": ", ".join(v1["files"][:2])}
            if not all(x is not None and x.get("main") for x in (d, p, q, c)):
                rows.append(row)
                continue
            D = d["main"]
            tasks = [t for t in P.PAPER_TASKS if t in D]

            def ros(x):
                return sum(x["main"][t] for t in tasks) / sum(D[t] for t in tasks)

            def mor(x):
                return sum(x["main"][t] / D[t] for t in tasks) / len(tasks)

            row.update(tasks=" ".join(tasks), sr_pq=ros(c), sr_p=ros(p), sr_q=ros(q),
                       isr_ratio_of_sums=ros(c) - ros(p) * ros(q), isr_mean_of_ratios=mor(c) - mor(p) * mor(q))
            rows.append(row)
    return rows


def _literal(node):
    """ast node -> value for literals and dict(key=literal, ...) calls (the form PUB uses); nothing is executed."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        return {kw.arg: _literal(kw.value) for kw in node.keywords}
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def pub_scores(path):
    """The PUB dict of analysis/reproduce_paper.py (published IJCNN 2025 scores), parsed, never executed."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUB" for t in node.targets):
            return _literal(node.value)
    return None


def ijcnn_tag(name):
    return name.split("-")[0].capitalize()          # Mistral-7B-v0.3 -> Mistral, LLaMA-3.1-8B -> Llama


def ijcnn_rows(paths):
    pub = pub_scores(paths.repro_script)
    rows = []
    for model, v in (pub or {}).items():
        for ctag, cell, (kc, kp, kq) in IJCNN_CELLS:
            for stag, tasks, idx in IJCNN_TASK_SETS:
                def sr(k):
                    return sum(v[k][i] for i in idx) / sum(v["dense"][i] for i in idx)
                rows.append({"model": model, "tag": ijcnn_tag(model), "cell": cell, "cell_tag": ctag, "tasks": tasks,
                             "set_tag": stag, "sr_pq": sr(kc), "sr_p": sr(kp), "sr_q": sr(kq), "isr": sr(kc) - sr(kp) * sr(kq),
                             "source": f"{paths.rel(paths.repro_script)} (PUB, read with ast)"})
    return rows


def analyze(store, e1, paths, missing):
    res = {"v2": v2_rows(store, e1), "v1": v1_check(paths), "ijcnn": ijcnn_rows(paths)}
    for r in res["v2"]:
        if r["isr"] is None and r["state_id"] is not None:
            missing.add("I_Sr (Table 2 definition)", f"{r['model']} {r['cell']} {r['order']}", "I_Sr not computable (see E1 I_Sr)")
    if not res["ijcnn"]:
        missing.add("I_Sr (Table 2 definition)", "IJCNN 7-8B rows", "PUB not found in analysis/reproduce_paper.py")
    return res


def write(res, paths):
    R = paths.reports
    write_csv(os.path.join(R, "isr_table2.csv"), res["v2"])
    write_csv(os.path.join(R, "isr_table2_v1_check.csv"), res["v1"])
    write_csv(os.path.join(R, "isr_ijcnn.csv"), res["ijcnn"])

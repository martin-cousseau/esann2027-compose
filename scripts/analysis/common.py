"""Shared loaders for the v2 analysis: result lines keyed by state_id, lookups, aligned per-item vectors.

Only reads (never writes) the result files:
  results/v2/<family>/<model>.jsonl      one line per model state (scripts/run_grid.py, run_unit). Families
                                         E1, E3, E4, E5, E6 share states (each state is run once), so they are
                                         merged by state_id; only lines with status "ok" are used.
  results/v2/<family>/failures__*.jsonl  units that raised an exception (listed as missing, never used)
  results/v2/items/<state_id>.json.gz    per-item correctness (src_v2/pipeline.py, write_items)
  results/v2/DET/<model>.jsonl           determinism re-run on a second machine (same state_id as in E1)
  results/v2/E2/<model>.jsonl            same-Hessian layer ablation (read by e2.py)
  results/results_full.jsonl, results/results_fix.jsonl, analysis/reproduce_paper.py   published v1 runs

The study plan (which states should exist) is read from scripts/make_queues.py, so a state that is planned
but absent is reported as missing instead of being skipped silently.
"""
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src_v2 import pipeline as P  # noqa: E402  (also puts review_code/ablation on sys.path)
from scripts import make_queues as MQ  # noqa: E402
from scripts.run_grid import short, state_id  # noqa: E402

PAPER_ORDERS = tuple(MQ.ORDERS)           # joint, p_then_q, q_then_p
ARMS = tuple(MQ.ARMS)                     # the E3 ablation arms
COMBINED = PAPER_ORDERS + ARMS
MECH_ARMS = ("q_then_p_refit", "q_then_p_pmask", "q_then_p_pmask_refit")   # what closes the Q->P penalty
DTYPE = "float32"
LAYER_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ORDER_TXT = {"dense": "dense", "prune": "prune only", "quant": "quant only", "joint": "joint", "p_then_q": "P->Q",
             "q_then_p": "Q->P", "q_then_p_refit": "Q->P refit", "q_then_p_pmask": "Q->P P-mask",
             "q_then_p_pmask_refit": "Q->P P-mask refit", "p_then_q_stale": "P->Q stale grid"}
ORDER_TEX = {"joint": "joint", "p_then_q": r"P$\to$Q", "q_then_p": r"Q$\to$P",
             "q_then_p_refit": r"Q$\to$P, grid re-fitted", "q_then_p_pmask": r"Q$\to$P, pruning-only mask",
             "q_then_p_pmask_refit": r"Q$\to$P, both", "p_then_q_stale": r"P$\to$Q, stale grid"}


class Paths:
    """Where the analysis reads (root/results) and writes (out_root/paper/generated, out_root/reports)."""

    def __init__(self, root=ROOT, out_root=None):
        self.root = root
        self.results_v2 = os.path.join(root, "results", "v2")
        self.results_v1 = os.path.join(root, "results")
        self.repro_script = os.path.join(root, "analysis", "reproduce_paper.py")
        out = out_root or root
        self.generated = os.path.join(out, "paper", "generated")
        self.figures = os.path.join(out, "reports", "figures")
        self.reports = os.path.join(out, "reports", "analysis")

    def rel(self, path):
        return os.path.relpath(path, self.root)


# ----------------------------------------------------------------------------- small helpers
def skey(s):
    return round(float(s), 4)


def pct(s):
    return round(float(s) * 100)


def cell_label(s, b):
    return f"{pct(s)}%+{int(b)}"


def cell_tex(s, b):
    return f"{pct(s)}\\% + {int(b)}"


def cell_sort_key(cell):
    s, b = cell
    return (-int(b), float(s))


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def layer_type(name):
    return name.split(".")[-1]


def clean(obj):
    """JSON-safe copy: NaN and infinities become None, tuples become lists, numpy scalars become floats."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            obj = obj.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(clean(obj), f, indent=1, sort_keys=False)
    os.replace(path + ".tmp", path)


def write_csv(path, rows, columns=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            out = []
            for c in columns:
                v = clean(r.get(c))
                out.append("" if v is None else (repr(v) if isinstance(v, float) else v))
            w.writerow(out)
    os.replace(path + ".tmp", path)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


class Missing:
    """Registry of everything that could not be shown, with the reason. Never filled in, only listed."""

    def __init__(self):
        self.items = []
        self._seen = set()

    def add(self, section, what, reason):
        k = (section, what, reason)
        if k not in self._seen:
            self._seen.add(k)
            self.items.append({"section": section, "what": what, "reason": reason})


# ----------------------------------------------------------------------------- the study plan
def planned_states():
    """{state_id: info} for every unit of every family in scripts/make_queues.py (the first family wins)."""
    out = {}
    for fam, fn in MQ.FAMILIES.items():
        for model, queue in fn().items():
            for u in queue:
                sid = state_id(model, u, DTYPE)
                if sid not in out:
                    out[sid] = {"family": fam, "model": model, "short": short(model), "order": u["order"],
                                "sparsity": float(u["sparsity"]), "bits": int(u["bits"]), "seed": int(u.get("seed", 0))}
    return out


def planned_models():
    seen = []
    for fam, fn in MQ.FAMILIES.items():
        for model in fn():
            if short(model) not in seen:
                seen.append(short(model))
    core = [short(m) for m in MQ.CORE]
    return core + [m for m in seen if m not in core]


# ----------------------------------------------------------------------------- result store
class Store:
    """Every usable v2 model state, keyed by state_id, plus the DET lines, failures and item files."""

    def __init__(self, paths=None):
        self.paths = paths or Paths()
        self.states, self.source = {}, {}
        self.det, self.det_source = {}, {}
        self.not_ok, self.failures, self.duplicates, self.other_dtype = [], [], [], []
        self.archived_failures, self.ignored_files = [], []
        self.item_problems = {}
        self._items = {}
        self.index = {}
        self.plan = planned_states()
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self):
        root = self.paths.results_v2
        fams = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != "items") \
            if os.path.isdir(root) else []
        for fam in fams:
            for f in sorted(os.listdir(os.path.join(root, fam))):
                path = os.path.join(root, fam, f)
                if not f.endswith(".jsonl"):
                    continue
                if "failures__" in f:  # failures__<model>.jsonl (current) or an archived copy (renamed by the orchestrator)
                    for r in P.read_jsonl(path):
                        if isinstance(r, dict):
                            rec = dict(r, _file=self.paths.rel(path))
                            (self.failures if f.startswith("failures__") else self.archived_failures).append(rec)
                    continue
                if fam == "E2":
                    continue  # different schema (one line per layer), read by e2.py
                if fam != "DET" and fam not in MQ.FAMILIES:
                    self.ignored_files.append(self.paths.rel(path))  # e.g. a pilot: not a study family
                    continue
                for r in P.read_jsonl(path):
                    if not isinstance(r, dict) or "state_id" not in r:
                        continue
                    sid = r["state_id"]
                    if r.get("status") != "ok":
                        self.not_ok.append({"state_id": sid, "status": r.get("status"), "file": self.paths.rel(path),
                                            "checks": r.get("checks")})
                        continue
                    if fam == "DET":
                        if sid not in self.det:
                            self.det[sid], self.det_source[sid] = r, self.paths.rel(path)
                        continue
                    if r.get("dtype", DTYPE) != DTYPE:
                        self.other_dtype.append({"state_id": sid, "file": self.paths.rel(path)})
                        continue
                    if sid in self.states:
                        self.duplicates.append({"state_id": sid, "kept": self.source[sid], "ignored": self.paths.rel(path)})
                        continue
                    self.states[sid], self.source[sid] = r, self.paths.rel(path)
        for sid, r in self.states.items():
            self.index[(short(r["model"]), r["order"], skey(r["sparsity"]), int(r["bits"]), int(r.get("seed", 0)))] = sid

    # ------------------------------------------------------------------ lookups
    def sid(self, model, order, s, b, seed=0):
        return self.index.get((model, order, skey(s), int(b), int(seed)))

    def get(self, model, order, s, b, seed=0):
        sid = self.sid(model, order, s, b, seed)
        return self.states.get(sid) if sid else None

    def dense(self, model):
        return self.get(model, "dense", 0.0, 16, 0)

    def prune_only(self, model, s, seed=0):
        return self.get(model, "prune", s, 16, seed)

    def quant_only(self, model, b, seed=0):
        return self.get(model, "quant", 0.0, b, seed)

    @staticmethod
    def planned_sid(model, order, s, b, seed=0):
        return f"{model}__{order}__s{float(s):.4f}__b{int(b)}__seed{int(seed)}__{DTYPE}"

    def is_planned(self, model, order, s, b, seed=0):
        return self.planned_sid(model, order, s, b, seed) in self.plan

    def models(self):
        found = sorted({short(r["model"]) for r in self.states.values()})
        plan = planned_models()
        return plan + [m for m in found if m not in plan]

    def model_states(self, model):
        return [sid for sid, r in self.states.items() if short(r["model"]) == model]

    def seeds(self, model):
        out = {int(r.get("seed", 0)) for sid, r in self.states.items() if short(r["model"]) == model}
        out |= {v["seed"] for v in self.plan.values() if v["short"] == model}
        return sorted(out)

    def cells(self, model, seed=0, orders=COMBINED):
        """(s, b) cells with a combined state present or planned for this model and seed, Table-1 order."""
        out = set()
        for r in self.states.values():
            if short(r["model"]) == model and r["order"] in orders and int(r.get("seed", 0)) == seed:
                out.add((skey(r["sparsity"]), int(r["bits"])))
        for v in self.plan.values():
            if v["short"] == model and v["order"] in orders and v["seed"] == seed:
                out.add((skey(v["sparsity"]), v["bits"]))
        return sorted(out, key=cell_sort_key)

    def quant_bits(self, model, seed=0):
        out = {int(r["bits"]) for r in self.states.values()
               if short(r["model"]) == model and r["order"] == "quant" and int(r.get("seed", 0)) == seed}
        out |= {v["bits"] for v in self.plan.values() if v["short"] == model and v["order"] == "quant" and v["seed"] == seed}
        return sorted(out, reverse=True)

    def label(self, sid):
        r = self.states.get(sid) or self.plan.get(sid)
        if r is None:
            return sid
        o = r["order"]
        if o == "dense":
            return "dense"
        if o == "quant":
            return f"{int(r['bits'])}-bit"
        if o == "prune":
            return f"{pct(r['sparsity'])}% pruned"
        return f"{cell_label(r['sparsity'], r['bits'])} {ORDER_TXT.get(o, o)}"

    # ------------------------------------------------------------------ per-item results
    def items_path(self, sid):
        cand = [os.path.join(self.paths.results_v2, "items", f"{sid}.json.gz")]
        r = self.states.get(sid)
        if r and r.get("items_file"):
            cand.append(os.path.join(self.paths.root, r["items_file"]))
        for c in cand:
            if os.path.exists(c):
                return c
        return None

    def items(self, sid):
        """{task: {"doc_id": [...], "acc": [...], ...}} or None (no file, unreadable, or sha256 differs)."""
        if sid in self._items:
            return self._items[sid]
        out, r = None, self.states.get(sid)
        if r is not None and r.get("items_file"):
            path = self.items_path(sid)
            if path is None:
                self.item_problems[sid] = "items file not mirrored yet"
            else:
                with open(path, "rb") as f:
                    blob = f.read()
                want = r.get("items_sha256")
                if want and hashlib.sha256(blob).hexdigest() != want:
                    self.item_problems[sid] = "items file sha256 differs from the result line (incomplete copy?)"
                else:
                    try:
                        out = json.loads(gzip.decompress(blob))
                    except Exception as e:  # noqa: BLE001
                        self.item_problems[sid] = f"items file unreadable ({type(e).__name__})"
        self._items[sid] = out
        return out

    def binary(self, sid, kind="v2", tasks=None):
        """{task: {doc_id: 0/1}}: v2 main metric per task ("v2") or the paper's metric ("paper")."""
        items = self.items(sid)
        if items is None:
            return None
        tasks = tasks or P.ZERO_SHOT_TASKS
        out = {}
        for t in tasks:
            if t not in items:
                continue
            m = task_metric(t, kind)
            if m is None or m not in items[t]:
                continue
            out[t] = {d: int(round(v)) for d, v in zip(items[t]["doc_id"], items[t][m])}
        return out


def task_metric(task, kind):
    """The metric of a task: 'v2' = the v2 protocol's main metric, 'paper' = the draft's choice (3 tasks)."""
    if kind == "v2":
        return P.main_metric(task)
    return P.PAPER_MAIN_METRIC.get(task)


def aligned_columns(store, sids, tasks, kinds=("v2", "paper")):
    """Item vectors of several states on common items, for the paired bootstrap.

    Returns ({task: (keys, X, doc_ids)}, notes): keys are (state_id, metric) pairs, X is an
    (n_keys, n_items) float array of 0/1, and the items are the doc_ids common to every state of `sids`
    that has the task (normally all of them). notes lists states whose doc_ids differ from the common set.
    """
    import numpy as np
    cols, notes = {}, []
    for t in tasks:
        metrics = sorted({task_metric(t, k) for k in kinds if task_metric(t, k)})
        have = []
        for sid in sids:
            it = store.items(sid)
            if it is not None and t in it and all(m in it[t] for m in metrics):
                have.append(sid)
        if not have:
            continue
        common = None
        for sid in have:
            ids = set(store.items(sid)[t]["doc_id"])
            common = ids if common is None else common & ids
        common = sorted(common)
        for sid in have:
            if len(store.items(sid)[t]["doc_id"]) != len(common):
                notes.append(f"{sid}: {t} has {len(store.items(sid)[t]['doc_id'])} items, {len(common)} common")
        keys, rows = [], []
        for sid in have:
            rec = store.items(sid)[t]
            pos = {d: i for i, d in enumerate(rec["doc_id"])}
            for m in metrics:
                vals = rec[m]
                keys.append((sid, m))
                rows.append([float(round(vals[pos[d]])) for d in common])
        cols[t] = (keys, np.asarray(rows, dtype=np.float64), common)
    return cols, notes


# ----------------------------------------------------------------------------- published v1 runs
def _affected_from_script(path):
    """The AFFECTED list of analysis/reproduce_paper.py, read with ast (the script is not executed)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "AFFECTED" for t in node.targets):
            return [tuple(x) for x in ast.literal_eval(node.value)]
    return None


def v1_key(r):
    return (short(r["model"]), r["order"], skey(r["sparsity"]), int(r["bits"]), int(r.get("seed", 0)))


def load_v1(paths):
    """The published runs merged as the paper does (analysis/reproduce_paper.py): results_full (16-bit) with
    the AFFECTED configurations replaced by their 32-bit re-runs from results_fix. None if files are absent."""
    fp, xp = os.path.join(paths.results_v1, "results_full.jsonl"), os.path.join(paths.results_v1, "results_fix.jsonl")
    if not (os.path.exists(fp) and os.path.exists(xp)):
        return None
    full = {v1_key(r): r for r in P.read_jsonl(fp)}
    fix = {v1_key(r): r for r in P.read_jsonl(xp)}
    affected = _affected_from_script(paths.repro_script) or []
    affected = [(k[0], k[1], skey(k[2]), int(k[3]), int(k[4])) for k in affected]
    merged = {k: dict(v, precision="fp16") for k, v in full.items()}
    for k in affected:
        if k in fix:
            merged[k] = dict(fix[k], precision="fp32")
    check = None
    mp = os.path.join(paths.results_v1, "merged_results.json")
    if os.path.exists(mp):
        with open(mp) as f:
            pub = json.load(f)
        diffs, n = [], 0
        for kk, v in pub.items():
            m, o, s, b, sd = kk.split("|")
            k = (m, o, skey(s), int(b), int(sd))
            if k in merged:
                n += 1
                same_prec = merged[k]["precision"] == v.get("precision")
                diffs.append(0.0 if (same_prec and merged[k]["ppl"] == v["ppl"]) else abs(merged[k]["ppl"] - v["ppl"]))
        check = {"file": paths.rel(mp), "n_common": n, "n_published": len(pub), "max_abs_ppl_diff": max(diffs) if diffs else None}
    return {"merged": merged, "full": full, "fix": fix, "affected": affected, "check": check,
            "files": [paths.rel(fp), paths.rel(xp), paths.rel(paths.repro_script)]}


def v1_layer_layout(paths):
    """{model short: (n_blocks, [linear names])} from the published per-layer diagnostics (all blocks logged)."""
    fp = os.path.join(paths.results_v1, "results_full.jsonl")
    out = {}
    if not os.path.exists(fp):
        return out
    for r in P.read_jsonl(fp):
        if r.get("diags"):
            m = short(r["model"])
            if m not in out:
                out[m] = (max(d["layer"] for d in r["diags"]) + 1, sorted({d["name"] for d in r["diags"]}))
    return out

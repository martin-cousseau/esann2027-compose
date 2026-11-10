"""Paired bootstrap over items, semantic retention (Sr), I_Sr, flips and the exact McNemar test.

Bootstrap: percentile intervals from B resamples of the items, drawn with replacement within each task.
One set of resamples per task is shared by every state of a model, so every difference, ratio or product
of states is paired item by item. The draws for a task depend only on (seed, task name, number of items,
B), so adding states never changes the interval of an existing comparison.

Sr follows the paper (and the review's analyze_results.py): the sum over tasks of the compressed model's
accuracy divided by the same sum for the dense model. The chance-normalized variant subtracts each task's
chance level from both sums (the review's sr(..., norm=True)). I_Sr = Sr_PQ - Sr_P * Sr_Q.
"""
import math
import zlib

import numpy as np

from .common import P, task_metric

BOOT_B = 10_000
BOOT_SEED = 20261110
BOOT_BATCH = 250
CI = (2.5, 97.5)

SR_SETS = {
    "sr3_v2": {"tasks": tuple(P.PAPER_TASKS), "kind": "v2", "chance": False,
               "label": "Sr over ARC-E, PIQA, HellaSwag, v2 metric"},
    "sr3_paper": {"tasks": tuple(P.PAPER_TASKS), "kind": "paper", "chance": False,
                  "label": "Sr over ARC-E, PIQA, HellaSwag, paper metric (acc, acc, acc_norm)"},
    "sr9_v2": {"tasks": tuple(P.ZERO_SHOT_TASKS), "kind": "v2", "chance": False,
               "label": "Sr-extended over the 9 zero-shot tasks, v2 metric"},
    "sr9_cn": {"tasks": tuple(P.ZERO_SHOT_TASKS), "kind": "v2", "chance": True,
               "label": "chance-normalized Sr over the 9 zero-shot tasks, v2 metric"},
}


class MissingData(KeyError):
    pass


class PairedBootstrap:
    """columns: {task: (keys, X, ...)} with X an (n_keys, n_items) 0/1 array over the same items per task."""

    def __init__(self, columns, B=BOOT_B, seed=BOOT_SEED, batch=BOOT_BATCH):
        self.B, self.seed = int(B), int(seed)
        self.point, self.boot, self.n = {}, {}, {}
        for task, spec in columns.items():
            keys, X = spec[0], np.asarray(spec[1], dtype=np.float64)
            n = X.shape[1]
            self.n[task] = n
            self.point[task] = {k: float(v) for k, v in zip(keys, X.mean(axis=1))}
            rng = np.random.default_rng([self.seed, zlib.crc32(task.encode())])
            out = np.empty((self.B, len(keys)))
            for a in range(0, self.B, batch):
                bb = min(batch, self.B - a)
                idx = rng.integers(0, n, size=(bb, n))
                counts = np.bincount((idx + n * np.arange(bb)[:, None]).ravel(), minlength=bb * n).reshape(bb, n)
                out[a:a + bb] = counts @ X.T / n
            self.boot[task] = ({k: i for i, k in enumerate(keys)}, out)

    def has(self, task, key):
        return task in self.point and key in self.point[task]

    def _point(self, task, key):
        try:
            return self.point[task][key]
        except KeyError:
            raise MissingData((task, key)) from None

    def _boot(self, task, key):
        try:
            pos, arr = self.boot[task]
            return arr[:, pos[key]]
        except KeyError:
            raise MissingData((task, key)) from None

    def stat(self, fn):
        """fn(get) -> value, where get(task, key) is a mean accuracy; fn is evaluated on the full sample (point)
        and on the B resamples at once (numpy arrays). Returns (point, lo, hi); raises MissingData."""
        pt = fn(self._point)
        bt = np.asarray(fn(self._boot), dtype=np.float64)
        bt = bt[np.isfinite(bt)]
        if not bt.size or not math.isfinite(pt):
            return float(pt), float("nan"), float("nan")
        lo, hi = np.percentile(bt, CI)
        return float(pt), float(lo), float(hi)


# ----------------------------------------------------------------------------- Sr and I_Sr
def chance_levels(boolq_rate):
    """Chance per task from src_v2/pipeline.CHANCE; BoolQ gets the majority-class rate (None if unknown)."""
    out = dict(P.CHANCE)
    out["boolq"] = boolq_rate
    return out


def sr(get, sid, dense_sid, spec, chance=None):
    num = den = 0.0
    for t in spec["tasks"]:
        m = task_metric(t, spec["kind"])
        c = 0.0
        if spec["chance"]:
            c = (chance or {}).get(t)
            if c is None:
                raise MissingData((t, "chance"))
        num = num + get(t, (sid, m)) - c
        den = den + get(t, (dense_sid, m)) - c
    return num / den


def isr(get, pq, p, q, dense_sid, spec, chance=None):
    return sr(get, pq, dense_sid, spec, chance) - sr(get, p, dense_sid, spec, chance) * sr(get, q, dense_sid, spec, chance)


def acc_mean(get, sid, tasks, kind):
    return sum(get(t, (sid, task_metric(t, kind))) for t in tasks) / len(tasks)


# ----------------------------------------------------------------------------- flips and McNemar
def flip_counts(state, dense, tasks):
    """Flips against the dense model from binary item dicts {task: {doc_id: 0/1}}.

    Per task and pooled over `tasks`: n (common items), cw (dense right, state wrong), wc (dense wrong, state
    right), flips = (cw + wc) / n. The pooled share equals extra_orders.flips on the same items."""
    out, N, CW, WC = {}, 0, 0, 0
    for t in tasks:
        if t not in state or t not in dense:
            continue
        common = set(state[t]) & set(dense[t])
        cw = sum(1 for k in common if dense[t][k] == 1 and state[t][k] == 0)
        wc = sum(1 for k in common if dense[t][k] == 0 and state[t][k] == 1)
        n = len(common)
        out[t] = {"n": n, "cw": cw, "wc": wc, "flips": (cw + wc) / n if n else float("nan")}
        N, CW, WC = N + n, CW + cw, WC + wc
    out["pooled"] = {"n": N, "cw": CW, "wc": WC, "flips": (CW + WC) / N if N else float("nan")}
    return out


def mcnemar(items_a, items_b, tasks):
    """extra_orders.mcnemar restricted to `tasks`: {task: (right in A only, right in B only, p)} + pooled."""
    X = P.extra()
    a = {t: items_a[t] for t in tasks if t in items_a and t in items_b}
    return X.mcnemar(a, items_b)


def flips_reference(items_a, items_b, tasks):
    X = P.extra()
    a = {t: items_a[t] for t in tasks if t in items_a and t in items_b}
    return X.flips(a, items_b)

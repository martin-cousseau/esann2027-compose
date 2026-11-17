"""E7: the order rules of Kim et al. and Harma et al., tested on the v2 states (analysis only, no GPU).

Kim et al. (ICLR 2026, arXiv 2603.18426), Progressive Intensity Hypothesis: "weaker perturbations should
precede stronger ones". The intensity of a method is its compression-equivalent ratio (CER, their
Definition 3): the compression ratio C' at which quantization alone reaches the same metric, found by
linear interpolation (their Sec. 3.2) on the quantization curve. As in their paper, b-bit quantization has
C = 16 / b (the dense model C = 1), and the CER of quantization is its own ratio. For a cell (s, b):
  prediction  P->Q when pruning alone at s is the weaker perturbation (CER_P < 16 / b), Q->P when it is the
              stronger one (CER_P > 16 / b);
  outcome     the better of P->Q and Q->P (Kim et al. test sequential orders only, so joint is left out);
  advantage   A = metric(Q->P) - metric(P->Q), signed so that A > 0 means P->Q is better; for perplexity and
              KL as log(metric(Q->P) / metric(P->Q)), so cells of different size compare. Their Hypothesis 1
              says A grows monotonically with the CER difference CER_Q - CER_P: Spearman rho per model.
A CER beyond the strongest quantization point on the curve is reported as such (no extrapolation); the
prediction is still defined, because that pruning is stronger than every quantizer on the curve.
Metrics: WikiText-2 and C4 perplexity, KL (WikiText-2), Sr-extended (9 tasks) and the paper's Sr (ARC-E acc, PIQA
acc, HellaSwag acc_norm); for the two Sr the outcome also gets the paired-bootstrap interval of Sr(P->Q) - Sr(Q->P).

Harma et al. (ICLR 2025, arXiv 2405.20935):
  threshold   their orthogonality threshold (Def. 3.11, Eq. 12) for a lower-is-better metric M:
              T_add = M(dense) + [M(P) - M(dense)] + [M(Q) - M(dense)], from the dense fp32 model; they expect a
              combined model to exceed it ("compounded errors"), i.e. non-orthogonality;
  null        the multiplicative null T_mult = M(dense) * [M(P) / M(dense)] * [M(Q) / M(dense)]: additivity in the
              loss (log perplexity) instead of in perplexity;
  order       their Theorem 3.5 (sparsity first adds no error; stated for magnitude pruning and max-scaled
              block formats, without compensation) predicts P->Q at least as good as Q->P.
Our setting is outside both papers' assumptions (OBS compensation, min-max asymmetric groups of 128); the
tests ask whether their rules still predict what happens.

Scope of the headline counts: the E1 cells of the core models and the E6 cells, seed 0, as the plan says
(E7 on E1 and E6). Other cells (E4's 60%) are in the CSV files with in_counts = False. Only seed 0: the
calibration-seed runs (E5) have no pruning-only states, so neither rule can be evaluated on them.
"""
import math
import os

from .common import MQ, PAPER_ORDERS, ORDER_TXT, cell_label, finite, write_csv, write_json
from .questions import _metric

# name, label, sign (+1 = lower is better), log-scale advantage
KIM_METRICS = (("wt_ppl", "WikiText-2 ppl", 1, True), ("c4_ppl", "C4 ppl", 1, True),
               ("wt_kl", "KL (WikiText-2)", 1, True), ("sr9_v2", "Sr-extended (9 tasks)", -1, False),
               ("sr3_paper", "Sr (ARC-E acc, PIQA acc, HellaSwag acc_norm: the paper's Sr)", -1, False))
NO_KL = {m.split("/")[-1] for m in MQ.E6_NO_KL}   # run without KL by design (a second fp32 copy does not fit)
HARMA_METRICS = (("wt_ppl", "WikiText-2 ppl"), ("c4_ppl", "C4 ppl"))
COUNT_FAMILIES = ("E1", "E6")
BASE_BITS = 16  # Kim et al.: C_Q = B_orig / B_Q, starting from a 16-bit model


def ratio_q(b):
    return BASE_BITS / float(b)


def quant_curve(store, e1, model, name, seed=0):
    """[(C, metric)] for dense (C = 1) and every quant-only state of the model, sorted by C."""
    pts = []
    d = store.dense(model)
    if d is not None:
        v = _metric(store, e1, d["state_id"], name)
        if finite(v):
            pts.append((1.0, v))
    for b in store.quant_bits(model, seed):
        q = store.quant_only(model, b, seed)
        if q is None:
            continue
        v = _metric(store, e1, q["state_id"], name)
        if finite(v):
            pts.append((ratio_q(b), v))
    return sorted(pts)


def cer(curve, value, sign):
    """Kim et al.'s CER of a state with this metric value: linear interpolation in C on the quantization curve.

    Returns (cer, where): where = "inside" (cer is a number), "below" (milder than the first point of the
    curve, the dense model) or "beyond" (stronger than its last point); cer is None outside the curve. The
    curve is walked from mild to strong; the first segment that brackets the value is used.
    """
    if len(curve) < 2 or not finite(value):
        return None, "no curve"
    g = [sign * m for _, m in curve]  # larger = worse
    x = sign * value
    if x < g[0]:
        return None, "below"
    for (c0, _), (c1, _), m0, m1 in zip(curve, curve[1:], g, g[1:]):
        lo, hi = min(m0, m1), max(m0, m1)
        if lo <= x <= hi:
            if m1 == m0:
                return c0, "inside"
            return c0 + (c1 - c0) * (x - m0) / (m1 - m0), "inside"
    return (None, "beyond") if x > max(g) else (None, "between non-monotone points")


def _advantage(qp, pq, sign, log):
    if not (finite(qp) and finite(pq)):
        return None
    if log:
        if qp <= 0 or pq <= 0:
            return None
        return math.log(qp / pq) * sign
    return sign * (qp - pq)


def _scope(store, model, s, b, seed=0):
    for o in ("p_then_q", "q_then_p", "joint"):
        p = store.plan.get(store.planned_sid(model, o, s, b, seed))
        if p:
            return p["family"]
    return "unplanned"


def _sr_order_ci(e1, model, s, b, seed, sr_set="sr9_v2"):
    """Paired-bootstrap 95% interval of Sr(P->Q) - Sr(Q->P) (Sr set sr_set) from the E1 analysis, or None."""
    for x in e1.get("sr_orders", []):
        if (x["model"] == model and x["cell"] == cell_label(s, b) and x["seed"] == seed and x["set"] == sr_set
                and {x["order_a"], x["order_b"]} == {"p_then_q", "q_then_p"}):
            f = 1 if x["order_a"] == "p_then_q" else -1
            lo, hi = sorted((f * x["lo"], f * x["hi"]))
            return f * x["diff"], lo, hi
    return None


def kim(store, e1, missing, seed=0):
    rows = []
    for model in store.models():
        curves = {name: quant_curve(store, e1, model, name, seed) for name, *_ in KIM_METRICS}
        for s, b in store.cells(model, seed, PAPER_ORDERS):
            fam = _scope(store, model, s, b, seed)
            pr, qr = store.prune_only(model, s, seed), store.quant_only(model, b, seed)
            pq, qp = store.get(model, "p_then_q", s, b, seed), store.get(model, "q_then_p", s, b, seed)
            lack = [n for n, x in (("prune-only", pr), ("quant-only", qr), ("P->Q", pq), ("Q->P", qp)) if x is None]
            if lack:
                missing.add("E7 order rules (Kim)", f"{model} {cell_label(s, b)} seed {seed}", "missing: " + ", ".join(lack))
                continue
            for name, label, sign, log in KIM_METRICS:
                mp, mq = _metric(store, e1, pr["state_id"], name), _metric(store, e1, qr["state_id"], name)
                mpq, mqp = _metric(store, e1, pq["state_id"], name), _metric(store, e1, qp["state_id"], name)
                if not all(finite(v) for v in (mp, mq, mpq, mqp)):
                    if name == "wt_kl" and model in NO_KL:
                        continue  # not measured by design (scripts/make_queues.py E6_NO_KL)
                    if not name.startswith("sr") or any(st.get("items_file") for st in (pr, qr, pq, qp)):
                        missing.add("E7 order rules (Kim)", f"{model} {cell_label(s, b)} {label}", "metric not available for all four states")
                    continue
                c_p, where = cer(curves[name], mp, sign)
                c_q = ratio_q(b)
                if where == "inside":
                    pred = "p_then_q" if c_p < c_q else ("q_then_p" if c_p > c_q else "tie")
                elif where == "below":
                    pred = "p_then_q"
                elif where == "beyond":
                    pred = "q_then_p"
                else:
                    pred = None
                direct = "p_then_q" if sign * mp < sign * mq else ("q_then_p" if sign * mp > sign * mq else "tie")
                adv = _advantage(mqp, mpq, sign, log)
                obs = None if adv is None else ("p_then_q" if adv > 0 else ("q_then_p" if adv < 0 else "tie"))
                ci = _sr_order_ci(e1, model, s, b, seed, name) if name.startswith("sr") else None
                sig = None if ci is None else (ci[1] > 0 or ci[2] < 0)  # interval excludes 0
                rows.append({
                    "model": model, "family": fam, "in_counts": fam in COUNT_FAMILIES, "seed": seed,
                    "cell": cell_label(s, b), "sparsity": s, "bits": b, "metric": name, "metric_label": label,
                    "prune_only": mp, "quant_only": mq, "p_then_q": mpq, "q_then_p": mqp,
                    "cer_prune": c_p, "cer_prune_where": where, "cer_quant": c_q,
                    "cer_diff": (c_q - c_p) if c_p is not None else None,
                    "curve_points": len(curves[name]), "curve": ";".join(f"{c:.4g}:{m:.6g}" for c, m in curves[name]),
                    "weaker": {"p_then_q": "prune", "q_then_p": "quant"}.get(pred, pred),
                    "prediction": pred, "prediction_direct": direct, "prediction_agrees_direct": pred == direct,
                    "advantage_pq": adv, "advantage_scale": "log ratio" if log else "difference",
                    "observed": obs, "observed_ci_lo": ci[1] if ci else None, "observed_ci_hi": ci[2] if ci else None,
                    "observed_significant": sig, "correct": (pred == obs) if pred in ("p_then_q", "q_then_p") and obs in ("p_then_q", "q_then_p") else None,
                    "states": ";".join(x["state_id"] for x in (pr, qr, pq, qp))})
    return rows


def harma(store, e1, missing, seed=0):
    rows = []
    for model in store.models():
        d = store.dense(model)
        for s, b in store.cells(model, seed, PAPER_ORDERS):
            fam = _scope(store, model, s, b, seed)
            pr, qr = store.prune_only(model, s, seed), store.quant_only(model, b, seed)
            if d is None or pr is None or qr is None:
                lack = [n for n, x in (("dense", d), ("prune-only", pr), ("quant-only", qr)) if x is None]
                missing.add("E7 order rules (Harma)", f"{model} {cell_label(s, b)} seed {seed}", "missing: " + ", ".join(lack))
                continue
            for name, label in HARMA_METRICS:
                md, mp, mq = (_metric(store, e1, x["state_id"], name) for x in (d, pr, qr))
                if not all(finite(v) and v > 0 for v in (md, mp, mq)):
                    continue
                t_add = md + (mp - md) + (mq - md)
                t_mult = md * (mp / md) * (mq / md)
                vals = {}
                for o in PAPER_ORDERS:
                    r = store.get(model, o, s, b, seed)
                    if r is None:
                        if store.is_planned(model, o, s, b, seed):
                            missing.add("E7 order rules (Harma)", f"{model} {cell_label(s, b)} {ORDER_TXT[o]}", "not available yet")
                        continue
                    m = _metric(store, e1, r["state_id"], name)
                    if not finite(m):
                        continue
                    vals[o] = m
                    rows.append({
                        "model": model, "family": fam, "in_counts": fam in COUNT_FAMILIES, "seed": seed,
                        "cell": cell_label(s, b), "sparsity": s, "bits": b, "order": o, "metric": name,
                        "metric_label": label, "dense": md, "prune_only": mp, "quant_only": mq, "combined": m,
                        "threshold_additive": t_add, "threshold_multiplicative": t_mult,
                        "exceeds_additive": m > t_add, "exceeds_multiplicative": m > t_mult,
                        "excess_additive_pct": 100 * (m / t_add - 1), "excess_multiplicative_pct": 100 * (m / t_mult - 1),
                        "state_id": r["state_id"]})
                if "p_then_q" in vals and "q_then_p" in vals:
                    for row in rows[-len(vals):]:
                        row["pq_better_than_qp"] = vals["p_then_q"] < vals["q_then_p"]
                        row["order_gap_qp_over_pq_pct"] = 100 * (vals["q_then_p"] / vals["p_then_q"] - 1)
    return rows


def _spearman(pairs):
    pairs = [(a, b) for a, b in pairs if finite(a) and finite(b)]
    if len(pairs) < 3:
        return None, None, len(pairs)
    from scipy import stats as sps
    res = sps.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    rho, p = float(res.statistic), float(res.pvalue)
    return (rho if math.isfinite(rho) else None), (p if math.isfinite(p) else None), len(pairs)


def summarize(kim_rows, harma_rows):
    out = {"kim": {}, "harma_order": {}, "harma_threshold": {}, "scope": list(COUNT_FAMILIES)}
    models = []
    for r in kim_rows + harma_rows:
        if r["model"] not in models:
            models.append(r["model"])
    for name, *_ in KIM_METRICS:
        for m in models + ["all"]:
            R = [r for r in kim_rows if r["metric"] == name and r["in_counts"] and (m == "all" or r["model"] == m)]
            D = [r for r in R if r["correct"] is not None]
            k = {"cells": len(R), "decided": len(D), "correct": sum(r["correct"] for r in D),
                 "predicted_qp": sum(r["prediction"] == "q_then_p" for r in D),
                 "predicted_qp_correct": sum(r["prediction"] == "q_then_p" and r["correct"] for r in D),
                 "predicted_pq": sum(r["prediction"] == "p_then_q" for r in D),
                 "predicted_pq_correct": sum(r["prediction"] == "p_then_q" and r["correct"] for r in D),
                 "observed_pq": sum(r["observed"] == "p_then_q" for r in R), "observed_qp": sum(r["observed"] == "q_then_p" for r in R),
                 "ties": sum(r["observed"] == "tie" or r["prediction"] == "tie" for r in R),
                 "cer_beyond_curve": sum(r["cer_prune_where"] == "beyond" for r in R),
                 "prediction_disagrees_direct": sum(not r["prediction_agrees_direct"] for r in R),
                 "smallest_abs_advantage": min((abs(r["advantage_pq"]) for r in R), default=None)}
            S_ = [r for r in D if r["observed_significant"]]
            if name.startswith("sr"):  # outcome decided by a paired-bootstrap interval that excludes 0
                k.update({"significant": len(S_), "significant_correct": sum(r["correct"] for r in S_)})
            if m != "all":  # Hypothesis 1: monotone in the CER difference, within one model
                k["rho"], k["rho_p"], k["rho_n"] = _spearman([(r["cer_diff"], r["advantage_pq"]) for r in R])
            out["kim"][f"{name}|{m}"] = k
    for name, _ in HARMA_METRICS:
        for m in models + ["all"]:
            R = [r for r in harma_rows if r["metric"] == name and r["in_counts"] and (m == "all" or r["model"] == m)]
            cells = {(r["model"], r["cell"]): r for r in R if "pq_better_than_qp" in r}
            out["harma_order"][f"{name}|{m}"] = {"cells": len(cells), "pq_better": sum(r["pq_better_than_qp"] for r in cells.values()),
                                                 "max_gap_pct": max((r["order_gap_qp_over_pq_pct"] for r in cells.values()), default=None),
                                                 "min_gap_pct": min((r["order_gap_qp_over_pq_pct"] for r in cells.values()), default=None)}
            for o in PAPER_ORDERS:
                Ro = [r for r in R if r["order"] == o]
                out["harma_threshold"][f"{name}|{o}|{m}"] = {
                    "n": len(Ro), "exceeds_additive": sum(r["exceeds_additive"] for r in Ro),
                    "exceeds_multiplicative": sum(r["exceeds_multiplicative"] for r in Ro),
                    "max_excess_additive_pct": max((r["excess_additive_pct"] for r in Ro), default=None),
                    "min_excess_additive_pct": min((r["excess_additive_pct"] for r in Ro), default=None),
                    "max_excess_multiplicative_pct": max((r["excess_multiplicative_pct"] for r in Ro), default=None),
                    "min_excess_multiplicative_pct": min((r["excess_multiplicative_pct"] for r in Ro), default=None)}
    out["models"] = models
    return out


def analyze(store, e1, missing, seed=0):
    k = kim(store, e1, missing, seed)
    h = harma(store, e1, missing, seed)
    return {"kim": k, "harma": h, "summary": summarize(k, h)}


def write(res, paths):
    R = paths.reports
    write_csv(os.path.join(R, "e7_kim.csv"), res["kim"])
    write_csv(os.path.join(R, "e7_harma.csv"), res["harma"])
    write_json(os.path.join(R, "e7_summary.json"), res["summary"])

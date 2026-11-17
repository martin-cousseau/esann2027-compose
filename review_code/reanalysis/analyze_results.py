"""Re-analysis of the 93 runs of the ESANN 2027 draft, from the repository's raw result files.

Usage (CPU, Python standard library only; reads nothing outside the repository):
    python analyze_results.py <path to esann2027-compose> [--csv storage_points.csv]

Sections
  A  reproduction: every number of Table 1, Table 2 and the text, recomputed and compared with the paper
  B  the six 16-bit runs that were replaced, the 32-bit controls, and which Table 1 cells mix precisions
  C  unit checks over every logged layer, and the number of weights actually compressed per model
  D  layer-level interaction per layer (not only the mean), by sublayer type, and by depth (block 0 is
     the only block calibrated on identical inputs in every run)
  E  model-level additivity in perplexity under two nulls (additive perplexity as in Harma et al.;
     additive loss, i.e. multiplicative perplexity, the analogue of the paper's I_Sr)
  F  the weaker-first rule of Kim et al. (ICLR 2026) on Table 1
  G  calibration-seed noise (paired by seed) and the joint vs P->Q gaps it has to explain
  H  task scores: standard errors, the chance floor of Sr, a rough interval for I_Sr
  I  storage with the sparsity mask counted, equal-storage comparisons, and a CSV for the chart
  J  exact zeros created by rounding, and the Q->P = 3-bit identity
  K  what each result line records (no GPU, precision or package versions)
"""
import argparse
import json
import math
import statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--csv", default=None, help="write the storage points used for the review's chart")
args = ap.parse_args()

K = lambda r: (r["model"].split("/")[1], r["order"], round(r["sparsity"], 4), r["bits"], r.get("seed", 0))
full = {K(r): r for r in map(json.loads, open(f"{args.repo}/results/results_full.jsonl"))}
fix = {K(r): r for r in map(json.loads, open(f"{args.repo}/results/results_fix.jsonl"))}
AFFECTED = [("SmolLM2-360M", "joint", 0.25, 3, 0), ("SmolLM2-360M", "p_then_q", 0.3333, 4, 0),
            ("SmolLM2-360M", "q_then_p", 0.3333, 3, 0), ("SmolLM2-360M", "joint", 0.3333, 3, 0),
            ("SmolLM2-360M", "joint", 0.5, 3, 0), ("SmolLM2-360M", "p_then_q", 0.5, 3, 0)]
M = {k: dict(v, precision="fp16") for k, v in full.items()}
for k in AFFECTED:
    M[k] = dict(fix[k], precision="fp32")


def g(m, o, s, b, sd=0):
    # the seed-1/2 pruning-only runs are stored with bits=4 (a logging slip); look them up either way
    k = (m, o, round(float(s), 4), b, sd)
    if k not in M and o == "prune":
        k = (m, o, round(float(s), 4), 4 if b == 16 else 16, sd)
    return M[k]


MODELS = ["SmolLM2-135M", "SmolLM2-360M", "Qwen2.5-0.5B"]
CELLS = [(0.25, 4), (1 / 3, 4), (0.5, 4), (0.25, 3), (1 / 3, 3), (0.5, 3)]
ORD = ["joint", "p_then_q", "q_then_p"]
NAME = {"joint": "joint", "p_then_q": "P->Q", "q_then_p": "Q->P"}
cell = lambda s, b: f"{round(s * 100)}%+{b}"


def hr(t):
    print("\n" + "=" * 10 + " " + t + " " + "=" * 10)


def p_t_df2(t):
    """Two-sided p-value of a t statistic with 2 degrees of freedom (closed form)."""
    return 1 - abs(t) / math.sqrt(t * t + 2)


def layer_map(r):
    return {(d["layer"], d["name"]): d for d in r["diags"]}


def interaction(m, o, s, b, sd=0):
    """The paper's Fig. 1 quantity: (mean E_PQ - mean E_P - mean E_Q) / (mean E_P + mean E_Q), in %."""
    ep, eq = g(m, "prune", s, 16, sd)["mean_rel_err"], g(m, "quant", 0, b, sd)["mean_rel_err"]
    return 100 * (g(m, o, s, b, sd)["mean_rel_err"] - ep - eq) / (ep + eq)


# ============================================================================= A
hr("A. reproduction of the paper's numbers")
PAPER_T1 = {  # (P only, Q only, joint, P->Q, Q->P) as printed in Table 1
    "SmolLM2-135M": [(16.48, 21.77, 23.04, 21.86, 24.03), (18.39, 21.77, 26.08, 24.11, 27.73),
                     (32.89, 21.77, 45.45, 40.15, 49.68), (16.48, 58.17, 60.87, 56.88, 58.17),
                     (18.39, 58.17, 63.18, 58.97, 68.00), (32.89, 58.17, 88.24, 97.97, 150.10)],
    "SmolLM2-360M": [(12.07, 13.29, 13.17, 13.32, 14.34), (12.86, 13.29, 14.12, 14.20, 15.53),
                     (18.64, 13.29, 20.82, 21.02, 23.82), (12.07, 25.92, 20.11, 20.77, 25.92),
                     (12.86, 25.92, 20.34, 20.88, 30.50), (18.64, 25.92, 28.71, 30.07, 57.80)],
    "Qwen2.5-0.5B": [(13.63, 14.24, 14.68, 14.71, 15.02), (14.52, 14.24, 15.62, 15.69, 15.99),
                     (20.21, 14.24, 21.61, 22.44, 22.74), (13.63, 21.95, 21.96, 22.11, 21.95),
                     (14.52, 21.95, 22.00, 22.33, 22.96), (20.21, 21.95, 28.41, 29.69, 35.61)]}
PAPER_DENSE = {"SmolLM2-135M": 15.42, "SmolLM2-360M": 11.46, "Qwen2.5-0.5B": 13.07}
worst = 0.0
for m in MODELS:
    worst = max(worst, abs(round(g(m, "dense", 0, 16)["ppl"], 2) - PAPER_DENSE[m]))
    for (s, b), row in zip(CELLS, PAPER_T1[m]):
        mine = (g(m, "prune", s, 16)["ppl"], g(m, "quant", 0, b)["ppl"]) + tuple(g(m, o, s, b)["ppl"] for o in ORD)
        worst = max(worst, max(abs(round(x, 2) - y) for x, y in zip(mine, row)))
print(f"  Table 1 (all 93 printed perplexities): max |recomputed - printed| = {worst:.3f}")

wins, worst_cnt, spreads = defaultdict(int), defaultdict(int), {}
for m in MODELS:
    for s, b in CELLS:
        v = {o: g(m, o, s, b)["ppl"] for o in ORD}
        wins[(m, min(v, key=v.get))] += 1
        worst_cnt[max(v, key=v.get)] += 1
        spreads[(m, s, b)] = max(v.values()) / min(v.values())
I = {(m, s, b, o): interaction(m, o, s, b) for m in MODELS for s, b in CELLS for o in ORD}
mf = [I[(m, s, b, o)] for m in MODELS for s, b in CELLS for o in ("joint", "p_then_q")]
q4 = [I[(m, s, b, "q_then_p")] for m in MODELS for s, b in CELLS if b == 4]
jr = [-I[(m, s, b, "joint")] for m in MODELS for s, b in CELLS]
pr = [-I[(m, s, b, "p_then_q")] for m in MODELS for s, b in CELLS]
print(f"  mask-first sub-additive (mean over layers): {sum(x < 0 for x in mf)}/36 "
      f"| joint {min(jr):.1f}-{max(jr):.1f}% | P->Q {min(pr):.1f}-{max(pr):.1f}%  [paper: 36/36, 6-24%, 3-18%]")
print(f"  Q->P super-additive at 4 bits: {sum(x > 0 for x in q4)}/9, {min(q4):+.1f} to {max(q4):+.1f}%  [paper: 9/9, +1 to +7%]")
print(f"  Q->P worst order: {worst_cnt['q_then_p']}/18; max worst/best ratio "
      f"{max(spreads.values()):.2f}  [paper: 16/18, up to 2x]")
print(f"  50%+3 worst/best: 135M {spreads[('SmolLM2-135M', 0.5, 3)]:.2f}x, 360M {spreads[('SmolLM2-360M', 0.5, 3)]:.2f}x  [paper: 1.7x, 2.0x]")
print(f"  best order: joint on 360M {wins[('SmolLM2-360M', 'joint')]}/6 + Qwen {wins[('Qwen2.5-0.5B', 'joint')]}/6;"
      f" P->Q on 135M {wins[('SmolLM2-135M', 'p_then_q')]}/6  [paper: 11 of 12, 5 of 6]")
for m in MODELS:
    sp4 = [100 * (spreads[(m, s, b)] - 1) for s, b in CELLS if b == 4]
    print(f"  {m}: 4-bit best-to-worst spread {min(sp4):.1f}-{max(sp4):.1f}%")
print("  [paper: 10-24% (135M), 9-14% (360M), 2-5% (Qwen)]")
j4 = all(I[(m, s, b, "joint")] < min(I[(m, s, b, "p_then_q")], I[(m, s, b, "q_then_p")])
         for m in MODELS for s, b in CELLS if b == 4)
print(f"  joint has the most negative interaction (smallest layer error) in every 4-bit cell: {j4}")


# ============================================================================= B
hr("B. precision: replaced 16-bit runs, 32-bit controls, mixed-precision cells")
for k in AFFECTED:
    a, b = full[k], fix[k]
    fail = "perplexity NaN (forward overflow)" if not math.isfinite(a["ppl"]) else \
        f"finite perplexity {a['ppl']:.2f} = {a['ppl'] / b['ppl']:.2f}x the 32-bit value, layer error NaN (Hessian overflow)"
    print(f"  {k[0]} {NAME[k[1]]:5s} {cell(k[2], k[3])}: {fail}; 32-bit {b['ppl']:.2f}")
d16, d32 = full[("SmolLM2-360M", "dense", 0.0, 16, 0)]["ppl"], fix[("SmolLM2-360M", "dense", 0.0, 16, 0)]["ppl"]
print(f"  dense 360M: 16-bit {d16:.4f} vs 32-bit {d32:.4f} (two separate sessions)")
for o in ORD:
    a, b = full[("SmolLM2-360M", o, 0.25, 4, 0)]["ppl"], fix[("SmolLM2-360M", o, 0.25, 4, 0)]["ppl"]
    print(f"  control 25%+4 {NAME[o]:5s}: 16-bit {a:.3f} 32-bit {b:.3f} ({100 * (b - a) / a:+.2f}%)")
gj16 = full[("SmolLM2-360M", "p_then_q", 0.25, 4, 0)]["ppl"] / full[("SmolLM2-360M", "joint", 0.25, 4, 0)]["ppl"] - 1
gj32 = fix[("SmolLM2-360M", "p_then_q", 0.25, 4, 0)]["ppl"] / fix[("SmolLM2-360M", "joint", 0.25, 4, 0)]["ppl"] - 1
print(f"  360M 25%+4 joint-vs-P->Q gap: {100 * gj16:.1f}% in 16-bit, {100 * gj32:.1f}% in 32-bit")
for s, b in CELLS:
    prec = {o: g("SmolLM2-360M", o, s, b)["precision"] for o in ORD}
    if len(set(prec.values())) > 1 or "fp32" in prec.values():
        print(f"  360M {cell(s, b)}: " + ", ".join(f"{NAME[o]} {p}" for o, p in prec.items())
              + "  (P only and Q only are 16-bit)")


# ============================================================================= C
hr("C. unit checks over every logged layer; weights actually compressed")
n = bad_kept = bad_lev = 0
dev = 0.0
for r in list(full.values()) + list(fix.values()):
    for d in r["diags"]:
        n += 1
        if r["order"] != "quant":
            dev = max(dev, abs(d["mask_frac"] - r["sparsity"]))
        bad_kept += not d["mask_kept"]
        if r["order"] != "prune" and r["bits"] < 16:
            bad_lev += d["max_levels"] > 2 ** r["bits"]
print(f"  {n} layer records: max |mask fraction - target| {dev:.1e}; pruned weights not zero: {bad_kept};"
      f" groups with more than 2^b levels: {bad_lev}")
for m in MODELS:
    r = g(m, "prune", 0.25, 16)
    tot = sum(d["shape"][0] * d["shape"][1] for d in r["diags"])
    print(f"  {m}: {max(d['layer'] for d in r['diags']) + 1} blocks, {len(r['diags'])} linear layers,"
          f" {tot / 1e6:.1f}M weights compressed")


# ============================================================================= D
hr("D. layer-level interaction per layer, by type, by depth")
tot, bytype = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
for m in MODELS:
    for s, b in CELLS:
        P, Q = layer_map(g(m, "prune", s, 16)), layer_map(g(m, "quant", 0, b))
        for o in ORD:
            C = layer_map(g(m, o, s, b))
            for key in C:
                sub = C[key]["rel_err"] < P[key]["rel_err"] + Q[key]["rel_err"]
                tot[(o, b)][0] += sub
                tot[(o, b)][1] += 1
                t = key[1].split(".")[-1]
                bytype[(o, b, t)][0] += sub
                bytype[(o, b, t)][1] += 1
for o in ORD:
    for b in (4, 3):
        a, nn_ = tot[(o, b)]
        types = sorted({t for (oo, bb, t) in bytype if oo == o and bb == b})
        print(f"  {NAME[o]:5s} {b}-bit: sub-additive layers {a}/{nn_} = {100 * a / nn_:.1f}% | "
              + " ".join(f"{t} {100 * bytype[(o, b, t)][0] / bytype[(o, b, t)][1]:.0f}%" for t in types))

print("  depth profile, I_E/(E_P+E_Q) in % from sums over the layers of each group:")
print("  (block 0: identical calibration inputs in every run; later blocks: inputs differ between runs)")
for m in MODELS:
    for s, b in CELLS:
        P, Q = layer_map(g(m, "prune", s, 16)), layer_map(g(m, "quant", 0, b))
        parts = []
        for o in ORD:
            C = layer_map(g(m, o, s, b))
            L = max(k[0] for k in C) + 1
            groups = {"block 0": [0], "rest": list(range(1, L))}
            vals = []
            for name, blocks in groups.items():
                ks = [k for k in C if k[0] in blocks]
                vals.append(100 * sum(C[k]["rel_err"] - P[k]["rel_err"] - Q[k]["rel_err"] for k in ks)
                            / sum(P[k]["rel_err"] + Q[k]["rel_err"] for k in ks))
            parts.append(f"{NAME[o]} {vals[0]:+6.1f} / {vals[1]:+6.1f}")
        print(f"  {m:13s} {cell(s, b):7s} block 0 / blocks 1+:  " + " | ".join(parts))
less = {o: 0 for o in ORD}
for m in MODELS:
    for s, b in CELLS:
        P, Q = layer_map(g(m, "prune", s, 16)), layer_map(g(m, "quant", 0, b))
        for o in ORD:
            C = layer_map(g(m, o, s, b))
            f = lambda blocks: sum(C[k]["rel_err"] - P[k]["rel_err"] - Q[k]["rel_err"] for k in C if k[0] in blocks) \
                / sum(P[k]["rel_err"] + Q[k]["rel_err"] for k in C if k[0] in blocks)
            L = max(k[0] for k in C) + 1
            less[o] += f([0]) > f(range(1, L))
print("  cells where block 0 is less sub-additive (or more super-additive) than blocks 1+: "
      + ", ".join(f"{NAME[o]} {less[o]}/18" for o in ORD))
ps = {o: sum(1 for m in MODELS for s, b in CELLS
             if (lambda C, P, Q: sum(C[k]["rel_err"] - P[k]["rel_err"] - Q[k]["rel_err"] for k in C if k[0] == 0))(
                 layer_map(g(m, o, s, b)), layer_map(g(m, "prune", s, 16)), layer_map(g(m, "quant", 0, b))) > 0)
      for o in ORD}
print("  cells where block 0 alone is super-additive: " + ", ".join(f"{NAME[o]} {ps[o]}/18" for o in ORD))


# ============================================================================= E
hr("E. model-level additivity in perplexity under two nulls")
cnt = defaultdict(int)
for m in MODELS:
    d = g(m, "dense", 0, 16)["ppl"]
    for s, b in CELLS:
        p, q = g(m, "prune", s, 16)["ppl"], g(m, "quant", 0, b)["ppl"]
        add, mult = p + q - d, p * q / d
        row = []
        for o in ORD:
            c = g(m, o, s, b)["ppl"]
            fam = "mask-first" if o != "q_then_p" else "Q->P"
            cnt[(fam, "add")] += c > add
            cnt[(fam, "mult")] += c > mult
            row.append(f"{NAME[o]} {100 * (c - add) / add:+6.1f}% / {100 * (c - mult) / mult:+6.1f}%")
        print(f"  {m:13s} {cell(s, b):7s} vs additive ppl {add:7.2f} / multiplicative ppl {mult:7.2f}:  " + " | ".join(row))
print(f"  worse than the null: mask-first {cnt[('mask-first', 'add')]}/36 (additive), "
      f"{cnt[('mask-first', 'mult')]}/36 (multiplicative); Q->P {cnt[('Q->P', 'add')]}/18, {cnt[('Q->P', 'mult')]}/18")


# ============================================================================= F
hr("F. weaker-first rule (Kim et al., ICLR 2026) against Table 1")
agree = 0
for m in MODELS:
    d = g(m, "dense", 0, 16)["ppl"]
    for s, b in CELLS:
        dp, dq = g(m, "prune", s, 16)["ppl"] - d, g(m, "quant", 0, b)["ppl"] - d
        pred = "P first" if dp < dq else "Q first"
        pq, qp = g(m, "p_then_q", s, b)["ppl"], g(m, "q_then_p", s, b)["ppl"]
        obs = "P first" if pq < qp else "Q first"
        agree += pred == obs
        flag = "" if pred == obs else "   <- rule fails"
        print(f"  {m:13s} {cell(s, b):7s} dPPL prune {dp:6.2f}, quant {dq:6.2f} -> predicts {pred};"
              f" P->Q {pq:6.2f}, Q->P {qp:6.2f}{flag}")
print(f"  rule holds in {agree}/18 cells")


# ============================================================================= G
hr("G. calibration-seed noise (SmolLM2-135M, seeds 0-2) and the gaps it has to explain")
for s in (0.25, 0.5):
    vals = {o: [g("SmolLM2-135M", o, s, 4, sd)["ppl"] for sd in (0, 1, 2)] for o in ORD}
    vals["prune"] = [g("SmolLM2-135M", "prune", s, 16, sd)["ppl"] for sd in (0, 1, 2)]
    vals["quant"] = [g("SmolLM2-135M", "quant", 0, 4, sd)["ppl"] for sd in (0, 1, 2)]
    for o, v in vals.items():
        print(f"  {cell(s, 4)} {o:9s} " + " ".join(f"{x:7.3f}" for x in v)
              + f"  mean {st.mean(v):6.2f} sd {st.stdev(v):.2f} ({100 * st.stdev(v) / st.mean(v):.1f}%)")
    for a, b in (("joint", "p_then_q"), ("joint", "q_then_p"), ("p_then_q", "q_then_p")):
        dd = [x - y for x, y in zip(vals[a], vals[b])]
        t = st.mean(dd) / (st.stdev(dd) / math.sqrt(3))
        print(f"     paired {NAME[a]}-{NAME[b]}: " + " ".join(f"{x:+.2f}" for x in dd)
              + f"  mean {st.mean(dd):+.2f}  t = {t:+.2f} (df 2, two-sided p = {p_t_df2(t):.3f})")
    for o in ORD:
        print(f"     layer interaction {NAME[o]:5s} by seed: "
              + " ".join(f"{interaction('SmolLM2-135M', o, s, 4, sd):+.1f}%" for sd in (0, 1, 2)))
print("  joint vs P->Q on the models without seeds (perplexity gap, P->Q over joint):")
for m in ("SmolLM2-360M", "Qwen2.5-0.5B"):
    print(f"  {m:13s} " + "  ".join(
        f"{cell(s, b)} {100 * (g(m, 'p_then_q', s, b)['ppl'] / g(m, 'joint', s, b)['ppl'] - 1):+.1f}%" for s, b in CELLS))


# ============================================================================= H
hr("H. task scores: noise, chance floor, I_Sr interval")
CH = {"arc_easy": 0.25, "piqa": 0.5, "hellaswag": 0.25}
NITEM = 500


def sr(m, r, norm=False):
    D = g(m, "dense", 0, 16)["main"]
    if norm:
        return sum(r["main"][t] - CH[t] for t in CH) / sum(D[t] - CH[t] for t in CH)
    return sum(r["main"].values()) / sum(D.values())


def sr_var(m, r):
    D = sum(g(m, "dense", 0, 16)["main"].values())
    return sum(p * (1 - p) / NITEM for p in r["main"].values()) / D ** 2


for m in MODELS:
    D = g(m, "dense", 0, 16)["main"]
    se = {t: math.sqrt(D[t] * (1 - D[t]) / NITEM) for t in CH}
    print(f"  {m}: dense " + " ".join(f"{t} {D[t]:.3f} (SE {se[t]:.3f})" for t in CH)
          + f" | chance floor of Sr {sum(CH.values()) / sum(D.values()):.3f}")
    for s, b in [(0.25, 4), (1 / 3, 3)]:
        P, Q, C = g(m, "prune", s, 16), g(m, "quant", 0, b), g(m, "p_then_q", s, b)
        i1 = sr(m, C) - sr(m, P) * sr(m, Q)
        i2 = sr(m, C, True) - sr(m, P, True) * sr(m, Q, True)
        se_i = math.sqrt(sr_var(m, C) + sr(m, Q) ** 2 * sr_var(m, P) + sr(m, P) ** 2 * sr_var(m, Q))
        print(f"     I_Sr {cell(s, b)}: {i1:+.3f} (unpaired SE ~{se_i:.3f}; chance-normalized {i2:+.3f})")
p25 = g("SmolLM2-135M", "prune", 0.25, 16)["main"]["hellaswag"]
print(f"  example: 25% pruning moves SmolLM2-135M HellaSwag from {g('SmolLM2-135M', 'dense', 0, 16)['main']['hellaswag']:.3f} to {p25:.3f}")


# ============================================================================= I
hr("I. storage with the mask counted")


def h2(p):
    return 0.0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def bpw(s, b, mask):
    v = (16 if b >= 16 else b) * (1 - s)
    if s <= 0 or mask == "none":
        return v
    return v + (1.0 if mask == "bitmap" else h2(s))


print("  bits per weight, excluding the per-group scale/zero overhead (about 0.15 bit for every 3/4-bit format):")
for s, b in [(0, 8), (0, 4), (0, 3)] + CELLS:
    print(f"    {cell(s, b):7s} TCr {1 - b / 16 * (1 - s):.3f} -> {bpw(s, b, 'none'):.2f} | bitmap {bpw(s, b, 'bitmap'):.2f}"
          f" | entropy bound {bpw(s, b, 'entropy'):.2f}")
print("  equal-storage comparisons (bitmap accounting, WikiText-2 perplexity, best order):")
best = lambda m, s, b: min((g(m, o, s, b)["ppl"], NAME[o]) for o in ORD)
for m in MODELS:
    q4, q3 = g(m, "quant", 0, 4)["ppl"], g(m, "quant", 0, 3)["ppl"]
    a, c, e, f = best(m, 0.25, 4), best(m, 1 / 3, 3), best(m, 0.5, 4), best(m, 0.25, 3)
    print(f"  {m:13s} 4.00 b/w: 4-bit {q4:.2f} vs 25%+4 {a[0]:.2f} ({a[1]}) | 3.00 b/w: 3-bit {q3:.2f}"
          f" vs 50%+4 {e[0]:.2f} ({e[1]}) vs 33%+3 {c[0]:.2f} ({c[1]}) | 3.25 b/w: 25%+3 {f[0]:.2f} ({f[1]})")
if args.csv:
    with open(args.csv, "w") as fh:
        fh.write("model,config,sparsity,bits,order,ppl,bits_tcr,bits_bitmap,bits_entropy,precision\n")
        for m in MODELS:
            for s, b in [(0, 8), (0, 4), (0, 3)]:
                r = g(m, "quant", 0, b)
                fh.write(f"{m},{b}-bit,0,{b},quant,{r['ppl']:.6f},{bpw(0, b, 'none'):.4f},{bpw(0, b, 'bitmap'):.4f},"
                         f"{bpw(0, b, 'entropy'):.4f},{r['precision']}\n")
            for s, b in CELLS:
                p, o = min((g(m, o, s, b)["ppl"], o) for o in ORD)
                fh.write(f"{m},{cell(s, b)},{s:.4f},{b},{o},{p:.6f},{bpw(s, b, 'none'):.4f},{bpw(s, b, 'bitmap'):.4f},"
                         f"{bpw(s, b, 'entropy'):.4f},{g(m, o, s, b)['precision']}\n")
    print(f"  wrote {args.csv}")


# ============================================================================= J
hr("J. exact zeros and the Q->P = 3-bit identity")
for m in MODELS:
    z = lambda r: st.mean(d["zero_frac"] for d in r["diags"])
    print(f"  {m:13s} zeros after quantization alone: 4-bit {z(g(m, 'quant', 0, 4)):.3f}, 3-bit {z(g(m, 'quant', 0, 3)):.3f}"
          f" | 25%+3: " + ", ".join(f"{NAME[o]} {z(g(m, o, 0.25, 3)):.3f}" for o in ORD))
    a, q = g(m, "q_then_p", 0.25, 3), g(m, "quant", 0, 3)
    same = all(x["rel_err"] == y["rel_err"] for x, y in zip(a["diags"], q["diags"])) and a["ppl"] == q["ppl"]
    short = {round(s * 100): sum(d["zero_frac"] < s for d in q["diags"]) for s in (0.25, 1 / 3, 0.5)}
    print(f"     Q->P 25%+3 identical to 3-bit alone (every layer error and the perplexity): {same};"
          f" layers with fewer 3-bit zeros than 25/33/50%: {short[25]}/{short[33]}/{short[50]} of {len(q['diags'])}")


# ============================================================================= K
hr("K. what each result line records")
keys = sorted(set().union(*[set(r) for r in full.values()]))
print(f"  keys: {keys}")
print("  missing: GPU name or UUID, driver/CUDA, torch/transformers/lm-eval versions, compute precision,"
      " per-item task results, safeguard warnings")
seed_prune = [k for k in full if k[1] == "prune" and k[4] > 0]
print(f"  seed-1/2 pruning-only runs are stored with bits = {sorted({k[3] for k in seed_prune})} (should be 16)")

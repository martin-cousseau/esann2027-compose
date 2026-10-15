"""Reproduce every number, table and figure of the paper from the raw result files.
Usage:  python analysis/reproduce_paper.py      (run from the repository root)
Outputs: results/merged_results.json, analysis/table1_rows.tex, analysis/fig1_interaction.pdf, and a printed report."""
import json, math, statistics as st, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"; A = "analysis"
K = lambda r: (r["model"].split("/")[1], r["order"], round(r["sparsity"], 4), r["bits"], r.get("seed", 0))
full = {K(r): r for r in map(json.loads, open(f"{R}/results_full.jsonl")) }
fix = {K(r): r for r in map(json.loads, open(f"{R}/results_fix.jsonl")) }

# 1. merge: the six SmolLM2-360M runs that overflowed in fp16 are replaced by their fp32 re-runs
AFFECTED = [("SmolLM2-360M", "joint", 0.25, 3, 0), ("SmolLM2-360M", "p_then_q", 0.3333, 4, 0),
            ("SmolLM2-360M", "q_then_p", 0.3333, 3, 0), ("SmolLM2-360M", "joint", 0.3333, 3, 0),
            ("SmolLM2-360M", "joint", 0.5, 3, 0), ("SmolLM2-360M", "p_then_q", 0.5, 3, 0)]
M = {k: dict(v, precision="fp16") for k, v in full.items()}
for k in AFFECTED: M[k] = dict(fix[k], precision="fp32")
assert all(math.isfinite(r["ppl"]) and math.isfinite(r["mean_rel_err"]) for r in M.values())
json.dump({"|".join(map(str, k)): v for k, v in M.items()}, open(f"{R}/merged_results.json", "w"))
print(f"runs: {len(full)} (fp16) + {len(fix)} (fp32 fix) -> {len(M)} merged")
print("fp32 controls (25%+4-bit):")
for o in ["joint", "p_then_q", "q_then_p"]:
    a, b = full[("SmolLM2-360M", o, 0.25, 4, 0)]["ppl"], fix[("SmolLM2-360M", o, 0.25, 4, 0)]["ppl"]
    print(f"  {o:9s} fp16 {a:.3f}  fp32 {b:.3f}  diff {100*(b-a)/a:+.2f}%")

g = lambda m, o, s, b, sd=0: M[(m, o, round(float(s), 4), b, sd)]
MODELS = ["SmolLM2-135M", "SmolLM2-360M", "Qwen2.5-0.5B"]
CELLS = [(0.25, 4), (1/3, 4), (0.5, 4), (0.25, 3), (1/3, 3), (0.5, 3)]
ORD = ["joint", "p_then_q", "q_then_p"]
tcr = lambda s, b: 1 - b / 16 * (1 - s)

# 2. Table 1 (perplexity) + order statistics
rows, wins, worst = [], {o: 0 for o in ORD}, {o: 0 for o in ORD}
print("\nTable 1 / Q2")
for m in MODELS:
    d = g(m, "dense", 0, 16)["ppl"]
    for i, (s, b) in enumerate(CELLS):
        v = {o: g(m, o, s, b)["ppl"] for o in ORD}
        best, w = min(v, key=v.get), max(v, key=v.get); wins[best] += 1; worst[w] += 1
        f = lambda o: ((r"\textbf{%.2f}" if o == best else "%.2f") % v[o]) + (r"$^{*}$" if g(m, o, s, b)["precision"] == "fp32" else "")
        name = {0: m, 1: f"(dense {d:.2f})"}.get(i, "")
        rows.append(f"{name} & {round(s*100):d}\\% + {b} & {tcr(s,b):.3f} & {g(m,'prune',s,16)['ppl']:.2f} & {g(m,'quant',0,b)['ppl']:.2f} & {f('joint')} & {f('p_then_q')} & {f('q_then_p')} \\\\")
        print(f"  {m:13s} {s:.2f}+{b}  " + "  ".join(f"{o}={v[o]:.2f}" for o in ORD) + f"  spread {(max(v.values())-min(v.values()))/min(v.values())*100:.1f}%  worst/best {max(v.values())/min(v.values()):.2f}x")
    rows.append(r"\midrule")
open(f"{A}/table1_rows.tex", "w").write("\n".join(rows[:-1]) + "\n")
print("  best order counts:", wins, "| worst order counts:", worst)

# 3. Q1: layer interaction + Fig. 1
print("\nQ1: I_E/(E_P+E_Q) (%)")
I = {}
for m in MODELS:
    for s, b in CELLS:
        ep, eq = g(m, "prune", s, 16)["mean_rel_err"], g(m, "quant", 0, b)["mean_rel_err"]
        for o in ORD: I[(m, s, b, o)] = 100 * (g(m, o, s, b)["mean_rel_err"] - ep - eq) / (ep + eq)
        print(f"  {m:13s} {s:.2f}+{b}  " + "  ".join(f"{o}={I[(m,s,b,o)]:+.1f}" for o in ORD))
mf = [I[(m, s, b, o)] for m in MODELS for s, b in CELLS for o in ("joint", "p_then_q")]
q4 = [I[(m, s, b, "q_then_p")] for m in MODELS for s, b in CELLS if b == 4]
print(f"  mask-first sub-additive: {sum(x < 0 for x in mf)}/{len(mf)} | Q->P super-additive at 4-bit: {sum(x > 0 for x in q4)}/{len(q4)}")
for m in MODELS:
    print(f"  {m}: zeros created by rounding  " + "  ".join(f"{b}-bit {st.mean(d['zero_frac'] for d in g(m,'quant',0,b)['diags']):.3f}" for b in (8, 4, 3))
          + f" | Q->P 25%+3 ppl {g(m,'q_then_p',0.25,3)['ppl']:.4f} vs 3-bit alone {g(m,'quant',0,3)['ppl']:.4f}")

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "hatch.linewidth": 0.5})
fig, axs = plt.subplots(1, 3, figsize=(4.75, 2.0), sharey=True)
x = np.arange(len(CELLS)); wd = 0.27
STY = [("joint", "Joint", "0.75", ""), ("p_then_q", "P$\\rightarrow$Q", "0.45", ""), ("q_then_p", "Q$\\rightarrow$P", "white", "////")]
for ax, m in zip(axs, MODELS):
    for i, (o, lab, col, h) in enumerate(STY):
        ax.bar(x + (i - 1) * wd, [I[(m, s, b, o)] for s, b in CELLS], wd, color=col, edgecolor="black", linewidth=0.4, hatch=h, label=lab)
    ax.axhline(0, color="black", lw=0.6); ax.axvline(2.5, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(["25/4", "33/4", "50/4", "25/3", "33/3", "50/3"], rotation=90, fontsize=6.5)
    ax.set_title(m, fontsize=7.5, pad=2); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel(r"$I_E/(E_P+E_Q)$ (%)"); axs[0].set_ylim(-26, 10)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4)
fig.tight_layout(pad=0.2, w_pad=0.3, rect=(0, 0, 1, 0.9)); fig.savefig(f"{A}/fig1_interaction.pdf")

# 4. seeds (SmolLM2-135M)
print("\nSeeds, SmolLM2-135M (mean ± sd over 3 calibration seeds)")
for s in (0.25, 0.5):
    for o in ORD:
        v = [g("SmolLM2-135M", o, s, 4, sd)["ppl"] for sd in (0, 1, 2)]
        print(f"  {s:.2f}+4 {o:9s} {st.mean(v):.2f} ± {st.stdev(v):.2f}")

# 5. Q3: model-level interaction (tasks), small models; 7-8B from the published IJCNN 2025 tables
print("\nQ3: I_Sr = Sr_PQ - Sr_P*Sr_Q (prune-then-quantize, tasks)")
for m in MODELS:
    D = sum(g(m, "dense", 0, 16)["main"].values()); out = []
    for s, b in [(0.25, 4), (1/3, 3)]:
        sp = sum(g(m, "prune", s, 16)["main"].values()) / D; sq = sum(g(m, "quant", 0, b)["main"].values()) / D
        out.append(f"{s:.2f}+{b}: {sum(g(m,'p_then_q',s,b)['main'].values())/D - sp*sq:+.3f}")
    print(f"  {m:13s} " + "  ".join(out))
PUB = {"Mistral-7B-v0.3": dict(dense=(30.3, 58.0, 12.8), p25=(29.4, 55.7, 10.9), p33=(27.9, 53.9, 10.0), q4=(28.3, 55.6, 10.5), q3=(23.3, 43.0, 6.4), j25_4=(26.9, 53.5, 10.1), j33_3=(20.2, 40.0, 5.3)),
       "LLaMA-3.1-8B": dict(dense=(35.4, 62.3, 17.7), p25=(33.3, 60.2, 14.9), p33=(30.2, 55.2, 11.5), q4=(32.0, 53.3, 0.2), q3=(21.3, 43.8, 4.2), j25_4=(30.5, 53.4, 10.5), j33_3=(20.3, 41.5, 0.8))}
for m, v in PUB.items():
    for idx, lab in [((0, 1, 2), "all"), ((0, 1), "no MATH")]:
        S = lambda k: sum(v[k][i] for i in idx) / sum(v["dense"][i] for i in idx)
        print(f"  {m:15s} ({lab:7s}) 25%+4: {S('j25_4')-S('p25')*S('q4'):+.3f}  33%+3: {S('j33_3')-S('p33')*S('q3'):+.3f}")

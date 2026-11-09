"""Synthetic-layer checks of the paper's decomposition (Sec. 3) with the paper's own obs_sweep.

Usage (CPU, a few minutes):
    python synthetic_layers.py <path to esann2027-compose>

1. The decomposition identity. Without compensation and with one fixed grid,
       I_E = E_PQ - E_P - E_Q = -||R_M||_H^2 + 2<P, R_keep>_H - 2<R_M, R_keep>_H
   holds exactly (up to the common normalization) for any H. Checked numerically.
2. What happens with compensation and correlated inputs. Synthetic 512x512 layers with Laplace weights
   and four input regimes, from independent features (H nearly diagonal: the decomposition's guarantee
   applies) to strongly correlated features with a few outlier channels (closer to LLM activations).
   Every arm of ../ablation/arms.py is run on the same H; reported: I_E/(E_P+E_Q) per arm (mean and range
   over seeds) and how good the mask chosen by Q->P is for pruning alone.
These layers are crude stand-ins for real ones: they show which way each effect can go, not its size
in SmolLM2 or Qwen. The real-model answer is ../ablation/ablation_same_hessian.py.
"""
import importlib.util
import math
import os
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ablation"))
import arms as A  # noqa: E402

spec = importlib.util.spec_from_file_location("esann_core", os.path.join(sys.argv[1], "src", "core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
torch.set_num_threads(max(1, os.cpu_count() or 1))

REGIMES = {  # name: (mean offset, outlier scale, spectrum exponent, rotate)
    "independent features": (0.0, 1.0, 0.0, False),
    "weak correlation": (0.0, 1.0, 0.3, True),
    "moderate (+ outlier channels x5)": (0.0, 5.0, 0.6, True),
    "strong (+ mean offset, outliers x15)": (0.3, 15.0, 0.6, True),
}


def make_layer(seed, mean_off, out_scale, alpha, rotate, d=512, n=8192):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g) * torch.arange(1, d + 1, dtype=torch.float32) ** -alpha
    if rotate:
        U, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        X = X @ U.T
    else:
        X = X * (torch.rand(d, generator=g) + 0.5)
    X += mean_off * torch.randn(1, d, generator=g)
    outliers = torch.randperm(d, generator=g)[:6]
    X[:, outliers] *= out_scale
    H = 2.0 / n * X.T @ X
    lap = torch.distributions.Laplace(0.0, 1.0)
    torch.manual_seed(seed)
    W = lap.sample((d, d)) / math.sqrt(2 * d) * torch.exp(0.3 * torch.randn(d, 1, generator=g))
    D = torch.diag(H).sqrt()
    C = H / D[:, None] / D[None, :]
    offdiag = (C - torch.diag(torch.diag(C))).abs().mean().item()
    return W, H, offdiag


# ----------------------------------------------------------------------------- 1. identity
print("1. decomposition identity (no compensation, one fixed grid, correlated H)")
W, H, _ = make_layer(0, 0.3, 15.0, 0.6, True, d=256, n=2048)
W = W[:64]
M = torch.rand(W.shape, generator=torch.Generator().manual_seed(3)) < 0.3
qW = torch.empty_like(W)
for c0 in range(0, W.shape[1], 128):
    scale, zero, maxq = core.find_params(W[:, c0:c0 + 128], 4)
    qW[:, c0:c0 + 128] = core.quantize(W[:, c0:c0 + 128], scale.unsqueeze(1), zero.unsqueeze(1), maxq)
P, R = W * M, W - qW
RM, Rk = R * M, R * (~M)
ip = lambda X, Y: torch.trace(X @ H @ Y.T).item()
lhs = ip(P + Rk, P + Rk) - ip(P, P) - ip(R, R)
absorption, cross1, cross2 = -ip(RM, RM), 2 * ip(P, Rk), -2 * ip(RM, Rk)
print(f"   I_E = {lhs:.6e}; absorption + cross terms = {absorption + cross1 + cross2:.6e}"
      f" (absorption {absorption:.3e}, 2<P,R_keep> {cross1:+.3e}, -2<R_M,R_keep> {cross2:+.3e})")
assert abs(lhs - (absorption + cross1 + cross2)) < 1e-6 * (ip(P, P) + ip(R, R))

# ----------------------------------------------------------------------------- 2. arms by regime
CELLS = [(0.25, 4), (0.5, 4), (0.25, 3), (0.5, 3)]
SEEDS = range(4)
print("\n2. I_E/(E_P+E_Q) in % on synthetic 512x512 layers, mean [min, max] over 4 seeds; all arms share one H")
for name, (mo, osc, al, rot) in REGIMES.items():
    res = {c: {a: [] for a in A.ALL_ARMS + ("mask_quality_q_then_p",)} for c in CELLS}
    ods = []
    for seed in SEEDS:
        W, H, od = make_layer(seed, mo, osc, al, rot)
        ods.append(od)
        out = A.layer_ablation(core, W, H, CELLS)
        for s, b in CELLS:
            c = out["cells"][A.key_cell(s, b)]
            eP, eQ = out["E_P"][A.key_s(s)], out["E_Q"][str(b)]
            for a in A.ALL_ARMS:
                res[(s, b)][a].append(100 * (c[a] - eP - eQ) / (eP + eQ))
            res[(s, b)]["mask_quality_q_then_p"].append(c["mask_quality_q_then_p"])
    print(f"\n  {name} (mean |off-diagonal correlation| of H = {st.mean(ods):.2f})")
    for s, b in CELLS:
        r = res[(s, b)]
        print(f"    {round(s * 100)}%+{b}: " + " | ".join(
            f"{a} {st.mean(r[a]):+5.1f} [{min(r[a]):+.1f},{max(r[a]):+.1f}]" for a in A.ALL_ARMS)
              + f" | Q->P mask, pruning alone: {st.mean(r['mask_quality_q_then_p']):.2f}x SparseGPT's error")

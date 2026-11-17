"""E9 (layer-level control without compensation): the identities scripts/run_e9.py relies on, on a random layer (CPU)."""
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src_v2 import pipeline as P  # noqa: E402,F401  (puts review_code/ablation on sys.path)
import ablation_same_hessian as ASH  # noqa: E402
import arms as A  # noqa: E402
import run_e9  # noqa: E402

CELLS = [(0.25, 4), (1 / 3, 3), (0.5, 3)]


def _layer(rows=48, cols=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(rows, cols, generator=g)
    X = torch.randn(512, 24, generator=g) @ torch.randn(24, cols, generator=g) + 0.3 * torch.randn(512, cols, generator=g)
    return W, 2 * X.T @ X / 512          # correlated inputs: off-diagonal H


def test_decomposition_and_keys():
    core = ASH.load_core(ROOT)
    W, H = _layer()
    res = run_e9.layer_e9(core, W, H, CELLS)
    for s, b in CELLS:
        c = res["nocomp"]["cells"][A.key_cell(s, b)]
        ep, eq = res["nocomp"]["E_P"][A.key_s(s)], res["nocomp"]["E_Q"][str(b)]
        assert abs(c["decomp_residual"]) <= 1e-9 * (ep + eq)
        assert c["structure_residual"] == 0.0
        assert c["absorption"] <= 0
        assert abs((c["p_then_q_stale"] - ep - eq) - c["I_E_stale_f64"]) <= 1e-4 * (ep + eq)
        for k in ("p_then_q", "q_then_p", "p_then_q_stale", "zeros_p_then_q", "zeros_q_then_p", "cross_PR", "cross_RR",
                  "mask_jaccard_q_then_p_vs_P", "overlap_mask_zeros", "share_zeros_pruned", "zeros"):
            assert k in c, k
        assert 0 <= c["overlap_mask_zeros"] <= 1 and 0 <= res["comp"]["cells"][A.key_cell(s, b)]["overlap_mask_zeros"] <= 1
    # compensation lowers the single-method errors
    assert all(res["comp"]["E_P"][k] <= res["nocomp"]["E_P"][k] + 1e-12 for k in res["comp"]["E_P"])


def test_diagonal_hessian_is_sub_additive():
    core = ASH.load_core(ROOT)
    W, H = _layer(seed=1)
    Hd = torch.diag(torch.diag(H))
    res = run_e9.layer_e9(core, W, Hd, CELLS)
    for s, b in CELLS:
        c = res["nocomp"]["cells"][A.key_cell(s, b)]
        assert abs(c["cross_PR"]) <= 1e-12 and abs(c["cross_RR"]) <= 1e-12
        assert c["I_E_stale_f64"] <= 0


def test_arms_that_coincide_without_compensation():
    core = ASH.load_core(ROOT)
    W, H = _layer(seed=2)
    Hd = torch.diag(torch.diag(H))
    nocomp = types.SimpleNamespace(obs_sweep=lambda W_in, _H, **kw: core.obs_sweep(W_in, Hd, **kw),
                                   rel_recon_error=core.rel_recon_error)
    for s, b in CELLS:
        Pz, Qz = A.single_sweeps(nocomp, W, H, [s], [b])
        j, _ = A.run_arm(nocomp, "joint", W, H, s, b, Pz, Qz)
        pq, _ = A.run_arm(nocomp, "p_then_q", W, H, s, b, Pz, Qz)
        qpm, _ = A.run_arm(nocomp, "q_then_p_pmask", W, H, s, b, Pz, Qz)
        pqs, _ = A.run_arm(nocomp, "p_then_q_stale", W, H, s, b, Pz, Qz)
        assert torch.equal(j, pq)
        assert torch.equal(qpm, pqs)


def test_analysis_sums(tmp_path):
    import json
    from scripts.analysis import e9 as E9
    core = ASH.load_core(ROOT)
    d = tmp_path / "results" / "v2" / "E9"
    d.mkdir(parents=True)
    with open(d / "Fake.jsonl", "w") as f:
        for blk in range(2):
            W, H = _layer(seed=10 + blk)
            res = run_e9.layer_e9(core, W, H, CELLS)
            f.write(json.dumps(dict(model="org/Fake", block=blk, name="mlp.down_proj", shape=list(W.shape), seconds=0, **res)) + "\n")
        f.write('{"model": "org/Fake", "block": 9, "na')   # a line cut by a lost runtime is skipped
    paths = types.SimpleNamespace(results_v2=str(tmp_path / "results" / "v2"), reports=str(tmp_path))
    out = E9.analyze(paths)
    assert out["layers"] == {"Fake": 2} and len(out["cells"]) == len(CELLS)
    for c in out["cells"]:
        assert c["n_layers"] == 2
        assert abs(c["ratio_p_then_q_stale_pct"] - (c["absorption_pct"] + c["cross_pct"])) < 1e-2
        assert abs(c["gap_pct"] - (c["mask_pct"] + c["grid_pct"])) < 1e-9
        assert 0 <= c["nocomp_overlap_mask_zeros"] <= 1
    E9.write(out, paths)
    assert (tmp_path / "e9_cells.csv").exists()

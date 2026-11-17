"""Compression arms for the order and mechanism ablation, built only from the paper's obs_sweep.

Every arm starts from the same dense weights W and the same Hessian H. Arms differ only in the order
of the two operations, in which mask is used, and in which quantization grid is used:

    arm                    first sweep          second sweep                   mask chosen on   grid fitted on
    prune      (E_P)       prune W              -                              W                -
    quant      (E_Q)       quantize W           -                              -                W
    joint                  prune + quantize W   -                              W (per block)    kept weights
    p_then_q               prune W -> W_P       quantize W_P, mask M_P fixed   W                kept weights
    q_then_p               quantize W -> W_Q    prune W_Q, grid G_Q fixed      W_Q              W (stale)
    q_then_p_refit         quantize W -> W_Q    prune W_Q, grid re-fitted      W_Q              kept weights
    q_then_p_pmask         quantize W -> W_Q    mask M_P, grid G_Q fixed       W (M_P)          W (stale)
    q_then_p_pmask_refit   quantize W -> W_Q    mask M_P, grid re-fitted       W (M_P)          kept weights
    p_then_q_stale         prune W -> W_P       quantize W_P, grid G_Q fixed   W                W (stale)

joint, p_then_q and q_then_p are exactly the paper's compress_weight orders (check_against_core
asserts it). Reading the ablation: if q_then_p_refit closes most of the gap between q_then_p and
p_then_q, the stale grid explains the quantize-first penalty (the paper's reading, Sec. 3); if
q_then_p_pmask[_refit] closes it, mask selection on rounded weights does (Harma et al.'s mechanism).
p_then_q_stale is the converse check: a stale grid imposed on the prune-first order.
"""
import types
import torch

PAPER_ARMS = ("joint", "p_then_q", "q_then_p")
ABLATION_ARMS = ("q_then_p_refit", "q_then_p_pmask", "q_then_p_pmask_refit", "p_then_q_stale")
ALL_ARMS = PAPER_ARMS + ABLATION_ARMS


def as_core(obj):
    """Accept the paper's core module, or any namespace (e.g. notebook globals) holding obs_sweep."""
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: obj[k] for k in ("obs_sweep", "rel_recon_error", "compress_weight") if k in obj})
    return obj


def single_sweeps(core, W, H, sparsities, bits_list, groupsize=128):
    """Pruning-only sweeps (one per sparsity) and quantization-only sweeps (one per bit width).

    Returns P = {s: (W_P, M_P)} and Q = {b: (W_Q, G_Q)}; every arm re-uses these.
    """
    core = as_core(core)
    P, Q = {}, {}
    for s in sparsities:
        Wp, mP, _ = core.obs_sweep(W, H, sparsity=s, bits=16, groupsize=groupsize)
        P[s] = (Wp, mP)
    for b in bits_list:
        Wq, _, gQ = core.obs_sweep(W, H, sparsity=0.0, bits=b, groupsize=groupsize)
        Q[b] = (Wq, gQ)
    return P, Q


def run_arm(core, arm, W, H, s, b, P, Q, groupsize=128):
    """Return (W_hat, mask) for one arm at sparsity s and bit width b."""
    core = as_core(core)
    Wp, mP = P[s]
    Wq, gQ = Q[b]

    def sweep(W_in, **kw):
        return core.obs_sweep(W_in, H, groupsize=groupsize, **kw)

    if arm == "joint":
        Wh, m, _ = sweep(W, sparsity=s, bits=b)
    elif arm == "p_then_q":
        Wh, m, _ = sweep(Wp, bits=b, fixed_mask=mP)
    elif arm == "q_then_p":
        Wh, m, _ = sweep(Wq, sparsity=s, bits=b, fixed_grid=gQ)
    elif arm == "q_then_p_refit":
        Wh, m, _ = sweep(Wq, sparsity=s, bits=b)
    elif arm == "q_then_p_pmask":
        Wh, m, _ = sweep(Wq, bits=b, fixed_mask=mP, fixed_grid=gQ)
    elif arm == "q_then_p_pmask_refit":
        Wh, m, _ = sweep(Wq, bits=b, fixed_mask=mP)
    elif arm == "p_then_q_stale":
        Wh, m, _ = sweep(Wp, bits=b, fixed_mask=mP, fixed_grid=gQ)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return Wh, m


@torch.no_grad()
def layer_ablation(core, W, H, cells, arms=ALL_ARMS, groupsize=128, mask_quality=True):
    """Every arm for one linear layer and several (sparsity, bits) cells, all on one shared H.

    Returns a JSON-ready dict:
      E_P[s], E_Q[b]                     normalized layer errors of the single methods
      zeros_Q[b]                         share of weights that quantization alone sets exactly to zero
      cells["s/b"][arm]                  normalized layer error of each arm
      cells["s/b"]["mask_quality_q_then_p"]  pruning-only error with the mask Q->P selects, divided by
                                         E_P with SparseGPT's own mask (1 = as good, >1 = worse)
      cells["s/b"]["mask_jaccard_q_then_p_vs_P"]  overlap of the Q->P mask with SparseGPT's mask
    """
    core = as_core(core)
    W = W.float()
    sparsities = sorted({s for s, _ in cells})
    bits_list = sorted({b for _, b in cells})
    P, Q = single_sweeps(core, W, H, sparsities, bits_list, groupsize)
    E = lambda Wh: core.rel_recon_error(W, Wh, H)
    out = {"E_P": {key_s(s): E(P[s][0]) for s in sparsities},
           "E_Q": {str(b): E(Q[b][0]) for b in bits_list},
           "zeros_Q": {str(b): (Q[b][0] == 0).float().mean().item() for b in bits_list},
           "cells": {}}
    for s, b in cells:
        c, mQP = {}, None
        for arm in arms:
            Wh, m = run_arm(core, arm, W, H, s, b, P, Q, groupsize)
            c[arm] = E(Wh)
            if arm == "q_then_p":
                mQP = m
                c["zeros_q_then_p"] = (Wh == 0).float().mean().item()
        if mask_quality and mQP is not None:
            mP = P[s][1]
            Wm, _, _ = core.obs_sweep(W, H, sparsity=0.0, bits=16, fixed_mask=mQP, groupsize=groupsize)
            c["mask_quality_q_then_p"] = E(Wm) / max(out["E_P"][key_s(s)], 1e-30)
            inter, union = (mQP & mP).sum().item(), (mQP | mP).sum().item()
            c["mask_jaccard_q_then_p_vs_P"] = inter / max(union, 1)
        out["cells"][key_cell(s, b)] = c
    return out


def key_s(s):
    return f"{s:.4f}"


def key_cell(s, b):
    return f"{s:.4f}/{b}"


@torch.no_grad()
def check_against_core(core, W, H, s, b, groupsize=128):
    """Max |difference| between run_arm and the paper's compress_weight for the three paper orders."""
    core = as_core(core)
    P, Q = single_sweeps(core, W, H, [s], [b], groupsize)
    diffs = {}
    for arm in PAPER_ARMS:
        mine, _ = run_arm(core, arm, W, H, s, b, P, Q, groupsize)
        ref, _ = core.compress_weight(W, H, s, b, arm, groupsize)
        diffs[arm] = (mine.float() - ref.float()).abs().max().item()
    return diffs

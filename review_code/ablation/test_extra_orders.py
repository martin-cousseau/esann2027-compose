"""CPU checks for extra_orders.py on a tiny random Llama model (no download).
Usage: python test_extra_orders.py --repo <path to esann2027-compose>"""
import argparse
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extra_orders as X  # noqa: E402
from ablation_same_hessian import load_core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    core = load_core(args.repo)
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=320, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    dense = LlamaForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(1)
    calib = [torch.randint(0, cfg.vocab_size, (1, 128), generator=g) for _ in range(8)]
    text_ids = torch.randint(0, cfg.vocab_size, (1, 128 * 4), generator=g)

    # 1. the paper's orders are unchanged after installing; the new orders run end to end
    ref = copy.deepcopy(dense)
    d_ref = core.compress_model(ref, calib, 0.5, 4, "q_then_p", "cpu", verbose=False)
    X.install_extra_orders(core.__dict__)
    X.install_extra_orders(core.__dict__)  # idempotent
    again = copy.deepcopy(dense)
    d_again = core.compress_model(again, calib, 0.5, 4, "q_then_p", "cpu", verbose=False)
    same = all(a["rel_err"] == b["rel_err"] for a, b in zip(d_ref, d_again))
    print(f"[1] paper order q_then_p unchanged after install: {same}")
    assert same
    ppl_ref = core.perplexity(ref, text_ids, 128, "cpu")[0]
    for order in X.EXTRA_ORDERS:
        m = copy.deepcopy(dense)
        diags = core.compress_model(m, calib, 0.5, 4, order, "cpu", verbose=False)
        assert all(d["mask_kept"] for d in diags), order
        assert all(abs(d["mask_frac"] - 0.5) < 1e-3 for d in diags), order
        assert all(d["max_levels"] <= 16 for d in diags), order
        ppl = core.perplexity(m, text_ids, 128, "cpu")[0]
        kl = X.kl_to_dense(m, dense, text_ids, 128, "cpu")
        print(f"    {order:22s} mean rel_err {sum(d['rel_err'] for d in diags) / len(diags):.4f}  ppl {ppl:8.2f}"
              f"  KL {kl['kl_nats_per_token']:.4f}  top-1 agreement {kl['top1_agreement']:.3f}")
    kl_ref = X.kl_to_dense(ref, dense, text_ids, 128, "cpu")
    kl_self = X.kl_to_dense(dense, dense, text_ids, 128, "cpu")
    print(f"    q_then_p (paper)       ppl {ppl_ref:8.2f}  KL {kl_ref['kl_nats_per_token']:.4f}; dense vs itself KL "
          f"{kl_self['kl_nats_per_token']:.2e}, agreement {kl_self['top1_agreement']:.3f}")
    assert abs(kl_self["kl_nats_per_token"]) < 1e-6 and kl_self["top1_agreement"] == 1.0

    # 2. strict safeguard raises on a non-finite Hessian, for the paper's orders and the ablation orders,
    #    with the documented call order (install_extra_orders first, then make_safeguard_strict)
    fresh = load_core(args.repo)
    X.install_extra_orders(fresh.__dict__)
    X.make_safeguard_strict(fresh.__dict__)
    H = torch.eye(8)
    H[0, 1] = float("inf")
    for order in ("q_then_p",) + X.EXTRA_ORDERS:
        try:
            fresh.compress_weight(torch.randn(4, 8), H, 0.5, 4, order)
            raise AssertionError(f"no error raised for {order}")
        except FloatingPointError as e:
            msg = str(e)
    print(f"[2] strict safeguard raises for q_then_p and all {len(X.EXTRA_ORDERS)} ablation orders: {msg}")

    # 3. flips and McNemar on a known example
    a = {"t": {i: 1 for i in range(100)}}
    b = {"t": {i: (0 if i < 20 else 1) for i in range(100)}}
    b["t"].update({i: 1 for i in range(5)})
    f, mc = X.flips(a, b), X.mcnemar(a, b)
    print(f"[3] flips {f['pooled']:.2f} (expected 0.15); McNemar right-in-A-only {mc['pooled'][0]}, "
          f"right-in-B-only {mc['pooled'][1]}, p = {mc['pooled'][2]:.2e}")
    assert abs(f["pooled"] - 0.15) < 1e-12 and mc["pooled"][:2] == (15, 0)
    assert abs(mc["pooled"][2] - 2 / 2 ** 15) < 1e-15

    # 4. bits per weight
    rows = [(0.25, 4), (0.5, 4), (1 / 3, 3), (0.0, 3)]
    for s, bb in rows:
        print(f"[4] {round(s * 100):2d}%+{bb}: TCr-equivalent {X.bits_per_weight(s, bb, 'none', groupsize=10**9):.2f}"
              f" | bitmap {X.bits_per_weight(s, bb, 'bitmap', groupsize=10**9):.2f}"
              f" | entropy {X.bits_per_weight(s, bb, 'entropy', groupsize=10**9):.2f}"
              f" | bitmap + group overhead {X.bits_per_weight(s, bb, 'bitmap'):.3f}")
    assert abs(X.bits_per_weight(0.25, 4, "bitmap", groupsize=10**9) - 4.0) < 1e-6
    assert abs(X.bits_per_weight(0.25, 4, "none", groupsize=10**9) - 16 * (1 - core.tcr(0.25, 4))) < 1e-6

    # 5. metadata (CPU: no GPU fields)
    md = X.run_metadata(repo=args.repo, dtype=torch.float32)
    print(f"[5] metadata keys: {sorted(md)}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

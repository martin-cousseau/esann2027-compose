"""Unit tests for the v2 pipeline. Run on CPU in the container and on the GPU runtime:

    python -m pytest -q tests/test_pipeline.py            # CPU (and CUDA tests if a GPU is present)
    python -m pytest -q tests/test_pipeline.py -m network # adds the lm-eval per-item test (downloads)
"""
import copy
import os
import random
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src_v2 import pipeline as P  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny_llama(seed=0, layers=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=320, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    m = LlamaForCausalLM(cfg).float().eval()
    m.config.use_cache = False
    return m


def random_layer(rows=96, cols=256, n=2048, seed=0, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(rows, cols, generator=g)
    X = torch.randn(n, cols, generator=g) @ (torch.eye(cols) + 0.3 * torch.randn(cols, cols, generator=g) / cols ** 0.5)
    H = 2 * X.T @ X / n
    return W.to(dev), H.to(dev)


# ----------------------------------------------------------------------------- arms vs compress_weight
@pytest.mark.parametrize("dev", DEVICES)
def test_arms_reproduce_compress_weight_exactly(dev):
    import arms as A
    core = P.load_core(strict=True)
    W, H = random_layer(dev=dev)
    for s, b in [(0.25, 4), (0.5, 4), (1 / 3, 3), (0.5, 3)]:
        diffs = A.check_against_core(core, W, H, s, b)
        assert max(diffs.values()) == 0.0, (s, b, diffs)


@pytest.mark.parametrize("dev", DEVICES)
def test_extra_orders_installed_and_paper_orders_unchanged(dev):
    core_plain = P.load_core(strict=False, extra_orders=False)
    core = P.load_core(strict=True)
    W, H = random_layer(dev=dev, seed=3)
    for order in ("joint", "p_then_q", "q_then_p", "prune", "quant"):
        a, _ = core_plain.compress_weight(W, H, 0.5, 4, order)
        b, _ = core.compress_weight(W, H, 0.5, 4, order)
        assert torch.equal(a, b), order
    for order in ("q_then_p_refit", "q_then_p_pmask", "q_then_p_pmask_refit", "p_then_q_stale"):
        Wh, m = core.compress_weight(W, H, 0.5, 4, order)
        assert abs(m.float().mean().item() - 0.5) < 1e-3
        assert bool((Wh[m] == 0).all())


# ----------------------------------------------------------------------------- strict safeguard
def test_strict_check_raises_for_every_order():
    core = P.load_core(strict=True)
    H = torch.eye(8)
    H[0, 1] = float("inf")
    orders = ("prune", "quant", "joint", "p_then_q", "q_then_p", "q_then_p_refit", "q_then_p_pmask",
              "q_then_p_pmask_refit", "p_then_q_stale")
    for order in orders:
        with pytest.raises(FloatingPointError):
            core.compress_weight(torch.randn(4, 8), H, 0.5, 4, order)


# ----------------------------------------------------------------------------- block-0 Hessian equality
def test_block0_hessian_equals_compress_model_hessian():
    """The same-Hessian ablation (dense Hessians) and the paper's sequential pipeline see the same H in
    block 0, bit for bit."""
    from ablation_same_hessian import dense_hessians
    core = P.load_core(strict=True)
    model = tiny_llama()
    g = torch.Generator().manual_seed(1)
    calib = [torch.randint(0, 512, (1, 64), generator=g) for _ in range(4)]
    Hd = dense_hessians(core, model, calib, "cpu", blocks=[0])
    seen = {}
    orig = core.compress_weight

    def spy(W, H, sparsity, bits, order, groupsize=128):
        seen.setdefault(len(seen), H.clone())
        return orig(W, H, sparsity, bits, order, groupsize)

    core.compress_weight = spy
    m2 = copy.deepcopy(model)
    core.compress_model(m2, calib, 0.5, 4, "p_then_q", "cpu", verbose=False)
    names = [n for n, m in core.get_layers(model)[0].named_modules() if isinstance(m, nn.Linear)]
    for k, name in enumerate(names):
        assert torch.equal(seen[k], Hd[(0, name)]), name


# ----------------------------------------------------------------------------- perplexity, KL
@pytest.mark.parametrize("dev", DEVICES)
def test_kl_dense_vs_dense_is_zero_and_ppl_matches_core(dev):
    core = P.load_core(strict=True)
    model = tiny_llama().to(dev)
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 512, (1, 64 * 3), generator=g)
    windows = ids[0, :192].reshape(3, 64)
    out = P.lm_metrics(model, windows, dev, dense=model)
    assert out["kl"] == 0.0 and out["top1"] == 1.0
    ppl_core, n = core.perplexity(model, ids, 64, dev)
    assert n == 3 and out["ppl"] == ppl_core


@pytest.mark.parametrize("dev", DEVICES)
def test_kl_positive_after_compression(dev):
    core = P.load_core(strict=True)
    dense = tiny_llama().to(dev)
    m = copy.deepcopy(dense)
    g = torch.Generator().manual_seed(5)
    calib = [torch.randint(0, 512, (1, 64), generator=g) for _ in range(4)]
    diags = core.compress_model(m, calib, 0.5, 3, "q_then_p", dev, verbose=False)
    assert all(d["mask_kept"] for d in diags)
    windows = torch.randint(0, 512, (2, 64), generator=g)
    out = P.lm_metrics(m, windows, dev, dense=dense)
    assert out["kl"] > 0 and 0 <= out["top1"] <= 1


# ----------------------------------------------------------------------------- flips, McNemar
def test_flips_and_mcnemar_known_example():
    X = P.extra()
    a = {"t": {i: 1 for i in range(100)}}
    b = {"t": {i: (0 if i < 20 else 1) for i in range(100)}}
    b["t"].update({i: 1 for i in range(5)})
    f, mc = X.flips(a, b), X.mcnemar(a, b)
    assert abs(f["pooled"] - 0.15) < 1e-12
    assert mc["pooled"][:2] == (15, 0) and abs(mc["pooled"][2] - 2 / 2 ** 15) < 1e-15


def test_binary_items_uses_v2_main_metric():
    items = {"arc_easy": {"doc_id": [0, 1], "acc": [1.0, 0.0], "acc_norm": [0.0, 1.0]},
             "boolq": {"doc_id": [3], "acc": [1.0]}}
    bi = P.binary_items(items)
    assert bi["arc_easy"] == {0: 0, 1: 1} and bi["boolq"] == {3: 1}
    assert P.binary_items(items, "acc")["arc_easy"] == {0: 1, 1: 0}


# ----------------------------------------------------------------------------- storage
def test_bits_per_weight_arithmetic():
    st = P.storage(0.25, 4)
    assert abs(st["tcr"] - 3.0) < 1e-9 and abs(st["bitmap"] - 4.0) < 1e-9
    assert abs(st["entropy_nominal"] - (3.0 + P.h2(0.25))) < 1e-9
    assert abs(P.storage(0.5, 4)["bitmap"] - 3.0) < 1e-9
    assert abs(P.storage(0.0, 3)["bitmap"] - 3.0) < 1e-9
    assert abs(P.storage(0.5, 3)["bitmap"] - 2.5) < 1e-9
    assert abs(P.storage(0.25, 4)["group_overhead"] - 20 / 128) < 1e-12
    # TCr identity: 16 (1 - TCr) = b (1 - s)
    core = P.load_core()
    for s, b in [(0.25, 4), (1 / 3, 3), (0.5, 3)]:
        assert abs(P.storage(s, b)["tcr"] - 16 * (1 - core.tcr(s, b))) < 1e-9
    diags = [{"shape": [10, 10], "mask_frac": 0.5, "zero_frac": 0.6}, {"shape": [10, 30], "mask_frac": 0.25, "zero_frac": 0.3}]
    st = P.storage(0.4, 4, diags)
    kept = (100 * 0.5 + 300 * 0.75) / 400
    ent = (100 * P.h2(0.5) + 300 * P.h2(0.25)) / 400
    assert abs(st["entropy_actual"] - (4 * kept + ent)) < 1e-12


def test_group_symbol_entropy():
    W = torch.zeros(2, 256)
    assert P.group_symbol_entropy(W) == 0.0
    W[:, ::2] = 1.0  # every group: half 0, half 1 -> 1 bit
    assert abs(P.group_symbol_entropy(W) - 1.0) < 1e-12
    W2 = torch.arange(128, dtype=torch.float32).repeat(1, 1)  # 128 distinct values -> 7 bits
    assert abs(P.group_symbol_entropy(W2) - 7.0) < 1e-12


# ----------------------------------------------------------------------------- fingerprint
def test_fingerprint_refuses_other_gpu_type():
    locked = {"gpu_name": "NVIDIA L4", "colab_image": "img-1", "torch": "2.x", "transformers": "5.x",
              "lm_eval": "0.4.13", "datasets": "5.x", "accelerate": "1.x"}
    same = dict(locked, gpu_uuid="GPU-aaa", commit="abc")
    assert P.check_fingerprint(same, locked)
    with pytest.raises(RuntimeError):
        P.check_fingerprint(dict(same, gpu_name="NVIDIA A100-SXM4-40GB"), locked)
    with pytest.raises(RuntimeError):
        P.check_fingerprint(dict(same, lm_eval="0.4.9"), locked)


# ----------------------------------------------------------------------------- calibration sampler
class FakeTok:
    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[ord(c) % 251 for c in text]])
        return type("Enc", (), {"input_ids": ids})()


def fake_c4(n=400, seed=0):
    rng = random.Random(seed)
    return [{"text": "x" * rng.choice([50, 100, 300, 600, 1200])} for _ in range(n)]


def test_calib_documents_in_lockstep_with_get_calib():
    tok, c4 = FakeTok(), fake_c4()
    for seed in (0, 1, 2):
        wins = P.get_calib(tok, c4, 8, 256, seed)
        where = P.calib_documents(tok, c4, 8, 256, seed)
        assert len(wins) == len(where) == 8
        for w, (i, s) in zip(wins, where):
            ref = tok(c4[i]["text"], return_tensors="pt").input_ids[:, s:s + 256]
            assert torch.equal(w, ref)


def test_c4_eval_windows_are_disjoint_from_calibration():
    tok, c4 = FakeTok(), fake_c4()
    excl = set()
    for seed in (0, 1, 2):
        excl |= {i for i, _ in P.calib_documents(tok, c4, 8, 256, seed)}
    wins, where = P.c4_eval_windows(tok, c4, n=10, seqlen=256, seed=7, exclude=excl)
    assert wins.shape == (10, 256)
    docs = [i for i, _ in where]
    assert len(set(docs)) == 10 and not (set(docs) & excl)


# ----------------------------------------------------------------------------- lm-eval per-item (network)
@pytest.mark.network
def test_run_lmeval_returns_per_item_records():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = "HuggingFaceTB/SmolLM2-135M"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
    agg, items, meta = P.run_lmeval(model, tok, ["arc_easy", "boolq"], batch_size=4, limit=3)
    assert set(items) >= {"arc_easy", "boolq"}
    assert len(items["arc_easy"]["doc_id"]) == 3 and len(items["arc_easy"]["acc_norm"]) == 3
    assert "acc_norm,none" in agg["arc_easy"] and agg["arc_easy"]["n_items"] == 3
    assert all(v in (0.0, 1.0) for v in items["boolq"]["acc"])


def test_storage_matches_extra_orders_bits_per_weight():
    X = P.extra()
    for s, b in [(0.0, 3), (0.0, 4), (0.25, 4), (1 / 3, 4), (0.5, 4), (0.25, 3), (1 / 3, 3), (0.5, 3), (0.6, 4)]:
        st = P.storage(s, b)
        for key, mode in (("tcr", "none"), ("bitmap", "bitmap"), ("entropy_nominal", "entropy")):
            ref = X.bits_per_weight(s, b, mode) - (16 + b) / 128
            assert abs(st[key] - ref) < 1e-9, (s, b, key, st[key], ref)

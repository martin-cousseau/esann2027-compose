# Findings (cumulative, cleaned)

Each finding names its evidence, the script that produces the numbers, and a confidence level:
**solid** (several models, or exact), **likely** (consistent but incomplete), **tentative** (one model or one
seed). Numbers are quoted from script output (`python scripts/make_tables.py` regenerates
`reports/analysis/summary.md` and the CSVs under `reports/analysis/`); the paper takes them from
`paper/generated/macros.tex`, never from this file.

Status of the review's findings R1-R7 (as of 2026-11-11 00:45 UTC; E1 67 of 75 states (SmolLM2-135M and
Qwen2.5-0.5B complete, SmolLM2-360M 17/25), E2 complete for the three core models at every second block,
E3 16 of 48 (SmolLM2-135M), E4 9 of 15, E5 0 of 20, E6 0 of 11).

| # | Review finding | Status | Evidence so far |
|---|---|---|---|
| R1 | TCr ignores the mask; at a true 3.0 bits, 50%+4 beats plain 3-bit | **mixed (2 of 3 models in)** | With a 1-bit mask: at 4.0 bits/w, 25%+4-bit never beats plain 4-bit (SmolLM2-135M, Qwen2.5-0.5B). At 3.0 bits/w, 50%+4-bit beats plain 3-bit clearly on SmolLM2-135M (WikiText-2, C4, KL) but only ties it on Qwen2.5-0.5B (best order 21.91 vs 21.84, KL 0.523 vs 0.514, inside the cross-hardware noise); the review's 'all three models' does not hold in fp32 so far. Below 3 bits plain 2-bit collapses on both complete models (WikiText-2 ppl about 94,000 on SmolLM2-135M and 7,400 on Qwen2.5-0.5B) while 50%+3 at 2.5 bits/w stays usable (86.0 and 28.6, best order) and dominates 60%+4 at 2.6 bits/w (97.6 and 42.8, best order joint on both) (E4). Under symbol-entropy coding plain 3-bit is cheaper than 50%+4-bit. Free zeros: Q->P at 25%+3 equals 3-bit exactly on both models. SmolLM2-360M pending (`reports/analysis/frontier_points.csv`) |
| R2 | "36/36 sub-additive" rests on separate Hessians; block 0 differs | **mixed, likely** | E2 (one dense Hessian per layer, every 2nd block): P->Q and joint are sub-additive in aggregate in all 18 cells (3 models x 6 cells); per layer, 60-92% of P->Q layers and 97-100% of joint layers are. Block 0 is atypical on the SmolLM2 models only: with one Hessian, P->Q in block 0 is super-additive in all 6 cells on SmolLM2-360M (deeper blocks sub-additive) and less sub-additive on SmolLM2-135M, while Qwen2.5-0.5B's block 0 behaves like its deeper blocks. So the review's block-0 observation is real (not a calibration artefact) but model-specific, and the claim survives with one Hessian in aggregate, not per layer (`reports/analysis/e2_*.csv`) |
| R3 | Q->P mechanism untested: stale grid vs mask chosen on rounded weights | **stale grid refuted; mask selection partial, likely** | E2 (layer level, 3 models): re-fitting the grid never reduces the Q->P penalty; the pruning-only mask removes at most 39% and often a negative share. E3 (model level, SmolLM2-135M, perplexity): re-fitting removes -183% to +8%, the pruning-only mask 9% to 122% depending on the cell, both together 23% to 82%. The paper's stale-grid reading is refuted at both levels; mask selection explains part of the penalty at model level in some cells, not consistently. 360M and Qwen E3 pending (`reports/analysis/e3_mechanism.csv`) |
| R4 | Kim's weaker-first rule fails; Harma expects a small effect for SparseGPT | **confirmed so far (2 of 3 models complete), likely** | E7 (`reports/analysis/e7_*.csv`, `scripts/analysis/e7.py`). Kim's rule with their CER (linear interpolation on the quantization curve, C = 16/b) picks the better of P->Q and Q->P in 10 of 15 cells by WikiText-2 ppl (SmolLM2-135M 5/6, SmolLM2-360M 2/3 so far, Qwen2.5-0.5B 3/6) and fails in all 4 cells where it predicts quantize-first: P->Q wins there too. Their Hypothesis 1 has the wrong sign on all three models (Spearman of the P->Q advantage vs the CER difference -0.54, -1.00 with n = 3, -0.60): P->Q gains most where pruning is the stronger perturbation. On accuracy (Sr-extended) the rule scores 12/15 because order differences in accuracy are small. Harma: sparsity first wins in 14 of 15 cells (the exception is Qwen2.5-0.5B 25%+3, -0.4%, the free-zeros cell); the order effect is large under OBS compensation (Q->P up to +77% ppl, SmolLM2-135M 50%+3), against their App. F expectation for SparseGPT (measured there at 6-8 bits only). Their additive threshold is exceeded by Q->P in 12 of 15 states but by P->Q in 5 of 15 and joint in 8 of 16; the multiplicative null (additive loss) only by Q->P (11 of 15; P->Q 1, joint 3). E6 pending |
| R5 | Scale range 3.4x, family changes at the last step | **pending** | E6 (SmolLM2-1.7B) not run yet |
| R6 | Noise: seeds, task SE, no per-item logs | **confirmed in a new form, solid** | Per-item logs and full splits now exist for every state. New: the same code at the same precision on two GPU types (16-bit, T4 vs L4) changes SmolLM2-135M perplexity by +1.5% (4-bit) and +4.6% (50%+4 P->Q) (`results/v2/REPRO16/`), the size of the calibration-seed spread; within one GPU type results are bit-exact (DET). Gaps of a few percent between orders are not meaningful without seeds. E5 pending |
| R7 | 16-bit overflow; safeguard only warns | **addressed, solid** | All v2 runs are fp32 with the strict check (raises on non-finite Hessians); no overflow so far. fp32 on L4 reproduces the paper's own fp32 re-runs of SmolLM2-360M within 0.4% (`results/v2/REPRO32/`) |

## Other findings

1. **Input correlation predicts the layer interaction (solid).** Each layer's P->Q interaction ratio correlates
   with the mean |corr| of its Hessian: Spearman 0.41-0.60, p < 0.001 in every cell of all 3 models (E2,
   `scripts/analysis/e2.py`). The review's synthetic prediction (sub-additivity erodes as correlation grows)
   holds on real layers.
2. **Free zeros at 3 bits (likely).** At 25%+3 and 33%+3, Q->P has lower layer error than P->Q on all
   three models (E2): 3-bit rounding already sets about a third of the weights to zero, so a 25% Q->P mask mostly
   picks free zeros. Model-level check (Q->P at 25%+3 equals 3-bit alone) waits for the E1 3-bit states.
3. **Dense models reproduce exactly (solid).** fp32 WikiText-2 perplexity equals the published 16-bit values to
   within 0.003% for SmolLM2-135M and Qwen2.5-0.5B (E1 dense states).
4. **Colab runtimes need a client keepalive (operational, solid).** Runtimes are reclaimed 20-35 min after the
   last client code execution, whatever runs inside (DECISIONS.md #18, #24).
5. **Layer error predicts KL across states, not the order within a cell (likely).** Spearman of the mean layer error vs KL over all compressed states: +0.83 (SmolLM2-135M), +0.96 (SmolLM2-360M, partial), +0.95 (Qwen2.5-0.5B). Within a cell, the order with the smallest layer error also has the smallest KL in only 1 of 7 cells on SmolLM2-135M (joint has the smallest layer error, P->Q the smallest KL), but in 3/3 and 4/6 cells on the other two; at 25%/33%+3 Q->P has the smallest layer error (free zeros) without the smallest KL (E8 analysis part, `reports/analysis/e8_layer_error_vs_kl.json`). The draft's open question has a split answer.

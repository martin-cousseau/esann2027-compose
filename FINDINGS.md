# Findings (cumulative, cleaned)

Each finding names its evidence, the script that produces the numbers, and a confidence level:
**solid** (several models, or exact), **likely** (consistent but incomplete), **tentative** (one model or one
seed). Numbers are quoted from script output (`python scripts/make_tables.py` regenerates
`reports/analysis/summary.md` and the CSVs under `reports/analysis/`); the paper takes them from
`paper/generated/macros.tex`, never from this file.

Status of the review's findings R1-R7 (as of 2026-11-10 11:50 UTC; E1 at 8 of 75 states, E2 complete
for the three core models at every second block, E3/E4/E5/E6 not yet run).

| # | Review finding | Status | Evidence so far |
|---|---|---|---|
| R1 | TCr ignores the mask; at a true 3.0 bits, 50%+4 beats plain 3-bit | **pending** | needs E1 3-bit and 50%+4 states on all models, then E4 (`reports/analysis/frontier_*.csv`) |
| R2 | "36/36 sub-additive" rests on separate Hessians; block 0 differs | **mixed, likely** | E2 (one dense Hessian per layer, every 2nd block): P->Q and joint are sub-additive in aggregate in all 18 cells (3 models x 6 cells); per layer, 60-92% of P->Q layers and 97-100% of joint layers are. Block 0 is atypical on the SmolLM2 models only: with one Hessian, P->Q in block 0 is super-additive in all 6 cells on SmolLM2-360M (deeper blocks sub-additive) and less sub-additive on SmolLM2-135M, while Qwen2.5-0.5B's block 0 behaves like its deeper blocks. So the review's block-0 observation is real (not a calibration artefact) but model-specific, and the claim survives with one Hessian in aggregate, not per layer (`reports/analysis/e2_*.csv`) |
| R3 | Q->P mechanism untested: stale grid vs mask chosen on rounded weights | **refutes both readings (layer level), likely** | E2: re-fitting the grid never reduces the Q->P penalty (it increases it in every cell with a penalty, all 3 models); the pruning-only mask removes at most 39% and often a negative share. Neither the paper's stale-grid reading nor Harma et al.'s mask-selection reading explains most of the penalty. Model-level confirmation waits for E3 |
| R4 | Kim's weaker-first rule fails; Harma expects a small effect for SparseGPT | **pending** | E7 (analysis only) needs E1. Literature check: Harma et al. measured the small SparseGPT/Wanda order effect at 6-8 bits only, so the 3-4-bit contrast is weaker than the review states (`reports/literature.md`) |
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

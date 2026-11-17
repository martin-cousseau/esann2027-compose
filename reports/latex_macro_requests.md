# Macro requests from the paper (phase L2, 2026-11-10)

`paper/main.tex` already uses the macro names below. Each one has a fallback in the preamble,
`\providecommand{\resX}{\TBD{Rn}}`, which prints a bold **[Rn]** until `scripts/make_tables.py` defines
`\resX` in `paper/generated/macros.tex` (that file is read first, so its definition wins and the fallback
does nothing). If a macro is added under the exact name listed here, the paper picks it up on the next build
with no edit to `main.tex`.

Conventions, as in `MACROS.md`: names are letters only; suffix `Pct` means the macro holds the number
without the `%` sign; signed values are wrapped in `\ensuremath{...}`; `--` means "not available". "Best
order" means the lowest WikiText-2 perplexity among joint, P->Q and Q->P (all three required), the rule
`fig_frontier_sparse_*.dat` and `table_main_rows.tex` already use. Unless stated otherwise, counts run over
the core models (SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B, plus SmolLM2-1.7B once E6 is in), seed 0.

"Expected now" is my recomputation from `reports/analysis/*.csv` (make_tables run of 2026-11-10 13:55 UTC).
It is there only to check an implementation; it is not used in the paper.

The PDF currently prints 37 placeholders (R1 to R29). R30 and R31 are inside the E5 slot, which is
switched off (`\Efivefalse`), so they do not print.

| R | Macro(s) | Definition | Source | Format | Expected now | Used in |
|---|---|---|---|---|---|---|
| R1 | `\resFrontierThreeWins` | Number of models where the best order of 50%+4 has a lower WikiText-2 ppl than plain 3-bit. Both cost 3.0 bits/w with a 1-bit mask. | `frontier_points.csv` (E1) | integer | 1 (SmolLM2-135M only) | abstract; Sec. 5 Storage |
| R2 | `\resFrontierThreeModels` | Number of models where 3-bit and all three 50%+4 orders are available. | same | integer | 1 | abstract; Sec. 5 Storage |
| R3 | `\resFrontierThreeGainMinPct` | Over the R1 models: smallest 100*(1 - ppl_best(50%+4)/ppl(3-bit)). | same | 1 decimal, unsigned | 28.4 | Sec. 5 Storage |
| R4 | `\resFrontierThreeGainMaxPct` | The same, largest. | same | 1 decimal, unsigned | 28.4 | Sec. 5 Storage |
| R5 | `\resFrontierFourGapMaxPct` | Over the models with 4-bit and all three 25%+4 orders: largest abs(100*(ppl_best(25%+4)/ppl(4-bit) - 1)). Both cost 4.0 bits/w. | same | 1 decimal, unsigned | 0.3 (0.257, SmolLM2-135M) | Sec. 5 Storage |
| R6 | `\resBitsSymSmolSmallQuantBThree`, `\resBitsSymSmolSmallSFiftyBFourPQ` | Symbol entropy in bits/w (plug-in entropy of the stored symbols per group of 128, zeros included). Best added as a new per-state column `BitsSym` in every state grid of MACROS.md. | field `storage.symbol_entropy` (= `bits_symbol_entropy` in `frontier_points.csv`) | 2 decimals | 2.42 and 2.62 | Sec. 5 Storage |
| R7 | `\resIdentityQPThreeCount` | Number of models where Q->P at 25%+3 is identical to plain 3-bit (ppl and every per-layer rel_err; `identical: true`). | `identity_qp25_3bit.json` | integer | 1 | Sec. 5 Free zeros |
| R8 | `\resIdentityQPThreeModels` | Number of models where both states exist (status not `pending`). | same | integer | 1 | Sec. 5 Free zeros |
| R9 | `\resETwoMaskFirstSub` | E2: number of (model, cell, order) with order in {joint, P->Q} whose summed-layer ratio I_E/(E_P+E_Q) is < 0 (`ratio_joint_pct`, `ratio_p_then_q_pct`). | `e2_cells.csv` | integer | 36 | Sec. 5 Sub-additivity |
| R10 | `\resETwoMaskFirstTotal` | E2: number of such (model, cell, order) combinations available. | same | integer | 36 | Sec. 5 Sub-additivity |
| R11 | `\resETwoSubJointMinPct` | E2: smallest share of sub-additive layers for joint over cells (`share_sub_joint`, %). | same | integer % | 97 | Sec. 5 Sub-additivity |
| R12 | `\resETwoSubJointMaxPct` | The same, largest. | same | integer % | 100 | Sec. 5 Sub-additivity |
| R13 | `\resETwoSubPQMinPct` | E2: smallest share of sub-additive layers for P->Q over cells (`share_sub_p_then_q`, %). | same | integer % | 60 (59.52) | Sec. 5 Sub-additivity |
| R14 | `\resETwoSubPQMaxPct` | The same, largest. | same | integer % | 92 (91.67) | Sec. 5 Sub-additivity |
| R15 | `\resETwoRhoMin` | E2: smallest Spearman rho over cells, P->Q per-layer ratio vs `h_corr_index` (arm `p_then_q`). | `e2_spearman.csv` | `\ensuremath{+0.41}` style | +0.41 (0.415) | Sec. 5 Sub-additivity |
| R16 | `\resETwoRhoMax` | The same, largest rho. | same | signed, 2 decimals | +0.60 (0.597) | Sec. 5 Sub-additivity |
| R17 | `\resETwoRhoPMax` | Largest p-value of those rhos. | same | `\ensuremath{5.43\times 10^{-6}}` style (as `\resETwoRhoP...`) | 5.43e-6 | Sec. 5 Sub-additivity |
| R18 | `\resETwoBlockZeroSuperPQSmolSmall`, `\resETwoBlockZeroSuperPQSmolMedium`, `\resETwoBlockZeroSuperPQQwen` | E2, per model: number of the six cells where the P->Q ratio of block 0 (`ratio_block0_p_then_q_pct`) is > 0. | `e2_cells.csv` | integer | 1, 6, 0 | Sec. 5 Sub-additivity |
| R19 | `\resQTwoQPWorstPplBFour` | E1, 4-bit cells only (25/33/50/60% + 4): number of cells where Q->P has the strictly highest WikiText-2 ppl (like `\resQTwoQPWorstPpl`, restricted to b = 4). | `q2_order_ranking.csv` | integer | 3 | Sec. 5 Order |
| R20 | `\resQTwoCellsPplBFour` | E1, 4-bit cells with all three orders ranked by WikiText-2 ppl. | same | integer | 3 | Sec. 5 Order |
| R21 | `\resETwoRefitWorse` | E2: among cells with a Q->P penalty (`qp_penalty_sum` > 0), the number where the re-fitted grid removes a negative share (`penalty_removed_q_then_p_refit_pct` < 0). | `e2_cells.csv` | integer | 12 | Sec. 5 Order |
| R22 | `\resETwoPenaltyCells` | E2: number of cells with `qp_penalty_sum` > 0. | same | integer | 12 | Sec. 5 Order |
| R23 | `\resETwoRemovedPmaskMaxPct` | E2: largest `penalty_removed_q_then_p_pmask_pct` over the R22 cells. | same | integer %, unsigned | 39 | Sec. 5 Order |
| R24 | `\resETwoStaleMaxAbsPct` | E2: largest abs(`stale_grid_on_pq_pct`) over all cells. | same | 1 decimal % | 0.4 (0.356) | Sec. 5 Order |
| R25 | `\resMcNemarSigPQQP` | E1: number of cells where the exact McNemar test of P->Q vs Q->P on the pooled nine tasks (`task = pooled9`) has p < 0.05. | `e1_mcnemar.csv` | integer | 4 | Sec. 5 Evaluation |
| R26 | `\resMcNemarCellsPQQP` | E1: number of cells with that test. | same | integer | 4 | Sec. 5 Evaluation |
| R27 | `\resHwDeltaSmolSmallQuantBFourPct`, `\resHwDeltaSmolSmallSFiftyBFourPQPct` | 100*(ppl_REPRO16/ppl_published - 1): same code, 16-bit, on an L4 (REPRO16) vs the published 16-bit T4 run of the same configuration, WikiText-2 ppl, SmolLM2-135M 4-bit and 50%+4 P->Q. | `results/v2/REPRO16/SmolLM2-135M.jsonl` vs `results/merged_results.json` | signed, 1 decimal | +1.5 (22.1005 vs 21.7742), +4.6 (42.0171 vs 40.1520) | Sec. 5 Noise |
| R28 | `\resReproThirtyTwoMaxAbsPct` | Largest abs(100*(ppl_REPRO32/ppl_fix - 1)) over the three REPRO32 states (SmolLM2-360M 25%+4, three orders) vs the paper's own fp32 re-runs on a T4. | `results/v2/REPRO32/SmolLM2-360M.jsonl` vs `results/results_fix.jsonl` | 1 decimal | 0.4 (0.37; DECISIONS.md #23) | Sec. 5 Noise |
| R29 | `\resAccDiffSmolSmallSFiftyBFourPQQP`, `\resAccDiffLoSmolSmallSFiftyBFourPQQP`, `\resAccDiffHiSmolSmallSFiftyBFourPQQP` | Paired accuracy difference P->Q minus Q->P at 50%+4, SmolLM2-135M: mean over the nine tasks (v2 metric), in points (x100), with its paired-bootstrap 95% interval. A general version `\resAccDiff<model><cell><pair>` for every cell and pair would also work. | `e1_acc_orders.csv`, row `task = mean9_v2`, columns diff/lo/hi | signed, 1 decimal | -1.4 [-2.1, -0.7] (-1.427 [-2.147, -0.710]) | Sec. 5 Evaluation |
| R30 | `\resSeedBestStable`, `\resSeedBestCells` | E5 (switch `\Efivetrue`): number of 4-bit cells, on the E5 models, whose best order by WikiText-2 ppl is the same for all three calibration seeds; and the number of cells with three seeds. | `e1_seeds.csv` | integer | pending | Sec. 5 Noise (E5 slot) |
| R31 | `\resSeedSdMaxSinglePct`, `\resSeedSdMaxCombinedPct` | E5: largest relative sample sd (100*sd/mean) of WikiText-2 ppl over three seeds: over single-method states, and over combined states. | `e1_seeds.csv` | 1 decimal | pending | Sec. 5 Noise (E5 slot) |

## Generated files I did not use, and why

* `table_main_rows.tex` (43 body rows, one per configuration and model, best order only) does not fit the
  6-page budget: about 0.8 page at footnotesize, while the final paper has about 2 lines of slack. Table 1 is
  built instead from the per-state macros
  `\resPpl / \resKl / \resFlipsPct` + model tag + {QuantBFour, STwentyFiveBFourJoint/PQ/QP, QuantBThree,
  SFiftyBFourJoint/PQ/QP}. It shows all three orders, so no best-order choice is made in the paper. The bold
  "best of the four at equal storage" is computed in TeX from the macro values, and no bold is shown while
  any of the four is `--`. If you prefer a generated body for it, the layout is: per model, three rows (PPL,
  KL, Flips) × 8 values (4-bit, 25%+4 joint/P->Q/Q->P, 3-bit, 50%+4 joint/P->Q/Q->P).
* `table_mechanism_rows.tex` (18 rows, 6 of them "n/a (--)") was replaced for the same reason. Table 2 is built from
  the E2 per-cell macros (`\resETwoPenalty`, `\resETwoRemovedRefit`, `\resETwoRemovedPmask`,
  `\resETwoRemovedBoth` + model + cell + `Pct`), with 6 rows and 3 models side by side. E2 is complete, so this
  table is final. E3 is quoted in the text for one cell (SmolLM2-135M, 50%+4) through the existing
  `\resEThreeRemoved...` macros. Optional, if you want E3 in the table: a body-rows file with the same
  13-column layout (cell & 3 × [penalty, refit, mask, both]) for the E3 cells.
* `table_kl_flips.tex`: KL and flips for the two equal-storage cells are already in Table 1; no room for more.

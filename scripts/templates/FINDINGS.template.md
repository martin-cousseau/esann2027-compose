# Findings (cumulative)

Generated: `python scripts/render_template.py scripts/templates/FINDINGS.template.md paper/generated/macros.tex -o FINDINGS.md --parts reports`.
Every number below is a macro of `paper/generated/macros.tex` (written by `scripts/make_tables.py` from the result
lines); nothing is typed by hand. Confidence: **solid** (several models, or exact), **likely** (consistent but
incomplete), **tentative** (one model or one seed).

## Status of the review's findings R1-R7 (final, 17 November 2026: seven models, 135M-3B)

Scope: SmolLM2-135M, SmolLM2-360M and Qwen2.5-0.5B (E1, every state, three calibration seeds), OLMo-2-1B,
Qwen2.5-1.5B, Qwen2.5-3B and SmolLM2-1.7B (E6/E6B: the same grid; three calibration seeds on Qwen2.5-1.5B),
layer-level controls on every second or fourth block of all seven (E2: one shared Hessian; E9: no compensation).
No state failed. The paper written on these results is the first author's revision (V4.1, private).

| # | Review finding | Status | Evidence |
|---|---|---|---|
| R1 | TCr ignores the mask; at a true 3.0 bits, 50%+4 beats plain 3-bit | **mixed, solid (four models); left out of V4.1** | Unchanged since 11 November (history below): with a 1-bit mask the gain of pruning at equal storage is model-dependent. V4.1 compares orders at equal (s, b), where the storage is the same for all three, and says that real storage adds a mask and grid parameters |
| R2 | The sub-additivity rests on separate Hessians; block 0 differs | **resolved, solid (seven models)** | With one dense Hessian shared by all orders, the mask-first orders are sub-additive in aggregate in {{resETwoMaskFirstSubAllModels}} of {{resETwoMaskFirstTotalAllModels}} (model, setting, order) cases; in the pipeline in {{resQOneMaskFirstSubAllModels}} of {{resQOneMaskFirstTotalAllModels}}; without compensation, where the paper's decomposition is exact, in {{resENineSubPQStaleAllModels}} of {{resENineCellsAllModels}} settings, with the summed cross terms within {{resENineCrossMaxAbsAllModelsPct}}% of E_P + E_Q. Not every layer is sub-additive (P->Q: {{resETwoSubPQMinPct}}-{{resETwoSubPQMaxPct}}% of the layers, three core models) |
| R3 | The mechanism of quantize-first is untested (stale grid vs mask chosen on rounded weights) | **resolved at the layer level, solid; partial at the model level** | The one-Hessian split of the Q->P / P->Q layer-error gap (seven models): grid at most {{resETwoSplitGridMaxAbsAllModelsPct}}%; mask {{resETwoSplitMaskMinAllModelsPct}} to {{resETwoSplitMaskMaxAllModelsPct}}% (the rounded-weight mask is the better one at 25% and 33% with 3 bits, {{resETwoSplitMaskNegFreeZerosAllModels}} of {{resETwoSplitFreeZerosCellsAllModels}}); update order {{resETwoSplitOrderMinAllModelsPct}} to {{resETwoSplitOrderMaxAllModelsPct}}%, positive in {{resETwoSplitOrderPosAllModels}} of {{resETwoSplitCellsAllModels}}, the largest part in {{resETwoSplitOrderLargestGapPosAllModels}} of the {{resETwoSplitGapPosAllModels}} settings where Q->P has the larger layer error ({{resETwoSplitRevOrderLargestGapPosAllModels}} with the grid exchanged first). Where Q->P has the smaller layer error ({{resETwoSplitGapNegAllModels}}, all at 3 bits) its perplexity is still the higher in {{resETwoSplitGapNegQPWorsePplAllModels}}: the layer error misses part of the penalty. Without compensation the update-order part is zero by construction and the mask alone makes Q->P worse at 4 bits ({{resENineSplitMaskMinAllModelsBFourPct}} to {{resENineSplitMaskMaxAllModelsBFourPct}}%) |
| R4 | Kim's weaker-first rule; Harma expects a small effect for SparseGPT | **solid by perplexity and KL, not by accuracy** | The rule (by WikiText-2 ppl) recommends Q->P in {{resESevenKimPredQPCellsPplAllModels}} of {{resQTwoCellsPplAllModels}} settings, all at {{resESevenKimPredQPMinSparsityPctAllModels}}% sparsity or more (beyond the range Kim et al. tested); there Q->P is never the best by ppl or KL and the worst in {{resESevenKimPredQPWorstPplAllModels}} (KL {{resESevenKimPredQPWorstKlAllModels}} of {{resESevenKimPredQPCellsKlAllModels}}); by nine-task accuracy it is the best in {{resESevenKimPredQPPplBestSrExtAllModels}} of the {{resESevenKimPredQPPplWithSrExtAllModels}} with benchmarks |
| R5 | Scale range 3.4x, family change at the last step | **addressed, likely** | Seven models up to {{resNWeightsQwenLargeB}}B compressed weights, three families. At 4 bits the order spread is {{resQThreeSpreadMinCoreBFourPct}}-{{resQThreeSpreadMaxCoreBFourPct}}% (median {{resQThreeSpreadMedianCoreBFourPct}}%) on the small models and {{resQThreeSpreadMinLargerBFourPct}}-{{resQThreeSpreadMaxLargerBFourPct}}% (median {{resQThreeSpreadMedianLargerBFourPct}}%) on the larger ones; it depends on the model (SmolLM2-360M vs Qwen2.5-0.5B: {{resScaleMatchedSmallSTwentyFiveBFourRatio}} and {{resScaleMatchedSmallSFiftyBFourRatio}} times); for every combination of calibration seeds it falls from SmolLM2-135M to -1.7B in {{resScaleSeedEndsShrinkFamSmol}} of {{resScaleSeedEndsCellsFamSmol}} settings, and Qwen2.5-1.5B has the largest spread of its family in only {{resScaleSeedPeakMidFamQwen}} |
| R6 | Noise: seeds, task standard errors, no per-item logs | **solid** | Calibration seeds 1 and 2 on four models: Q->P is the worst order in {{resSeedRunsQPWorstAllBoth}} of {{resSeedRunsAllBoth}} runs and the best in {{resSeedRunsQPBestAllBoth}}. Per-item logs and full splits for every state with benchmarks; paired bootstrap intervals. The {{resRerunStates}} states run more than once (DET, REPRO32, BATCH, E6B, IMG) are bit-identical on other GPU units; the batch-size check (BATCH): {{resBatchNllIdentical}} of {{resBatchStates}} states with identical perplexity, {{resBatchItemsDiffer}} of {{resBatchItemsTotal}} items changed |
| R7 | 16-bit overflow; the safeguard only warns | **addressed, solid** | All v2 runs are fp32 with the strict check; fp32 on the L4 reproduces the paper's own fp32 re-runs within {{resReproThirtyTwoMaxAbsPct}}% (REPRO32) |

## Other findings

1. **Free zeros at 3 bits (solid, seven models).** 3-bit GPTQ alone zeroes {{resFreeZerosPctMinAllModelsQuantBThree}}-{{resFreeZerosPctMaxAllModelsQuantBThree}}% of the weights; Q->P at 25%+3 is
   the 3-bit model exactly on {{resQOneQPEqualsQuantAllModelsSTwentyFiveBThree}} of 7 models.
2. **Input correlation predicts the layer interaction (solid, seven models).** In every (model, setting) the
   per-layer P->Q interaction correlates with the mean |corr| of the layer's Hessian (Spearman {{resETwoRhoMin}} to {{resETwoRhoMax}},
   largest p-value {{resETwoRhoPMax}}).
3. **The model-level interaction survives a chance correction (solid).** I_Sr of P->Q at 33%+3-bit is positive on
   all {{resISrPaperNAllModelsSThirtyThreeBThreePQ}} benchmarked models with raw retention and on {{resISrPaperCnPosAllModelsSThirtyThreeBThreePQ}} with chance-corrected retention ({{resISrPaperCnExclZeroPosAllModelsSThirtyThreeBThreePQ}} intervals above zero in both).
4. **A mask-first order is almost always best; which one depends on the model (solid).** By WikiText-2 ppl (seed 0,
   {{resQTwoCellsPplAllModels}} settings) Q->P is the best in {{resQTwoQPBestPplAllModels}}; on the three small models joint is the best in {{resQTwoBestJointPpl}} and P->Q in {{resQTwoBestPQPpl}}.

## Caveats

- **Cholesky fallback.** `src/core.py` retries a failed Cholesky factorization with damping x 10^k (k <= 4) without
  recording it; the paper states the base damping.
- **Best order chosen on the test set.** The bold entries of Table 1 take the minimum of three orders on WikiText-2
  test perplexity; near-ties (a few hundredths of perplexity) are within the calibration-seed spread
  (`paper/supplement/seeds.csv`).
- **Resolved since 11 November:** the lm-eval batch configuration of four early states (the BATCH check above).

{{file:FINDINGS_history_2026-11-11.md}}

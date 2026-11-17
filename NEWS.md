# NEWS (newest first)

## Read this first
- **Now (17 Nov, 06:00 UTC):** V4.1, the final pass before submission, is delivered with the full submission package
  (private zip `context/esann2027_submission_v4.1_2026-11-17.zip`, backed up to the private dataset): 6 pages, margins
  centred to the ESANN rule, 40 worded claims checked; an independent audit found no numerical error; two simulated
  reviews (weak accept) and the fixes are in the package (`REFLECTION.md`, acceptance estimate 60-75%;
  `COVER_NOTE.md`, changes and checklist). Supplement extended and re-uploaded; Hub READMEs brought up to date
  (DECISIONS.md #66). ESANN 2027 extended deadline: 25 November 2026.
- **16 Nov, 20:30 UTC:** the V4 update with the Qwen2.5-1.5B seed check is delivered (private package
  `context/esann2027_author_package_v4_update_2026-11-16.zip`, backed up to the private dataset). All 13 missing
  seed states ran. Whichever calibration seeds are compared, the 1.5B peak within Qwen2.5 holds in 3 of the 5 settings
  where seed 0 shows it, and the fall from SmolLM2-135M to -1.7B in 5 of 6; Q3 says so (DECISIONS.md #65). 6 pages,
  37 claims hold; `diff_v4_update.pdf` shows the four changed lines against the V4 sent at 16:45.
- **16 Nov, 16:45 UTC:** V4, the final revision for the first author, is delivered (private package, backed up
  to the private dataset): 6 pages, every number generated, 36 worded claims checked against the data, independently
  audited (one blocker and four major points, all fixed: DECISIONS.md #63), the team's review inputs sorted one by one
  (`REVIEW_V4.md` in the package).
- **What V4 adds:** the introduction positions the paper against Harma, Kim, Bouriel, a 2026 preprint (Nema et al.),
  JSQ and OBR; the penalty of quantizing first is traced to the order of the compensated updates (same-Hessian
  ablation, all seven models: +4.5 to +18.1% layer error with the same mask and grid, the grid at most 0.5%); the
  paper's decomposition is checked without compensation (new experiment E9, all seven models: cross terms at most
  0.13% of E_P+E_Q, mask-first sub-additive in 42 of 42 settings); a generated supplement with every state's
  results (`paper/supplement/`, also on the results dataset).
- **GPU:** no runtime is assigned; the hourly check-in is deleted. Compute units: `logs/budget.md` (about 55 of the 80
  left, estimate; the seed check used about 5.9 L4-hours).
- **Links:** PR https://github.com/martin-cousseau/esann2027-compose/pull/1 (draft) ; models
  https://huggingface.co/martincousseau/esann2027-compose-v2 ; results and supplement
  https://huggingface.co/datasets/martincousseau/esann2027-v2-results ; runs https://wandb.ai/martin-css/esann2027-compose-v2

## 2026-11-17 06:00 UTC - V4.1 and the submission package
- Audit and two simulated reviews of the V4.1 draft; fixes, macros and supplement tables (#66). No GPU used.

## 2026-11-16 20:30 UTC - Qwen2.5-1.5B seed check, V4 update
- Runtimes esann-a14 and esann-b13 ran the 13 seed-2 states (#64), then were deleted. The tables, the supplement's
  `seeds.csv` (re-uploaded to the results dataset) and V4 were rebuilt (#65).

## 2026-11-16 16:45 UTC - V4 final runs and package
- E2 (one shared Hessian) and E9 (no compensation) now cover all seven models; the runtimes were lost three times
  while this session was paused by its usage limit (DECISIONS.md #58-#61), nothing finished was lost.
- New: `scripts/run_e9.py`, `scripts/analysis/e9.py`, `scripts/make_supplement.py`, `scripts/hub_sync.py`, tests.

## 2026-11-15 18:30 UTC - Qwen2.5-3B complete, seven-model text (v3)
- The last five Qwen2.5-3B states ran on one runtime with the compute units added at 15:00 (#55). The paper text,
  claims and note are updated for seven models (#57); the Q3 sentence about Qwen2.5 is rewritten.
- New analysis: small-model scope for the seed macros, and a seed check of the scale trend (`macros_scale_seeds`).
  On SmolLM2 it already shows that SmolLM2-360M's apparent peak at seed 0 does not survive the seeds, while the
  fall from 135M to 1.7B does in 5 of 6 cells.

## 2026-11-15 08:00 UTC - final package v2 (six models), audited
- Final package sent: six complete models, Table 2 rows at 1-2B, seeds of the three small models. Qwen2.5-3B left out
  (incomplete). An independent audit found one false statement (the shared-Hessian scope in the abstract) and several
  overstatements; all fixed (DECISIONS.md #52).
- The runtimes were reclaimed twice on 15 Nov while a runtime command waited for approval during worker restarts
  (#50, #51); GPU work stopped.

## 2026-11-14 16:35 UTC - interim package v2 sent
- At your request, an interim package for the team (private; sent as a zip and backed up to the private dataset):
  the revised paper (6 pages) now counts SmolLM2-1.7B with the three small models (full grid, Table 2 row) and
  leaves out the models still running (DECISIONS.md #48). Every number is generated; every sentence that turns
  numbers into words is checked by `check_claims.py` (20 of 20 hold); the note lists the changes since v1.1.
- The final package (seven models, Table 2 rows at 1-2B, seeds of the three small models) follows when the runs end
  (about 15:30 UTC on 15 November).

## 2026-11-14 13:30 UTC - SmolLM2-1.7B complete; Qwen2.5-3B to the full grid
- SmolLM2-1.7B, full grid: quantize-first is the worst order by perplexity in 5 of 6 cells (not at 33%+4, where the
  joint sweep is worst by 2.4%); its penalty at 50%+3 is 1.34 times the best perplexity. At 25%+3, quantize-first
  equals 3-bit quantization alone (same WikiText-2 NLL sum): the 3-bit grid rounds 30% of the weights to zero, and
  the 25% pruning then removes only zeros.
- Table 2 at 1.7B: I_Sr for P->Q is -0.002 [-0.010, +0.005] at 25%+4 and +0.011 [-0.000, +0.023] at 33%+3. In the
  weaker-first cell (50%+4), quantize-first is the worst order on all metrics, including accuracy over nine tasks:
  the accuracy reversal of the small models does not appear at 1.7B.
- Runtime B completes the Qwen2.5-3B grid after its own queue (13 states, its own result file; DECISIONS.md #47).

## 2026-11-13 13:45 UTC - author package v1; runs stopped; Colab connector down
- **Package v1 delivered:** `revised/main.pdf` (6 pages), `diff/main_diff.pdf`, `NOTE.md` (change log C1-C38, every
  quote checked against the paper by `check_quotes.py`), corrected bibliography and literature check. Independent
  audit running.
- **Tools (public):** `scripts/flatten_macros.py` (inline macros for latexdiff), `scripts/render_template.py` (reports
  that quote no number by hand), `scripts/note_tables.py` (the note's tables from the analysis outputs).
- **Analysis (subagent, commit 7a68106):** seeds at 4 and 3 bits, family vs scale at matched size, BATCH check,
  Table-2 I_Sr with intervals, the 7-8B I_Sr parsed from the author's script, Fig. 1 v2 with the E2 values.
- **Literature check:** four wrong references fixed (OBR is ICLR 2026, SmolLM2 COLM 2025, OPTQ title, lm-eval v0.4.13);
  OBC is prune-then-quantize; Kim et al. tested pruning up to 30%; the IJCNN 7-8B models did not keep the mask.
- **Runs:** stopped 12 Nov ~21:35 UTC (see above); restart needs the Colab connector.

## 2026-11-11 08:55 UTC - all runs done, runtimes deleted, release public
- **E6 (SmolLM2-1.7B, 11 states):** at 3.0 bits per weight plain 3-bit beats the best 50%+4-bit order, at 4.0 bits
  plain 4-bit beats 25%+4-bit; within SmolLM2 the 4-bit order spread shrinks with size (FINDINGS.md R1, R5).
- **E5 (seeds, SmolLM2-360M):** joint beats P->Q in every seed of every 4-bit cell; Q->P is always worst; seed sd of
  perplexity at most 1.0% (R6).
- **E3 (mechanism, 40 states):** re-fitting the grid helps in 2 of 10 cells; the pruning-only mask helps in 7 of 10, all
  on SmolLM2, in none on Qwen2.5-0.5B (R3). E3 Qwen2.5-0.5B was reduced to its two 50% cells for budget (#35).
- **Order rules (E7):** Kim et al.'s rule right in 14 of 20 cells, wrong in all 5 where it predicts quantize-first;
  sparsity first wins in 19 of 20 (R4).
- **Paper:** switched on E5 and E6, a scale paragraph, mechanism claims matched to E2 + E3, Fig. 1 made readable
  (plain 2-bit off scale, stated in the caption), Table 1 with perplexity and KL for all four models.
- **Release:** HF model repository (157 exported states verified by size and sha256, card, standalone loader) and
  results dataset public (#36); README of this branch links them and W&B.
- **Runtimes:** A deleted 07:21, B deleted 08:48 UTC.

## 2026-11-11 03:25 UTC - E1 complete; angle B + C confirmed on three models; paper revised
- **E1 complete (75 states):** with the mask counted, 25%+4-bit ties plain 4-bit on every core model; 50%+4-bit
  beats plain 3-bit at 3.0 bits/w on both SmolLM2 models and ties it on Qwen2.5-0.5B; below 3 bits plain 2-bit
  collapses on all three. Q->P at 25%+3 is the 3-bit model on all three (DECISIONS.md #33, FINDINGS.md R1).
- **Order:** a mask-first order is best in all 18 cells (joint 11, P->Q 7); Q->P strictly worst in 16. Kim et al.'s
  rule is right in 13 of 18 cells and wrong in all 4 where it predicts quantize-first, as the review said.
- **Paper:** abstract, order paragraph (with the two order rules) and conclusion updated; the per-cell mechanism
  table moved to the repository for space; still 6 pages, every number a macro.
- **Operations:** the container of this session restarted at about 01:50 (watcher restarted, nothing lost). The
  E6 chain on runtime A waited 5 min on a finished but unreaped (zombie) process; reaped by hand. Runtime B now
  runs the seeds (E5) before E3 (#32).

## 2026-11-11 00:55 UTC - order rules, E6 extended, release tooling
- **E7 (analysis only):** Kim et al.'s weaker-first rule, with their compression-equivalent ratio, picks the
  better of P->Q and Q->P in 10 of 15 complete cells by WikiText-2 perplexity and fails in all 4 cells where it
  predicts quantize-first; their monotonicity hypothesis has the wrong sign on all three models. Harma et al.'s
  sparsity-first order wins in 14 of 15 cells; their additive threshold is exceeded by Q->P in 12 of 15 states
  but by P->Q in only 5 (`reports/analysis/summary.md`, section E7).
- **E8 (analysis part):** the mean layer error predicts KL across states (Spearman +0.83 to +0.96) but on
  SmolLM2-135M not which order wins within a cell (1 of 7 cells).
- **E6 extended (#31):** the 1.7B queue now also has Q->P at 25%+4 and 50%+4 and pruning alone at 25% and 50%,
  so E6 can measure the order spread at scale and the order rules; about 4.5 CU more.
- **Release tooling:** `release/load_esann2027.py` (standalone loader, bit-exact with the pipeline) and
  `release/make_hf_readme.py` (model card generated from the result lines, Hub sizes and hashes verified).
- **Private inputs** backed up to a separate private HF dataset (DECISIONS.md #2).

## 2026-11-10 18:15 UTC - SmolLM2-135M complete; draft paper; restart after a suspension
- **SmolLM2-135M, E1 complete** (`results/v2/E1/SmolLM2-135M.jsonl`, `reports/analysis/summary.md`): with the
  1-bit mask counted, 50%+4-bit P->Q beats plain 3-bit at 3.0 bits/w (WikiText-2 41.2 vs 57.6, KL 0.97 vs 1.27),
  while 25%+4-bit does not beat plain 4-bit at 4.0 bits/w (21.99 vs 21.94). Q->P at 25%+3 equals 3-bit exactly
  (identical perplexity and all 210 layer errors). P->Q is the best order in every complete cell by WikiText-2,
  C4 and KL. Caveat: under entropy coding of the stored symbols, plain 3-bit costs less than 50%+4-bit.
- **E2 complete for all 3 models:** sub-additivity of mask-first orders holds with one Hessian in aggregate
  (all 18 cells); neither re-fitting the grid nor the pruning-only mask explains the quantize-first penalty on
  any model; the interaction correlates with input correlation (Spearman 0.41-0.60).
- **Paper:** 6-page draft for angle B + C, 15 verified references, every number a macro.
- **Runtimes:** the session was suspended 14:48-18:00 (usage limit), the tokens expired, the runtimes died at
  about 15:50; all finished units were recovered from the HF results dataset. Restarted at 18:04 on a newer
  Colab image, admitted after a bit-exact check.
- **Next:** E3 135M, E1 360M (A); E1 Qwen, then E4/E3 for Qwen and 360M (B); macros; NEWS every few hours.

## 2026-11-10 11:20 UTC - reproduction and determinism pass; runtimes lost again, keepalive added
- **fp32 reproduction passes:** SmolLM2-360M fp32 on L4 vs the paper's own fp32 re-runs (T4): joint 25%+4
  -0.11%, P->Q 25%+4 +0.37%, Q->P 25%+4 +0.37% (`results/v2/REPRO32/`). The SmolLM2-135M gaps vs 16-bit T4
  runs are precision + hardware: the same code in 16-bit on L4 differs from the 16-bit T4 runs by +1.5%
  (4-bit) and +4.6% (50%+4 P->Q) (`results/v2/REPRO16/`). Gaps of a few % between orders are therefore not
  meaningful across hardware; within the locked type they are exact.
- **Determinism passes:** SmolLM2-135M joint 50%+4 computed on two different L4 machines: identical WikiText-2
  NLL and identical per-layer errors (`results/v2/DET/` vs `results/v2/E1/`).
- **Runtimes:** `esann-e1-b2` (07:35) and `esann-e1-a3` (about 08:05) died like the earlier ones; units they
  finished after the watcher stopped were recovered from the HF results dataset. New runtimes since 11:06 with
  a kernel-channel keepalive (`scripts/colab_keepalive.py`).
- **W&B:** 11 runs synced to the public project.
- **Next:** SmolLM2-135M E1 rest (about 14:30), E4 and E3 on 135M (about 16:30), angle decision draft.

## 2026-11-10 07:35 UTC - reproduction investigation, E2 for both SmolLM2 models
- **Reproduction:** dense perplexity matches the published runs to 1e-5, but compressed states do not hold
  0.5% across GPU types: SmolLM2-135M 4-bit +0.74% (fp32 L4 vs 16-bit T4), 50%+4 P->Q +2.62%; Qwen2.5-0.5B
  4-bit +0.50%. The same code in 16-bit on the L4 gives +1.50% for SmolLM2-135M 4-bit, so neither precision
  nor the pipeline explains it: kernel-level differences flip greedy rounding/mask decisions, which cascade.
  E1 proceeds (DECISIONS.md #22); fp32 SmolLM2-360M vs the paper's own fp32 re-runs still running.
- **E2 complete for SmolLM2-135M and -360M** (`reports/analysis/summary.md`, `scripts/make_tables.py`): P->Q and
  joint are sub-additive in all 6 cells on both models with one Hessian. Neither the re-fitted grid (always
  worse) nor the pruning-only mask (-19% to +39%) removes most of the quantize-first penalty on either
  model, so angle A's evidence bar is not met; the default angle (B + C, honest storage frontier) stands.
- **Next:** E1 SmolLM2-135M rest and 360M on A3; E1 Qwen then E4 and E3 on B2; E2 Qwen on A3.

## 2026-11-10 06:55 UTC - fixes, first results
- **Colab reclaims runtimes whose kernel is idle** (about 20 min after the last code execution), even with GPU
  jobs running in the background: both runtimes died at about 06:33, and the pilot overnight the same way (so
  the pilot cost about 1-2 CU, not 14). Finished units were safe (resume-safe runners, results mirrored every
  90 s); two units in progress were lost. Chains now run from a kernel cell that blocks until they finish.
- **lm-eval ran out of memory at batch 64** (fp32 logits of long BoolQ/SciQ passages). Now two fixed batch
  groups: SmolLM2 48 (short tasks) / 16 (BoolQ, SciQ, LAMBADA); Qwen 24 / 8. Results write to `/content/out`
  on the runtimes so code updates never touch live result files.
- **Dense fp32 vs published 16-bit WikiText-2 perplexity:** 15.41532 vs 15.41535 (SmolLM2-135M), 13.06988 vs
  13.07026 (Qwen2.5-0.5B) (`reports/analysis/`).
- **E2 on SmolLM2-135M** (same Hessian, every 2nd block, 105 layers; `results/v2/E2/SmolLM2-135M.jsonl`,
  `scripts/make_tables.py`): P->Q is sub-additive in all 6 cells (-4.1% to -12.2%), joint in all 6 (-6.4% to
  -18.1%). Re-fitting the grid does not reduce the quantize-first penalty (it makes it 3-14% worse); reusing
  the pruning-only mask removes 6-39%. So neither the stale grid (the paper's reading) nor mask selection
  (Harma et al.'s) explains most of it on this model: angle A's 60% bar is not met here.
- **E2, correlation:** each layer's P->Q interaction correlates with the mean |corr| of its Hessian (Spearman
  0.48-0.55 on 135M, 0.54-0.61 on partial 360M): the review's synthetic prediction holds on real layers. With
  one Hessian, block 0 of SmolLM2-360M is super-additive for P->Q in all 6 cells while deeper blocks are
  sub-additive (partial data).
- **Literature:** 30 BibTeX entries verified (`paper/refs.bib`, `reports/literature.md`). Caveats for the text:
  Harma et al. measured the small SparseGPT order effect at 6-8 bits only; Kim et al.'s "often marginally
  positive" is in their vision section; SparseGPT shows 50%+4-bit vs 3-bit only from 2.7B up.
- **LaTeX:** the draft is re-created exactly in the ESANN template (`paper/v1/`, same page breaks).
- **Next:** reproduction gate (3 SmolLM2-135M units), then the rest of E1; E2 on 360M and Qwen; CU plan for
  E3-E6 from the measured timings.

## 2026-11-10 06:00 UTC - pilot done, E1 and E2 running
- The session was suspended 23:51-05:49 UTC (usage limit). The pilot runtime had finished its checks at about
  23:56 and was idle until Colab reclaimed it (time unknown; Colab's usual idle timeout is about 90 min, so about
  4 CU; worst case the full 6.2 h, about 14 CU). Logged in logs/runtimes.jsonl and logs/budget.md.
- GPU-side checks, all PASS (logs/env_check.txt): L4 in passthrough mode, 22 GiB, 12.8 TFLOPS fp32 with TF32
  off; config.json for the 6 models; C4 and WikiText-2; all 9 task splits have the expected sizes (ARC-E 2376,
  PIQA 1838, HellaSwag 10042, ARC-C 1172, WinoGrande 1267, OBQA 500, BoolQ 3270, SciQ 1000, LAMBADA 5153);
  lm-eval with per-item logs; the review's smoke tests; 19 pytest tests on CPU + CUDA.
- Every compressed state is now exported exactly (codes + scales + mask, bit-exact round trip tested) and uploaded
  to the HF model repo; every run will be synced to a public W&B project. Both at your request.
- Running now: runtime A = E1 SmolLM2-135M (the first 3 units feed an automatic reproduction gate vs the published
  runs, 0.5% tolerance; the chain stops if it fails), then E1 SmolLM2-360M; E2 same-Hessian ablation (every 2nd
  block, 6 cells, 3 models) alongside. Runtime B = determinism re-run on a second machine, then E1 Qwen2.5-0.5B.
- Next: gate result (about 06:30), first timings -> CU plan for E3-E6; analysis subagent on the first results.

## 2026-11-09 23:45 UTC - phase 0, container side
- Finished: token, HF identity (fine-grained, write scope), private dataset `martincousseau/esann2027-v2-results`
  (upload and delete OK), review code unpacked, baseline re-analysis `max |recomputed - printed| = 0.000`
  (logs/env_check.txt, logs/reanalysis_v1_output.txt).
- Drive folder has only "Review" and "Code": no Explainer. You chose to proceed.
- Built and tested on CPU: `src_v2/pipeline.py`, `scripts/run_grid.py` (resume-safe runner), 14 unit tests,
  the review's smoke tests (all PASSED), and an end-to-end CPU smoke run of the runner.
- Next: L4 pilot runtime: bootstrap, runtime checks, timing, determinism and reproduction checks.
- CU: 0 spent, 100 left; no runtime alive.

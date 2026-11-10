# NEWS (newest first)

## Read this first
- **Angle:** B + C (honest storage frontier with the mask counted, plus free zeros), provisional
  (DECISIONS.md #25). The 6-page LaTeX draft exists (`paper/main.tex`, `paper/main.pdf`); its numbers are
  generated macros, and 29 new macros are being added.
- **Status (21:00 UTC):** SmolLM2-135M complete (E1 25, E2, E3 16, E4 5): angle B + C confirmed on it. E1 Qwen
  19/25, E1 SmolLM2-360M 5/25. Queued: A: 360M E1 (to ~02:30), then E6 SmolLM2-1.7B (7 states, to ~06:00);
  B: Qwen E1, E4 Qwen/360M, E3 360M/Qwen (to ~06:00), then E5 SmolLM2-360M seeds (to ~08:30).
- **Blockers:** none. Risk: this session was suspended twice by usage limits; each time the runtimes died about
  an hour later. A runtime-side keepalive is now running too (DECISIONS.md #27); finished units are never lost.
- **Decisions waiting on you:** none urgent (QUESTIONS.md). The draft is authored "A. Yaacoub" (from README.md),
  affiliation to be confirmed.
- **GPU type locked:** NVIDIA L4, fp32, TF32 off; images `..._20261106-060046_RC00` and `..._20261109-060054_RC00`
  (the second admitted after a bit-exact check, #26).
- **CU (estimate; check the Colab website):** about 39 spent, about 61 left; the armed queues need about 53
  (total about 92); 2 runtimes alive (about 4.5 CU/h). Runtimes free themselves when their queues end.
- **Links:** PR https://github.com/martin-cousseau/esann2027-compose/pull/1 ; W&B (public)
  https://wandb.ai/martin-css/esann2027-compose-v2 ; models (private until the end)
  https://huggingface.co/martincousseau/esann2027-compose-v2

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

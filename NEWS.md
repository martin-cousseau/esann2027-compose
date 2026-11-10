# NEWS (newest first)

## Read this first
- **Angle:** not chosen yet (needs E1-E4). Early E2 evidence weakens angle A on SmolLM2-135M (see below).
- **Status:** E1 running on two L4 runtimes (`esann-e1-a3`, `esann-e1-b2`); E2 done for both SmolLM2 models,
  Qwen running. Reproduction gate re-read after an investigation (see below): proceeding. Runtime A was lost
  twice (idle reclaim, then an unexplained loss); every finished unit was kept.
- **Blockers:** none. Colab allows 2 GPU runtimes at once here.
- **Decisions waiting on you:** none urgent (QUESTIONS.md).
- **GPU type locked:** NVIDIA L4, image `release-colab-external-images_20261106-060046_RC00`, fp32, TF32 off.
- **CU (estimate; check the Colab website):** about 7 spent (pilot about 1-2, four runtimes about 0.7 h each,
  two alive since 06:41), about 93 left; 2 runtimes alive, about 4.5 CU/h together.
- **Links:** PR https://github.com/martin-cousseau/esann2027-compose/pull/1 ; W&B (public)
  https://wandb.ai/martin-css/esann2027-compose-v2 ; models (private until the end)
  https://huggingface.co/martincousseau/esann2027-compose-v2

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

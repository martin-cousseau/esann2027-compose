# NEWS (newest first)

## Read this first
- **Angle:** not chosen yet (needs E1-E4 results).
- **Status:** phase 0 passed (GPU checks all PASS; the Explainer is missing and you chose to proceed). E1 started
  at 05:56 UTC on two L4 runtimes; E2 runs alongside E1 on runtime A. Results mirror into `results/v2/`.
- **Blockers:** none. Colab allows only 2 GPU runtimes at once on this plan (a 3rd was refused).
- **Decisions waiting on you:** none urgent (QUESTIONS.md: the Explainer and the original LaTeX source, if you have them).
- **GPU type locked:** NVIDIA L4, image `release-colab-external-images_20261106-060046_RC00`, fp32, TF32 off.
- **CU (estimate, check the real balance on the Colab website):** about 4 spent (pilot; upper bound 14, see below),
  about 96 left; 2 runtimes alive (`esann-e1-a`, `esann-e1-b`), burning about 4.5 CU/h together.
- **Links:** draft PR https://github.com/martin-cousseau/esann2027-compose/pull/1 ; models (private until the end,
  then public) https://huggingface.co/martincousseau/esann2027-compose-v2 ; W&B project being set up (public).

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

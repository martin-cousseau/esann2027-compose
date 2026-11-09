# NEWS (newest first)

## Read this first
- **Angle:** not chosen yet (needs E1-E4).
- **Status:** phase 0 container checks passed except the missing Explainer (you chose to proceed). Colab connector OK (Pro tier).
- **Blockers:** none.
- **Decisions waiting on you:** see QUESTIONS.md (Explainer, original LaTeX source).
- **GPU type:** L4 planned (pilot pending); G4 and A100 cost 3-5x more per hour (logs/budget.md).
- **CU:** 0 spent (estimate), 100 left. Check the real balance on the Colab website: the connector does not report it.

## 2026-11-09 23:45 UTC - phase 0, container side
- Finished: token, HF identity (fine-grained, write scope), private dataset `martincousseau/esann2027-v2-results`
  (upload and delete OK), review code unpacked, baseline re-analysis `max |recomputed - printed| = 0.000`
  (logs/env_check.txt, logs/reanalysis_v1_output.txt).
- Drive folder has only "Review" and "Code": no Explainer. You chose to proceed.
- Built and tested on CPU: `src_v2/pipeline.py`, `scripts/run_grid.py` (resume-safe runner), 14 unit tests,
  the review's smoke tests (all PASSED), and an end-to-end CPU smoke run of the runner.
- Next: L4 pilot runtime: bootstrap, runtime checks, timing, determinism and reproduction checks.
- CU: 0 spent, 100 left; no runtime alive.

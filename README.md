# Do Pruning and Quantization Compose? — code and results

Code, raw results and paper source for:

> A. Yaacoub. *Do Pruning and Quantization Compose? Order and Interaction in Post-Training Compression of Language Models.* Submitted to ESANN 2027.

It extends: S. Laborde, M. Cousseau, A. Yaacoub, L. Prevost. *Semantic Retention and Extreme Compression in LLMs: Can We Have Both?* IJCNN 2025 (arXiv:2505.07289).

## v2 study (November 2026, branch `claude/busy-hopper-ghtynk`)

A complete fp32 re-run of the study on one GPU type, answering a review of the draft, with a revised paper
(`paper/main.tex`, "Counting the Mask: When Does Pruning Pay in Quantized Small Language Models?"). Everything
below is on this branch; the original draft and its files are unchanged (`paper/v1/` is an exact rebuild of it).

| What | Where |
|---|---|
| Compressed model states (exact codes, scales, masks) and a standalone loader | https://huggingface.co/martincousseau/esann2027-compose-v2 |
| Result lines and per-item benchmark logs, as uploaded by the GPU runners | https://huggingface.co/datasets/martincousseau/esann2027-v2-results (and `results/v2/` here) |
| One Weights & Biases run per model state (metrics, layer diagnostics, links back here) | https://wandb.ai/martin-css/esann2027-compose-v2 |
| What was found, what was decided, the session log | `FINDINGS.md`, `DECISIONS.md`, `NEWS.md`, `QUESTIONS.md` |

Layout of the v2 files:

```
src_v2/pipeline.py         v2 library on top of src/core.py (fp32, strict overflow check, metrics, storage, export)
scripts/run_grid.py        resume-safe runner: one result line per model state (results/v2/<family>/<model>.jsonl)
scripts/make_queues.py     the study plan (families E1, E3-E6) and the queue files in colab/queues/
scripts/run_e2.py          E2, the same-Hessian layer ablation (review code, ablation/)
scripts/make_tables.py     CPU only: every table, macro, figure and report from the result files
scripts/analysis/          the analyses (E1 bootstrap and McNemar, Q1-Q3, frontier, E2, E3, E5, E6, E7, E8)
colab/                     runtime chains, locked environment fingerprint, keepalive
paper/generated/           macros.tex (every number the paper quotes), MACROS.md (the source of each), tables
reports/                   analysis outputs, literature check, LaTeX notes, audit
release/                   model loader and the generated model card of the Hugging Face repository
tests/                     CPU tests of the pipeline and of the analysis
```

Reproduce the analysis (CPU, a few minutes): `pip install -r requirements-analysis.txt`, then
`python scripts/make_tables.py`. GPU runs (NVIDIA L4, fp32, TF32 off; the environment fingerprint is checked
against `colab/locked_fingerprint.json`): `python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-135M
--queue colab/queues/E1__SmolLM2-135M.json --locked colab/locked_fingerprint.json`.

The text below describes the original (v1) study.

## What is in this repository

```
notebooks/
  esann2027_compose.ipynb    main study (Colab, free T4 GPU): pilot mode + the 93 runs of the paper
  esann2027_fix_fp32.ipynb   32-bit re-run of the 6 SmolLM2-360M runs that overflowed in 16-bit (+ 3 controls)
src/
  core.py                    the compression code used in both notebooks (same as notebook cell 3)
results/
  results_full.jsonl         93 runs, one JSON object per line (16-bit)
  results_fix.jsonl          10 runs in 32-bit (6 replacements, dense model, 3 controls)
  merged_results.json        the 93 runs used in the paper (fix applied); written by the script below
analysis/
  reproduce_paper.py         recomputes every number, Table 1, Table 2 and Fig. 1 from the raw files
paper/
  main.tex, refs.bib, esannV2.cls, fig1_interaction.pdf, main.pdf
```

## Method in one paragraph

Pruning (SparseGPT-style) and quantization (GPTQ-style, min–max asymmetric grid, groups of 128 columns) are implemented
from scratch as options of one column-wise Optimal-Brain-Surgeon sweep. Three combinations reach the same storage cost:
**joint** (one sweep; mask chosen first, grid fitted on the kept weights), **P→Q** (prune, then quantize with the mask fixed)
and **Q→P** (quantize, then prune and project back onto the fixed grid). Every linear layer of every decoder block is
compressed sequentially, calibrated on 128 × 2048 C4 tokens.

## Experimental grid

* Models: SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B.
* Per model (25 runs): dense; pruning only at 25 / 33.3 / 50 %; quantization only at 8 / 4 / 3 bits;
  {25, 33.3, 50 %} × {4, 3 bits} × {joint, P→Q, Q→P}.
* SmolLM2-135M: 18 extra runs with calibration seeds 1 and 2 (25 % and 50 % at 4 bits, all orders, plus 4-bit alone).
* Metrics: WikiText-2 perplexity (2048-token windows); zero-shot ARC-Easy, PIQA, HellaSwag (first 500 items, lm-eval);
  per-layer diagnostics (mask ratio, zeros, levels per group, normalized layer error `rel_err`).

## Result file format (`results/*.jsonl`)

One configuration per line: `model, order, sparsity, bits, seed, tcr, ppl, tasks` (all lm-eval accuracy metrics),
`main` (accuracy used in the paper: acc for ARC-Easy and PIQA, acc_norm for HellaSwag), `mean_rel_err`,
`diags` (one entry per linear layer: shape, mask_frac, zero_frac, mask_kept, max_levels, rel_err),
timings and run settings.

## Reproduce the paper's numbers (no GPU needed)

```bash
pip install numpy matplotlib
python analysis/reproduce_paper.py
```

## Re-run the experiments

Open `notebooks/esann2027_compose.ipynb` in Google Colab, select a T4 GPU, run all.
`MODE = "pilot"` runs a validation on one small model with a PASS/FAIL report; `MODE = "full"` runs the 93 runs
(≈15 GPU-hours on a T4; results are saved to Google Drive after every run and the notebook resumes where it stopped).

Notes:
* The six SmolLM2-360M runs listed in `analysis/reproduce_paper.py` (`AFFECTED`) overflowed in 16-bit and were
  re-run in 32-bit with `esann2027_fix_fp32.ipynb`. Three 16-bit controls re-run in 32-bit differ by at most 0.4 % in perplexity.
* After the overflow, a safeguard was added to `obs_sweep` (non-finite Hessian entries set to zero, diagonal fallback if
  Cholesky still fails, with a printed WARNING). It only acts where the previous code crashed, so all other runs are unchanged.

## License

Code: MIT. Results: CC BY 4.0.

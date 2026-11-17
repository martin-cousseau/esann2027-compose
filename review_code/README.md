# Review code: "Do Pruning and Quantization Compose?" (ESANN 2027 draft)

Scripts that accompany the review of the draft and its repository
(https://github.com/antounyaacoub/esann2027-compose). Everything builds on the paper's own
`src/core.py`; nothing in the compression algorithms is re-implemented.

```
ablation/
  arms.py                    the seven compression arms (3 paper orders + 4 ablation arms) on one shared Hessian
  ablation_same_hessian.py   layer-level order and mechanism ablation on real models (GPU; --smoke-test on CPU)
  extra_orders.py            ablation arms for full sequential runs, KL to dense, per-item results, flips,
                             McNemar test, mask-aware bits per weight, strict overflow check, run metadata
  test_extra_orders.py       CPU checks for extra_orders.py on a tiny random model
  ablation_colab.ipynb       self-contained Colab notebook (T4) that runs the ablation on the three models
reanalysis/
  analyze_results.py         re-analysis of the 93 published runs (standard library only, seconds)
  synthetic_layers.py        decomposition identity + all arms on synthetic layers (CPU, about a minute)
outputs/
  analyze_results_output.txt, synthetic_layers_output.txt, smoke_test_output.txt,
  test_extra_orders_output.txt, storage_points.csv
```

## Re-analysis of the published results (no GPU)

```bash
git clone https://github.com/antounyaacoub/esann2027-compose
python reanalysis/analyze_results.py esann2027-compose --csv outputs/storage_points.csv
pip install torch
python reanalysis/synthetic_layers.py esann2027-compose
```

## Ablation on real models (free Colab T4)

Open `ablation/ablation_colab.ipynb` in Colab, select a T4 GPU, run all. Or, from a shell:

```bash
python ablation/ablation_same_hessian.py --repo esann2027-compose --smoke-test        # CPU, 15 s
python ablation/ablation_same_hessian.py --repo esann2027-compose \
       --model HuggingFaceTB/SmolLM2-135M --cells 1/4:4,1/2:4,1/3:3 --blocks stride4 --out ablation.jsonl
python ablation/ablation_same_hessian.py --repo esann2027-compose --model HuggingFaceTB/SmolLM2-135M \
       --out ablation.jsonl --summary-only
```

Estimated T4 time with the defaults (3 cells, every 4th block, 29 sweeps per layer): about 20 min for
SmolLM2-135M and 40 min each for SmolLM2-360M and Qwen2.5-0.5B. `--blocks all` takes about four times
longer. Results are appended after every layer, so a disconnect loses nothing.

## Full sequential runs with the ablation arms, KL and flips

In the paper's notebook, after cells 3 and 4:

```python
%run -i extra_orders.py
install_extra_orders(globals())    # compress_model now accepts q_then_p_refit, q_then_p_pmask, ...
make_safeguard_strict(globals())   # raise on non-finite Hessians instead of degrading silently
```

Then add rows such as `("q_then_p_refit", 0.25, 3, 0)` to `GRID`, and in `run_config` store
`kl_to_dense(...)`, the per-item dict from `lmeval_items(...)` and `run_metadata(...)` with each result.

## What was tested here

The review environment had no access to Hugging Face, so no real model was run. Tested on CPU:
`arms.py` reproduces the paper's `compress_weight` exactly for joint, P->Q and Q->P; the block-0
errors of `ablation_same_hessian.py` equal the paper's own `compress_model` errors exactly; the
ablation, KL, flips, McNemar, storage and metadata functions run end to end on a tiny random Llama
model (outputs in `outputs/`). The numbers in those smoke tests come from random weights and say
nothing about the paper's claims.

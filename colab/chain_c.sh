#!/bin/bash
# Runtime C: E2 same-Hessian ablation, every 2nd block (stride2), six cells, the three core models.
cd /content/work
for M in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B; do
  python scripts/run_e2.py --model $M --blocks stride2 --locked colab/locked_fingerprint.json --runtime-name esann-e2-a --session e2-main
done
echo "CHAIN C DONE"

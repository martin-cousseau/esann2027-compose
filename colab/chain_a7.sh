#!/bin/bash
# Runtime A: rest of E3 SmolLM2-135M, then E1 SmolLM2-360M.
cd /content/work
export ESANN_OUT=/content/out
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-a5 --session e1-main"
python scripts/run_grid.py --family E3 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E3__SmolLM2-135M.json --eval-batch 16 $COMMON
python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E1__SmolLM2-360M.json --eval-batch 16 $COMMON
echo "CHAIN A7 DONE"

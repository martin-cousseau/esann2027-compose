#!/bin/bash
# Runtime A: E1 SmolLM2-135M (first 4 units = reproduction units) -> gate -> rest of E1 135M -> E1 360M
cd /content/work
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-e1-a --session e1-main"
python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E1__SmolLM2-135M.json --max-units 3 --eval-batch 64 $COMMON
python colab/gate.py || { echo "GATE FAILED: chain stopped"; exit 1; }
python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E1__SmolLM2-135M.json --eval-batch 64 $COMMON
python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E1__SmolLM2-360M.json --eval-batch 32 $COMMON
echo "CHAIN A DONE"

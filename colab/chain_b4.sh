#!/bin/bash
# Runtime B: rest of E1 Qwen2.5-0.5B, then E4 (Qwen, SmolLM2-360M), then E3 (SmolLM2-360M, Qwen).
cd /content/work
export ESANN_OUT=/content/out
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-b4 --session e1-main"
python scripts/run_grid.py --family E1 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E1__Qwen2.5-0.5B.json --eval-batch 8 $COMMON
python scripts/run_grid.py --family E4 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E4__Qwen2.5-0.5B.json --eval-batch 8 $COMMON
python scripts/run_grid.py --family E4 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E4__SmolLM2-360M.json --eval-batch 16 $COMMON
python scripts/run_grid.py --family E3 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E3__SmolLM2-360M.json --eval-batch 16 $COMMON
python scripts/run_grid.py --family E3 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E3__Qwen2.5-0.5B.json --eval-batch 8 $COMMON
echo "CHAIN B4 DONE"

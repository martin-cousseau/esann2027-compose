#!/bin/bash
# Runtime B: determinism re-run of SmolLM2-135M joint 50%+4 on a second machine (perplexity and layer errors only),
# then E1 Qwen2.5-0.5B.
cd /content/work
COMMON="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-e1-b --session e1-main"
python scripts/run_grid.py --family DET --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/DET__SmolLM2-135M.json --eval-batch 64 $COMMON
python scripts/run_grid.py --family E1 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E1__Qwen2.5-0.5B.json --eval-batch 16 --export-models --hf-models-repo martincousseau/esann2027-compose-v2 $COMMON
echo "CHAIN B DONE"

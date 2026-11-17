#!/bin/bash
# Runtime B: E1 Qwen2.5-0.5B (rest), then E4 frontier points (full metrics), then E3 mechanism arms (ppl/KL).
cd /content/work
export ESANN_OUT=/content/out
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-b3 --session e1-main"
python scripts/run_grid.py --family E1 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E1__Qwen2.5-0.5B.json --eval-batch 8 $COMMON
python scripts/run_grid.py --family E4 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E4__Qwen2.5-0.5B.json --eval-batch 8 $COMMON
python scripts/run_grid.py --family E4 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E4__SmolLM2-135M.json --eval-batch 16 $COMMON
python scripts/run_grid.py --family E4 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E4__SmolLM2-360M.json --eval-batch 16 $COMMON
for M in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B; do
  S=$(basename $M)
  python scripts/run_grid.py --family E3 --model $M --queue colab/queues/E3__$S.json --eval-batch 8 $COMMON
done
echo "CHAIN B3 DONE"

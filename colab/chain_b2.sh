#!/bin/bash
# Runtime B, after chain_b (E1 Qwen2.5-0.5B): E4 frontier points (full metrics), then E3 mechanism arms
# (perplexity, C4 and KL). Waits for chain_b's process to end first.
cd /content/work
export ESANN_OUT=/content/out
WAIT_PID=$(python -c "import json; print(json.load(open('/content/out/logs/chain_pids.json')).get('chain_b', 0))")
while [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-e1-b2 --session e1-main"
python scripts/run_grid.py --family E4 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E4__Qwen2.5-0.5B.json --eval-batch 8 $COMMON
python scripts/run_grid.py --family E4 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E4__SmolLM2-135M.json --eval-batch 16 $COMMON
python scripts/run_grid.py --family E4 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E4__SmolLM2-360M.json --eval-batch 16 $COMMON
for M in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B; do
  S=$(basename $M)
  python scripts/run_grid.py --family E3 --model $M --queue colab/queues/E3__$S.json --eval-batch 8 $COMMON
done
echo "CHAIN B2 DONE"

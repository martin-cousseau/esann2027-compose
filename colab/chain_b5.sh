#!/bin/bash
# Runtime B, after chain_b4: E5 reduced (SmolLM2-360M, calibration seeds 1 and 2, 4-bit alone and every 4-bit
# cell in the three orders; perplexity, C4, KL).
cd /content/work
export ESANN_OUT=/content/out
WAIT_PID=$(python -c "import json; print(json.load(open('/content/out/logs/chain_pids.json')).get('chain_b4', 0))")
while [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-b4 --session e5-main"
python scripts/run_grid.py --family E5 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E5__SmolLM2-360M.json --eval-batch 16 $COMMON
echo "CHAIN B5 DONE"

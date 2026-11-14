#!/bin/bash
# Runtime A, after the phase-0 gate failed (fp32 vs published 16-bit): reproduction investigation, perplexity only.
cd /content/work
export ESANN_OUT=/content/out
COMMON="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-e1-a2 --session e1-main"
python scripts/run_grid.py --family REPRO16 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/REPRO16__SmolLM2-135M.json --dtype float16 --no-dense-copy $COMMON
python colab/gate2.py --part fp16
python scripts/run_grid.py --family REPRO32 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/REPRO32__SmolLM2-360M.json --no-dense-copy $COMMON
python colab/gate2.py --part fp32
echo "CHAIN A3 DONE"

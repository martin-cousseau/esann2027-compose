#!/bin/bash
# Runtime A (2026-11-14, DECISIONS.md #46): resumes chain_a12 after the second stop. The orchestrator pushes the
# existing lines of E6B/SmolLM2-1.7B and E6/Qwen2.5-1.5B here first (one writer per result file), so finished states
# are skipped. Then the E6B re-runs of Qwen2.5-1.5B, Qwen2.5-1.5B to the full grid, and Qwen2.5-3B.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-a8 || exit 1
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a8"
X="--export-models --hf-models-repo martincousseau/esann2027-compose-v2"
python scripts/run_grid.py --family E6B --model HuggingFaceTB/SmolLM2-1.7B --queue colab/queues/E6B__SmolLM2-1.7B.json --eval-batch 8 $H --session e6b-smol
python scripts/run_grid.py --family E6B --model Qwen/Qwen2.5-1.5B --queue colab/queues/E6B__Qwen2.5-1.5B.json --eval-batch 8 $H --session e6b-qwen
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E6__Qwen2.5-1.5B.json --eval-batch 8 $H $X --session e6-qwen
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B.json --eval-batch 4 --no-dense-copy $H --session e6-qwen3b
echo "CHAIN A13 DONE"

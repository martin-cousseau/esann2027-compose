#!/bin/bash
# Runtime B (2026-11-13, DECISIONS.md #44): the 4- and 3-bit seeds of Qwen2.5-0.5B, then SmolLM2-1.7B to the full
# grid and OLMo-2-1B (their I_Sr states with benchmarks). Each result file has one writer runtime: the orchestrator
# pushes the existing lines of E5/Qwen2.5-0.5B and E6/SmolLM2-1.7B here before the start.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-b6 || exit 1
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b6"
X="--export-models --hf-models-repo martincousseau/esann2027-compose-v2"
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E5__Qwen2.5-0.5B.json --eval-batch 8 $H $X --session e5-qwen
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E5__Qwen2.5-0.5B__b3.json --eval-batch 8 $H $X --session e5-qwen3
python scripts/run_grid.py --family E6 --model HuggingFaceTB/SmolLM2-1.7B --queue colab/queues/E6__SmolLM2-1.7B.json --eval-batch 8 $H $X --session e6-smol
python scripts/run_grid.py --family E6 --model allenai/OLMo-2-0425-1B --queue colab/queues/E6__OLMo-2-0425-1B.json --eval-batch 8 $H $X --session e6-olmo
echo "CHAIN B9 DONE"

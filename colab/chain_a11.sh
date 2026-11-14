#!/bin/bash
# Runtime A (2026-11-13, DECISIONS.md #44): benchmarks for Table 2 at 1-2B (E6B re-runs of E6 states that ran without
# them, SmolLM2-1.7B's weaker-first cell), then Qwen2.5-1.5B to the full grid (its I_Sr states with benchmarks), then
# the 3-bit seeds of SmolLM2-360M and the seeds of SmolLM2-135M. Each result file has one writer runtime: the
# orchestrator pushes the existing lines of E6/Qwen2.5-1.5B and E5/SmolLM2-360M here before the start.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-a7 || exit 1
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a7"
X="--export-models --hf-models-repo martincousseau/esann2027-compose-v2"
python scripts/run_grid.py --family E6B --model HuggingFaceTB/SmolLM2-1.7B --queue colab/queues/E6B__SmolLM2-1.7B.json --eval-batch 8 $H --session e6b-smol
python scripts/run_grid.py --family E6B --model Qwen/Qwen2.5-1.5B --queue colab/queues/E6B__Qwen2.5-1.5B.json --eval-batch 8 $H --session e6b-qwen
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E6__Qwen2.5-1.5B.json --eval-batch 8 $H $X --session e6-qwen
python scripts/run_grid.py --family E5 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E5__SmolLM2-360M.json --eval-batch 16 $H $X --session e5-360m
python scripts/run_grid.py --family E5 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E5__SmolLM2-135M.json --eval-batch 16 $H $X --session e5-135m
echo "CHAIN A11 DONE"

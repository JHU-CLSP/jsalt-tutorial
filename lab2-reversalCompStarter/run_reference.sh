#!/bin/bash
# Reference pipeline: ~45 minutes on a DGX Spark (GB10).
# SFT on 100 traces -> evaluate -> GRPO 300 steps -> evaluate.
set -e

echo "=== SFT (100 traces, rank-8 LoRA) ==="
python train_sft.py

echo "=== SFT eval (greedy) ==="
python evaluate.py --adapter-path models/sft-lora --out preds_sft.jsonl

echo "=== GRPO (300 steps) ==="
python train_grpo.py --adapter-path models/sft-lora

echo "=== GRPO eval (on-policy, temp 0.7) ==="
python evaluate.py --adapter-path models/grpo-lora --temp 0.7 --out preds_grpo.jsonl

echo "=== done ==="

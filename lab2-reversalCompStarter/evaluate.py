"""Evaluate a model on the word-reversal task.

Decoding via mlx-lm continuous batching. Two metrics:
  - exact-match %: parsed <answer> equals the reversed word (case/whitespace
    insensitive). Responses with no <answer> tag score 0.
  - avg Levenshtein distance: between parsed answer and target, computed over
    responses that have an <answer> tag (tag rate is reported alongside).

Evaluation protocol: SFT/base models decode greedily (default). GRPO-trained
policies were optimized as temperature samplers — evaluate them at their
training temperature (--temp 0.7) for an on-policy measurement.

Usage:
    python evaluate.py --adapter-path models/sft-lora --out preds_sft.jsonl
    python evaluate.py --adapter-path models/grpo-lora --temp 0.7 --out preds_grpo.jsonl
"""

import os

# CUDA graphs off by default — the graph-capture decode path crashes
# intermittently on some discrete GPUs; export MLX_USE_CUDA_GRAPHS=1 for
# a modest speedup on unified-memory machines
os.environ.setdefault("MLX_USE_CUDA_GRAPHS", "0")

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from task import levenshtein, make_prompt, parse_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA adapter dir to load on top of --model")
    ap.add_argument("--words", default="data/eval_words.txt")
    ap.add_argument("--out", default=None, help="write per-word predictions jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=448)
    ap.add_argument("--completion-batch-size", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature; 0 = greedy. Evaluate GRPO "
                    "policies at their training temperature (on-policy)")
    ap.add_argument("--top-p", type=float, default=0.0, help="0 = off")
    ap.add_argument("--top-k", type=int, default=0, help="0 = off")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mx.random.seed(args.seed)

    words = Path(args.words).read_text().split()
    if args.limit:
        words = words[: args.limit]

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": make_prompt(w)}],
            add_generation_prompt=True,
        )
        for w in words
    ]

    t = time.time()
    resp = batch_generate(
        model,
        tokenizer,
        prompts,
        max_tokens=args.max_tokens,
        completion_batch_size=args.completion_batch_size,
        sampler=make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k)
        if args.temp > 0
        else None,
    )
    dt = time.time() - t

    n = len(words)
    tagged = exact = lev_sum = out_toks = 0
    rows = []
    for word, text in zip(words, resp.texts):
        target = word[::-1]
        parsed = parse_answer(text)
        row = {"word": word, "target": target, "parsed": parsed}
        out_toks += len(tokenizer.encode(text))
        if parsed is not None:
            tagged += 1
            norm = parsed.lower()
            row["lev"] = levenshtein(norm, target)
            lev_sum += row["lev"]
            if norm == target:
                exact += 1
        row["exact"] = parsed is not None and parsed.lower() == target
        rows.append(row)

    adapter = f" + {args.adapter_path}" if args.adapter_path else ""
    print(f"\nmodel: {args.model}{adapter}  "
          f"({'greedy' if args.temp == 0 else f'temp={args.temp}'})")
    print(f"n={n}  wall={dt:.0f}s  ({out_toks / dt:.0f} tok/s aggregate)")
    print(f"tag rate:    {tagged}/{n} = {100 * tagged / n:.1f}%")
    print(f"exact match: {exact}/{n} = {100 * exact / n:.1f}%")
    if tagged:
        print(f"avg levenshtein (over {tagged} tagged): {lev_sum / tagged:.2f}")

    if args.out:
        with open(args.out, "w") as f:
            for row, text in zip(rows, resp.texts):
                row["response"] = text
                f.write(json.dumps(row) + "\n")
        print(f"predictions -> {args.out}")


if __name__ == "__main__":
    main()

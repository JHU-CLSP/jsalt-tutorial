"""SFT on the 100 synthesized reversal traces.

Masked-loss next-token training: loss is computed on assistant (completion)
tokens only. Defaults reproduce the reference recipe: rank-8 LoRA (scale
32/rank) at lr 1e-4. Pass --lora-rank 0 for a full fine-tune (use a ~10x
lower --lr, e.g. 1e-5, and expect a full model dir instead of an adapter).

Usage:
    python train_sft.py                    # reference recipe
"""

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.utils import hf_repo_to_path, load_config, save

from lora_util import apply_lora, save_adapter_dir


def encode_example(tokenizer, messages, max_seq_len):
    full = tokenizer.apply_chat_template(messages)
    prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True)
    assert full[: len(prompt)] == prompt, "prompt is not a prefix of the full chat"
    assert len(full) <= max_seq_len, f"example too long: {len(full)}"
    return full, len(prompt)


def make_batches(examples, batch_size, pad_id, pad_multiple=32):
    """examples: list of (tokens, prompt_len), length-sorted batches.

    Sequence lengths are padded up to a multiple of pad_multiple so the run
    cycles through a handful of batch shapes instead of one per batch — each
    distinct shape captures (and pins) a fwd+bwd CUDA graph on the CUDA
    backend, so unbounded shape churn leaks memory off MLX's books.
    """
    examples = sorted(examples, key=lambda e: len(e[0]))
    batches = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        max_len = max(len(t) for t, _ in chunk)
        max_len = ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple
        x = mx.array(
            [t + [pad_id] * (max_len - len(t)) for t, _ in chunk]
        )
        # target-space weights: target t (= token t+1) trains iff it is a
        # completion token, i.e. prompt_len <= t+1 < len(tokens)
        w = mx.array(
            [
                [1.0 if p <= t + 1 < len(toks) else 0.0 for t in range(max_len - 1)]
                for toks, p in chunk
            ]
        )
        batches.append((x, w))
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out", default="models/sft-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="tuned for the default LoRA; use ~1e-5 for full FT")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-rank", type=int, default=8,
                    help="0 = full fine-tune; >0 trains a LoRA of this rank "
                    "on all attn/mlp linears and saves an adapter dir")
    ap.add_argument("--lora-scale", type=float, default=None,
                    help="default 32/rank (the alpha=32 convention the 10x-lr "
                    "heuristic assumes)")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    args = ap.parse_args()

    mx.random.seed(args.seed)  # pins the LoRA init
    model, tokenizer = load(args.model)
    config = load_config(hf_repo_to_path(args.model))
    lora_parameters = None
    if args.lora_rank > 0:
        lora_parameters = apply_lora(
            model, args.lora_rank, args.lora_scale, args.lora_dropout
        )

    rows = [json.loads(l) for l in open(args.data)]
    examples = [
        encode_example(tokenizer, r["messages"], args.max_seq_len) for r in rows
    ]
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = next(iter(tokenizer.eos_token_ids))
    batches = make_batches(examples, args.batch_size, pad_id)
    total_steps = len(batches) * args.epochs
    print(f"{len(examples)} examples, {len(batches)} batches/epoch, "
          f"{total_steps} total steps")

    def loss_fn(model, x, w):
        logits = model(x[:, :-1]).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, x[:, 1:], reduction="none")
        return (ce * w).sum() / w.sum()

    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, args.lr, args.warmup),
            optim.cosine_decay(args.lr, total_steps - args.warmup),
        ],
        [args.warmup],
    )
    opt = optim.AdamW(learning_rate=schedule)
    lvg = nn.value_and_grad(model, loss_fn)

    rng = random.Random(args.seed)
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        for i in order:
            x, w = batches[i]
            loss, grads = lvg(model, x, w)
            opt.update(model, grads)
            mx.eval(loss, model.parameters(), opt.state)
            step += 1
            # essential on unified memory: without periodic cache clears the
            # buffer cache balloons (see ml-explore/mlx-lm#986)
            if step % 8 == 0:
                mx.clear_cache()
            if step % 10 == 0 or step == total_steps:
                print(f"epoch {epoch} step {step}/{total_steps} "
                      f"loss {loss.item():.4f} "
                      f"({step / (time.time() - t0):.2f} steps/s, "
                      f"peak {mx.get_peak_memory() / 1e9:.1f} GB)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if lora_parameters is not None:
        save_adapter_dir(out, model, lora_parameters, len(model.layers), args.model)
        print(f"saved adapter dir -> {out}")
    else:
        save(out, args.model, model, tokenizer, config)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()

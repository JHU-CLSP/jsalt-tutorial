"""GRPO trainer for the word-reversal task.

Fully on-policy: rollouts are sampled from the SAME weights being trained
(one MLX model object), so the PPO importance ratio is identically 1 and the
surrogate collapses to plain  -advantage * logp  (REINFORCE with a group
baseline). No reference model / KL term.

Reward    : negative Levenshtein distance between the parsed <answer> and the
            reversed word. No <answer> tag counts as an empty answer, i.e.
            reward = -len(target).
Advantage : reward - mean(group rewards). NO std normalization.
Loss      : -(1/N) sum_g adv_g * sum_t logp_{g,t} / max_tokens   (fixed-
            constant normalization a la Dr.GRPO — NOT per-sequence length,
            which rewards failures for getting longer; computed on the exact
            sampled token ids, eos included). logp is scored at the SAMPLING
            temperature (logits / temp), so the on-policy invariant holds
            for any --temp.

Stability features (each guards against a real failure mode):
  - linear lr warmup (default 50 steps): un-warmed Adam steps at RL onset
    wreck the SFT policy before the second-moment estimates calibrate
  - truncated rollouts count toward the group mean but are masked from the
    gradient: no policy gradient through degenerate repetition loops
  - groups whose rewards are all identical are dropped (zero advantage)

Defaults reproduce the reference run. Usage:
    python train_grpo.py --adapter-path models/sft-lora
"""

import os

# CUDA graphs off by default: the graph cache pins workspace memory as
# generation churns through batch shapes, which silently page-thrashes (and
# appears to hang) on discrete-VRAM GPUs. Costs some throughput on
# unified-memory machines; export MLX_USE_CUDA_GRAPHS=1 to re-enable there.
os.environ.setdefault("MLX_USE_CUDA_GRAPHS", "0")

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import load_adapters
from mlx_lm.utils import hf_repo_to_path, load_config, save_config, save_model

from lora_util import apply_lora, save_adapter_dir
from task import levenshtein, make_prompt, parse_answer

PAD_MULTIPLE = 32


def rollout(model, tokenizer, prompts, max_tokens, sampler, completion_batch_size):
    """Sample one completion per entry of `prompts` (token-id lists).

    Returns (completions, finish_reasons); completions are the exact sampled
    token ids, including the stop token when generation ended with one.
    (Training on re-tokenized text instead of the sampled ids would be
    subtly off-policy whenever the tokenizer round-trip doesn't reproduce
    the sampled tokens.)
    """
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        sampler=sampler,
        completion_batch_size=completion_batch_size,
    )
    uids = gen.insert(prompts, [max_tokens] * len(prompts))
    tokens = {u: [] for u in uids}
    finish = {}
    while responses := gen.next_generated():
        for r in responses:
            tokens[r.uid].append(r.token)
            if r.finish_reason is not None:
                finish[r.uid] = r.finish_reason
    gen.close()
    return [tokens[u] for u in uids], [finish[u] for u in uids]


def reward_fn(completion_text, word):
    target = word[::-1]
    parsed = parse_answer(completion_text)
    answer = "" if parsed is None else parsed.lower()
    return -levenshtein(answer, target), parsed is not None


def sampler_truncation_mask(logits, targets, top_p, top_k):
    """Reproduce mlx_lm.sample_utils truncation on training-pass logits.

    Mirrors make_sampler's chain EXACTLY: the keep-set is computed on the
    temp-1 distribution (the sampler masks logprobs BEFORE temperature is
    applied in categorical_sampling), top_p first (HF-style: keep tokens
    whose ascending-order cumulative prob exceeds 1 - top_p), then top_k on
    the survivors. Returns logits with dropped tokens at -inf, so
    log_softmax(masked / temp) is the exact sampling distribution.

    The sampled target token is force-kept: generation-time logits (bf16,
    incremental KV path) and training-time logits (fp32, batched) can drift
    at the truncation boundary, and a dropped target would mean logp = -inf.
    """
    # the keep-set is part of the SAMPLING procedure — a constant w.r.t. the
    # parameters. Build it entirely on stop_gradient values (index ops like
    # argsort/put_along_axis have no VJP), and let gradient reach the logits
    # only through the final where().
    sg = mx.stop_gradient(logits)
    inf = mx.array(-float("inf"), logits.dtype)
    lp = sg - mx.logsumexp(sg, axis=-1, keepdims=True)
    if 0 < top_p < 1:
        probs = mx.exp(lp)
        sorted_idx = mx.argsort(lp, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
        cum = mx.cumsum(sorted_probs, axis=-1)
        inv = mx.put_along_axis(
            mx.zeros_like(sorted_idx),
            sorted_idx,
            mx.broadcast_to(
                mx.arange(sorted_idx.shape[-1], dtype=sorted_idx.dtype),
                sorted_idx.shape,
            ),
            axis=-1,
        )
        cum = mx.take_along_axis(cum, inv, axis=-1)
        lp = mx.where(cum > 1 - top_p, lp, inf)
    if top_k > 0:
        # threshold at the kth-largest surviving logprob (boundary ties may
        # keep a few extra tokens vs argpartition's arbitrary pick — the
        # resulting distribution difference is negligible)
        thr = mx.min(mx.topk(lp, k=top_k, axis=-1), axis=-1, keepdims=True)
        lp = mx.where(lp >= thr, lp, inf)
    keep = lp != inf
    keep = mx.put_along_axis(
        keep, targets[..., None], mx.array(True), axis=-1
    )
    return mx.where(keep, logits, inf)


def pad_batch(rows, pad_id):
    """rows: list of (tokens, prompt_len). Returns x (B, L), w (B, L-1).

    Lengths are padded to a multiple of PAD_MULTIPLE to bound the number of
    distinct batch shapes (each shape pins a fwd+bwd CUDA graph).
    """
    max_len = max(len(t) for t, _ in rows)
    max_len = ((max_len + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    x = mx.array([t + [pad_id] * (max_len - len(t)) for t, _ in rows])
    w = mx.array(
        [
            [1.0 if p <= t + 1 < len(toks) else 0.0 for t in range(max_len - 1)]
            for toks, p in rows
        ]
    )
    return x, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--words", default="data/rl_words.txt")
    ap.add_argument("--out", default="models/grpo-lora")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.0,
                    help="nucleus sampling for rollouts; 0 = off. Mirrored "
                    "exactly in the logp computation, so on-policy holds")
    ap.add_argument("--top-k", type=int, default=0,
                    help="top-k sampling for rollouts; 0 = off. Mirrored "
                    "exactly in the logp computation, so on-policy holds")
    # must clear the longest LEGITIMATE trace, else correct rollouts get
    # truncated and punished for it
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--train-truncated", action="store_true",
                    help="include truncated rollouts in the gradient (default: "
                    "they count toward the group mean but are masked from the "
                    "loss, so degenerate loops never receive policy gradient)")
    ap.add_argument("--lr", type=float, default=3e-6,
                    help="3e-6 is stable for BOTH full-FT and LoRA here; "
                    "higher LoRA lrs that work for SFT degrade RL")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--microbatch", type=int, default=8, help="sequences per fwd/bwd")
    ap.add_argument("--rollout-batch-size", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter-path", default=None,
                    help="continue training an existing LoRA adapter dir "
                    "(e.g. from train_sft.py) on top of --model")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="0 = train --model as-is; >0 trains a FRESH LoRA of "
                    "this rank on --model (mutually exclusive with "
                    "--adapter-path)")
    ap.add_argument("--lora-scale", type=float, default=None,
                    help="default 32/rank")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()
    assert not (args.adapter_path and args.lora_rank > 0), \
        "--adapter-path and --lora-rank are mutually exclusive"

    run = None
    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
        )

    model, tokenizer = load(args.model)
    model_path = Path(args.model)
    config = load_config(
        model_path if model_path.exists() else hf_repo_to_path(args.model)
    )

    lora_parameters = None
    if args.adapter_path:
        # established mlx_lm practice: freeze the base, then load_adapters
        # re-applies linear_to_lora_layers from the saved adapter_config and
        # loads the weights — only LoRA params remain trainable
        model.freeze()
        load_adapters(model, args.adapter_path)
        lora_parameters = json.load(
            open(Path(args.adapter_path) / "adapter_config.json")
        )["lora_parameters"]
    elif args.lora_rank > 0:
        lora_parameters = apply_lora(
            model, args.lora_rank, args.lora_scale, args.lora_dropout
        )
    if lora_parameters is not None:
        # dropout would inject noise into BOTH rollouts and scoring (MLX
        # modules don't get switched to eval mode here), making sampled and
        # scored distributions diverge — the on-policy contract requires 0
        assert lora_parameters.get("dropout", 0.0) == 0.0, \
            "GRPO requires lora dropout == 0 (on-policy contract)"

    mx.random.seed(args.seed)  # rollout sampling is part of the recipe
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = next(iter(tokenizer.eos_token_ids))

    words = Path(args.words).read_text().split()
    rng = random.Random(args.seed)
    rng.shuffle(words)
    # ON-POLICY CONTRACT: every knob passed to make_sampler MUST be mirrored
    # in loss_fn's scored distribution (see sampler_truncation_mask), or the
    # "no importance ratio" simplification silently becomes a lie. temp,
    # top_p, and top_k are mirrored; do NOT add min_p/xtc/etc. without
    # extending the mirror.
    assert args.temp > 0, "GRPO needs stochastic rollouts (temp > 0)"
    sampler = make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k)

    prompt_ids = {}  # word -> chat-templated token ids

    def get_prompt(word):
        if word not in prompt_ids:
            prompt_ids[word] = tokenizer.apply_chat_template(
                [{"role": "user", "content": make_prompt(word)}],
                add_generation_prompt=True,
            )
        return prompt_ids[word]

    def loss_fn(model, x, w, adv):
        # score the SAME distribution the rollouts were sampled from:
        # truncate (top_p/top_k, computed at temp 1 like the sampler does),
        # THEN apply temperature — otherwise the on-policy (ratio == 1)
        # invariant silently breaks
        logits = model(x[:, :-1]).astype(mx.float32)
        targets = x[:, 1:]
        if 0 < args.top_p < 1 or args.top_k > 0:
            logits = sampler_truncation_mask(logits, targets, args.top_p, args.top_k)
        token_logp = -nn.losses.cross_entropy(
            logits / args.temp, targets, reduction="none"
        )
        # fixed-constant normalization (Dr.GRPO): 1/|o| normalization
        # dilutes per-token punishment for longer failures, teaching the
        # policy to bloat wrong answers until they hit the token cap
        seq_logp = (token_logp * w).sum(axis=1) / args.max_tokens
        # sum here; scaled by 1/N outside so microbatches accumulate into
        # the full-batch mean
        return -(adv * seq_logp).sum()

    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, args.lr, args.warmup),
            lambda _: args.lr,
        ],
        [args.warmup],
    )
    opt = optim.AdamW(learning_rate=schedule)
    lvg = nn.value_and_grad(model, loss_fn)

    def save_checkpoint(dst):
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        if lora_parameters is not None:
            save_adapter_dir(
                dst, model, lora_parameters, len(model.layers), args.model
            )
            return
        # mlx_lm.utils.save() hardcodes donate_model=True, which would destroy
        # the live training weights — replicate it with donate_model=False
        save_model(dst, model, donate_model=False)
        save_config(config, config_path=dst / "config.json")
        tokenizer.save_pretrained(dst)

    P, G = args.prompts_per_step, args.group_size
    word_iter, epoch = iter(words), 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch_words = []
        for _ in range(P):
            try:
                batch_words.append(next(word_iter))
            except StopIteration:
                epoch += 1
                rng.shuffle(words)
                word_iter = iter(words)
                batch_words.append(next(word_iter))

        # --- rollouts: G samples per prompt, same weights we train ---
        t_roll = time.time()
        prompts = [get_prompt(w) for w in batch_words for _ in range(G)]
        completions, finish = rollout(
            model, tokenizer, prompts, args.max_tokens, sampler,
            args.rollout_batch_size,
        )
        roll_dt = time.time() - t_roll

        rewards, taggeds = [], []
        for i, comp in enumerate(completions):
            r, tagged = reward_fn(tokenizer.decode(comp), batch_words[i // G])
            rewards.append(float(r))
            taggeds.append(tagged)

        # --- advantages: reward - group mean, no std normalization ---
        # truncated rollouts count toward the group mean (siblings are still
        # rewarded relative to them) but are masked from the gradient unless
        # --train-truncated: no policy gradient through degenerate loops
        rows, advs = [], []
        kept_groups, masked_trunc = 0, 0
        for g in range(P):
            group_r = rewards[g * G : (g + 1) * G]
            mean_r = sum(group_r) / G
            if all(r == group_r[0] for r in group_r):
                continue  # zero advantage everywhere -> no gradient
            kept_groups += 1
            p_ids = get_prompt(batch_words[g])
            for k in range(G):
                i = g * G + k
                if finish[i] == "length" and not args.train_truncated:
                    masked_trunc += 1
                    continue
                rows.append((p_ids + completions[i], len(p_ids)))
                advs.append(group_r[k] - mean_r)

        n_tokens = sum(len(c) for c in completions)
        stats = {
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "exact_rate": sum(r == 0 for r in rewards) / len(rewards),
            "tag_rate": sum(taggeds) / len(taggeds),
            "trunc_rate": sum(f == "length" for f in finish) / len(finish),
            "kept_groups": kept_groups,
            "masked_trunc": masked_trunc,
            "len_mean": n_tokens / len(completions),
            "rollout_tps": n_tokens / roll_dt,
            "epoch": epoch,
        }

        # --- policy gradient over microbatches ---
        if rows:
            t_train = time.time()
            n_total = len(rows)
            loss_val, acc = 0.0, None
            for i in range(0, n_total, args.microbatch):
                chunk = rows[i : i + args.microbatch]
                x, w = pad_batch(chunk, pad_id)
                adv = mx.array(advs[i : i + args.microbatch]) / n_total
                loss, grads = lvg(model, x, w, adv)
                acc = grads if acc is None else tree_map(mx.add, acc, grads)
                mx.eval(loss, acc)
                loss_val += loss.item()
            if args.grad_clip > 0:
                acc, _ = optim.clip_grad_norm(acc, args.grad_clip)
            opt.update(model, acc)
            mx.eval(model.parameters(), opt.state)
            stats["loss"] = loss_val
            stats["train_s"] = time.time() - t_train

        line = " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in stats.items()
        )
        print(f"step {step}/{args.steps} {line} "
              f"({(time.time() - t0) / step:.1f} s/step, "
              f"peak {mx.get_peak_memory() / 1e9:.1f} GB)", flush=True)
        if run:
            run.log(stats, step=step)

        # essential on unified memory: without periodic cache clears the
        # buffer cache balloons (see ml-explore/mlx-lm#986)
        if step % 8 == 0:
            mx.clear_cache()

        if args.save_every and step % args.save_every == 0:
            save_checkpoint(f"{args.out}-step{step}")
            print(f"checkpoint -> {args.out}-step{step}", flush=True)

    save_checkpoint(args.out)
    print(f"saved -> {args.out}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()

# Word Reversal: an SFT + GRPO Challenge

Teach a 1B language model to reverse words — `"pluto"` → `"otulp"` — with
chain-of-thought reasoning. You get **100 supervised examples** and **1,000
RL prompts**. The supervised set is enough to teach the *format*; actually
learning the *skill* is your reinforcement learning loop's job.

Everything runs on a single GPU with [MLX](https://github.com/ml-explore/mlx).
Because MLX is one framework for both fast generation and training — the
same weights, the same memory — the GRPO loop needs no separate inference
engine, no weight syncing, and no importance ratios: rollouts are always
exactly on-policy.

## The task

The model receives:

> Reverse the word "pluto". Reason in \<think\> XML tags, and put your answer
> in \<answer\> tags.

and should respond with reasoning in `<think>` tags and the reversed word in
`<answer>` tags:

```
<think>
"pluto" is spelled "p", "l", "u", "t", "o". That's:

1. p
2. l
3. u
4. t
5. o
So reversed is 5 -> o, 4 -> t, 3 -> u, 2 -> l, 1 -> p, or "o", "t", "u",
"l", "p". Which is "otulp." Answer is "otulp".
</think>
<answer>
otulp
</answer>
```

## Data (ships in `data/`)

| split | size | file | use |
|---|---|---|---|
| sft | 100 | `data/sft_train.jsonl` (+ `sft_words.txt`) | gold reasoning traces for supervised fine-tuning |
| rl | 1,000 | `data/rl_words.txt` | prompts for GRPO |
| eval | 1,000 | `data/eval_words.txt` | held-out evaluation — **never train on these** |

Words are the most frequent English words of length ≥ 5 (Google corpus
frequency). The splits are disjoint. `make_dataset.py` regenerates
everything deterministically if you want to inspect the process.

## Setup

**NVIDIA GPU (CUDA), e.g. DGX Spark.** Requirements: driver ≥ 580 (CUDA 13
stack), Python ≥ 3.10. In a fresh environment (conda or venv) - make sure to pin the version number to avoid accidentally installing a newer, incompatible version:

```bash
pip install "mlx-lm[cuda13]==0.31.2"
```

**Apple silicon Mac.** Just `pip install mlx-lm` — everything here runs
unchanged on the Metal backend (a 1B model trains comfortably on ≥ 32 GB).

Optional: `pip install wandb` to get live training curves from
`train_grpo.py --wandb-project <name>`.

## Quickstart

```bash
./run_reference.sh        # the full reference pipeline, ~45 min on a GB10
```

or step by step:

```bash
python train_sft.py                                                  # ~1 min
python evaluate.py --adapter-path models/sft-lora                    # ~5 min
python train_grpo.py --adapter-path models/sft-lora                 # ~35 min
python evaluate.py --adapter-path models/grpo-lora --temp 0.7        # ~5 min
```

## Reference results (your baseline to beat)

| model | decode | exact match | avg Levenshtein |
|---|---|---|---|
| Llama-3.2-1B-Instruct (base) | greedy | 0.0% | ~104 |
| + SFT on the 100 traces | greedy | 34.5% | 1.49 |
| + GRPO, 300 steps | temp 0.7 | **51.6%** | 0.98 |

**Beat 51.6% exact match on the eval set.** The reference pipeline is
fully seeded (`--seed 42`), so rerunning it reproduces these numbers on
the same hardware (a different GPU model may land a couple of points
away). The reference GRPO curve is still climbing at step 300 and not
plateauing, so there is real headroom — through longer runs, better
hyperparameters, or smarter ideas.

## Scoring

`evaluate.py` reports, over the 1,000 eval words:

- **exact match %** (the leaderboard metric): the parsed `<answer>` equals
  the reversed word, case/whitespace-insensitive. No answer tag = wrong.
- **avg Levenshtein**: edit distance between parsed answer and target,
  over tagged responses (tag rate reported alongside).

Decode however you like — greedy and temperature sampling are both fair;
the reference evaluates each policy the way it was trained (greedy for SFT,
temp 0.7 for GRPO). Report the command you used.

## How the reference GRPO works

`train_grpo.py` is ~400 lines and deliberately readable — it is the thing
this challenge wants you to understand and improve. The essentials:

- **Rollouts**: each step samples G=8 completions for each of 4 words via
  MLX's continuous-batching generator, keeping the exact sampled token ids.
- **Reward**: negative Levenshtein distance between the parsed answer and
  the target (a missing `<answer>` tag counts as an empty answer).
- **Advantage**: reward minus the group mean. No std normalization.
- **Loss**: `-advantage * logp` on the sampled tokens. Same weights sample
  and train, so there is no importance ratio and no reference model. The
  log-probs are scored at the sampling temperature (and under any top-p/
  top-k truncation), so the scored distribution is exactly the sampling
  distribution.
- **Stability**: 50-step lr warmup (un-warmed Adam at RL onset wrecks the
  policy), sequence log-probs normalized by a fixed constant rather than
  sequence length (length-normalization teaches the policy to bloat its
  failures), and truncated rollouts contribute to the group baseline but
  receive no gradient.

Each stabilizer guards against a failure mode we hit while building this —
remove them and watch what happens.

## Ideas worth trying

- More steps / more rollouts per step (the reference curve hasn't flattened)
- Group size, temperature, exploration schedules
- Reward shaping: exact-match bonus, length penalties, format shaping
- Curriculum: short words first
- A bigger or different base model, different LoRA rank — or full fine-tuning

## Notes

- Training the 1B uses ~30 GB peak in the reference configuration.
- The scripts disable CUDA graphs (`MLX_USE_CUDA_GRAPHS=0`) so they behave
  identically on discrete-VRAM and unified-memory GPUs; set
  `MLX_USE_CUDA_GRAPHS=1` for a modest speedup on unified-memory machines.
  They also call `mx.clear_cache()` periodically.
- First-ever run JIT-compiles kernels (~1 min one-time cost, cached on disk
  afterwards).
- LoRA learning rates: the SFT default (1e-4) follows the ~10x-of-full-FT
  heuristic at LoRA scale 32/rank. That heuristic does NOT transfer to the
  RL phase — GRPO is stable at 3e-6 (the full-FT rate) and degrades at
  higher rates, which is why the defaults differ.

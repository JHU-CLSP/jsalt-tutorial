"""LoRA helpers shared by the SFT and GRPO trainers.

Follows mlx_lm.lora's established practice: freeze the model, convert target
linears via mlx_lm.tuner.utils.linear_to_lora_layers (LoRALinear.from_base
under the hood), train model.trainable_parameters(), and save adapter dirs
(adapters.safetensors + adapter_config.json) that mlx_lm can load with
load(model, adapter_path=...) or tuner.utils.load_adapters.
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.tuner.utils import linear_to_lora_layers

# all linears except embeddings / lm_head
LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def apply_lora(model, rank, scale=None, dropout=0.0):
    """Freeze the model and convert target linears in ALL layers to LoRA.

    scale=None uses 32 / rank (alpha=32, the convention the "LoRA wants
    ~10x lr" heuristic assumes — https://thinkingmachines.ai/blog/lora/).

    Returns the lora_parameters dict (the adapter_config.json payload).
    """
    if scale is None:
        scale = 32.0 / rank
    model.freeze()
    lora_parameters = {
        "rank": rank,
        "scale": scale,
        "dropout": dropout,
        "keys": LORA_KEYS,
    }
    linear_to_lora_layers(model, len(model.layers), lora_parameters)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    n_total = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(
        f"LoRA rank {rank} scale {scale}: {n_train:,} trainable / "
        f"{n_total:,} total ({100 * n_train / n_total:.3f}%)"
    )
    return lora_parameters


def save_adapter_dir(out, model, lora_parameters, num_layers, base_model):
    """Write an mlx_lm-compatible adapter dir (config + trainable weights)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_parameters,
        "base_model": str(base_model),
    }
    with open(out / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    mx.save_safetensors(
        str(out / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )

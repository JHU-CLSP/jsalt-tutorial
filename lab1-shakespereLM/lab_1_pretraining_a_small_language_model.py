# -*- coding: utf-8 -*-
"""Lab 1: Pretraining A Small Language Model

Original Colab notebook:
    https://colab.research.google.com/drive/1i9CactH7yeXVfWTKbpWo5Wyp5WqTQZxL
"""

import math
import random
import subprocess
import sys
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparameters / constants
# ---------------------------------------------------------------------------

MODEL_DIM  = 384
N_HEADS    = 6
N_LAYERS   = 6
DOC_SIZE   = 255
BATCH_SIZE = 64
EPOCHS     = 5
VOCAB_SIZE = 256 + 1  # +1 for BOS token
BOS_TOKEN  = 256

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(txt: str) -> list[int]:
    return list(txt.encode("utf-8"))


def detokenize(arr: list[int]) -> str:
    return bytes(arr).decode("utf-8")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AttnHead(nn.Module):
    def __init__(self, model_dim: int, head_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(model_dim, head_dim)
        self.k_proj = nn.Linear(model_dim, head_dim)
        self.v_proj = nn.Linear(model_dim, head_dim)
        self.d = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape
        q = self.q_proj(x)                              # (B, L, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_scores = q @ k.transpose(-1, -2)           # (B, L, L)
        attn_scores = attn_scores / (self.d ** 0.5)
        mask = torch.tril(torch.ones(L, L, device=x.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_scores = F.softmax(attn_scores, dim=-1)
        return attn_scores @ v                          # (B, L, head_dim)


class Attn(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttnHead(model_dim, model_dim // num_heads) for _ in range(num_heads)]
        )
        self.o_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()
        self.up_proj   = nn.Linear(model_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.up_proj(x) * F.silu(self.gate_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, n_heads: int):
        super().__init__()
        self.attn = Attn(model_dim, n_heads)
        self.mlp  = MLP(model_dim, model_dim * 4)
        self.input_layernorm          = nn.RMSNorm(model_dim, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(model_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, n_layers: int, max_pos: int):
        super().__init__()
        self.embed     = nn.Embedding(VOCAB_SIZE, model_dim)
        self.pos_embed = nn.Embedding(max_pos, model_dim)
        self.blocks    = nn.ModuleList([TransformerBlock(model_dim, n_heads) for _ in range(n_layers)])
        self.out_norm  = nn.RMSNorm(model_dim, eps=1e-6)
        self.lm_head   = nn.Linear(model_dim, VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L = x.shape
        pos = torch.arange(L, device=x.device)
        x = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.out_norm(x))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(url: str) -> str:
    return urllib.request.urlopen(url).read().decode()


def build_batches(text: str, device: torch.device) -> tuple[list, list]:
    docs = [text[i:i + DOC_SIZE] for i in range(0, len(text) - DOC_SIZE, DOC_SIZE)]
    docs = [[BOS_TOKEN] + tokenize(doc) for doc in docs]
    random.shuffle(docs)

    batches = [
        torch.tensor(docs[i:i + BATCH_SIZE], device=device)
        for i in range(0, len(docs) - BATCH_SIZE + 1, BATCH_SIZE)
    ]

    split = int(0.9 * len(batches))
    return batches[:split], batches[split:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: Transformer, train_batches: list, val_batches: list, optimizer):
    for epoch in range(EPOCHS):
        model.train()
        for i, batch in enumerate(train_batches):
            x, y = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(train_batches)} | Train Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            val_losses = [
                F.cross_entropy(model(b[:, :-1]).view(-1, VOCAB_SIZE), b[:, 1:].reshape(-1)).item()
                for b in val_batches
            ]
        val_t = torch.tensor(val_losses)
        mean  = val_t.mean().item()
        stderr = (val_t.std() / math.sqrt(len(val_losses))).item()
        print(f"====> Validation Loss at epoch {epoch}: {mean:.4f} +/- {stderr:.4f}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model: Transformer, prompt: str, device: torch.device,
             max_new_tokens: int = 200, temperature: float = 0.5) -> str:
    model.eval()
    tokens = [BOS_TOKEN] + tokenize(prompt)
    x = torch.tensor(tokens, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -DOC_SIZE:]
            logits = model(x_cond)
            if temperature == 0.0:
                next_token = torch.argmax(logits[0, -1, :]).view(1, 1)
            else:
                probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            x = torch.cat((x, next_token), dim=1)

    return bytes(x[0].tolist()[1:]).decode("utf-8", errors="replace")  # skip BOS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Install Muon optimizer if needed
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "git+https://github.com/KellerJordan/Muon"])
    from muon import SingleDeviceMuonWithAdam as SingleDeviceMuonWithAuxAdam  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading data...")
    text = load_data(DATA_URL)
    train_batches, val_batches = build_batches(text, device)
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")

    # Model
    model = Transformer(MODEL_DIM, N_HEADS, N_LAYERS, max_pos=DOC_SIZE + 1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (Muon for weight matrices, Adam for embeddings/head)
    hidden_matrices = [p for n, p in model.named_parameters()
                       if p.ndim == 2 and "embed" not in n and "lm_head" not in n]
    adam_params     = [p for n, p in model.named_parameters()
                       if p.ndim != 2 or "embed" in n or "lm_head" in n]
    optimizer = SingleDeviceMuonWithAuxAdam([
        dict(params=hidden_matrices, use_muon=True,  lr=0.02,  weight_decay=0.0),
        dict(params=adam_params,     use_muon=False, lr=3e-4,  betas=(0.9, 0.999), weight_decay=0.01),
    ])

    # Train
    train(model, train_batches, val_batches, optimizer)

    # Generate
    output = generate(model, "O Romeo, Romeo!", device)
    print("\n--- Generated text ---")
    print(output)


if __name__ == "__main__":
    main()

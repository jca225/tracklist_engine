"""M2 — small causal sequence models with a discrete softmax head.

Two architectures (causal self-attention and GRU) over the shared VQ token
sequence, evaluated under a strict **prequential expanding-window** protocol:

    warm-up train on tokens[0:W]
    for each block [s, e):
        predict tokens[s:e]  using a model trained only on [0:s]   <-- recorded
        then warm-start train on tokens[0:e]

Because the block's predictions are recorded *before* any gradient step sees the
block's targets — and each next-token forecast for frame t reads the position
that attends only to tokens[0:t-1] — no information at or after frame t ever
touches its own prediction. Frames in the warm-up prefix are left NaN (no honest
out-of-sample forecast exists for them).

Signals: ``surprisal`` (NLL of the true token), ``entropy`` (predictive
entropy), ``fwd_kl`` (predictive-information *proxy* — KL between the one-step
forecast after vs. before observing the current token). ``fwd_kl`` is labelled a
proxy: it is not the paper's exact I(x|z).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import StudyData
from .signals import SignalSet


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class CausalTransformer(nn.Module):
    """Decoder-only LM over K tokens; optional banded (windowed) attention."""

    def __init__(
        self,
        n_tokens: int,
        *,
        d_model: int = 48,
        n_layers: int = 2,
        n_heads: int = 2,
        max_len: int = 2100,
        context: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context = context
        self.tok_emb = nn.Embedding(n_tokens, d_model)
        # Fixed sinusoidal PE — generalizes to positions never seen during the
        # expanding-window warm-start (where every block's eval frontier lives).
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_emb", pe, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_tokens)

    def _mask(self, t: int, device: torch.device) -> torch.Tensor:
        # True = disallowed. Causal + optional band limit (context window).
        i = torch.arange(t, device=device).unsqueeze(1)
        j = torch.arange(t, device=device).unsqueeze(0)
        disallow = j > i  # future
        if self.context is not None:
            disallow = disallow | (j <= i - self.context)
        return disallow

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (T,) long  ->  logits (T, K)
        t = seq.shape[0]
        h = self.tok_emb(seq) + self.pos_emb[:t]
        h = self.encoder(h.unsqueeze(0), mask=self._mask(t, seq.device)).squeeze(0)
        return self.head(h)


class GRUModel(nn.Module):
    """GRU LM over K tokens — the recurrent memory alternative."""

    def __init__(self, n_tokens: int, *, d_model: int = 48, n_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(n_tokens, d_model)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, n_tokens)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.tok_emb(seq)).unsqueeze(0)
        out, _ = self.gru(h)
        return self.head(self.drop(out.squeeze(0)))


def _build(arch, n_tokens, context, d_model, n_layers, n_heads, dropout):
    if arch == "attention":
        return CausalTransformer(
            n_tokens, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context=context, dropout=dropout,
        )
    if arch == "gru":
        return GRUModel(n_tokens, d_model=d_model, n_layers=max(1, n_layers), dropout=dropout)
    raise ValueError(f"unknown arch {arch!r}")


def _train_prefix(model, optim, tokens_t: torch.Tensor, *, epochs: int) -> None:
    """Warm-start teacher-forced training on tokens[0:len] for `epochs` passes."""
    if tokens_t.shape[0] < 2:
        return
    inp, tgt = tokens_t[:-1], tokens_t[1:]
    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        logits = model(inp)
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()


def run_m2(
    data: StudyData,
    *,
    arch: str = "attention",
    context: int | None = None,
    warmup: int = 128,
    block: int = 32,
    warmup_epochs: int = 30,
    refit_epochs: int = 5,
    d_model: int = 48,
    n_layers: int = 2,
    n_heads: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
    label: str | None = None,
) -> SignalSet:
    _seed_everything(seed)
    dev = torch.device(device)
    tokens = torch.tensor(data.tokens, dtype=torch.long, device=dev)
    n = tokens.shape[0]
    K = data.n_tokens

    model = _build(arch, K, context, d_model, n_layers, n_heads, dropout).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    surprisal = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    fwd_kl = np.full(n, np.nan)

    warmup = min(max(warmup, 2), n - 1)
    _train_prefix(model, optim, tokens[:warmup], epochs=warmup_epochs)

    for s in range(warmup, n, block):
        e = min(s + block, n)
        model.eval()
        with torch.no_grad():
            logits = model(tokens[: e - 1])             # (e-1, K): logits[i] predicts token i+1
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            ent = -(p * logp).sum(dim=-1)               # (e-1,)
            for t in range(s, e):
                lp_t = logp[t - 1]                      # forecast of token t from tokens[0:t-1]
                surprisal[t] = float(-lp_t[tokens[t]].item())
                entropy[t] = float(ent[t - 1].item())
                if t <= e - 2:                          # KL(F_t || F_{t-1}); F_t needs logits[t]
                    fwd_kl[t] = float(
                        (p[t] * (logp[t] - logp[t - 1])).sum().clamp_min(0.0).item()
                    )
        _train_prefix(model, optim, tokens[:e], epochs=refit_epochs)

    name = label or f"M2-{arch}" + (f"-ctx{context}" if context is not None else "")
    return SignalSet(
        model=name,
        n_frames=n,
        signals={
            "surprisal": surprisal,
            "entropy": entropy,
            "fwd_kl": fwd_kl,
        },
    )

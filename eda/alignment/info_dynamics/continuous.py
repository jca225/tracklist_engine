"""Continuous (no-VQ) information-dynamics models.

Drops the k-means codebook bottleneck: the predictive model works directly on
the (PCA-whitened) MERT vectors and surprise is a proper Gaussian negative
log-likelihood, so it is comparable across models in nats.

- ``M0c`` — memoryless continuous nulls: persistence (predict zₜ = zₜ₋₁) and
  running-mean, each with an online-estimated homoscedastic variance.
- ``M2c`` — causal GRU / attention regressing the next vector with a
  heteroscedastic scalar variance head, trained under the same prequential
  expanding-window protocol as the discrete M2.

Whitening is fit on the whole mix (unsupervised, boundary-blind) — same status
as the global codebook.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .data import StudyData, normalized_mert
from .seqmodel import CausalTransformer, _seed_everything
from .signals import SignalSet

_LOG2PI = math.log(2.0 * math.pi)


def pca_whiten(data: StudyData, *, d: int = 64) -> np.ndarray:
    """L2-norm MERT → centered → top-d PCA, unit variance per component."""
    x = normalized_mert(data).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    d = min(d, vt.shape[0])
    z = u[:, :d] * math.sqrt(x.shape[0] - 1)  # whitened scores, unit variance
    return z.astype(np.float32)


def _gauss_nll(z: np.ndarray, mu: np.ndarray, var: float | np.ndarray) -> float:
    d = z.shape[0]
    var = np.maximum(var, 1e-6)
    sq = float(np.sum((z - mu) ** 2))
    if np.isscalar(var):
        return 0.5 * (d * (math.log(float(var)) + _LOG2PI) + sq / float(var))
    return 0.5 * (float(np.sum(np.log(var))) + d * _LOG2PI + sq / float(var))


def run_m0_continuous(data: StudyData, z: np.ndarray, *, warmup: int = 16) -> SignalSet:
    """Persistence & running-mean prediction, online homoscedastic variance."""
    n, d = z.shape
    persist = np.full(n, np.nan)
    meanpred = np.full(n, np.nan)

    # online residual variance (per-dim, isotropic) for each predictor
    sse_p, cnt_p = 1.0, 1
    run_sum = z[0].astype(np.float64).copy()
    sse_m, cnt_m = 1.0, 1
    for t in range(1, n):
        # persistence: predict z_t = z_{t-1}
        var_p = sse_p / (cnt_p * d)
        persist[t] = _gauss_nll(z[t], z[t - 1], var_p)
        sse_p += float(np.sum((z[t] - z[t - 1]) ** 2)); cnt_p += 1
        # running mean predictor
        mu_m = run_sum / t
        var_m = sse_m / (cnt_m * d)
        meanpred[t] = _gauss_nll(z[t], mu_m, var_m)
        sse_m += float(np.sum((z[t] - mu_m) ** 2)); cnt_m += 1
        run_sum += z[t]

    # warm-up frames have unstable variance estimates
    persist[:warmup] = np.nan
    meanpred[:warmup] = np.nan
    return SignalSet(
        model="M0c", n_frames=n,
        signals={"surprisal": persist, "meanpred_nll": meanpred},
    )


class ContTransformer(nn.Module):
    def __init__(self, d_in, *, d_model=96, n_layers=2, n_heads=2,
                 context=None, dropout=0.1, max_len=2100):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.core = CausalTransformer(  # reuse PE + masking; ignore its tok_emb/head
            2, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            context=context, dropout=dropout, max_len=max_len,
        )
        self.mu = nn.Linear(d_model, d_in)
        self.logvar = nn.Linear(d_model, 1)

    def forward(self, z):
        t = z.shape[0]
        h = self.proj(z) + self.core.pos_emb[:t]
        h = self.core.encoder(h.unsqueeze(0), mask=self.core._mask(t, z.device)).squeeze(0)
        return self.mu(h), self.logvar(h).squeeze(-1)


class ContGRU(nn.Module):
    def __init__(self, d_in, *, d_model=96, n_layers=1, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.mu = nn.Linear(d_model, d_in)
        self.logvar = nn.Linear(d_model, 1)

    def forward(self, z):
        h = self.drop(self.proj(z)).unsqueeze(0)
        out, _ = self.gru(h)
        out = self.drop(out.squeeze(0))
        return self.mu(out), self.logvar(out).squeeze(-1)


def _nll_loss(mu, logvar, target):
    d = target.shape[1]
    var = torch.exp(logvar).clamp_min(1e-6)
    sq = ((target - mu) ** 2).sum(dim=1)
    return 0.5 * (d * logvar + sq / var).mean()


def _train_cont(model, optim, z, *, epochs):
    if z.shape[0] < 2:
        return
    inp, tgt = z[:-1], z[1:]
    model.train()
    for _ in range(epochs):
        optim.zero_grad()
        mu, logvar = model(inp)
        loss = _nll_loss(mu, logvar, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()


def run_m2_continuous(
    data: StudyData,
    z_np: np.ndarray,
    *,
    arch: str = "gru",
    context: int | None = None,
    warmup: int = 128,
    block: int = 32,
    warmup_epochs: int = 30,
    refit_epochs: int = 5,
    d_model: int = 96,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
    label: str | None = None,
) -> SignalSet:
    _seed_everything(seed)
    dev = torch.device(device)
    z = torch.tensor(z_np, dtype=torch.float32, device=dev)
    n, d = z.shape

    if arch == "attention":
        model = ContTransformer(d, d_model=d_model, context=context).to(dev)
    elif arch == "gru":
        model = ContGRU(d, d_model=d_model).to(dev)
    else:
        raise ValueError(arch)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    surprisal = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    pred_change = np.full(n, np.nan)

    warmup = min(max(warmup, 2), n - 1)
    _train_cont(model, optim, z[:warmup], epochs=warmup_epochs)

    for s in range(warmup, n, block):
        e = min(s + block, n)
        model.eval()
        with torch.no_grad():
            mu, logvar = model(z[: e - 1])      # predicts z[i+1] at row i
            for t in range(s, e):
                m = mu[t - 1]
                lv = float(logvar[t - 1].item())
                var = math.exp(lv)
                sq = float(((z[t] - m) ** 2).sum().item())
                surprisal[t] = 0.5 * (d * (lv + _LOG2PI) + sq / max(var, 1e-6))
                entropy[t] = 0.5 * d * (lv + 1.0 + _LOG2PI)   # Gaussian entropy ∝ logvar
                if t <= e - 2:
                    pred_change[t] = float(((mu[t] - mu[t - 1]) ** 2).sum().item())
        _train_cont(model, optim, z[:e], epochs=refit_epochs)

    name = label or f"M2c-{arch}" + (f"-ctx{context}" if context is not None else "")
    return SignalSet(
        model=name, n_frames=n,
        signals={"surprisal": surprisal, "entropy": entropy, "pred_change": pred_change},
    )

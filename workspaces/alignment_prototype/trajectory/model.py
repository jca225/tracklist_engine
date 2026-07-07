"""Minimal learned trajectory decoder.

A small conv stack over the (match, mel) cross-similarity channels produces
a per-(mix bin, ref bin) score; a per-frame head produces the NULL logit.
Deliberately tiny (~10k params) — 313 GT spans cannot feed more, and the
point of v1 is the training loop + held-out protocol, not architecture.
The conv receptive field is what the hand Viterbi cannot do: the score of
(t, r) sees its diagonal neighborhood, so evidence accumulates across
frames before any decode — the accumulation-not-per-window lesson from the
reconstruction probes, made learnable.

Per-feature-kind affine on the match channel: chroma and HuBERT cosines
live on different scales; a learned scale/bias per kind beats normalizing
them offline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryDecoder(nn.Module):
    def __init__(self, hidden: int = 16, n_feat_kinds: int = 2) -> None:
        super().__init__()
        self.kind_affine = nn.Embedding(n_feat_kinds, 2)  # scale, bias
        nn.init.constant_(self.kind_affine.weight[:, 0], 1.0)
        nn.init.constant_(self.kind_affine.weight[:, 1], 0.0)
        self.conv = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        # NULL logit from per-frame pooled evidence: (max, mean) of each
        # sim channel over ref -> how well ANY ref position explains frame t
        self.null_head = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        sim: torch.Tensor,  # (B, 2, Tm, Tr)
        feat_kind: torch.Tensor,  # (B,)
        ref_valid: torch.Tensor,  # (B, Tr)
    ) -> torch.Tensor:  # (B, Tm, Tr+1)
        aff = self.kind_affine(feat_kind)  # (B, 2)
        scale = aff[:, 0].view(-1, 1, 1, 1)
        bias = aff[:, 1].view(-1, 1, 1, 1)
        x = torch.cat([sim[:, :1] * scale + bias, sim[:, 1:]], dim=1)
        grid = self.conv(x).squeeze(1)  # (B, Tm, Tr)

        rv = ref_valid[:, None, None, :]  # (B, 1, 1, Tr)
        masked = sim.masked_fill(~rv, float("-inf"))
        pooled_max = masked.amax(dim=-1)  # (B, 2, Tm)
        denom = rv.sum(dim=-1).clamp(min=1)  # (B, 1, 1)
        pooled_mean = (sim * rv).sum(dim=-1) / denom  # (B, 2, Tm)
        null_in = torch.cat([pooled_max, pooled_mean], dim=1).transpose(1, 2)
        null_logit = self.null_head(null_in)  # (B, Tm, 1)

        return torch.cat([grid, null_logit], dim=-1)

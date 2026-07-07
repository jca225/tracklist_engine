"""Learned trajectory decoder.

A conv stack over cross-similarity channels produces a per-(mix bin, ref bin)
score; a per-frame head produces the NULL logit. The conv receptive field is
what the hand Viterbi cannot do: the score of (t, r) sees its diagonal
neighborhood, so evidence accumulates across frames before any decode — the
accumulation-not-per-window lesson from the reconstruction probes, made
learnable.

v2 (hidden=32, diag_windows=(8, 24)) appended fixed diagonal-mean channels
and a third conv — and REGRESSED on held-out BB11 (viterbi .424 -> .329,
2026-07-07 log trajectory_v2_20260707.log). Likely: the slope-1 diagonal is
wrong for tempo-stretched/oddratio spans (noise, not context), and the
smoother grid starves the Viterbi of sharp evidence. Defaults are back at v1
(hidden=16, no diag channels); the knobs stay for controlled ablations.

Per-feature-kind affine on the match channel: chroma and HuBERT cosines
live on different scales; a learned scale/bias per kind beats normalizing
them offline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def diag_mean(sim: torch.Tensor, w: int) -> torch.Tensor:
    """Mean of sim[..., t+k, r+k] over k < w — accumulation along the
    alignment diagonal (a straight clip is a line in (mix, ref) time)."""
    tm, tr = sim.shape[-2], sim.shape[-1]
    acc = torch.zeros_like(sim)
    cnt = torch.zeros(tm, tr, device=sim.device, dtype=sim.dtype)
    for k in range(min(w, tm, tr)):
        acc[..., : tm - k, : tr - k] += sim[..., k:, k:]
        cnt[: tm - k, : tr - k] += 1
    return acc / cnt.clamp(min=1)


class TrajectoryDecoder(nn.Module):
    def __init__(
        self,
        hidden: int = 16,
        n_feat_kinds: int = 2,
        diag_windows: tuple[int, ...] = (),  # (8, 24) = the v2 regression
        deep: bool = False,  # third dilated conv (part of the v2 regression)
    ) -> None:
        super().__init__()
        self.diag_windows = tuple(diag_windows)
        self.kind_affine = nn.Embedding(n_feat_kinds, 2)  # scale, bias
        nn.init.constant_(self.kind_affine.weight[:, 0], 1.0)
        nn.init.constant_(self.kind_affine.weight[:, 1], 0.0)
        c_in = 2 * (1 + len(self.diag_windows))
        layers: list[nn.Module] = [
            nn.Conv2d(c_in, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(),
        ]
        if deep:
            layers += [
                nn.Conv2d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
                nn.ReLU(),
            ]
        layers.append(nn.Conv2d(hidden, 1, kernel_size=1))
        self.conv = nn.Sequential(*layers)
        # NULL logit from per-frame pooled evidence: (max, mean) of each
        # input channel over ref -> how well ANY ref position explains frame t
        self.null_head = nn.Sequential(
            nn.Linear(2 * c_in, hidden),
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
        base = torch.cat([sim[:, :1] * scale + bias, sim[:, 1:]], dim=1)
        chans = [base] + [diag_mean(base, w) for w in self.diag_windows]
        x = torch.cat(chans, dim=1)  # (B, c_in, Tm, Tr)
        grid = self.conv(x).squeeze(1)  # (B, Tm, Tr)

        rv = ref_valid[:, None, None, :]  # (B, 1, 1, Tr)
        masked = x.masked_fill(~rv, float("-inf"))
        pooled_max = masked.amax(dim=-1)  # (B, c_in, Tm)
        denom = rv.sum(dim=-1).clamp(min=1)  # (B, 1, 1)
        pooled_mean = (x * rv).sum(dim=-1) / denom  # (B, c_in, Tm)
        null_in = torch.cat([pooled_max, pooled_mean], dim=1).transpose(1, 2)
        null_logit = self.null_head(null_in)  # (B, Tm, 1)

        return torch.cat([grid, null_logit], dim=-1)

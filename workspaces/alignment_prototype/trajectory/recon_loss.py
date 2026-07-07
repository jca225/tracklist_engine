"""Training objectives: supervised trajectory CE + label-free reconstruction.

Reconstruction (docs/reconstruction_supervision_plan.md, permanent design
call): the mix is a free answer key, but ONLY for the host/regular channel —
per-window vocal reconstruction sits at the noise floor (0.16 cosine) and
must never be a loss. The objective here is differentiable analysis-by-
synthesis in mel space: the model's per-frame distribution over ref bins is
a soft alignment; the expected ref mel frame under it should reconstruct the
observed mix mel frame. NULL probability mass reconstructs silence (the zero
vector), so predicting NULL where the mix doesn't contain the ref is free,
and pointing at the wrong ref section costs cosine. Label-free: usable on
the 20k unlabeled sets once the harvest starts.

Supervised CE: per-frame classification over ref bins + a NULL class, with
Gaussian label smoothing across neighboring bins (the scorer's 2 s tolerance
means adjacent bins are nearly as correct — hard one-hot would fight it).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_NEG = -1e9


def masked_full_logits(logits: torch.Tensor, ref_valid: torch.Tensor) -> torch.Tensor:
    """(B, Tm, Tr+1) with padded ref bins forced to -inf (NULL stays live)."""
    tr = ref_valid.shape[1]
    ref_part = logits[..., :tr].masked_fill(~ref_valid[:, None, :], _NEG)
    return torch.cat([ref_part, logits[..., tr:]], dim=-1)


def trajectory_ce(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    target_null: torch.Tensor,
    mix_valid: torch.Tensor,
    ref_valid: torch.Tensor,
    sigma_bins: float = 2.0,
    null_weight: float = 0.3,
) -> torch.Tensor:
    """Soft-label CE over (ref bins + NULL).

    logits (B, Tm, Tr+1); target_idx (B, Tm) with -1 = ignore; target_null
    marks frames whose truth is the NULL class. `sigma_bins` = Gaussian
    smoothing width (2 bins at 0.5 s/bin ~ half the scorer's 2 s tolerance).
    `null_weight` down-weights NULL frames so the (rare) class doesn't need
    a balancing scheme but still teaches abstention.
    """
    b, tm, k = logits.shape
    tr = k - 1
    logp = F.log_softmax(masked_full_logits(logits, ref_valid), dim=-1)

    pos_mask = (target_idx >= 0) & mix_valid
    null_mask = target_null & mix_valid
    total = torch.zeros((), device=logits.device)
    denom = 0.0

    if pos_mask.any():
        idx = target_idx[pos_mask]  # (N,)
        bins = torch.arange(tr, device=logits.device)
        soft = torch.exp(-0.5 * ((bins[None, :] - idx[:, None]) / sigma_bins) ** 2)
        # zero the smoothing on each example's padded ref bins, then renormalize
        bidx = pos_mask.nonzero(as_tuple=True)[0]
        soft = soft * ref_valid[bidx].to(soft.dtype)
        soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
        lp = logp[..., :tr][pos_mask]  # (N, Tr)
        total = total + -(soft * lp).sum()
        denom += float(idx.numel())

    if null_mask.any():
        lp_null = logp[..., tr][null_mask]  # (N,)
        total = total + null_weight * -lp_null.sum()
        denom += null_weight * float(lp_null.numel())

    if denom == 0.0:
        return torch.zeros((), device=logits.device, requires_grad=True)
    return total / denom


def reconstruction_loss(
    logits: torch.Tensor,
    ref_mel: torch.Tensor,
    mix_mel: torch.Tensor,
    mix_valid: torch.Tensor,
    ref_valid: torch.Tensor,
    recon_ok: torch.Tensor,
) -> torch.Tensor:
    """1 - mean mel cosine between the soft-aligned render and the mix.

    logits (B, Tm, Tr+1); ref_mel (B, Tr, 64); mix_mel (B, Tm, 64). Applied
    only to frames of host/regular spans (`recon_ok`): expected-ref-frame
    under softmax(logits) vs the observed mix frame, NULL mass -> silence.
    Label-free by construction (never reads target_idx).
    """
    tr = ref_mel.shape[1]
    p = F.softmax(masked_full_logits(logits, ref_valid), dim=-1)
    render = torch.einsum("btr,brd->btd", p[..., :tr], ref_mel)
    render = F.normalize(render, dim=-1, eps=1e-8)
    mix = F.normalize(mix_mel, dim=-1, eps=1e-8)
    cos = (render * mix).sum(dim=-1)  # (B, Tm)
    m = mix_valid & recon_ok[:, None]
    if not m.any():
        return torch.zeros((), device=logits.device, requires_grad=True)
    return 1.0 - (cos * m).sum() / m.sum()

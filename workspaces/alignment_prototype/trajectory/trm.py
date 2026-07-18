"""Tiny Recursive Model core for the trajectory decoder (bake-off §2.1-2.3).

A 2-layer block iteratively refines a proposed answer `y` against a latent
scratchpad `z`, given a fixed input embedding `x`. Deep supervision carries
(y, z) detached between improvement steps; each improvement step backprops
through ALL `T` inner recursion rounds (the HRM->TRM fix — not a 1-step
gradient). A Q-head emits a per-example ACT halt signal ("is the answer good
enough to stop thinking?", depth not length).

Token mixing is self-attention (size-agnostic over the variable mix-frame axis
`Tm`); an MLP-mixer variant — TRM's headline choice on *fixed*-grid puzzles — is
a later ablation once the answer grid is fixed. The answer is emitted in
OFFSET coordinates (`offset_coords.OffsetVocab`), so it is fixed-size regardless
of the span's ref length; `sim_to_offset_evidence` re-indexes the raw
cross-similarity grid into those coordinates for the input embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .offset_coords import OffsetVocab


class _Block(nn.Module):
    """Pre-norm attention + MLP — the shared 2-layer `net` of the recursion."""

    def __init__(self, d_model: int, n_heads: int = 2, mlp_mult: int = 2) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_mult),
            nn.GELU(),
            nn.Linear(d_model * mlp_mult, d_model),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        h = self.n1(a)
        a = a + self.attn(h, h, h, need_weights=False)[0]
        a = a + self.mlp(self.n2(a))
        return a


@dataclass
class TRMOutput:
    logits: list[torch.Tensor]  # one (B, N, V) grid per supervision step
    q: list[torch.Tensor]  # one (B,) ACT halt logit per supervision step
    y: torch.Tensor  # final refined answer latent (B, N, d), detached
    z: torch.Tensor  # final scratchpad (B, N, d), detached


class TRMCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        n_sup: int = 6,
        n_inner: int = 6,
        T: int = 3,
        y_mode: str = "once",
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        if y_mode not in ("once", "every"):
            raise ValueError(f"y_mode must be once|every, got {y_mode!r}")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_sup = n_sup
        self.n_inner = n_inner
        self.T = T
        self.y_mode = y_mode
        self.net = _Block(d_model, n_heads=n_heads)
        # normalize the answer latent before projecting to the (large) vocab —
        # the residual stream grows across ~n_sup*T*n_inner net applications;
        # without this the logits explode (v0 saw CE ~8e7). Bounds logit scale.
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.q_head = nn.Linear(d_model, 1)
        self.y0 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.z0 = nn.Parameter(torch.zeros(1, 1, d_model))

    def _latent_recursion(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.n_inner):
            z = self.net(x + y + z)  # refine reasoning
            if self.y_mode == "every":
                y = self.net(y + z)
        if self.y_mode == "once":
            y = self.net(y + z)  # refine the answer (TRM-faithful)
        return y, z

    def _deep_recursion(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():  # no-grad warmup rounds
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(x, y, z)
        y, z = self._latent_recursion(x, y, z)  # one grad round (full n_inner)
        return y, z

    def forward(self, x: torch.Tensor) -> TRMOutput:
        B, N, _ = x.shape
        y = self.y0.expand(B, N, -1).contiguous()
        z = self.z0.expand(B, N, -1).contiguous()
        logits: list[torch.Tensor] = []
        qs: list[torch.Tensor] = []
        for _ in range(self.n_sup):
            y, z = self._deep_recursion(x, y, z)
            yn = self.head_norm(y)
            logits.append(self.head(yn))
            qs.append(self.q_head(yn.mean(dim=1)).squeeze(-1))
            y, z = y.detach(), z.detach()  # carry detached between steps
        return TRMOutput(logits=logits, q=qs, y=y, z=z)


def _sinusoidal_pos(n: int, d: int, device) -> torch.Tensor:
    """Size-agnostic positional encoding over the variable mix-frame axis."""
    pos = torch.arange(n, device=device).float().unsqueeze(1)
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(-torch.log(torch.tensor(10000.0)) * i / d)
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class TRMDecoder(nn.Module):
    """Drop-in trajectory decoder: same `(sim, feat_kind, ref_valid)` call as the
    conv `TrajectoryDecoder`, but emits per-frame logits over the fixed OFFSET
    vocabulary (`TRMOutput`, one grid per supervision step). The front re-indexes
    `sim` into offset coordinates, embeds each mix frame's offset-evidence row,
    adds positional encoding, and hands the token sequence to `TRMCore`."""

    def __init__(
        self,
        vocab: OffsetVocab,
        *,
        d_model: int = 32,
        n_feat_kinds: int = 2,
        **trm_kwargs,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        # per-feature-kind affine on the match channel (chroma vs HuBERT scale),
        # mirroring the conv decoder's kind_affine
        self.kind_affine = nn.Embedding(n_feat_kinds, 2)
        nn.init.constant_(self.kind_affine.weight[:, 0], 1.0)
        nn.init.constant_(self.kind_affine.weight[:, 1], 0.0)
        self.embed = nn.Linear(2 * vocab.n_offsets, d_model)
        self.core = TRMCore(d_model=d_model, vocab_size=vocab.size, **trm_kwargs)

    def forward(
        self,
        sim: torch.Tensor,  # (B, 2, Tm, Tr)
        feat_kind: torch.Tensor,  # (B,)
        ref_valid: torch.Tensor,  # (B, Tr) — reserved; evidence self-masks range
    ) -> TRMOutput:
        aff = self.kind_affine(feat_kind)  # (B, 2)
        scale = aff[:, 0].view(-1, 1, 1, 1)
        bias = aff[:, 1].view(-1, 1, 1, 1)
        base = torch.cat([sim[:, :1] * scale + bias, sim[:, 1:]], dim=1)
        ev = sim_to_offset_evidence(base, self.vocab)  # (B, 2, Tm, n_off)
        B, _, Tm, n_off = ev.shape
        tok = ev.permute(0, 2, 1, 3).reshape(B, Tm, 2 * n_off)
        x = self.embed(tok)  # (B, Tm, d)
        x = x + _sinusoidal_pos(Tm, self.d_model, x.device).unsqueeze(0)
        return self.core(x)


def sim_to_offset_evidence(sim: torch.Tensor, vocab: OffsetVocab) -> torch.Tensor:
    """Re-index a `(B, C, Tm, Tr)` cross-similarity grid into fixed offset
    coordinates `(B, C, Tm, n_offsets)`: entry `[.., t, i]` = `sim[.., t, t+bin]`
    where `bin = lo_bin + i`. Out-of-range ref positions read 0."""
    B, C, Tm, Tr = sim.shape
    n_off = vocab.n_offsets
    t = torch.arange(Tm, device=sim.device).view(Tm, 1)
    off = torch.arange(n_off, device=sim.device).view(1, n_off) + vocab.lo_bin
    ref = t + off  # (Tm, n_off) target ref bin per (frame, offset)
    valid = (ref >= 0) & (ref < Tr)
    ref_c = ref.clamp(0, Tr - 1)
    idx = ref_c.view(1, 1, Tm, n_off).expand(B, C, Tm, n_off)
    ev = torch.gather(sim, dim=3, index=idx)
    return ev * valid.view(1, 1, Tm, n_off)


def trm_offset_targets(
    target_idx: torch.Tensor,  # (B, Tm) ref-bin index per frame, -1 = ignore
    target_null: torch.Tensor,  # (B, Tm) bool
    mix_valid: torch.Tensor,  # (B, Tm) bool
    vocab: OffsetVocab,
) -> torch.Tensor:
    """Convert the conv path's ref-bin targets to per-frame OFFSET-class labels
    (offset = ref_bin - frame). NULL frames -> null_index; ignore / invalid ->
    IGNORE_INDEX; positions clamped into the fixed vocab."""
    from .offset_coords import IGNORE_INDEX

    B, Tm = target_idx.shape
    t = torch.arange(Tm, device=target_idx.device).view(1, Tm)
    off_bin = target_idx - t
    idx = (off_bin - vocab.lo_bin).clamp(0, vocab.n_offsets - 1)
    labels = torch.full(
        (B, Tm), IGNORE_INDEX, dtype=torch.long, device=target_idx.device
    )
    pos = (target_idx >= 0) & mix_valid & ~target_null
    labels[pos] = idx[pos]
    labels[target_null & mix_valid] = vocab.null_index
    return labels


def trm_offset_ce(out: TRMOutput, labels: torch.Tensor) -> torch.Tensor:
    """Deep-supervision cross-entropy over offset labels: mean CE across every
    improvement step's logits (IGNORE_INDEX frames excluded)."""
    from .offset_coords import IGNORE_INDEX

    V = out.logits[-1].shape[-1]
    terms = [
        F.cross_entropy(
            lg.reshape(-1, V), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        for lg in out.logits
    ]
    return torch.stack(terms).mean()


def trm_decode_segments(
    out: TRMOutput, bin_s: float, vocab: OffsetVocab
) -> list[tuple[float, float, float]]:
    """Argmax the final improvement step's offset logits and collapse runs of
    equal offset into `(mix_start, ref_start, ref_end)` segments — the same
    coordinate `trajectory_acc` consumes."""
    import numpy as np

    from .offset_coords import offset_labels_to_ref_bins
    from .targets import frames_to_segments

    logits = out.logits[-1]  # (B, Tm, V)
    labels = logits[0].argmax(dim=-1).cpu().numpy().astype(np.int64)
    ref_bin, null_mask = offset_labels_to_ref_bins(labels, vocab)
    return frames_to_segments(ref_bin, null_mask, bin_s)


__all__ = [
    "TRMCore",
    "TRMDecoder",
    "TRMOutput",
    "sim_to_offset_evidence",
    "trm_offset_targets",
    "trm_offset_ce",
    "trm_decode_segments",
]

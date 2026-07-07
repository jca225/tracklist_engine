"""Structured decode over the learned score grid.

Greedy per-frame argmax throws away sequence structure — no jump costs, no
monotonic prior — which is exactly what `path_decode._viterbi` already
encodes (stay-on-diagonal free, forward jump cheap, backward jump dear).
This bridges the two: transform the model's (Tm, Tr) grid into the
clip-start-offset coordinates `_viterbi` expects, decode, then overlay the
model's NULL logit (a frame goes NULL where silence out-scores the decoded
path by `null_margin`).

The jump penalty `lam` lives on the LOGIT scale of a particular checkpoint,
not the matched-filter scale path_decode tunes for — sweep it on the TRAIN
split (never eval) via train.py --lam-sweep.
"""

from __future__ import annotations

import numpy as np
import torch

from workspaces.alignment_prototype.path_decode import _viterbi

from .targets import frames_to_segments

_NEG = -1e9


def viterbi_segments(
    logits: torch.Tensor,
    bin_s: float,
    lam: float = 4.0,
    back_ratio: float = 3.0,
    null_margin: float = 0.0,
) -> list[tuple[float, float, float]]:
    """(Tm, Tr+1) logits -> segment triples via offset-state Viterbi.

    States are clip-start offsets r0 = ref_bin - mix_frame (the diagonal
    index): staying on one diagonal is free, a forward jump costs `lam`, a
    backward jump (loop/replay) costs `lam * back_ratio` — the monotonic
    prior from path_decode, applied to learned emissions.
    """
    g = logits[:, :-1].detach().cpu().numpy().astype(np.float64)  # (Tm, Tr)
    null = logits[:, -1].detach().cpu().numpy().astype(np.float64)  # (Tm,)
    tm, tr = g.shape
    if tm == 0 or tr == 0:
        return []
    # reward[t, r0] = g[t, r0 + t]; diagonals that run off the ref are dead
    reward = np.full((tm, tr), _NEG)
    for t in range(tm):
        n = tr - t
        if n > 0:
            reward[t, :n] = g[t, t:]
    _, path_r0 = _viterbi(reward, lam_fwd=lam, lam_back=lam * back_ratio)
    ref_bin = path_r0 + np.arange(tm)
    ref_bin = np.clip(ref_bin, 0, tr - 1)
    on_path = g[np.arange(tm), ref_bin]
    null_mask = null > (on_path + null_margin)
    return frames_to_segments(ref_bin, null_mask, bin_s)

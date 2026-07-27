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

from dataclasses import asdict, dataclass
import json
from typing import Any

import numpy as np
import torch

from workspaces.alignment_prototype.path_decode import _viterbi

from .targets import frames_to_segments

_NEG = -1e9

# Softmax masses from a model checkpoint — never a calibrated STRUCTURE belief.
EVIDENCE_KIND_CONV = "uncalibrated_trajectory_logits"
EVIDENCE_KIND_TRM = "uncalibrated_trm_offset_logits"
_EVIDENCE_KINDS = frozenset({EVIDENCE_KIND_CONV, EVIDENCE_KIND_TRM})


@dataclass(frozen=True)
class FrameCandidate:
    """One member of a frame-local, uncalibrated model distribution."""

    ref_bin: int | None
    probability: float


@dataclass(frozen=True)
class FrameEvidence:
    """Compact uncertainty retained for one decoded mix frame."""

    decoded_ref_bin: int | None
    decoded_probability: float
    null_probability: float
    normalized_entropy: float
    top1_top2_margin: float
    candidates: tuple[FrameCandidate, ...]


@dataclass(frozen=True)
class TrajectoryEvidence:
    """Versioned, JSON-safe trajectory-model evidence.

    These are softmax masses from a model checkpoint, not calibrated STRUCTURE
    probabilities.  A held-out-set calibrator must turn this evidence into an
    AxisBelief before it can govern a timeline.
    """

    schema_version: int
    evidence_kind: str
    frames: tuple[FrameEvidence, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> TrajectoryEvidence:
        raw: dict[str, Any] = json.loads(payload)
        if raw.get("schema_version") != 1:
            raise ValueError(
                f"unsupported trajectory evidence schema: {raw.get('schema_version')}"
            )
        kind = raw.get("evidence_kind")
        if kind not in _EVIDENCE_KINDS:
            raise ValueError(f"unexpected trajectory evidence kind: {kind}")
        frames = tuple(
            FrameEvidence(
                decoded_ref_bin=frame["decoded_ref_bin"],
                decoded_probability=float(frame["decoded_probability"]),
                null_probability=float(frame["null_probability"]),
                normalized_entropy=float(frame["normalized_entropy"]),
                top1_top2_margin=float(frame["top1_top2_margin"]),
                candidates=tuple(
                    FrameCandidate(
                        ref_bin=candidate["ref_bin"],
                        probability=float(candidate["probability"]),
                    )
                    for candidate in frame["candidates"]
                ),
            )
            for frame in raw["frames"]
        )
        return cls(
            schema_version=1,
            evidence_kind=str(kind),
            frames=frames,
        )


def trajectory_evidence(
    logits: torch.Tensor,
    decoded_ref_bins: np.ndarray,
    null_mask: np.ndarray,
    *,
    top_k: int = 3,
) -> TrajectoryEvidence:
    """Preserve frame-local candidate mass without claiming calibration."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (mix_frames, ref_bins + NULL)")
    if len(decoded_ref_bins) != logits.shape[0] or len(null_mask) != logits.shape[0]:
        raise ValueError("decoded path and NULL mask must match the frame count")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    probabilities = torch.softmax(
        logits.detach().float().cpu().to(dtype=torch.float64), dim=-1
    )
    p = probabilities.cpu().numpy()
    tr = logits.shape[1] - 1
    denom = np.log(logits.shape[1])
    frames: list[FrameEvidence] = []
    for t, row in enumerate(p):
        order = np.argsort(row)[::-1]
        candidates = tuple(
            FrameCandidate(
                ref_bin=None if int(index) == tr else int(index),
                probability=float(row[index]),
            )
            for index in order[: min(top_k, len(order))]
        )
        selected = tr if bool(null_mask[t]) else int(decoded_ref_bins[t])
        positive = row[row > 0.0]
        entropy = float(-(positive * np.log(positive)).sum() / denom)
        top_margin = float(row[order[0]] - row[order[1]])
        frames.append(
            FrameEvidence(
                decoded_ref_bin=None
                if bool(null_mask[t])
                else int(decoded_ref_bins[t]),
                decoded_probability=float(row[selected]),
                null_probability=float(row[tr]),
                normalized_entropy=entropy,
                top1_top2_margin=top_margin,
                candidates=candidates,
            )
        )
    return TrajectoryEvidence(
        schema_version=1,
        evidence_kind=EVIDENCE_KIND_CONV,
        frames=tuple(frames),
    )


def _decode(
    logits: torch.Tensor,
    lam: float,
    back_ratio: float,
    null_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = logits[:, :-1].detach().cpu().numpy().astype(np.float64)
    null = logits[:, -1].detach().cpu().numpy().astype(np.float64)
    tm, tr = g.shape
    if tm == 0 or tr == 0:
        return g, np.array([], dtype=np.int64), np.array([], dtype=np.bool_)
    reward = np.full((tm, tr), _NEG)
    for t in range(tm):
        n = tr - t
        if n > 0:
            reward[t, :n] = g[t, t:]
    _, path_r0 = _viterbi(reward, lam_fwd=lam, lam_back=lam * back_ratio)
    ref_bin = np.clip(path_r0 + np.arange(tm), 0, tr - 1)
    on_path = g[np.arange(tm), ref_bin]
    null_mask = null > (on_path + null_margin)
    return g, ref_bin, null_mask


def viterbi_segments(
    logits: torch.Tensor,
    bin_s: float,
    lam: float = 4.0,
    back_ratio: float = 3.0,
    null_margin: float = 0.0,
    return_score: bool = False,
) -> list[tuple[float, float, float]] | tuple[list[tuple[float, float, float]], float]:
    """(Tm, Tr+1) logits -> segment triples via offset-state Viterbi.

    States are clip-start offsets r0 = ref_bin - mix_frame (the diagonal
    index): staying on one diagonal is free, a forward jump costs `lam`, a
    backward jump (loop/replay) costs `lam * back_ratio` — the monotonic
    prior from path_decode, applied to learned emissions.

    With ``return_score=True`` also returns a scalar DECODE CONFIDENCE: the mean
    per-frame margin of the decoded path over the frame's average ref emission
    (``on_path - row_mean``), restricted to non-NULL frames. High = the model
    decisively prefers this diagonal over the alternatives; near zero = the grid
    is flat and the path is a coin-flip. It is the gate signal for the ml driver
    (trust learned segments only where the model is sure). Logit-scaled, so the
    threshold is per-checkpoint and swept, not absolute.
    """
    g, ref_bin, null_mask = _decode(logits, lam, back_ratio, null_margin)
    tm, tr = g.shape
    if tm == 0 or tr == 0:
        return ([], 0.0) if return_score else []
    on_path = g[np.arange(tm), ref_bin]
    segs = frames_to_segments(ref_bin, null_mask, bin_s)
    if not return_score:
        return segs
    live = ~null_mask
    if live.any():
        margin = on_path[live] - g[live].mean(axis=1)
        score = float(np.mean(margin))
    else:
        score = 0.0
    return segs, score


def decode_with_evidence(
    logits: torch.Tensor,
    bin_s: float,
    lam: float = 4.0,
    back_ratio: float = 3.0,
    null_margin: float = 0.0,
    *,
    top_k: int = 3,
) -> tuple[list[tuple[float, float, float]], TrajectoryEvidence]:
    """Decode segments and retain the model's frame-local candidate evidence."""
    _, ref_bin, null_mask = _decode(logits, lam, back_ratio, null_margin)
    if logits.shape[0] == 0 or logits.shape[1] <= 1:
        return [], TrajectoryEvidence(1, EVIDENCE_KIND_CONV, ())
    segments = frames_to_segments(ref_bin, null_mask, bin_s)
    return segments, trajectory_evidence(logits, ref_bin, null_mask, top_k=top_k)

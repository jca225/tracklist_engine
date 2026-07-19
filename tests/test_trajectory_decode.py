"""Viterbi bridge + diagonal accumulation channels (pure, no audio)."""

from __future__ import annotations

import pytest

# Heavy deps (torch) are excluded from requirements-ci.txt — this test must
# skip gracefully in the lightweight CI rather than break collection. The
# workspaces imports below pull torch transitively, so skip BEFORE them.
pytest.importorskip("torch")

import torch  # noqa: E402

from workspaces.alignment_prototype.trajectory.decode import (  # noqa: E402
    viterbi_segments,
)
from workspaces.alignment_prototype.trajectory.model import diag_mean  # noqa: E402

BIN = 0.5


def _grid_logits(
    tm: int, tr: int, path: list[int], null: float = -10.0
) -> torch.Tensor:
    """Logits peaked (+5) at the given ref bin per frame, NULL flat at `null`."""
    g = torch.zeros(tm, tr + 1)
    g[:, tr] = null
    for t, r in enumerate(path):
        g[t, r] = 5.0
    return g


class TestViterbiSegments:
    def test_clean_diagonal_is_one_segment(self) -> None:
        path = list(range(80, 100))
        segs = viterbi_segments(_grid_logits(20, 200, path), BIN, lam=4.0)
        assert len(segs) == 1
        assert segs[0] == (0.0, 40.0, 50.0)

    def test_jump_survives_when_evidence_strong(self) -> None:
        path = list(range(80, 90)) + list(range(300, 310))
        segs = viterbi_segments(_grid_logits(20, 400, path), BIN, lam=4.0)
        assert len(segs) == 2
        assert segs[1][1] == 150.0  # jumped to ref bin 300

    def test_high_lam_suppresses_spurious_jump(self) -> None:
        # one noisy frame off-diagonal must not split the segment
        path = list(range(80, 100))
        g = _grid_logits(20, 200, path)
        g[10, 150] = 6.0  # spurious spike, barely beats the diagonal
        segs = viterbi_segments(g, BIN, lam=8.0)
        assert len(segs) == 1  # staying beats one frame's +1 logit gain

    def test_null_overlay(self) -> None:
        path = list(range(80, 100))
        g = _grid_logits(20, 200, path)
        g[15:, 200] = 20.0  # silence out-scores the path on the tail
        segs = viterbi_segments(g, BIN, lam=4.0)
        assert len(segs) == 1
        assert segs[0][2] == 47.5  # segment ends where NULL takes over


class TestDiagMean:
    def test_constant_diagonal_preserved(self) -> None:
        # a perfect diagonal line keeps value 1 under diagonal averaging
        tm = tr = 12
        sim = torch.zeros(1, 1, tm, tr)
        for i in range(tm):
            sim[0, 0, i, i] = 1.0
        out = diag_mean(sim, 4)
        assert torch.isclose(out[0, 0, 0, 0], torch.tensor(1.0))
        # off-diagonal stays 0 (no leakage across diagonals)
        assert out[0, 0, 0, 5] == 0.0

    def test_isolated_spike_diluted(self) -> None:
        sim = torch.zeros(1, 1, 12, 12)
        sim[0, 0, 3, 3] = 1.0
        out = diag_mean(sim, 4)
        assert torch.isclose(out[0, 0, 3, 3], torch.tensor(0.25))

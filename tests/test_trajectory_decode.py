"""Viterbi bridge + diagonal accumulation channels (pure, no audio)."""

from __future__ import annotations

import pytest

# Heavy deps (torch) are excluded from requirements-ci.txt — this test must
# skip gracefully in the lightweight CI rather than break collection. The
# workspaces imports below pull torch transitively, so skip BEFORE them.
pytest.importorskip("torch")

import torch  # noqa: E402

from workspaces.alignment_prototype.trajectory.decode import (  # noqa: E402
    TrajectoryEvidence,
    decode_with_evidence,
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


class TestTrajectoryEvidence:
    def test_peaked_and_flat_logits_retain_different_uncertainty(self) -> None:
        path = list(range(4, 8))
        peaked = _grid_logits(4, 12, path)
        flat = torch.zeros(4, 13)

        peaked_segments, peaked_evidence = decode_with_evidence(peaked, BIN)
        _, flat_evidence = decode_with_evidence(flat, BIN)

        assert peaked_segments
        assert peaked_evidence.evidence_kind == "uncalibrated_trajectory_logits"
        assert peaked_evidence.frames[0].normalized_entropy < 0.5
        assert peaked_evidence.frames[0].top1_top2_margin > 0.5
        assert flat_evidence.frames[0].normalized_entropy == pytest.approx(1.0)
        assert flat_evidence.frames[0].top1_top2_margin == pytest.approx(0.0)

    def test_null_dominance_is_explicit(self) -> None:
        logits = _grid_logits(3, 8, [2, 3, 4])
        logits[1, -1] = 20.0

        segments, evidence = decode_with_evidence(logits, BIN)

        assert segments
        assert evidence.frames[1].decoded_ref_bin is None
        assert evidence.frames[1].null_probability > 0.99
        assert evidence.frames[1].candidates[0].ref_bin is None

    def test_json_round_trip_is_deterministic(self) -> None:
        logits = _grid_logits(3, 8, [2, 3, 4])
        _, evidence = decode_with_evidence(logits, BIN)

        payload = evidence.to_json()
        restored = TrajectoryEvidence.from_json(payload)

        assert restored == evidence
        assert restored.to_json() == payload

    def test_rejects_mislabeled_or_future_schema(self) -> None:
        with pytest.raises(ValueError, match="unsupported"):
            TrajectoryEvidence.from_json(
                '{"schema_version":2,"evidence_kind":"uncalibrated_trajectory_logits","frames":[]}'
            )
        with pytest.raises(ValueError, match="unexpected"):
            TrajectoryEvidence.from_json(
                '{"schema_version":1,"evidence_kind":"structure_probability","frames":[]}'
            )


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

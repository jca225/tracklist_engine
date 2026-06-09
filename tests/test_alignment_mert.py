"""Tests for MERT feature wiring and learned aligner head."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # mert_model needs torch; absent in deps-light CI

from workspaces.alignment_prototype.mert_features import build_examples, candidate_list
from workspaces.alignment_prototype.mert_model import TrainConfig, train_head
from workspaces.alignment_prototype.mert_store import MertSeries
from workspaces.alignment_prototype.records import SlotCandidate, SpanTarget


def _synthetic_series(n: int, dim: int, *, offset: float = 0.0) -> MertSeries:
    t = np.arange(n, dtype=np.float64) * 2.0
    vecs = np.zeros((n, dim), dtype=np.float32)
    vecs[:, 0] = offset
    vecs[:, 1] = np.linspace(0, 1, n, dtype=np.float32) + offset
    return MertSeries(start_s=t, end_s=t + 1.0, vectors=vecs)


def test_candidate_list_prefers_slot_pool():
    pools = {"002": (SlotCandidate("a", "regular"), SlotCandidate("b", "regular"))}
    ids, stems = candidate_list("002", pools, ("x", "y"))
    assert ids == ("a", "b")
    assert stems == ("regular", "regular")


def test_build_examples_and_train_head():
    dim = 16
    mix = _synthetic_series(200, dim)
    refs = {
        "track_a": _synthetic_series(80, dim, offset=1.0),
        "track_b": _synthetic_series(80, dim, offset=5.0),
    }
    targets = (
        SpanTarget(
            slot_label="002",
            recording_id="track_a",
            claimed_stem="regular",
            set_start_s=20.0,
            set_end_s=40.0,
            ref_start_s=4.0,
            ref_end_s=12.0,
            tempo_ratio=1.0,
            pitch_shift_semi=0,
            label="A",
        ),
        SpanTarget(
            slot_label="003",
            recording_id="track_b",
            claimed_stem="regular",
            set_start_s=60.0,
            set_end_s=80.0,
            ref_start_s=6.0,
            ref_end_s=14.0,
            tempo_ratio=1.0,
            pitch_shift_semi=0,
            label="B",
        ),
    )
    pools = {
        "002": (SlotCandidate("track_a", "regular"), SlotCandidate("track_b", "regular")),
        "003": (SlotCandidate("track_a", "regular"), SlotCandidate("track_b", "regular")),
    }
    examples = build_examples(targets, mix, refs, pools, search_margin_s=30.0, max_negatives=1)
    assert len(examples) == 2
    head = train_head(examples, cfg=TrainConfig(epochs=80, lr=5e-2), device="cpu")
    mix_seg = np.stack([ex.mix_segment for ex in examples])
    ref_seg = np.stack([ex.ref_segments for ex in examples])
    import torch

    with torch.no_grad():
        logits = head.identity_logits(torch.from_numpy(mix_seg), torch.from_numpy(ref_seg))
    assert logits.shape == (2, 2)
    assert int(logits[0].argmax()) == 0
    assert int(logits[1].argmax()) == 1

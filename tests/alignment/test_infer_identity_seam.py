"""End-to-end (GPU-free) test of the infer.py Phase 1B identity seam.

Injects L3/L22 features via a saved IdentityFeatureBundle (the runner's output)
and drives _apply_open_set_identity directly, proving candidate_pool +
identity_override + the bundle compose through the seam to rewrite recording_id.
"""

from __future__ import annotations

import types

import numpy as np

from alignment import infer
from alignment.extract_stem_mert import IdentityFeatureBundle
from alignment.records import SpanPrediction


def _pred(slot, rid):
    return SpanPrediction(slot, rid, "regular", 0.0, 10.0, 0.0, None)


def test_seam_overrides_recording_id_from_bundle(tmp_path):
    rival = np.array([[0.0], [1.0]], dtype=np.float32)
    claim = np.array([[1.0], [0.0]], dtype=np.float32)
    IdentityFeatureBundle(
        set_id="s",
        queries={"1": rival},  # span 1's audio looks like the rival, not the claim
        refs={"claimed": claim, "rival": rival},
        set_pool_by_stem={"instrumental": ("claimed", "rival"), "acappella": ()},
        spans={},
    ).save(tmp_path)

    preds = (_pred("1", "claimed"),)
    rows = ({"slot_label": "1", "recording_id": "claimed", "claimed_stem": "regular"},)
    args = types.SimpleNamespace(
        identity_cache_dir=tmp_path, identity_tau=0.3, identity_floor=0.5, set_id="s"
    )
    new_preds, id_source, id_prov, counts = infer._apply_open_set_identity(
        preds, rows, args
    )
    assert new_preds[0].recording_id == "rival"  # claim overridden by audio
    assert id_source[0] == "open_set_mert"
    assert id_prov[0]["decision"] == "override"
    assert id_prov[0]["claim"] == "claimed"


def test_seam_abstains_null_recording_id(tmp_path):
    # Claimless span, nothing scorable -> abstain -> recording_id nulled (persisted).
    IdentityFeatureBundle(
        set_id="s",
        queries={},
        refs={},
        set_pool_by_stem={"instrumental": (), "acappella": ()},
        spans={},
    ).save(tmp_path)
    preds = (_pred("1", None),)
    rows = ({"slot_label": "1", "recording_id": None, "claimed_stem": "regular"},)
    args = types.SimpleNamespace(
        identity_cache_dir=tmp_path, identity_tau=0.3, identity_floor=0.5, set_id="s"
    )
    new_preds, id_source, id_prov, _ = infer._apply_open_set_identity(preds, rows, args)
    assert new_preds[0].recording_id is None
    assert id_source[0] == "abstain"


def test_seam_keeps_claim_when_audio_agrees(tmp_path):
    claim = np.array([[1.0], [0.0]], dtype=np.float32)
    rival = np.array([[0.0], [1.0]], dtype=np.float32)
    IdentityFeatureBundle(
        set_id="s",
        queries={"1": claim},  # audio matches the claim
        refs={"claimed": claim, "rival": rival},
        set_pool_by_stem={"instrumental": ("claimed", "rival"), "acappella": ()},
        spans={},
    ).save(tmp_path)
    preds = (_pred("1", "claimed"),)
    rows = ({"slot_label": "1", "recording_id": "claimed", "claimed_stem": "regular"},)
    args = types.SimpleNamespace(
        identity_cache_dir=tmp_path, identity_tau=0.3, identity_floor=0.5, set_id="s"
    )
    new_preds, id_source, _, _ = infer._apply_open_set_identity(preds, rows, args)
    assert new_preds[0].recording_id == "claimed"
    assert id_source[0] == "claim"

from __future__ import annotations

import pytest

pytest.importorskip("librosa")  # audio_enrich → recon_probe

from alignment.candidate_arbiter.audio_enrich import (
    with_verification,
)
from alignment.candidate_arbiter.audio_verify import (
    PairVerification,
)
from alignment.candidate_arbiter.schema import (
    CandidateEvidence,
    PlacementCandidate,
)


def test_verification_maps_into_candidate_evidence() -> None:
    candidate = PlacementCandidate(
        set_id="set",
        slot_label="1",
        recording_id="rid",
        claimed_stem="regular",
        source="mert_decode",
        rank=0,
        set_start_s=10.0,
        ref_start_s=None,
        native_confidence=None,
        evidence=CandidateEvidence(baseline_delta_s=2.0),
    )
    verification = PairVerification(
        match_mean=0.8,
        match_p10=0.5,
        margin=0.3,
        continuity=0.75,
        support_bins=24,
        background_count=4,
    )

    enriched = with_verification(candidate, verification)

    assert enriched.evidence.verify_match_mean == 0.8
    assert enriched.evidence.verify_match_p10 == 0.5
    assert enriched.evidence.verify_margin == 0.3
    assert enriched.evidence.verify_continuity == 0.75
    assert enriched.evidence.verify_support_bins == 24
    assert enriched.evidence.verify_background_count == 4

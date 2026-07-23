"""Candidate binding generation for GT completion (abstained-appearance recovery).

Each abstained GT row already carries the human-labeled song NAME (from the
.als) + timing, but no `track_id`. We propose candidate recording_ids from (a)
name-match to the catalog and (b) the aligner's prediction at that time, rank
them, and hand them to the spectrogram UI for human confirmation. Truth comes
from the human's spectrogram verdict, so it never inflates the aligner score.
"""

from __future__ import annotations

from eda.alignment.spectrogram_review.gt_completion import (
    aligner_candidate,
    name_candidates,
    norm_name,
    apply_verdicts,
    rank_candidates,
    to_render_row,
)


def test_apply_verdicts_fills_confirmed_abstained_rows_only():
    gt = [
        {
            "slot_label": "007",
            "track": "Calvin Harris - Outside (Acappella)",
            "claimed_stem": "acappella",
        },  # abstained (no track_id)
        {"slot_label": "013", "track": "CCR - Fortunate Son", "track_id": "already"},
        {
            "slot_label": "021",
            "track": "Some Song",
            "claimed_stem": "regular",
        },  # abstained, no verdict
    ]
    verdicts = [
        {
            "slot": "007",
            "name": "Calvin Harris - Outside (Acappella)  [name+aligner]",
            "recording_id": "r_out",
            "verdict": "confirm",
        },
        {
            "slot": "007",
            "name": "Calvin Harris - Outside (Acappella)  [name-multi]",
            "recording_id": "r_wrong",
            "verdict": "reject",
        },
    ]
    out, stats = apply_verdicts(gt, verdicts)
    # confirmed abstained row is filled with the confirmed recording
    assert out[0]["track_id"] == "r_out"
    assert out[0]["id_source"] == "human_spectrogram"
    # already-bound row untouched; no-verdict abstained row stays abstained
    assert out[1]["track_id"] == "already"
    assert "track_id" not in out[2] or not out[2].get("track_id")
    assert stats["filled"] == 1 and stats["confirmed_verdicts"] == 1


def test_apply_verdicts_rejects_do_not_fill():
    gt = [{"slot_label": "021", "track": "Some Song", "claimed_stem": "regular"}]
    verdicts = [
        {
            "slot": "021",
            "name": "Some Song  [aligner-only]",
            "recording_id": "r_x",
            "verdict": "reject",
        }
    ]
    out, stats = apply_verdicts(gt, verdicts)
    assert not out[0].get("track_id")
    assert stats["filled"] == 0


def test_to_render_row_maps_candidate_into_spectrogram_schema():
    abst = {
        "slot_label": "007",
        "track": "Calvin Harris - Outside (Acappella)",
        "claimed_stem": "acappella",
        "set_start_s": 222.6,
        "set_end_s": 290.9,
        "ref_start_s": 8.1,
        "ref_end_s": 51.5,
    }
    row = to_render_row(abst, "r_out", "name+aligner", "BB12", "1fsnxchk", 3)
    # renders as a "predicted == GT candidate" span so the boxes overlap
    assert row["recording_id"] == "r_out"
    assert row["gt_set_start_s"] == row["pred_set_start_s"] == 222.6
    assert row["gt_ref_start_s"] == row["pred_ref_start_s"] == 8.1
    assert row["gt_stem"] == "acappella"
    assert row["slot"] == "007"
    assert row["span_class"]  # non-empty so the renderer includes it
    assert "name+aligner" in row["name"]  # confidence surfaced to the reviewer
    assert row["set_id"] == "1fsnxchk"


def test_norm_name_strips_qualifiers_and_punct():
    assert norm_name("Calvin Harris - Outside (Acappella)") == norm_name(
        "calvin harris  outside"
    )
    assert norm_name("Two Friends - Emily (Remix)") == norm_name("Two Friends Emily")


def test_name_candidates_exact_and_unique():
    recs = [
        {"rid": "r_out", "full": "Calvin Harris ft. Ellie Goulding - Outside"},
        {"rid": "r_emily", "full": "Two Friends - Emily"},
        {"rid": "r_other", "full": "Some Other Song"},
    ]
    c = name_candidates("Calvin Harris - Outside (Acappella)", recs)
    assert c == ["r_out"]


def test_name_candidates_none_when_no_overlap():
    recs = [{"rid": "r1", "full": "Totally Different"}]
    assert name_candidates("Snakehips - All My Friends", recs) == []


def test_aligner_candidate_is_recording_overlapping_in_time():
    spans = [
        {"recording_id": "r_a", "set_start_s": 100.0, "set_end_s": 140.0},
        {"recording_id": "r_b", "set_start_s": 500.0, "set_end_s": 560.0},
    ]
    row = {"set_start_s": 505.0, "set_end_s": 550.0}
    assert aligner_candidate(row, spans) == "r_b"
    row2 = {"set_start_s": 900.0, "set_end_s": 950.0}
    assert aligner_candidate(row2, spans) is None


def test_rank_prefers_name_and_aligner_agreement():
    # when the name-match and the aligner agree, that recording ranks first
    # with the highest confidence.
    ranked = rank_candidates(name_ids=["r_x", "r_y"], aligner_id="r_y")
    assert ranked[0]["recording_id"] == "r_y"
    assert ranked[0]["confidence"] == "name+aligner"
    # the other name candidate is still offered, lower.
    assert [c["recording_id"] for c in ranked] == ["r_y", "r_x"]


def test_rank_offers_aligner_only_when_no_name_match():
    ranked = rank_candidates(name_ids=[], aligner_id="r_z")
    assert ranked == [{"recording_id": "r_z", "confidence": "aligner-only"}]


def test_rank_unique_name_match_is_high_confidence():
    ranked = rank_candidates(name_ids=["r_solo"], aligner_id=None)
    assert ranked == [{"recording_id": "r_solo", "confidence": "name-unique"}]


def test_rank_empty_when_no_candidates():
    assert rank_candidates(name_ids=[], aligner_id=None) == []

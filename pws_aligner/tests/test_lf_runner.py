from __future__ import annotations

"""Tests for the per-span LF runner (B1).

The runner is what turns "4 audio probes" into real PWS breadth: it calls
every enabled LF per span and concatenates their votes — keeping the two
axes separate:

  continuous (offset EM)   audio probes + cue_time (converted to the
                           relative-offset frame) → ContinuousLabelModel
  categorical              operation / identity / ordering / LLM votes →
                           collected + summarized, NEVER fed to the offset EM
"""

import json

from pws_aligner.continuous_model import ContinuousLabelModel
from pws_aligner.fit_corpus import fit_cohort, pool_set_lf_votes
from pws_aligner.identity_lf import ClaimedIdentityVote
from pws_aligner.lf_runner import (
    SpanLfVotes,
    categorical_lf_summary,
    run_all_spans,
    run_span_lfs,
)
from pws_aligner.operations import OperationVote
from pws_aligner.ordering_lf import OrderingVote
from pws_aligner.tracklist_lf import CUE_BIAS_S
from pws_aligner.votes import AbstainReason, Vote


def _span_doc(
    span_id: str = "1",
    set_start_s: float = 100.0,
    cue: float | None = 101.0,
    claimed_stem: str | None = "acappella",
) -> dict:
    doc = {
        "span_id": span_id,
        "slot_label": span_id,
        "recording_id": "rec_1",
        "set_start_s": set_start_s,
        "set_end_s": set_start_s + 60.0,
        "ref_start_s": 0.0,
        "ref_end_s": 60.0,
        "name": "Track",
        "probes": [
            {
                "probe": "fp",
                "recording_id": "rec_1",
                "offset_s": -99.0,
                "confidence": 0.9,
                "abstain": False,
                "features": [1.0],
            }
        ],
    }
    if cue is not None:
        doc["cue_anchor_s"] = cue
    if claimed_stem is not None:
        doc["claimed_stem"] = claimed_stem
    return doc


def test_audio_probe_votes_on_continuous_axis():
    out = run_span_lfs(_span_doc(), set_id="s")
    assert isinstance(out, SpanLfVotes)
    probes = [v.probe for v in out.continuous if not v.abstained]
    assert "fp" in probes
    assert all(isinstance(v, Vote) for v in out.continuous)


def test_cue_vote_converted_to_relative_offset_frame():
    """cue_anchor_s → a cue_time Vote at offset −(cue − bias), keyed on recording_id."""
    out = run_span_lfs(_span_doc(cue=101.0), set_id="s")
    cue_votes = [v for v in out.continuous if v.probe == "cue_time"]
    assert len(cue_votes) == 1
    v = cue_votes[0]
    assert not v.abstained
    assert v.recording_id == "rec_1"
    assert abs(v.offset_s - (-(101.0 - CUE_BIAS_S))) < 1e-9


def test_no_cue_yields_abstaining_cue_vote():
    out = run_span_lfs(_span_doc(cue=None), set_id="s")
    cue_votes = [v for v in out.continuous if v.probe == "cue_time"]
    assert len(cue_votes) == 1
    assert cue_votes[0].abstained
    assert cue_votes[0].reason == AbstainReason.NO_DATA


def test_claimed_identity_on_categorical_axis():
    out = run_span_lfs(_span_doc(claimed_stem="acappella"), set_id="s")
    ident = [v for v in out.categorical if isinstance(v, ClaimedIdentityVote)]
    assert len(ident) == 1
    assert not ident[0].abstained


def test_categorical_never_leaks_into_continuous():
    """The offset EM must see ONLY Vote objects — no categorical labels."""
    out = run_span_lfs(_span_doc(), set_id="s")
    assert all(isinstance(v, Vote) for v in out.continuous)
    assert not any(isinstance(v, Vote) for v in out.categorical)


def test_operation_votes_passed_through_to_categorical():
    op = OperationVote(
        lf="echo_tail",
        span_id="1",
        label="echo",
        confidence=0.8,
        abstained=False,
        reason=AbstainReason.NONE,
    )
    out = run_span_lfs(_span_doc(), set_id="s", operation_votes=(op,))
    assert op in out.categorical


def test_run_all_spans_emits_ordering_votes_from_neighbours():
    docs = [
        _span_doc("1", set_start_s=10.0),
        _span_doc("2", set_start_s=100.0),
        _span_doc("3", set_start_s=50.0),  # backward step vs span 2
    ]
    all_votes = run_all_spans(docs, set_id="s")
    assert len(all_votes) == 3

    def ordering(sv: SpanLfVotes) -> OrderingVote:
        matches = [v for v in sv.categorical if isinstance(v, OrderingVote)]
        assert len(matches) == 1
        return matches[0]

    assert ordering(all_votes[1]).label == "out_of_order"  # next neighbour behind
    assert ordering(all_votes[2]).label == "out_of_order"  # behind its prev


def test_categorical_lf_summary_counts():
    docs = [_span_doc("1", set_start_s=10.0), _span_doc("2", set_start_s=100.0)]
    all_votes = run_all_spans(docs, set_id="s")
    summary = categorical_lf_summary(all_votes)
    assert "claimed_axes_identity" in summary
    assert "ordering_soft" in summary
    assert summary["claimed_axes_identity"]["fired"] == 2
    assert summary["ordering_soft"]["fired"] == 2


def test_pool_set_lf_votes_full_lf_set(tmp_path):
    docs = [
        _span_doc("1", set_start_s=10.0, cue=11.0),
        _span_doc("2", set_start_s=100.0, cue=100.5),
    ]
    (tmp_path / "aaa_probe_votes.json").write_text(json.dumps(docs))
    pooled, categorical, missing = pool_set_lf_votes(["aaa", "bbb"], tmp_path)
    assert missing == ["bbb"]
    assert len(pooled) == 2
    # continuous spans include the cue votes
    assert any(v.probe == "cue_time" for span in pooled for v in span)
    # categorical votes collected across spans
    assert any(isinstance(v, ClaimedIdentityVote) for v in categorical)
    assert any(isinstance(v, OrderingVote) for v in categorical)


def test_fit_runs_with_cue_lf_and_learns_its_noise(tmp_path):
    """The pooled continuous spans (audio + cue) fit; cue_time gets learned noise."""
    docs = [
        _span_doc(str(i), set_start_s=10.0 * i + 5.0, cue=10.0 * i + 6.0)
        for i in range(1, 6)
    ]
    (tmp_path / "aaa_probe_votes.json").write_text(json.dumps(docs))
    pooled, _categorical, _missing = pool_set_lf_votes(["aaa"], tmp_path)
    model = fit_cohort(pooled)
    assert isinstance(model, ContinuousLabelModel)
    assert "cue_time" in model.probe_noise()

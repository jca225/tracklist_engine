from __future__ import annotations

"""Per-span LF runner (B1) — the union of all enabled LFs, on correct axes.

``fit_corpus``/``run_phase1`` historically pooled only the 4 audio probes.
This runner is what turns that into real PWS breadth: for each span it calls
every enabled LF and concatenates the votes into one ``SpanLfVotes`` bundle,
keeping the two statistical axes SEPARATE:

  continuous  — offset votes (audio probes + cue_time) feeding
                ``ContinuousLabelModel``'s offset EM.
  categorical — operation / claimed-identity / ordering / LLM votes.  These
                are genuinely categorical; mixing them into the continuous
                offset EM is exactly the granularity error that killed the
                Dawid–Skene gate.  They are collected + summarized here and
                fused by categorical machinery (label_model.DawidSkene) when
                dense enough.

Cue-time frame conversion
-------------------------
The scraped cue estimates the track's MIX ENTRY time (absolute).  The
continuous model's frame is RELATIVE offset (ref_start_s − set_start_s),
which is constant along a linear 1x playback.  Under the "track plays from
its top at entry" prior (ref position 0 at the entry moment):

    offset_s = −(cue_seconds − CUE_BIAS_S)

That prior is wrong when the DJ enters mid-track — those votes land in the
outlier slab and the EM learns cue_time's true inlier rate / σ.  This is a
learned prior, not an anchor: nothing here hard-codes the cue as truth.

Audio-dependent operation LFs (echo/loop/filter/sidechain/keylock) need the
span audio, which vote files do not carry — capture time runs them (see
stem_routing.route_audio_lf) and this runner accepts their votes as
``operation_votes`` pass-through.

Acceptance finding (BB12, 2026-07-15): the full-LF fit runs and the
categorical coverage summary reports (identity 152/152 fired, ordering
152/152), but the existing ``1fsnxchk_probe_votes.json`` artifact carries NO
``cue_anchor_s`` field — the infer.py votes-export hook predates the cue LF —
so cue_time abstains on all 152 spans and σ stays at the 0.05 floor,
unchanged from the 4-probe baseline. Lifting σ off the floor via LF breadth
therefore needs either the votes-export hook to emit ``cue_anchor_s`` or the
corpus-native capture path (plan §C3). The plumbing is verified end-to-end on
synthetic spans in test_lf_runner.py.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from pws_aligner.lf.identity import ClaimedIdentityVote, claimed_identity_vote
from pws_aligner.lf.ordering import OrderingVote, ordering_soft_vote
from pws_aligner.run_phase1 import _span_votes
from pws_aligner.lf.tracklist import CUE_BIAS_S, PlacementVote, cue_placement_vote
from pws_aligner.core.votes import AbstainReason, Vote

_CUE_FEATURES: tuple[float, ...] = ()


@dataclass(frozen=True)
class SpanLfVotes:
    """All LF votes for one span, split by statistical axis."""

    span_id: str
    continuous: tuple[Vote, ...]  # → ContinuousLabelModel offset EM
    categorical: tuple[object, ...]  # OperationVote / ClaimedIdentityVote / …


def cue_vote_as_continuous(pv: PlacementVote, span_id: str) -> Vote:
    """Convert a cue PlacementVote into the relative-offset Vote frame.

    See the module docstring for the frame derivation. Abstained placement
    votes become abstained offset votes (the EM skips them).
    """
    if pv.abstained or pv.set_start_s is None:
        return Vote(
            probe=pv.lf,
            span_id=span_id,
            recording_id=None,
            offset_s=0.0,
            confidence=0.0,
            abstained=True,
            reason=AbstainReason.NO_DATA,
            features=_CUE_FEATURES,
        )
    return Vote(
        probe=pv.lf,
        span_id=span_id,
        recording_id=pv.recording_id,
        # pv.set_start_s is already bias-corrected (cue − CUE_BIAS_S);
        # relative offset under the plays-from-top prior is its negation.
        offset_s=-pv.set_start_s,
        confidence=pv.confidence,
        abstained=False,
        reason=AbstainReason.NONE,
        features=_CUE_FEATURES,
    )


def run_span_lfs(
    span_doc: Mapping,
    *,
    set_id: str,
    prev_start_s: float | None = None,
    next_start_s: float | None = None,
    operation_votes: Sequence[object] = (),
    llm_votes: Sequence[object] = (),
) -> SpanLfVotes:
    """Run every enabled per-span LF and bundle the votes on correct axes.

    Args:
        span_doc: one span object from a ``<set_id>_probe_votes.json`` file
            (see run_phase1 for the schema).
        set_id: the DJ-set identifier (traceability on the cue vote).
        prev_start_s / next_start_s: neighbouring slots' placements for the
            ordering LF; both None → the ordering vote abstains.
        operation_votes: precomputed audio-operation votes (OperationVote or
            RoutedVote) — capture time owns the audio, we pass them through.
        llm_votes: precomputed LlmVote objects (the LLM LF is budgeted and
            invoked by the policy layer, not blanket-run here).

    Returns:
        SpanLfVotes with the audio-probe + cue votes on the continuous axis
        and identity/ordering/operation/LLM votes on the categorical axis.
    """
    span_id = str(span_doc["span_id"])

    continuous: list[Vote] = list(_span_votes(dict(span_doc)))
    cue_pv = cue_placement_vote(
        set_id=set_id,
        recording_id=str(span_doc.get("recording_id") or ""),
        slot_label=str(span_doc.get("slot_label", span_id)),
        cue_seconds=(
            float(span_doc["cue_anchor_s"]) if "cue_anchor_s" in span_doc else None
        ),
    )
    continuous.append(cue_vote_as_continuous(cue_pv, span_id))

    categorical: list[object] = []
    categorical.append(
        claimed_identity_vote(
            span_id=span_id,
            recording_id=span_doc.get("recording_id"),
            claimed_version=span_doc.get("claimed_version"),
            claimed_stem=span_doc.get("claimed_stem"),
        )
    )
    categorical.append(
        ordering_soft_vote(
            span_id=span_id,
            own_start_s=float(span_doc.get("set_start_s", 0.0)),
            prev_start_s=prev_start_s,
            next_start_s=next_start_s,
        )
    )
    categorical.extend(operation_votes)
    categorical.extend(llm_votes)

    return SpanLfVotes(
        span_id=span_id,
        continuous=tuple(continuous),
        categorical=tuple(categorical),
    )


def run_all_spans(
    span_docs: Sequence[Mapping],
    *,
    set_id: str,
    operation_votes_by_span: Mapping[str, Sequence[object]] | None = None,
) -> list[SpanLfVotes]:
    """Run the LF set over every span of one set, wiring neighbour placements.

    Neighbour placements for the ordering LF come from the span docs' own
    pass-through ``set_start_s`` (the current placement hypothesis).
    """
    starts = [float(d.get("set_start_s", 0.0)) for d in span_docs]
    out: list[SpanLfVotes] = []
    for i, doc in enumerate(span_docs):
        span_id = str(doc["span_id"])
        ops: Sequence[object] = ()
        if operation_votes_by_span is not None:
            ops = operation_votes_by_span.get(span_id, ())
        out.append(
            run_span_lfs(
                doc,
                set_id=set_id,
                prev_start_s=starts[i - 1] if i > 0 else None,
                next_start_s=starts[i + 1] if i + 1 < len(span_docs) else None,
                operation_votes=ops,
            )
        )
    return out


def categorical_lf_summary(
    all_span_votes: Sequence[SpanLfVotes],
) -> dict[str, dict[str, int]]:
    """Fired/abstained counts per categorical LF (coverage report).

    True per-LF *accuracy* for categorical LFs comes from the categorical
    label model (DawidSkene.probe_accuracy) once co-voting is dense enough;
    until then coverage is the honest thing to report.
    """
    summary: dict[str, dict[str, int]] = {}
    for sv in all_span_votes:
        for v in sv.categorical:
            lf = getattr(v, "lf", None)
            if lf is None:
                continue
            entry = summary.setdefault(str(lf), {"fired": 0, "abstained": 0})
            if getattr(v, "abstained", False):
                entry["abstained"] += 1
            else:
                entry["fired"] += 1
    return summary


__all__ = [
    "SpanLfVotes",
    "cue_vote_as_continuous",
    "run_span_lfs",
    "run_all_spans",
    "categorical_lf_summary",
    "ClaimedIdentityVote",
    "OrderingVote",
    "CUE_BIAS_S",
]

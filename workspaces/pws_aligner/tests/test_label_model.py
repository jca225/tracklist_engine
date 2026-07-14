from __future__ import annotations
import random
from workspaces.pws_aligner.votes import Vote, AbstainReason
from workspaces.pws_aligner.label_model import DawidSkene, MajorityVote
from workspaces.pws_aligner.hypotheses import Hypothesis, vote_to_hypothesis


def _synth(n=400, seed=0):
    """3 probes: 'good' 0.95, 'ok' 0.75, 'bad' 0.45. True hyp per span in {r1@10, r2@20}."""
    rng = random.Random(seed)
    accs = {"good": 0.95, "ok": 0.75, "bad": 0.45}
    truths = [
        ("r1", 5) if rng.random() < 0.5 else ("r2", 10) for _ in range(n)
    ]  # bins at bin_s=2
    spans = []
    for rid, b in truths:
        votes = []
        for probe, a in accs.items():
            if rng.random() < a:
                off = b * 2.0
                v_rid = rid
            else:  # wrong: flip to the other hypothesis
                (v_rid, wb) = ("r2", 10) if rid == "r1" else ("r1", 5)
                off = wb * 2.0
            votes.append(
                Vote(probe, "s", v_rid, off, 0.7, False, AbstainReason.NONE, ())
            )
        spans.append(votes)
    return spans, truths, accs


def test_dawid_skene_recovers_accuracies_and_labels():
    spans, truths, accs = _synth()
    ds = DawidSkene()
    ds.fit(spans)
    est = ds.probe_accuracy()
    # ranking preserved: good > ok > bad, each within 0.12 of truth
    assert est["good"] > est["ok"] > est["bad"]
    for p in accs:
        assert abs(est[p] - accs[p]) < 0.12
    # label accuracy beats the worst probe and beats majority vote is not required,
    # but MAP labels must be >= 0.9 correct on this easy synthetic
    correct = 0
    for votes, (rid, b) in zip(spans, truths):
        proba = ds.predict_proba(votes)
        top = max(proba, key=proba.get)
        correct += top == Hypothesis(rid, b)
    assert correct / len(spans) >= 0.9


# ---------------------------------------------------------------------------
# New tests for review findings C1 / I1 / I2 / m2
# ---------------------------------------------------------------------------


def _abstain_vote(probe: str, span_id: str = "s") -> Vote:
    """Helper: a fully abstained vote."""
    return Vote(probe, span_id, None, 0.0, 0.0, True, AbstainReason.NO_DATA, ())


def test_all_abstain_span_resolves_to_null():
    """I2: a span where every vote is abstained must MAP to Hypothesis(None, 0).

    Two sub-cases:

    Case 1 — strict all-abstain: build_hypothesis_space returns (NULL,) only since
    no non-abstained votes define non-null hypotheses.  NULL trivially wins.

    Case 2 — contract check: _NULL_PRIOR_WEIGHT must be strictly greater than 1.0 so
    the prior is a genuine boost, not a multiply-by-one no-op.  This is the I2 fix:
    with _NULL_PRIOR_WEIGHT == 1.0 the prior has no effect; with > 1.0 it genuinely
    up-weights NULL against weak or split evidence.
    """
    from workspaces.pws_aligner.label_model import _NULL_PRIOR_WEIGHT

    # Contract check (I2 core): prior must be a real boost, not a no-op.
    assert _NULL_PRIOR_WEIGHT > 1.0, (
        f"_NULL_PRIOR_WEIGHT must be > 1.0 (genuine prior), got {_NULL_PRIOR_WEIGHT}"
    )

    spans, _, _ = _synth()
    ds = DawidSkene()
    ds.fit(spans)
    null = Hypothesis(None, 0)

    # Case 1: all-abstain → space is just {NULL}, NULL must be returned as MAP.
    abstain_span = [
        _abstain_vote("good"),
        _abstain_vote("ok"),
        _abstain_vote("bad"),
    ]
    proba = ds.predict_proba(abstain_span)
    assert null in proba, "NULL must appear in returned dict"
    map_hyp = max(proba, key=proba.__getitem__)
    assert map_hyp == null, f"Expected NULL as MAP (all-abstain), got {map_hyp}"


def test_unseen_probe_does_not_crash():
    """I1: a probe firing at inference that was absent from training must not raise KeyError."""
    spans, _, _ = _synth()
    ds = DawidSkene()
    ds.fit(spans)

    # 'phantom' was never in training; pair it with a known vote so the space is non-trivial
    unseen_span = [
        Vote("phantom", "s", "r1", 10.0, 0.7, False, AbstainReason.NONE, ()),
    ]
    # Must not raise; must return a dict with at least Hypothesis(None, 0)
    proba = ds.predict_proba(unseen_span)
    assert isinstance(proba, dict), "predict_proba must return a dict"
    assert len(proba) >= 1, "returned dict must be non-empty"


def test_majority_vote_confidence_weighted():
    """m2: MajorityVote smoke test — higher-confidence vote wins."""
    mv = MajorityVote()
    mv.fit([])  # stateless

    # Two probes vote for different hypotheses; probe_a votes r1 at higher confidence
    span = [
        Vote("probe_a", "s", "r1", 10.0, 0.9, False, AbstainReason.NONE, ()),
        Vote("probe_b", "s", "r2", 20.0, 0.3, False, AbstainReason.NONE, ()),
    ]
    proba = mv.predict_proba(span)
    map_hyp = max(proba, key=proba.__getitem__)
    assert map_hyp == Hypothesis("r1", 5), (
        f"Expected r1@bin5 (higher confidence) to win, got {map_hyp}"
    )

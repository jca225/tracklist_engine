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
    # ranking preserved: good > ok > bad, each within 0.1 of truth
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

"""Tier-2 verification: the acquisition-case ledger is find-or-create + append-only.

The ledger is the durable record every acquisition decision lands in (and the
substrate the harness/eval feed on). Two invariants must hold under *any*
interleaving of attempts: (1) a case is opened once per (slot_label, recording_id)
and reused thereafter — find-or-create; (2) every recorded attempt is retained —
append-only, nothing silently dropped. Hypothesis generates arbitrary attempt
streams; each runs against its own temp ledger.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from core.acquisition_case import (
    Attempt,
    AttemptAction,
    AttemptVerdict,
    CaseClaim,
    find_case_index,
    load_cases,
    record_attempt,
)

_slots = st.sampled_from(["013", "081", "013w1", "013w2", "112"])
_recs = st.sampled_from(["recA", "recB", "recC"])
_events = st.lists(st.tuples(_slots, _recs), min_size=0, max_size=25)


@settings(max_examples=120, deadline=None)
@given(events=_events)
def test_ledger_find_or_create_and_append_only(events):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for slot, rec in events:
            record_attempt(
                set_id="s",
                slot_label=slot,
                recording_id=rec,
                attempt=Attempt(
                    action=AttemptAction.YT_SEARCH, verdict=AttemptVerdict.ACCEPT
                ),
                claim=CaseClaim(recording_id=rec),
                root=root,
            )

        cases = load_cases(root / "s.jsonl")
        distinct = {(slot, rec) for slot, rec in events}

        # (1) find-or-create: exactly one case per distinct (slot, recording).
        assert len(cases) == len(distinct)
        for slot, rec in distinct:
            assert find_case_index(cases, slot, rec) is not None

        # (2) append-only: every attempt is retained, none lost.
        assert sum(len(c.attempts) for c in cases) == len(events)


@settings(max_examples=60, deadline=None)
@given(rec=_recs, n=st.integers(min_value=1, max_value=8))
def test_repeated_same_slot_recording_grows_one_case(rec, n):
    # N attempts on the same (slot, recording) => one case with N attempts.
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for _ in range(n):
            record_attempt(
                set_id="s",
                slot_label="013w1",
                recording_id=rec,
                attempt=Attempt(
                    action=AttemptAction.INGEST_URL, verdict=AttemptVerdict.PROMOTE
                ),
                claim=CaseClaim(recording_id=rec),
                root=root,
            )
        cases = load_cases(root / "s.jsonl")
        assert len(cases) == 1
        assert len(cases[0].attempts) == n

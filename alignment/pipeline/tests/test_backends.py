"""End-to-end backend: dispatch an actor binding to its driver, and the TRM
skeleton's loud not-implemented pointer. Driver execution over real audio is
integration; dispatch is unit-tested with a fake driver."""

from __future__ import annotations

from pathlib import Path

import pytest

from alignment.pipeline.adapters import backends
from alignment.pipeline.runner import Grain, RunContext
from alignment.pipeline.stages import Pipeline

BB11, BB12 = "2nvzlh2k", "1fsnxchk"


def _ctx(pipeline):
    return RunContext(
        set_id=BB11,
        source_set_id=BB12,
        grain=Grain.COMPOSE,
        split="set",
        seed=0,
        pipeline=pipeline,
    )


def test_backend_dispatches_actor_binding_to_its_driver():
    seen = []

    class FakeDriver:
        name = "fake"

        def align_set(self, sc):
            seen.append(sc.set_id)
            return Path("/tmp/fake_timeline.json")

    be = backends.EndToEndBackend(driver_by_actor={"viterbi": FakeDriver})
    p = Pipeline.of("classical", actor="viterbi")
    out = be.run(p, _ctx(p))
    assert out == Path("/tmp/fake_timeline.json")
    assert seen == [BB11]  # driver received the eval set


def test_backend_builds_loso_setcontext_without_preflight_io():
    seen = []

    class FakeDriver:
        name = "fake"

        def align_set(self, sc):
            seen.append((sc.set_id, sc.source_set_id))
            return Path("/tmp/x.json")

    be = backends.EndToEndBackend(driver_by_actor={"viterbi": FakeDriver})
    p = Pipeline.of("classical", actor="viterbi")
    be.run(p, _ctx(p))
    assert seen == [(BB11, BB12)]  # cross-set source threaded through, no I/O


def test_trm_actor_is_a_loud_skeleton_pointing_at_the_bakeoff():
    be = backends.EndToEndBackend(driver_by_actor={})
    p = Pipeline.of("trm", actor="trm")
    with pytest.raises(NotImplementedError, match="trm_decoder_bakeoff"):
        be.run(p, _ctx(p))

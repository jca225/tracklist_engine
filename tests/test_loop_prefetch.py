"""Unit tests for the WS1 one-slot input-prefetch primitive.

PrefetchSlot is pure orchestration over three injected callables (pick /
pull / hydrate), so it is tested here with plain lambdas — no torch, no
network, no vast_loop import (importing vast_loop would set
TRACKLIST_DISABLE_FK and drag the GPU stack in).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.loop_prefetch import PrefetchFailure, PrefetchItem, PrefetchSlot


def _slot(pick=None, pull=None, hydrate=None) -> PrefetchSlot:
    return PrefetchSlot(
        pick=pick or (lambda skip: (7, "/remote/7.m4a")),
        pull=pull or (lambda tid, remote: Path(f"/local/{tid}.m4a")),
        hydrate=hydrate or (lambda tid, local: {"tid": tid}),
    )


def test_take_returns_item_on_success() -> None:
    slot = _slot()
    slot.start(frozenset())
    item = slot.take()
    assert isinstance(item, PrefetchItem)
    assert item.tid == 7
    assert item.local_audio == Path("/local/7.m4a")
    assert item.asset == {"tid": 7}
    assert item.pull_s >= 0.0


def test_take_returns_none_when_drained() -> None:
    slot = _slot(pick=lambda skip: None)
    slot.start(frozenset())
    assert slot.take() is None


def test_skip_tids_forwarded_to_pick() -> None:
    seen: list[frozenset[int]] = []

    def pick(skip: frozenset[int]):
        seen.append(skip)
        return None

    slot = _slot(pick=pick)
    slot.start(frozenset({3, 9}))
    slot.take()
    assert seen == [frozenset({3, 9})]


def test_pull_error_becomes_failure_with_tid() -> None:
    def pull(tid: int, remote: str) -> Path:
        raise RuntimeError("rsync exploded")

    slot = _slot(pull=pull)
    slot.start(frozenset())
    item = slot.take()
    assert isinstance(item, PrefetchFailure)
    assert item.tid == 7
    assert "rsync exploded" in item.detail


def test_hydrate_error_becomes_failure_with_tid() -> None:
    def hydrate(tid: int, local: Path):
        raise RuntimeError("ssh exploded")

    slot = _slot(hydrate=hydrate)
    slot.start(frozenset())
    item = slot.take()
    assert isinstance(item, PrefetchFailure)
    assert item.tid == 7


def test_single_slot_enforced() -> None:
    slot = _slot()
    slot.start(frozenset())
    with pytest.raises(AssertionError):
        slot.start(frozenset())
    slot.take()
    slot.start(frozenset())  # legal again after take()
    assert isinstance(slot.take(), PrefetchItem)


def test_take_without_start_raises() -> None:
    with pytest.raises(AssertionError):
        _slot().take()


def test_pending_reflects_slot_state() -> None:
    slot = _slot()
    assert not slot.pending
    slot.start(frozenset())
    assert slot.pending
    slot.take()
    assert not slot.pending

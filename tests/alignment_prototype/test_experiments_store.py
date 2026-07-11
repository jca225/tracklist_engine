from __future__ import annotations
from pathlib import Path
from workspaces.alignment_prototype.experiments.store import Store
from workspaces.alignment_prototype.experiments.matrix import Cell
from workspaces.alignment_prototype.score_timeline_vs_gt import SpanScore


def _row(slot: str, strict: float) -> SpanScore:
    return SpanScore(
        slot, "recX", "acappella", "multiseg", True, 3.0, strict, strict, None, 2
    )


def test_upsert_is_idempotent_and_fetch_filters(tmp_path: Path):
    s = Store(tmp_path / "scores.db")
    c = Cell("classical", "1fsnxchk", decoder="looptrace")
    # Two distinct spans (different slot), each with their own strict value.
    s.upsert(c, [_row("6", 0.4), _row("7", 0.5)])
    s.upsert(c, [_row("6", 0.4), _row("7", 0.5)])  # re-run: no duplication
    got = s.fetch(set_id="1fsnxchk")
    assert len(got) == 2
    assert got[0]["driver"] == "classical" and got[0]["decoder"] == "looptrace"
    assert {r["strict"] for r in got} == {0.4, 0.5}
    assert s.fetch(driver="agentic") == []


def test_rescore_same_cell_replaces_not_appends(tmp_path: Path):
    """Re-scoring the same (cell, slot, recording_id) with different strict/fiber
    must REPLACE the existing row, not insert a second one (idempotency invariant)."""
    s = Store(tmp_path / "scores.db")
    c = Cell("classical", "1fsnxchk", decoder="looptrace")

    # First pass: strict=0.4, fiber=0.4
    first_row = SpanScore(
        "6", "recX", "acappella", "multiseg", True, 3.0, 0.4, 0.4, None, 2
    )
    s.upsert(c, [first_row])

    # Second pass: same (cell, slot="6", recording_id="recX") but different strict/fiber
    second_row = SpanScore(
        "6", "recX", "acappella", "multiseg", True, 3.0, 0.9, 0.9, None, 2
    )
    s.upsert(c, [second_row])

    rows = s.fetch(set_id="1fsnxchk")

    # (a) Exactly ONE row for that span — second upsert replaced the first
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"

    # (b) The surviving row carries the second (latest) strict value
    assert rows[0]["strict"] == 0.9, f"Expected strict=0.9, got {rows[0]['strict']}"

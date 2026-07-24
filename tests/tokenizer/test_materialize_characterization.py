import pickle
from pathlib import Path
from tokenizer.materialize import materialize
from tests.tokenizer.conftest import snapshot

_GOLDEN = Path("tests/tokenizer/golden_materialize.pkl")


def test_materialize_output_is_stable(fixture_db: Path):
    counts = materialize(fixture_db, batch_size=10_000)
    # NOTE: materialize()'s actual return dict keys are
    # {"track_metadata", "set_track_slots", "track_suggestions", "errors"} —
    # the brief's `counts["slot"]` does not match the real interface.
    assert counts["set_track_slots"] > 0, f"fixture produced no slots: {counts}"
    # Guard: the fixture also exercises the suggestion (sugTog) path, so a
    # regression there can't silently pass against an empty golden table.
    assert counts["track_suggestions"] > 0, f"fixture produced no suggestions: {counts}"
    snap = snapshot(fixture_db)
    if not _GOLDEN.exists():  # first run: record
        _GOLDEN.write_bytes(pickle.dumps(snap))
    golden = pickle.loads(_GOLDEN.read_bytes())
    assert snap == golden, "materialize output changed vs golden snapshot"

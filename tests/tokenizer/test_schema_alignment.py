import sqlite3
from pathlib import Path
from tokenizer import model
from tokenizer.materialize import _MATERIALIZE_DDL

_SCHEMA = Path("web_crawler/database/schema.sql")


def _table_columns(table: str) -> set[str]:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA.read_text())
    cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    conn.close()
    return cols


def test_slot_row_is_subset_of_schema():
    # Every dataclass field must be a real column. (track_id/parsed_at etc. may
    # exist in schema without a struct field; the struct must not invent columns.)
    schema_cols = _table_columns("set_track_slots")
    missing = set(model.columns(model.SetTrackSlotRow)) - schema_cols
    assert not missing, f"struct fields not in set_track_slots schema: {missing}"


def test_suggestion_row_is_subset_of_schema():
    schema_cols = _table_columns("track_suggestions")
    missing = set(model.columns(model.TrackSuggestionRow)) - schema_cols
    assert not missing, f"struct fields not in track_suggestions schema: {missing}"


def _columns_from_ddl(ddl_text: str, table: str) -> set[str]:
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl_text)
    cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    conn.close()
    return cols


def test_inline_materialize_ddl_matches_schema_for_written_tables():
    # materialize.py carries its own inline DDL (_MATERIALIZE_DDL) that creates the
    # output tables on a fresh DB. It MUST stay column-identical to the canonical
    # schema.sql for the tables it writes, or a fresh-DB run silently drops columns
    # (the exact bug fixed on this branch: layer_role/constituents_json).
    # NOTE: This compares column NAMES only; it does not verify column types/defaults,
    # nor that dataclass structs cover every writable column (a future gotcha).
    for table in ("set_track_slots", "track_suggestions", "track_metadata"):
        inline = _columns_from_ddl(_MATERIALIZE_DDL, table)
        canonical = _table_columns(table)
        assert inline == canonical, (
            f"{table}: _MATERIALIZE_DDL columns {inline} != schema.sql {canonical}"
        )

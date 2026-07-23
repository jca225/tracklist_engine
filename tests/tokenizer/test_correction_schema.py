import sqlite3

from tokenizer.materialize import _MATERIALIZE_DDL


def _cols(conn, t):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}


def test_correction_layer_tables_and_columns():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"set_notices", "set_slot_id_meta", "track_id_links"} <= tables
    sug = _cols(conn, "track_suggestions")
    assert {
        "data_type",
        "cue_seconds",
        "poll_correct",
        "poll_not_correct",
        "poll_unsure",
        "is_id_remix",
        "has_apple",
        "labels_json",
    } <= sug

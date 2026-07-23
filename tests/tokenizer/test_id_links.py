import sqlite3

from tokenizer.materialize import _MATERIALIZE_DDL, _flush_id_meta, _id_meta_tuple


class _FakeID:
    protected = True
    rbcst = False
    watchers = 42
    spotify_presave_count = None


def test_id_meta_persists():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    _flush_id_meta(conn, [_id_meta_tuple(_FakeID(), "SET1", 3, tlp_id=555, is_id=True)])
    row = conn.execute(
        "SELECT is_id, protected, rbcst, watchers FROM set_slot_id_meta"
    ).fetchone()
    assert row == (1, 1, 0, 42)

import sqlite3

from tokenizer.materialize import _MATERIALIZE_DDL, _flush_notices, _notice_tuple
from tokenizer.text_tokenizer import parse_bItmH_row

RECYCLE = (
    '<div class="bItmH"><i class="fa fa-recycle fa-24"></i> '
    "This tracklist contains identical tracklist(s) (parts)</div>"
)


def test_notice_row_type_persists():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    tok = parse_bItmH_row(RECYCLE)
    _flush_notices(conn, [_notice_tuple(tok, "SET1", 12)])
    row = conn.execute("SELECT set_id, row_index, row_type FROM set_notices").fetchone()
    assert row == ("SET1", 12, "recycle_notice")

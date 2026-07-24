"""End-to-end: materialize() dispatches every row type into the WS-C tables."""

import sqlite3

from tokenizer.materialize import materialize

TRACK = (
    '<div class="bItm tlpItem tlpTog" data-id="100" data-trackid="abc123" '
    'data-trno="0"><div class="bPlay">'
    '<span id="tlp0_tracknumber_value">01</span></div>'
    '<div class="bCont"><div class="fontL" '
    'itemtype="http://schema.org/MusicRecording">'
    '<meta itemprop="name" content="Artist - Title"/>'
    '<span class="trackValue">Artist - Title</span></div></div></div>'
)
ID_ROW = (
    '<div class="bItm tlpItem tlpTog" data-id="200" data-isided="true" '
    'data-protected="true" data-trno="1"><div class="bPlay">'
    '<span id="tlp0_tracknumber_value">02</span></div>'
    '<div class="bCont"><div class="fontL" '
    'itemtype="http://schema.org/MusicRecording">'
    '<meta itemprop="name" content="ID - ID"/>'
    '<span class="trackValue">ID - ID</span></div></div></div>'
)
SUG = (
    '<div class="bItm con ntB sugTog" data-type="5" data-tlp="100" data-pos="1" '
    'data-guest="9">track wasn\'t played [24-11-09 10:01:29] MeCaddy</div>'
)
NOTICE = (
    '<div class="bItmH"><i class="fa fa-recycle fa-24"></i> '
    "This tracklist contains identical tracklist(s) (parts)</div>"
)


def test_materialize_dispatches_all_row_types(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE dj_set_rows (row_id INTEGER PRIMARY KEY, set_id TEXT, "
        "row_index INTEGER, raw_html TEXT);"
    )
    for i, html in enumerate([TRACK, ID_ROW, SUG, NOTICE]):
        conn.execute(
            "INSERT INTO dj_set_rows (row_id, set_id, row_index, raw_html) "
            "VALUES (?,?,?,?)",
            (i + 1, "SETX", i, html),
        )
    conn.commit()
    conn.close()

    materialize(db)

    c = sqlite3.connect(db)
    (slots,) = c.execute("SELECT COUNT(*) FROM set_track_slots").fetchone()
    assert slots == 2  # track + ID row both become slots

    dt = c.execute(
        "SELECT data_type FROM track_suggestions WHERE data_type IS NOT NULL"
    ).fetchone()
    assert dt == (5,)  # "track wasn't played" persisted with its type

    nt = c.execute("SELECT row_type FROM set_notices").fetchone()
    assert nt == ("recycle_notice",)

    idm = c.execute(
        "SELECT is_id, protected FROM set_slot_id_meta WHERE is_id=1"
    ).fetchone()
    assert idm == (1, 1)  # ID row captured with protected flag

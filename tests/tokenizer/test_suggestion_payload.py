import sqlite3

from bs4 import BeautifulSoup

from tokenizer.materialize import (
    _MATERIALIZE_DDL,
    _flush_suggestions,
    _suggestion_tuple,
)
from tokenizer.suggestion_tokenizer import parse_suggestion_row

# A type-5 "track wasn't played" suggestion row with a community poll.
# Poll is real markup: div#pollres_* with greenTxt/redTxt/blueTxt counts.
ROW = (
    '<div class="bItm con ntB sugTog" data-type="5" data-tlp="123" data-pos="7" '
    'data-guest="999">'
    "track wasn't played [24-11-09 10:01:29] MeCaddy "
    '<div id="pollres_123">[poll:'
    '<div class="greenTxt iB">3</div>/'
    '<div class="redTxt iB">1</div>/'
    '<div class="blueTxt iB">0</div>]</div>'
    "</div>"
)


def _buf_from(html, set_id="SET1"):
    sug = parse_suggestion_row(BeautifulSoup(html, "lxml").find("div"))
    return _suggestion_tuple(sug, set_id)


def test_suggestion_type_and_poll_persist():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    _flush_suggestions(conn, [_buf_from(ROW)])
    row = conn.execute(
        "SELECT data_type, poll_correct, poll_not_correct, poll_unsure "
        "FROM track_suggestions"
    ).fetchone()
    assert row == (5, 3, 1, 0)


def test_legacy_db_still_writes_identity(tmp_path):
    # A pre-migration track_suggestions (14 cols) must still accept writes.
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE track_suggestions ("
        "sug_id INTEGER PRIMARY KEY, set_id TEXT, tlp_id INTEGER, pos INTEGER, "
        "track_slug TEXT, track_display TEXT, artist_title TEXT, "
        "suggester_user_id INTEGER, suggester_name TEXT, suggestion_timestamp TEXT, "
        "is_remix INTEGER, has_youtube INTEGER, has_soundcloud INTEGER, has_spotify INTEGER)"
    )
    _flush_suggestions(conn, [_buf_from(ROW)])
    (n,) = conn.execute("SELECT COUNT(*) FROM track_suggestions").fetchone()
    assert n == 1

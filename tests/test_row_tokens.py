"""Tests for the HTML row tokenizer dispatcher."""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer import classify_row, tokenize_row


FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIX / name).read_text()


# ---------- classify_row -----------------------------------------------------

def test_classify_track_row():
    assert classify_row(_load("row_track.html")) == "track"


def test_classify_concurrent_track_row():
    assert classify_row(_load("row_track_concurrent.html")) == "track"


def test_classify_player_widget():
    assert classify_row(_load("row_player_widget.html")) == "player_widget"


def test_classify_save_footer():
    assert classify_row(_load("row_save_footer.html")) == "save_footer"


def test_classify_suggestion():
    assert classify_row(_load("row_suggestion.html")) == "suggestion"


def test_classify_text_row():
    # bItmH rows wrap notices / headers / recycle links
    assert classify_row(_load("row_text.html")) == "text"


def test_classify_unknown_returns_label():
    assert classify_row("<div class='completely-unrelated'>x</div>") == "unknown"


# ---------- tokenize_row -----------------------------------------------------

def test_tokenize_track_row_extracts_core_fields():
    tok = tokenize_row(_load("row_track.html"))
    assert tok is not None
    assert type(tok).__name__ == "TrackRow"
    # Every track row has at least these, regardless of IDing status:
    assert tok.data_trno == 0          # the first real track row in Vol. 26 is trno=0
    assert tok.is_ided is True
    assert tok.title                   # non-empty string


def test_tokenize_extracts_artwork_url():
    """The UI needs the real artwork URL — `src` on img.artwork is a lazy-load
    placeholder, the actual CDN URL lives in `data-src`."""
    tok = tokenize_row(_load("row_track.html"))
    assert tok.artwork_url is not None
    assert tok.artwork_url.startswith("http")
    assert "empty.png" not in tok.artwork_url  # rejected placeholder


def test_tokenize_rejects_placeholder_artwork():
    """Rows whose only <img class="artwork"> points at the empty.png placeholder
    should produce artwork_url=None (not the placeholder)."""
    html = '<div class="tlpTog bItm tlpItem"><img class="artwork artM" src="/images/static/empty.png"/></div>'
    tok = tokenize_row(html)
    assert tok.artwork_url is None


def test_tokenize_concurrent_track_sets_flag():
    tok = tokenize_row(_load("row_track_concurrent.html"))
    assert tok is not None
    assert tok.is_concurrent is True   # the second row (trno=1) is layered "w/"


def test_tokenize_player_widget_returns_none():
    # Page chrome rows dispatch to no parser.
    assert tokenize_row(_load("row_player_widget.html")) is None


def test_tokenize_save_footer_returns_none():
    assert tokenize_row(_load("row_save_footer.html")) is None


def test_tokenize_suggestion_returns_object():
    tok = tokenize_row(_load("row_suggestion.html"))
    assert tok is not None
    assert type(tok).__name__.endswith("Row") or type(tok).__name__.endswith("Token")


@pytest.mark.parametrize("name", [
    "row_track.html",
    "row_track_concurrent.html",
    "row_suggestion.html",
    "row_text.html",
])
def test_tokenize_never_raises_on_valid_fixtures(name: str):
    # Domain contract: the tokenizer should never raise on a well-formed row;
    # unparseable cases return None or an empty-but-typed record.
    tokenize_row(_load(name))   # assertion is the absence of an exception

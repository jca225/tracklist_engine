from __future__ import annotations

from core.labels import labels_overlap


def test_disjoint_titles_do_not_overlap():
    # the 20911 mis-attach: acquired song vs target recording title
    assert labels_overlap("Come On Over Baby (All I Want Is You)", "Good Time") is False


def test_same_song_titles_overlap():
    assert (
        labels_overlap("Nelly Furtado - Say It Right", "Say It Right (Studio acapella)")
        is True
    )


def test_empty_side_never_overlaps():
    assert labels_overlap("", "Good Time") is False

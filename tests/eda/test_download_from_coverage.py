"""Unit tests for the coverage → ingest job-file executor (pure surface).

Covers the pieces that carry logic without touching pi-storage:
  - candidate → job-row mapping (with + without fetched metadata)
  - SoundCloud-over-YouTube URL preference (via SetMeta construction)
  - top-N slicing / rank order preservation
The DB-touching edges (fetch_rows, fetch_meta) are NOT exercised here.
"""

from __future__ import annotations

from eda.alignment.generalization.download_from_coverage import (
    JOB_FIELDS,
    SetMeta,
    build_job_rows,
    candidate_to_job_row,
    rank_candidates,
)
from eda.alignment.generalization.grammar_coverage import (
    SetFingerprint,
    build_coverage,
)


def _fp(
    sid,
    *,
    w=0.0,
    ver=0.0,
    stem=0.0,
    tracks=20,
    pt=60.0,
    audio=False,
    fetch=True,
    self_frac=0.0,
    styles="",
):
    return SetFingerprint(
        set_id=sid,
        title=f"title-{sid}",
        n_slots=tracks,
        total_tracks=tracks,
        play_time_min=pt,
        w_frac=w,
        version_frac=ver,
        stem_frac=stem,
        id_frac=0.0,
        mean_cue_gap=None,
        styles=styles,
        has_audio=audio,
        fetchable=fetch and not audio,
        self_frac=self_frac,
        is_live_pa=self_frac >= 0.7,
    )


def _meta(sid, *, sc="", yt="", title="", views=100, styles="House"):
    if sc:
        media_url, plat = sc, "soundcloud"
    elif yt:
        media_url, plat = yt, "youtube"
    else:
        media_url, plat = "", "none"
    return SetMeta(
        set_id=sid,
        title=title or f"meta-{sid}",
        set_url=f"https://1001tracklists.com/{sid}",
        date="2025-01-01",
        creator_name="dj_x",
        creator_url="https://1001tracklists.com/user/dj_x",
        views=views,
        ided_tracks=18,
        total_tracks=20,
        play_time="1h 20m",
        likes=5,
        styles=styles,
        media_url=media_url,
        link_platform=plat,
    )


# --- SetMeta URL preference (SC over YT) ------------------------------------


def test_setmeta_prefers_soundcloud_over_youtube():
    m = _meta("a", sc="https://sc/x", yt="https://yt/x")
    assert m.link_platform == "soundcloud"
    assert m.media_url == "https://sc/x"


def test_setmeta_falls_back_to_youtube():
    m = _meta("a", sc="", yt="https://yt/x")
    assert m.link_platform == "youtube"
    assert m.media_url == "https://yt/x"


def test_setmeta_no_link():
    m = _meta("a")
    assert m.link_platform == "none"
    assert m.media_url == ""


# --- candidate → job row ----------------------------------------------------


def test_job_row_schema_matches_job_file_fields():
    row = candidate_to_job_row(_fp("a"), _meta("a", sc="https://sc/a"))
    assert tuple(row.keys()) == JOB_FIELDS
    assert row["tracklist_id"] == "a"


def test_job_row_url_is_preferred_media_url_not_page():
    # url must be the MEDIA url (mix audio), SC preferred — not the 1001TL page.
    row = candidate_to_job_row(
        _fp("a"), _meta("a", sc="https://sc/a", yt="https://yt/a")
    )
    assert row["url"] == "https://sc/a"
    row_yt = candidate_to_job_row(_fp("b"), _meta("b", yt="https://yt/b"))
    assert row_yt["url"] == "https://yt/b"


def test_job_row_carries_metadata():
    row = candidate_to_job_row(
        _fp("a"), _meta("a", sc="https://sc/a", views=999, styles="Techno")
    )
    assert row["views"] == 999
    assert row["styles"] == "Techno"
    assert row["ided_tracks"] == 18
    assert row["total_tracks"] == 20
    assert row["creator_name"] == "dj_x"


def test_job_row_missing_metadata_falls_back_to_fingerprint():
    fp = _fp("ghost", tracks=33, styles="Trance")
    row = candidate_to_job_row(fp, None)
    assert tuple(row.keys()) == JOB_FIELDS
    assert row["tracklist_id"] == "ghost"
    assert row["title"] == "title-ghost"  # fp title fallback
    assert row["url"] == ""  # no media url known
    assert row["total_tracks"] == 33  # fp total_tracks fallback
    assert row["styles"] == "Trance"
    assert row["views"] is None


def test_job_row_prefers_meta_title_over_fingerprint():
    row = candidate_to_job_row(
        _fp("a"), _meta("a", sc="https://sc/a", title="Real Title")
    )
    assert row["title"] == "Real Title"


# --- ranking + top-N slicing ------------------------------------------------


def test_rank_candidates_excludes_downloaded_and_live_pa():
    rows = [
        _fp("dl", stem=0.5, audio=True),  # already downloaded -> not fetchable
        _fp("live", stem=0.5, self_frac=0.9),  # live PA -> excluded
        _fp("good", stem=0.5, self_frac=0.1),  # keep
        _fp("nolink", stem=0.5, fetch=False),  # not fetchable -> drop
    ]
    cov = build_coverage(rows)
    ranked = rank_candidates(rows, cov, live_pa_thresh=0.7)
    ids = {r.set_id for r in ranked}
    assert ids == {"good"}


def test_build_job_rows_respects_top_n_and_order():
    # three fetchable candidates; ranking should be by fill weight desc.
    rows = [
        _fp("hi", w=0.0, ver=0.0, stem=0.0),  # starved corners -> high fill
        _fp("mid", w=0.5, ver=0.5, stem=0.5),
        _fp("lo", w=0.9, ver=0.9, stem=0.9),
    ]
    cov = build_coverage(rows)
    ranked = rank_candidates(rows, cov, live_pa_thresh=0.7)
    meta = {r.set_id: _meta(r.set_id, sc=f"https://sc/{r.set_id}") for r in rows}

    top2 = build_job_rows(ranked, meta, top=2)
    assert len(top2) == 2
    # order preserved from ranked
    assert [r["tracklist_id"] for r in top2] == [
        ranked[0].set_id,
        ranked[1].set_id,
    ]

    top_all = build_job_rows(ranked, meta, top=99)
    assert len(top_all) == 3


def test_build_job_rows_empty():
    assert build_job_rows([], {}, top=500) == []

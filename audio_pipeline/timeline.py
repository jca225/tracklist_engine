"""Timeline construction: tokenize rows → ordered list of TimelineSegment.

Pure functions (no I/O, no Result — the one library boundary is reading rows
from the DB, which happens in the caller).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable

import pandas as pd

from .models import SetTimeline, TimelineSegment


_SEGMENT_FIELDS = (
    "row_index", "track_id", "tlp_id", "title", "artists",
    "cue_seconds_section", "is_ided", "is_concurrent", "is_remixish",
    "has_yt", "has_sc", "has_sp",
)


def build_timeline(set_id: str, tokens_for_set: pd.DataFrame, set_audio_id: int | None = None) -> SetTimeline:
    """Given the tokenized rows for ONE set (already `cue_seconds_section`-resolved
    by `big_bootie._resolve_cue_sections`), return an ordered SetTimeline."""
    tracks = tokens_for_set[tokens_for_set["row_kind"] == "track"].copy()
    tracks = tracks.sort_values("row_index").reset_index(drop=True)

    segments: list[TimelineSegment] = []
    for _, r in tracks.iterrows():
        artists_raw = r.get("artists")
        if isinstance(artists_raw, str) and artists_raw:
            artists = tuple(artists_raw.split("|"))
        else:
            artists = ()
        cue = r.get("cue_seconds_section")
        segments.append(TimelineSegment(
            row_index=int(r["row_index"]),
            track_id=(r.get("track_key") or None),
            tlp_id=(r.get("row_dom_id") or None),
            title=(r.get("title") or None),
            artists=artists,
            cue_seconds_section=(float(cue) if pd.notna(cue) else None),
            is_ided=bool(r.get("is_ided") or False),
            is_concurrent=bool(r.get("is_concurrent") or False),
            is_remixish=bool(r.get("is_remixish") or False),
            has_yt=bool(r.get("has_yt") or False),
            has_sc=bool(r.get("has_sc") or False),
            has_sp=bool(r.get("has_sp") or False),
        ))
    return SetTimeline(set_id=set_id, set_audio_id=set_audio_id, segments=tuple(segments))


def timeline_to_json(tl: SetTimeline) -> str:
    """Serialize a SetTimeline to a stable JSON string (ordered keys)."""
    payload = {
        "set_id": tl.set_id,
        "set_audio_id": tl.set_audio_id,
        "segments": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(s).items()}
            for s in tl.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---------- pipeline-aware helpers -------------------------------------------
# These are about using tokenizer output to drive download/reference-selection
# decisions. They're pure (operate on DataFrames), no I/O.

def reference_track_ids(tokens_for_set: pd.DataFrame) -> tuple[str, ...]:
    """The subset of track_ids worth computing MERT-full on: IDed AND not remixish.

    `is_remixish` means the row is a remix/rework/acappella/alt variant — the
    audio is not the canonical commercial release, so we'd rather fetch the
    reference from the normal tracklist and let the transposition adapter
    handle any key/tempo shift."""
    t = tokens_for_set
    mask = (t["row_kind"] == "track") & t["is_ided"].fillna(False).astype(bool) & ~t["is_remixish"].fillna(False).astype(bool)
    ids = t.loc[mask, "track_key"].dropna().unique().tolist()
    return tuple(str(i) for i in ids if i)


def concurrent_groups(tokens_for_set: pd.DataFrame) -> tuple[tuple[int, ...], ...]:
    """Group track rows that share a cue_seconds_section into tuples of row_index.

    A group of size 1 is a standalone track; size > 1 is a mashup layer. The
    cutup-plan code uses these groupings as the unit of alignment.
    """
    tracks = tokens_for_set[tokens_for_set["row_kind"] == "track"].copy()
    tracks = tracks.sort_values("row_index")
    out: list[tuple[int, ...]] = []
    for _, g in tracks.groupby("cue_seconds_section", dropna=False):
        out.append(tuple(int(i) for i in g["row_index"]))
    return tuple(out)

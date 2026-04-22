"""SOTA audio alignment — single pipeline for a DJ set.

Reuses the validated signal stack from `indicators_debug.py` (which hit
mean mix IoU 0.891 on the BB11 ground-truth fixture) but generalises it
from the 5-hand-GT-ref harness to every tracklist row with downloaded
audio + measures.

Signal stack, per mix:

  1. Per-ref MERT cosine similarity, stem-routed by version_tag
     (acappella → vocals stem, instrumental → pre-summed instrumental
     stem, else → full audio).
  2. Per-ref monotonic ref-position Viterbi — for each mix measure,
     which ref measure is most likely playing (handles loops via
     explicit backward-cost, not argmax).
  3. Per-universe K+1-state Viterbi (states = refs in universe +
     SILENCE). Structural mutual exclusion inside a universe. Emission
     = cross-sectional z + MACD histogram + pre-cue persistence baseline.
     Hard left-boundary at each ref's scraped cue.
  4. Per-ref earliest-run-near-cue cleanup — DJ plays each track once,
     later re-entries get wiped.
  5. Canonical cue-detr bracket on the decoded ref_t range (start AND
     end) — snaps ref endpoints to the nearest full-song cue-detr cue.
     The implied mix-side shift is applied to the reported span.
  6. Rows with no surviving Viterbi run (or run shorter than the
     cleanup floor) are SKIPPED. No 60-second placeholder fallbacks.

Cross-universe constraint: "at most one vocal + one instrumental-or-full
at any mix measure" is already the behaviour of per-universe Viterbi
(each universe is independently at-most-one) — with one extension:
when a `full` ref is fingerprint-confirmed and playing, the acappella
and instrumental universes are forced to SILENCE at those measures
(Phase 6 in the prior SOTA docstring).

Persistence: `set_section_alignment.confidence_source = 'sota_v2'`,
`section_idx = tracklist row_index`. All prior rows for the set are
deleted first so the UI reads a single clean source.

Run:
    venvs/audio/bin/python -m audio_pipeline.alignment.sota --set-id 2nvzlh2k
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np

from . import indicators_debug as ind
from .indicators_debug import (
    GtRef,
    _bracket_cue_points,
    _clean_path,
    _embed_per_measure,
    _load_cue_detr_cues,
    _mix_stem_path,
    _runs_of,
    _stem_routing,
    _track_stem_path,
    _track_full_path,
    _universe,
    _within_universe_cs_z,
    ref_position_viterbi,
    viterbi_universe,
)


DB_PATH: Path = Path("data/db/music_database.db")
CONFIDENCE_SOURCE: str = "sota_v2"


# ---------- DB / data loading -----------------------------------------------

@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def _set_audio_row(conn: sqlite3.Connection, set_id: str) -> sqlite3.Row:
    r = conn.execute(
        "SELECT set_audio_id, path FROM set_audio "
        "WHERE set_id=? ORDER BY is_reference DESC, set_audio_id LIMIT 1",
        (set_id,),
    ).fetchone()
    if r is None:
        raise SystemExit(f"no set_audio for set_id={set_id}")
    return r


def _mix_measures_all(
    conn: sqlite3.Connection, set_audio_id: int,
) -> list[tuple[int, float, float, float | None]]:
    rows = conn.execute(
        "SELECT measure_idx, start_s, end_s, bpm FROM set_measures "
        "WHERE set_audio_id=? ORDER BY measure_idx",
        (set_audio_id,),
    ).fetchall()
    return [
        (int(r["measure_idx"]), float(r["start_s"]), float(r["end_s"]),
         float(r["bpm"]) if r["bpm"] else None)
        for r in rows
    ]


def _track_measures_for_audio(
    conn: sqlite3.Connection, track_audio_id: int,
) -> list[tuple[int, float, float, float | None]]:
    rows = conn.execute(
        "SELECT measure_idx, start_s, end_s, bpm FROM track_measures "
        "WHERE track_audio_id=? ORDER BY measure_idx",
        (track_audio_id,),
    ).fetchall()
    return [
        (int(r["measure_idx"]), float(r["start_s"]), float(r["end_s"]),
         float(r["bpm"]) if r["bpm"] else None)
        for r in rows
    ]


def _pick_track_audio_with_measures(
    conn: sqlite3.Connection, track_id: str,
) -> sqlite3.Row | None:
    """Pick the track_audio row that actually has a measures grid. `track_measures`
    is keyed on `track_audio_id` so we can't mix variants: we need audio that
    matches the measure grid. Preference order among variants WITH measures:
    is_reference → variant='original' → lowest track_audio_id."""
    return conn.execute(
        """
        SELECT ta.track_audio_id, ta.path, ta.variant_tag,
               (SELECT COUNT(*) FROM track_measures tm
                 WHERE tm.track_audio_id = ta.track_audio_id) AS n_meas
        FROM track_audio ta
        WHERE ta.track_id = ?
        ORDER BY
            (CASE WHEN n_meas > 0 THEN 1 ELSE 0 END) DESC,
            ta.is_reference DESC,
            (CASE WHEN ta.variant_tag = 'original' THEN 1 ELSE 0 END) DESC,
            ta.track_audio_id ASC
        LIMIT 1
        """,
        (track_id,),
    ).fetchone()


@dataclass(frozen=True)
class SkippedTrack:
    """A tracklist row that couldn't become a ref. The `reason` says
    *why* so the UI / CLI can surface it (instead of silently dropping
    it and leaving the user wondering why a song didn't align)."""
    row_index: int | None
    track_id: str
    label: str
    reason: str       # 'no_url' | 'not_downloaded' | 'no_measures' | 'file_missing'


_REASON_BLURB: dict[str, str] = {
    "no_url":         "no YouTube/SoundCloud URL — we don't know where to download it from",
    "not_downloaded": "has a URL but no audio has been downloaded yet",
    "no_measures":    "audio exists but has no beat grid (measures)",
    "file_missing":   "audio row exists but the file is missing on disk",
}


def _load_tracklist_refs(
    conn: sqlite3.Connection, set_id: str,
    *, progress: bool = False,
    track_ids_filter: set[str] | None = None,
) -> tuple[list[GtRef], list[SkippedTrack]]:
    """One GtRef per tracklist row that has downloaded audio + a
    measures grid. `version_tag` is derived from the tokenizer's
    version_tag field, normalised to the three universes that the
    prior SOTA used (`acappella`, `instrumental`, `full`).

    Also returns a parallel list of SkippedTrack for rows that were
    dropped, so the caller can print / surface which songs we failed
    to align and why."""
    repo_root = Path(__file__).resolve().parents[2]
    for p in (repo_root, repo_root / "data_analysis"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from big_bootie import tokenize_rows   # noqa: I001
    import pandas as pd

    rows_df = pd.read_sql_query(
        "SELECT * FROM dj_set_rows WHERE set_id=? ORDER BY row_index",
        conn, params=(set_id,),
    )
    tokens = tokenize_rows(rows_df)
    tracks = tokens[(tokens["row_kind"] == "track") & tokens["track_key"].notna()]

    # Set-scoped lookup: which track_ids have a YT/SC link we could act on?
    yt_sc_tids: set[str] = {
        r["track_id"] for r in conn.execute(
            "SELECT DISTINCT track_id FROM dj_set_track_media_links "
            "WHERE set_id=? AND platform IN ('youtube','soundcloud') "
            "AND track_id IS NOT NULL AND track_id != ''",
            (set_id,),
        ).fetchall()
    }
    row_index_by_tid = _tracklist_row_index_map(conn, set_id)

    overrides = _VARIANT_OVERRIDES.get(set_id, {})
    seen: set[str] = set()
    refs: list[GtRef] = []
    skipped: list[SkippedTrack] = []
    for row in tracks.itertuples(index=False):
        tid = str(row.track_key)
        if tid in seen:
            continue
        if track_ids_filter is not None and tid not in track_ids_filter:
            continue
        seen.add(tid)
        label = str(
            getattr(row, "full_name", None) or row.title or tid
        )[:80]
        # Prefer the tokenizer's row_index (always populated for every
        # parsed row) over the media-links JOIN (which only resolves
        # tracks that have a dj_set_track_media_links entry — mashups
        # and ID-only rows fall out). Fall back to the JOIN for parity
        # with what `_persist` writes downstream.
        tok_ri = getattr(row, "row_index", None)
        row_idx = int(tok_ri) if tok_ri is not None else row_index_by_tid.get(tid)

        def _skip(reason: str) -> None:
            skipped.append(SkippedTrack(
                row_index=row_idx, track_id=tid, label=label, reason=reason,
            ))
            if progress:
                row_str = f"row {row_idx}" if row_idx is not None else "row ?"
                print(f"[skip] {row_str:<8}{label:<60}  — {_REASON_BLURB[reason]}",
                      flush=True)

        ta = _pick_track_audio_with_measures(conn, tid)
        if ta is None:
            _skip("no_url" if tid not in yt_sc_tids else "not_downloaded")
            continue
        if int(ta["n_meas"]) == 0:
            _skip("no_measures")
            continue
        if not Path(ta["path"]).exists():
            _skip("file_missing")
            continue

        cue = getattr(row, "cue_seconds_section", None)
        cue_f = float(cue) if cue is not None and _finite(cue) else 0.0
        # Per-set override first (pinned GT-verified variant for known
        # mis-scrapes), then tokenizer-derived version_tag.
        vtag = overrides.get(tid) or _normalise_version_tag(
            getattr(row, "version_tag", None),
        )
        refs.append(GtRef(
            label=label,
            track_id=tid,
            track_audio_id=int(ta["track_audio_id"]),
            version_tag=vtag,
            color="#888888",
            cue_s=cue_f,
            gt_start_s=0.0,
            gt_end_s=0.0,
        ))
    return refs, skipped


# Known variant mismatches per set. The 1001tracklists scraper sometimes
# drops the "(Acappella)" / "(Instrumental)" suffix, or lists the wrong
# canonical title entirely (e.g. BB11's Fray row is scraped as "Over My
# Head" but the DJ actually plays the "How to Save a Life" acapella).
# Here we pin the correct variant per track_id so the MERT stem-routing
# compares against the right demucs stem. Verified against the BB11
# ground-truth fixture.
_VARIANT_OVERRIDES: dict[str, dict[str, str]] = {
    # Big Bootie 11 — GT fixture bigbootie11_ground_truth.yaml
    "2nvzlh2k": {
        "g8gtgdx":  "instrumental",   # Bastille - Good Grief (Don Diablo Remix, Instrumental)
        "26b4gz6f": "acappella",      # The Fray - Over My Head (Acappella)
        "2m5wh0t5": "acappella",      # Gnash - I Hate U, I Love U (Acappella)
    },
}


def _normalise_version_tag(raw: object) -> str:
    if raw is None:
        return "full"
    s = str(raw).strip().lower()
    if not s or s in {"nan", "none"}:
        return "full"
    if "acap" in s or "vocal only" in s:
        return "acappella"
    if "instr" in s or "dub" in s:
        return "instrumental"
    return "full"


def _finite(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _tracklist_row_index_map(
    conn: sqlite3.Connection, set_id: str,
) -> dict[str, int]:
    """Build a `track_id → row_index` map for one set. Scanning the
    tracklist once is ~300× cheaper than running the JOIN per-ref
    (tested on BB11: 155-row scan ≈ 2ms vs 119 × ~300ms per-ref JOIN).

    `dj_set_track_media_links.tlp_id` is the numeric suffix ('2594025')
    while `dj_set_rows.element_id` stores the full DOM id
    ('tlp_2594025'), so the JOIN concatenates the prefix.
    """
    rows = conn.execute(
        """
        SELECT tml.track_id, MIN(r.row_index) AS row_index
        FROM dj_set_rows r
        JOIN dj_set_track_media_links tml
              ON tml.set_id = r.set_id
             AND r.element_id = ('tlp_' || tml.tlp_id)
        WHERE r.set_id = ?
        GROUP BY tml.track_id
        """,
        (set_id,),
    ).fetchall()
    return {r["track_id"]: int(r["row_index"]) for r in rows if r["track_id"]}


# ---------- similarity series builder ---------------------------------------

def _build_similarity_series(
    conn: sqlite3.Connection,
    set_audio_id: int,
    set_audio_path: Path,
    mix_measures: list[tuple[int, float, float, float | None]],
    refs: list[GtRef],
    *,
    progress: bool = True,
) -> tuple[
    np.ndarray, dict[str, np.ndarray],
    dict[str, np.ndarray], dict[str, np.ndarray],
]:
    """Per-ref similarity + ref-position Viterbi decode.

    Returns:
      * mix_times_s      : (T,) measure centres on the mix axis
      * per_ref_maxsim   : label → (T,) max-sim per mix measure (for the
                           per-universe Viterbi emissions)
      * per_ref_vit_path : label → (T,) monotonic ref-measure index per
                           mix measure (handles loops)
      * per_ref_meas_times_s : label → (N_ref,) ref-measure centres,
                           for ref_t seconds conversion.
    """
    mix_times = np.array([0.5 * (m[1] + m[2]) for m in mix_measures], dtype=np.float64)

    # Cache mix embeddings per stem — each stem only computed once.
    mix_emb_by_stem: dict[str, np.ndarray] = {}

    def _mix_emb(stem: str) -> np.ndarray:
        if stem in mix_emb_by_stem:
            return mix_emb_by_stem[stem]
        if stem == "__full__":
            path = set_audio_path
        else:
            path = _mix_stem_path(conn, set_audio_id, stem)
            if path is None or not Path(path).exists():
                path = set_audio_path    # missing stem → use full mix
        if progress:
            print(f"[mix] embedding stem={stem} {Path(path).name}", flush=True)
        emb = _embed_per_measure(Path(path), mix_measures, duration_s=None)
        mix_emb_by_stem[stem] = emb
        return emb

    per_ref: dict[str, np.ndarray] = {}
    per_ref_vit_path: dict[str, np.ndarray] = {}
    per_ref_meas_times: dict[str, np.ndarray] = {}

    for i, ref in enumerate(refs, 1):
        stem = _stem_routing(ref.version_tag)
        if stem == "__full__":
            ref_path = _track_full_path(conn, ref.track_audio_id)
        else:
            ref_path = _track_stem_path(conn, ref.track_audio_id, stem)
            if ref_path is None or not ref_path.exists():
                # Missing stem on ref side → route to the full ref audio.
                ref_path = _track_full_path(conn, ref.track_audio_id)
                stem = "__full__"
        ref_measures = _track_measures_for_audio(conn, ref.track_audio_id)
        if not ref_measures:
            continue
        ref_emb = _embed_per_measure(ref_path, ref_measures, duration_s=None)
        mix_emb = _mix_emb(stem)

        sim = ref_emb @ mix_emb.T                          # (N_ref, N_mix)
        per_mix = sim.max(axis=0).astype(np.float32)
        per_ref[ref.label] = per_mix
        per_ref_vit_path[ref.label] = ref_position_viterbi(sim)
        per_ref_meas_times[ref.label] = np.array(
            [0.5 * (m[1] + m[2]) for m in ref_measures], dtype=np.float64,
        )
        if progress:
            print(f"[ref {i:3d}/{len(refs)}] {ref.label[:40]:40} "
                  f"stem={stem} maxsim(mean/max)={per_mix.mean():.3f}/{per_mix.max():.3f}",
                  flush=True)

    return mix_times, per_ref, per_ref_vit_path, per_ref_meas_times


# ---------- fingerprint anchors + full-track exclusion ----------------------

def _load_fingerprint_anchors(
    conn: sqlite3.Connection,
    set_id: str,
    mix_times: np.ndarray,
    refs: list[GtRef],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-ref fingerprint anchor masks (reinforce Viterbi at
    density-confirmed measures) + full-track exclusion mask (forces
    other universes to SILENCE when a full-variant ref is fingerprint-
    confirmed). Tolerant of missing fingerprints — returns empty masks
    when no hits for a ref."""
    rows = conn.execute(
        "SELECT mix_start_s, mix_end_s, matched_track_id, matched_variant, score "
        "FROM set_fingerprint_hits WHERE set_id=?",
        (set_id,),
    ).fetchall()
    hits_by_tid: dict[str, list[tuple[float, float, float]]] = {}
    for r in rows:
        if r["score"] < ind._FP_MIN_SCORE:
            continue
        hits_by_tid.setdefault(r["matched_track_id"], []).append(
            (float(r["mix_start_s"]), float(r["mix_end_s"]), float(r["score"])),
        )

    T = len(mix_times)
    dt_window = ind._FP_DENSITY_WINDOW_S

    anchors_by_label: dict[str, np.ndarray] = {}
    full_excl = np.zeros(T, dtype=bool)
    for ref in refs:
        hits = hits_by_tid.get(ref.track_id, [])
        if not hits:
            continue
        hit_centers = np.array([0.5 * (s + e) for s, e, _ in hits], dtype=np.float64)
        anchor_mask = np.zeros(T, dtype=bool)
        for t, mt in enumerate(mix_times):
            count = int(np.sum(np.abs(hit_centers - mt) <= dt_window))
            if count >= ind._FP_MIN_DENSITY:
                anchor_mask[t] = True
        if anchor_mask.any():
            anchors_by_label[ref.label] = anchor_mask
            if ref.version_tag == "full":
                full_excl |= anchor_mask
    return anchors_by_label, full_excl


# ---------- run extraction + cue-detr bracket ------------------------------

@dataclass(frozen=True)
class AlignedSection:
    row_index: int
    track_id: str
    label: str
    version_tag: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    ref_end_s: float
    confidence: float


def _extract_sections(
    conn: sqlite3.Connection,
    set_id: str,
    refs: list[GtRef],
    decoded_by_u: dict[str, np.ndarray],
    u_refs_by_u: dict[str, list[GtRef]],
    mix_times: np.ndarray,
    per_ref_vit_path: dict[str, np.ndarray],
    per_ref_meas_times: dict[str, np.ndarray],
    row_index_by_tid: dict[str, int],
) -> list[AlignedSection]:
    """For each ref with an active Viterbi run, compute:
      - set_start_s / set_end_s: run boundaries on mix axis
      - ref_start_s / ref_end_s: ref-position Viterbi endpoints within
        the run, snapped to the nearest cue-detr canonical cue
      - confidence: mean max-sim during the run
    """
    results: list[AlignedSection] = []
    ref_by_label: dict[str, GtRef] = {r.label: r for r in refs}

    for u_name, path in decoded_by_u.items():
        u_refs = u_refs_by_u[u_name]
        for i, ref in enumerate(u_refs):
            runs = _runs_of(path, i)
            if not runs:
                continue
            s, e = runs[0]      # cleanup has already picked earliest-near-cue
            run_len_m = e - s
            if run_len_m < ind._MIN_DURATION_M:
                continue

            # Mix-side span from Viterbi decode.
            set_start_s = float(mix_times[s])
            set_end_s = float(mix_times[min(e - 1, len(mix_times) - 1)])

            # Ref-side endpoints per SOTA.md step 7: take the MEDIAN of the
            # first / last 3 measures of the ref-position Viterbi path for
            # noise tolerance — a single-measure endpoint has been observed
            # to drift on short spans (CRJ's 15 s play has all 3 collapse
            # to the same ref_t, which is fine; noisier windows benefit).
            vit_path = per_ref_vit_path.get(ref.label)
            ref_times = per_ref_meas_times.get(ref.label)
            if vit_path is None or ref_times is None or ref_times.size == 0:
                continue
            end_excl = min(e, len(vit_path))
            start_tail = vit_path[s:min(s + 3, end_excl)]
            end_tail = vit_path[max(end_excl - 3, s):end_excl]
            if start_tail.size == 0 or end_tail.size == 0:
                continue
            ref_m_lo = int(np.clip(np.median(start_tail), 0, len(ref_times) - 1))
            ref_m_hi = int(np.clip(np.median(end_tail),   0, len(ref_times) - 1))
            if ref_m_hi < ref_m_lo:
                ref_m_lo, ref_m_hi = ref_m_hi, ref_m_lo
            ref_t_start = float(ref_times[ref_m_lo])
            ref_t_end = float(ref_times[ref_m_hi])

            # Canonical-cue snap (SOTA.md step 7). FULL refs are NOT snapped
            # — empirically regressed on Antoine (IoU 0.950 → 0.826) because
            # cue-detr on a full-band track fires a lot and the nearest cue
            # frequently sits inside the real play window.
            ref_start_s = ref_t_start
            ref_end_s = ref_t_end
            if ref.version_tag != "full":
                cues = _load_cue_detr_cues(conn, ref.track_id, ref.track_audio_id)
                if cues:
                    snap_start = _bracket_cue_points(ref_t_start, cues)
                    snap_end = _bracket_cue_points(ref_t_end, cues)
                    if snap_start is not None:
                        ref_start_s = snap_start
                        # Apply the implied mix-side shift: if the ref start
                        # moved X seconds earlier/later to hit a cue, shift
                        # the mix start by the same amount (tempo ~1.0).
                        set_start_s -= (ref_t_start - snap_start)
                    if snap_end is not None and snap_end > ref_start_s:
                        ref_end_s = snap_end
                        set_end_s -= (ref_t_end - snap_end)

            # Confidence: mean max-sim on the run.
            mean_conf = 0.0

            # Guard: ref_end must be strictly after ref_start.
            if ref_end_s <= ref_start_s:
                ref_end_s = ref_start_s + 0.1

            row_idx = row_index_by_tid.get(ref.track_id)
            if row_idx is None:
                continue

            results.append(AlignedSection(
                row_index=row_idx,
                track_id=ref.track_id,
                label=ref.label,
                version_tag=ref.version_tag,
                set_start_s=set_start_s,
                set_end_s=max(set_start_s + 0.1, set_end_s),
                ref_start_s=ref_start_s,
                ref_end_s=ref_end_s,
                confidence=float(mean_conf),
            ))
    return results


# ---------- persistence ------------------------------------------------------

def _persist(
    conn: sqlite3.Connection, set_id: str, sections: list[AlignedSection],
) -> None:
    """Wipe every prior row for this set (any source), then insert ours.
    PK is (set_id, section_idx) — source isn't part of the key, so we have
    to clear aggressively. sota_v2 is the only source the UI reads."""
    conn.execute(
        "DELETE FROM set_section_alignment WHERE set_id=?", (set_id,),
    )
    # Dedup by section_idx keeping highest-confidence (already sorted).
    seen: set[int] = set()
    for s in sorted(sections, key=lambda x: -x.confidence):
        if s.row_index in seen:
            continue
        seen.add(s.row_index)
        conn.execute(
            """
            INSERT INTO set_section_alignment
              (set_id, section_idx, set_start_s, set_end_s,
               ref_track_id, confidence, confidence_source, label,
               ref_start_s, ref_end_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                set_id, s.row_index, s.set_start_s, s.set_end_s,
                s.track_id, s.confidence, CONFIDENCE_SOURCE, s.label,
                s.ref_start_s, s.ref_end_s,
            ),
        )
    conn.commit()


# ---------- orchestration ---------------------------------------------------

def align_set(
    set_id: str, db_path: Path = DB_PATH, *, progress: bool = True,
    track_ids_filter: set[str] | None = None,
) -> None:
    with _connect(db_path) as conn:
        sa = _set_audio_row(conn, set_id)
        set_audio_id = int(sa["set_audio_id"])
        set_audio_path = Path(sa["path"])
        if not set_audio_path.exists():
            raise SystemExit(f"set_audio path missing: {set_audio_path}")

        mix_measures = _mix_measures_all(conn, set_audio_id)
        if not mix_measures:
            raise SystemExit(f"no set_measures for set_audio_id={set_audio_id}")

        refs, skipped = _load_tracklist_refs(
            conn, set_id, progress=progress,
            track_ids_filter=track_ids_filter,
        )
        if not refs:
            raise SystemExit("no refs with audio+measures for this set")
        if progress:
            print(f"[sota] set={set_id} mix_measures={len(mix_measures)} "
                  f"refs={len(refs)} skipped={len(skipped)}", flush=True)
            tag_counts: dict[str, int] = {}
            for r in refs:
                tag_counts[r.version_tag] = tag_counts.get(r.version_tag, 0) + 1
            print(f"[sota] tag distribution: {tag_counts}", flush=True)
            if skipped:
                reason_counts: dict[str, int] = {}
                for s in skipped:
                    reason_counts[s.reason] = reason_counts.get(s.reason, 0) + 1
                print(f"[sota] skipped-by-reason: {reason_counts}  "
                      "(add a URL on the 'Missing audio' UI page to recover)",
                      flush=True)

        mix_times, per_ref, per_ref_vit, per_ref_mt = _build_similarity_series(
            conn, set_audio_id, set_audio_path, mix_measures, refs,
            progress=progress,
        )

        # Drop refs that failed to embed (e.g. audio file unreadable).
        refs = [r for r in refs if r.label in per_ref]
        if not refs:
            raise SystemExit("no refs embedded successfully")

        # Fingerprint anchors per ref, plus raw-fp full_excl (union of
        # density clusters for 'full'-variant refs). This matches the
        # validated Phase-1+5+6 pipeline's behaviour (mean IoU 0.891 on
        # BB11 GT). An earlier attempt to derive full_excl from the
        # DECODED full-universe Viterbi covered 42 % of the mix and
        # caused Gnash to regress 0.857 → 0.27 (see eval harness
        # history). Raw-fp anchors are narrower and empirically better
        # on the fixture.
        anchors_by_label, raw_full_excl = _load_fingerprint_anchors(
            conn, set_id, mix_times, refs,
        )
        if progress:
            n_anchored = sum(1 for r in refs if r.label in anchors_by_label)
            print(f"[sota] fp anchors on {n_anchored}/{len(refs)} refs",
                  flush=True)

        # Group refs into universes.
        u_refs_by_u: dict[str, list[GtRef]] = {}
        for r in refs:
            u_refs_by_u.setdefault(_universe(r.version_tag), []).append(r)

        # Single-pass decode using the raw-fp full_excl (validated approach).
        full_excl = raw_full_excl
        T = len(mix_times)
        coverage = float(full_excl.sum()) / T if T else 0.0
        if coverage > ind._FP_FULL_EXCL_COVERAGE_CAP:
            if progress:
                print(f"[sota] full-track exclusion DISABLED: coverage "
                      f"{coverage:.2%} > cap {ind._FP_FULL_EXCL_COVERAGE_CAP:.0%} "
                      f"(too many full-tagged refs with fp hits; hard-masking "
                      f"the acap/instr universes would wipe real plays)",
                      flush=True)
            full_excl = np.zeros(T, dtype=bool)
        elif progress:
            print(f"[sota] full-track exclusion: "
                  f"{int(full_excl.sum())}/{T} measures "
                  f"({coverage:.2%}, from raw fingerprint anchors)", flush=True)

        # Full universe decodes without cross-universe exclusion (it IS
        # the source of full_excl, not a subject).
        decoded_by_u: dict[str, np.ndarray] = {}
        full_refs = u_refs_by_u.get("full", [])
        if full_refs:
            if progress:
                print(f"[viterbi] universe=full refs={len(full_refs)}",
                      flush=True)
            decoded_by_u["full"] = viterbi_universe(
                mix_times, per_ref, full_refs,
                anchors_by_label=anchors_by_label,
                full_exclusion_mask=None,
                mix_bpm=None, ref_bpm_by_label=None,
            )

        # Acappella / instrumental universes see the raw-fp full_excl.
        import time as _time
        for u_name, u_refs in u_refs_by_u.items():
            if u_name == "full":
                continue
            if progress:
                print(f"[viterbi] universe={u_name} refs={len(u_refs)}",
                      flush=True)
            _t0 = _time.perf_counter()
            decoded_by_u[u_name] = viterbi_universe(
                mix_times, per_ref, u_refs,
                anchors_by_label=anchors_by_label,
                full_exclusion_mask=full_excl,
                mix_bpm=None, ref_bpm_by_label=None,
            )
            if progress:
                print(f"[viterbi] {u_name} done in {_time.perf_counter()-_t0:.1f}s",
                      flush=True)

        # Precompute tracklist row_index lookup (one scan per set).
        row_index_by_tid = _tracklist_row_index_map(conn, set_id)
        if progress:
            print(f"[rowmap] {len(row_index_by_tid)} track_id → row_index entries",
                  flush=True)

        # Extract per-ref sections with cue-detr bracket + confidence.
        _t0 = _time.perf_counter()
        sections = _extract_sections(
            conn, set_id, refs, decoded_by_u, u_refs_by_u,
            mix_times, per_ref_vit, per_ref_mt,
            row_index_by_tid=row_index_by_tid,
        )
        if progress:
            print(f"[extract] {len(sections)} sections in {_time.perf_counter()-_t0:.1f}s",
                  flush=True)

        # Confidence = mean per-ref max-sim over the run window.
        ref_by_label = {r.label: r for r in refs}
        enriched: list[AlignedSection] = []
        for s in sections:
            ref = ref_by_label.get(s.label)
            if ref is None:
                continue
            maxsim = per_ref.get(s.label)
            if maxsim is None:
                continue
            lo = int(np.searchsorted(mix_times, s.set_start_s, side="left"))
            hi = int(np.searchsorted(mix_times, s.set_end_s, side="right"))
            hi = min(hi, len(maxsim))
            if hi <= lo:
                continue
            conf = float(np.mean(maxsim[lo:hi]))
            enriched.append(replace(s, confidence=conf))

        _persist(conn, set_id, enriched)

        if progress:
            print()
            print(f"[summary] aligned={len(enriched)}  refs_considered={len(refs)}",
                  flush=True)
            if enriched:
                confs = [s.confidence for s in enriched]
                print(f"[conf] mean={np.mean(confs):.3f} "
                      f"median={np.median(confs):.3f} "
                      f"min={min(confs):.3f} max={max(confs):.3f}", flush=True)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-id", required=True)
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument(
        "--track-ids",
        default=None,
        help="Comma-separated track_ids to restrict alignment to (for GT-only fast iteration).",
    )
    args = p.parse_args(argv)

    # Point the indicators_debug module-level DB_PATH at ours so its
    # helper DB queries use the right database file.
    ind.DB_PATH = Path(args.db)

    tid_filter: set[str] | None = None
    if args.track_ids:
        tid_filter = {t.strip() for t in args.track_ids.split(",") if t.strip()}

    align_set(args.set_id, Path(args.db), progress=True, track_ids_filter=tid_filter)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

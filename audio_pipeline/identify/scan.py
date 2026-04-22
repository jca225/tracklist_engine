"""Stage-1b: slide chromaprint over the full mix, produce per-window hits.

For each `hop_s`-spaced window of the mix, query every reference
fingerprint in the corpus and record hits above `min_similarity`.
Hits are persisted to `set_fingerprint_hits` as the input signal to
stage-3's MAP inference (identity term of the posterior).

Why this works where CCC didn't:

* Chromaprint hashes spectral-peak constellations — coincident
  chroma averages don't fool it, only matching peak patterns do.
* A real 3-min play produces ~30 aligned windows in a row; a chroma
  coincidence produces one. Post-hoc density gating filters the
  latter for free.
* Cue-independent: works on unidentified segments too, supplies
  evidence where the scraped tracklist is silent.

The window-by-window approach is O(N_windows × N_refs × fingerprint
length). For 3,000 refs × 15-min mix at 2s hop × 120 hashes per
compare ≈ 162M ops per set — half a second in numpy.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..errors import DbError
from ..result import Err, Ok, Result
from .acoustid_adapter import Fingerprint, compute, decode_hashes, similarity


# Chromaprint default hash rate is ~8 hashes/sec. Window and hop are
# expressed in seconds here and converted to hash counts before the
# slide — easier to reason about against wall-clock mix time.
HASHES_PER_SECOND: float = 7.8125
DEFAULT_WINDOW_S: float = 10.0
DEFAULT_HOP_S: float = 2.0
DEFAULT_MIN_SIMILARITY: float = 0.65


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


@dataclass(frozen=True)
class FingerprintHit:
    set_id: str
    mix_start_s: float
    mix_end_s: float
    matched_track_id: str
    matched_variant: str
    score: float


def persist_track_fingerprint(
    db_path: Path, track_id: str, fp: Fingerprint, variant_tag: str = "original",
) -> Result[None, DbError]:
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO track_fingerprints
                  (track_id, variant_tag, fingerprint, duration_s)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(track_id, variant_tag) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    duration_s  = excluded.duration_s,
                    created_at  = CURRENT_TIMESTAMP
                """,
                (track_id, variant_tag, fp.raw, fp.duration_s),
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def load_corpus_fingerprints(
    db_path: Path, track_ids: tuple[str, ...] | None = None,
) -> Result[dict[tuple[str, str], np.ndarray], DbError]:
    """Load all persisted fingerprints as (track_id, variant) → hash array.

    Passing `track_ids` scopes the corpus to a subset — useful when
    scanning a specific mix against only its scraped tracks (hot path
    for Phase-3 inference). None loads everything (full-corpus scan,
    used for identifying gaps).
    """
    try:
        with _connect(db_path) as conn:
            if track_ids:
                placeholders = ",".join("?" for _ in track_ids)
                rows = conn.execute(
                    f"""
                    SELECT track_id, variant_tag, fingerprint
                    FROM track_fingerprints
                    WHERE track_id IN ({placeholders})
                    """,
                    track_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT track_id, variant_tag, fingerprint FROM track_fingerprints"
                ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    out: dict[tuple[str, str], np.ndarray] = {}
    for r in rows:
        try:
            hashes = decode_hashes(r["fingerprint"])
        except Exception:   # noqa: BLE001 — malformed rows skipped
            continue
        if hashes.size > 0:
            out[(r["track_id"], r["variant_tag"])] = hashes
    return Ok(out)


def scan_mix(
    mix_hashes: np.ndarray,
    corpus: dict[tuple[str, str], np.ndarray],
    *,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[tuple[float, float, str, str, float]]:
    """Slide `window_s` over `mix_hashes` and emit hits for any
    reference whose fingerprint locally matches above `min_similarity`.

    Output tuples are (mix_start_s, mix_end_s, track_id, variant, score),
    ordered by (start, score desc). Ties at the same window are kept —
    mashup sections legitimately trigger multiple tracks and the
    downstream MAP inference should see them all.
    """
    if mix_hashes.size == 0 or not corpus:
        return []

    window_h = max(1, int(window_s * HASHES_PER_SECOND))
    hop_h = max(1, int(hop_s * HASHES_PER_SECOND))
    hits: list[tuple[float, float, str, str, float]] = []

    for start_h in range(0, mix_hashes.size - window_h + 1, hop_h):
        slice_ = mix_hashes[start_h:start_h + window_h]
        start_s = start_h / HASHES_PER_SECOND
        end_s = (start_h + window_h) / HASHES_PER_SECOND

        for (track_id, variant), ref_hashes in corpus.items():
            if ref_hashes.size == 0:
                continue
            # For long refs, scan for best offset within the ref (the
            # mix window may align to any point in the track). Coarse
            # step — every 4 hashes ≈ 0.5s — is enough to find the
            # rough placement; exact placement isn't what this stage
            # is solving, that's alignment's job.
            best = 0.0
            stride = max(4, window_h // 2)
            for off in range(0, ref_hashes.size - window_h + 1, stride):
                s = similarity(slice_, ref_hashes[off:off + window_h])
                if s > best:
                    best = s
                if best >= 0.95:
                    break       # saturated, stop scanning this ref
            if best >= min_similarity:
                hits.append((start_s, end_s, track_id, variant, best))
    return hits


def persist_hits(
    db_path: Path, set_id: str,
    hits: list[tuple[float, float, str, str, float]],
) -> Result[int, DbError]:
    """Bulk-insert fingerprint hits into `set_fingerprint_hits`.

    Replaces any prior hits for this set — the scan output is
    deterministic given current corpus + current mix audio, so leaving
    stale rows around would just confuse downstream consumers.
    """
    try:
        with _connect(db_path) as conn:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM set_fingerprint_hits WHERE set_id = ?", (set_id,))
            conn.executemany(
                """
                INSERT INTO set_fingerprint_hits
                  (set_id, mix_start_s, mix_end_s, matched_track_id, matched_variant, score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (set_id, start_s, end_s, tid, variant, score)
                    for (start_s, end_s, tid, variant, score) in hits
                ],
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(len(hits))


def run_scan_for_set(
    db_path: Path, set_id: str, *,
    scope_to_scraped_tracks: bool = True,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> Result[int, DbError]:
    """End-to-end: compute mix fingerprint, scan against corpus, persist hits.

    `scope_to_scraped_tracks=True` (the default) restricts the corpus to
    the track_ids linked to this set via `dj_set_track_media_links` —
    dramatically faster and avoids spurious cross-set hits on remixes
    of the same base track. Set False for "find anything playing here"
    usage (e.g. identifying unidentified rows).
    """
    # Mix audio path.
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT path FROM set_audio WHERE set_id = ?
                ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1
                """,
                (set_id,),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    if row is None:
        return Err(DbError(kind="not_found", detail=f"no set_audio for {set_id!r}"))

    mix_fp_r = compute(Path(row["path"]))
    if not mix_fp_r.is_ok():
        return Err(DbError(kind="query_failed", detail=f"mix fingerprint: {mix_fp_r.error}"))
    mix_hashes = mix_fp_r.value.hashes

    # Corpus scope.
    scoped_ids: tuple[str, ...] | None = None
    if scope_to_scraped_tracks:
        try:
            with _connect(db_path) as conn:
                tids = conn.execute(
                    """
                    SELECT DISTINCT track_id FROM dj_set_track_media_links
                    WHERE set_id = ? AND track_id IS NOT NULL
                    """,
                    (set_id,),
                ).fetchall()
        except sqlite3.DatabaseError as e:
            return Err(DbError(kind="query_failed", detail=str(e)))
        scoped_ids = tuple(r["track_id"] for r in tids)

    corpus_r = load_corpus_fingerprints(db_path, scoped_ids)
    if not corpus_r.is_ok():
        return corpus_r

    hits = scan_mix(
        mix_hashes, corpus_r.value,
        window_s=window_s, hop_s=hop_s, min_similarity=min_similarity,
    )
    return persist_hits(db_path, set_id, hits)


def ingest_all_refs(
    db_path: Path, track_ids: tuple[str, ...] | None = None,
) -> Result[tuple[int, int], DbError]:
    """Compute + persist fingerprints for every track_audio row (or a
    filtered subset). Skips entries already present. Returns
    (n_persisted, n_skipped)."""
    try:
        with _connect(db_path) as conn:
            if track_ids:
                placeholders = ",".join("?" for _ in track_ids)
                rows = conn.execute(
                    f"""
                    SELECT ta.track_id, ta.path
                    FROM track_audio ta
                    WHERE ta.track_id IN ({placeholders})
                      AND ta.track_id NOT IN (
                          SELECT track_id FROM track_fingerprints WHERE variant_tag = 'original'
                      )
                    ORDER BY ta.is_reference DESC, ta.downloaded_at DESC
                    """,
                    track_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ta.track_id, ta.path
                    FROM track_audio ta
                    WHERE ta.track_id NOT IN (
                        SELECT track_id FROM track_fingerprints WHERE variant_tag = 'original'
                    )
                    ORDER BY ta.is_reference DESC, ta.downloaded_at DESC
                    """
                ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    persisted = 0
    skipped = 0
    seen: set[str] = set()
    for r in rows:
        tid = r["track_id"]
        if tid in seen:
            continue
        seen.add(tid)
        fp_r = compute(Path(r["path"]))
        if not fp_r.is_ok():
            skipped += 1
            continue
        pr = persist_track_fingerprint(db_path, tid, fp_r.value)
        if pr.is_ok():
            persisted += 1
        else:
            skipped += 1
    return Ok((persisted, skipped))


# Variants to fingerprint beyond the full track. Tracks used as
# acappellas need `vocals`-stem fingerprints to be detectable when
# the mix is playing only the vocal layer; same logic in reverse
# for instrumentals. Full-audio fingerprints alone miss these
# layered plays, which is exactly what happens to Call Me Maybe
# at 86-101s in Big Bootie 11.
STEM_VARIANTS: tuple[str, ...] = ("vocals", "instrumental")


def ingest_all_refs_variants(
    db_path: Path,
    track_ids: tuple[str, ...] | None = None,
    variants: tuple[str, ...] = STEM_VARIANTS,
) -> Result[dict[str, tuple[int, int]], DbError]:
    """Compute + persist per-stem-variant fingerprints for every track
    that has the corresponding demucs stem on disk.

    Returns {variant → (persisted, skipped)} so callers can report
    per-variant coverage.
    """
    # Pull (track_id, stem_name, stem_path) rows we should fingerprint,
    # skipping (track_id, variant) pairs that already have a fingerprint.
    placeholders = ""
    params: tuple = ()
    where_scope = ""
    if track_ids:
        placeholders = ",".join("?" for _ in track_ids)
        where_scope = f"AND ta.track_id IN ({placeholders})"
        params = track_ids

    counts: dict[str, tuple[int, int]] = {}
    for variant in variants:
        try:
            with _connect(db_path) as conn:
                rows = conn.execute(
                    f"""
                    SELECT ta.track_id, ts.path
                    FROM track_stems ts
                    JOIN track_audio ta ON ta.track_audio_id = ts.track_audio_id
                    WHERE ts.stem_name = ?
                      {where_scope}
                      AND NOT EXISTS (
                          SELECT 1 FROM track_fingerprints tf
                          WHERE tf.track_id = ta.track_id AND tf.variant_tag = ?
                      )
                    ORDER BY ta.is_reference DESC, ta.downloaded_at DESC
                    """,
                    (variant, *params, variant),
                ).fetchall()
        except sqlite3.DatabaseError as e:
            return Err(DbError(kind="query_failed", detail=str(e)))

        persisted = 0
        skipped = 0
        seen: set[str] = set()
        for r in rows:
            tid = r["track_id"]
            if tid in seen:
                continue
            seen.add(tid)
            fp_r = compute(Path(r["path"]))
            if not fp_r.is_ok():
                skipped += 1
                continue
            pr = persist_track_fingerprint(db_path, tid, fp_r.value, variant_tag=variant)
            if pr.is_ok():
                persisted += 1
            else:
                skipped += 1
        counts[variant] = (persisted, skipped)
    return Ok(counts)


def run_scan_for_set_variants(
    db_path: Path,
    set_id: str,
    *,
    variants: tuple[str, ...] = ("original", *STEM_VARIANTS),
    scope_to_scraped_tracks: bool = True,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> Result[dict[str, int], DbError]:
    """Multi-variant scan: for each variant, fingerprint the matching
    mix stream and scan against refs with that variant.

    * `original` → fingerprint full-mix audio, scan against
      `variant_tag='original'` corpus (what `run_scan_for_set` does).
    * `vocals` → fingerprint mix's vocals stem, scan against
      `variant_tag='vocals'` corpus. Detects acappella plays the
      full-audio scan misses.
    * `instrumental` → fingerprint mix's `instrumental.wav` (the
      pre-summed drums+bass+other), scan against `variant_tag=
      'instrumental'`. Detects instrumental plays layered under
      acappellas.

    Hits across variants are *appended* — persist_hits wipes prior
    rows for the set, so we collect everything first and write once.
    Returns {variant → n_hits_added}.
    """
    mix_paths_by_variant = _resolve_mix_paths_by_variant(db_path, set_id, variants)
    if not mix_paths_by_variant.is_ok():
        return mix_paths_by_variant

    # Corpus scope.
    scoped_ids: tuple[str, ...] | None = None
    if scope_to_scraped_tracks:
        try:
            with _connect(db_path) as conn:
                tids = conn.execute(
                    """
                    SELECT DISTINCT track_id FROM dj_set_track_media_links
                    WHERE set_id = ? AND track_id IS NOT NULL
                    """,
                    (set_id,),
                ).fetchall()
        except sqlite3.DatabaseError as e:
            return Err(DbError(kind="query_failed", detail=str(e)))
        scoped_ids = tuple(r["track_id"] for r in tids)

    all_hits: list[tuple[float, float, str, str, float]] = []
    counts: dict[str, int] = {}

    for variant, mix_path in mix_paths_by_variant.value.items():
        mix_fp_r = compute(mix_path)
        if not mix_fp_r.is_ok():
            counts[variant] = 0
            continue
        mix_hashes = mix_fp_r.value.hashes

        corpus_r = load_corpus_fingerprints_by_variant(
            db_path, variant, scoped_ids,
        )
        if not corpus_r.is_ok():
            counts[variant] = 0
            continue
        corpus = corpus_r.value

        hits = scan_mix(
            mix_hashes, corpus,
            window_s=window_s, hop_s=hop_s, min_similarity=min_similarity,
        )
        counts[variant] = len(hits)
        all_hits.extend(hits)

    # Single write of the fused hit list.
    persist_r = persist_hits(db_path, set_id, all_hits)
    if not persist_r.is_ok():
        return persist_r
    return Ok(counts)


def _resolve_mix_paths_by_variant(
    db_path: Path, set_id: str, variants: tuple[str, ...],
) -> Result[dict[str, Path], DbError]:
    """Map each requested variant → path to the matching mix audio.

    Variants without an available on-disk file (e.g. set with no
    demucs output) are silently omitted — caller sees the gap via
    the returned dict's keys.
    """
    try:
        with _connect(db_path) as conn:
            main = conn.execute(
                """
                SELECT path, set_audio_id FROM set_audio WHERE set_id = ?
                ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1
                """,
                (set_id,),
            ).fetchone()
            if main is None:
                return Err(DbError(kind="not_found", detail=f"no set_audio for {set_id!r}"))
            stem_rows = conn.execute(
                """
                SELECT stem_name, path FROM set_stems WHERE set_audio_id = ?
                """,
                (int(main["set_audio_id"]),),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    stems = {r["stem_name"]: Path(r["path"]) for r in stem_rows}
    out: dict[str, Path] = {}
    for v in variants:
        if v == "original":
            out[v] = Path(main["path"])
        elif v in stems:
            out[v] = stems[v]
    return Ok(out)


def load_corpus_fingerprints_by_variant(
    db_path: Path, variant_tag: str, track_ids: tuple[str, ...] | None = None,
) -> Result[dict[tuple[str, str], np.ndarray], DbError]:
    """Load fingerprints for a specific variant_tag.

    Wraps `load_corpus_fingerprints`'s general case — kept separate
    so the multi-variant scan stays readable at the call site.
    """
    try:
        with _connect(db_path) as conn:
            if track_ids:
                placeholders = ",".join("?" for _ in track_ids)
                rows = conn.execute(
                    f"""
                    SELECT track_id, variant_tag, fingerprint
                    FROM track_fingerprints
                    WHERE variant_tag = ? AND track_id IN ({placeholders})
                    """,
                    (variant_tag, *track_ids),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT track_id, variant_tag, fingerprint FROM track_fingerprints "
                    "WHERE variant_tag = ?",
                    (variant_tag,),
                ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    out: dict[tuple[str, str], np.ndarray] = {}
    for r in rows:
        try:
            hashes = decode_hashes(r["fingerprint"])
        except Exception:   # noqa: BLE001
            continue
        if hashes.size > 0:
            out[(r["track_id"], r["variant_tag"])] = hashes
    return Ok(out)

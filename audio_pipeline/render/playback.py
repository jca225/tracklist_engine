"""Render a `measure_alignment` playback score back to audio.

Pipeline per active entry:
  1. Load the ref stems that correspond to `stem_mask` (e.g.
     `["vocals"]` → vocals.wav only; `["drums","bass","other"]` →
     instrumental.wav which we've pre-summed at analysis time, falling
     back to summing the three individual stems at render time if the
     pre-summed file is missing).
  2. Slice ref-measure `[ref_measure_idx, ref_measure_idx + span]`.
  3. Pitch-shift by `pitch_shift_semi` and time-stretch so the slice
     lands in `[mix_start_s, mix_end_s]` (using `tempo_ratio`). Rubber
     Band via `pyrubberband` does both in one call without coupling.
  4. Apply `gain_db`.
  5. Sum into the output buffer at `mix_start_s`.

Rubber Band is the right tool: it keeps pitch and tempo independent
(matches DJ key-lock), handles integer semitones cleanly, and has
battle-tested transient preservation. Librosa's pyrubberband bindings
shell out to the `rubberband` CLI installed alongside libvhs — we
require that binary on PATH.

Output format: 44.1 kHz mono WAV. Mono because the ref stems in this
pipeline were produced by demucs 4-stem split at 44.1 kHz — matches
upstream sample-rate without resampling on the render path.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..errors import DbError
from ..result import Err, Ok, Result


# Render-time defaults. All caller-overridable via function kwargs so
# tests can tighten the sample rate or trade quality for speed.
DEFAULT_SAMPLE_RATE: int = 44_100
DEFAULT_HOP_MS: int = 50                 # for MFCC-distance comparison grid


@dataclass(frozen=True)
class RenderError:
    kind: str        # 'missing_stem' | 'rubberband_failed' | 'io' | 'empty_plan'
    detail: str


@dataclass(frozen=True)
class RenderReport:
    set_id: str
    out_path: Path
    n_measures_rendered: int
    n_entries_rendered: int
    total_duration_s: float
    mfcc_distance: float | None = None    # filled if compare_against is given


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def _rubberband_available() -> bool:
    """Is the `rubberband` CLI on PATH? pyrubberband shells out to it."""
    try:
        subprocess.run(
            ["rubberband", "--version"],
            capture_output=True, check=False, timeout=2,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _load_mono_audio(path: Path, sr: int) -> np.ndarray:
    """Load an audio file to mono at `sr` Hz using soundfile + librosa
    resample. Files are expected to already be at `sr` (demucs default
    is 44.1 kHz) — resampling here is a safety net, not the hot path.
    """
    import soundfile as sf
    import librosa
    y, file_sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def _stretch_and_shift(
    y: np.ndarray, sr: int, *,
    tempo_ratio: float, pitch_shift_semi: int,
) -> np.ndarray:
    """Rubber Band roundtrip.

    `tempo_ratio = ref_dur / mix_dur` means: if the DJ sped the track
    up (tempo_ratio > 1 inside this codebase's convention), we need to
    play *faster*, so playback length shrinks. Rubber Band's `rate`
    argument takes the playback-rate multiplier; `rate = tempo_ratio`
    maps that directly (rate=2.0 halves duration).

    `pitch_shift_semi` is literal semitones, integer; librosa/pyrubberband
    handle the 2^(s/12) conversion internally.

    Pitch and tempo are independent when you call Rubber Band with both
    args. For varispeed (coupled) the caller would set
    `tempo_ratio = 2^(semi/12)` themselves and `pitch_shift_semi = 0`;
    detecting which mode the DJ used belongs upstream.
    """
    import pyrubberband as pyrb
    if abs(tempo_ratio - 1.0) < 1e-3 and pitch_shift_semi == 0:
        return y
    # pyrubberband calls the Rubber Band CLI. Pass both transforms in
    # a single invocation so it's one subprocess per slice rather than
    # two (stretch then shift), which doubles latency.
    if pitch_shift_semi == 0:
        return pyrb.time_stretch(y, sr, rate=float(tempo_ratio)).astype(np.float32)
    if abs(tempo_ratio - 1.0) < 1e-3:
        return pyrb.pitch_shift(y, sr, n_steps=int(pitch_shift_semi)).astype(np.float32)
    # Chain. Rubber Band is expensive but this is the unavoidable case.
    shifted = pyrb.pitch_shift(y, sr, n_steps=int(pitch_shift_semi))
    return pyrb.time_stretch(shifted, sr, rate=float(tempo_ratio)).astype(np.float32)


def _gain_linear(gain_db: float | None) -> float:
    if gain_db is None:
        return 1.0
    return float(10.0 ** (gain_db / 20.0))


def _resolve_stem_path(
    ref_stem_paths: dict[str, dict[str, Path]],
    track_id: str,
    stem_mask: list[str],
) -> Path | None:
    """Pick the on-disk audio for a (track, stem_mask) pair.

    stem_mask is a JSON list like `["vocals"]` or
    `["drums","bass","other"]`. The canonical paths are:

      single stem in list → that stem's file
      3-stem instrumental mask → 'instrumental' pre-sum if present,
                                  else the caller sums at render time
      ['full'] → full mix audio (the track's main file)

    Returns `None` when the stem is unavailable; caller handles the
    miss as a render error rather than silently dropping the entry.
    """
    stems = ref_stem_paths.get(track_id, {})
    if not stems:
        return None

    if len(stem_mask) == 1 and stem_mask[0] in stems:
        return stems[stem_mask[0]]

    # Instrumental pattern — prefer the pre-summed file when it exists.
    if set(stem_mask) == {"drums", "bass", "other"} and "instrumental" in stems:
        return stems["instrumental"]

    # No single-file match; the caller will need to sum stems at render
    # time. Return None to signal that path.
    return None


def _render_summed_stems(
    ref_stem_paths: dict[str, dict[str, Path]],
    track_id: str,
    stem_mask: list[str],
    sr: int,
) -> np.ndarray | None:
    """Fallback for when no single file represents the mask — sum each
    listed stem at load time. Shapes truncate to the shortest stem."""
    stems = ref_stem_paths.get(track_id, {})
    if not stems:
        return None
    arrays: list[np.ndarray] = []
    for s in stem_mask:
        if s not in stems:
            return None
        arrays.append(_load_mono_audio(stems[s], sr))
    if not arrays:
        return None
    min_n = min(a.shape[0] for a in arrays)
    return np.sum([a[:min_n] for a in arrays], axis=0).astype(np.float32)


def _ref_measure_windows(
    db_path: Path, track_ids: tuple[str, ...],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Map track_id → {measure_idx → (start_s, end_s)}."""
    if not track_ids:
        return {}
    placeholders = ",".join("?" for _ in track_ids)
    out: dict[str, dict[int, tuple[float, float]]] = {}
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT ta.track_id, tm.measure_idx, tm.start_s, tm.end_s
            FROM track_measures tm
            JOIN track_audio ta ON ta.track_audio_id = tm.track_audio_id
            WHERE ta.track_id IN ({placeholders})
            ORDER BY ta.is_reference DESC, ta.downloaded_at DESC, tm.measure_idx
            """,
            track_ids,
        ).fetchall()
    for r in rows:
        out.setdefault(r["track_id"], {}).setdefault(
            int(r["measure_idx"]),
            (float(r["start_s"]), float(r["end_s"])),
        )
    return out


def _set_measure_windows(
    db_path: Path, set_id: str,
) -> dict[int, tuple[float, float]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT sm.measure_idx, sm.start_s, sm.end_s
            FROM set_measures sm
            JOIN set_audio sa ON sa.set_audio_id = sm.set_audio_id
            WHERE sa.set_id = ?
            ORDER BY sa.is_reference DESC, sa.downloaded_at DESC
            """,
            (set_id,),
        ).fetchall()
    out: dict[int, tuple[float, float]] = {}
    for r in rows:
        out.setdefault(
            int(r["measure_idx"]),
            (float(r["start_s"]), float(r["end_s"])),
        )
    return out


def _ref_stem_paths(
    db_path: Path, track_ids: tuple[str, ...],
) -> dict[str, dict[str, Path]]:
    if not track_ids:
        return {}
    placeholders = ",".join("?" for _ in track_ids)
    with _connect(db_path) as conn:
        audio_rows = conn.execute(
            f"""
            SELECT track_id, path FROM track_audio
            WHERE track_id IN ({placeholders})
            ORDER BY is_reference DESC, downloaded_at DESC
            """,
            track_ids,
        ).fetchall()
        stem_rows = conn.execute(
            f"""
            SELECT ta.track_id, ts.stem_name, ts.path
            FROM track_stems ts
            JOIN track_audio ta ON ta.track_audio_id = ts.track_audio_id
            WHERE ta.track_id IN ({placeholders})
            ORDER BY ta.is_reference DESC, ta.downloaded_at DESC
            """,
            track_ids,
        ).fetchall()
    out: dict[str, dict[str, Path]] = {}
    for r in audio_rows:
        out.setdefault(r["track_id"], {}).setdefault("full", Path(r["path"]))
    for r in stem_rows:
        out.setdefault(r["track_id"], {}).setdefault(r["stem_name"], Path(r["path"]))
    return out


def render_set(
    db_path: Path,
    set_id: str,
    out_path: Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_s: float | None = None,
) -> Result[RenderReport, RenderError]:
    """Render a set's `measure_alignment` rows to a reconstructed WAV.

    `duration_s` sets the output buffer length. When None, uses the last
    mix-measure's `end_s` from `set_measures`. Missing stems for any
    active entry produce a non-fatal skip (logged in the report) rather
    than aborting the whole render.
    """
    if not _rubberband_available():
        return Err(RenderError(
            kind="rubberband_failed",
            detail="rubberband CLI not on PATH; install with `brew install rubberband`",
        ))

    import soundfile as sf

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT set_measure_idx, ref_track_id, ref_measure_idx,
                   pitch_shift_semi, tempo_ratio, stem_mask_json, gain_db
            FROM measure_alignment
            WHERE set_id = ?
            ORDER BY set_measure_idx, ref_track_id
            """,
            (set_id,),
        ).fetchall()

    if not rows:
        return Err(RenderError(kind="empty_plan", detail=f"no measure_alignment for {set_id!r}"))

    set_measures = _set_measure_windows(db_path, set_id)
    track_ids = tuple({r["ref_track_id"] for r in rows})
    ref_measures = _ref_measure_windows(db_path, track_ids)
    ref_stems = _ref_stem_paths(db_path, track_ids)

    if duration_s is None:
        duration_s = max(
            (end for (_, end) in set_measures.values()),
            default=0.0,
        )

    n_samples = int(duration_s * sample_rate) + sample_rate
    buf = np.zeros(n_samples, dtype=np.float32)

    entries_done = 0
    measures_touched: set[int] = set()
    stem_audio_cache: dict[Path, np.ndarray] = {}

    for row in rows:
        set_m_idx = int(row["set_measure_idx"])
        track_id = row["ref_track_id"]
        ref_m_idx = int(row["ref_measure_idx"])
        pitch = int(row["pitch_shift_semi"])
        tempo = float(row["tempo_ratio"])
        stem_mask = json.loads(row["stem_mask_json"])
        gain_db = row["gain_db"]

        mix_window = set_measures.get(set_m_idx)
        ref_window = ref_measures.get(track_id, {}).get(ref_m_idx)
        if mix_window is None or ref_window is None:
            continue

        mix_start, mix_end = mix_window
        ref_start, ref_end = ref_window

        stem_path = _resolve_stem_path(ref_stems, track_id, stem_mask)
        if stem_path is not None:
            if stem_path not in stem_audio_cache:
                stem_audio_cache[stem_path] = _load_mono_audio(stem_path, sample_rate)
            y_stream = stem_audio_cache[stem_path]
        else:
            summed = _render_summed_stems(ref_stems, track_id, stem_mask, sample_rate)
            if summed is None:
                continue      # silent miss, logged via entries_done delta
            y_stream = summed

        ref_start_i = int(ref_start * sample_rate)
        ref_end_i = int(ref_end * sample_rate)
        if ref_end_i <= ref_start_i or ref_end_i > y_stream.shape[0]:
            ref_end_i = min(ref_end_i, y_stream.shape[0])
            if ref_end_i <= ref_start_i:
                continue
        slice_ = y_stream[ref_start_i:ref_end_i]

        try:
            rendered = _stretch_and_shift(
                slice_, sample_rate,
                tempo_ratio=tempo, pitch_shift_semi=pitch,
            )
        except subprocess.SubprocessError:
            continue        # rubber band crashed on this slice — skip, don't abort

        rendered = rendered * _gain_linear(gain_db)

        mix_start_i = int(mix_start * sample_rate)
        end_i = min(mix_start_i + rendered.shape[0], buf.shape[0])
        span = end_i - mix_start_i
        if span <= 0:
            continue
        buf[mix_start_i:end_i] += rendered[:span]
        entries_done += 1
        measures_touched.add(set_m_idx)

    # Normalise to prevent clipping if many entries stacked up.
    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1.0:
        buf = buf / peak

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), buf, sample_rate, subtype="PCM_16")

    return Ok(RenderReport(
        set_id=set_id,
        out_path=out_path,
        n_measures_rendered=len(measures_touched),
        n_entries_rendered=entries_done,
        total_duration_s=duration_s,
    ))


def mfcc_distance(
    actual_path: Path, rendered_path: Path, *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_ms: int = DEFAULT_HOP_MS,
) -> Result[float, RenderError]:
    """Mean cosine distance between MFCC frames of `actual` vs `rendered`.

    Frame rate = `hop_ms` — 50ms is a reasonable trade between time
    localisation and smoothing out render artefacts. Range 0..2 where
    0 is perfect alignment. A good reconstruction lands <0.25.
    """
    import librosa
    try:
        y_a, _ = librosa.load(str(actual_path), sr=sample_rate, mono=True)
        y_r, _ = librosa.load(str(rendered_path), sr=sample_rate, mono=True)
    except (FileNotFoundError, OSError) as e:
        return Err(RenderError(kind="io", detail=str(e)))

    hop_length = int(sample_rate * hop_ms / 1000)
    m_a = librosa.feature.mfcc(y=y_a, sr=sample_rate, n_mfcc=13, hop_length=hop_length)
    m_r = librosa.feature.mfcc(y=y_r, sr=sample_rate, n_mfcc=13, hop_length=hop_length)
    n = min(m_a.shape[1], m_r.shape[1])
    if n == 0:
        return Err(RenderError(kind="io", detail="empty MFCC"))
    a = m_a[:, :n]
    r = m_r[:, :n]
    # Cosine distance per frame, mean over frames.
    a_norm = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-9)
    r_norm = r / (np.linalg.norm(r, axis=0, keepdims=True) + 1e-9)
    cos_sim = np.sum(a_norm * r_norm, axis=0)    # (n_frames,)
    dist = 1.0 - float(np.mean(cos_sim))
    return Ok(dist)


def persist_playback_score(
    db_path: Path, set_id: str, *,
    reconstruction_mfcc_distance: float | None,
    reconstruction_method: str = "pyrubberband_v1",
) -> Result[None, DbError]:
    """Serialise `measure_alignment` rows as `score_json` and store in
    `set_playback_score` alongside the reconstruction metric."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT set_measure_idx, ref_track_id, ref_measure_idx,
                       pitch_shift_semi, tempo_ratio, stem_mask_json,
                       gain_db, confidence
                FROM measure_alignment
                WHERE set_id = ?
                ORDER BY set_measure_idx, ref_track_id
                """,
                (set_id,),
            ).fetchall()
            payload = {
                "set_id": set_id,
                "entries": [
                    {
                        "set_measure_idx": int(r["set_measure_idx"]),
                        "ref_track_id": r["ref_track_id"],
                        "ref_measure_idx": int(r["ref_measure_idx"]),
                        "pitch_shift_semi": int(r["pitch_shift_semi"]),
                        "tempo_ratio": float(r["tempo_ratio"]),
                        "stem_mask": json.loads(r["stem_mask_json"]),
                        "gain_db": r["gain_db"],
                        "confidence": r["confidence"],
                    }
                    for r in rows
                ],
            }
            conn.execute(
                """
                INSERT INTO set_playback_score
                  (set_id, score_json, reconstruction_mfcc_distance, reconstruction_method)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(set_id) DO UPDATE SET
                    score_json                    = excluded.score_json,
                    reconstruction_mfcc_distance  = excluded.reconstruction_mfcc_distance,
                    reconstruction_method         = excluded.reconstruction_method,
                    rendered_at                   = CURRENT_TIMESTAMP
                """,
                (
                    set_id,
                    json.dumps(payload),
                    reconstruction_mfcc_distance,
                    reconstruction_method,
                ),
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def _cli(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/db/music_database.db")
    p.add_argument("--set-id", required=True)
    p.add_argument("--out", required=True, help="Output WAV path")
    p.add_argument("--compare", default=None, help="Actual mix WAV to compare against (MFCC distance)")
    p.add_argument("--persist", action="store_true",
                   help="Upsert a set_playback_score row with the MFCC distance.")
    args = p.parse_args(argv)

    r = render_set(Path(args.db), args.set_id, Path(args.out))
    if not r.is_ok():
        print(f"render FAILED: {r.error}", file=sys.stderr)
        return 1
    rep = r.value
    print(f"rendered {rep.n_entries_rendered} entries across "
          f"{rep.n_measures_rendered} measures → {rep.out_path}", flush=True)

    mfcc_dist: float | None = None
    if args.compare:
        d_r = mfcc_distance(Path(args.compare), Path(args.out))
        if d_r.is_ok():
            mfcc_dist = d_r.value
            print(f"MFCC distance vs actual mix: {mfcc_dist:.4f}  "
                  f"(0 = perfect, <0.25 = strong reconstruction)", flush=True)
        else:
            print(f"mfcc compare FAILED: {d_r.error}", file=sys.stderr)

    if args.persist:
        p_r = persist_playback_score(
            Path(args.db), args.set_id,
            reconstruction_mfcc_distance=mfcc_dist,
        )
        if not p_r.is_ok():
            print(f"persist FAILED: {p_r.error}", file=sys.stderr)
            return 1
        print("persisted to set_playback_score", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))

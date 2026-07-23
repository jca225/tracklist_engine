"""Robust stem-path resolution: manifest, audio_index, then unique disk fallback.

`labeling/acquire/pull_set_for_alignment.py` writes each track's ``stems`` field ONCE at
pull time; stems separated AFTER the pull (re-stem, phase-cancel, the annotator's
``[NNNbpm KK]`` tagging) never get written back, so ``manifest.stems.{vocals,
instrumental}`` drifts stale — measured: it records ~117 of 328 instrumental
stems actually on disk. The files DO exist, keyed by SLOT
(``stems/<slot>__<artist - title>/{stem}.flac``), often with several dirs per
slot (plain + annotator-tagged). ~50 aligner modules read the manifest field and
silently drop the spans it misses (this bit the agentic loop, the instrumental
probe, and joint_ref_decode).

This is the shared resolver: check the manifest field first, then
``audio_index.json`` by ``track_audio_id``, then accept an on-disk slot fallback
only when it identifies exactly one stem file. Ambiguous fallbacks abstain
rather than silently selecting a content-divergent copy.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from labeling.audio_index import has_audio_index, load_audio_index, lookup_stem

# identity stem axis -> Demucs/Roformer stem file basename. 'regular' = the full
# track (no stem file), so it is intentionally absent.
STEM_FILE = {"acappella": "vocals", "instrumental": "instrumental"}


def _slot_variants(slot_label: str) -> list[str]:
    """The on-disk stem dirs are zero-padded (``001w1__*``) but callers pass
    mixed forms — GT keeps ``001w1``, timelines normalize to ``1w1``. Try the
    raw value, the leading-zeros-stripped form, and the 3-digit zero-padded
    form so a lookup keyed either way still hits the disk dir."""
    s = str(slot_label)
    out = [s]
    m = re.match(r"^0*(\d+)(.*)$", s)
    if m:
        for cand in (m.group(1) + m.group(2), m.group(1).zfill(3) + m.group(2)):
            if cand not in out:
                out.append(cand)
    return out


def resolve_stem(
    set_dir: Path | None,
    slot_label: str | None,
    track: dict | None,
    stem_name: str,
) -> Path | None:
    """Real path to a track's ``stem_name`` ('vocals'|'instrumental') stem.

    1. ``audio_index.json`` by ``track_audio_id`` when present;
    2. ``track['stems'][stem_name]`` only when no index has been built;
    3. else collect ``set_dir/stems/<slot>__*/<stem_name>.flac`` across both
       slot forms;
    4. return the fallback only when exactly one distinct file exists;
    5. warn and abstain when multiple files exist.
    """
    if track:
        if has_audio_index(set_dir):
            return lookup_stem(
                load_audio_index(set_dir), track.get("track_audio_id"), stem_name
            )
        p = (track.get("stems") or {}).get(stem_name)
        if p and Path(p).is_file():
            return Path(p)
    if not set_dir or not slot_label or slot_label == "?":
        return None
    stems_root = Path(set_dir) / "stems"
    if not stems_root.is_dir():
        return None
    candidates: dict[Path, Path] = {}
    for slot in _slot_variants(slot_label):
        for directory in sorted(stems_root.glob(f"{slot}__*")):
            candidate = directory / f"{stem_name}.flac"
            if candidate.is_file():
                candidates[candidate.resolve()] = candidate
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if len(candidates) > 1:
        warnings.warn(
            "ambiguous stem fallback: "
            f"track_audio_id={(track or {}).get('track_audio_id', '?')} "
            f"slot={slot_label} stem={stem_name} candidates={len(candidates)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return None


def resolve_stem_for_span(
    set_dir: Path | None, span: dict, track: dict | None
) -> Path | None:
    """Convenience: resolve the stem implied by ``span['claimed_stem']``.

    Returns None for 'regular' (the full track is not a stem) — callers keep
    using the track's ``local_path`` for regular spans."""
    stem_key = STEM_FILE.get(span.get("claimed_stem") or "regular")
    if not stem_key:
        return None
    return resolve_stem(set_dir, span.get("slot_label"), track, stem_key)

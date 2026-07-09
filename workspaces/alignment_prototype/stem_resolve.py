"""Robust stem-path resolution: disk is truth, the manifest field is a hint.

`labeling/pull_set_for_alignment.py` writes each track's ``stems`` field ONCE at
pull time; stems separated AFTER the pull (re-stem, phase-cancel, the annotator's
``[NNNbpm KK]`` tagging) never get written back, so ``manifest.stems.{vocals,
instrumental}`` drifts stale — measured: it records ~117 of 328 instrumental
stems actually on disk. The files DO exist, keyed by SLOT
(``stems/<slot>__<artist - title>/{stem}.flac``), often with several dirs per
slot (plain + annotator-tagged). ~50 aligner modules read the manifest field and
silently drop the spans it misses (this bit the agentic loop, the instrumental
probe, and joint_ref_decode).

This is the shared resolver: check the manifest field first, then fall back to
the on-disk slot dirs. New code should call this instead of reading
``track['stems'][...]`` directly; existing readers can adopt incrementally.
"""

from __future__ import annotations

import re
from pathlib import Path

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


def _tagged_first(dirs: list[Path]) -> list[Path]:
    """Prefer the annotator's ``[NNNbpm KK]``-tagged dir — that's the canonical,
    re-pitched/tagged stem the human chose — over the plain pull-time dir."""
    tagged = [d for d in dirs if re.search(r"\[\d+bpm ", d.name)]
    return tagged + [d for d in dirs if d not in tagged]


def resolve_stem(
    set_dir: Path | None,
    slot_label: str | None,
    track: dict | None,
    stem_name: str,
) -> Path | None:
    """Real path to a track's ``stem_name`` ('vocals'|'instrumental') stem.

    1. ``track['stems'][stem_name]`` if it exists on disk (the manifest hint);
    2. else glob ``set_dir/stems/<slot>__*/<stem_name>.flac`` (both slot forms,
       annotator-tagged dir preferred) — the disk truth;
    3. else None.
    """
    if track:
        p = (track.get("stems") or {}).get(stem_name)
        if p and Path(p).is_file():
            return Path(p)
    if not set_dir or not slot_label or slot_label == "?":
        return None
    stems_root = Path(set_dir) / "stems"
    if not stems_root.is_dir():
        return None
    for slot in _slot_variants(slot_label):
        for d in _tagged_first(sorted(stems_root.glob(f"{slot}__*"))):
            f = d / f"{stem_name}.flac"
            if f.is_file():
                return f
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

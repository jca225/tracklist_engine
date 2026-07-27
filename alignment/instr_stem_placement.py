"""Instrumental stem-to-stem fingerprint placement channel for `infer`.

The regular-stem landmark fp localizes the mix<->ref diagonal to ~0.2s, but it
runs on the *full* mix and ref — for instrumental spans the vocals-carrying mix
bus and the vocal-less ref stem are timbrally mismatched, so the full-mix fp
(and chroma) fail on placement. Fingerprinting the INSTRUMENTAL stem on BOTH
sides (mix_instrumental.flac <-fp-> ref Demucs instrumental stem) removes the
mismatch and recovers regular-stem-level placement (probe: BB11 instrumental
88% id, 5.0s median set_start — see alignment/instrumental_probe.py).

This module is the productionized decode half of that probe, reusing its
fp-build + slot-resolution primitives verbatim (no re-derived math). `infer`
calls `place_instr_spans` under the opt-in `--instr-stem-placement` flag.

Gate note (STALE claimed_stem): `infer` reads `claimed_stem` from
set_track_slots on pi, which the row-text-marker drop bug left with ~2
instruments/set (vs ~25 real). If we gated purely on the span's DB stem this
channel would fire on 2 spans and be useless. `stem_overrides` (built from a GT
yaml — the same source the scorer trusts for the axis label, NOT for placement)
supplies the true per-span stem + the original zero-padded slot_label needed for
slot-glob ref resolution. Absent an override the caller logs the shortfall and
the channel fires only on the DB-visible instrumentals — it never silently
no-ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alignment.instrumental_probe import (
    _instr_stem,
    _manifest_by_tid,
    _resolve_stem_by_slot,
    build_ref_fp,
    mix_instr_hashes,
)
from alignment.mix_fp_hits import decode_placements


@dataclass(frozen=True)
class StemOverride:
    """Per-recording routing truth from a GT yaml: the (unambiguous) stem +
    the ORIGINAL zero-padded slot_label for slot-glob ref resolution. `infer`
    normalizes slots (strips leading zeros: 001w1->1w1), but the stem dirs on
    disk are zero-padded (stems/001w1__*/), so the glob needs the raw form."""

    claimed_stem: str
    raw_slot_label: str | None


def load_stem_overrides(
    gt_yaml: Path, *, id_map_dir: Path | None = None
) -> dict[str, StemOverride]:
    """{recording_id: StemOverride} — the instrumental-routing map.

    Keyed by the GT `track_id` (mapped tlp->recording_id via
    labeling/fixtures/id_maps/<set>.json, the same bridge the scorer uses, so
    keys line up with `infer` spans' recording_ids).

    AMBIGUITY GUARD: a recording that appears in the set as BOTH instrumental
    and some other stem (BB12: 6/23 instrumental recordings also play as
    regular/acappella) cannot be routed by recording_id alone — infer has no
    reliable per-span stem to disambiguate the siblings (GT slot_labels are the
    human's numbering, not infer's tracklist slots). We only emit an entry for a
    recording whose GT stems are EXACTLY {instrumental}; ambiguous recordings
    are omitted so the channel abstains on them rather than mis-marking the
    sibling regular/acappella span. This keeps the channel from ever regressing
    another stem, at the cost of not firing on the 6 shared recordings."""
    import json

    import yaml

    gt = yaml.safe_load(gt_yaml.read_text())
    set_id = str(gt.get("set_id") or "")
    id_map: dict[str, str] = {}
    if id_map_dir is not None and set_id:
        mp = id_map_dir / f"{set_id}.json"
        if mp.exists():
            id_map = json.loads(mp.read_text())

    stems_of: dict[str, set[str]] = {}
    slot_of: dict[str, str | None] = {}
    for r in gt.get("tracks", []):
        tid = r.get("track_id")
        if not tid:
            continue
        rid = id_map.get(tid, tid)
        stems_of.setdefault(rid, set()).add(r.get("claimed_stem") or "regular")
        if r.get("claimed_stem") == "instrumental":
            slot_of[rid] = r.get("slot_label") or None

    out: dict[str, StemOverride] = {}
    for rid, stems in stems_of.items():
        if stems == {"instrumental"}:
            out[rid] = StemOverride(
                claimed_stem="instrumental", raw_slot_label=slot_of.get(rid)
            )
    return out


def _resolve_ref_instr(
    set_dir: Path,
    by_tid: dict[str, dict],
    recording_id: str,
    raw_slot_label: str | None,
) -> str | None:
    """Ref instrumental-stem path for one span. Manifest field first (by
    recording_id), slot-glob fallback second — the probe's resolution order.
    The manifest `stems.instrumental` field is stale/incomplete (~5/26 BB11
    spans only resolve by slot), hence the fallback. Slot-glob needs the raw
    zero-padded slot_label (see StemOverride)."""
    ip = _instr_stem(by_tid.get(recording_id))
    if ip:
        return ip
    if raw_slot_label:
        sp = _resolve_stem_by_slot(set_dir, raw_slot_label)
        if sp:
            return sp
    return None


def place_instr_spans(
    preds: list,
    *,
    set_dir: Path,
    set_id: str,
    mix_dur_s: float,
    stem_overrides: dict[str, StemOverride] | None = None,
    topk: int = 6,
    gap_s: float = 6.0,
) -> tuple[dict[int, tuple[float, float, float]], dict[str, int]]:
    """Instrumental stem-fp placements keyed by span index.

    Returns ({idx: (set_start_s, set_end_s, offset_s)}, stats). Mirrors the
    `--fp-placement` decode: mix_instrumental.flac hashed once, each
    instrumental span's ref instrumental fp decoded via the SAME monotonic
    `decode_placements` primitive (with_offset), so ref_start_s = set_start_s +
    offset_s. The CALLER applies the consistency clamp against the prior
    placement (fp is precise but unleashed) exactly like `--fp-placement` — this
    returns the raw fp proposals ungated.

    Stem gate: when `stem_overrides` is given it is AUTHORITATIVE — a span is
    instrumental iff its recording_id is in the (ambiguity-filtered) override
    map, so shared-recording siblings never get mis-routed and the stale DB
    stem is ignored entirely. Without overrides we fall back to the DB stem
    (which marks ~2/set — the channel barely fires; the caller warns).
    `stats["db_instr"]` / `stats["gt_instr"]` expose the staleness."""
    ov = stem_overrides
    by_tid = _manifest_by_tid(set_dir, set_id)

    def _is_instr(i: int, p) -> bool:
        rid = getattr(p, "recording_id", None)
        if ov is not None:
            return rid in ov  # override map holds only instrumental recordings
        return (getattr(p, "claimed_stem", None) or "regular") == "instrumental"

    idxs = [i for i, p in enumerate(preds) if _is_instr(i, p)]
    db_instr = sum(
        1
        for p in preds
        if (getattr(p, "claimed_stem", None) or "regular") == "instrumental"
    )
    stats = {
        "gt_instr": len(idxs),
        "db_instr": db_instr,
        "resolved": 0,
        "placed": 0,
    }
    if not idxs:
        return {}, stats

    # resolve ref instrumental stems + build/load their fps (cached in fp_instr/)
    resolved: list[int] = []
    ref_fps: list = []
    for i in idxs:
        p = preds[i]
        rid = getattr(p, "recording_id", None)
        if not rid:
            continue
        o = ov.get(rid) if ov is not None else None
        raw_slot = o.raw_slot_label if o else None
        ipath = _resolve_ref_instr(set_dir, by_tid, rid, raw_slot)
        if ipath is None:
            continue
        fp = build_ref_fp(rid, ipath)
        if fp is None:
            continue
        resolved.append(i)
        ref_fps.append(fp)
    stats["resolved"] = len(resolved)
    if not ref_fps:
        return {}, stats

    mix_hashes = mix_instr_hashes(set_dir)
    # Feed resolved spans in their ORIGINAL (tracklist / slot) order — `preds`
    # come from predict_sequence in tracklist order, and decode_placements
    # enforces non-decreasing set_start over the list. Re-sorting by the prior
    # set_start would let a MERT-mislocated span scramble tracklist order and
    # corrupt the monotonic constraint; --fp-placement feeds `keep` in index
    # order for the same reason.
    placements = decode_placements(
        mix_hashes,
        ref_fps,
        mix_dur_s=mix_dur_s,
        topk=topk,
        gap_s=gap_s,
        with_offset=True,
    )
    out: dict[int, tuple[float, float, float]] = {}
    for i, pl in zip(resolved, placements):
        if pl is None:
            continue
        ss, se, off = pl
        out[i] = (float(ss), float(se), float(off))
    stats["placed"] = len(out)
    return out, stats

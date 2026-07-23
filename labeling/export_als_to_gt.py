"""Export BB12 (or any pulled set) Ableton `.als` → ground-truth YAML.

Reads the live `.als`, maps clips through 1-mix warp markers into mix seconds,
reads identity from each clip's file path, optionally attaches ``track_id`` when
the path exactly matches a row in `~/aligning/<set>/manifest.json` (pull
inventory only — not used for labels or timing), applies clip hygiene, and
writes a `*_ground_truth.yaml` consumable by `labeling.write_back_ground_truth`.

Usage (Mac, from repo root):

    venvs/audio/bin/python -m labeling.export_als_to_gt \\
        --als "$HOME/aligning/_labeling/1fsnxchk/BB12 align Project/bb12_align.als" \\
        --set-dir "$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12" \\
        --out labeling/fixtures/bb12_ground_truth.yaml

Review table only:

    venvs/audio/bin/python -m labeling.export_als_to_gt ... --review
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als import (
    ArrangementMapper,
    MUTE_THR,
    ParsedClip,
    audible_from_curve,
    clip_gain_breakpoints,
    build_manifest_index,
    classify_path,
    load_als_xml,
    parse_layer_clips,
    resolve_identity,
    select_arrangement_mapper,
    split_clip_at_mix_span_edges,
    tempo_ratio,
    track_display_name,
)
from labeling.als.validate import has_errors, validate_session
from labeling.identity.content_hash import file_sha256, mdat_sha256
from labeling.identity.content_resolver import (
    CatalogEntry,
    ContentCatalog,
    resolve_clip_identity,
)
from labeling.schema import (
    GroundTruthSet,
    GroundTruthTrack,
    RefSegment,
    save,
)
from core.result import Err, Ok

# Canonical BB12 labeling session, relocated 2026-07-21 (master plan §4): the
# .als project moved to ~/aligning/_labeling/1fsnxchk/, but the set-dir (manifest
# + tracks/ + stems/) stayed at the old path — hence the split below. The path
# convention itself (D22) is an open operator decision; update here if it lands.
DEFAULT_ALS = (
    Path.home() / "aligning/_labeling/1fsnxchk/BB12 align Project/bb12_align.als"
)
DEFAULT_SET_DIR = (
    Path.home() / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
)

PARKING_TOL_S = 5.0
SLIVER_MAX_S = 3.0
SLIVER_ADJACENCY_S = 0.1
SLIVER_REF_CONTINUITY_S = 1.5
MICRO_SLIVER_DROP_S = 0.01  # clip-boundary specks (e.g. slot 107)
LOOP_OVERLAP_MIN = 0.35
BLINK_182_SLOTS = frozenset({"024", "024w1", "029"})
AUDIBLE_EPS_S = 0.05


GainCurve = tuple[tuple[float, float], ...]


def _merge_curves(*curves: GainCurve) -> GainCurve:
    """Concatenate per-clip gain curves into one set-time-ordered envelope.

    Slot-merge (slivers / loop iterations) unions clips that occupy distinct
    mix regions; sorting by set-time stitches their fader curves into a single
    curve over the slot's whole mix span. Near-coincident points are deduped so
    a shared boundary doesn't create a zero-width segment."""
    pts: list[tuple[float, float]] = []
    for c in curves:
        pts.extend(c)
    pts.sort(key=lambda p: p[0])
    out: list[tuple[float, float]] = []
    for x, g in pts:
        if out and abs(x - out[-1][0]) < 1e-4:
            out[-1] = (x, max(out[-1][1], g))  # coincident -> keep louder
        else:
            out.append((x, g))
    return tuple(out)


def _audible_fields(
    curve: GainCurve, set_start: float, set_end: float
) -> tuple[float | None, float | None, float | None, bool]:
    """Derive (audible_frac, audible_start_s, audible_end_s, skip) from ONE
    gain curve — the single source that keeps the three fields consistent.

    Sparse-output convention: frac omitted when fully audible (==1.0), and
    start/end omitted when they coincide (within AUDIBLE_EPS_S) with the clip
    extent — i.e. absence of a field means 'audible across the whole span'."""
    frac, start, end = audible_from_curve(curve)
    frac_out = None if frac >= 1.0 else round(frac, 3)
    start_out = (
        None if start is None or abs(start - set_start) < AUDIBLE_EPS_S else start
    )
    end_out = None if end is None or abs(end - set_end) < AUDIBLE_EPS_S else end
    return frac_out, start_out, end_out, frac < MUTE_THR


@dataclass(frozen=True)
class ReviewRow:
    action: str  # kept | dropped | merged
    reason: str
    group: str
    slot: str
    track: str
    set_start_s: float | None = None
    set_end_s: float | None = None
    recording_id: str | None = None


@dataclass(frozen=True)
class ClipRow:
    clip: ParsedClip
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    ref_end_s: float
    recording_id: str | None
    slot_label: str
    display: str
    claimed_stem: str
    ref_source: str
    tempo_ratio: float | None
    pitch_shift_semi: int
    is_loop: bool = False
    ref_segments: tuple[RefSegment, ...] = ()
    audible_frac: float | None = None
    audible_start_s: float | None = None
    audible_end_s: float | None = None
    gain_curve: GainCurve = ()
    skip_training: bool = False
    id_source: str = ""
    claimed_variant: str = "regular"


def _mix_track(root) -> object | None:
    for track_el in root.xpath(".//LiveSet/Tracks/*"):
        if track_el.tag != "AudioTrack":
            continue
        if track_display_name(track_el) == "1-mix":
            return track_el
    return None


def _load_content_catalog(set_dir: Path) -> ContentCatalog | None:
    """Read `set_dir/content_catalog.json` into a `ContentCatalog`, or None if absent.

    Registers TWO `CatalogEntry` rows per catalog entry (one keyed on
    `content_sha256`, one on `payload_sha256` when present) so a clip can bind
    either by full-file bytes or by the tag-invariant mdat payload — both
    pointing at the same identity.

    A content hash key that maps to more than one DISTINCT identity AXIS
    TUPLE `(recording_id, stem, variant)` across the catalog is AMBIGUOUS —
    the same bytes were ingested under two different identities (duplicate
    rip / mislabeled stem / mashup constituent / wrong-version). Keying on
    bare `recording_id` alone is NOT enough: two rows can share a
    `recording_id` but disagree on `stem` (e.g. a duplicate-rip mislabeled as
    the acappella of its own regular master), and that must still abstain —
    right work, wrong stem, is a wrong label. Binding confidently to either
    would be the exact confidently-wrong-id poison this branch kills, so such
    a key is dropped entirely: a clip whose bytes hash to it falls through to
    abstain instead of last-wins-binding. This is decided per hash key — a
    row can keep a clean `content_sha256` while its `payload_sha256` (or
    vice-versa) is excluded as ambiguous.
    """
    p = set_dir / "content_catalog.json"
    if not p.is_file():
        return None
    payload = json.loads(p.read_text())
    rows = payload.get("entries") or []

    # Pass 1: collect every (recording_id, stem, variant) axis tuple seen
    # under each hash key, across both key types, so an ambiguous key can be
    # identified before any CatalogEntry is emitted. Both stem and variant
    # default to "regular" here (matching Pass 2 below) so the same missing
    # field can't read as two different axis values across the two passes.
    axes_by_key: dict[str, set[tuple[str | None, str, str]]] = {}
    for e in rows:
        axis = (
            e.get("recording_id"),
            str(e.get("stem") or "regular"),
            str(e.get("variant") or "regular"),
        )
        for key in (e.get("content_sha256"), e.get("payload_sha256")):
            if key:
                axes_by_key.setdefault(str(key), set()).add(axis)
    ambiguous_keys = {key for key, axes in axes_by_key.items() if len(axes) > 1}

    # Pass 2: emit CatalogEntry rows, skipping any key found ambiguous above.
    entries: list[CatalogEntry] = []
    for e in rows:
        rid = e.get("recording_id")
        taid = str(e.get("track_audio_id") or "")
        stem = str(e.get("stem") or "regular")
        variant = str(e.get("variant") or "regular")
        for key in (e.get("content_sha256"), e.get("payload_sha256")):
            if key and str(key) not in ambiguous_keys:
                entries.append(
                    CatalogEntry(
                        track_audio_id=taid,
                        recording_id=rid,
                        stem=stem,
                        variant=variant,
                        head_hash=str(key),
                    )
                )
    return ContentCatalog.from_entries(entries)


def _content_bind(
    clip: ParsedClip, catalog: ContentCatalog | None
) -> tuple[str | None, str | None, str | None, str]:
    """(recording_id, stem, variant, id_source): bind a clip by content, or abstain.

    Two passes: `head_hash_of=file_sha256` (the full-file hash — matches a
    pristine downloaded master); on a miss, for an mp4-family path
    (`.m4a`/`.mp4`/`.m4b`), a second pass with `head_hash_of=mdat_sha256` (the
    tag-invariant mdat payload — matches a locally iTunes-tagged copy of the
    same master). A missing file or any OSError abstains; this never raises.

    A content bind resolves a COMPLETE axis point — `stem` and `variant` come
    off the matched `ClipIdentity`, not derived out-of-band from the path.
    """
    if catalog is None:
        return None, None, None, "abstain"

    def _safe(hasher):
        def _inner(path: str) -> str | None:
            try:
                return hasher(path)
            except OSError:
                return None

        return _inner

    r = resolve_clip_identity(clip, catalog, head_hash_of=_safe(file_sha256))
    if not r.is_ok() and clip.path.lower().endswith((".m4a", ".mp4", ".m4b")):
        r = resolve_clip_identity(clip, catalog, head_hash_of=_safe(mdat_sha256))
    if r.is_ok():
        return r.value.recording_id, r.value.stem, r.value.variant, "content"
    return None, None, None, "abstain"


def _clip_row(
    clip: ParsedClip,
    mapper: ArrangementMapper,
    manifest,
    catalog: ContentCatalog | None = None,
) -> ClipRow | None:
    set_start = mapper.arr_to_set_sec(clip.arr_start)
    set_end = mapper.arr_to_set_sec(clip.arr_end)
    if set_start is None or set_end is None:
        return None
    _rid_unused, slot_label, display, claimed_stem = resolve_identity(clip, manifest)
    recording_id, bind_stem, bind_variant, id_source = _content_bind(clip, catalog)
    if id_source == "content":
        # A content bind resolved a specific track_audio row: its stem/variant
        # are the SOUND identity and must win over the weaker path/manifest
        # guess (resolve_identity above) — right work, wrong stem is a wrong
        # label. On abstain, keep the path/manifest claimed_stem as today and
        # default claimed_variant to "regular" (no sound bind to read it from).
        claimed_stem = bind_stem or claimed_stem
        claimed_variant = bind_variant or "regular"
    else:
        claimed_variant = "regular"
    _, ref_source = classify_path(clip.path)
    ref_start = clip.ref_start_s()
    ref_end = clip.ref_end_s()
    set_span = set_end - set_start
    ref_span = ref_end - ref_start
    # The real fader ride over this clip, in SET seconds: every volume
    # breakpoint mapped through the warp. audible_frac/start/end are then
    # derived from this one curve (no independent computation to drift).
    curve: list[tuple[float, float]] = []
    for arr_b, gain in clip_gain_breakpoints(
        clip.vol_points, clip.arr_start, clip.arr_end
    ):
        sec = mapper.arr_to_set_sec(arr_b)
        if sec is not None:
            curve.append((round(sec, 3), round(gain, 4)))
    gain_curve = _merge_curves(tuple(curve))  # sort + dedup coincident points
    aud_frac_out, aud_start_out, aud_end_out, skip = _audible_fields(
        gain_curve, set_start, set_end
    )
    return ClipRow(
        clip=clip,
        set_start_s=set_start,
        set_end_s=set_end,
        ref_start_s=ref_start,
        ref_end_s=ref_end,
        recording_id=recording_id,
        slot_label=slot_label,
        display=display,
        claimed_stem=claimed_stem,
        claimed_variant=claimed_variant,
        ref_source=ref_source,
        tempo_ratio=tempo_ratio(set_span, ref_span),
        pitch_shift_semi=clip.pitch_coarse,
        audible_frac=aud_frac_out,
        audible_start_s=aud_start_out,
        audible_end_s=aud_end_out,
        gain_curve=gain_curve,
        skip_training=skip,
        id_source=id_source,
    )


def _drop_parking(
    rows: list[ClipRow], mix_end_s: float
) -> tuple[list[ClipRow], list[ReviewRow]]:
    kept: list[ClipRow] = []
    review: list[ReviewRow] = []
    cutoff = mix_end_s + PARKING_TOL_S
    for row in rows:
        if row.set_start_s > cutoff:
            review.append(
                ReviewRow(
                    action="dropped",
                    reason=f"parking-lot (set_start_s>{cutoff:.1f})",
                    group=row.clip.group_name,
                    slot=row.slot_label,
                    track=row.display,
                    set_start_s=row.set_start_s,
                    set_end_s=row.set_end_s,
                    recording_id=row.recording_id,
                )
            )
        else:
            kept.append(row)
    return kept, review


def _merge_slivers(rows: list[ClipRow]) -> tuple[list[ClipRow], list[ReviewRow]]:
    if not rows:
        return rows, []
    rows = sorted(rows, key=lambda r: (r.clip.path, r.set_start_s))
    kept: list[ClipRow] = []
    review: list[ReviewRow] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        span = row.set_end_s - row.set_start_s
        prev = kept[-1] if kept else None
        gap_s = row.set_start_s - prev.set_end_s if prev is not None else float("inf")
        ref_gap_s = (
            row.ref_start_s - prev.ref_end_s if prev is not None else float("inf")
        )
        same_identity = (
            prev is not None
            and prev.clip.path == row.clip.path
            and prev.recording_id == row.recording_id
            and prev.claimed_stem == row.claimed_stem
            and prev.clip.is_warped == row.clip.is_warped
        )
        # A sliver is a split-off tail of the preceding clip, not merely a
        # short later occurrence of the same file.  The old path-only test
        # swallowed reprises separated by 12–51 seconds (and even parking
        # clips thousands of seconds later), corrupting the GT trajectory.
        contiguous = (
            -SLIVER_ADJACENCY_S <= gap_s <= SLIVER_ADJACENCY_S
            and abs(ref_gap_s) <= SLIVER_REF_CONTINUITY_S
        )
        if 0 < span <= SLIVER_MAX_S and same_identity and contiguous:
            assert prev is not None
            prev = kept[-1]
            m_end = max(prev.set_end_s, row.set_end_s)
            m_curve = _merge_curves(prev.gain_curve, row.gain_curve)
            frac, a0, a1, skip = _audible_fields(m_curve, prev.set_start_s, m_end)
            merged = replace(
                prev,
                set_end_s=m_end,
                ref_end_s=max(prev.ref_end_s, row.ref_end_s),
                tempo_ratio=tempo_ratio(
                    m_end - prev.set_start_s,
                    max(prev.ref_end_s, row.ref_end_s) - prev.ref_start_s,
                ),
                audible_frac=frac,
                audible_start_s=a0,
                audible_end_s=a1,
                gain_curve=m_curve,
                skip_training=skip,
            )
            kept[-1] = merged
            review.append(
                ReviewRow(
                    action="merged",
                    reason=(
                        f"sliver ({span:.2f}s, gap={gap_s:.3f}s, "
                        f"ref_gap={ref_gap_s:.3f}s) into adjacent neighbor"
                    ),
                    group=row.clip.group_name,
                    slot=row.slot_label,
                    track=row.display,
                    set_start_s=row.set_start_s,
                    set_end_s=row.set_end_s,
                    recording_id=row.recording_id,
                )
            )
            i += 1
            continue
        kept.append(row)
        i += 1
    return kept, review


def _drop_micro_slivers(rows: list[ClipRow]) -> tuple[list[ClipRow], list[ReviewRow]]:
    kept: list[ClipRow] = []
    review: list[ReviewRow] = []
    for row in rows:
        span = row.set_end_s - row.set_start_s
        if span < MICRO_SLIVER_DROP_S:
            review.append(
                ReviewRow(
                    action="dropped",
                    reason=f"micro-sliver ({span:.4f}s < {MICRO_SLIVER_DROP_S}s)",
                    group=row.clip.group_name,
                    slot=row.slot_label,
                    track=row.display,
                    set_start_s=row.set_start_s,
                    set_end_s=row.set_end_s,
                    recording_id=row.recording_id,
                )
            )
        else:
            kept.append(row)
    return kept, review


def _detect_loops(rows: list[ClipRow]) -> list[ClipRow]:
    """Group same-path clips into loop rows when they overlap in the mix."""
    if not rows:
        return rows
    by_key: dict[tuple[str, str, str], list[ClipRow]] = {}
    for row in rows:
        key = (row.clip.path, row.slot_label, row.claimed_stem)
        by_key.setdefault(key, []).append(row)
    out: list[ClipRow] = []
    for key_rows in by_key.values():
        key_rows.sort(key=lambda r: (r.clip.arr_start, r.set_start_s))
        if len(key_rows) == 1:
            out.append(key_rows[0])
            continue
        slot_base = key_rows[0].slot_label.split("w", 1)[0]
        if slot_base in BLINK_182_SLOTS:
            out.extend(key_rows)
            continue
        segments: list[RefSegment] = []
        for row in key_rows:
            segments.append(
                RefSegment(
                    ref_start_s=row.ref_start_s,
                    ref_end_s=row.ref_end_s,
                    mix_start_s=row.set_start_s,
                    mix_end_s=row.set_end_s,
                    tempo_ratio=tempo_ratio(
                        row.set_end_s - row.set_start_s,
                        row.ref_end_s - row.ref_start_s,
                    ),
                )
            )
        # LOOP = a bit-identical ref segment re-triggered BACK-TO-BACK more than
        # once (>=2 identical, temporally-ADJACENT plays — the repeat starts
        # ~where it ended; Avicii ref 153-157 at mix 79/83/87, MJ "Just Beat It").
        # NOT a loop: distinct sections (SPLIT/CUT, Emily); or the SAME section
        # replayed far apart with other content between (reprise/callback, Beach
        # Boys "Wouldn't It Be Nice" ending at mix 851 then 882, ~30s gap).
        # Adjacency — not occurrence count — is the discriminator.
        _by_mix = sorted(segments, key=lambda s: s.mix_start_s)
        looped = False
        for a, b in zip(_by_mix, _by_mix[1:]):
            # tolerance, not exact: warp jitter gives the same loop iteration
            # ref_end 157.1 vs 157.0 — rounding would split them.
            same = (
                abs(a.ref_start_s - b.ref_start_s) < 1.5
                and abs(a.ref_end_s - b.ref_end_s) < 1.5
            )
            dur = a.ref_end_s - a.ref_start_s
            adjacent = abs((b.mix_start_s - a.mix_start_s) - dur) < max(2.0, 0.4 * dur)
            if same and adjacent:  # one back-to-back identical repeat = a loop
                looped = True
                break
        first = key_rows[0]
        set_span = sum(max(0.0, row.set_end_s - row.set_start_s) for row in key_rows)
        # tempo_ratio is the PLAYBACK SPEED: sum of the segment ref-durations
        # actually played, NOT the outer ref envelope. The envelope counts the
        # ref region the DJ JUMPED OVER between non-contiguous segments — that
        # inflated Emily's instrumental (slot 003) to 2.69x when its 3 segments
        # each played at 1.0x (65.3s song played over 65.3s of mix).
        ref_span = sum(max(0.0, row.ref_end_s - row.ref_start_s) for row in key_rows)
        # ref envelope = MIN start / MAX end over all segments, not
        # first-start/last-end: a loop can jump BACKWARD in the song (Avicii
        # Fade slot 004 loops ref 153-157s x3 then drops back to ref 39-65s),
        # so last-in-mix.ref_end (65s) < first-in-mix.ref_start (153s) gave a
        # negative ref span. The segments carry the real (non-monotonic) path.
        ref_lo = min(row.ref_start_s for row in key_rows)
        ref_hi = max(row.ref_end_s for row in key_rows)
        m_start = min(row.set_start_s for row in key_rows)
        m_end = max(row.set_end_s for row in key_rows)
        m_curve = _merge_curves(*(row.gain_curve for row in key_rows))
        frac, a0, a1, skip = _audible_fields(m_curve, m_start, m_end)
        out.append(
            replace(
                first,
                set_start_s=m_start,
                set_end_s=m_end,
                ref_start_s=ref_lo,
                ref_end_s=ref_hi,
                tempo_ratio=tempo_ratio(set_span, ref_span),
                is_loop=looped,
                ref_segments=tuple(segments),
                audible_frac=frac,
                audible_start_s=a0,
                audible_end_s=a1,
                gain_curve=m_curve,
                skip_training=skip,
            )
        )
    return sorted(out, key=lambda r: (r.set_start_s, r.slot_label, r.claimed_stem))


import re as _re

# A clip whose audio IS the set's OWN mix / mix-instrumental is the human's
# UNALIGNABLE marker: too hard to align OR the source doesn't exist anywhere
# (e.g. Lux Omega). Not a song placement — a positive abstain LABEL.
# NOTE: an imported `instrumental-N.flac` is a REAL instrumental the human
# dragged in (e.g. Mako - Smoke Filled Room at lane 202), NOT a placeholder —
# its identity just isn't encoded in the generic filename (needs a manual map).
_PLACEHOLDER_RE = _re.compile(
    r"^(mix\.(m4a|flac|wav)|mix_instrumental\.(m4a|flac|wav))$"
)


def _placeholder_note(path: str, group: str) -> str | None:
    fname = Path(path).name.lower()
    if not _PLACEHOLDER_RE.match(fname):
        return None
    if fname.startswith("mix_instrumental"):
        return f"mix_instrumental substituted as host — original unavailable ({group})"
    return f"mix self-reference — too difficult to align ({group})"


def _to_gt_track(row: ClipRow) -> GroundTruthTrack:
    note = _placeholder_note(row.clip.path, row.clip.group_name)
    return GroundTruthTrack(
        label=row.display,
        track_id=row.recording_id,
        claimed_stem=row.claimed_stem,
        claimed_variant=row.claimed_variant,
        set_start_s=row.set_start_s,
        set_end_s=row.set_end_s,
        ref_start_s=row.ref_start_s,
        ref_end_s=row.ref_end_s,
        slot_label=row.slot_label,
        ref_source=row.ref_source,
        tempo_ratio=row.tempo_ratio,
        pitch_shift_semi=row.pitch_shift_semi,
        is_loop=row.is_loop,
        ref_segments=row.ref_segments,
        audible_frac=row.audible_frac,
        audible_start_s=row.audible_start_s,
        audible_end_s=row.audible_end_s,
        gain_curve=row.gain_curve,
        skip_training=row.skip_training,
        unalignable=note is not None,
        source_note=note,
        id_source=row.id_source,
    )


def collect_kept_clip_rows(
    als_path: Path,
    set_dir: Path,
    *,
    include_all: bool = False,
) -> tuple[str, list[ClipRow], list[ReviewRow]]:
    """Run the ALS export pipeline and return hygiene-passed clip rows."""
    manifest_path = set_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = build_manifest_index(manifest_path)
    manifest_json = __import__("json").loads(manifest_path.read_text())
    set_id = str(manifest_json.get("set_id") or "").strip()
    if not set_id:
        raise ValueError("manifest.json missing set_id")
    catalog = _load_content_catalog(set_dir)
    mix_duration_s = float(manifest_json.get("mix_duration_s") or 0.0)

    root = load_als_xml(als_path)
    mix_track = _mix_track(root)
    if mix_track is None:
        raise ValueError("no 1-mix track in .als")
    clips = parse_layer_clips(root)
    label_arr_max = max((c.arr_end for c in clips), default=0.0)
    mapper = select_arrangement_mapper(
        root, mix_track, mix_duration_s=mix_duration_s, label_arr_max=label_arr_max
    )

    raw_rows: list[ClipRow] = []
    review: list[ReviewRow] = []
    for clip in clips:
        if clip.silence_reason:
            # Deactivated track / disabled clip / fader at 0 — silent, so not
            # ground truth even though Live keeps the clip in the arrangement.
            review.append(
                ReviewRow(
                    action="dropped",
                    reason=clip.silence_reason,
                    group=clip.group_name,
                    slot=slot_from_path(clip.path) or "",
                    track=clip.track_name,
                )
            )
            continue
        for part in split_clip_at_mix_span_edges(clip, mapper):
            row = _clip_row(part, mapper, manifest, catalog)
            if row is None:
                review.append(
                    ReviewRow(
                        action="dropped",
                        reason="outside mix warp span",
                        group=part.group_name,
                        slot=slot_from_path(part.path) or "",
                        track=part.track_name,
                    )
                )
                continue
            if row.set_end_s <= row.set_start_s:
                review.append(
                    ReviewRow(
                        action="dropped",
                        reason="non-positive set span after mix-span split",
                        group=part.group_name,
                        slot=row.slot_label or "",
                        track=row.display,
                        set_start_s=row.set_start_s,
                        set_end_s=row.set_end_s,
                        recording_id=row.recording_id,
                    )
                )
                continue
            raw_rows.append(row)

    if include_all:
        for row in raw_rows:
            review.append(
                ReviewRow(
                    action="kept",
                    reason="include-all",
                    group=row.clip.group_name,
                    slot=row.slot_label,
                    track=row.display,
                    set_start_s=row.set_start_s,
                    set_end_s=row.set_end_s,
                    recording_id=row.recording_id,
                )
            )
        return set_id, sorted(raw_rows, key=lambda r: r.set_start_s), review

    rows, rev = _drop_parking(raw_rows, mix_duration_s)
    review.extend(rev)
    rows, rev = _merge_slivers(rows)
    review.extend(rev)
    rows, rev = _drop_micro_slivers(rows)
    review.extend(rev)
    rows = _detect_loops(rows)

    for row in rows:
        review.append(
            ReviewRow(
                action="kept",
                reason="hygiene pass",
                group=row.clip.group_name,
                slot=row.slot_label,
                track=row.display,
                set_start_s=row.set_start_s,
                set_end_s=row.set_end_s,
                recording_id=row.recording_id,
            )
        )

    return set_id, rows, review


def export_gt(
    als_path: Path,
    set_dir: Path,
    *,
    include_all: bool = False,
) -> tuple[GroundTruthSet, list[ReviewRow]]:
    set_id, rows, review = collect_kept_clip_rows(
        als_path,
        set_dir,
        include_all=include_all,
    )
    return GroundTruthSet(
        set_id=set_id,
        tracks=tuple(_to_gt_track(r) for r in rows),
    ), review


def slot_from_path(path: str) -> str | None:
    from labeling.als import slot_from_path as _slot

    return _slot(path)


def print_review(review: list[ReviewRow]) -> None:
    kept = sum(1 for r in review if r.action == "kept")
    dropped = sum(1 for r in review if r.action == "dropped")
    merged = sum(1 for r in review if r.action == "merged")
    unresolved = sorted(
        {r.slot for r in review if r.action == "kept" and not r.recording_id and r.slot}
    )
    print(
        f"review: kept={kept} dropped={dropped} merged={merged} unresolved_slots={len(unresolved)}"
    )
    if unresolved:
        print(
            "  unresolved:",
            ", ".join(unresolved[:20]),
            ("..." if len(unresolved) > 20 else ""),
        )
    print(f"{'action':8} {'slot':8} {'set_span':22} {'recording':12} reason")
    for row in review[:40]:
        span = ""
        if row.set_start_s is not None and row.set_end_s is not None:
            span = f"{row.set_start_s:.1f}-{row.set_end_s:.1f}s"
        print(
            f"{row.action:8} {row.slot or '-':8} {span:22} "
            f"{row.recording_id or 'NULL':12} {row.reason}"
        )
    if len(review) > 40:
        print(f"  ... +{len(review) - 40} more")


def _reader_drop_count(root) -> int:
    """Non-mix AudioClips in the tree minus those ``parse_layer_clips`` returned.

    Mirrors the reader's track filter (skips ``1-mix``/``2-mix``); any positive
    delta is clips the *total* reader dropped silently — no resolvable path,
    missing CurrentStart/CurrentEnd/Loop, or malformed numerics. Each dropped
    clip is a GT label that would vanish without a trace.
    """
    raw = 0
    for track_el in root.xpath(".//LiveSet/Tracks/*"):
        if track_el.tag != "AudioTrack":
            continue
        nm = track_display_name(track_el)
        if nm.startswith("1-mix") or nm.startswith("2-mix"):
            continue
        raw += len(track_el.xpath(".//AudioClip"))
    return raw - len(parse_layer_clips(root))


# Minimum fraction of exported GT tracks that must resolve a manifest
# recording_id. resolve_identity fills track_id ONLY on an exact manifest match;
# a stale .als (clip file-refs not relinked after a slot/tag rename) resolves ~0
# ids and produces a fixture that joins to nothing downstream (BB12 re-export
# 2026-07-12 resolved 1/163 and was written silently). A healthy export resolves
# ~99%; a broken one ~0% — 0.5 cleanly separates them.
ID_COVERAGE_MIN = 0.5


def id_coverage(tracks) -> tuple[int, int, float]:
    """(content-bound, total, fraction) of GT tracks whose id came from content."""
    total = len(tracks)
    resolved = sum(1 for t in tracks if getattr(t, "id_source", "") == "content")
    return resolved, total, (resolved / total if total else 1.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--als", type=Path, default=DEFAULT_ALS)
    p.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR)
    p.add_argument(
        "--out", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--review", action="store_true", help="print review table only")
    p.add_argument("--include-all-clips", action="store_true")
    p.add_argument(
        "--allow-invalid",
        action="store_true",
        help="export even when the .als has error-severity diagnostics "
        "(clips the reader silently drops / malformed timing). Default: refuse.",
    )
    args = p.parse_args(argv)

    if not args.als.is_file():
        print(f"not found: {args.als}", file=sys.stderr)
        return 2
    if not args.set_dir.is_dir():
        print(f"not found: {args.set_dir}", file=sys.stderr)
        return 2

    # Gate: the reader (parse_layer_clips) is *total* — it silently drops clips
    # with no path / missing CurrentStart|CurrentEnd|Loop / malformed numerics,
    # any one of which is a silently-missing GT label. Two blocking signals:
    #   (1) validate error-severity diagnostics, and
    #   (2) a direct drop count — raw non-mix AudioClips vs what the reader
    #       returned. This catches drops `validate` grades below "error"
    #       (clip-incomplete is a warning) or doesn't model (missing Path).
    # Both fail-fast at the edge; --allow-invalid overrides. Warnings print but
    # do not block.
    root = load_als_xml(args.als)
    diags = validate_session(root)
    if diags:
        errs = [d for d in diags if d.severity == "error"]
        warns = [d for d in diags if d.severity != "error"]
        print(
            f"validate: {len(errs)} error(s), {len(warns)} warning(s) in {args.als.name}",
            file=sys.stderr,
        )
        for d in diags:
            print(f"  {d}", file=sys.stderr)

    dropped = _reader_drop_count(root)
    if dropped:
        print(
            f"reader silently dropped {dropped} non-mix clip(s) "
            "(no path / missing timing / malformed numerics) — each is a lost GT label",
            file=sys.stderr,
        )

    if (has_errors(diags) or dropped) and not args.allow_invalid:
        print(
            "REFUSING to export: the .als has clips the reader drops silently "
            "(bad/missing labels). Fix the .als, or pass --allow-invalid to override.",
            file=sys.stderr,
        )
        return 1

    try:
        gt, review = export_gt(
            args.als, args.set_dir, include_all=args.include_all_clips
        )
    except (OSError, ValueError) as e:
        print(f"export failed: {e}", file=sys.stderr)
        return 1

    print_review(review)
    if args.review:
        return 0

    # Gate: identity coverage. A stale .als whose clip file-refs no longer match
    # the manifest (e.g. not relinked after a slot/tag rename) resolves almost no
    # recording_ids; the resulting fixture joins to nothing in the scorer and
    # would silently corrupt canonical set_ground_truth on write-back. Refuse.
    resolved, total, coverage = id_coverage(gt.tracks)
    if total and coverage < ID_COVERAGE_MIN and not args.allow_invalid:
        abstained = [
            t.slot_label or t.label
            for t in gt.tracks
            if getattr(t, "id_source", "") != "content"
        ]
        print(
            f"REFUSING to export: only {resolved}/{total} tracks ({coverage:.0%}) "
            f"content-bound (min {ID_COVERAGE_MIN:.0%}). Abstained: "
            f"{', '.join(abstained[:30])}. Rebuild content_catalog.json (re-pull), "
            "or pass --allow-invalid to override.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    title = args.set_dir.name
    match save(gt, args.out, title=title):
        case Err(e):
            print(f"write failed: {e.detail}", file=sys.stderr)
            return 1
        case Ok(path):
            print(f"wrote {len(gt.tracks)} tracks -> {path}")
            _write_inventory_bundle(gt, path)
    return 0


def _write_inventory_bundle(gt, yaml_path: Path) -> None:
    """Sidecar JSON for alignment training (ref_source + slot labels)."""
    from collections import Counter

    bundle_path = yaml_path.with_suffix(".inventory.json")
    rows = []
    for t in gt.tracks:
        rows.append(
            {
                "label": t.slot_label or t.label,
                "track_id": t.track_id,
                "claimed_stem": t.claimed_stem,
                "ref_source": t.ref_source,
            }
        )
    payload = {
        "set_id": gt.set_id,
        "tracks": rows,
        "ref_source_counts": dict(Counter(t.ref_source for t in gt.tracks)),
    }
    bundle_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote inventory bundle -> {bundle_path}")


if __name__ == "__main__":
    sys.exit(main())

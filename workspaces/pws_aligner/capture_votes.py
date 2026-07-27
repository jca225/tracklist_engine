"""Genuine per-probe vote capture via harness probes.

This module runs the REAL harness probes (ChromaProbe, FingerprintProbe,
HubertProbe, ContinuityProbe) per span and records every genuine AlignmentResult
before any merge, producing ``out/<set_id>_probe_votes.json`` in the same schema
as ``export_votes.py`` — but with GENUINE per-probe values instead of
reconstructions from a pre-merged scalar.

The problem with ``export_votes.py`` output: every probe's ``recording_id`` was
copied from the merged identity decision (identity pinned to MERT), and every
non-abstaining confidence was flat 0.5.  That means Dawid–Skene sees no
per-probe opinion variance → degenerates to winner-take-all.

What this module produces instead:
  - Each probe's ``recording_id`` is the probe's OWN ``AlignmentResult.recording_id``
    (i.e. what that probe would have chosen — from ``ref.recording_id``, which is
    the candidate the driver presented).
  - Each probe's ``confidence`` is the probe's native [0,1] signal.
  - Each probe's ``abstain`` is ``True`` if the probe itself abstained (weak signal
    or missing cache) or raised an exception.
  - A probe that errors (missing cache, import failure, etc.) → explicit abstain
    entry with a ``stderr`` warning naming the missing artifact; NEVER crash the
    whole export; NEVER silently drop.

Usage
-----
    venvs/audio/bin/python -m workspaces.pws_aligner.capture_votes \\
        --set-id <id> [--out <path>] [--probes <csv>]

    # BB12
    venvs/audio/bin/python -m workspaces.pws_aligner.capture_votes \\
        --set-id 1fsnxchk

Output: ``workspaces/alignment_prototype/out/<set_id>_probe_votes.json``

Cache prerequisites (BB12 = 1fsnxchk)
---------------------------------------
1. ``~/aligning/1fsnxchk__*/{mix.m4a,mix_vocals.flac,mix_instrumental.flac}``
   — the set-level aligning audio pulled by ``labeling/acquire/pull_set_for_alignment.py``.
2. ``~/aligning/1fsnxchk__*/manifest.json``
   — per-track rows with ``local_path`` and ``stems`` (Demucs output paths).
3. Per-probe fingerprint cache at
   ``workspaces/alignment_prototype/.cache/fp_index/<recording_id>__<stem>.landmark``
   (or in the canonical DB ``track_fingerprints`` table) — built by
   ``scripts/backfill_track_fingerprints.py`` + ``scripts/cache_set_fingerprint_hits.py``.
4. The predicted timeline at
   ``workspaces/alignment_prototype/out/<set_id>_predicted_timeline.json``
   — written by ``infer.py``; provides span stubs (set_start/end, slot_label,
   claimed_stem, recording_id) without requiring a fresh pi SSH.

Probes captured (axis-routed per claimed_stem)
----------------------------------------------
  acappella  → hubert, continuity, chroma
  instrumental → fp, chroma, continuity
  regular    → fp, chroma

Wall-clock order of magnitude per probe (inferred from code):
  chroma      ~5–15 s / span (librosa load + CQT + matched filter)
  fp          ~2–8 s  / span (landmark hashing; fast when cache hot)
  hubert      ~20–60 s / span (HuBERT L9 inference via similarity_probe; MPS)
  continuity  ~15–45 s / span (K×chroma stacked windows)

Expected BB12 command
---------------------
    venvs/audio/bin/python -m workspaces.pws_aligner.capture_votes \\
        --set-id 1fsnxchk
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[2] / "workspaces" / "alignment_prototype" / "out"
)
_ALIGNING_ROOT = Path.home() / "aligning"

# Canonical probe name order in the output file (matches the classical driver's
# axis routing table in harness/axes.py).
_ALL_PROBE_NAMES: tuple[str, ...] = ("fp", "chroma", "hubert", "continuity")

# Stem → probe names to run (matches axes.py AXES dict).
_STEM_TO_PROBES: dict[str, tuple[str, ...]] = {
    "acappella": ("hubert", "continuity", "chroma"),
    "instrumental": ("fp", "chroma", "continuity"),
    "regular": ("fp", "chroma"),
}

# Offset frame per probe, verified EMPIRICALLY against BB12 (2026-07-14):
# chroma/continuity/hubert emit ABSOLUTE ref-time positions in offset_s
# (contract.py's "primary ref-time placement"); the fp path feeds the landmark
# DIAGONAL (ref_t − mix_t, i.e. already relative) through the same field name.
# The votes-file convention is RELATIVE (offset_s = ref_start_s − set_start_s,
# per run_phase1's schema), so absolute-frame probes are converted at capture:
#     relative = absolute − span.set_start_s
# Getting this wrong is catastrophic and self-consistent: the absolute-frame
# probes agree with EACH OTHER in the wrong frame and outvote fp (observed:
# DS crowned chroma 0.99 / floored fp 0.01; gt_measured said the opposite).
_ABSOLUTE_FRAME_PROBES: frozenset[str] = frozenset({"chroma", "continuity", "hubert"})


# ---------------------------------------------------------------------------
# Data types (same schema as export_votes.ProbeEntry / SpanVotesDoc)
# ---------------------------------------------------------------------------


def _probe_entry(
    *,
    probe: str,
    recording_id: str | None,
    offset_s: float | None,
    confidence: float,
    abstain: bool,
) -> dict:
    return {
        "probe": probe,
        "recording_id": recording_id,
        "offset_s": offset_s,
        "confidence": confidence,
        "abstain": abstain,
        "features": [],
    }


def _abstain_entry(probe: str) -> dict:
    return _probe_entry(
        probe=probe,
        recording_id=None,
        offset_s=None,
        confidence=0.0,
        abstain=True,
    )


# ---------------------------------------------------------------------------
# Probe factory (lazy-import to avoid crashing when a dep is absent)
# ---------------------------------------------------------------------------


def _build_probes(probe_names: tuple[str, ...]) -> dict[str, object]:
    """Instantiate each named probe; return {name: probe_instance | None}.

    A ``None`` value means the probe could not be imported (missing dep) — the
    caller should emit an abstain with a stderr warning instead of crashing.
    """
    # Early import-error guard: if contract.py itself is broken, fail loud here
    # rather than silently inside the per-probe try/except below.
    from workspaces.alignment_prototype.harness.contract import Probe  # noqa: F401

    instances: dict[str, object] = {}
    for name in probe_names:
        if name in instances:
            continue
        try:
            if name == "fp":
                from workspaces.alignment_prototype.harness.fingerprint_probe import (
                    FingerprintProbe,
                )

                instances[name] = FingerprintProbe()
            elif name == "chroma":
                from workspaces.alignment_prototype.harness.chroma_probe import (
                    ChromaProbe,
                )

                instances[name] = ChromaProbe()
            elif name == "hubert":
                from workspaces.alignment_prototype.harness.hubert_probe import (
                    HubertProbe,
                )

                instances[name] = HubertProbe()
            elif name == "continuity":
                from workspaces.alignment_prototype.harness.continuity_probe import (
                    ContinuityProbe,
                )

                instances[name] = ContinuityProbe()
            else:
                print(
                    f"WARNING: unknown probe name {name!r}, skipping",
                    file=sys.stderr,
                )
                instances[name] = None
        except Exception as exc:
            print(
                f"WARNING: probe {name!r} failed to import — will abstain: {exc}",
                file=sys.stderr,
            )
            instances[name] = None
    return instances


# ---------------------------------------------------------------------------
# Audio path resolution (from manifest.json in the aligning dir)
# ---------------------------------------------------------------------------


def _find_aligning_dir(set_id: str) -> Path | None:
    hits = sorted(_ALIGNING_ROOT.glob(f"{set_id}__*"))
    return hits[0] if hits else None


def _load_manifest_by_rid(set_dir: Path, set_id: str) -> dict[str, dict]:
    """Manifest rows keyed by recording_id AND track_id.

    Mirrors infer.py's ``_manifest_by_tid``: bridges the scrape (tlp*) namespace
    to canonical recording_id via labeling/fixtures/id_maps/<set>.json.
    """
    manifest_path = set_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    rows = json.loads(manifest_path.read_text())["tracks"]
    by_rid: dict[str, dict] = {}
    for row in rows:
        # Key by track_id (primary) and recording_id (secondary, setdefault so
        # track_id entries win on collision).
        tid = row.get("track_id", "")
        if tid:
            by_rid[tid] = row
        rid = row.get("recording_id", "")
        if rid:
            by_rid.setdefault(rid, row)
    # apply id_maps bridge
    map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    if map_path.is_file():
        for tlp, rec in json.loads(map_path.read_text()).items():
            if tlp in by_rid:
                by_rid.setdefault(rec, by_rid[tlp])
    return by_rid


def _ref_audio_path(row: dict, claimed_stem: str) -> Path | None:
    """Best reference audio path for a manifest row given the claimed stem.

    - acappella: prefer separated vocals stem, else the acappella audio itself.
    - instrumental: prefer separated instrumental stem.
    - regular: use the full-track audio.
    """
    if not row:
        return None
    if claimed_stem == "acappella":
        vpath = (row.get("stems") or {}).get("vocals")
        if vpath:
            return Path(vpath)
        if row.get("stem") == "acappella":
            lp = row.get("local_path")
            return Path(lp) if lp else None
        return None
    if claimed_stem == "instrumental":
        ipath = (row.get("stems") or {}).get("instrumental")
        if ipath:
            return Path(ipath)
    lp = row.get("local_path")
    return Path(lp) if lp else None


def _mix_audio_path(set_dir: Path, claimed_stem: str) -> Path | None:
    """Mix-side audio for the given claimed_stem, mirroring axes.py routing."""
    if claimed_stem == "acappella":
        p = set_dir / "mix_vocals.flac"
        return p if p.is_file() else None
    elif claimed_stem == "instrumental":
        p = set_dir / "mix_instrumental.flac"
        return p if p.is_file() else None
    else:
        # regular / fallback
        for name in ("mix.m4a", "mix.flac", "mix.wav", "mix.mp3"):
            candidate = set_dir / name
            if candidate.is_file():
                return candidate
        return None


# ---------------------------------------------------------------------------
# Per-span probe run (the core of this module)
# ---------------------------------------------------------------------------


def _run_probe_safe(
    probe_name: str,
    probe_instance: object | None,
    mix_ctx,
    ref_ctx,
    candidate_pool,
    *,
    span_set_start_s: float = 0.0,
    debug: bool = False,
) -> dict:
    """Run one probe and return a ProbeEntry dict; absorb errors as abstain.

    If ``probe_instance`` is None (failed import) or if the probe raises, emits
    an explicit abstain entry with a stderr warning naming the missing artifact.

    Parameters
    ----------
    debug:
        When True, print full tracebacks for probe errors to stderr.
        Passed through from ``capture_votes(debug=True)``; never reads
        ``sys.argv`` so programmatic callers can enable tracebacks cleanly.
    """
    from workspaces.alignment_prototype.harness.contract import AlignmentResult

    if probe_instance is None:
        print(
            f"WARNING: probe {probe_name!r} not available (import failed) — abstaining",
            file=sys.stderr,
        )
        return _abstain_entry(probe_name)
    try:
        result: AlignmentResult = probe_instance.run(mix_ctx, ref_ctx, candidate_pool)
        # Normalize to the votes-file RELATIVE frame (see _ABSOLUTE_FRAME_PROBES).
        offset = result.offset_s
        if not result.abstain and probe_name in _ABSOLUTE_FRAME_PROBES:
            assert result.offset_s is not None
            offset = result.offset_s - span_set_start_s
        return _probe_entry(
            probe=probe_name,
            recording_id=result.recording_id,
            offset_s=offset,
            confidence=result.confidence,
            abstain=result.abstain,
        )
    except FileNotFoundError as exc:
        print(
            f"WARNING: probe {probe_name!r} raised FileNotFoundError "
            f"(missing cache/audio: {exc}) — abstaining",
            file=sys.stderr,
        )
        return _abstain_entry(probe_name)
    except Exception as exc:
        # Catch-all: never crash the whole export for one probe
        print(
            f"WARNING: probe {probe_name!r} raised {type(exc).__name__}: {exc} — abstaining",
            file=sys.stderr,
        )
        if debug:
            traceback.print_exc(file=sys.stderr)
        return _abstain_entry(probe_name)


def _capture_span(
    span_doc: dict,
    *,
    probe_names: tuple[str, ...],
    probe_instances: dict[str, object | None],
    set_dir: Path | None,
    manifest_by_rid: dict[str, dict],
    debug: bool = False,
) -> dict:
    """Capture genuine probe votes for one span.

    Returns a span-level dict in the probe_votes.json schema.
    """
    from workspaces.alignment_prototype.harness.contract import (
        AlignmentResult,
        CandidatePool,
        MixContext,
        RefContext,
    )
    from workspaces.alignment_prototype.records import SlotCandidate

    slot_label = str(span_doc.get("slot_label", ""))
    recording_id = span_doc.get("recording_id") or None
    claimed_stem = str(span_doc.get("claimed_stem", "regular"))
    set_start_s = float(span_doc.get("set_start_s", 0.0))
    set_end_s = float(span_doc.get("set_end_s", 0.0))
    ref_start_s = float(span_doc.get("ref_start_s", 0.0))
    ref_end_s_raw = span_doc.get("ref_end_s")
    ref_end_s = float(ref_end_s_raw) if ref_end_s_raw is not None else None
    confidence = float(span_doc.get("confidence", 0.0))
    name = str(span_doc.get("name", ""))

    # Resolve mix audio path
    mix_audio: Path | None = None
    if set_dir is not None:
        mix_audio = _mix_audio_path(set_dir, claimed_stem)
        if mix_audio is None:
            # Fallback to full mix
            mix_audio = _mix_audio_path(set_dir, "regular")

    # Resolve ref audio path
    ref_row = manifest_by_rid.get(recording_id or "")
    ref_audio: Path | None = None
    if ref_row:
        ref_audio = _ref_audio_path(ref_row, claimed_stem)

    # Determine which probes to run for this stem
    stem_probes = _STEM_TO_PROBES.get(claimed_stem, _STEM_TO_PROBES["regular"])
    # Filter to only the probes we have instances for
    active_probes = tuple(p for p in probe_names if p in stem_probes)

    probe_entries: list[dict] = []

    if mix_audio is None or not mix_audio.is_file():
        # No mix audio → all probes abstain with a warning
        missing_what = f"mix audio for {claimed_stem!r} stem under {set_dir}"
        print(
            f"WARNING [{slot_label}]: {missing_what} missing — all probes abstain",
            file=sys.stderr,
        )
        for pname in active_probes:
            probe_entries.append(_abstain_entry(pname))
    elif ref_audio is None or not ref_audio.is_file():
        # No ref audio → all probes abstain with a warning
        missing_what = (
            f"ref audio for recording_id={recording_id!r} stem={claimed_stem!r}"
        )
        print(
            f"WARNING [{slot_label}]: {missing_what} missing — all probes abstain",
            file=sys.stderr,
        )
        for pname in active_probes:
            probe_entries.append(_abstain_entry(pname))
    else:
        # Build contexts
        mix_ctx = MixContext(
            audio_path=mix_audio,
            set_id=str(span_doc.get("set_id", "")),
            span_start_s=set_start_s,
            span_end_s=set_end_s,
        )
        ref_ctx = RefContext(
            recording_id=recording_id or "",
            audio_path=ref_audio,
            stem=claimed_stem,
        )
        candidate = SlotCandidate(
            recording_id=recording_id or "",
            claimed_stem=claimed_stem,
        )
        candidate_pool = CandidatePool(candidates=(candidate,))

        for pname in active_probes:
            entry = _run_probe_safe(
                pname,
                probe_instances.get(pname),
                mix_ctx,
                ref_ctx,
                candidate_pool,
                span_set_start_s=set_start_s,
                debug=debug,
            )
            probe_entries.append(entry)

    # Probes NOT in stem_probes for this stem are omitted (not emitted as abstains).
    # This matches the axis-routing design: chroma is not run on acappella by the
    # canonical driver when hubert is the invariant axis.
    # However, for DS we want ALL probes present; probes not routed to this stem
    # are emitted as abstains so DS has a uniform probe set across all spans.
    routed_set = set(active_probes)
    for pname in probe_names:
        if pname not in routed_set:
            probe_entries.append(_abstain_entry(pname))

    # Sort probe_entries to canonical order (probe_names order)
    order = {p: i for i, p in enumerate(probe_names)}
    probe_entries.sort(key=lambda e: order.get(e["probe"], len(probe_names)))

    return {
        "span_id": slot_label,
        "slot_label": slot_label,
        "recording_id": recording_id,
        "claimed_stem": claimed_stem,
        "set_start_s": set_start_s,
        "set_end_s": set_end_s,
        "ref_start_s": ref_start_s,
        "ref_end_s": ref_end_s,
        "confidence": confidence,
        "name": name,
        "probes": probe_entries,
    }


# ---------------------------------------------------------------------------
# File-level I/O
# ---------------------------------------------------------------------------


def _load_timeline(timeline_path: Path) -> list[dict]:
    """Load spans from a predicted_timeline.json."""
    raw = json.loads(timeline_path.read_text())
    spans = raw.get("spans")
    if spans is None:
        raise ValueError(
            f"timeline at {timeline_path} has no 'spans' key — run infer.py first"
        )
    return list(spans)


def capture_votes(
    set_id: str,
    *,
    timeline_path: Path | None = None,
    out_path: Path | None = None,
    probe_names: tuple[str, ...] = _ALL_PROBE_NAMES,
    debug: bool = False,
) -> Path:
    """End-to-end: run harness probes per span and write probe_votes.json.

    Parameters
    ----------
    set_id:
        DJ-set identifier (e.g. ``"1fsnxchk"`` for BB12).
    timeline_path:
        Path to ``<set_id>_predicted_timeline.json`` for span stubs.
        Defaults to ``workspaces/alignment_prototype/out/<set_id>_predicted_timeline.json``.
    out_path:
        Path to write the output ``<set_id>_probe_votes.json``.
        Defaults to ``workspaces/alignment_prototype/out/<set_id>_probe_votes.json``.
    probe_names:
        Probes to include in the output (in this canonical order).
        Each probe is run only when the span's axis routing includes it;
        probes not routed for a span's stem are emitted as abstains so DS
        has a uniform vector.

    Returns
    -------
    Path to the written ``<set_id>_probe_votes.json``.
    """
    out_dir = _DEFAULT_OUT_DIR
    if timeline_path is None:
        timeline_path = out_dir / f"{set_id}_predicted_timeline.json"
    if out_path is None:
        out_path = out_dir / f"{set_id}_probe_votes.json"

    if not timeline_path.exists():
        sys.exit(
            f"ERROR: predicted timeline not found: {timeline_path}\n"
            "Run infer.py first:\n"
            f"  venvs/audio/bin/python -m workspaces.alignment_prototype.infer "
            f"--set-id {set_id}\n"
        )

    spans = _load_timeline(timeline_path)
    if not spans:
        sys.exit(f"ERROR: no spans in timeline: {timeline_path}")

    # Resolve set-level resources once
    set_dir = _find_aligning_dir(set_id)
    if set_dir is None:
        print(
            f"WARNING: no aligning dir found for {set_id} under {_ALIGNING_ROOT} — "
            "all audio-dependent probes will abstain",
            file=sys.stderr,
        )
    manifest_by_rid = _load_manifest_by_rid(set_dir, set_id) if set_dir else {}

    # Tripwire: if most of the timeline's recording_ids are absent from the
    # manifest index, the run would silently degrade to all-abstain garbage —
    # warn loudly up front rather than after hours of probe compute.
    span_rids = [s.get("recording_id") for s in spans if s.get("recording_id")]
    if manifest_by_rid and span_rids:
        missing = sum(1 for r in span_rids if r not in manifest_by_rid)
        if missing / len(span_rids) > 0.5:
            print(
                f"WARNING: {missing}/{len(span_rids)} timeline recording_ids are "
                "missing from the manifest index — most probes will abstain. "
                "Check manifest.json track_id/recording_id keys and the "
                f"id_maps bridge for {set_id}.",
                file=sys.stderr,
            )

    # Collect all probe names needed across all stems
    all_needed: set[str] = set()
    for pname in probe_names:
        all_needed.add(pname)
    probe_instances = _build_probes(tuple(all_needed))

    # Capture per span
    span_docs: list[dict] = []
    for i, span in enumerate(spans):
        slot_label = span.get("slot_label", str(i))
        print(f"  span {i + 1}/{len(spans)} [{slot_label}] …", flush=True)
        doc = _capture_span(
            span,
            probe_names=probe_names,
            probe_instances=probe_instances,
            set_dir=set_dir,
            manifest_by_rid=manifest_by_rid,
            debug=debug,
        )
        span_docs.append(doc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(span_docs, indent=2))

    n_probes_total = sum(len(d["probes"]) for d in span_docs)
    n_abstain = sum(sum(1 for p in d["probes"] if p["abstain"]) for d in span_docs)
    print(
        f"wrote {out_path}  "
        f"({len(span_docs)} spans, {n_probes_total} probe entries, "
        f"{n_abstain} abstaining)"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Capture genuine per-probe AlignmentResults via harness probes "
            "into probe_votes.json for PWS phase-1 aggregation. "
            "Each probe runs against its own candidate and emits its OWN "
            "recording_id/confidence/abstain — no reconstructed scalars."
        )
    )
    p.add_argument(
        "--set-id",
        required=True,
        help="DJ-set identifier (e.g. 1fsnxchk for BB12)",
    )
    p.add_argument(
        "--timeline",
        type=Path,
        default=None,
        dest="timeline_path",
        help=(
            "path to <set_id>_predicted_timeline.json for span stubs "
            "(default: workspaces/alignment_prototype/out/<set_id>_predicted_timeline.json)"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        dest="out_path",
        help=(
            "output path for probe_votes.json "
            "(default: workspaces/alignment_prototype/out/<set_id>_probe_votes.json)"
        ),
    )
    p.add_argument(
        "--probes",
        type=str,
        default=None,
        help=(
            "comma-separated probe names to run "
            f"(default: {','.join(_ALL_PROBE_NAMES)})"
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="print full tracebacks for probe errors",
    )
    args = p.parse_args(argv)

    probe_names: tuple[str, ...] = _ALL_PROBE_NAMES
    if args.probes:
        probe_names = tuple(n.strip() for n in args.probes.split(",") if n.strip())

    capture_votes(
        args.set_id,
        timeline_path=args.timeline_path,
        out_path=args.out_path,
        probe_names=probe_names,
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Votes-export hook: predicted_timeline.json -> <set_id>_probe_votes.json.

Usage
-----
    venvs/audio/bin/python -m workspaces.pws_aligner.export_votes \\
        --set-id <id> [--timeline <path>] [--out <dir>]

What this does
--------------
``infer.py`` (alignment_prototype) writes a predicted timeline with a
``probe_proposals`` dict per span — ``{probe_name: set_start_s}`` — capturing
every channel's raw placement before ``merge()`` collapses them.  That dict is
the cheapest available per-probe evidence: no audio re-read, no heavy ML, fully
offline once the timeline exists.

This exporter lifts those per-span, per-probe scalars into the
``<set_id>_probe_votes.json`` schema that ``run_phase1`` reads.  The mapping:

  probe entry present in probe_proposals -> abstain=False, confidence=0.5
  probe entry absent                     -> abstain=True,  confidence=0.0

``offset_s`` (ref-frame offset, so ref_start_s = set_start_s + offset_s) is
approximated as ``span.ref_start_s - probe_set_start_s``.  This is exact when
the ref_start_s the probe would have produced equals the final span ref_start_s
(true for the winning probe; an approximation for the others), but gives the
PWS label model a per-probe offset hypothesis — the main benefit over re-using
the merged scalar.

``features`` are left empty (``[]``) in this path because the neuro/precision.py
sharpness proxies (margin, z, prominence) require the full matched-filter curve,
which is not stored in the predicted timeline.  The Dawid-Skene label model
tolerates empty features in Phase 1.

Alignment_prototype touched: **NO**.  The exporter is a pure reader of the
existing output artifact.

Cache prerequisites (BB12 = 1fsnxchk)
---------------------------------------
Run ``infer.py`` first — it writes the predicted timeline that this exporter
reads:

    venvs/audio/bin/python -m workspaces.alignment_prototype.infer \\
        --set-id 1fsnxchk

Then export:

    venvs/audio/bin/python -m workspaces.pws_aligner.export_votes \\
        --set-id 1fsnxchk

Output: ``workspaces/alignment_prototype/out/1fsnxchk_probe_votes.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[2] / "workspaces" / "alignment_prototype" / "out"
)

# Probe names present in infer.py's probe_proposals dict.  The order here is
# the canonical order in the votes doc's probes list (for deterministic output).
_KNOWN_PROBES: tuple[str, ...] = (
    "mert_decode",
    "fp",
    "lyrics",
    "stem_hubert",
    "instr_fp",
)

# Confidence assigned to every non-abstaining probe entry.  The true per-probe
# confidence is not stored in the predicted timeline; 0.5 is a neutral prior
# that lets Dawid-Skene weight by consistency (EM agreement), not by an
# arbitrary scalar.
_DEFAULT_CONFIDENCE: float = 0.5


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeEntry:
    """One probe's vote for one span, in the votes-file schema."""

    probe: str
    recording_id: str | None
    offset_s: float
    confidence: float
    abstain: bool
    features: tuple[float, ...]


@dataclass(frozen=True)
class SpanVotesDoc:
    """Full votes document for one span."""

    span_id: str
    slot_label: str
    recording_id: str | None
    claimed_stem: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    ref_end_s: float | None
    confidence: float
    name: str
    probes: tuple[ProbeEntry, ...]

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "slot_label": self.slot_label,
            "recording_id": self.recording_id,
            "claimed_stem": self.claimed_stem,
            "set_start_s": self.set_start_s,
            "set_end_s": self.set_end_s,
            "ref_start_s": self.ref_start_s,
            "ref_end_s": self.ref_end_s,
            "confidence": self.confidence,
            "name": self.name,
            "probes": [
                {
                    "probe": p.probe,
                    "recording_id": p.recording_id,
                    "offset_s": p.offset_s,
                    "confidence": p.confidence,
                    "abstain": p.abstain,
                    "features": list(p.features),
                }
                for p in self.probes
            ],
        }


# ---------------------------------------------------------------------------
# Conversion: predicted timeline span -> SpanVotesDoc
# ---------------------------------------------------------------------------


def _probe_entry_from_proposal(
    probe_name: str,
    proposal_set_start_s: float | None,
    *,
    span_set_start_s: float,
    span_ref_start_s: float,
    recording_id: str | None,
) -> ProbeEntry:
    """Build one ProbeEntry from a probe_proposals scalar.

    Parameters
    ----------
    probe_name:
        The probe identifier (e.g. "fp", "mert_decode").
    proposal_set_start_s:
        The probe's proposed set_start_s (from probe_proposals), or None when
        the probe abstained (was absent from the dict).
    span_set_start_s / span_ref_start_s:
        The final (merged) placement for the span — used to compute offset_s
        for the winning probe and as a fallback for others.
    recording_id:
        The MERT identity decision for this span (shared across probes in
        infer.py; the PWS label model may later correct it).
    """
    if proposal_set_start_s is None:
        return ProbeEntry(
            probe=probe_name,
            recording_id=None,
            offset_s=0.0,
            confidence=0.0,
            abstain=True,
            features=(),
        )
    # offset_s = ref_start_s - probe_set_start_s.
    # Rationale: if this probe's set_start is correct and we hold ref_start_s
    # fixed (the best available estimate), the implied ref offset changes by the
    # same delta that probe_set_start_s differs from span_set_start_s.
    offset_s = span_ref_start_s - proposal_set_start_s
    return ProbeEntry(
        probe=probe_name,
        recording_id=recording_id,
        offset_s=offset_s,
        confidence=_DEFAULT_CONFIDENCE,
        abstain=False,
        features=(),  # neuro/precision.py sharpness proxies unavailable offline
    )


def span_to_votes_doc(
    span_doc: dict,
    *,
    span_index: int,
    probe_names: tuple[str, ...] = _KNOWN_PROBES,
) -> SpanVotesDoc:
    """Convert one span dict from predicted_timeline.json to a SpanVotesDoc.

    Parameters
    ----------
    span_doc:
        A span entry from ``predicted_timeline.json`` (the ``infer.py`` output).
        Must have ``slot_label``, ``set_start_s``, ``set_end_s``, ``ref_start_s``,
        ``recording_id`` and optionally ``probe_proposals``.
    span_index:
        Zero-based ordinal position (used to generate span_id when slot_label
        is absent).
    probe_names:
        Ordered tuple of probe names to emit, in this order in the votes doc.
        Probes absent from ``probe_proposals`` are emitted as abstaining entries.
    """
    slot_label = str(span_doc.get("slot_label", span_index))
    recording_id = span_doc.get("recording_id") or None
    set_start_s = float(span_doc.get("set_start_s", 0.0))
    set_end_s = float(span_doc.get("set_end_s", 0.0))
    ref_start_s = float(span_doc.get("ref_start_s", 0.0))
    ref_end_s_raw = span_doc.get("ref_end_s")
    ref_end_s = float(ref_end_s_raw) if ref_end_s_raw is not None else None
    confidence = float(span_doc.get("confidence", 0.0))
    name = str(span_doc.get("name", ""))
    claimed_stem = str(span_doc.get("claimed_stem", "regular"))

    proposals: dict[str, float] = span_doc.get("probe_proposals") or {}

    # Emit an entry for every known probe name: present = non-abstain, absent = abstain.
    probe_entries: list[ProbeEntry] = [
        _probe_entry_from_proposal(
            probe_name=pname,
            proposal_set_start_s=proposals.get(pname),
            span_set_start_s=set_start_s,
            span_ref_start_s=ref_start_s,
            recording_id=recording_id,
        )
        for pname in probe_names
    ]
    # Also include any probe names present in proposals but NOT in probe_names
    # (forward-compatibility: new probes added to infer.py appear automatically).
    extra_probes = sorted(k for k in proposals if k not in probe_names)
    for pname in extra_probes:
        probe_entries.append(
            _probe_entry_from_proposal(
                probe_name=pname,
                proposal_set_start_s=proposals[pname],
                span_set_start_s=set_start_s,
                span_ref_start_s=ref_start_s,
                recording_id=recording_id,
            )
        )

    return SpanVotesDoc(
        span_id=slot_label,
        slot_label=slot_label,
        recording_id=recording_id,
        claimed_stem=claimed_stem,
        set_start_s=set_start_s,
        set_end_s=set_end_s,
        ref_start_s=ref_start_s,
        ref_end_s=ref_end_s,
        confidence=confidence,
        name=name,
        probes=tuple(probe_entries),
    )


# ---------------------------------------------------------------------------
# File-level I/O
# ---------------------------------------------------------------------------


def load_predicted_timeline(timeline_path: Path) -> dict:
    """Load and minimally validate a predicted_timeline.json."""
    raw = json.loads(timeline_path.read_text())
    if "spans" not in raw:
        raise ValueError(
            f"predicted_timeline at {timeline_path} has no 'spans' key — "
            "run infer.py first"
        )
    return raw


def timeline_to_votes_docs(
    timeline: dict,
    *,
    probe_names: tuple[str, ...] = _KNOWN_PROBES,
) -> list[SpanVotesDoc]:
    """Convert all spans in a predicted timeline to SpanVotesDoc objects."""
    return [
        span_to_votes_doc(span, span_index=i, probe_names=probe_names)
        for i, span in enumerate(timeline.get("spans", []))
    ]


def write_votes_file(
    docs: list[SpanVotesDoc],
    out_path: Path,
) -> Path:
    """Serialize SpanVotesDoc list to the probe_votes.json schema."""
    payload = [doc.to_dict() for doc in docs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def export_votes(
    set_id: str,
    *,
    timeline_path: Path | None = None,
    out_dir: Path | None = None,
    probe_names: tuple[str, ...] = _KNOWN_PROBES,
) -> Path:
    """End-to-end: predicted_timeline.json -> probe_votes.json.

    Parameters
    ----------
    set_id:
        DJ-set identifier (e.g. "1fsnxchk").
    timeline_path:
        Path to the predicted_timeline JSON.  Defaults to
        ``out/<set_id>_predicted_timeline.json`` under alignment_prototype.
    out_dir:
        Output directory.  Defaults to the same ``out/`` directory as the
        predicted timeline.
    probe_names:
        Ordered tuple of probe names to include in every span's probes list.

    Returns
    -------
    Path to the written ``<set_id>_probe_votes.json``.
    """
    if out_dir is None:
        out_dir = _DEFAULT_OUT_DIR
    if timeline_path is None:
        timeline_path = out_dir / f"{set_id}_predicted_timeline.json"

    if not timeline_path.exists():
        sys.exit(
            f"ERROR: predicted timeline not found: {timeline_path}\n"
            "Run infer.py first:\n"
            f"  venvs/audio/bin/python -m workspaces.alignment_prototype.infer "
            f"--set-id {set_id}\n"
        )

    timeline = load_predicted_timeline(timeline_path)
    spans = timeline.get("spans", [])
    if not spans:
        sys.exit(f"ERROR: no spans in timeline: {timeline_path}")

    docs = timeline_to_votes_docs(timeline, probe_names=probe_names)
    out_path = out_dir / f"{set_id}_probe_votes.json"
    write_votes_file(docs, out_path)

    n_probes_total = sum(len(d.probes) for d in docs)
    n_abstain = sum(sum(1 for p in d.probes if p.abstain) for d in docs)
    print(
        f"wrote {out_path}  "
        f"({len(docs)} spans, {n_probes_total} probe entries, "
        f"{n_abstain} abstaining)"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Export per-probe AlignmentResults from a predicted timeline "
            "into probe_votes.json for PWS phase-1 aggregation."
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
        help=(
            "path to <set_id>_predicted_timeline.json "
            "(default: workspaces/alignment_prototype/out/<set_id>_predicted_timeline.json)"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        dest="out_dir",
        help=(
            "output directory for probe_votes.json "
            "(default: workspaces/alignment_prototype/out/)"
        ),
    )
    args = p.parse_args(argv)

    export_votes(
        args.set_id,
        timeline_path=args.timeline,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

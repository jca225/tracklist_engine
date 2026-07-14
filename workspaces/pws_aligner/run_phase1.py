"""PWS phase-1 CLI: votes file -> aggregation -> pws_timeline.json.

Usage
-----
    venvs/audio/bin/python -m workspaces.pws_aligner.run_phase1 --set-id <id>

Design constraint (per task brief)
-----------------------------------
The existing ``infer.py`` pipeline writes only ``probe_proposals`` in the
timeline JSON — a ``dict[probe_name -> set_start_s]`` scalar per probe, **not**
full ``AlignmentResult`` objects with ``recording_id``, ``confidence``, or
``abstain``.  Running heavy inference inside ``run_phase1`` is explicitly
prohibited; and no per-probe per-span artifact with identity information exists.

Therefore ``run_phase1`` reads a **votes file** (``out/<set_id>_probe_votes.json``)
whose format is documented below.  A votes-export hook that writes this file
from the existing ``infer.py`` pipeline is the **missing piece the controller
must wire**.  That is an honest outcome for Task 5.

Votes-file schema
-----------------
A JSON array of span objects, one per span in play order::

    [
      {
        "span_id": "1",
        "slot_label": "1",
        "recording_id": "r1",          // current predicted recording_id (pass-through)
        "claimed_stem": "regular",
        "set_start_s": 10.0,
        "set_end_s": 50.0,
        "ref_start_s": 0.0,
        "ref_end_s": 40.0,
        "confidence": 0.0,             // pass-through (overwritten by PWS)
        "name": "Track 1",
        "probes": [
          {
            "probe": "fp",             // probe name (e.g. "fp", "mert", "hubert")
            "recording_id": "r1",      // null to abstain
            "offset_s": 8.0,           // ref-frame offset (set_start_s - ref_start_s)
            "confidence": 0.8,         // calibrated [0,1]
            "abstain": false,
            "features": [0.8, 1.2, 0.5]  // sharpness proxies (margin, z, prominence)
          }
        ]
      }
    ]

Output
------
``out/<set_id>_pws_timeline.json``
    Timeline JSON in the same schema as ``out/<set_id>_predicted_timeline.json``
    (the schema ``score_timeline_vs_gt`` reads).  Non-placement fields
    (slot_label, claimed_stem, set_start_s, set_end_s, name, …) are copied
    through unchanged; only ``recording_id``, ``ref_start_s``, ``confidence``,
    and a new ``abstain`` field are overwritten by the PWS placement.

``out/<set_id>_pws_probe_accuracy.json``
    Written only when ``DawidSkene`` was the chosen aggregator.  Contains
    ``{probe_name: accuracy}`` from ``DawidSkene.probe_accuracy()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.harness.contract import AlignmentResult
from workspaces.pws_aligner.decode_bridge import posterior_to_placement
from workspaces.pws_aligner.density_gate import choose_aggregator
from workspaces.pws_aligner.hypotheses import Hypothesis
from workspaces.pws_aligner.label_model import DawidSkene, MajorityVote
from workspaces.pws_aligner.votes import AbstainReason, Vote, collect_votes

# Default out/ directory relative to the alignment_prototype package (mirrors
# where infer.py writes predicted_timeline.json).
# Assumption: pws_aligner/ sits exactly two levels below repo root
# (repo_root/workspaces/pws_aligner/), so parents[2] == repo root.
_DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[2] / "workspaces" / "alignment_prototype" / "out"
)


# ---------------------------------------------------------------------------
# Votes-file loader
# ---------------------------------------------------------------------------


def _load_votes_file(votes_path: Path) -> list[dict]:
    """Load and validate the probe votes JSON.  Raises ValueError on bad shape."""
    raw = json.loads(votes_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Votes file must be a JSON array, got {type(raw).__name__}")
    return raw


def _span_votes(span_doc: dict) -> tuple[Vote, ...]:
    """Convert one span doc's probe entries to Vote objects."""
    span_id = str(span_doc["span_id"])
    probe_docs = span_doc.get("probes") or []
    results: list[tuple[AlignmentResult, tuple[float, ...], AbstainReason]] = []
    for p in probe_docs:
        abstain = bool(p.get("abstain", False))
        rid = p.get("recording_id") if not abstain else None
        result = AlignmentResult(
            recording_id=rid,
            offset_s=float(p.get("offset_s", 0.0)),
            confidence=float(p.get("confidence", 0.0)),
            abstain=abstain,
            source=str(p.get("probe", "unknown")),
        )
        features = tuple(float(x) for x in (p.get("features") or []))
        reason = AbstainReason.NONE if not abstain else AbstainReason.NO_DATA
        results.append((result, features, reason))
    return collect_votes(span_id, results)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_phase1(
    set_id: str,
    *,
    votes_path: Path | None = None,
    out_dir: Path | None = None,
    bin_s: float = 2.0,
) -> Path:
    """Run PWS aggregation for *set_id* and write the pws_timeline JSON.

    Parameters
    ----------
    set_id:
        The DJ-set identifier (e.g. ``"1fsnxchk"``).
    votes_path:
        Path to the ``<set_id>_probe_votes.json`` file.  Defaults to
        ``out/<set_id>_probe_votes.json`` under the alignment_prototype package.
    out_dir:
        Directory to write output files.  Defaults to the alignment_prototype
        ``out/`` directory.
    bin_s:
        Seconds per offset bin (must match what produced the votes file).

    Returns
    -------
    Path to the written ``<set_id>_pws_timeline.json``.
    """
    if out_dir is None:
        out_dir = _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if votes_path is None:
        votes_path = out_dir / f"{set_id}_probe_votes.json"

    if not votes_path.exists():
        sys.exit(
            f"ERROR: votes file not found: {votes_path}\n"
            "Run the votes-export hook (infer.py --export-votes) first.\n"
            "See workspaces/pws_aligner/run_phase1.py for the votes-file schema."
        )

    span_docs = _load_votes_file(votes_path)
    if not span_docs:
        sys.exit(f"ERROR: votes file is empty: {votes_path}")

    # Build per-span Vote tuples
    all_votes: list[tuple[Vote, ...]] = [_span_votes(s) for s in span_docs]

    # Choose aggregator based on label density across the set
    aggregator_name = choose_aggregator(all_votes)

    if aggregator_name == "label_model":
        model: DawidSkene | MajorityVote = DawidSkene()
    else:
        model = MajorityVote()

    # Fit on all spans (unsupervised)
    model.fit(all_votes)

    # Predict per span and decode to placement
    placements: list[dict] = []
    for span_doc, votes in zip(span_docs, all_votes):
        posterior = model.predict_proba(votes)
        placement = posterior_to_placement(
            str(span_doc["span_id"]), posterior, bin_s=bin_s
        )
        placements.append(placement)

    # Write probe accuracy when DawidSkene ran
    if isinstance(model, DawidSkene):
        acc_path = out_dir / f"{set_id}_pws_probe_accuracy.json"
        acc_path.write_text(json.dumps(model.probe_accuracy(), indent=2))
        print(f"wrote {acc_path}")

    # Build pws_timeline.json: copy span fields through, overwrite placement fields
    spans_out: list[dict] = []
    for span_doc, pl in zip(span_docs, placements):
        # Pass-through non-placement fields from the votes-file span doc.
        # Passthrough keys that score_timeline_vs_gt needs:
        #   slot_label, claimed_stem, set_start_s, set_end_s, name, confidence,
        #   ref_start_s, ref_end_s, cue_anchor_s (optional).
        span_out: dict = {
            "slot_label": span_doc.get("slot_label", span_doc["span_id"]),
            "claimed_stem": span_doc.get("claimed_stem", "regular"),
            "set_start_s": float(span_doc.get("set_start_s", 0.0)),
            "set_end_s": float(span_doc.get("set_end_s", 0.0)),
            "name": span_doc.get("name", ""),
            # PWS placement overwrites these.
            # Abstain spans use "" as sentinel (TimelineSpan.recording_id is str,
            # non-optional) so the JSON round-trips through msgspec without
            # ValidationError and scores as an identity miss.
            "recording_id": pl["recording_id"]
            if pl["recording_id"] is not None
            else "",
            # ref_start_s: offset_s is ref_frame - mix_frame; set_start_s + offset_s
            # recovers the ref start that infer.py would write.  When abstaining,
            # preserve the pass-through ref_start_s from the votes file.
            "ref_start_s": (
                span_doc.get("ref_start_s", 0.0)
                if pl["abstain"]
                else float(span_doc.get("set_start_s", 0.0)) + pl["offset_s"]
            ),
            "ref_end_s": span_doc.get("ref_end_s"),
            "confidence": pl["confidence"],
            "abstain": pl["abstain"],
            "pws_aggregator": aggregator_name,
        }
        # Optional fields
        if "cue_anchor_s" in span_doc:
            span_out["cue_anchor_s"] = span_doc["cue_anchor_s"]
        if "tempo_ratio" in span_doc:
            span_out["tempo_ratio"] = span_doc["tempo_ratio"]
        if "ref_segments" in span_doc:
            span_out["ref_segments"] = span_doc["ref_segments"]
        spans_out.append(span_out)

    timeline = {
        "set_id": set_id,
        "pws_aggregator": aggregator_name,
        "spans": spans_out,
        "skipped": [],
    }

    tl_path = out_dir / f"{set_id}_pws_timeline.json"
    tl_path.write_text(json.dumps(timeline, indent=2))
    print(f"wrote {tl_path}  ({aggregator_name}, {len(spans_out)} spans)")
    return tl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PWS phase-1: aggregate probe votes into a timeline for score_timeline_vs_gt."
    )
    p.add_argument("--set-id", required=True, help="DJ-set identifier (e.g. 1fsnxchk)")
    p.add_argument(
        "--votes",
        type=Path,
        default=None,
        help="path to <set_id>_probe_votes.json (default: out/<set_id>_probe_votes.json)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: workspaces/alignment_prototype/out/)",
    )
    p.add_argument(
        "--bin-s",
        type=float,
        default=2.0,
        help="seconds per offset bin (must match votes-export hook, default 2.0)",
    )
    args = p.parse_args(argv)

    run_phase1(
        args.set_id,
        votes_path=args.votes,
        out_dir=args.out_dir,
        bin_s=args.bin_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

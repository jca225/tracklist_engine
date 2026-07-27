"""Harvest executor — the co-training flywheel's write-side.

Turns ACCEPTed candidate→mix agreements (from ``cotrain_seam.propose_from_candidate``)
into a **harvest ledger** of pseudo-labelled (ref ↔ mix-span) training pairs.

Gated by the ACCEPT-precision certification (2026-07-18) via ``CERTIFIED_POLICY``,
a per-stem band map: keys are the certified axes, each value is the band that made
that axis poison-free (1.000 precision, 0 false-accepts) on both BB GT sets —
``regular`` at 2-channel agreement, ``instrumental`` at UNANIMOUS 3-channel (at
2-channel it was ~0.59, fooled by confusable EDM textures). acappella (untested)
is absent → never harvested. Coupling axis→band in one object means a caller
cannot enable instrumental without also getting its stricter band, so a mis-tuned
caller cannot silently harvest a poisoned axis.

Invariant (inherited from ``cotrain_seam``): ZERO autonomous canonical mutation.
This writes ONLY to the harvest-ledger JSONL (training data) and carries any
correction as a PROPOSED entry. It never touches the canonical DB or the
correction ledger. The ledger append is idempotent — a ``span_key`` already in
the file is not re-written.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pws_aligner.cotrain_seam import (
    DEFAULT_THRESHOLDS,
    Band,
    BandThresholds,
    CandidateProposal,
    MixSpan,
    ProposedCorrection,
    RefCandidate,
    RefMixScorer,
    propose_from_candidate,
)

# The certified-safe axis allowlist for the low-level record helper. Only stems
# here are harvestable by ``record_from_proposal``. regular: certified poison-free
# (2026-07-18). Conservative default; ``harvest()`` uses CERTIFIED_POLICY below.
DEFAULT_ALLOWED_STEMS: frozenset[str] = frozenset({"regular"})

# Certified per-stem banding policy (rigorous ACCEPT-precision run, 2026-07-18;
# see ACCEPT_PRECISION_RESULTS.md). Keys = the axes cleared for auto-harvest;
# each value is the band that made THAT axis poison-free (1.000 precision, 0
# false-accepts) on both BB GT sets. Coupling axis→band in one object closes a
# footgun: you cannot enable instrumental without also getting its stricter band.
#   - regular:      2-channel agreement (fp+chroma) — it only runs 2 probes, so
#                   3-channel is physically impossible; 2 was already poison-free.
#   - instrumental: UNANIMOUS 3-channel (fp+chroma+continuity). At 2-channel it
#                   was ~0.59 precision (confusable EDM textures fool 2 sensors);
#                   requiring all 3 to agree drives false-accepts to 0.
# acappella is absent (untested off-Mac) → never auto-harvested.
CERTIFIED_POLICY: dict[str, BandThresholds] = {
    "regular": BandThresholds(),
    "instrumental": BandThresholds(min_agreeing=3),
}


@dataclass(frozen=True)
class HarvestRecord:
    """One pseudo-labelled training pair harvested from an ACCEPT."""

    span_key: str
    set_id: str
    slot_label: str
    recording_id: str
    winner: str  # ref source_path or source_url (the accepted reference)
    stem: str
    consensus_offset_s: float | None
    n_agreeing: int
    mean_confidence: float
    preference_pairs: tuple[tuple[str, str], ...]
    correction: dict | None = None  # PROPOSED track_audio_correction, or None

    def to_json(self) -> dict:
        return {
            "span_key": self.span_key,
            "set_id": self.set_id,
            "slot_label": self.slot_label,
            "recording_id": self.recording_id,
            "winner": self.winner,
            "stem": self.stem,
            "consensus_offset_s": self.consensus_offset_s,
            "n_agreeing": self.n_agreeing,
            "mean_confidence": self.mean_confidence,
            "preference_pairs": [list(p) for p in self.preference_pairs],
            "correction": self.correction,
        }


def _correction_json(c: ProposedCorrection | None) -> dict | None:
    if c is None:
        return None
    return {
        "axis": c.axis,
        "old_value": c.old_value,
        "new_value": c.new_value,
        "source_url": c.source_url,
        "reason": c.reason,
        "dry_run": c.dry_run,
    }


def record_from_proposal(
    prop: CandidateProposal,
    *,
    allowed_stems: frozenset[str] = DEFAULT_ALLOWED_STEMS,
) -> HarvestRecord | None:
    """Turn one proposal into a HarvestRecord, or None if not harvestable.

    Harvested iff band is ACCEPT AND the candidate's stem is in ``allowed_stems``.
    REVIEW/ABSTAIN and disallowed-axis ACCEPTs return None.
    """
    if prop.agreement.band is not Band.ACCEPT:
        return None
    stem = prop.candidate.stem or "regular"
    if stem not in allowed_stems:
        return None

    span = prop.span
    span_key = f"{span.set_id}:{span.slot_label}@{span.set_start_s:.1f}s"
    winner = prop.candidate.source_path or prop.candidate.source_url
    pairs = (
        prop.training_signal.preference_pairs
        if prop.training_signal is not None
        else ()
    )
    return HarvestRecord(
        span_key=span_key,
        set_id=span.set_id,
        slot_label=span.slot_label,
        recording_id=prop.candidate.recording_id,
        winner=winner,
        stem=stem,
        consensus_offset_s=prop.agreement.consensus_offset_s,
        n_agreeing=prop.agreement.n_agreeing,
        mean_confidence=prop.agreement.mean_confidence,
        preference_pairs=pairs,
        correction=_correction_json(prop.correction),
    )


def harvest(
    cases: Iterable[tuple[RefCandidate, MixSpan, dict | None]],
    scorer: RefMixScorer,
    *,
    policy: dict[str, BandThresholds] = CERTIFIED_POLICY,
) -> list[HarvestRecord]:
    """Score each (candidate, span, claim_axes) case and keep certified ACCEPTs.

    A case is harvestable iff its stem is a key in ``policy`` (the certified
    allowlist) AND it banded ACCEPT under THAT stem's certified thresholds. The
    per-stem band is the safety mechanism: instrumental is banded at its stricter
    3-channel-unanimity threshold, regular at 2-channel — a stem outside the
    policy is skipped before scoring.
    """
    allowed = frozenset(policy)
    out: list[HarvestRecord] = []
    for candidate, span, claim_axes in cases:
        stem = candidate.stem or "regular"
        thresholds = policy.get(stem)
        if thresholds is None:
            continue  # uncertified axis — never harvested
        prop = propose_from_candidate(
            candidate, span, scorer, thresholds=thresholds, claim_axes=claim_axes
        )
        rec = record_from_proposal(prop, allowed_stems=allowed)
        if rec is not None:
            out.append(rec)
    return out


def _existing_span_keys(ledger: Path) -> set[str]:
    if not ledger.is_file():
        return set()
    keys: set[str] = set()
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            keys.add(json.loads(line)["span_key"])
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


def write_ledger(records: Sequence[HarvestRecord], ledger: Path) -> int:
    """Append records to the JSONL ledger, deduped by span_key. Returns n written.

    Idempotent: a span_key already present in the file (or earlier in this batch)
    is skipped, so re-running the harvest never duplicates a span.
    """
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    seen = _existing_span_keys(ledger)
    written = 0
    with ledger.open("a") as fh:
        for rec in records:
            if rec.span_key in seen:
                continue
            seen.add(rec.span_key)
            fh.write(json.dumps(rec.to_json()) + "\n")
            written += 1
    return written


# ── CLI: harvest a BB set's certified spans (integration smoke on real audio) ─
#
# Real flywheel input is acquisition-proposed candidate refs on the ~1,016
# downloaded corpus sets — BLOCKED on the mix-side analysis pass (set_measures=0,
# set_stems=4 as of 2026-07-17). This CLI runs the write-side on a BB GT set,
# using each span's GT-correct ref as the candidate, so the executor is
# exercised on real audio + produces a real ledger. Banding uses CERTIFIED_POLICY
# (regular@2ch, instrumental@3ch); acappella is never harvested.


def _harvest_set(
    set_arg: str,
    *,
    stem: str | None,
    policy: dict[str, BandThresholds],
    out: Path,
) -> int:
    from pws_aligner import validate_accept_precision as vap
    from pws_aligner.capture_votes import (
        _find_aligning_dir,
        _load_manifest_by_rid,
    )
    from pws_aligner.cotrain_seam import real_probe_scorer

    set_id = vap._SET_ALIASES.get(set_arg, set_arg)
    gt = vap._load_gt(set_id)
    set_dir = _find_aligning_dir(set_id)
    if set_dir is None:
        raise SystemExit(f"no ~/aligning/{set_id}__* dir — pull the set first")
    manifest = _load_manifest_by_rid(set_dir, set_id)
    resolver = vap._manifest_resolver(manifest)
    # positives only (n_decoys=0): the GT-correct ref is the candidate.
    cases = vap.build_gt_cases(gt, resolver, n_decoys=0, stem_filter=stem)
    tuples = [
        (c.candidate, c.span, dict(c.claim_axes) if c.claim_axes else None)
        for c in cases
    ]
    scorer = real_probe_scorer(set_dir)
    records = harvest(tuples, scorer, policy=policy)
    n = write_ledger(records, out)
    bands = {s: t.min_agreeing for s, t in policy.items()}
    print(f"=== harvest {set_arg} ({set_id}) stem={stem or 'all'} ===")
    print(f"cases={len(tuples)} harvested={len(records)} written={n} policy={bands}")
    print(f"ledger={out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Harvest ACCEPTed spans into a ledger.")
    ap.add_argument("--set", required=True, help="bb11 | bb12 | <set_id>")
    ap.add_argument(
        "--stem", default=None, help="restrict to one claimed_stem (default: all)"
    )
    ap.add_argument(
        "--allow-stems",
        default=None,
        help="comma-list subset of the certified axes to harvest (default: all certified)",
    )
    ap.add_argument("--out", required=True, help="harvest-ledger JSONL path (appended)")
    args = ap.parse_args(argv)
    policy = CERTIFIED_POLICY
    if args.allow_stems:
        want = {s.strip() for s in args.allow_stems.split(",") if s.strip()}
        unknown = want - set(CERTIFIED_POLICY)
        if unknown:
            ap.error(f"uncertified stem(s) {sorted(unknown)} — not in CERTIFIED_POLICY")
        policy = {s: CERTIFIED_POLICY[s] for s in want}
    return _harvest_set(args.set, stem=args.stem, policy=policy, out=Path(args.out))


if __name__ == "__main__":
    sys.exit(main())

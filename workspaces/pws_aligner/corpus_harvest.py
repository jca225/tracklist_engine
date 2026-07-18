"""Corpus-harvest CLI — co-training flywheel step 2 (batch runner over the corpus).

Turns already-downloaded+analyzed pi-storage corpus sets into a harvest-ledger of
pseudo-labelled (ref ↔ mix-span) training pairs. It is GLUE: a corpus case-builder
(pi DB → cases) + a batch loop, delegating all alignment logic to the tested
machinery (``harvest.harvest`` / ``harvest.write_ledger`` /
``cotrain_seam.real_probe_scorer`` / ``cotrain_seam.corpus_mix_resolver``).

Placement anchor = the scraped 1001TL cue time
(``set_track_slots.cue_time_seconds`` / ``cue_seconds``) — the corpus has no GT
window. A noisy window costs RECALL not PRECISION: bad window → certified probes
disagree → ABSTAIN → not harvested, so the 2026-07-18 ACCEPT-precision
certification (regular @2-channel, instrumental @3-channel-unanimity) transfers.
Only the certified axes are harvestable (CERTIFIED_POLICY); acappella is never
harvested and needs no HuBERT, so this runs CPU-only on pi-storage.

Invariant (inherited from cotrain_seam/harvest): ZERO canonical mutation — writes
ONLY the harvest-ledger JSONL, idempotent by span_key.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from workspaces.pws_aligner.cotrain_seam import (
    BandThresholds,
    MixSpan,
    RefCandidate,
    RefMixScorer,
    corpus_mix_resolver,
    real_probe_scorer,
)
from workspaces.pws_aligner.harvest import CERTIFIED_POLICY, harvest, write_ledger

# Canonical pi-storage defaults (all overridable via CLI args for tests/other hosts).
DEFAULT_DB = Path("/mnt/storage/data/db/music_database.db")
DEFAULT_STEMS_ROOT = Path("/mnt/storage/stems/set")
DEFAULT_SPAN_S = 40.0


@dataclass(frozen=True)
class CorpusSlot:
    """A DB-projected harvest-eligible slot (decouples SQL from case-building)."""

    set_id: str
    set_audio_id: int
    slot_label: str
    recording_id: str
    ref_path: str
    claimed_version: str
    claimed_stem: str
    claimed_variant: str
    cue_time_s: float
    duration_s: float | None
    mix_full_path: str


def query_corpus_slots(
    conn: sqlite3.Connection,
    *,
    policy_stems: Iterable[str],
    limit: int | None = None,
) -> list[CorpusSlot]:
    """Strict inner-join eligibility: reference mix + reference ref @ claimed_stem
    + a scraped cue time + a certified axis. ``conn.row_factory`` must be
    ``sqlite3.Row``. On-disk audio existence is NOT checked here (disk is truth;
    the scorer abstains when absent) — see ``census`` for the disk funnel.

    Precondition: at most one ``is_reference=1`` row per set (mix) and per
    ``(recording_id, stem)`` (ref). ``is_reference`` is not UNIQUE in the schema,
    so a set with duplicate reference rows would fan a slot out into multiple
    cases; the ledger still dedupes by ``span_key`` (no double-harvest), but
    ``n_cases`` would inflate and the scorer would run redundantly.
    """
    stems = tuple(policy_stems)
    if not stems:
        return []
    placeholders = ",".join("?" for _ in stems)
    sql = f"""
        SELECT s.set_id AS set_id, sa.set_audio_id AS set_audio_id,
               s.slot_label AS slot_label, s.recording_id AS recording_id,
               ta.path AS ref_path, s.claimed_version AS claimed_version,
               s.claimed_stem AS claimed_stem, s.claimed_variant AS claimed_variant,
               COALESCE(s.cue_time_seconds, s.cue_seconds) AS cue_time_s,
               s.duration_seconds AS duration_s, sa.path AS mix_full_path
        FROM set_track_slots s
        JOIN set_audio sa ON sa.set_id = s.set_id AND sa.is_reference = 1
        JOIN track_audio ta ON ta.recording_id = s.recording_id
                            AND ta.stem = s.claimed_stem
                            AND ta.is_reference = 1
        WHERE s.claimed_stem IN ({placeholders})
          AND s.recording_id IS NOT NULL
          AND COALESCE(s.cue_time_seconds, s.cue_seconds) IS NOT NULL
        ORDER BY s.set_id, s.row_index
    """
    params: list[object] = list(stems)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    out: list[CorpusSlot] = []
    for r in conn.execute(sql, params).fetchall():
        out.append(
            CorpusSlot(
                set_id=r["set_id"],
                set_audio_id=int(r["set_audio_id"]),
                slot_label=r["slot_label"] or "",
                recording_id=r["recording_id"],
                ref_path=r["ref_path"],
                claimed_version=r["claimed_version"] or "original",
                claimed_stem=r["claimed_stem"],
                claimed_variant=r["claimed_variant"] or "regular",
                cue_time_s=float(r["cue_time_s"]),
                duration_s=float(r["duration_s"])
                if r["duration_s"] is not None
                else None,
                mix_full_path=r["mix_full_path"],
            )
        )
    return out


def _resolve(path: str, root: Path | None) -> Path:
    """Join ``root`` to ``path`` only when ``path`` is relative; else pass through."""
    p = Path(path)
    if root is not None and not p.is_absolute():
        return Path(root) / p
    return p


def build_corpus_cases(
    slots: Sequence[CorpusSlot],
    *,
    ref_audio_root: Path | None = None,
) -> list[tuple[RefCandidate, MixSpan, dict[str, str]]]:
    """One positive case per slot (no decoys — harvesting keeps confident
    agreements; decoys were only for the precision gate). ``claim_axes`` mirrors
    the slot claim so ``cotrain_seam`` can propose a correction if the accepted
    candidate ever differs (here they match by construction → correction None).

    ``span_dur_s`` comes from ``set_track_slots.duration_seconds``, which the
    tokenizer fills from the scraped *track* length — an UPPER BOUND on the
    actual play span, not the play-span length itself. The window is therefore
    typically wider than the DJ's play; per the design's abstain-heavy argument
    this is a RECALL cost (more to scan) not a PRECISION cost (certified banding
    still requires cross-channel offset agreement), so the certification holds.
    """
    cases: list[tuple[RefCandidate, MixSpan, dict[str, str]]] = []
    for s in slots:
        candidate = RefCandidate(
            recording_id=s.recording_id,
            source_url=f"corpus://{s.set_id}/{s.slot_label}",
            source_path=str(_resolve(s.ref_path, ref_audio_root)),
            version=s.claimed_version,
            stem=s.claimed_stem,
            variant=s.claimed_variant,
        )
        span = MixSpan(
            set_id=s.set_id,
            slot_label=s.slot_label,
            set_start_s=s.cue_time_s,
            span_dur_s=s.duration_s if s.duration_s is not None else DEFAULT_SPAN_S,
        )
        claim_axes = {
            "version": s.claimed_version,
            "stem": s.claimed_stem,
            "variant": s.claimed_variant,
        }
        cases.append((candidate, span, claim_axes))
    return cases


@dataclass(frozen=True)
class HarvestSummary:
    """Counts from one batch harvest run."""

    n_sets: int
    n_cases: int
    n_harvested: int
    n_written: int

    def to_json(self) -> dict:
        return {
            "n_sets": self.n_sets,
            "n_cases": self.n_cases,
            "n_harvested": self.n_harvested,
            "n_written": self.n_written,
        }


# A ScorerFactory builds a per-set scorer from (mix_full_path, mix_stem_dir).
# Injected so the batch loop is testable with a fake scorer offline.
ScorerFactory = Callable[[Path, Path], RefMixScorer]


def _default_scorer_factory(mix_full_path: Path, mix_stem_dir: Path) -> RefMixScorer:
    """Real corpus scorer: certified probes over the pi-storage layout."""
    return real_probe_scorer(
        mix_resolver=corpus_mix_resolver(mix_full_path, mix_stem_dir)
    )


def run_corpus_harvest(
    slots: Sequence[CorpusSlot],
    *,
    stems_root: Path,
    out: Path,
    policy: dict[str, BandThresholds] = CERTIFIED_POLICY,
    set_audio_root: Path | None = None,
    ref_audio_root: Path | None = None,
    scorer_factory: ScorerFactory = _default_scorer_factory,
) -> HarvestSummary:
    """Group slots by ``set_audio_id``, build ONE scorer per set (so the mix
    feature cache is reused across the set's slots), harvest under ``policy``, and
    append incrementally to the idempotent ledger (crash-safe + resumable).
    """
    stems_root = Path(stems_root)
    by_set: dict[int, list[CorpusSlot]] = {}
    for s in slots:
        by_set.setdefault(s.set_audio_id, []).append(s)

    n_cases = n_harvested = n_written = 0
    for set_audio_id, set_slots in by_set.items():
        mix_full = _resolve(set_slots[0].mix_full_path, set_audio_root)
        mix_stem_dir = stems_root / str(set_audio_id)
        scorer = scorer_factory(mix_full, mix_stem_dir)
        cases = build_corpus_cases(set_slots, ref_audio_root=ref_audio_root)
        n_cases += len(cases)
        records = harvest(cases, scorer, policy=policy)
        n_harvested += len(records)
        n_written += write_ledger(records, out)
    return HarvestSummary(len(by_set), n_cases, n_harvested, n_written)


_CENSUS_CATEGORIES: tuple[str, ...] = (
    "eligible-now",
    "no-cue-time",
    "no-ref-audio",
    "no-mix-audio",
    "no-mix-stem",
)


@dataclass(frozen=True)
class CensusReport:
    """Per-axis eligibility funnel — the flywheel's recall ceiling today."""

    by_axis: dict[str, dict[str, int]]

    def total(self) -> int:
        return sum(sum(cats.values()) for cats in self.by_axis.values())

    def to_json(self) -> dict:
        return {"by_axis": self.by_axis, "total": self.total()}

    def render(self) -> str:
        lines = ["=== corpus-harvest eligibility census ==="]
        for axis in sorted(self.by_axis):
            cats = self.by_axis[axis]
            lines.append(f"[{axis}] total={sum(cats.values())}")
            for cat in _CENSUS_CATEGORIES:
                lines.append(f"    {cat:<14} {cats.get(cat, 0)}")
        lines.append(f"TOTAL slots (certified axes): {self.total()}")
        return "\n".join(lines)


def census_rows(
    conn: sqlite3.Connection, *, policy_stems: Iterable[str]
) -> list[sqlite3.Row]:
    """LEFT-join funnel over certified-axis slots: every slot with its DB pieces
    (cue / ref audio / reference mix), so the classifier can name what blocks it.
    """
    stems = tuple(policy_stems)
    if not stems:
        return []
    placeholders = ",".join("?" for _ in stems)
    sql = f"""
        SELECT s.set_id AS set_id, s.slot_label AS slot_label,
               s.claimed_stem AS claimed_stem,
               COALESCE(s.cue_time_seconds, s.cue_seconds) AS cue_time_s,
               sa.set_audio_id AS set_audio_id, sa.path AS mix_full_path,
               ta.path AS ref_path
        FROM set_track_slots s
        LEFT JOIN set_audio sa ON sa.set_id = s.set_id AND sa.is_reference = 1
        LEFT JOIN track_audio ta ON ta.recording_id = s.recording_id
                                 AND ta.stem = s.claimed_stem
                                 AND ta.is_reference = 1
        WHERE s.claimed_stem IN ({placeholders})
          AND s.recording_id IS NOT NULL
        ORDER BY s.set_id, s.row_index
    """
    return list(conn.execute(sql, list(stems)).fetchall())


def _classify(
    row: sqlite3.Row, *, stems_root: Path, set_audio_root: Path | None
) -> str:
    """First-missing wins: cue → ref → mix(row/file) → mix-stem(instrumental) → ok."""
    if row["cue_time_s"] is None:
        return "no-cue-time"
    if row["ref_path"] is None:
        return "no-ref-audio"
    if row["mix_full_path"] is None:
        return "no-mix-audio"
    mix = _resolve(row["mix_full_path"], set_audio_root)
    if not mix.is_file():
        return "no-mix-audio"
    if row["claimed_stem"] == "instrumental":
        # set_audio_id is non-NULL here: mix_full_path NULL + is_file() guards above passed.
        stem_file = Path(stems_root) / str(row["set_audio_id"]) / "instrumental.flac"
        if not stem_file.is_file():
            return "no-mix-stem"
    return "eligible-now"


def census(
    conn: sqlite3.Connection,
    *,
    stems_root: Path,
    policy_stems: Iterable[str],
    set_audio_root: Path | None = None,
) -> CensusReport:
    """Classify every certified-axis slot by what blocks harvest, disk-checked."""
    by_axis: dict[str, dict[str, int]] = {}
    for row in census_rows(conn, policy_stems=policy_stems):
        axis = row["claimed_stem"]
        cat = _classify(row, stems_root=stems_root, set_audio_root=set_audio_root)
        bucket = by_axis.setdefault(axis, {c: 0 for c in _CENSUS_CATEGORIES})
        bucket[cat] += 1
    return CensusReport(by_axis)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Corpus-harvest CLI — co-training flywheel step 2."
    )
    ap.add_argument("--db", default=str(DEFAULT_DB), help="canonical DB path")
    ap.add_argument(
        "--stems-root",
        default=str(DEFAULT_STEMS_ROOT),
        help="mix-side stems root: <root>/<set_audio_id>/instrumental.flac",
    )
    ap.add_argument(
        "--set-audio-root",
        default=None,
        help="prefix for relative set_audio.path values (optional)",
    )
    ap.add_argument(
        "--ref-audio-root",
        default=None,
        help="prefix for relative track_audio.path values (optional)",
    )
    ap.add_argument(
        "--out", default=None, help="harvest-ledger JSONL (required unless --census)"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="cap eligible slots (harvest mode)"
    )
    ap.add_argument("--stem", default=None, help="restrict to one certified axis")
    ap.add_argument(
        "--census",
        action="store_true",
        help="report eligibility without running probes (no --out needed)",
    )
    args = ap.parse_args(argv)

    policy: dict[str, BandThresholds] = CERTIFIED_POLICY
    if args.stem:
        if args.stem not in CERTIFIED_POLICY:
            ap.error(f"uncertified stem {args.stem!r} — not in CERTIFIED_POLICY")
        policy = {args.stem: CERTIFIED_POLICY[args.stem]}
    policy_stems = tuple(policy)

    set_audio_root = Path(args.set_audio_root) if args.set_audio_root else None
    ref_audio_root = Path(args.ref_audio_root) if args.ref_audio_root else None
    stems_root = Path(args.stems_root)

    if not args.census and not args.out:
        ap.error("--out is required unless --census")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        if args.census:
            report = census(
                conn,
                stems_root=stems_root,
                policy_stems=policy_stems,
                set_audio_root=set_audio_root,
            )
            print(report.render())
            print(json.dumps(report.to_json()))
            return 0

        slots = query_corpus_slots(conn, policy_stems=policy_stems, limit=args.limit)
        summary = run_corpus_harvest(
            slots,
            stems_root=stems_root,
            out=Path(args.out),
            policy=policy,
            set_audio_root=set_audio_root,
            ref_audio_root=ref_audio_root,
        )
        print(json.dumps(summary.to_json()))
        print(f"ledger={args.out}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

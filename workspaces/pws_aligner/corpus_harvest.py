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

import sqlite3
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

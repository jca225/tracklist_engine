"""Multi-candidate identity pool — Part A of Phase 1B (spec §2A).

The shipped pipeline builds ONE ``(recording_id, stem)`` per slot from
``set_track_slots`` (``infer.slot_pools_from_rows``), so ``predict_sequence``
"selects" from a pool of size 1 and can only confirm the tokenizer claim
(state-of-record decision #19). This module builds the real pool per slot:

    pool = {the tokenizer claim} ∪ {the set's stem-appropriate reference pool}

so the aligner *can* pick a different recording than claimed. Scoring that pool
(blind MERT chamfer) lives in [open_set_identity.py]; the gated override that
consumes both lives in the ``infer.py`` seam. This core is PURE over an injected
stem-partitioned inventory (mirrors ``open_set_identity`` / ``stem_mert``): the
I/O that reads which of the set's references are acappella vs instrumental is the
seam's job, not this module's — so this stays unit-testable with no DB/GPU/pi.

Routing (decision #22, final): acappella slots identify against the set's
acappella refs; regular/instrumental slots identify against the instrumental refs
(full vocal songs identify by their instrumental backbone — there is no third
regime). The claim is ALWAYS in the pool so "no override" is representable and
``margin_gate``'s accept-claim branch has the claim present to fall back on.
"""

from __future__ import annotations

from dataclasses import dataclass

from alignment.records import SlotCandidate

# claimed_stem -> which of the set's stem-partitioned ref pools to identify
# against. Mirrors harness/axes.py routing but at the identity (not placement)
# layer: only two identity regimes (vocal vs instrumental backbone).
_POOL_STEM = {
    "acappella": "acappella",
    "instrumental": "instrumental",
    "regular": "instrumental",
}

# ref_source tags on the emitted SlotCandidate, so downstream provenance can tell
# the claim apart from a pool-added rival.
CLAIM_SOURCE = "claim"
POOL_SOURCE = "set_pool"


def pool_stem_for(claimed_stem: str | None) -> str:
    """Which stem-partitioned ref pool a slot identifies against (defaults to
    instrumental — the regular/full-song backbone regime)."""
    return _POOL_STEM.get(claimed_stem or "regular", "instrumental")


@dataclass(frozen=True)
class CandidatePool:
    """The identity candidates for one slot + provenance for the AxisBelief.

    ``candidates`` always leads with the claim (when the slot has one), so
    ``candidates[0]`` is the "no override" default and the claim is guaranteed
    present for ``margin_gate``'s accept-claim fallback. ``claim`` is the claim
    candidate (or None for a claimless slot). ``pool_stem`` records which ref
    regime this slot was routed to; ``source_size`` is how many stem-appropriate
    refs were offered before dedup against the claim.
    """

    slot_label: str
    claim: SlotCandidate | None
    candidates: tuple[SlotCandidate, ...]
    pool_stem: str
    source_size: int

    @property
    def size(self) -> int:
        return len(self.candidates)

    @property
    def is_overridable(self) -> bool:
        """True when the pool offers a real choice (a rival beyond the claim)."""
        return self.size > (1 if self.claim is not None else 0)


def build_pool(
    slot_label: str,
    claim_recording_id: str | None,
    claimed_stem: str,
    set_pool_by_stem: dict[str, tuple[str, ...]],
) -> CandidatePool:
    """Pool for one slot = {claim} ∪ {stem-routed set refs}, claim first.

    ``set_pool_by_stem`` maps ``"acappella"``/``"instrumental"`` -> the set's
    reference recording_ids of that stem (the seam builds this from the loaded
    refs + track_audio stems). Rivals equal to the claim are deduped; the claim's
    own stem is preserved on its candidate (it may be regular even when routed to
    the instrumental identity pool). Order: claim, then refs in the given order.
    """
    stem = pool_stem_for(claimed_stem)
    refs = set_pool_by_stem.get(stem, ())

    claim_cand = (
        SlotCandidate(
            recording_id=claim_recording_id,
            claimed_stem=claimed_stem,
            ref_source=CLAIM_SOURCE,
        )
        if claim_recording_id
        else None
    )

    candidates: list[SlotCandidate] = []
    seen: set[str] = set()
    if claim_cand is not None:
        candidates.append(claim_cand)
        seen.add(claim_cand.recording_id)
    for rid in refs:
        if rid in seen:
            continue
        seen.add(rid)
        candidates.append(
            SlotCandidate(recording_id=rid, claimed_stem=stem, ref_source=POOL_SOURCE)
        )

    return CandidatePool(
        slot_label=slot_label,
        claim=claim_cand,
        candidates=tuple(candidates),
        pool_stem=stem,
        source_size=len(refs),
    )


def build_pools(
    rows: tuple[dict, ...],
    set_pool_by_stem: dict[str, tuple[str, ...]],
) -> dict[str, CandidatePool]:
    """Per-slot pools for a set's tracklist rows (drop-in richer replacement for
    ``infer.slot_pools_from_rows``). Rows without a claim recording still get a
    pool of the stem-routed refs (they can only override, never accept-claim)."""
    pools: dict[str, CandidatePool] = {}
    for r in rows:
        label = r["slot_label"]
        if label in pools:
            continue  # concurrent (`w`) rows share the parent slot's claim/pool
        pools[label] = build_pool(
            slot_label=label,
            claim_recording_id=r.get("recording_id"),
            claimed_stem=r.get("claimed_stem", "regular"),
            set_pool_by_stem=set_pool_by_stem,
        )
    return pools


def as_slot_pools(
    pools: dict[str, CandidatePool],
) -> dict[str, tuple[SlotCandidate, ...]]:
    """Adapter to the legacy ``MertLearnedAligner(slot_pools=...)`` shape
    (``dict[slot_label, tuple[SlotCandidate]]``), so the seam can widen the pool
    without touching ``_assign_slot``'s signature."""
    return {label: p.candidates for label, p in pools.items()}

"""Mashup-compatibility supervision: extract (bed, payload) pairs from BB12 GT.

A *mashup pair* is two tracks the DJ layered simultaneously — recoverable from the
hand-labeled ground truth as a **mix-time overlap** between an acappella *payload*
and an instrumental/regular *bed* (the instr-anchor / acap-payload asymmetry).

Positives = overlapping (bed, payload). Negatives = non-co-occurring (bed, payload)
sampled from the same track pool, so a probe can't cheat on per-track identity.

Pure logic, no audio/GPU — validate the supervision before embedding anything.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from labeling.ground_truth.schema import GroundTruthTrack, load

BED_STEMS = {"instrumental", "regular"}
PAYLOAD_STEMS = {"acappella"}
MIN_OVERLAP_S = 8.0          # ignore incidental few-second brushes


@dataclass(frozen=True)
class Stem:
    """One stem-token to embed: a recording played in a given role."""
    track_id: str
    role: str                # 'bed' (-> demucs instrumental) | 'payload' (-> demucs vocals)
    label: str               # human-readable, for logs

    @property
    def demucs_stem(self) -> str:
        return "instrumental" if self.role == "bed" else "vocals"


@dataclass(frozen=True)
class MashupPair:
    bed: Stem
    payload: Stem
    overlap_s: float
    positive: bool


def _interval(t: GroundTruthTrack) -> tuple[float, float]:
    return (t.set_start_s, t.set_end_s)


def _overlap_s(a: GroundTruthTrack, b: GroundTruthTrack) -> float:
    a0, a1 = _interval(a)
    b0, b1 = _interval(b)
    return max(0.0, min(a1, b1) - max(a0, b0))


def _eligible(t: GroundTruthTrack) -> bool:
    return bool(t.track_id) and not t.unalignable


def extract_pairs(gt_path: Path | str, neg_per_pos: int = 3) -> list[MashupPair]:
    res = load(gt_path)
    if not res.is_ok():
        raise SystemExit(f"GT load failed: {res.error}")
    tracks = [t for t in res.value.tracks if _eligible(t)]

    beds = [t for t in tracks if t.claimed_stem in BED_STEMS]
    payloads = [t for t in tracks if t.claimed_stem in PAYLOAD_STEMS]

    def stem_of(t: GroundTruthTrack, role: str) -> Stem:
        return Stem(track_id=t.track_id, role=role,
                    label=f"{t.slot_label}:{(t.label or '')[:32]}")

    positives: list[MashupPair] = []
    pos_keys: set[tuple[str, str]] = set()
    for p in payloads:
        for b in beds:
            if b.track_id == p.track_id:
                continue
            ov = _overlap_s(b, p)
            if ov >= MIN_OVERLAP_S:
                positives.append(MashupPair(stem_of(b, "bed"), stem_of(p, "payload"), ov, True))
                pos_keys.add((b.track_id, p.track_id))

    # Negatives: (bed, payload) that never co-occur. Deterministic stride sampling
    # (no RNG — varies coverage without Math.random), capped at neg_per_pos x.
    negatives: list[MashupPair] = []
    want = len(positives) * neg_per_pos
    cand = [(b, p) for b in beds for p in payloads
            if b.track_id != p.track_id and (b.track_id, p.track_id) not in pos_keys
            and _overlap_s(b, p) == 0.0]
    stride = max(1, len(cand) // want) if want else 1
    for b, p in cand[::stride][:want]:
        negatives.append(MashupPair(stem_of(b, "bed"), stem_of(p, "payload"), 0.0, False))

    return positives + negatives


def needed_stems(pairs: list[MashupPair]) -> list[Stem]:
    seen: dict[tuple[str, str], Stem] = {}
    for mp in pairs:
        for s in (mp.bed, mp.payload):
            seen[(s.track_id, s.role)] = s
    return list(seen.values())


if __name__ == "__main__":
    gt = sys.argv[1] if len(sys.argv) > 1 else "labeling/fixtures/bb12_ground_truth.yaml"
    pairs = extract_pairs(gt)
    pos = [p for p in pairs if p.positive]
    neg = [p for p in pairs if not p.positive]
    stems = needed_stems(pairs)
    print(f"positives={len(pos)}  negatives={len(neg)}  distinct stems to embed={len(stems)}")
    print(f"  beds={sum(s.role=='bed' for s in stems)}  payloads={sum(s.role=='payload' for s in stems)}")
    print("\nsample positive mashup pairs (acap payload OVER instr/regular bed):")
    for mp in sorted(pos, key=lambda m: -m.overlap_s)[:12]:
        print(f"  {mp.overlap_s:6.1f}s  BED {mp.bed.label:38s} <- PAYLOAD {mp.payload.label}")

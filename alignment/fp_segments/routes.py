"""Strict like-for-like channel routes for constituent segment alignment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentLane:
    name: str
    mix_channel: str
    mix_file: str
    mix_cache_suffix: str
    reference_stem: str
    claimed_stems: frozenset[str]


LANES = {
    "instrumental": SegmentLane(
        name="instrumental",
        mix_channel="instrumental",
        mix_file="mix_instrumental.flac",
        mix_cache_suffix="instrumental_mix_hashes.pkl",
        reference_stem="instrumental",
        claimed_stems=frozenset({"instrumental"}),
    ),
    "vocal": SegmentLane(
        name="vocal",
        mix_channel="vocals",
        mix_file="mix_vocals.flac",
        mix_cache_suffix="vocal_mix_hashes.pkl",
        reference_stem="vocals",
        claimed_stems=frozenset({"acappella"}),
    ),
}


def lane(name: str) -> SegmentLane:
    try:
        return LANES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown segment lane {name!r}; expected one of {sorted(LANES)}"
        ) from exc


def validate_route(*, mix_channel: str, reference_stem: str) -> SegmentLane:
    """Reject unlike-channel matching mechanically."""
    for item in LANES.values():
        if (
            item.mix_channel == mix_channel
            and item.reference_stem == reference_stem
        ):
            return item
    raise ValueError(
        f"unlike fingerprint route forbidden: "
        f"mix_{mix_channel}->reference_{reference_stem}"
    )

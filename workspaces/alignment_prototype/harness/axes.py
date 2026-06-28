"""Axis decomposition — route each stem to its nuisance-invariant feature/probes.

A song ≈ timbre × harmony × language, near-orthogonal. The placement/fiber math
must run on the axis that is INVARIANT to the variation that does NOT carry
section identity — otherwise slight variation (a key change, a re-sung chorus,
added production) makes the matcher believe a different section played:

  axis      feature   invariant to          used for
  language  HuBERT    key, timbre           vocal placement ("lyrics don't transpose")
  harmony   chroma    timbre, lyrics        instrumental/full placement
  timbre    MERT      —                     IDENTITY ONLY (cannot localize: ~900s argmax)

This is the code form of that principle: given a span's ``claimed_stem``, which
mix-stem file to load, which reference Demucs stem to match against, the
invariant feature, and which placement probes to run. It supersedes/extends
``refine_ref_offsets._MIX_SOURCE`` (which only carried mix_file + ref_stem) by
adding the invariant feature and the probe routing. Route by the stem the AUDIO
actually is, not the scraped label (continuity_refine._run_cross_channel is the
validator) — see the 2026-06-10 "trusted the acappella label" regression.

MERT is deliberately absent from ``placement_probes`` everywhere: it votes
identity (candidate selection), never placement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AxisRoute:
    mix_file: str  # mix-side stem file under ~/aligning/<set>/
    ref_stem: str | None  # reference Demucs stem key (None = full track)
    invariant_feature: str  # 'hubert' (language) | 'chroma' (harmony)
    placement_probes: tuple[str, ...]  # probe names to run for placement, by leverage


# claimed_stem -> route. Vocals lead with HuBERT (language axis); instrumental
# leads with fingerprint+chroma (harmony); both add the continuity stack for
# repeat-robustness. fingerprint is omitted for vocals (constellation hashing is
# weak on vocals-only audio).
AXES: dict[str, AxisRoute] = {
    "acappella": AxisRoute(
        mix_file="mix_vocals.flac",
        ref_stem="vocals",
        invariant_feature="hubert",
        placement_probes=("hubert", "continuity", "chroma"),
    ),
    "instrumental": AxisRoute(
        mix_file="mix_instrumental.flac",
        ref_stem="instrumental",
        invariant_feature="chroma",
        placement_probes=("fp", "chroma", "continuity"),
    ),
    "regular": AxisRoute(
        mix_file="mix.m4a",
        ref_stem=None,
        invariant_feature="chroma",
        placement_probes=("fp", "chroma"),
    ),
}


def route_for_stem(claimed_stem: str | None) -> AxisRoute:
    """Axis route for a span's claimed_stem (defaults to the regular/full route)."""
    return AXES.get((claimed_stem or "regular"), AXES["regular"])

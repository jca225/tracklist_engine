"""Stage-4: classify a per-stem match-rate dict into a stem-mask label.

Takes the `stem_match_rates_json` payload produced by stem-aware Stage-1
and turns it into a categorical read on what's actually audible in the
mix at that section:

  - **full**         — all four stems aligned; the DJ played the track straight.
  - **acappella**    — only vocals aligned; instrumentals came from a different track.
  - **instrumental** — drums/bass/other aligned but vocals didn't.
  - **partial**      — some stems in, some out; classic mashup component.
  - **none**         — no stem cleared the threshold; alignment isn't trustworthy.

Pure function, no I/O. Consumed by the UI and by Stage-6 mashup attribution.
"""
from __future__ import annotations

import re
from typing import Literal, Mapping

StemLabel = Literal["full", "acappella", "instrumental", "partial", "none"]

DEFAULT_THRESHOLD: float = 0.4
DEFAULT_MARGIN: float = 0.08
INSTRUMENT_STEMS: tuple[str, ...] = ("drums", "bass", "other")


# Tracklist version tags — parsed out of row text like
# "Bastille - Good Grief (Don Diablo Remix) (Instrumental Mix)"
# and used as a hard prior for section-level classification when present.
# Order matters: more specific patterns first. Keys are regexes; values are
# the label they imply about what's audible in the mix.
_VERSION_PATTERNS: tuple[tuple[re.Pattern[str], StemLabel], ...] = (
    (re.compile(r"\ba\s*cap+[eé]l+a\b",                re.IGNORECASE), "acappella"),
    (re.compile(r"\b(vocal|vox)\s+(only|version|mix)", re.IGNORECASE), "acappella"),
    (re.compile(r"\(\s*acap\s*\)",                     re.IGNORECASE), "acappella"),
    (re.compile(r"\binstrumental(?:\s+(?:mix|version|edit))?\b", re.IGNORECASE), "instrumental"),
    (re.compile(r"\bdub\s*(?:mix|version)?\b",         re.IGNORECASE), "instrumental"),
    (re.compile(r"\bkaraoke\b",                        re.IGNORECASE), "instrumental"),
    (re.compile(r"\(\s*instr\s*\)",                    re.IGNORECASE), "instrumental"),
    (re.compile(r"\(\s*full\s*\)",                     re.IGNORECASE), "full"),
)


def parse_version_tag(text: str | None) -> StemLabel | None:
    """Extract a stem-mask prior from tracklist row text when present.

    Returns None when no explicit version marker is found (the vast
    majority of rows). The caller should prefer the parsed tag over the
    stem-energy classifier when it exists, because the DJ explicitly
    labeled the version — nothing more authoritative than that.
    """
    if not text:
        return None
    for pat, label in _VERSION_PATTERNS:
        if pat.search(text):
            return label
    return None


def classify(
    stem_rates: Mapping[str, float],
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> StemLabel:
    """Section-level classifier from per-stem DTW match rates.

    `threshold` is the paper's 0.4 'aligned' cutoff. `margin` guards
    against noise wins — if the top stem beats the second-best by less
    than `margin`, we downgrade to `partial` rather than trust a
    borderline signal (the "Good Grief acappella" failure mode).
    """
    if not stem_rates:
        return "none"

    v = stem_rates.get("vocals", 0.0)
    inst = [stem_rates.get(s, 0.0) for s in INSTRUMENT_STEMS]
    f = stem_rates.get("full", 0.0)

    v_ok = v >= threshold
    inst_ok = [r >= threshold for r in inst]
    n_inst_ok = sum(inst_ok)
    any_inst_ok = n_inst_ok > 0
    all_inst_ok = n_inst_ok == len(INSTRUMENT_STEMS)
    f_ok = f >= threshold

    if max(v, max(inst) if inst else 0.0, f) < threshold:
        return "none"

    # Margin guard: top stem must dominate the next clearly.
    ordered = sorted(stem_rates.values(), reverse=True)
    if len(ordered) >= 2 and (ordered[0] - ordered[1]) < margin and not (f_ok and any_inst_ok):
        return "partial"

    # Clean full: vocals + every instrument aligned.
    if v_ok and all_inst_ok:
        return "full"
    if f_ok and (v_ok or any_inst_ok):
        return "full"
    if v_ok and not any_inst_ok:
        return "acappella"
    if not v_ok and any_inst_ok:
        return "instrumental"
    return "partial"


# UI-friendly render map. Keep labels short; the UI stacks many of them.
BADGE: dict[StemLabel, tuple[str, str]] = {
    "full":         ("🎛", "full"),
    "acappella":    ("🎤", "acappella"),
    "instrumental": ("🥁", "instrumental"),
    "partial":      ("🧩", "partial"),
    "none":         ("⚠",  "unaligned"),
}

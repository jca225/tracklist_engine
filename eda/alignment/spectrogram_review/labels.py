"""Human-readable copy for the spectrogram review gallery."""

from __future__ import annotations

from eda.alignment.spectrogram_review.spans import SpanCard

CAUSE_TITLE = {
    "identity": "Wrong song",
    "placement": "Wrong time in the mix",
    "mis-route": "Used the wrong stem channel",
    "tempo/octave": "Tempo / octave mismatch",
    "loop-instance": "Loop / repeat confusion",
    "instance-ambiguity": "Ambiguous repeat of the same song",
    "decode-residual": "Right ballpark, wrong detail inside the song",
}

CAUSE_BLURB = {
    "identity": "The model matched a different track than the one that was actually playing.",
    "placement": "It found the song but dropped it in the wrong place on the mix timeline (>15s off).",
    "mis-route": "It treated vocals like a full track (or vice versa), so the wrong matcher ran.",
    "tempo/octave": "Stretch / pitch is off enough that the timeline doesn't line up.",
    "loop-instance": "The song loops; the model latched onto the wrong loop cycle.",
    "instance-ambiguity": "The mix repeats similar material; hard to tell which repeat is which.",
    "decode-residual": "Start time is roughly right, but the path through the song is still wrong.",
}

STEM_TITLE = {
    "acappella": "Vocals / acapella",
    "instrumental": "Instrumental",
    "regular": "Full track",
}

SPAN_TITLE = {
    "linear": "straight play",
    "multiseg": "jumps around inside the song",
    "loop": "loops",
    "oddratio": "odd tempo stretch",
}


def fmt_clock(seconds: float) -> str:
    """Format mix/source seconds as m:ss (or h:mm:ss if long)."""
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    if h:
        return f"{h}:{m:02d}:{sec:04.1f}"
    return f"{m}:{sec:04.1f}"


def fmt_delta(seconds: float) -> str:
    """Signed human delta, e.g. '12s too late' / '3s too early'."""
    if abs(seconds) < 0.5:
        return "about on time"
    late = seconds > 0
    mag = abs(seconds)
    if mag >= 60:
        body = f"{mag / 60:.1f} min"
    else:
        body = f"{mag:.0f}s"
    return f"{body} too late" if late else f"{body} too early"


def place_delta_s(card: SpanCard) -> float:
    """Pred − actual on the mix (positive = we placed later than truth)."""
    return card.pred_set_start_s - card.gt_set_start_s


def ref_delta_s(card: SpanCard) -> float:
    return card.pred_ref_start_s - card.gt_ref_start_s


def group_title(group_key: str, *, success: bool) -> str:
    if not success:
        return CAUSE_TITLE.get(group_key, group_key.replace("-", " ").title())
    # success keys look like: "acappella · loop · tight"
    parts = [p.strip() for p in group_key.split("·")]
    stem = STEM_TITLE.get(parts[0], parts[0]) if parts else group_key
    structure = SPAN_TITLE.get(parts[1], parts[1]) if len(parts) > 1 else ""
    tight = parts[2] if len(parts) > 2 else ""
    place = (
        "start time close"
        if tight == "tight"
        else "start time a bit off"
        if tight == "loose-place"
        else tight
    )
    if structure:
        return f"{stem} — {structure} — {place}"
    return stem


def group_blurb(group_key: str, *, success: bool) -> str:
    if not success:
        return CAUSE_BLURB.get(group_key, "")
    return "These lined up well enough (≥50% of the span decoded correctly)."


def one_liner(card: SpanCard) -> str:
    """Single sentence under the track title."""
    stem = STEM_TITLE.get(card.gt_stem, card.gt_stem)
    if card.success:
        place = fmt_delta(place_delta_s(card))
        return f"Looks good — {stem}. Mix start is {place}."
    title = CAUSE_TITLE.get(card.cause, card.cause)
    if card.cause == "identity":
        return f"Missed it — {title.lower()}."
    if card.cause == "placement":
        return f"Missed it — put the {stem.lower()} {fmt_delta(place_delta_s(card))} in the mix."
    if card.cause == "decode-residual":
        return (
            f"Close on the mix ({fmt_delta(place_delta_s(card))}), "
            f"but inside the song we're {fmt_delta(ref_delta_s(card))}."
        )
    return f"Missed it — {title.lower()} ({stem.lower()})."


def accuracy_plain(card: SpanCard) -> str:
    pct = int(round(card.traj_strict * 100))
    if pct >= 80:
        return f"Matched about {pct}% of this clip"
    if pct >= 50:
        return f"Partly matched ({pct}% of this clip)"
    if pct > 0:
        return f"Mostly wrong ({pct}% matched)"
    return "Didn't match this clip"


def stem_plain(stem: str) -> str:
    return STEM_TITLE.get(stem, stem)

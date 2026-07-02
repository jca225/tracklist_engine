#!/usr/bin/env python3
"""Lyrics-as-GRADER precision test — the VOCAL analog of recon_probe's host-grader.

Question (from the "why can't lyrics be the vocal grader?" thread): on the vocal channel,
when the lyrics matcher is CONFIDENT (big score margin — the top diagonal's distinctive
rare-word anchors beat the runner-up), are its placements actually right? If a lyrics-margin
gate reaches high precision, the vocal parts of the 20k can be auto-labeled self-supervised
the same way reconstruction labels host tracks — a division of labor, both label-free.

Grades placement (set_start) with identity given (the tracklist supplies the candidate ref),
which is exactly how a labeler would use it. Runs on the cached Whisper transcripts (no GPU).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.lyrics_grader_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.ground_truth import schema  # noqa: E402
from workspaces.alignment_prototype.lyrics_align import (  # noqa: E402
    _bigram_times,
    _norm,
    _slot_order,
    candidate_diagonals,
    load_cached,
    monotonic_decode,
    transcribe_words,
)

SET_DIR = Path.home() / "aligning" / "1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
GT_YAML = _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"
import json  # noqa: E402


def _harvest(pairs: list[tuple[float, int]], label: str) -> None:
    """Precision/recall of a confidence gate (highest confidence first)."""
    pairs = [p for p in pairs if p[0] is not None]
    if not pairs or not any(l for _, l in pairs) or all(l for _, l in pairs):
        print(f"  [{label}] insufficient spans / no class variety")
        return
    order = sorted(pairs, key=lambda p: -p[0])
    total = sum(l for _, l in pairs)
    print(f"  [{label}] n={len(pairs)} correct={total}")
    print(f"    {'gate':>8} {'kept':>5} {'prec':>6} {'recall':>7}")
    marks = {int(len(order) * f) for f in (0.1, 0.25, 0.5, 1.0)}
    best = None
    tp = 0
    for rank, (conf, lab) in enumerate(order, 1):
        tp += lab
        prec, rec = tp / rank, tp / total
        if prec >= 0.90 and best is None and rank >= 5:
            best = (conf, rank, prec, rec)
        if rank in marks:
            print(f"    {conf:>8.3f} {rank:>5} {prec:>5.0%} {rec:>7.0%}")
    top = order[: max(5, len(order) // 10)]
    tp10 = sum(l for _, l in top) / len(top)
    if best:
        conf, k, p, r = best
        print(
            f"    → τ@P≥90%: conf≥{conf:.3f} keeps {k}/{len(pairs)} ({r:.0%} of correct) "
            f"→ ~{r:.0%} of vocal spans harvestable CLEAN. GRADER CLEARS THE BAR."
        )
    else:
        print(f"    → no τ reaches 90% precision (top-decile precision {tp10:.0%})")


def main() -> int:
    r = schema.load(str(GT_YAML))
    if not r.is_ok:
        print("GT load failed", file=sys.stderr)
        return 2
    gt = r.value
    man = json.loads((SET_DIR / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in man["tracks"]}
    mix_dur = man.get("mix_duration_s") or max(t.set_end_s for t in gt.tracks)
    max_slot = (
        max(
            (_slot_order(t.slot_label)[0] for t in gt.tracks if t.slot_label), default=1
        )
        or 1
    )

    print("lyrics: mix_vocals transcript (cached)…")
    mix_bt = _bigram_times(_norm(transcribe_words(SET_DIR / "mix_vocals.flac")))

    aca = sorted(
        [t for t in gt.tracks if t.claimed_stem == "acappella"],
        key=lambda t: _slot_order(t.slot_label),
    )
    rows = []
    lyr_spans = []
    for t in aca:
        if (t.set_end_s - t.set_start_s) < 4.0:
            continue
        mt = by_tid.get(t.track_id)
        voc = (mt.get("stems") or {}).get("vocals") if mt else None
        cw = load_cached(voc) if voc else None
        if not cw:
            continue
        cands = candidate_diagonals(_norm(cw), mix_bt)
        epos = _slot_order(t.slot_label)[0] / max_slot * mix_dur
        rows.append({"t": t, "cands": cands})
        lyr_spans.append((cands, epos))

    chosen = monotonic_decode(lyr_spans)

    errs, harvest_ss, harvest_score = [], [], []
    for i, row in enumerate(rows):
        t = row["t"]
        ss_l, _rs_l = chosen[i]
        if ss_l is None or not row["cands"]:
            continue
        scores = sorted((c[2] for c in row["cands"]), reverse=True)
        margin = scores[0] - (scores[1] if len(scores) > 1 else 0.0)
        err = abs(ss_l - t.set_start_s)
        errs.append(err)
        lab = int(err <= 15.0)
        harvest_ss.append((margin, lab))  # confidence = score margin (distinctiveness)
        harvest_score.append((scores[0], lab))  # confidence = raw top score

    print(f"\n== LYRICS as vocal grader (BB12 acappella, set_start; identity given) ==")
    print(f"  scorable spans: {len(errs)}")
    if errs:
        e = np.array(errs)
        print(
            f"  placement: median {np.median(e):.1f}s  <5s {np.mean(e <= 5):.0%}  "
            f"<15s {np.mean(e <= 15):.0%}  (sanity vs prior ~2.2s median)"
        )
    print("\n-- pseudo-label GATE (does confidence pick clean vocal labels?) --")
    _harvest(harvest_ss, "lyrics MARGIN (top - runner-up)")
    _harvest(harvest_score, "lyrics raw top score")
    print(
        "\n  reference: reconstruction HOST grader top-decile precision ~78% (recon_probe);"
    )
    print("  if lyrics MARGIN clears 90%, vocals are a YES for self-labeling the 20k.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

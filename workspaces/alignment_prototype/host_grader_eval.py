#!/usr/bin/env python3
"""Host-grader precision test with FINGERPRINT SHARPNESS — symmetric to the lyrics
(vocal) and reconstruction (host) grader tests.

The open asymmetry: lyrics CLEARS 90% precision for vocals; reconstruction stalls at 78%
for host. The deferred lever is fingerprint sharpness (the "is this alignment diagonal
unique / peaked" signal — fp localizes to 0.2s where recon's mel is fuzzy). Question: does
fp sharpness (alone, no audio) gate host pseudo-labels at 90%? If yes, BOTH channels are
cleanly self-labelable (vocals→lyrics, host→fp) and the 20k is in reach.

Cheap: reads the cached fp hits + predicted timeline + GT. No audio, no model.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.host_grader_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT_DIR = Path(__file__).resolve().parent / "out"
FP_CACHE = Path(__file__).resolve().parent / ".cache" / "set_fp_hits"
GT_YAML = _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"


def _harvest(pairs: list[tuple[float, int]], label: str) -> None:
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
            f"→ ~{r:.0%} of host spans harvestable CLEAN. CLEARS THE BAR."
        )
    else:
        print(f"    → no τ reaches 90% precision (top-decile precision {tp10:.0%})")


def main() -> int:
    import yaml

    set_id = "1fsnxchk"
    pred = json.loads((OUT_DIR / f"{set_id}_predicted_timeline.json").read_text())[
        "spans"
    ]
    gt_rows = [
        r
        for r in yaml.safe_load(GT_YAML.read_text())["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")
    ]
    fp_hits = json.loads((FP_CACHE / f"{set_id}.json").read_text())

    # fp hits indexed by recording_id -> list of (mid_s, votes, sharpness, score)
    by_rec: dict[str, list] = {}
    for h in fp_hits:
        if h.get("stem") != "regular":
            continue
        mid = 0.5 * (h["mix_start_s"] + h["mix_end_s"])
        by_rec.setdefault(str(h["recording_id"]), []).append(
            (mid, h["votes"], h["sharpness"], h["score"])
        )

    def fp_for(rec: str, set_start: float):
        """Nearest fp hit for this recording to the predicted set_start (<=30s)."""
        hits = by_rec.get(rec)
        if not hits:
            return None
        mid, v, s, sc = min(hits, key=lambda x: abs(x[0] - set_start))
        if abs(mid - set_start) > 30:
            return None
        return v, s, sc

    sharp_g, votes_g, vs_g, score_g = [], [], [], []
    n_host = n_fp = 0
    for p in pred:
        if (p.get("claimed_stem") or "regular") != "regular":
            continue
        try:
            ss, se = float(p["set_start_s"]), float(p["set_end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if se - ss < 4:
            continue
        rec = str(p.get("recording_id") or "")
        n_host += 1
        # correctness vs GT (identity + set_start<15s) — same rule as recon grader
        ov = [
            r
            for r in gt_rows
            if float(r["set_start_s"]) < se + 5 and float(r["set_end_s"]) > ss - 5
        ]
        id_ok = any(str(r["track_id"]) == rec for r in ov)
        same = [r for r in ov if str(r["track_id"]) == rec]
        place_ok = (
            bool(same) and min(abs(float(r["set_start_s"]) - ss) for r in same) <= 15
        )
        lab = int(id_ok and place_ok)
        fp = fp_for(rec, ss)
        if fp is None:
            continue  # fp abstains → not a harvest candidate
        n_fp += 1
        v, s, sc = fp
        sharp_g.append((s, lab))
        votes_g.append((float(v), lab))
        vs_g.append((float(v) * s, lab))
        score_g.append((sc, lab))

    print(f"\n== HOST grader via FINGERPRINT SHARPNESS (BB12 regular) ==")
    print(f"  host spans: {n_host} | with an fp hit near predicted placement: {n_fp}")
    print("\n-- pseudo-label GATE (does fp confidence pick clean host labels?) --")
    _harvest(sharp_g, "fp sharpness (peak/second z)")
    _harvest(votes_g, "fp votes")
    _harvest(vs_g, "fp votes*sharpness")
    _harvest(score_g, "fp score")
    print(
        "\n  references: lyrics VOCAL grader clears 90% @61% recall; recon HOST grader ~78%."
    )
    print("  if fp sharpness clears 90%, BOTH channels self-labelable → 20k in reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

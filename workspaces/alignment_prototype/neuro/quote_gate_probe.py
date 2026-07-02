"""Prototype: does a spoken-QUOTE gate recover the quote-driven placement tail?

The contamination check found the lyrics-placement tail lands in quote-dense mix
windows (tail 0.07 vs good 0.01 out-of-vocab density; Spearman +0.32 @ GT window).
But out-of-vocab words themselves are INERT — they never match a candidate bigram.
The damage is done by ordinary common words *inside* a spoken-quote region forming
spurious common-bigram diagonals. So the gate is two-stage:

  1. DETECT quote regions unsupervised: local density of out-of-vocab words
     (words in NO cached song transcript) over mix time; a region above THRESH is
     a spoken-quote / dialogue zone.
  2. SUPPRESS: drop every mix bigram (in-vocab included) whose time falls in a
     quote region, then re-run the exact lyrics anchor decode.

Compares baseline vs gated placement error per acappella span. Read-only; writes
only neuro/out/. No edits to lyrics_align.py / infer.py.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.neuro.quote_gate_probe \
        [--thresh 0.15] [--half 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml as _yaml

from workspaces.alignment_prototype.lyrics_align import (
    _norm,
    _slot_order,
    candidate_diagonals,
    load_cached,
    monotonic_decode,
    transcribe_words,
)
from workspaces.alignment_prototype.path_decode import find_aligning_dir

GT_YAML = _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"
SET_ID = "1fsnxchk"


def _gated_bigram_times(mix_seq, in_quote) -> dict[str, list[float]]:
    """_bigram_times, but skip any bigram touching a quote-region word."""
    g: dict[str, list[float]] = {}
    for i in range(len(mix_seq) - 1):
        if in_quote[i] or in_quote[i + 1]:
            continue
        g.setdefault(mix_seq[i][0] + " " + mix_seq[i + 1][0], []).append(
            0.5 * (mix_seq[i][1] + mix_seq[i + 1][1])
        )
    return g


def _plain_bigram_times(mix_seq) -> dict[str, list[float]]:
    g: dict[str, list[float]] = {}
    for i in range(len(mix_seq) - 1):
        g.setdefault(mix_seq[i][0] + " " + mix_seq[i + 1][0], []).append(
            0.5 * (mix_seq[i][1] + mix_seq[i + 1][1])
        )
    return g


def _anchors(mix_bt, raw_tracks, by_tid, manifest_dur, max_slot) -> dict:
    aca = [t for t in raw_tracks if t.get("claimed_stem") == "acappella"]
    aca.sort(key=lambda t: _slot_order(t["slot_label"]))
    spans, slots = [], []
    for s in aca:
        voc = (by_tid.get(s["track_id"], {}).get("stems") or {}).get("vocals")
        if not voc or not Path(voc).is_file():
            continue
        cw = load_cached(voc)
        if not cw:
            continue
        cands = candidate_diagonals(_norm(cw), mix_bt)
        if not cands:
            continue
        epos = _slot_order(s["slot_label"])[0] / max_slot * manifest_dur
        spans.append((cands, epos))
        slots.append(s["slot_label"])
    chosen = monotonic_decode(spans)
    return {sl: ss for sl, (ss, _rs) in zip(slots, chosen) if ss is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--thresh",
        type=float,
        default=0.15,
        help="out-of-vocab density above which a region is a quote",
    )
    ap.add_argument(
        "--half",
        type=float,
        default=6.0,
        help="half-window (s) for local out-of-vocab density",
    )
    args = ap.parse_args()

    set_dir = find_aligning_dir(SET_ID)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    # song vocab from every cached ref transcript
    song_vocab: set[str] = set()
    for t in manifest["tracks"]:
        voc = (t.get("stems") or {}).get("vocals")
        if voc and Path(voc).is_file():
            cw = load_cached(voc)
            if cw:
                song_vocab.update(w for w, _ in _norm(cw))

    mix_seq = _norm(transcribe_words(set_dir / "mix_vocals.flac"))
    times = np.array([tm for _, tm in mix_seq])
    oov = np.array([w not in song_vocab for w, _ in mix_seq], bool)

    # local out-of-vocab density → quote-region mask per word
    in_quote = np.zeros(len(mix_seq), bool)
    for i, t0 in enumerate(times):
        m = np.abs(times - t0) <= args.half
        if m.sum() >= 3 and oov[m].mean() >= args.thresh:
            in_quote[i] = True
    print(
        f"quote gate: {in_quote.sum()}/{len(mix_seq)} mix words masked "
        f"({in_quote.mean():.1%}) at thresh={args.thresh} half={args.half}s",
        file=sys.stderr,
    )

    raw_tracks = _yaml.safe_load(GT_YAML.read_text()).get("tracks", [])
    gt_ss = {
        str(r["slot_label"]): float(r["set_start_s"])
        for r in raw_tracks
        if r.get("claimed_stem") == "acappella" and r.get("set_start_s") is not None
    }
    manifest_dur = max(
        (float(t.get("set_end_s") or 0) for t in raw_tracks), default=0.0
    )
    max_slot = (
        max(
            (
                _slot_order(t["slot_label"])[0]
                for t in raw_tracks
                if t.get("slot_label")
            ),
            default=1,
        )
        or 1
    )

    base = _anchors(
        _plain_bigram_times(mix_seq), raw_tracks, by_tid, manifest_dur, max_slot
    )
    gated = _anchors(
        _gated_bigram_times(mix_seq, in_quote),
        raw_tracks,
        by_tid,
        manifest_dur,
        max_slot,
    )

    rows = []
    for slot, gt in gt_ss.items():
        if slot not in base and slot not in gated:
            continue
        eb = abs(base[slot] - gt) if slot in base else None
        eg = abs(gated[slot] - gt) if slot in gated else None
        rows.append((slot, gt, eb, eg))

    def med(vals):
        v = [x for x in vals if x is not None]
        return np.median(v) if v else float("nan")

    eb_all = [r[2] for r in rows]
    eg_all = [r[3] for r in rows]
    print(f"\n=== quote-gate A/B — {len(rows)} acappella spans (BB12) ===")
    print(
        f"  baseline placement err: median {med(eb_all):5.1f}s  "
        f"tail(>30s) {sum(1 for e in eb_all if e and e > 30):2d}"
    )
    print(
        f"  gated    placement err: median {med(eg_all):5.1f}s  "
        f"tail(>30s) {sum(1 for e in eg_all if e and e > 30):2d}"
    )

    print("\n  per-span changes (|Δ|>3s):")
    for slot, gt, eb, eg in sorted(rows, key=lambda r: -((r[2] or 0) - (r[3] or 0))):
        if eb is None or eg is None:
            print(
                f"    {slot:6} gt {gt:7.1f}  base {str(eb):>7} -> gated {str(eg):>7}"
                f"  (abstain change)"
            )
            continue
        if abs(eb - eg) > 3:
            arrow = "IMPROVE" if eg < eb else "REGRESS"
            print(
                f"    {slot:6} gt {gt:7.1f}  base {eb:6.1f}s -> gated {eg:6.1f}s  "
                f"{arrow} ({eb - eg:+.0f}s)"
            )

    out = {
        "thresh": args.thresh,
        "half": args.half,
        "n": len(rows),
        "base_median": float(med(eb_all)),
        "gated_median": float(med(eg_all)),
        "base_tail": sum(1 for e in eb_all if e and e > 30),
        "gated_tail": sum(1 for e in eg_all if e and e > 30),
    }
    op = Path(__file__).resolve().parent / "out" / "quote_gate_ab.json"
    op.parent.mkdir(exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

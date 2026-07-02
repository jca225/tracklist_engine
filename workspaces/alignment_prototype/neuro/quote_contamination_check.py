"""Does the lyrics-placement error correlate with QUOTE-heavy mix windows?

Big Bootie mixes are stuffed with spoken movie/comedy/TV quotes that are NOT in
the tracklist and NOT in GT, but ARE transcribed by Whisper into the mix-vocal
stream the lyrics channel matches against. Hypothesis: the confidently-wrong
lyrics-placement tail (Drake -232s, Chromeo -43s) sits in windows dense with
spoken quotes.

Method (read-only, no production edits):
  1. Build a SONG vocabulary = union of normalized words across every track's
     cached vocals transcript (acappella AND regular). A mix word absent from
     this vocab is an "unexplained" word — a quote / dialogue / Whisper-noise
     token.
  2. quote-density(t, W) = fraction of mix-vocal words in [t, t+W] that are
     unexplained.
  3. For each acappella span: lyrics placement error e = |lyr_ss - gt_ss|, and
     quote-density at the GT window and at the lyrics-PREDICTED window.
  4. Test: Spearman(e, quote-density) + tail-vs-good density contrast.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.neuro.quote_contamination_check
"""

from __future__ import annotations

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
    load_cached,
    transcribe_words,
)
from workspaces.alignment_prototype.neuro.lyrics_segment_eval import _lyrics_anchors
from workspaces.alignment_prototype.path_decode import find_aligning_dir

GT_YAML = _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"
SET_ID = "1fsnxchk"
WIN_S = 12.0  # matched-filter window width used by the lyrics channel


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    set_dir = find_aligning_dir(SET_ID)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    # --- 1. song vocabulary from every cached vocals transcript -------------
    song_vocab: set[str] = set()
    n_ref = 0
    for t in manifest["tracks"]:
        voc = (t.get("stems") or {}).get("vocals")
        if not voc or not Path(voc).is_file():
            continue
        cw = load_cached(voc)
        if not cw:
            continue
        n_ref += 1
        for w, _ in _norm(cw):
            song_vocab.add(w)
    print(
        f"song vocab: {len(song_vocab)} tokens from {n_ref} cached ref transcripts",
        file=sys.stderr,
    )

    # --- 2. mix stream + per-word quote flag --------------------------------
    mix_seq = _norm(transcribe_words(set_dir / "mix_vocals.flac"))
    times = np.array([tm for _, tm in mix_seq], dtype=np.float64)
    is_quote = np.array([w not in song_vocab for w, _ in mix_seq], dtype=bool)
    frac_quote = is_quote.mean() if len(is_quote) else 0.0
    print(
        f"mix vocal words: {len(mix_seq)}; unexplained (quote) fraction overall: "
        f"{frac_quote:.1%}",
        file=sys.stderr,
    )

    def qdensity(t0: float, w: float = WIN_S) -> float:
        m = (times >= t0) & (times < t0 + w)
        return float(is_quote[m].mean()) if m.any() else float("nan")

    # --- 3. lyrics anchors + GT set_start -----------------------------------
    raw_tracks = _yaml.safe_load(GT_YAML.read_text()).get("tracks", [])
    gt_ss = {
        str(r.get("slot_label")): float(r.get("set_start_s"))
        for r in raw_tracks
        if r.get("claimed_stem") == "acappella" and r.get("set_start_s") is not None
    }
    anchors = _lyrics_anchors(set_dir, raw_tracks, by_tid)  # slot -> (lyr_ss, lyr_rs)

    rows = []  # (slot, err, q_gt, q_pred, lyr_ss, gt)
    for slot, (lyr_ss, _rs) in anchors.items():
        if slot not in gt_ss or lyr_ss is None:
            continue
        gt = gt_ss[slot]
        err = abs(lyr_ss - gt)
        rows.append((slot, err, qdensity(gt), qdensity(lyr_ss), lyr_ss, gt))

    if len(rows) < 3:
        print("too few spans", file=sys.stderr)
        return 1

    err = np.array([r[1] for r in rows])
    q_gt = np.array([r[2] for r in rows])
    q_pred = np.array([r[3] for r in rows])
    ok = ~np.isnan(q_pred)

    print(f"\n=== quote-contamination check — {len(rows)} acappella spans (BB12) ===")
    print(f"  overall mix-vocal quote fraction: {frac_quote:.1%}")
    print(
        f"  Spearman(err, quote-density @ PREDICTED window): "
        f"{_spearman(err[ok], q_pred[ok]):+.2f}"
    )
    print(
        f"  Spearman(err, quote-density @ GT window):        "
        f"{_spearman(err, q_gt):+.2f}"
    )

    tail = err > 30.0
    good = err < 5.0
    print(f"\n  quote-density @ predicted window:")
    print(
        f"    confidently-wrong tail (err>30s, n={tail.sum():2d}): "
        f"mean {np.nanmean(q_pred[tail]) if tail.any() else float('nan'):.2f}"
    )
    print(
        f"    good placements   (err<5s,  n={good.sum():2d}): "
        f"mean {np.nanmean(q_pred[good]) if good.any() else float('nan'):.2f}"
    )

    print(f"\n  worst spans (by placement error):")
    for slot, e, qg, qp, lss, gt in sorted(rows, key=lambda r: -r[1])[:8]:
        print(
            f"    {slot:6} err {e:6.1f}s  q@pred {qp:.2f} q@gt {qg:.2f}  "
            f"(lyr {lss:6.1f}s / gt {gt:6.1f}s)"
        )

    out = {
        "n": len(rows),
        "overall_quote_fraction": frac_quote,
        "spearman_err_qpred": _spearman(err[ok], q_pred[ok]),
        "spearman_err_qgt": _spearman(err, q_gt),
        "tail_qpred_mean": float(np.nanmean(q_pred[tail])) if tail.any() else None,
        "good_qpred_mean": float(np.nanmean(q_pred[good])) if good.any() else None,
    }
    op = Path(__file__).resolve().parent / "out" / "quote_contamination.json"
    op.parent.mkdir(exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

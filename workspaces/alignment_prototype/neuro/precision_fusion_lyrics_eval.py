"""WS1 (revised) — 3-channel precision fusion INCLUDING the lyrics channel.

The first WS1 eval fused only chroma + HuBERT and tied the hand-set HuBERT
priority. That omitted the strongest acappella channel — lyrics-ASR
([[project_lyrics_alignment_channel]], already wired in infer.py via
--lyrics-placement). This eval adds lyrics as a third ref_start channel and asks
the question that actually matters:

    Does precision-weighted fusion of {chroma, hubert, lyrics} beat the CURRENT
    production rule (lyrics places, HuBERT fills abstentions)?

This is the case precision-fusion was built for: Whisper deletes stacked vocals
and the lyrics channel deliberately abstains on dense-overlap "w" rows, so its
per-span reliability genuinely varies — exactly what an inverse-variance arbiter
should exploit, rather than a fixed channel order.

Axis = ref_start (which part of the song), the apples-to-apples metric with the
first eval. Lyrics ref_start comes from its own monotonic decode (tracklist order
+ position prior); chroma/hubert from the per-span matched filter. GT from the
YAML; cached transcriptions (no GPU). Read-only; no harness/infer edits.

Cross-channel precisions live on different scales (MF prominence vs lyric IDF
score-margin), so each channel's precision is STANDARDIZED within-channel before
the cross-channel argmax — that is the calibration the raw arbiter lacked.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.neuro.precision_fusion_lyrics_eval \
        [--tol-s 5] [--max-win-s 15]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.ground_truth import schema
from workspaces.alignment_prototype.lyrics_align import (
    _bigram_times,
    _norm,
    _slot_order,
    candidate_diagonals,
    load_cached,
    monotonic_decode,
    transcribe_words,
)
from workspaces.alignment_prototype.neuro.precision import (
    SR,
    detect_offset_curve,
    precision_from_curve,
)
from workspaces.alignment_prototype.refine_ref_offsets import STRETCHES, chroma

SET_DIR = Path.home() / "aligning" / "1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
GT_YAML = _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"


def _load(
    path: Path, *, offset: float = 0.0, duration: float | None = None
) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(
            str(path), sr=SR, mono=True, offset=offset, duration=duration
        )
    return y


def _hub(y: np.ndarray, layer: int) -> np.ndarray:
    from workspaces.section_hsmm.similarity_probe import _hubert

    return _hubert(y, layer)


def _pct(errs: list[float], thr: float) -> float:
    return 100.0 * sum(1 for e in errs if e <= thr) / len(errs) if errs else 0.0


def _zstd(vals: list[float | None]) -> list[float | None]:
    """Standardize present values within-channel (None passes through)."""
    present = [v for v in vals if v is not None]
    if len(present) < 2:
        return [0.0 if v is not None else None for v in vals]
    mu, sd = float(np.mean(present)), float(np.std(present)) + 1e-9
    return [None if v is None else (v - mu) / sd for v in vals]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tol-s", type=float, default=5.0)
    p.add_argument("--max-win-s", type=float, default=15.0)
    p.add_argument("--min-win-s", type=float, default=4.0)
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument(
        "--abstain-z",
        type=float,
        default=-1.0,
        help="fusion abstains if best channel's standardized precision < this",
    )
    args = p.parse_args(argv)

    r = schema.load(GT_YAML)
    if not r.is_ok():
        print(f"GT load failed: {r.error}", file=sys.stderr)
        return 2
    gt = r.value
    manifest = json.loads((SET_DIR / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    mix_dur = float(manifest.get("mix_duration_s") or 0.0) or max(
        t.set_end_s for t in gt.tracks
    )
    max_slot = (
        max(
            (_slot_order(t.slot_label)[0] for t in gt.tracks if t.slot_label), default=1
        )
        or 1
    )

    # --- lyrics channel: build spans (cached) + monotonic decode -------------
    print("lyrics: transcribing mix_vocals (cached)…")
    mix_bt = _bigram_times(_norm(transcribe_words(SET_DIR / "mix_vocals.flac")))

    aca = [t for t in gt.tracks if t.claimed_stem == "acappella"]
    aca.sort(key=lambda t: _slot_order(t.slot_label))

    rows: list[dict] = []  # per usable span
    lyr_spans = []  # (cands, epos) for monotonic_decode
    for t in aca:
        dur = min(args.max_win_s, t.set_end_s - t.set_start_s)
        if dur < args.min_win_s:
            continue
        mt = by_tid.get(t.track_id)
        voc = (mt.get("stems") or {}).get("vocals") if mt else None
        if not voc or not Path(voc).is_file():
            continue
        cw = load_cached(voc)
        if not cw:
            continue
        cands = candidate_diagonals(_norm(cw), mix_bt)
        epos = _slot_order(t.slot_label)[0] / max_slot * mix_dur
        rows.append(
            {"t": t, "voc": voc, "dur": dur, "cands": cands, "gt": t.ref_start_s}
        )
        lyr_spans.append((cands, epos))

    chosen = monotonic_decode(lyr_spans)  # [(set_start, ref_start)] aligned to rows

    # --- chroma + hubert matched filters per span ----------------------------
    print(f"chroma+hubert matched filter on {len(rows)} spans…")
    mixy_v = _load(SET_DIR / "mix_vocals.flac")
    ref_cache: dict[tuple[str, str], np.ndarray] = {}
    for i, row in enumerate(rows):
        t, voc, dur = row["t"], row["voc"], row["dur"]
        i0 = int(t.set_start_s * SR)
        mw = mixy_v[i0 : i0 + int(dur * SR)]
        for f in ("chroma", "hubert"):
            rf = ref_cache.get((t.track_id, f))
            if rf is None:
                ry = _load(Path(voc))
                rf = chroma(ry) if f == "chroma" else _hub(ry, args.hubert_layer)
                ref_cache[(t.track_id, f)] = rf
            wf = chroma(mw) if f == "chroma" else _hub(mw, args.hubert_layer)
            if wf.shape[1] >= rf.shape[1]:
                row[f] = None
                continue
            pred, peak, st, curve = detect_offset_curve(wf, rf, STRETCHES)
            if curve.size < 4:
                row[f] = None
                continue
            row[f] = (pred, precision_from_curve(curve).prominence)
        # lyrics prediction + precision (score margin of chosen diagonal)
        ss_l, rs_l = chosen[i]
        if rs_l is None or not row["cands"]:
            row["lyrics"] = None
        else:
            scores = sorted((c[2] for c in row["cands"]), reverse=True)
            # margin of top diagonal over runner-up (0 if single) — lyric precision
            margin = scores[0] - (scores[1] if len(scores) > 1 else 0.0)
            row["lyrics"] = (rs_l, scores[0] + margin)

    # --- standardize each channel's precision within-channel -----------------
    chans = ("chroma", "hubert", "lyrics")
    std_prec: dict[str, list[float | None]] = {}
    for ch in chans:
        std_prec[ch] = _zstd([row[ch][1] if row.get(ch) else None for row in rows])

    # per-channel raw error table
    print(f"\n=== {len(rows)} acappella spans (ref_start) ===")
    print(f"{'channel':9}{'n':>5}{'median':>8}{'<2s%':>7}{'<5s%':>7}{'<15s%':>7}")
    for ch in chans:
        e = [abs(row[ch][0] - row["gt"]) for row in rows if row.get(ch)]
        if e:
            print(
                f"{ch:9}{len(e):5d}{np.median(e):8.1f}"
                f"{_pct(e, 2):7.0f}{_pct(e, 5):7.0f}{_pct(e, 15):7.0f}"
            )

    # --- strategies ----------------------------------------------------------
    def err_of(ch, row):
        return abs(row[ch][0] - row["gt"]) if row.get(ch) else None

    strat: dict[str, list[float]] = {
        "lyrics-only": [],
        "lyrics->hubert (prod)": [],
        "raw-peak(3)": [],
        "precision-fusion(3)": [],
        "oracle(3)": [],
    }
    abstain_ct = 0
    for i, row in enumerate(rows):
        present = [ch for ch in chans if row.get(ch)]
        if not present:
            continue
        # lyrics-only (abstains -> skip from its list)
        if row.get("lyrics"):
            strat["lyrics-only"].append(err_of("lyrics", row))
        # production: lyrics if present else hubert else chroma
        for ch in ("lyrics", "hubert", "chroma"):
            if row.get(ch):
                strat["lyrics->hubert (prod)"].append(err_of(ch, row))
                break
        # raw-peak: max raw precision (NOT standardized) across channels
        rp = max(present, key=lambda ch: row[ch][1])
        strat["raw-peak(3)"].append(err_of(rp, row))
        # precision fusion: max standardized precision, with abstain floor
        zbest = max(present, key=lambda ch: std_prec[ch][i])
        if std_prec[zbest][i] < args.abstain_z:
            abstain_ct += 1
        else:
            strat["precision-fusion(3)"].append(err_of(zbest, row))
        # oracle
        strat["oracle(3)"].append(min(err_of(ch, row) for ch in present))

    print(
        f"\n--- strategies (ref_start err vs GT; precision-fusion abstained "
        f"{abstain_ct}/{len(rows)}) ---"
    )
    print(f"{'strategy':24}{'n':>5}{'median':>8}{'<2s%':>7}{'<5s%':>7}{'<15s%':>7}")
    for name, e in strat.items():
        if e:
            print(
                f"{name:24}{len(e):5d}{np.median(e):8.1f}"
                f"{_pct(e, 2):7.0f}{_pct(e, 5):7.0f}{_pct(e, 15):7.0f}"
            )

    out_path = Path(__file__).resolve().parent / "out" / "ws1_lyrics_fusion_eval.json"
    out_path.write_text(
        json.dumps(
            {
                "n": len(rows),
                "abstain": abstain_ct,
                "median": {
                    k: (float(np.median(v)) if v else None) for k, v in strat.items()
                },
                "lt5": {k: _pct(v, 5) for k, v in strat.items()},
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

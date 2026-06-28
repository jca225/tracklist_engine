"""Acappella ref-offset eval on a local aligning folder (no corpus / recording_id needed).

For each acappella span in a session's `.als`, measure how well a matched-filter
recovers the GT ``ref_start`` (which part of the candidate acappella plays in the
mix) using ``mix_vocals.flac`` vs the candidate acappella file directly — the
weak-spot test for vocal placement. Compares chroma vs HuBERT-L9 features on the
*same* `detect_offset` harness (identical stretch search, ref_start = k*HOP/SR),
so the only variable is the feature.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.acappella_ref_offset_eval \
        --als "$HOME/aligning/<set>/<proj>/<name>.als" \
        --set-dir "$HOME/aligning/<set>"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.export_als_to_gt import collect_kept_clip_rows
from workspaces.alignment_prototype.refine_ref_offsets import (
    HOP,
    SR,
    STRETCHES,
    chroma,
    detect_offset,
)
from workspaces.section_hsmm.similarity_probe import _hubert

FEATURES = ("chroma", "hubert")


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


def _resolve(path: str, als: Path, set_dir: Path) -> Path | None:
    for cand in (Path(path), als.parent / path, set_dir / path):
        if cand.is_file():
            return cand
    return None


def _feat(name: str, y: np.ndarray, layer: int) -> np.ndarray:
    return chroma(y) if name == "chroma" else _hubert(y, layer)


def _pct(errs: list[float], thr: float) -> float:
    return 100.0 * sum(1 for e in errs if e <= thr) / len(errs) if errs else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--als", type=Path, required=True)
    p.add_argument("--set-dir", type=Path, required=True)
    p.add_argument("--max-win-s", type=float, default=15.0)
    p.add_argument("--min-win-s", type=float, default=4.0)
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument(
        "--single-stretch",
        action="store_true",
        help="fix stretch=1.0 (isolate warp effect)",
    )
    p.add_argument("--limit", type=int, default=0, help="cap spans (0=all)")
    args = p.parse_args(argv)

    set_id, rows, _ = collect_kept_clip_rows(args.als, args.set_dir)
    aca = [r for r in rows if r.claimed_stem == "acappella"]
    mix_vocals = args.set_dir / "mix_vocals.flac"
    if not mix_vocals.is_file():
        print(f"missing {mix_vocals}", file=sys.stderr)
        return 2
    print(f"set {set_id}: {len(aca)} acappella spans; loading mix_vocals...")
    mixy = _load(mix_vocals)
    stretches = (1.0,) if args.single_stretch else STRETCHES

    per: dict[str, list[float]] = {f: [] for f in FEATURES}
    detail: list[tuple] = []
    skipped = 0
    n = 0
    for r in aca:
        if args.limit and n >= args.limit:
            break
        dur = min(args.max_win_s, r.set_end_s - r.set_start_s)
        if dur < args.min_win_s:
            skipped += 1
            continue
        ref_path = _resolve(r.clip.path, args.als, args.set_dir)
        if ref_path is None:
            skipped += 1
            continue
        i0 = int(r.set_start_s * SR)
        mw = mixy[i0 : i0 + int(dur * SR)]
        ry = _load(ref_path)
        if ry.size < int(dur * SR):
            skipped += 1
            continue
        gt = r.ref_start_s
        row_out = {}
        for f in FEATURES:
            wf, rf = _feat(f, mw, args.hubert_layer), _feat(f, ry, args.hubert_layer)
            pred, peak, st = detect_offset(wf, rf, stretches)
            err = abs(pred - gt)
            per[f].append(err)
            row_out[f] = (pred, peak, err)
        detail.append((r.slot_label, gt, row_out, r.display[:34]))
        n += 1
        print(
            f"  {r.slot_label:8} gt={gt:6.1f}  "
            + "  ".join(
                f"{f}=pred {row_out[f][0]:6.1f} err {row_out[f][2]:5.1f} (pk{row_out[f][1]:.2f})"
                for f in FEATURES
            )
            + f"  | {r.display[:30]}"
        )

    print(f"\n=== acappella ref-offset: {n} spans scored, {skipped} skipped ===")
    print(
        f"{'feature':8} {'median':>8} {'mean':>8} {'<2s%':>7} {'<5s%':>7} {'<10s%':>7}"
    )
    for f in FEATURES:
        e = per[f]
        if not e:
            continue
        print(
            f"{f:8} {np.median(e):8.1f} {np.mean(e):8.1f} "
            f"{_pct(e, 2):7.0f} {_pct(e, 5):7.0f} {_pct(e, 10):7.0f}"
        )
    # head-to-head
    if per["chroma"] and per["hubert"]:
        wins = {"chroma": 0, "hubert": 0, "tie": 0}
        for _, _, ro, _ in detail:
            ce, he = ro["chroma"][2], ro["hubert"][2]
            wins[
                "tie" if abs(ce - he) < 0.5 else ("chroma" if ce < he else "hubert")
            ] += 1
        print(f"\nhead-to-head (err<0.5s=tie): {wins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

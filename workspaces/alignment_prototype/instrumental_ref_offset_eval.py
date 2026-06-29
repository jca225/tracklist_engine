"""Instrumental ref-offset eval on a local aligning folder (no corpus / recording_id).

The instrumental mirror of ``acappella_ref_offset_eval``. For each instrumental
span in a session's ``.als``, measure how well we recover the GT ``ref_start``
(which part of the instrumental plays in the mix) from ``mix_instrumental.flac``
vs the candidate instrumental file (a Demucs stem or a downloaded instrumental).

The point: chroma matched-filter is the known *0%* baseline on instrumental
(the separated mix_instrumental carries other layered tracks → chroma locks onto
the wrong content). This compares chroma against a **landmark fingerprint** on the
SAME spans — fingerprint votes are sharp enough to find the right diagonal despite
the crosstalk. If fp clears chroma here, it justifies wiring a stem-fp instrumental
channel (Phase 2 of docs/stem_routing_plan.md); if not, instrumental needs a
beat-grid signal instead.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.instrumental_ref_offset_eval \
        --als "$HOME/aligning/<set>/<name>.als" \
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
from workspaces.alignment_prototype.landmark_fp import fp_offset
from workspaces.alignment_prototype.refine_ref_offsets import (
    SR,
    STRETCHES,
    chroma,
    detect_offset,
)


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


def _pct(errs: list[float], thr: float) -> float:
    return 100.0 * sum(1 for e in errs if e <= thr) / len(errs) if errs else 0.0


def _chroma_offset(mix_w: np.ndarray, ref_y: np.ndarray, stretches) -> float:
    """ref_start_s via the same matched-filter harness as the acappella eval."""
    wf, rf = chroma(mix_w), chroma(ref_y)
    pred, _peak, _st = detect_offset(wf, rf, stretches)
    return float(pred)


def _fp_offset(
    mix_w: np.ndarray, ref_y: np.ndarray, stretches
) -> tuple[float, int, float]:
    """ref_start_s, votes, sharpness via landmark fingerprint on the raw waveforms.

    fp_offset hashes at FP_SR internally; pass SR-rate audio and let it resample.
    """
    ref_start_s, votes, _st, sharp = fp_offset(mix_w, ref_y, stretches=tuple(stretches))
    return float(ref_start_s), int(votes), float(sharp)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--als", type=Path, required=True)
    p.add_argument("--set-dir", type=Path, required=True)
    p.add_argument(
        "--win-s",
        type=float,
        default=20.0,
        help="mix window length (fp likes more context than the 15s acappella eval)",
    )
    p.add_argument("--min-win-s", type=float, default=4.0)
    p.add_argument(
        "--single-stretch",
        action="store_true",
        help="fix stretch=1.0 (isolate warp effect)",
    )
    p.add_argument("--limit", type=int, default=0, help="cap spans (0=all)")
    args = p.parse_args(argv)

    set_id, rows, _ = collect_kept_clip_rows(args.als, args.set_dir)
    inst = [r for r in rows if r.claimed_stem == "instrumental"]
    mix_instr = args.set_dir / "mix_instrumental.flac"
    if not mix_instr.is_file():
        print(f"missing {mix_instr}", file=sys.stderr)
        return 2
    print(f"set {set_id}: {len(inst)} instrumental spans; loading mix_instrumental...")
    mixy = _load(mix_instr)
    stretches = (1.0,) if args.single_stretch else STRETCHES

    chroma_err: list[float] = []
    fp_err: list[float] = []
    fp_votes: list[int] = []
    skipped = 0
    n = 0
    for r in inst:
        if args.limit and n >= args.limit:
            break
        dur = min(args.win_s, r.set_end_s - r.set_start_s)
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
        if ry.size < int(dur * SR) or mw.size < int(args.min_win_s * SR):
            skipped += 1
            continue
        gt = r.ref_start_s

        c_pred = _chroma_offset(mw, ry, stretches)
        f_pred, votes, sharp = _fp_offset(mw, ry, stretches)
        ce, fe = abs(c_pred - gt), abs(f_pred - gt)
        chroma_err.append(ce)
        fp_err.append(fe)
        fp_votes.append(votes)
        n += 1
        print(
            f"  {r.slot_label:8} gt={gt:6.1f}  "
            f"chroma=pred {c_pred:6.1f} err {ce:5.1f}   "
            f"fp=pred {f_pred:6.1f} err {fe:5.1f} (v{votes:4d} s{sharp:.1f})  "
            f"| {r.display[:30]}"
        )

    print(f"\n=== instrumental ref-offset: {n} spans scored, {skipped} skipped ===")
    print(
        f"{'feature':8} {'median':>8} {'mean':>8} {'<2s%':>7} {'<5s%':>7} {'<10s%':>7}"
    )
    for name, e in (("chroma", chroma_err), ("fp", fp_err)):
        if not e:
            continue
        print(
            f"{name:8} {np.median(e):8.1f} {np.mean(e):8.1f} "
            f"{_pct(e, 2):7.0f} {_pct(e, 5):7.0f} {_pct(e, 10):7.0f}"
        )
    if chroma_err and fp_err:
        wins = {"chroma": 0, "fp": 0, "tie": 0}
        for ce, fe in zip(chroma_err, fp_err):
            wins["tie" if abs(ce - fe) < 0.5 else ("chroma" if ce < fe else "fp")] += 1
        print(f"\nhead-to-head (err<0.5s=tie): {wins}")
        print(
            f"fp votes: median {int(np.median(fp_votes))}  min {min(fp_votes)}  max {max(fp_votes)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

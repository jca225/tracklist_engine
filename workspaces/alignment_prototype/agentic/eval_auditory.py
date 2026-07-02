"""Validation gate: measure an auditory probe's precision against GT.

The probes in ``auditory.py`` are registered ``validated=False`` with provisional
precisions. This module turns the flagship ``onset_align`` into a real
observation on real audio and measures P(correct | fired) vs ground truth —
the one experiment that validates or kills the "missing rank" thesis before we
trust the probe. Regular/instrumental spans only (onset content is strong;
acappella onset structure is weak and belongs to the vocal channel).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.agentic.eval_auditory \
        --set-id 2nvzlh2k --gt labeling/fixtures/bb11_ground_truth.yaml [--max 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.agentic.auditory import ENV_FPS, onset_align
from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir

SR = 22050
HOP = 512  # → ENV_FPS ≈ 43 Hz onset frames


def _onset_env(path: Path) -> np.ndarray | None:
    import librosa

    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    except Exception:
        return None
    if y.size < HOP * 4:
        return None
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    peak = float(env.max())
    return env / peak if peak > 0 else env


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--max", type=int, default=20, help="cap regular spans evaluated")
    p.add_argument(
        "--min-sharp", type=float, default=1.3, help="fire only above this sharpness"
    )
    p.add_argument(
        "--band-s",
        type=float,
        default=0.0,
        help="if >0, band the mix search to GT±band (measures ceiling as a refiner)",
    )
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    if set_dir is None:
        sys.exit(f"no aligning folder for {args.set_id}")
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}

    rows = [
        r
        for r in yaml.safe_load(args.gt.read_text())["tracks"]
        if str(r.get("slot_label")) != "mix"
        and (r.get("claimed_stem") or "regular") in ("regular", "instrumental")
        and r.get("track_id")
    ]
    rows = rows[: args.max]

    print(f"mix onset envelope… ({set_dir.name})")
    mix_env = _onset_env(set_dir / "mix.m4a")
    if mix_env is None:
        sys.exit("no mix.m4a")
    print(f"  mix: {mix_env.size / ENV_FPS / 60:.1f} min of onset frames")

    fired, correct5, correct15, errs = 0, 0, 0, []
    for r in rows:
        row = by_tid.get(str(r["track_id"]))
        ref_path = Path(row["local_path"]) if row and row.get("local_path") else None
        if ref_path is None or not ref_path.is_file():
            continue
        ref_env = _onset_env(ref_path)
        if ref_env is None:
            continue
        offset_s = 0.0
        search_env = mix_env
        if args.band_s > 0:  # band the search around GT (ceiling as a refiner)
            center = r["set_start_s"]
            lo = max(0, int((center - args.band_s) * ENV_FPS))
            hi = min(mix_env.size, int((center + args.band_s) * ENV_FPS) + ref_env.size)
            if hi - lo < ref_env.size:
                continue
            search_env = mix_env[lo:hi]
            offset_s = lo / ENV_FPS
        res = onset_align(search_env, ref_env)
        if res is None or res.sharpness < args.min_sharp:
            continue  # probe abstains on a non-distinct alignment
        fired += 1
        err = abs((res.lag_s + offset_s) - r["set_start_s"])
        errs.append(err)
        correct5 += err <= 5
        correct15 += err <= 15
        flag = "OK " if err <= 15 else "MISS"
        print(
            f"  {flag} {r['slot_label']:5} lag={res.lag_s + offset_s:7.1f} gt={r['set_start_s']:7.1f} "
            f"|Δ|={err:6.1f} peak={res.peak:.2f} sharp={res.sharpness:.1f}"
        )

    print(
        f"\nonset_align fired on {fired}/{len(rows)} regular spans "
        f"(sharpness ≥ {args.min_sharp})"
    )
    if fired:
        e = sorted(errs)
        prec15 = correct15 / fired
        print(
            f"  measured precision (fire→correct): <=15s {prec15:.0%}  <=5s {correct5 / fired:.0%}"
        )
        print(f"  median |Δ| = {e[len(e) // 2]:.1f}s")
        print(
            f"  → registry precision for onset_align should be ~{prec15:.2f} (validated once n is enough)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

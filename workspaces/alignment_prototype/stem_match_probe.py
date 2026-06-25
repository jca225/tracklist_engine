#!/usr/bin/env python3
"""Stem->stem matching robustness probe — the open-lane litmus.

The prior-art table has ONE empty column: vocal/instrumental-aware alignment
(nobody does it). This probe bootstraps that lane WITHOUT a trained model or
clean data, using only locally-cached separated stems. The question is NOT
accuracy on any single example -- it is ROBUSTNESS: how many songs can overlay
on a stem channel before stem-conditioned identity + placement breaks?

Two arms (run either or both):

  synthetic  Build a DIRTY mix channel by summing K real vocal (or
             instrumental) stems at known internal offsets and comparable
             gains -- "very dirty and incorrect", but with PERFECT ground
             truth. For the host stem recover (a) WHERE in its own ref the
             window came from and (b) WHICH ref it is, against a candidate pool
             of clean stems. Sweep K = overlap depth -> the breaking-point
             curve. Compare features (chroma vs mfcc vs hubert).

  real       BB12 mix_vocals vs ref vocal stems on the actual GT acappella
             spans (mix_instrumental vs ref instrumental on instr spans):
             does stem->stem survive REAL separation noise, and does it beat
             the chroma-on-full baseline the CLAUDE.md records as 0-14% exact?

Self-contained: synthetic needs only data/mashup_compat/stems/<taid>/{vocals,
instrumental}.flac (already pulled). No pi, no DB, no model.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.stem_match_probe \
        --arm synthetic --stem vocals --features chroma,mfcc \
        --depths 1,2,3,4,5 --trials 20

    venvs/audio/bin/python -m workspaces.alignment_prototype.stem_match_probe \
        --arm synthetic --stem vocals --features chroma,hubert --trials 8   # GPU
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP,
    SR,
    chroma,
    detect_offset,
)
from workspaces.section_hsmm.similarity_probe import _hubert, _mfcc  # noqa: E402

STEM_ROOT = _REPO / "data" / "mashup_compat" / "stems"
FPS = SR / HOP


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def _feat(y: np.ndarray, name: str, layer: int = 9) -> np.ndarray:
    if name == "chroma":
        return chroma(y)
    if name == "mfcc":
        return _mfcc(y)
    if name == "hubert":
        return _hubert(y, layer)
    raise ValueError(f"unknown feature {name}")


def _load(path: Path) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


class FeatCache:
    """Cache whole-stem feature series per (path, feature) -- reused across trials."""

    def __init__(self, feature: str, layer: int) -> None:
        self.feature = feature
        self.layer = layer
        self._audio: dict[Path, np.ndarray] = {}
        self._feat: dict[Path, np.ndarray] = {}

    def audio(self, path: Path) -> np.ndarray:
        if path not in self._audio:
            self._audio[path] = _load(path)
        return self._audio[path]

    def feat(self, path: Path) -> np.ndarray:
        if path not in self._feat:
            self._feat[path] = _feat(self.audio(path), self.feature, self.layer)
        return self._feat[path]


# --------------------------------------------------------------------------- #
# synthetic arm
# --------------------------------------------------------------------------- #
def _stem_paths(stem: str) -> list[Path]:
    fname = f"{stem}.flac"
    return sorted(p / fname for p in STEM_ROOT.iterdir() if (p / fname).is_file())


def _add_source(
    win: np.ndarray, src: np.ndarray, shift: int, internal0: int, gain: float
) -> None:
    """Add gain*src into the window buffer `win` (len Wn) at sample `shift`.

    shift<0 => source started before the window; internal0 = where in the source
    its placed segment begins. Mutates win in place (the physical superposition)."""
    wn = win.shape[0]
    w_lo = max(0, shift)
    w_hi = wn
    s_lo = internal0 + max(0, -shift)
    seg = min(w_hi - w_lo, src.shape[0] - s_lo)
    if seg <= 0:
        return
    win[w_lo : w_lo + seg] += gain * src[s_lo : s_lo + seg]


@dataclass
class TrialResult:
    depth: int
    feature: str
    id_correct: bool
    margin: float  # host peak - best distractor peak
    place_err_s: float  # |recovered offset - true offset| (only if id_correct)


def _one_trial(
    rng: np.random.Generator,
    pool: list[Path],
    cache: FeatCache,
    depth: int,
    extra_distractors: int,
    window_s: float,
    gain_lo: float,
) -> TrialResult | None:
    wn = int(window_s * SR)
    chosen = list(rng.choice(len(pool), size=min(depth, len(pool)), replace=False))
    host_path = pool[chosen[0]]
    host = cache.audio(host_path)
    if host.shape[0] < wn + int(15 * SR):
        return None  # too short to take a mid-song window

    # host window: start somewhere non-trivial inside the host stem
    o0 = int(rng.integers(int(10 * SR), host.shape[0] - wn - int(5 * SR)))
    win = np.zeros(wn, dtype=np.float32)
    _add_source(win, host, shift=0, internal0=o0, gain=1.0)

    # overlay the other (depth-1) stems with comparable gain, forced to overlap
    for ci in chosen[1:]:
        d = cache.audio(pool[ci])
        if d.shape[0] < int(20 * SR):
            continue
        shift = int(rng.integers(-wn, wn))  # partial overlap
        idur = max(1, d.shape[0] - int(5 * SR))
        internal0 = int(rng.integers(0, idur))
        gain = float(rng.uniform(gain_lo, 1.0))
        _add_source(win, d, shift, internal0, gain)

    win_f = _feat(win, cache.feature, cache.layer)

    # candidate pool = in-mix confusers + extra clean stems not in the mix
    extra = [
        i
        for i in rng.choice(
            len(pool), size=min(extra_distractors + depth, len(pool)), replace=False
        )
        if i not in chosen
    ][:extra_distractors]
    candidates = chosen + extra
    host_off, host_peak, _ = detect_offset(win_f, cache.feat(host_path))
    best_other = 0.0
    for ci in candidates[1:]:
        _, peak, _ = detect_offset(win_f, cache.feat(pool[ci]))
        best_other = max(best_other, peak)

    id_correct = host_peak > best_other
    return TrialResult(
        depth=depth,
        feature=cache.feature,
        id_correct=id_correct,
        margin=host_peak - best_other,
        place_err_s=abs(host_off - o0 / SR) if id_correct else float("nan"),
    )


def run_synthetic(args: argparse.Namespace) -> int:
    pool = _stem_paths(args.stem)
    if len(pool) < 3:
        sys.exit(f"need >=3 local {args.stem} stems in {STEM_ROOT}, found {len(pool)}")
    print(
        f"synthetic arm: {len(pool)} {args.stem} stems | "
        f"depths={args.depths} trials={args.trials} window={args.window_s}s "
        f"features={args.features}\n"
    )

    rows: list[TrialResult] = []
    for feature in args.features:
        cache = FeatCache(feature, args.hubert_layer)
        rng = np.random.default_rng(args.seed)
        for depth in args.depths:
            got = 0
            tries = 0
            while got < args.trials and tries < args.trials * 4:
                tries += 1
                r = _one_trial(
                    rng,
                    pool,
                    cache,
                    depth,
                    args.extra_distractors,
                    args.window_s,
                    args.gain_lo,
                )
                if r is not None:
                    rows.append(r)
                    got += 1
            print(f"  {feature:7s} depth={depth} done ({got} trials)")

    print("\n=== breaking-point curve (host = 1 stem + (depth-1) overlaid) ===")
    print(
        f"{'feature':8s} {'depth':>5s} {'id@1%':>7s} {'margin':>8s} "
        f"{'place_med_s':>12s} {'place<2s%':>10s}"
    )
    for feature in args.features:
        for depth in args.depths:
            sub = [r for r in rows if r.feature == feature and r.depth == depth]
            if not sub:
                continue
            id_acc = 100.0 * np.mean([r.id_correct for r in sub])
            margin = float(np.median([r.margin for r in sub]))
            errs = np.array(
                [
                    r.place_err_s
                    for r in sub
                    if r.id_correct and np.isfinite(r.place_err_s)
                ]
            )
            pmed = float(np.median(errs)) if errs.size else float("nan")
            pexact = 100.0 * np.mean(errs < 2.0) if errs.size else float("nan")
            print(
                f"{feature:8s} {depth:5d} {id_acc:7.1f} {margin:8.3f} "
                f"{pmed:12.2f} {pexact:10.1f}"
            )
    return 0


# --------------------------------------------------------------------------- #
# real arm (BB12 mix stems vs local ref stems on actual GT spans)
# --------------------------------------------------------------------------- #
def run_real(args: argparse.Namespace) -> int:
    import yaml

    from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir

    gt = yaml.safe_load(Path(args.gt).read_text())
    set_dir = find_aligning_dir(gt["set_id"])
    mix_file = {"vocals": "mix_vocals.flac", "instrumental": "mix_instrumental.flac"}[
        args.stem
    ]
    claimed = {"vocals": "acappella", "instrumental": "instrumental"}[args.stem]
    mix_path = set_dir / mix_file
    if not mix_path.is_file():
        sys.exit(f"no {mix_file} in {set_dir}")

    # recording_id -> local stem dir (taid) via one pi query, cached
    taid_map = _resolve_local_stem_dirs(args.stem)
    spans = [
        t
        for t in gt["tracks"]
        if t.get("claimed_stem") == claimed
        and not t.get("is_loop")
        and t.get("track_id") in taid_map
        and "set_start_s" in t
        and "ref_start_s" in t
    ]
    print(f"real arm ({args.stem}): {len(spans)} GT {claimed} spans with local stems")
    if not spans:
        return 0

    pool = _stem_paths(args.stem)
    cache = FeatCache(args.features[0], args.hubert_layer)
    mix_y = cache.audio(mix_path)
    wn = int(args.window_s * SR)
    rng = np.random.default_rng(args.seed)

    id_ok = 0
    perrs: list[float] = []
    for t in spans:
        s0 = int(float(t["set_start_s"]) * SR)
        win = mix_y[s0 : s0 + wn]
        if win.shape[0] < wn // 2:
            continue
        win_f = _feat(win, cache.feature, cache.layer)
        host_path = taid_map[t["track_id"]]
        host_off, host_peak, _ = detect_offset(win_f, cache.feat(host_path))
        dpool = [p for p in pool if p != host_path]
        dsel = [
            dpool[i]
            for i in rng.choice(
                len(dpool), size=min(args.extra_distractors, len(dpool)), replace=False
            )
        ]
        best_other = max(
            (detect_offset(win_f, cache.feat(p))[1] for p in dsel), default=0.0
        )
        ok = host_peak > best_other
        id_ok += ok
        if ok:
            perrs.append(abs(host_off - float(t["ref_start_s"])))
        print(
            f"  {str(t.get('slot_label', '????')):>4s} {t['track'][:34]:34s} "
            f"id={'OK ' if ok else 'MISS'} peak={host_peak:.3f} "
            f"off={host_off:7.1f}s gt={float(t['ref_start_s']):7.1f}s"
        )

    print(
        f"\nreal {args.stem}: identity {id_ok}/{len(spans)} = {100 * id_ok / len(spans):.0f}%  "
        f"| placement median {np.median(perrs):.1f}s  <2s {100 * np.mean(np.array(perrs) < 2):.0f}% "
        f"(n={len(perrs)})"
        if perrs
        else f"\nreal {args.stem}: identity {id_ok}/{len(spans)}"
    )
    return 0


def _resolve_local_stem_dirs(stem: str) -> dict[str, Path]:
    """recording_id -> local <taid>/<stem>.flac, via one pi query (cached to json)."""
    import json
    import subprocess

    fname = f"{stem}.flac"
    have = {p.name for p in STEM_ROOT.iterdir() if (p / fname).is_file()}
    cache_f = STEM_ROOT.parent / f"taid_map_{stem}.json"
    if cache_f.is_file():
        raw = json.loads(cache_f.read_text())
    else:
        ids = ",".join(f"'{x}'" for x in [])  # resolved below via all-regular query
        sql = (
            "SELECT recording_id, track_audio_id FROM track_audio "
            "WHERE stem='regular' ORDER BY is_reference DESC, track_audio_id;"
        )
        try:
            out = subprocess.run(
                [
                    "ssh",
                    "pi-storage",
                    f'sqlite3 /mnt/storage/data/db/music_database.db "{sql}"',
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout
        except Exception as e:  # noqa: BLE001
            print(f"  (pi resolve failed: {e}; real arm needs the recording->taid map)")
            return {}
        raw = {}
        for line in out.strip().splitlines():
            rid, taid = line.split("|")
            raw.setdefault(rid, taid)
        cache_f.write_text(json.dumps(raw))
    return {rid: STEM_ROOT / taid / fname for rid, taid in raw.items() if taid in have}


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arm", choices=["synthetic", "real", "both"], default="synthetic")
    p.add_argument("--stem", choices=["vocals", "instrumental"], default="vocals")
    p.add_argument(
        "--features", default="chroma,mfcc", help="comma list of chroma,mfcc,hubert"
    )
    p.add_argument("--depths", default="1,2,3,4,5")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--window-s", type=float, default=12.0)
    p.add_argument("--extra-distractors", type=int, default=8)
    p.add_argument("--gain-lo", type=float, default=0.5)
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gt", default=str(_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    )
    args = p.parse_args(argv)
    args.features = [f.strip() for f in args.features.split(",") if f.strip()]
    args.depths = [int(d) for d in str(args.depths).split(",") if d.strip()]

    rc = 0
    if args.arm in ("synthetic", "both"):
        rc |= run_synthetic(args)
    if args.arm in ("real", "both"):
        rc |= run_real(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

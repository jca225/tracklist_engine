#!/usr/bin/env python3
"""nmf_baseline — reference-conditioned NMF for DJ-mix reverse engineering.

A v0 reproduction of the André/Schwarz/Fourer 2024 idea (the current SOTA): model
the mix magnitude spectrogram V as a non-negative sum of the KNOWN source tracks'
spectrograms,  V ≈ [W_1 | W_2 | …] · H,  with W_k = track k's spectrogram (fixed
dictionary) and H the activations over mix time. Unlike our matched-filter/DTW
methods, this models the mix as a SUM of sources — attacking superposition (our
documented root cause) head-on, and the activation matrix yields, per track:

  - set_start  — first mix frame the track's activation turns on (placement)
  - tempo      — slope of the activation ridge (track-frame vs mix-frame) = warp
  - gain       — the activation envelope itself (the fader ride)

v0 = fixed-W KL-NMF, affine warp read from the ridge. v1 (TODO) = multi-pass:
re-warp the dictionary from the ridge and re-solve (handles loops/jumps).

    venvs/audio/bin/python -m workspaces.alignment_prototype.nmf_baseline --synthetic
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

SR = 22050
NMF_FPS = 8.0  # coarse activation grid (0.125 s) — keeps H small & fast
N_MELS = 128
_EPS = 1e-9


@dataclass(frozen=True)
class NmfPred:
    track_idx: int
    set_start_s: float
    tempo_ratio: float
    gain_peak: float
    present: bool


_WARP_GRID = (0.90, 0.95, 1.0, 1.05, 1.10)


def _stretch_cols(W: np.ndarray, s: float) -> np.ndarray:
    tk = W.shape[1]
    mm = max(2, int(round(tk * s)))
    idx = np.clip((np.arange(mm) / s).astype(int), 0, tk - 1)
    return W[:, idx]


def _solve_H(
    V: np.ndarray,
    W: np.ndarray,
    iters: int,
    l1: float = 0.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fixed-W KL-NMF: V≈W·H. W=(F,R) column-normalized, returns H=(R,Tm).
    l1>0 adds an L1 sparsity penalty (term in the denominator) → sharper, less
    smeared activations. mask (R,Tm) confines activation to a region — used by
    the fingerprint-banded variant to enforce temporal continuity (kills the
    cross-talk between spectrally-similar tracks that wrecked the unbanded NMF)."""
    Wn = W / (W.sum(axis=0, keepdims=True) + _EPS)
    V = V.astype(np.float64)
    rng = np.random.default_rng(0)
    H = rng.random((Wn.shape[1], V.shape[1])) + _EPS
    if mask is not None:
        H *= mask
    Wt1 = Wn.T @ np.ones((V.shape[0], V.shape[1]))
    for _ in range(iters):
        WH = Wn @ H + _EPS
        H *= (Wn.T @ (V / WH)) / (Wt1 + l1 + _EPS)
        if mask is not None:
            H *= mask
    return H


def _sustained_onset(presence: np.ndarray, thr: float, k: int = 3) -> int:
    """First frame where presence stays above thr for >=k frames — rejects the
    spurious single-frame blips that wreck a plain argmax onset."""
    run = 0
    for i, a in enumerate(presence > thr):
        run = run + 1 if a else 0
        if run >= k:
            return i - k + 1
    above = np.flatnonzero(presence > thr)
    return int(above[0]) if above.size else 0


def recover(
    V: np.ndarray,
    dicts: dict[int, np.ndarray],
    *,
    fps: float = NMF_FPS,
    iters: int = 60,
    present_frac: float = 0.15,
    warps: tuple[float, ...] = _WARP_GRID,
    l1: float = 0.5,
) -> dict[int, NmfPred]:
    """Reference-conditioned NMF, multi-pass (André-style v1).

    Pass 1 — STRETCHED dictionary: every track enters at several tempo-warps; NMF
    picks which warp activates (= the winning block). Pass 2 — rebuild the
    dictionary with ONLY each track's winning warp and re-solve: removes the
    competing-warp atoms so the surviving activation is sharp (less cross-talk),
    which is what makes the onset/gain readable on real spectra. Sparsity (l1) +
    a sustained-onset rule do the rest."""
    blocks = [(k, s, _stretch_cols(W, s)) for k, W in dicts.items() for s in warps]
    bigW = np.concatenate([b[2] for b in blocks], axis=1).astype(np.float64)
    H = _solve_H(V, bigW, iters, l1)
    best_warp: dict[int, float] = {}
    per: dict[int, list[tuple[float, float]]] = {}
    off = 0
    for k, s, Wks in blocks:
        tk = Wks.shape[1]
        # MEAN activation per atom — size-invariant. Total sum is biased to the
        # largest stretch (more atoms -> bigger sum), which pinned warp at the grid
        # max and corrupted placement.
        per.setdefault(k, []).append((s, float(H[off : off + tk].sum()) / tk))
        off += tk
    for k, lst in per.items():
        best_warp[k] = max(lst, key=lambda z: z[1])[0]

    # Pass 2: winning-warp-only dictionary → clean per-track activation
    wblocks = [(k, _stretch_cols(dicts[k], best_warp[k])) for k in dicts]
    bigW2 = np.concatenate([b[1] for b in wblocks], axis=1).astype(np.float64)
    H2 = _solve_H(V, bigW2, iters, l1)

    out, off = {}, 0
    for k, Wk in wblocks:
        tk = Wk.shape[1]
        Hk = H2[off : off + tk]
        off += tk
        presence = Hk.sum(axis=0)
        pk = float(presence.max())
        thr = present_frac * pk
        present = bool((presence > thr).sum() >= 3)
        start_f = _sustained_onset(presence, thr) if present else 0
        out[k] = NmfPred(
            track_idx=k,
            set_start_s=start_f / fps,
            tempo_ratio=best_warp[k] if present else 1.0,
            gain_peak=pk,
            present=present,
        )
    return out


def recover_banded(
    V: np.ndarray,
    dicts: dict[int, np.ndarray],
    anchors: dict[int, tuple[float, float]],
    *,
    fps: float = NMF_FPS,
    iters: int = 60,
    l1: float = 0.3,
    band_frames: int = 24,
) -> dict[int, np.ndarray]:
    """Fingerprint-banded NMF — our shortcut to André's continuity trick.

    anchors[k] = (set_start_s, stretch) from the fingerprint. We build each
    track's dictionary at its stretch and CONFINE its activation to a band around
    its known diagonal (r ≈ t - start_f, slope 1 after the stretch). That enforces
    temporal continuity for free, so a track can no longer splatter onto a
    spectrally-similar neighbour. Returns per-track GAIN curve (the activation
    summed along the band = the recovered volume/fade envelope) — the editable
    quantity fingerprinting alone can't give."""
    tm = V.shape[1]
    wblocks, masks = [], []
    for k, W in dicts.items():
        s_start, st = anchors[k]
        Wk = _stretch_cols(W, st)
        tk = Wk.shape[1]
        start_f = int(round(s_start * fps))
        r = np.arange(tk)[:, None]
        t = np.arange(tm)[None, :]
        masks.append((np.abs(r - (t - start_f)) <= band_frames).astype(np.float64))
        wblocks.append((k, Wk))
    bigW = np.concatenate([b[1] for b in wblocks], axis=1).astype(np.float64)
    bigMask = np.concatenate(masks, axis=0)
    H = _solve_H(V, bigW, iters, l1, mask=bigMask)
    out, off = {}, 0
    for k, Wk in wblocks:
        tk = Wk.shape[1]
        out[k] = H[off : off + tk].sum(axis=0)  # gain envelope over mix time
        off += tk
    return out


def recover_editable(
    V: np.ndarray,
    dicts: dict[int, np.ndarray],
    anchors: dict[int, tuple[float, float]],
    *,
    fps: float = NMF_FPS,
    iters: int = 60,
    l1: float = 0.3,
    band_frames: int = 24,
    n_bands: int = 8,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The editable trio per track: (gain curve, per-band EQ).

    Banded NMF as in recover_banded, but also keep each track's reconstruction
    R_k = W_k·H_k. In a track's SOLO frames (it plays, neighbours don't), the
    per-band ratio (mix energy / reconstruction energy) is the EQ the DJ applied:
    H carries only broadband gain (it scales whole atom columns), so any per-band
    discrepancy between the mix and the un-EQ'd source model IS the equalization.
    Returns eq as `n_bands` values normalized to median 1 (flat ⇒ no EQ)."""
    tm = V.shape[1]
    wblocks, masks = [], []
    for k, W in dicts.items():
        s_start, st = anchors[k]
        Wk = _stretch_cols(W, st)
        start_f = int(round(s_start * fps))
        r = np.arange(Wk.shape[1])[:, None]
        t = np.arange(tm)[None, :]
        masks.append((np.abs(r - (t - start_f)) <= band_frames).astype(np.float64))
        wblocks.append((k, Wk))
    bigW = np.concatenate([b[1] for b in wblocks], axis=1).astype(np.float64)
    H = _solve_H(V, bigW, iters, l1, mask=np.concatenate(masks, axis=0))

    gains, recons, off = {}, {}, 0
    for k, Wk in wblocks:
        tk = Wk.shape[1]
        Hk = H[off : off + tk]
        off += tk
        gains[k] = Hk.sum(axis=0)
        recons[k] = Wk @ Hk
    edges = np.linspace(0, V.shape[0], n_bands + 1).astype(int)
    out = {}
    for k in dicts:
        gk = gains[k]
        act = gk > 0.2 * (gk.max() + _EPS)
        other = np.zeros_like(gk)
        for j in dicts:
            if j != k:
                other = np.maximum(other, gains[j] / (gains[j].max() + _EPS))
        solo = act & (other < 0.2)
        if solo.sum() < 3:
            solo = act
        mix_b = V[:, solo].sum(axis=1)
        rec_b = recons[k][:, solo].sum(axis=1) + _EPS
        eq_full = mix_b / rec_b
        eq = np.array([eq_full[edges[b] : edges[b + 1]].mean() for b in range(n_bands)])
        out[k] = (gk, eq / (np.median(eq) + _EPS))
    return out


# ----------------------------------------------------------------------------- audio
def mel_spec(y: np.ndarray, fps: float = NMF_FPS) -> np.ndarray:
    import librosa

    hop = int(round(SR / fps))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=hop)
    return S.astype(np.float32)


def recover_audio(
    mix_path: Path, track_paths: dict[int, Path], **kw
) -> dict[int, NmfPred]:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        my, _ = librosa.load(str(mix_path), sr=SR, mono=True)
        V = mel_spec(my)
        dicts = {}
        for k, p in track_paths.items():
            ty, _ = librosa.load(str(p), sr=SR, mono=True)
            dicts[k] = mel_spec(ty)
    return recover(V, dicts, **kw)


# ----------------------------------------------------------------------------- smoke
def _synthetic_case(seed: int = 0):
    """Build spectra directly: 3 distinguishable tracks placed in a mix at known
    (start, stretch). Returns (V, dicts, gt_starts, gt_tempos)."""
    rng = np.random.default_rng(seed)
    F, Tm = N_MELS, 320
    V = rng.random((F, Tm)).astype(np.float64) * 0.05  # noise floor
    dicts, gt_start, gt_tempo = {}, {}, {}
    bands_per = F // 3
    for k in range(3):
        Tk = int(rng.integers(70, 110))
        # each track: DENSE random columns within its own disjoint band-set. Dense
        # random => each column distinguishable (sharp diagonal activation, so warp
        # is readable); disjoint bands => low cross-talk between tracks.
        Wk = np.full((F, Tk), 0.02)
        lo = k * bands_per
        Wk[lo : lo + bands_per, :] = rng.random((bands_per, Tk)) + 0.3
        dicts[k] = Wk.astype(np.float64)
        stretch = float(rng.choice([0.9, 1.0, 1.1]))
        mm = int(round(Tk * stretch))
        start = int(rng.integers(0, max(1, Tm - mm - 1)))
        idx = np.clip((np.arange(mm) / stretch).astype(int), 0, Tk - 1)
        V[:, start : start + mm] += Wk[:, idx]  # warped, placed
        gt_start[k] = start / NMF_FPS
        gt_tempo[k] = stretch
    return V, dicts, gt_start, gt_tempo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--cases", type=int, default=4)
    args = p.parse_args(argv)
    if not args.synthetic:
        p.error("only --synthetic is wired (audio path used via eval_bench)")

    se, te, pres = [], [], []
    for c in range(args.cases):
        V, dicts, gs, gt = _synthetic_case(seed=c)
        pred = recover(V, dicts)
        for k in dicts:
            se.append(abs(pred[k].set_start_s - gs[k]))
            te.append(abs(pred[k].tempo_ratio - gt[k]))
            pres.append(pred[k].present)
    se, te = np.array(se), np.array(te)
    print(f"NMF baseline on {args.cases} synthetic mixes ({len(se)} tracks):")
    print(f"  set_start MAE : {se.mean():.2f}s  (median {np.median(se):.2f}s)")
    print(f"  tempo     MAE : {te.mean():.3f}")
    print(f"  presence      : {100 * np.mean(pres):.0f}% detected")
    print(
        "  (recovers placement+warp+gain from a SUM of sources — superposition-aware)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

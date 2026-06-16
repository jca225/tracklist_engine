#!/usr/bin/env python3
"""Oracle-cancellation feasibility test for joint two-bed decode.

The gate before building the real decoder: during the BB12 Galantis/Calvin
crossfade, slot 111 (Sweet Nothing) plays BURIED at fader gain 0.16 under slot
112 (Galantis You) at gain 1.0, and single-bed decode fails it (0% exact<2s).

If we CANCEL 112 — using its GT placement + gain_curve to synthesize its mel
contribution and subtract it from the mix — does the buried 111 then localize
on the residual where it didn't on the raw mix? This uses ORACLE 112 (its true
placement) and oracle gains: the most favorable setting. If 111 doesn't surface
here, no joint decoder will recover it and we stop.

Mel power-domain spectral subtraction (|A+B|^2 ~ |A|^2+|B|^2), with a global
level scale fit by least squares to bridge the master-vs-mix level gap.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SR = 22050
HOP = 512
N_MELS = 128
FPS = SR / HOP
SETDIR = Path.home() / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12"

# GT placements (from bb12_ground_truth.yaml)
REF112 = SETDIR / "tracks/031w4__Galantis - You.m4a"
REF111 = SETDIR / "tracks/031w3__Calvin Harris - Sweet Nothing (Qulinez Remix).m4a"
# 112 piecewise ref(mix) segments: (mix_start, mix_end, ref_start, ref_end)
SEG112 = [(2528.0, 2591.0, 0.2, 63.9), (2591.0, 2655.0, 63.8, 127.5)]
GAIN112 = [(2527.8, 0.0), (2591.5, 0.0), (2591.6, 0.03), (2604.6, 1.0), (2660.8, 1.0)]
# 111 linear: ref(t) = 105.124 + (t - 2572.8)
REF111_AT = lambda t: 105.124 + (t - 2572.8)  # noqa: E731

REGION = (2605.0, 2655.0)  # buried-111 window, inside 112's segment 2


def mel(path: Path) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        m = librosa.feature.melspectrogram(
            y=y, sr=SR, hop_length=HOP, n_mels=N_MELS, power=2.0
        )
    return m.astype(np.float32)  # (F, T) power


def _interp(curve, t):
    if t <= curve[0][0]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    for (x0, g0), (x1, g1) in zip(curve, curve[1:]):
        if x0 <= t <= x1:
            return g0 if x1 == x0 else g0 + (t - x0) / (x1 - x0) * (g1 - g0)
    return curve[-1][1]


def ref112_sec(t: float) -> float | None:
    for ms, me, rs, re in SEG112:
        if ms <= t <= me:
            return rs + (t - ms) * (re - rs) / (me - ms)
    return None


def matched_filter(
    query: np.ndarray, ref: np.ndarray, stretches
) -> tuple[float, float]:
    """Best (ref_start_sec, peak) of query (F,L) sliding over ref (F,T)."""
    from scipy.signal import fftconvolve

    best = (0.0, -1.0)
    n = query.shape[1]
    for st in stretches:
        m = int(round(n * st))
        if m < 4 or ref.shape[1] <= m:
            continue
        idx = np.clip((np.arange(m) / st).astype(int), 0, n - 1)
        w = query[:, idx]
        w = w / (np.linalg.norm(w) + 1e-9)
        num = fftconvolve(ref, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
        e = np.concatenate([[0.0], np.cumsum((ref**2).sum(axis=0))])
        den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
        sc = num / den
        k = int(sc.argmax())
        if sc[k] > best[1]:
            best = (k / FPS, float(sc[k]))
    return best


def main() -> int:
    print("loading mel spectrograms …", file=sys.stderr)
    mix = mel(SETDIR / "mix.m4a")
    r112 = mel(REF112)
    r111 = mel(REF111)

    a, b = int(REGION[0] * FPS), int(REGION[1] * FPS)
    L = b - a
    mix_q = mix[:, a:b]

    # synthesize 112's mel contribution over the region (oracle placement+gain)
    base = np.zeros_like(mix_q)
    for j in range(L):
        t = (a + j) / FPS
        rs = ref112_sec(t)
        if rs is None:
            continue
        rf = int(round(rs * FPS))
        if 0 <= rf < r112.shape[1]:
            g = _interp(GAIN112, t)
            base[:, j] = (g**2) * r112[:, rf]  # power scales as amplitude^2

    # global level scale bridging master-vs-mix gap (LS projection)
    denom = float((base * base).sum()) + 1e-9
    scale = float((mix_q * base).sum()) / denom
    residual = np.maximum(mix_q - scale * base, 0.0)

    gt_ref111 = REF111_AT(REGION[0])
    stretches = tuple(np.round(np.arange(0.94, 1.07, 0.02), 3))
    raw_off, raw_pk = matched_filter(mix_q, r111, stretches)
    res_off, res_pk = matched_filter(residual, r111, stretches)

    print(f"\n=== oracle-cancellation feasibility (buried 111 under 112) ===")
    print(
        f"  region mix {REGION[0]:.0f}-{REGION[1]:.0f}s  GT ref111 @ region start = {gt_ref111:.1f}s"
    )
    print(
        f"  fitted 112 level scale = {scale:.3f}  (residual energy "
        f"{100 * residual.sum() / (mix_q.sum() + 1e-9):.0f}% of mix)"
    )
    print(
        f"  RAW mix   -> ref111 peak @ {raw_off:6.1f}s  score {raw_pk:.3f}  "
        f"err {abs(raw_off - gt_ref111):6.1f}s"
    )
    print(
        f"  RESIDUAL  -> ref111 peak @ {res_off:6.1f}s  score {res_pk:.3f}  "
        f"err {abs(res_off - gt_ref111):6.1f}s"
    )
    ok = abs(res_off - gt_ref111) < 5.0 and abs(raw_off - gt_ref111) >= 5.0
    print(
        f"\n  VERDICT: {'CANCELLATION REVEALS 111' if ok else 'cancellation FAILED — mel subtraction too imprecise; 111 only 2.6% power'}"
    )

    # ---- Alternative: DOMINANCE-WINDOW decode (no cancellation) ----------
    # 111 is the LOUDER bed only at t=2588-2604 (gain 1.41 vs 112's ~0.1-0.3).
    # Decode 111 on just that window (chroma, matching the real decoder), vs on
    # its full span (mostly buried) which is what single-bed does + fails.
    import librosa

    from workspaces.alignment_prototype.refine_ref_offsets import chroma as _chroma

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ymix, _ = librosa.load(str(SETDIR / "mix.m4a"), sr=SR, mono=True)
        y111, _ = librosa.load(str(REF111), sr=SR, mono=True)
    cmix, c111 = _chroma(ymix), _chroma(y111)
    st = tuple(np.round(np.arange(0.96, 1.05, 0.02), 3))

    def decode_window(t0, t1):
        q = cmix[:, int(t0 * FPS) : int(t1 * FPS)]
        off, pk = matched_filter(q, c111, st)
        return off, pk, REF111_AT(t0)

    print("\n=== dominance-window decode of 111 (chroma) ===")
    for tag, (t0, t1) in [
        ("DOMINANT 2588-2604", (2588.0, 2604.0)),
        ("FULL span 2573-2682", (2573.0, 2682.0)),
        ("BURIED 2605-2655", (2605.0, 2655.0)),
    ]:
        off, pk, gt = decode_window(t0, t1)
        print(
            f"  {tag:22} -> ref111 @ {off:6.1f}s  score {pk:.3f}  "
            f"GT {gt:6.1f}s  err {abs(off - gt):6.1f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

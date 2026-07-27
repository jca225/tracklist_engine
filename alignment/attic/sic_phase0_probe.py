"""Medley SIC Phase 0 — oracle feasibility on BB12 finale (3440-3670s).

Cancel GT-aligned known layers from the mix (per-window offset+gain matched
subtraction, SIC order = loudness); measure (a) energy removed, (b) salience
of the MISSED layers in residual vs mix. Gate: >=3 dB removal AND missed
salience improves.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/Users/johnnycabrahams/Desktop/tracklist_engine")
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from alignment.looptrace.selfsim import load_audio

SR = 22050
BLOCK = (3440.0, 3670.0)
WIN, HOP = int(1.0 * SR), int(0.5 * SR)
PAD = int(0.08 * SR)  # +-80ms per-window offset search

base = Path.home() / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
man = json.loads((base / "manifest.json").read_text())
by_tid = {t["track_id"]: t for t in man["tracks"]}
gt = yaml.safe_load(
    Path(
        "/Users/johnnycabrahams/Desktop/tracklist_engine/labeling/fixtures/bb12_ground_truth.yaml"
    ).read_text()
)
gt_all = [r for r in gt["tracks"] if str(r.get("slot_label")) != "mix"]
# rows overlapping the block, classified via span_table (matched by GT start)
import csv

st = {
    round(float(r["gt_set_start_s"]), 1): r
    for r in csv.DictReader(
        open(
            "/Users/johnnycabrahams/Desktop/tracklist_engine/eda/alignment/failure_analysis/out/span_table.csv"
        )
    )
    if r["set_id"] == "1fsnxchk" and r.get("gt_set_start_s")
}
CANCEL, MISSED = [], []
for r in gt_all:
    s0, s1 = float(r["set_start_s"]), float(r["set_end_s"])
    if s1 <= BLOCK[0] + 5 or s0 >= BLOCK[1] - 5:
        continue
    q = st.get(round(s0, 1))
    key = str(r.get("slot_label"))
    if q and q["identity_hit"] == "1" and float(q["set_start_err_s"] or 999) < 20:
        CANCEL.append((key, r, q["slot"], q["name"][:30]))
    else:
        MISSED.append(
            (
                key,
                r,
                q["slot"] if q else "?",
                (q["name"][:30] if q else r.get("track", "")),
            )
        )
print("cancel set:", [(c[2], c[3]) for c in CANCEL])
print("missed set:", [(m[2], m[3]) for m in MISSED])

_STEM_FILE = {"acappella": "vocals", "instrumental": "instrumental"}


def sources_for(row) -> list[Path]:
    t = by_tid.get(str(row.get("track_id"))) or {}
    out = []
    p = t.get("local_path")
    if p and Path(p).is_file():
        out.append(Path(p))
    sk = _STEM_FILE.get(row.get("claimed_stem") or "regular")
    sp = (t.get("stems") or {}).get(sk) if sk else None
    if sp and Path(sp).is_file():
        out.append(Path(sp))
    return out


def gt_segments(row) -> list[tuple[float, float, float, float]]:
    segs = row.get("ref_segments")
    if segs:
        out = []
        for j, s in enumerate(segs):
            a = float(s["mix_start_s"])
            b = (
                float(segs[j + 1]["mix_start_s"])
                if j + 1 < len(segs)
                else float(row["set_end_s"])
            )
            out.append((a, b, float(s["ref_start_s"]), float(s["ref_end_s"])))
        return out
    return [
        (
            float(row["set_start_s"]),
            float(row["set_end_s"]),
            float(row["ref_start_s"]),
            float(row["ref_end_s"]),
        )
    ]


def warped_layer(row, ref: np.ndarray, n: int, t0: float) -> np.ndarray:
    """Ref audio mapped onto the mix-block timeline via GT (linear per segment)."""
    out = np.zeros(n, dtype=np.float64)
    for a, b, ra, rb in gt_segments(row):
        a_, b_ = max(a, BLOCK[0]), min(b, BLOCK[1])
        if b_ - a_ < 0.5:
            continue
        # crop ref range proportionally to the cropped mix range
        f0 = (a_ - a) / max(b - a, 1e-9)
        f1 = (b_ - a) / max(b - a, 1e-9)
        ra_, rb_ = ra + f0 * (rb - ra), ra + f1 * (rb - ra)
        i0, i1 = int((a_ - t0) * SR), int((b_ - t0) * SR)
        src = ref[int(ra_ * SR) : int(rb_ * SR)]
        if src.size < 100:
            continue
        idx = np.linspace(0, src.size - 1, i1 - i0)
        out[i0:i1] = np.interp(idx, np.arange(src.size), src)
    return out


NFFT, NHOP = 4096, 1024  # cancel.py winner's grid


def _stft(y):
    import librosa

    return librosa.stft(y.astype(np.float32), n_fft=NFFT, hop_length=NHOP)


def cancel_layer(mix: np.ndarray, layer: np.ndarray) -> tuple[np.ndarray, float]:
    """SPECTRAL-magnitude subtraction (cancel.py's adaptive design): keylock
    (phase-vocoder) warping destroys waveform phase but preserves magnitude.
    Per-frame per-bin adaptive gain (smoothed, capped 4), subtract magnitudes,
    resynthesize with mix phase. Returns residual, energy removed."""
    import librosa
    from scipy.ndimage import uniform_filter1d

    S_m = _stft(mix)
    S_l = _stft(layer)
    n = min(S_m.shape[1], S_l.shape[1])
    M, L = np.abs(S_m[:, :n]), np.abs(S_l[:, :n])
    # adaptive per-bin gain, time-smoothed (~0.5 s = 10 frames)
    g = M / np.maximum(L, 1e-9)
    g = uniform_filter1d(g, 10, axis=1, mode="nearest")
    g = np.clip(g, 0.0, 4.0)
    # only subtract where the layer actually explains energy (L above floor)
    g[L < np.percentile(L, 30)] = 0.0
    R = np.maximum(M - g * L, 0.0)
    before = float((M**2).sum())
    after = float((R**2).sum())
    res = librosa.istft(
        R * np.exp(1j * np.angle(S_m[:, :n])), hop_length=NHOP, length=len(mix)
    ).astype(np.float64)
    return res, max(0.0, before - after) / max(before, 1e-9)


def salience(mix: np.ndarray, layer: np.ndarray) -> float:
    """Median per-frame cosine of MAGNITUDE spectra (keylock-tolerant)."""
    S_m, S_l = np.abs(_stft(mix)), np.abs(_stft(layer))
    n = min(S_m.shape[1], S_l.shape[1])
    rs = []
    for i in range(n):
        a, b = S_l[:, i], S_m[:, i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-3 or nb < 1e-6:
            continue
        rs.append(float(np.dot(a, b)) / (na * nb))
    return float(np.median(rs)) if rs else 0.0


t0 = BLOCK[0]
mix_full = load_audio(
    str(base / [p for p in ["mix.m4a"] if (base / p).is_file()][0])
    if (base / "mix.m4a").is_file()
    else man["mix_local_path"],
    offset_s=t0,
    duration_s=BLOCK[1] - t0,
)
n = mix_full.size
e_mix = float(np.dot(mix_full, mix_full))
print(f"block {BLOCK[0]:.0f}-{BLOCK[1]:.0f}s  mix energy 0 dB ref")

# build + rank cancel layers by in-block energy, best source per layer
layers = []
for _key, row, slot, _nm in CANCEL:
    best = None
    for src in sources_for(row):
        ref = load_audio(str(src))
        lay = warped_layer(row, ref, n, t0)
        sal = salience(mix_full, lay)
        if best is None or sal > best[2]:
            best = (slot, lay, sal, src.name[:40])
    if best:
        layers.append(best)
        print(f"  cancel layer {best[0]}: salience={best[2]:.3f} src={best[3]}")
layers.sort(key=lambda x: -x[2])

res = mix_full.copy()
for slot, lay, sal, srcn in layers:
    res, removed = cancel_layer(res, lay)
    print(
        f"  cancelled {slot}: removed {100 * removed:.1f}% of remaining spectral energy"
    )

print(
    f"\nTOTAL residual: {10 * np.log10(float(np.dot(res, res)) / e_mix):.2f} dB re mix"
)

print("\nmissed-layer salience (mix -> residual):")
for _key, row, slot, _nm in MISSED:
    best = (0.0, 0.0, "-")
    for src in sources_for(row):
        ref = load_audio(str(src))
        lay = warped_layer(row, ref, n, t0)
        s_mix, s_res = salience(mix_full, lay), salience(res, lay)
        if s_res > best[1]:
            best = (s_mix, s_res, src.name[:40])
    d = (
        "IMPROVED"
        if best[1] > best[0] * 1.1
        else ("flat" if best[1] > best[0] * 0.9 else "WORSE")
    )
    print(f"  {slot}: {best[0]:.3f} -> {best[1]:.3f}  {d}  ({best[2]})")

outdir = Path.home() / "aligning/_review/_sic"
outdir.mkdir(parents=True, exist_ok=True)
sf.write(
    outdir / "finale_mix.wav",
    (mix_full / np.abs(mix_full).max() * 0.7).astype(np.float32),
    SR,
)
sf.write(
    outdir / "finale_residual.wav",
    (res / max(np.abs(res).max(), 1e-9) * 0.7).astype(np.float32),
    SR,
)
print(f"\nwavs -> {outdir}")

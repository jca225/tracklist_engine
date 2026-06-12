"""EDA: is BB12 mashup compatibility about ENERGY/brightness match (not harmony)?

The harmonic EDA showed key barely separates real mashups (AUC 0.60). The other
big DJ principle is energy/section matching — a soaring vocal over a big drop. Test
whether real (bed, payload) sections match in loudness (RMS) and brightness
(spectral centroid) more than random pairs. Uses the local stem slices (no MERT).
If energy ALSO caps ~0.6, audio-only compatibility is exhausted -> pivot to the
taste/curator-conditioned model.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from analysis.adapters import audio_io, mert_adapter
from workspaces.mashup_compat.embed import LOCAL_STEMS, _pull_stem, _resolve_track_audio_ids
from workspaces.mashup_compat.section import build_pairs

SR = mert_adapter.MERT_SR


def _feats(samples: np.ndarray, ref_start: float, ref_end: float) -> tuple[float, float] | None:
    a, b = int(ref_start * SR), int(ref_end * SR)
    seg = samples[a:b]
    if seg.size < SR // 2:
        return None
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-9)
    # spectral centroid (brightness) via magnitude spectrum
    mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    freqs = np.fft.rfftfreq(seg.size, 1 / SR)
    centroid = float((freqs * mag).sum() / (mag.sum() + 1e-9))
    return (np.log(rms), centroid)


def main() -> int:
    pos, neg = build_pairs()
    pairs = [(b, p, True) for b, p in pos] + [(b, p, False) for b, p in neg]
    sections = {s.key: s for b, p, _ in pairs for s in (b, p)}
    taids = _resolve_track_audio_ids([s.track_id for s in sections.values()])

    feat: dict = {}
    cache_samples: dict = {}
    for key, s in sections.items():
        taid = taids.get(s.track_id)
        if taid is None:
            continue
        if (taid, s.stem_file) not in cache_samples:
            sp = _pull_stem(taid, s.stem_file)
            if sp is None:
                cache_samples[(taid, s.stem_file)] = None
            else:
                wf = audio_io.load_mono(sp, target_sr=SR)
                cache_samples[(taid, s.stem_file)] = wf.value.samples if wf.is_ok() else None
        samples = cache_samples[(taid, s.stem_file)]
        if samples is not None:
            f = _feats(samples, s.ref_start, s.ref_end)
            if f:
                feat[key] = f

    rms_d, cen_d, y = [], [], []
    for b, p, pos_ in pairs:
        if b.key in feat and p.key in feat:
            rms_d.append(abs(feat[b.key][0] - feat[p.key][0]))
            cen_d.append(abs(feat[b.key][1] - feat[p.key][1]))
            y.append(1 if pos_ else 0)
    y = np.array(y)
    print(f"usable pairs: {len(y)} (pos={int(y.sum())}, neg={int((1-y).sum())})\n")
    print("AUC (compatible = small difference):")
    print(f"  RMS loudness match   : {roc_auc_score(y, -np.array(rms_d)):.3f}")
    print(f"  spectral centroid    : {roc_auc_score(y, -np.array(cen_d)):.3f}")
    print(f"\n  (compare: key 0.628, section-MERT 0.62 — is energy any better?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

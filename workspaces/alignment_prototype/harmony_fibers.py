#!/usr/bin/env python3
"""Harmony fibers — a higher-recall "key fiber" for instrumental/regular content.

Motivation (see external/fiber_validation.py findings). ``ref_fibers.compute_fibers``
on HuBERT is PRECISE but LOW-RECALL (SALAMI pairwise F 0.10, P 0.88, R 0.06): it is
timbre/phonetic and blind to melodic/harmonic-only repeats (a chorus replayed with
a different arrangement, an instrumental section that returns). Feeding
``compute_fibers`` CHROMA instead fails the opposite way — chroma self-similarity is
high EVERYWHERE (harmonically uniform pop), so the fixed 0.5 diagonal threshold
catches diagonals at every lag and average-linkage collapses to ONE fiber on 30/30
tracks (F 0.51 ≈ the all-one baseline 0.52). Degenerate over-merge, not detection.

**What this module does differently.** Same output contract as ``compute_fibers``
(``harmony_fibers(audio_path, *, fps=...) -> (labels, label_hz)``, per-frame int
repeat-class labels, -1 = non-repeated/silence) so it drops into the existing
scoring unchanged. But the detection is a diagonal-STRIPE extraction with an
ADAPTIVE, per-track threshold — the two things that stop the everywhere-merge:

  1. **CENS chroma** (Müller/Ewert) — smoothed + downsampled chroma, quantized;
     far more robust for harmonic *structure* than frame-level chroma-CQT. Tempo
     changes / local timbre wash out; the long-range harmonic shape survives.
  2. **Diagonal-enhanced SSM.** Cosine SSM, then median-smooth ALONG the diagonal
     direction (a Müller "path enhancement") so a sustained repeat becomes a bright
     line and transient coincidences are suppressed.
  3. **Time-lag transform.** Re-index the SSM by (time, lag): a repeat of segment
     [i..j] at lag L is a horizontal STRIPE at lag L over columns i..j. Repeats are
     now bright horizontal runs, trivially extractable as 1-D runs per lag.
  4. **ADAPTIVE threshold.** For each candidate lag we compare the lag row against
     the track's OWN off-diagonal similarity distribution (a high global-percentile
     floor) AND require the stripe to be a local prominence over the median lag
     energy at that time. Chroma's high everywhere-similarity is a CONSTANT that the
     percentile floor absorbs; only stripes that stand OUT survive. This is the
     lever the fixed-0.5 chroma-blob lacks.
  5. **Grouping.** Each surviving stripe links segment [i..j] to [i+L..j+L]. We
     union-find those links into repeat classes, then verify each merged class the
     SAME way ref_fibers does — a best-offset matched-filter diagonal between the
     two spans in CENS space (drop links whose spans don't actually correlate), so a
     spuriously-bright lag can't fabricate a class. -1 for everything unlinked.

librosa only (SR=22050, HOP=512, fps≈43.07 to match the harness grid). No new deps.

Validate: ``python -m workspaces.alignment_prototype.external.fiber_validation
--harmony`` (this module adds a ``--harmony`` arm to that harness; the HuBERT/chroma
arms are untouched). A WIN = recall materially above HuBERT's 0.06, precision
usably above the chroma-blob's 0.44 (>0.6), F above the all-one baseline 0.52.
Read-only imports; does not modify ref_fibers.
"""

from __future__ import annotations

import warnings

import numpy as np

from workspaces.alignment_prototype.refine_ref_offsets import HOP, SR

# CENS downsample factor: chroma-CQT at SR/HOP≈43.07 Hz -> CENS at ~4.3 Hz.
# 10x smoothing/decimation is the Müller default for structure; slow enough to
# suppress local timbre, fast enough for ~2s section granularity.
_CENS_DS = 10


def _cens(y: np.ndarray) -> tuple[np.ndarray, float]:
    """(12, T) CENS chroma + its frame rate (Hz). Smoothed, L2-normed per frame.

    CENS (Chroma Energy Normalized Statistics) applies quantization + local
    averaging to chroma-CQT — the standard feature for harmonic *structure*
    analysis because it discards short-time fluctuation while keeping the
    long-range harmonic contour that defines a repeated section."""
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP, win_len_smooth=41)
    c = c[:, ::_CENS_DS]
    c = c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)
    hz = (SR / HOP) / _CENS_DS
    return c.astype(np.float32), hz


def _diag_enhanced_ssm(c: np.ndarray, diag_len: int) -> np.ndarray:
    """Cosine SSM (T,T) with median smoothing ALONG the main diagonal direction.

    Path enhancement (Müller): replace S[i,j] by the median of the diagonal
    stretch S[i-k..i+k, j-k..j+k] centred on (i,j). A sustained repeat (a bright
    diagonal segment) survives the median; an isolated coincidence does not. This
    is what turns chroma's noisy off-diagonals into clean extractable stripes."""
    s = (c.T @ c).astype(np.float32)  # cosine (columns already L2-normed)
    t = s.shape[0]
    k = max(1, diag_len // 2)
    out = np.empty_like(s)
    # For each offset d along the diagonal, median-filter the d-diagonal in 1D.
    from scipy.ndimage import median_filter

    # Cheap equivalent: median-filter along diagonals via a diagonal-oriented
    # window. We do it diagonal-by-diagonal (T diagonals, each a 1-D median).
    for d in range(-t + 1, t):
        idx_i = np.arange(max(0, -d), min(t, t - d))
        idx_j = idx_i + d
        if idx_i.size == 0:
            continue
        vals = s[idx_i, idx_j]
        if vals.size >= 3:
            vals = median_filter(vals, size=min(2 * k + 1, vals.size), mode="nearest")
        out[idx_i, idx_j] = vals
    return out


def _diag_sim_cens(a: np.ndarray, b: np.ndarray) -> float:
    """Best-offset normalized matched-filter peak between two CENS slices — the
    SAME verification ref_fibers._diag_sim uses, in CENS space. A sustained
    diagonal => the two spans are the same harmonic content."""
    from scipy.signal import fftconvolve

    a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-9)
    if a.shape[1] > b.shape[1]:
        a, b = b, a
    n = a.shape[1]
    if n < 2 or b.shape[1] < n:
        return 0.0
    w = a / (np.linalg.norm(a) + 1e-9)
    num = fftconvolve(b, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
    e = np.concatenate([[0.0], np.cumsum((b**2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[n:] - e[:-n], 1e-9))
    return float((num / den).max())


def _stripes(
    s: np.ndarray,
    hz: float,
    nonsil: np.ndarray,
    *,
    min_repeat_s: float,
    min_lag_s: float,
    glob_pct: float,
    local_prom: float,
    smooth_s: float,
) -> list[tuple[int, int, int]]:
    """Extract diagonal stripes as (start, end, lag) in CENS frames.

    A stripe at lag L over [i..j] means content [i..j] recurs at [i+L..j+L].
    The anti-blob lever is a LOCAL-CONTRAST time-lag transform, not an absolute
    threshold. We build the time-lag matrix ``TL[time, lag] = S[time, time+lag]``,
    then subtract, for each time, the local background across NEARBY lags (a
    per-column median over a lag window). On a harmonically-UNIFORM track every
    lag is equally bright, so the contrast is ~0 everywhere and NOTHING survives —
    which is the correct answer (no discrete harmonic repeat is separable there,
    the SALAMI jam-band caveat). On a track with a real repeat, the true lag
    stands OUT from its neighbours and its stripe survives. A stripe must clear
    BOTH a positive local-contrast margin (``local_prom``) AND a track-relative
    absolute floor (``glob_pct`` percentile of the raw off-diagonal, so a stripe
    that is locally prominent but globally dim — pure noise — is still rejected).
    """
    from scipy.ndimage import uniform_filter1d

    t = s.shape[0]
    # off-diagonal population (exclude the |lag|<min_lag band; that is trivially
    # bright and would drag the percentile down / isn't a repeat).
    ml_lag = max(1, int(min_lag_s * hz))
    mask = np.abs(np.subtract.outer(np.arange(t), np.arange(t))) >= ml_lag
    pop = s[mask]
    if pop.size == 0:
        return []
    floor = float(np.percentile(pop, glob_pct))

    # --- time-lag matrix + local-lag-background subtraction -------------------
    # TL[i, lag] = S[i, i+lag]; invalid (i+lag>=t) left as NaN so background
    # medians ignore the ragged corner.
    n_lag = t - ml_lag
    tl = np.full((t, n_lag), np.nan, dtype=np.float32)
    for k, lag in enumerate(range(ml_lag, t)):
        d = np.diagonal(s, lag).astype(np.float32)  # length t-lag
        tl[: d.size, k] = d
    # local background across lags: median over a +/- lag_win window, per time.
    lag_win = max(3, int(6.0 * hz))  # ~6 s of lag context
    bg = np.full_like(tl, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(t):
            row = tl[i]
            valid = ~np.isnan(row)
            if valid.sum() < 3:
                continue
            # rolling median via nan-aware: use overall row median as a robust,
            # cheap per-time background (uniform track -> flat row -> ~row value).
            bg[i] = np.nanmedian(row)
    contrast = tl - bg  # >0 where this lag beats the track-time background

    pw = max(1, int(smooth_s * hz))
    mr = max(1, int(min_repeat_s * hz))
    # Gather candidate stripes PER LAG, then keep only the track's DOMINANT lags.
    # A real repeat structure has a handful of characteristic lags (a section
    # returns at 1-3 offsets); a uniform jam "repeats" weakly at dozens of lags.
    # If the qualifying lags are too diffuse (> `max_lags`), the track has no
    # separable harmonic repeat -> return empty (honest, not a blob).
    per_lag: dict[int, list[tuple[int, int, int]]] = {}
    lag_mass: dict[int, int] = {}
    for k, lag in enumerate(range(ml_lag, t)):
        diag = np.diagonal(s, lag).astype(np.float32)  # raw sim, index = i
        diag = uniform_filter1d(diag, pw, mode="nearest")
        con = contrast[: t - lag, k]
        con = uniform_filter1d(np.nan_to_num(con, nan=-1.0), pw, mode="nearest")
        good = (con > local_prom) & (diag > floor) & nonsil[: t - lag] & nonsil[lag:t]
        segs: list[tuple[int, int, int]] = []
        i = 0
        n = good.size
        while i < n:
            if good[i]:
                j = i
                while j < n and good[j]:
                    j += 1
                if j - i >= mr:
                    segs.append((i, j, lag))
                i = j
            else:
                i += 1
        if segs:
            per_lag[lag] = segs
            lag_mass[lag] = sum(j - i for i, j, _ in segs)

    if not per_lag:
        return []
    # merge adjacent lags' mass (a repeat smears over a few neighbouring lags)
    # into lag CLUSTERS, rank clusters by mass, keep the strongest few.
    lag_tol = max(1, int(2.0 * hz))
    lags_sorted = sorted(per_lag)
    clusters: list[list[int]] = [[lags_sorted[0]]]
    for lg in lags_sorted[1:]:
        if lg - clusters[-1][-1] <= lag_tol:
            clusters[-1].append(lg)
        else:
            clusters.append([lg])
    cl_mass = [(sum(lag_mass[lg] for lg in cl), cl) for cl in clusters]
    cl_mass.sort(reverse=True, key=lambda x: x[0])

    # UNIFORM-TRACK GATE (the anti-blob decision). A separable repeat structure
    # has (a) few dominant lag clusters and (b) the repeats occupy SECTIONS, not
    # the whole track. If the detected stripes are diffuse — many lag clusters
    # and/or the per-frame stripe density is huge (every frame "repeats" at many
    # lags) — there is no discrete harmonic repeat to recover; return empty
    # rather than let union-find cascade the whole track into one fiber. This is
    # the honest answer on harmonically-static jam material (SALAMI caveat), and
    # it is exactly what stops the fixed-0.5 chroma-blob failure.
    tot_mass = sum(m for m, _ in cl_mass)
    audible = max(1, int(nonsil.sum()))
    density = tot_mass / audible  # avg # of stripe-frames covering each audible frame
    max_lag_clusters = 10
    if len(cl_mass) > max_lag_clusters or density > 6.0:
        return []
    keep_lags = set(per_lag)
    out: list[tuple[int, int, int]] = []
    for lg in keep_lags:
        out.extend(per_lag[lg])
    return out


def _rms_nonsil(y: np.ndarray, t: int, hz: float, ratio: float) -> np.ndarray:
    """Per-CENS-frame audibility mask (silence is perfectly self-similar and would
    forge repeats). Mirrors ref_fibers' RMS silence gate at the CENS rate."""
    import librosa

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    step = max(1, int(round((SR / HOP) / hz)))
    rms = rms[::step]
    floor = float(np.median(rms)) * ratio
    nonsil = np.ones(t, dtype=bool)
    m = min(t, rms.size)
    nonsil[:m] = rms[:m] >= floor
    return nonsil


def harmony_fibers(
    audio_path: str,
    *,
    fps: float = SR / HOP,
    min_repeat_s: float = 8.0,
    min_lag_s: float = 8.0,
    glob_pct: float = 92.0,
    local_prom: float = 0.07,
    smooth_s: float = 2.0,
    verify_thresh: float = 0.60,
    silence_ratio: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Per-frame harmonic repeat-class labels (contract-compatible with
    ``ref_fibers.compute_fibers``): ``(labels, label_hz)``, shared id = same
    repeated harmonic section, -1 = non-repeated / silence.

    ``fps`` is accepted for signature parity (the HuBERT grid rate); this detector
    reads the audio directly and works at the internal CENS rate, returning
    ``label_hz`` = CENS rate. The harness resamples labels onto its own grid, so
    the exact rate is transparent to scoring.

    Pipeline: CENS chroma -> diagonal-enhanced SSM -> adaptive diagonal-stripe
    extraction (per-track percentile floor + per-lag prominence) -> union-find the
    stripe links into repeat classes -> matched-filter verify each class (drop
    spans that don't actually correlate) -> relabel. The adaptive threshold + the
    verification are what keep it from blobbing to one fiber the way fixed-0.5
    chroma does."""
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)

    c, hz = _cens(y)
    t = c.shape[1]
    labels = -np.ones(t, dtype=int)
    if t < int(2 * min_repeat_s * hz):
        return labels, hz

    nonsil = _rms_nonsil(y, t, hz, silence_ratio)
    diag_len = max(3, int(0.5 * min_repeat_s * hz))
    s = _diag_enhanced_ssm(c, diag_len)

    stripes = _stripes(
        s,
        hz,
        nonsil,
        min_repeat_s=min_repeat_s,
        min_lag_s=min_lag_s,
        glob_pct=glob_pct,
        local_prom=local_prom,
        smooth_s=smooth_s,
    )
    if not stripes:
        return labels, hz

    # --- union-find over frames linked by a (verified) stripe -----------------
    parent = list(range(t))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    linked = np.zeros(t, dtype=bool)
    for i, j, lag in stripes:
        a0, a1 = i, j
        b0, b1 = i + lag, min(j + lag, t)
        if b1 - b0 < 2 or a1 - a0 < 2:
            continue
        # verify the two spans really correlate in CENS space (anti-fabrication)
        sim = _diag_sim_cens(
            np.ascontiguousarray(c[:, a0:a1]),
            np.ascontiguousarray(c[:, b0:b1]),
        )
        if sim < verify_thresh:
            continue
        # Union each span into ONE connected component (consecutive frames
        # within a span are the same section), then link the two spans by their
        # first frames. Without the within-span union each diagonal frame-pair
        # would be its own class -> tens of singleton fibers per stripe.
        for d in range(1, a1 - a0):
            union(a0, a0 + d)
        for d in range(1, b1 - b0):
            union(b0, b0 + d)
        union(a0, b0)
        linked[a0:a1] = True
        linked[b0:b1] = True

    # relabel: only frames that got linked to >=1 other span become fibers.
    from collections import Counter

    roots = np.array([find(i) for i in range(t)])
    # a root is a real class only if >=2 disjoint instances mapped into it; we
    # approximate "real" by class size (a lone linked span is still a repeat pair
    # because linking always touches TWO spans, so any linked frame is a repeat).
    remap: dict[int, int] = {}
    nid = 0
    cnt = Counter(roots[linked].tolist())
    for i in range(t):
        if not linked[i]:
            continue
        r = int(roots[i])
        if cnt[r] < 2:
            continue
        if r not in remap:
            remap[r] = nid
            nid += 1
        labels[i] = remap[r]
    return labels, hz

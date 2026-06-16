#!/usr/bin/env python3
"""Fibers — a track's self-repeat equivalence classes, computed a priori.

Think of playback as a map ref-time -> audio-content. A repeated chorus means
many ref positions share one content value; the *fiber* over that content is the
set of repeat instances. Every "repeat error" the placement decoder makes is it
landing on the wrong point of the RIGHT fiber. Knowing the fibers lets us score
within-fiber picks as correct and collapse repeats instead of competing them.

**Lesson from human audit (2026-06-16):** the first version (spectral cluster
labels + pooled-cosine display) over-merged badly — John heard false merges
(diff sections, pooled sim 0.9+) and silence "fibers" (the vocal stem goes
quiet; silence is perfectly self-similar). Pooled cosine and spectral labels are
NOT trustworthy equivalence. Fixes, all validated on his examples:

  1. SILENCE GATE — drop sections whose RMS << the track median (the silence
     fibers had RMS ~0.0).
  2. DIAGONAL verification — equivalence = a sustained best-offset matched-filter
     diagonal between two sections (real repeats scored 0.56-0.84, false merges
     0.21-0.37; pooled cosine couldn't tell them apart).
  3. AVERAGE-LINKAGE grouping, not connected-components — a section that matches
     ONE member of a group but not the others (18s~174s=0.62 but 18s~96s=0.00)
     must not transitively join; averaging the links blocks that.

Feed HuBERT frames for vocals (phonetic -> robust to a singer varying delivery),
chroma for harmonic beds.

NOTE: with these (HuBERT, silence-gated) fibers the path_decode figure is
53% strict -> 59% fiber-aware (+6pp). The old "70%" was v1 over-merge inflation;
computing fibers on the chroma decode feature instead blobs to one fiber (fake
100%) — fibers MUST be HuBERT + silence-gated, never chroma.
"""

from __future__ import annotations

import warnings

import numpy as np

from workspaces.alignment_prototype.refine_ref_offsets import HOP, SR


def _long_repeats(g, g_hz, nonsil, min_repeat_s, thresh):
    """Candidate repeated SEGMENTS by scanning self-similarity diagonals.

    A sustained high run on the lag-L diagonal means [i..j] recurs at [i+L..j+L]
    — this captures the LONG, loud repeats (a 29 s chorus) that fixed short
    sections fragment and miss. Both ends must be audible (silence is perfectly
    self-similar and would otherwise form spurious repeats). Overlapping detected
    intervals are merged into canonical segments. Returns (start, end) in
    ds-frames."""
    from scipy.ndimage import uniform_filter1d

    t = g.shape[1]
    s = (g.T @ g).astype(np.float32)
    pw = max(1, int(2 * g_hz))
    ml = int(min_repeat_s * g_hz)
    ivs: list[tuple[int, int]] = []
    for lag in range(max(1, int(4 * g_hz)), t):
        d = uniform_filter1d(np.diagonal(s, lag).astype(np.float32), pw, mode="nearest")
        good = (d > thresh) & nonsil[: t - lag] & nonsil[lag:t]
        i = 0
        while i < len(good):
            if good[i]:
                j = i
                while j < len(good) and good[j]:
                    j += 1
                if j - i >= ml:
                    ivs.append((i, j))
                    ivs.append((i + lag, j + lag))
                i = j
            else:
                i += 1
    if not ivs:
        return []
    ivs.sort()
    merged = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= merged[-1][1] + pw:  # overlapping/adjacent -> one canonical segment
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _diag_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Best-offset normalized matched-filter peak between two feature slices —
    a sustained diagonal means the two sections are the SAME content (robust to
    where in the section they align). Slides the shorter over the longer."""
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


def _avg_linkage(sim: np.ndarray, thresh: float) -> list[int]:
    """Agglomerative grouping by AVERAGE linkage at `thresh`. Returns a group id
    per item. Average linkage (not single/connected-components) stops a section
    that matches one member but not the rest from transitively joining."""
    n = sim.shape[0]
    groups = [[i] for i in range(n)]
    while len(groups) > 1:
        best, bi, bj = thresh, -1, -1
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                m = np.mean([sim[a, b] for a in groups[i] for b in groups[j]])
                if m > best:
                    best, bi, bj = m, i, j
        if bi < 0:
            break
        groups[bi].extend(groups[bj])
        groups.pop(bj)
    label = [0] * n
    for gid, g in enumerate(groups):
        for i in g:
            label[i] = gid
    return label


def compute_fibers(
    feat: np.ndarray,
    fps: float,
    *,
    ds_hz: float = 8.0,
    min_repeat_s: float = 6.0,
    repeat_thresh: float = 0.5,
    verify_thresh: float = 0.5,
    audio_path: str | None = None,
    silence_ratio: float = 0.35,
    k: int | None = None,  # back-compat (ignored)
    min_section_s: float | None = None,  # back-compat (ignored)
) -> tuple[np.ndarray, float]:
    """(labels, label_hz). feat=(D, T) per-column features at `fps`.

    Pipeline: downsample -> RMS silence mask (if `audio_path`) -> scan
    self-similarity diagonals for LONG sustained repeats (the long/loud
    hook/chorus, not short fragments) -> verify+group candidate segments by
    average-linkage on the best-offset diagonal similarity. labels are per
    downsampled frame: shared id = same fiber, -1 = silence / non-repeated.
    Use `same_fiber` for the question the decoder/scorer actually asks."""
    step = max(1, int(round(fps / ds_hz)))
    g = feat[:, ::step].astype(np.float32)
    g_hz = fps / step
    g = g / (np.linalg.norm(g, axis=0, keepdims=True) + 1e-9)
    t = g.shape[1]
    labels = -np.ones(t, dtype=int)

    nonsil = np.ones(t, dtype=bool)
    if audio_path is not None:
        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(audio_path, sr=SR, mono=True)
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0][::step]
        floor = float(np.median(rms)) * silence_ratio
        nonsil[: rms.size] = rms[:t] >= floor

    secs = _long_repeats(g, g_hz, nonsil, min_repeat_s, repeat_thresh)
    if not secs:
        return labels, g_hz

    feats = [np.ascontiguousarray(g[:, a:b]) for a, b in secs]
    m = len(secs)
    sim = np.eye(m, dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            sim[i, j] = sim[j, i] = _diag_sim(feats[i], feats[j])
    grp = _avg_linkage(sim, verify_thresh) if m > 1 else [0]
    remap: dict[int, int] = {}
    next_id = 0
    for (a, b), gid in zip(secs, grp):
        if gid not in remap:
            remap[gid] = next_id
            next_id += 1
        labels[a : min(b, t)] = remap[gid]
    return labels, g_hz


def compute_fibers_fp(
    audio_path: str,
    *,
    label_hz: float = 8.0,
    min_lag_s: float = 8.0,
    lag_tol_s: float = 0.5,
    peak_frac: float = 0.15,
    dens_frac: float = 0.30,
    min_repeat_s: float = 6.0,
    close_s: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Constellation match-density localizer (the "proper localizer").

    HuBERT is blind to MELODIC repeats (John's audit: 0:00 ≈ 2:32 of Love On Me
    scores 0.11) and chroma is harmonically uniform (its diagonal is high
    EVERYWHERE at a real lag, so it can't localize). The discriminative signal is
    the landmark-fingerprint match-density: peak-pair hashes give (1) robust
    repeat LAGS by vote, and (2) per-time density of hashes at time t that recur
    at t+L — high ONLY in the actual repeated region. Pipeline: hashes -> strong
    lags (clustered, vote-thresholded) -> per-lag covered-density -> runs ->
    union [s,e] with [s+L,e+L]. labels: shared id = same fiber, -1 = none.

    STATUS (2026-06-16): recovers melodic repeats HuBERT misses (Love On Me,
    Emily) but is THRESHOLD-SENSITIVE across tracks (Congratulations/Freeze Time
    can come out empty at one param set). Full robustness needs per-track
    ADAPTIVE thresholding (or msaf / learned contrastive embeddings — not
    installed/built). Offered as a selectable method, not the default, so it
    doesn't regress HuBERT-validated tracks."""
    import librosa
    from collections import Counter, defaultdict

    from scipy.ndimage import binary_closing, uniform_filter1d

    from workspaces.alignment_prototype.fp_probe import _FPS, constellation, hashes

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
    nfr = max(1, int(len(y) / SR * label_hz))
    labels = -np.ones(nfr, dtype=int)
    h = hashes(*constellation(y))
    lh: dict = defaultdict(int)
    lag_ti: dict = defaultdict(list)
    minlag = int(min_lag_s * _FPS)
    for _key, ts in h.items():
        s = sorted(set(ts))
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                lg = s[j] - s[i]
                if lg >= minlag:
                    lh[lg] += 1
                    lag_ti[lg].append(s[i])
    if not lh:
        return labels, label_hz
    tol = int(lag_tol_s * _FPS)
    clusters: list = []
    for lg, v in sorted(lh.items()):
        if clusters and lg - clusters[-1][0] <= tol:
            clusters[-1] = [
                clusters[-1][0],
                clusters[-1][1] + v,
                clusters[-1][2] + lag_ti[lg],
            ]
        else:
            clusters.append([lg, v, list(lag_ti[lg])])
    mx = max(c[1] for c in clusters)
    parent = list(range(nfr))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    minrep = int(min_repeat_s * label_hz)
    for lag_c, v, tis in clusters:
        if v < peak_frac * mx:
            continue
        lb = int(round(lag_c / _FPS * label_hz))
        if lb < minrep or lb >= nfr:
            continue
        dens = np.zeros(nfr)
        for ti in tis:
            b = int(ti / _FPS * label_hz)
            if 0 <= b < nfr:
                dens[b] += 1
        dens = uniform_filter1d(dens, int(2 * label_hz))
        if dens.max() <= 0:
            continue
        cov = binary_closing(
            dens > dens_frac * dens.max(), structure=np.ones(int(close_s * label_hz))
        )
        i = 0
        while i < nfr:
            if cov[i]:
                k = i
                while k < nfr and cov[k]:
                    k += 1
                if k - i >= minrep:
                    for b in range(i, k):
                        if b + lb < nfr:
                            parent[find(b)] = find(b + lb)
                i = k
            else:
                i += 1
    lab = np.array([find(i) for i in range(nfr)])
    cnt = Counter(lab.tolist())
    out = -np.ones(nfr, dtype=int)
    remap: dict = {}
    nid = 0
    for i in range(nfr):
        c = lab[i]
        if cnt[c] >= int(1.5 * minrep):
            if c not in remap:
                remap[c] = nid
                nid += 1
            out[i] = remap[c]
    return out, label_hz


def fiber_at(labels: np.ndarray, label_hz: float, sec: float) -> int:
    if labels.size == 0:
        return -1
    b = int(round(sec * label_hz))
    return int(labels[min(max(b, 0), labels.size - 1)])


def same_fiber(labels: np.ndarray, label_hz: float, a_s: float, b_s: float) -> bool:
    """Do two ref times fall in the same self-repeat class? (-1 = silence /
    ungrouped never counts as equivalent.)"""
    fa = fiber_at(labels, label_hz, a_s)
    fb = fiber_at(labels, label_hz, b_s)
    return fa >= 0 and fa == fb


def fiber_intervals(
    labels: np.ndarray, label_hz: float, min_len_s: float = 4.0
) -> list[tuple[float, float, int]]:
    """Contiguous (start_s, end_s, label) runs >= min_len_s, excluding silence
    (label -1) — for inspection / UI."""
    out: list[tuple[float, float, int]] = []
    if labels.size == 0:
        return out
    s = 0
    for i in range(1, labels.size + 1):
        if i == labels.size or labels[i] != labels[i - 1]:
            if labels[s] >= 0 and (i - s) / label_hz >= min_len_s:
                out.append((s / label_hz, i / label_hz, int(labels[s])))
            s = i
    return out

#!/usr/bin/env python3
"""Synthetic fixtures for looptrace (brief §9: ALL thresholds tune here,
never on the BB11/BB12 answer keys).

`make_song` synthesizes a non-stationary pseudo-vocal: a stream of distinct
"syllables" (harmonic tones with random f0/envelope + noise consonants), so
every second has unique content unless we plant structure deliberately.

`make_mix_span` builds a fake DJ-mix span from a song: straight play, N×
loops (digital copies, as a DJ's loop button produces), section jumps, and
tempo-stretch in both DJ modes — repitch (resample: tempo+pitch together)
and key-locked (phase-vocoder time stretch). Returns the audio plus the
ground-truth segment list and loop structure, in ORIGINAL song seconds and
span-relative mix seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from workspaces.alignment_prototype.looptrace.selfsim import SR


@dataclass(frozen=True)
class GTLoop:
    start_s: float  # span-relative mix time
    end_s: float
    period_s: float
    n_iter: int


@dataclass(frozen=True)
class GTSegment:
    mix_start_s: float  # span-relative
    mix_end_s: float
    song_start_s: float
    song_end_s: float


@dataclass(frozen=True)
class MixSpan:
    y: np.ndarray
    segments: tuple[GTSegment, ...]
    loops: tuple[GTLoop, ...]
    stretch: float  # mix_time * stretch = song_time (slope)
    mode: str  # "none" | "repitch" | "keylock"
    # the song's own repeat lags in SONG seconds — what a perfect Phase-1
    # audit of the reference would report (feeds the structural exclusion)
    song_lags_s: tuple[float, ...] = ()


def make_song(dur_s: float = 90.0, seed: int = 0) -> np.ndarray:
    """Non-stationary pseudo-vocal; distinct content everywhere."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    y = np.zeros(n)
    t = 0
    while t < n:
        syl = rng.uniform(0.15, 0.45)  # syllable length
        m = min(int(syl * SR), n - t)
        tt = np.arange(m) / SR
        f0 = rng.uniform(110, 330)
        tone = sum(
            rng.uniform(0.2, 1.0)
            * np.sin(2 * np.pi * f0 * (k + 1) * tt + rng.uniform(0, 6.28))
            for k in range(4)
        )
        env = np.minimum(1.0, np.minimum(tt / 0.02 + 1e-6, (m / SR - tt) / 0.05 + 1e-6))
        y[t : t + m] += tone * env * rng.uniform(0.4, 1.0)
        if rng.random() < 0.5:  # noise consonant at the boundary
            c = min(int(0.05 * SR), n - t)
            y[t : t + c] += rng.standard_normal(c) * 0.3
        t += m + int(rng.uniform(0.0, 0.12) * SR)  # tiny inter-syllable gap
    return y / (np.abs(y).max() + 1e-9)


def distinct_take(seg: np.ndarray, seed: int = 0) -> np.ndarray:
    """A surrogate re-sung take: same content, micro-timing jitter
    (piecewise ±2.5% resample drift) + a small amp ride. Distinct takes are
    what the audit found the corpora's natural repeats to be; the loop
    detector must NOT flag them as DJ loops."""
    from scipy.signal import resample_poly

    rng = np.random.default_rng(seed)
    out = []
    c = int(0.4 * SR)
    for i in range(0, len(seg), c):
        f = rng.uniform(0.975, 1.025)
        out.append(
            resample_poly(seg[i : i + c], int(round(f * 1000)), 1000)
            * rng.uniform(0.85, 1.15)
        )
    return np.concatenate(out)


def _bleed(y: np.ndarray, bleed_db: float | None, rng) -> np.ndarray:
    if bleed_db is None:
        return y
    n = rng.standard_normal(y.size)
    n *= (np.linalg.norm(y) / (np.linalg.norm(n) + 1e-12)) * 10 ** (bleed_db / 20)
    return y + n


def _stretch(y: np.ndarray, factor: float, mode: str) -> np.ndarray:
    """factor >1 = played slower in the mix (song time / mix time < 1...).

    Here `factor` = output_duration / input_duration. repitch = resample
    (tempo and pitch move together); keylock = phase-vocoder stretch."""
    if mode == "none" or abs(factor - 1.0) < 1e-9:
        return y
    if mode == "repitch":
        from scipy.signal import resample_poly

        num, den = int(round(factor * 1000)), 1000
        return resample_poly(y, num, den)
    if mode == "keylock":
        import librosa

        return librosa.effects.time_stretch(y=y.astype(np.float32), rate=1.0 / factor)
    raise ValueError(mode)


def make_mix_span(
    song: np.ndarray,
    kind: str,
    *,
    seed: int = 0,
    stretch_factor: float = 1.0,
    stretch_mode: str = "none",
    bleed_db: float | None = None,
) -> MixSpan:
    """kind: straight | loop2 | loop4 | replay | jump | natural_repeat.

    bleed_db: add INDEPENDENT noise per loop iteration at this level below
    the signal — models Roformer separation crosstalk, which differs per
    iteration because the instrumental bed differs (a real BB12 loop
    measured -6.7 dB residual between iterations of mix_vocals).

    Loops model the Ableton production reality of the GT mixes: the DJ
    duplicates a WARPED clip, so the stretch happens FIRST and the tiled
    iterations are digital copies of the stretched audio. (The live-hardware
    order — tile, then master-tempo vocoder — defeats waveform cloneness and
    is a documented detector limitation; see tests' xfail.) For non-loop
    kinds the stretch is applied to the assembled span; either order gives
    the same GT geometry. GT slope (song per mix second) = 1/factor."""
    rng = np.random.default_rng(seed + hash(kind) % 1000)
    dur = len(song) / SR

    def cut(t0: float, ln: float) -> np.ndarray:
        return song[int(t0 * SR) : int((t0 + ln) * SR)]

    segs: list[GTSegment] = []
    loops: list[GTLoop] = []
    if kind == "straight":
        ln = rng.uniform(18, 28)
        t0 = rng.uniform(0, dur - ln)
        y = cut(t0, ln)
        segs = [GTSegment(0.0, ln, t0, t0 + ln)]
    elif kind in ("loop2", "loop4"):
        n_iter = 2 if kind == "loop2" else 4
        period = float(rng.choice([1.875, 3.75, 7.5]))  # 1/2/4 bars @128bpm
        pre = rng.uniform(4, 8)
        post = rng.uniform(6, 10)
        t0 = rng.uniform(0, dur - (pre + period + post) - 1)
        loop_at = t0 + pre
        pre_y = _stretch(cut(t0, pre), stretch_factor, stretch_mode)
        loop_y = _stretch(cut(loop_at, period), stretch_factor, stretch_mode)
        post_y = _stretch(cut(loop_at + period, post), stretch_factor, stretch_mode)
        iters = [_bleed(loop_y, bleed_db, rng) for _ in range(n_iter)]
        y = np.concatenate([pre_y, *iters, post_y])
        f = stretch_factor if stretch_mode != "none" else 1.0
        pre_m, period_m = len(pre_y) / SR, len(loop_y) / SR
        m = pre_m
        segs = [GTSegment(0.0, pre_m, t0, loop_at)]
        for _ in range(n_iter):
            segs.append(GTSegment(m, m + period_m, loop_at, loop_at + period))
            m += period_m
        segs.append(
            GTSegment(
                m, m + len(post_y) / SR, loop_at + period, loop_at + period + post
            )
        )
        loops = [GTLoop(pre_m, pre_m + n_iter * period_m, period_m, n_iter)]
        return MixSpan(y, tuple(segs), tuple(loops), 1.0 / f, stretch_mode)
    elif kind == "replay":
        # non-contiguous A ... then jump BACK to A (section replay): lag >
        # matched region length; the second pass is truncated by span end.
        # Stretch-first like the loop kinds (duplicated warped clips).
        block = rng.uniform(16, 24)
        second = rng.uniform(8, block - 2)
        t0 = rng.uniform(0, dur - block - 1)
        b1 = _bleed(
            _stretch(cut(t0, block), stretch_factor, stretch_mode), bleed_db, rng
        )
        b2 = _bleed(
            _stretch(cut(t0, second), stretch_factor, stretch_mode), bleed_db, rng
        )
        y = np.concatenate([b1, b2])
        f = stretch_factor if stretch_mode != "none" else 1.0
        block_m, second_m = len(b1) / SR, len(b2) / SR
        segs = [
            GTSegment(0.0, block_m, t0, t0 + block),
            GTSegment(block_m, block_m + second_m, t0, t0 + second),
        ]
        loops = [GTLoop(0.0, block_m + second_m, block_m, 2)]
        return MixSpan(y, tuple(segs), tuple(loops), 1.0 / f, stretch_mode)
    elif kind == "jump":
        l1, l2 = rng.uniform(8, 14), rng.uniform(8, 14)
        a = rng.uniform(0, dur / 2 - l1)
        b = rng.uniform(dur / 2, dur - l2)
        y = np.concatenate([cut(a, l1), cut(b, l2)])
        segs = [GTSegment(0.0, l1, a, a + l1), GTSegment(l1, l1 + l2, b, b + l2)]
    elif kind == "natural_repeat":
        # the SONG repeats a phrase as two DISTINCT takes and the DJ plays
        # it straight — must NOT be detected as a loop (GT loops empty).
        # The repeat is part of the song, so a Phase-1 audit would report
        # its lag: exposed via MixSpan.song_lags_s for the structural test.
        ph = rng.uniform(3.0, 5.0)
        gap = rng.uniform(2.0, 4.0)
        t0 = rng.uniform(0, dur - (ph * 2 + gap) - 20)
        seg = cut(t0, ph)
        take2 = distinct_take(seg, seed=seed + 1)
        y = np.concatenate([seg, cut(t0 + ph, gap), take2, cut(t0 + ph + gap, 8.0)])
        m2 = ph + gap
        segs = [GTSegment(0.0, m2 + ph + 8.0, t0, t0 + ph + gap + 8.0)]
        y = _stretch(y, stretch_factor, stretch_mode)
        f = stretch_factor if stretch_mode != "none" else 1.0
        segs = [
            GTSegment(s.mix_start_s * f, s.mix_end_s * f, s.song_start_s, s.song_end_s)
            for s in segs
        ]
        return MixSpan(y, tuple(segs), (), 1.0 / f, stretch_mode, (ph + gap,))
    else:
        raise ValueError(kind)

    y = _stretch(y, stretch_factor, stretch_mode)
    f = stretch_factor if stretch_mode != "none" else 1.0
    segs = [
        GTSegment(s.mix_start_s * f, s.mix_end_s * f, s.song_start_s, s.song_end_s)
        for s in segs
    ]
    loops = [
        GTLoop(lo.start_s * f, lo.end_s * f, lo.period_s * f, lo.n_iter) for lo in loops
    ]
    return MixSpan(y, tuple(segs), tuple(loops), 1.0 / f, stretch_mode)

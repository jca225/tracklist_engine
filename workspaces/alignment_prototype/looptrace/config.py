#!/usr/bin/env python3
"""Versioned thresholds for looptrace. ALL tunables live here.

Rule of the effort (brief §9): thresholds are tuned on synthetic fixtures
only, never on the BB11/BB12 answer keys. Any post-hoc change after seeing
real-mix results must be logged in NOTES.md and everything rerun.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditConfig:
    """Phase 1 — ill-posedness audit (CLONE vs DISTINCT-TAKE repeats).

    audit-v1 (dead): HuBERT fibers as the repeat detector — under-detected
    (26 pairs / 19 tracks; HuBERT is blind to melodic repeats). audit-v2
    (dead): per-frame silence gate fragmented runs at every breath gap
    (median voiced run 1.6 s < the 4 s minimum -> 8/15 BB11 tracks got zero
    pairs). audit-v3: gap-closed gate + per-run voiced-fraction check."""

    version: str = "audit-v3"
    # selfsim.repeat_pairs — mel lag-diagonal scan
    min_lag_s: float = 2.0
    min_repeat_s: float = 4.0
    diag_thresh: float = 0.6
    silence_ratio: float = 0.35
    gap_close_s: float = 1.5
    min_voiced_frac: float = 0.4
    # sample-accurate verification around the coarse lag
    verify_pad_s: float = 1.0
    # CLONE test: residual after best-lag + best-gain alignment.
    # residual_energy = 1 - r^2 (r = normalized xcorr peak), in dB.
    # r=0.97 -> -12 dB. True digital copy-paste sits far below; distinct
    # vocal takes decorrelate at waveform level (r<0.5 -> ~-1 dB).
    clone_residual_db: float = -12.0
    # calibration: random non-repeat window pairs give the "unrelated"
    # residual distribution the CLONE threshold must sit far below.
    calib_pairs: int = 40
    seed: int = 7


@dataclass(frozen=True)
class LoopConfig:
    """Phase 2 — DJ loop/replay detection inside a mix span.

    loop-v1 (dead): clone floor (-12 dB) as the discriminator, max_lag 16 s.
    Real-mix measurement killed both: Roformer separation under different
    beds degrades a TRUE loop's iterations to -6.7 dB (inside the
    distinct-take band), and the GT repeats are section-scale (periods
    15-41 s) including non-contiguous replays. loop-v2: the discriminator
    is the STRUCTURAL song-lag test (a mix self-repeat at a lag the song's
    Phase-1 repeat map doesn't contain must be a DJ edit); the residual
    floor only rejects noise proposals."""

    version: str = "loop-v2"
    min_lag_s: float = 1.2  # 1 bar @ ~128 bpm after stretch slack
    max_lag_s: float = 120.0  # replays are section-scale (BB11 GT has a 109 s one); precision comes from verify + the structural test, not the lag cap
    min_repeat_s: float = 1.2
    diag_thresh: float = 0.45  # propose generously (verify + structure dispose)
    smooth_s: float = 0.75
    silence_ratio: float = 0.35
    gap_close_s: float = 1.0
    min_voiced_frac: float = 0.4
    verify_pad_s: float = 0.5
    fragment_gap_s: float = 2.5  # merge same-lag proposals across bleed gaps
    min_residual_db: float = -3.0  # noise rejection only (r >= ~0.65)
    core_trim_frac: float = 0.15  # trim frayed mel-run edges before verify
    min_iters: int = 2
    period_multiple_tol: float = 0.12
    # structural song-lag exclusion (needs the Phase-1 audit repeat map)
    song_lag_tol_s: float = 1.0
    song_lag_tol_frac: float = 0.04
    # KNOWN LIMITATION (fixture-measured): loops tiled BEFORE a key-locked
    # master-tempo stretch (live-hardware order) defeat waveform cloneness
    # (r~0.3) and spectral-magnitude + timing-drift verification cannot
    # separate them from natural distinct-take repeats (distributions
    # overlap; vocoder hop quantization injects ~15 ms fake drift). The GT
    # corpora are Ableton-produced (duplicated warped clips = copies AFTER
    # the stretch), so waveform verification + the structural test suffice.


@dataclass(frozen=True)
class EvalConfig:
    """Per-second segment accuracy (brief §9) on top of the frozen legacy
    trajectory_acc (tol=2.0 s, step=1.0 s)."""

    version: str = "eval-v1"
    step_s: float = 1.0
    tolerances_s: tuple[float, ...] = (0.25, 1.0, 2.0)


AUDIT_V3 = AuditConfig()
LOOP_V2 = LoopConfig()
EVAL_V1 = EvalConfig()

#!/usr/bin/env python3
"""Piecewise-linear path decode — one algorithm for ALL span types.

The single-line refiner (`refine_ref_offsets` / `continuity_refine`) emits one
`ref_start` + one `stretch`: a straight diagonal in (mix-time, ref-time). That
covers only 60/164 BB12 GT spans. The other 104 are excluded because they are
NOT one line:

  - section-jumps (73): DJ plays section A then jumps to C -> 2+ diagonals,
  - loops (10):          one short phrase repeated -> diagonal resets backward,
  - half/double-time (21): one diagonal too steep for the narrow stretch grid.

All three are the SAME object once the output is a *segment list* instead of a
line. We decode a piecewise-linear path with a Viterbi over ref-offset states:
staying on the diagonal is free, a section-jump costs `lam`. The decoded path's
runs of constant offset ARE the segments; discontinuities are the DJ's jumps.
Linear spans fall out as a path with zero jumps, loops as periodic backward
jumps, big-stretch clips by searching octave multiples of the grid stretch.

Scoring is unified too: a *trajectory* metric. Sample mix times across the span,
compute predicted vs GT ref position (piecewise-linear interpolation of the
segment list), and report the fraction within 2 s. This is the honest
generalization of "exact-<2s" and is defined for every span type.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.path_decode \
        --eval [--feature chroma|hubert] [--hubert-layer 9] \
        [--stems regular,acappella,instrumental] [--lam 0.6] [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core.timebase import Trajectory

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.continuity_refine import (  # noqa: E402
    _FEAT_CACHE,
    _compute_hubert,
)
from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP,
    SR,
    _MIX_SOURCE,
    _STEM_FILE,
    chroma,
    find_aligning_dir,
)

FPS = SR / HOP


# --- unified, mmap-able feature cache (chroma OR hubert) ------------------
def _feat_path(src_key, feature: str, layer: int) -> Path:
    h = hashlib.md5(str(src_key).encode()).hexdigest()[:16]
    tag = f"hubertL{layer}" if feature == "hubert" else feature
    return _FEAT_CACHE / f"{h}_{tag}.npy"


def _ensure_feat(audio_path, src_key, feature: str, layer: int) -> Path:
    """Compute (or reuse) a (D, T) feature, persisted so workers can mmap it."""
    if feature == "hubert":
        # _compute_hubert keys on the audio path; reuse its cache verbatim.
        _compute_hubert(audio_path, layer)
        from workspaces.alignment_prototype.continuity_refine import _hubert_cache_path

        return _hubert_cache_path(audio_path, layer)
    cf = _feat_path(src_key, feature, layer)
    if cf.is_file():
        return cf
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    _FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cf, chroma(y))
    return cf


# --- the decode -----------------------------------------------------------
def _cummax_arg(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive running max value and its argindex (vectorized)."""
    m = np.maximum.accumulate(x)
    arg = np.maximum.accumulate(np.where(x >= m, np.arange(x.size), -1))
    return m, arg


def _viterbi(
    reward: np.ndarray,
    lam_fwd: float,
    lam_back: float | None = None,
    *,
    fwd_slope: float = 0.0,
    back_slope: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Best piecewise-constant-offset path with a DIRECTIONAL jump penalty.

    reward[t, k] is the emission for window t at clip-start state k (larger k =
    later in the ref). Staying is free; a FORWARD jump (k increases — the DJ
    skips ahead, 51% of plays) costs ``lam_fwd + fwd_slope*(k-j)``; a BACKWARD
    jump (replay an earlier section) costs ``lam_back + back_slope*(j-k)``, with
    lam_back >> lam_fwd. This is the monotonic prior: among equivalent repeats
    (same fiber) the path prefers the forward-consistent instance, converting
    within-fiber ambiguity into correct placement.

    The optional *distance-graded* slopes (``fwd_slope``/``back_slope``, 0 =
    the original flat penalty, exact-behaviour default) make a jump's cost scale
    with its magnitude |Δk| in ref frames. This breaks the within-fiber tie the
    flat penalty leaves — when several equivalent instances have identical
    emission, the graded cost prefers the nearer (prior-consistent) one instead
    of an arbitrary DP end-bias. Calibrated from the warp prior's ref-jump
    scale (``warp_prior.json`` cut_up: forward jumps typical to +91s, backward
    rare). Still O(K) per step: a linear-in-distance penalty keeps the source
    search a prefix/suffix max on the shifted potential dp[j] ± slope*j.
    """
    if lam_back is None:
        lam_back = lam_fwd
    tm, K = reward.shape
    dp = reward[0].astype(np.float64).copy()
    src = np.empty((tm, K), dtype=np.int32)
    ar = np.arange(K)
    arf = ar.astype(np.float64)
    for t in range(1, tm):
        # forward source: max over j<k of dp[j] - fwd_slope*(k-j)
        #   = max_{j<k}(dp[j] + fwd_slope*j) - fwd_slope*k  (prefix max of g_f)
        g_f = dp + fwd_slope * arf
        pm, pa = _cummax_arg(g_f)
        pmv = np.empty(K)
        pmv[0] = -np.inf
        pmv[1:] = pm[:-1]
        pmv[1:] -= fwd_slope * arf[1:]  # subtract the -slope*k term
        pmi = np.empty(K, np.int32)
        pmi[0] = 0
        pmi[1:] = pa[:-1]
        # backward source: max over j>k of dp[j] - back_slope*(j-k)
        #   = max_{j>k}(dp[j] - back_slope*j) + back_slope*k  (suffix max of g_b)
        g_b = dp - back_slope * arf
        rm, ra = _cummax_arg(g_b[::-1])
        sm_inc = rm[::-1]
        sa_inc = (K - 1 - ra)[::-1]
        smv = np.empty(K)
        smv[-1] = -np.inf
        smv[:-1] = sm_inc[1:]
        smv[:-1] += back_slope * arf[:-1]  # add the +slope*k term
        smi = np.empty(K, np.int32)
        smi[-1] = 0
        smi[:-1] = sa_inc[1:]
        cand = np.stack([dp, pmv - lam_fwd, smv - lam_back])  # stay / fwd / back
        csrc = np.stack([ar, pmi, smi])
        ci = cand.argmax(0)
        dp = reward[t] + cand[ci, ar]
        src[t] = csrc[ci, ar]
    end = int(dp.argmax())
    path = np.empty(tm, dtype=np.int32)
    cur = end
    for t in range(tm - 1, -1, -1):
        path[t] = cur
        cur = int(src[t, cur])
    return float(dp[end]), path


def _scores_at_stretch(
    win_f: np.ndarray, ref_f: np.ndarray, st: float, wgt: np.ndarray | None = None
) -> np.ndarray:
    """Normalized matched-filter curve of one window over the whole ref (the
    proven localizer — normalizes by each ref window's energy, so it localizes
    where a raw per-frame cosine wanders).

    `wgt` (length = win frames) down-weights query frames where THIS bed is
    faded down: during a crossfade the mix is gain·thisBed + gain·otherBed, so
    low-gain frames carry mostly the other bed and only mislead the match.
    Weighting by the bed's own fader (its GT gain_curve) concentrates the
    correlation on the frames where the bed is actually audible."""
    from scipy.signal import fftconvolve

    n = win_f.shape[1]
    m = int(round(n * st))
    if m < 2 or ref_f.shape[1] <= m:
        return np.zeros(0, np.float32)
    idx = np.clip((np.arange(m) / st).astype(int), 0, n - 1)
    w = win_f[:, idx]
    if wgt is not None:
        w = w * np.asarray(wgt, np.float32)[idx][None, :]
    w = w / (np.linalg.norm(w) + 1e-9)
    num = fftconvolve(ref_f, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
    e = np.concatenate([[0.0], np.cumsum((ref_f**2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    return (num / den).astype(np.float32)


def warp_jump_slopes(
    fps: float, *, kappa: float, prior_path: Path | None = None
) -> tuple[float, float]:
    """Per-ref-frame (fwd, back) jump slopes calibrated from the warp prior.

    The empirical DJ ref-jump distribution (``warp_prior.json`` cut_up) is
    forward-skewed: forward skips reach ~+p90 s, backward replays only ~p10 s
    and are rarer. We set each direction's per-second slope so a jump at that
    direction's tail percentile costs a fixed ``kappa`` — so a big *forward*
    skip (large p90) is cheap per second while a *backward* jump (small |p10|)
    is expensive per second. Both scale to frames by /fps. This graduates the
    flat lam penalty by jump magnitude, nudging the within-fiber tie toward the
    nearer, prior-consistent instance (vs the flat penalty's arbitrary pick)."""
    prior_path = prior_path or (
        Path(__file__).resolve().parent / "synthetic_mix" / "warp_prior.json"
    )
    cut = json.loads(prior_path.read_text())["cut_up"]["ref_jump_s"]
    p90 = abs(float(cut["p90"])) or 91.4
    p10 = abs(float(cut["p10"])) or 38.4
    return kappa / (p90 * fps), kappa / (p10 * fps)


def decode_path(
    M: np.ndarray,
    R: np.ndarray,
    stretches: tuple[float, ...],
    lam: float,
    wlen_frames: int = 516,  # ~12 s matched-filter window
    hop_frames: int = 86,  # ~2 s window hop
    lam_back: float | None = None,  # backward-jump penalty (monotonic prior)
    weight: np.ndarray | None = None,  # per-span-frame fader gain (gain_curve)
    fwd_slope: float = 0.0,  # per-ref-frame forward-jump cost (0 = flat, warp-off)
    back_slope: float = 0.0,  # per-ref-frame backward-jump cost (0 = flat)
) -> tuple[list[tuple[float, float, float]], float]:
    """(segments, score). M=(D,Tm) span, R=(D,Tr) ref, both L2-normed per col.

    segments = [(mix_start_s, ref_start_s, ref_end_s)] relative to span start.

    Decode a piecewise-linear path over WINDOWED matched-filter emissions (not
    raw per-frame cosine — that's too noisy and, on the full mix, each frame is
    ref+other-layers). For each stretch we slide a ~12 s window across the span;
    each window's normalized score curve over ref offsets is the emission. A
    Viterbi over the clip-start state (offset) lets the path stay on one
    diagonal (free) or jump to another section (cost `lam`) — so linear spans
    decode to one segment, section-jumps to several, loops to backward jumps."""
    tm = M.shape[1]
    if tm < 8 or R.shape[1] < wlen_frames:
        return [], -1.0
    last = max(1, tm - wlen_frames + 1)
    aps = list(range(0, last, hop_frames)) or [0]
    best = None
    for s in stretches:
        curves, rel = [], []
        for ap in aps:
            win = np.ascontiguousarray(M[:, ap : ap + wlen_frames])
            if win.shape[1] < wlen_frames // 2:
                continue
            wgt = None if weight is None else weight[ap : ap + win.shape[1]]
            c = _scores_at_stretch(win, R, s, wgt)
            if c.size:
                curves.append(c)
                rel.append(ap)
        if not curves:
            continue
        # emission E[p, r0] = curve_p at clip-start r0 (= ref start minus the
        # window's own diagonal advance round(rel_p * s))
        shifts = [int(round(r * s)) for r in rel]
        # keep rel alongside the (c, sh) filter — else the stored `rel` (all
        # windows) desyncs from `path_r0` (only valid windows), overrunning the
        # segment loop below when a large stretch drops late windows.
        valid = [(c, sh, r) for c, sh, r in zip(curves, shifts, rel) if c.size - sh > 1]
        if not valid:
            continue
        lr0 = min(c.size - sh for c, sh, _ in valid)
        e = np.stack([c[sh : sh + lr0] for c, sh, _ in valid]).astype(np.float32)
        score, path_r0 = _viterbi(
            e, lam, lam_back, fwd_slope=fwd_slope, back_slope=back_slope
        )  # path over windows
        rel_v = [r for _, _, r in valid]
        if best is None or score > best[0]:
            best = (score, s, np.asarray(rel_v), path_r0)
    if best is None:
        return [], -1.0
    score, s, rel, path_r0 = best
    # collapse runs of equal clip-start into segments; ref advances at slope s
    segs: list[tuple[float, float, float]] = []
    p0 = 0
    P = len(rel)
    for p in range(1, P + 1):
        if p == P or path_r0[p] != path_r0[p - 1]:
            r0 = int(path_r0[p0])
            u0 = int(rel[p0])
            u1 = tm if p == P else int(rel[p])  # extend last run to span end
            mix_start = u0 / FPS
            ref_start = (r0 + u0 * s) / FPS
            ref_end = (r0 + u1 * s) / FPS
            segs.append((mix_start, ref_start, ref_end))
            p0 = p
    return segs, score


# --- trajectory scoring (works for linear / loop / multi-seg alike) -------
# Piecewise ref(mix_t) evaluation lives in core.timebase.Trajectory (one
# implementation shared by decoder + scorers). These shims keep the historic
# list-of-tuples calling convention; new code should build Trajectory directly.
def _pieces(seg_list, span_start: float, span_end: float, default_slope: float):
    """Normalize a segment list to [(mix_lo, mix_hi, ref_at_lo, slope)] in
    absolute mix seconds, for piecewise-linear ref(mix_t) interpolation."""
    return list(
        Trajectory.from_segments(
            "mix", "ref", seg_list, span_end=span_end, default_slope=default_slope
        ).pieces
    )


def _gt_pieces(row: dict):
    segs = row.get("ref_segments")
    s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
    slope = float(row.get("tempo_ratio") or 1.0)
    if segs:
        seq = [
            (float(s["mix_start_s"]), float(s["ref_start_s"]), float(s["ref_end_s"]))
            for s in segs
        ]
        return _pieces(seq, s0, s1, slope)
    return list(
        Trajectory.linear(
            "mix", "ref", span_start=s0, span_end=s1,
            dst_start=float(row["ref_start_s"]), slope=slope,
        ).pieces
    )


def _ref_at(pieces, t: float) -> float:
    return Trajectory("mix", "ref", tuple(pieces)).value_at(t)


def trajectory_acc(
    pred_segs, row: dict, tol: float = 2.0, step: float = 1.0, fiber=None
) -> tuple[float, int, float]:
    """(strict_acc, n_pred_segments, fiber_acc).

    strict_acc = fraction of sampled mix times whose predicted ref is within
    `tol` s of GT. fiber_acc additionally credits a sample when predicted and
    GT ref fall in the SAME self-repeat class (fiber=(labels, label_hz)) — i.e.
    the decoder picked a different-but-equivalent repeat. With no fiber it
    equals strict_acc."""
    s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
    if s1 <= s0 or not pred_segs:
        return 0.0, len(pred_segs), 0.0
    gt = _gt_pieces(row)
    slope = float(row.get("tempo_ratio") or 1.0)
    # decode_path returns mix-start RELATIVE to the span, ref times absolute
    pred = _pieces([(s0 + ms, rs, re) for (ms, rs, re) in pred_segs], s0, s1, slope)
    ts = np.arange(s0, s1, step)
    pr = np.array([_ref_at(pred, t) for t in ts])
    gr = np.array([_ref_at(gt, t) for t in ts])
    near = np.abs(pr - gr) < tol
    strict = float(near.mean())
    if fiber is None:
        return strict, len(pred_segs), strict
    from workspaces.alignment_prototype.ref_fibers import same_fiber

    labels, hz = fiber
    # same_fiber excludes label -1 (silence / ungrouped): two ungrouped frames
    # are NOT equivalent, so within-fiber credit only applies to real repeats.
    eq = np.array(
        [same_fiber(labels, hz, float(p), float(gq)) for p, gq in zip(pr, gr)]
    )
    return strict, len(pred_segs), float((near | eq).mean())


def _span_class(row: dict) -> str:
    if row.get("is_loop"):
        return "loop"
    if row.get("ref_segments"):
        return "multiseg"
    r = float(row.get("tempo_ratio") or 1.0)
    if not (0.9 <= r <= 1.15):
        return "oddratio"
    return "linear"


# --- stretch band: grid center x octave multiples x fine ------------------
def _stretch_band(t, mix_series, ref_series) -> tuple[float, ...]:
    e = 1.0
    if t.recording_id in ref_series:
        j = int(np.searchsorted(mix_series.start_s, t.set_start_s))
        lo, hi = max(0, j - 2), min(mix_series.n_measures, j + 3)
        mix_bar = float(np.median(mix_series.end_s[lo:hi] - mix_series.start_s[lo:hi]))
        rser = ref_series[t.recording_id]
        ref_bar = float(np.median(rser.end_s - rser.start_s))
        if mix_bar > 0 and ref_bar > 0:
            e = ref_bar / mix_bar
            while e > 1.45:
                e *= 0.5
            while e < 0.7:
                e *= 2.0
    fine = (0.96, 0.98, 1.0, 1.02, 1.04)
    band = set()
    for oct_mult in (0.5, 1.0, 2.0):  # admit genuine half / double-time clips
        for f in fine:
            band.add(round(e * oct_mult * f, 4))
    return tuple(sorted(band))


def _job(args: tuple) -> dict:
    (
        idx,
        mix_npy,
        a,
        n,
        ref_npy,
        stretches,
        lam,
        wlen,
        hop,
        lam_back,
        fwd_sl,
        back_sl,
    ) = args
    M = np.load(mix_npy, mmap_mode="r")[:, a : a + n]
    R = np.load(ref_npy, mmap_mode="r")
    M = np.ascontiguousarray(M, dtype=np.float32)
    R = np.ascontiguousarray(R, dtype=np.float32)
    segs, score = decode_path(
        M,
        R,
        tuple(stretches),
        lam,
        wlen,
        hop,
        lam_back,
        fwd_slope=fwd_sl,
        back_slope=back_sl,
    )
    return {"idx": idx, "segs": segs, "score": round(score, 3)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", action="store_true")
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--feature", choices=["chroma", "hubert"], default="chroma")
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--stems", default="regular,acappella,instrumental")
    p.add_argument("--lam", type=float, default=0.15, help="forward-jump penalty")
    p.add_argument(
        "--lam-back",
        type=float,
        default=None,
        help="backward-jump penalty. Default = --lam (symmetric/neutral). "
        "Raising it favors forward-consistent instances, but the cleaner answer "
        "to repeat-ambiguity is to score fibers as equivalent (--fibers), not "
        "to chase a specific instance (the lam-back sweep showed it hurts).",
    )
    p.add_argument("--window-s", type=float, default=12.0, help="matched-filter window")
    p.add_argument("--hop-s", type=float, default=2.0, help="window hop")
    p.add_argument(
        "--fibers",
        action="store_true",
        help="also report fiber-aware accuracy (credit same self-repeat class — "
        "isolates true placement error from within-fiber repeat ambiguity)",
    )
    p.add_argument(
        "--warp-jump",
        action="store_true",
        help="grade the jump penalty by magnitude, calibrated from the warp "
        "prior's ref-jump percentiles (forward skips cheap, backward jumps "
        "expensive). Breaks the within-fiber tie toward the nearer instance. "
        "NB the flat --lam-back sweep HURT — this is a magnitude-graded variant, "
        "A/B it against baseline (default off) before trusting it.",
    )
    p.add_argument(
        "--warp-kappa",
        type=float,
        default=None,
        help="cost (in emission units) of a tail-percentile jump; default = --lam",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="write per-span predicted segment lists as JSON (for external "
        "scorers, e.g. looptrace.eval clone-aware accuracy)",
    )
    args = p.parse_args(argv)
    if not args.eval:
        p.error("only --eval is wired")
    want_stems = {s.strip() for s in args.stems.split(",") if s.strip()}
    fwd_sl, back_sl = (0.0, 0.0)
    if args.warp_jump:
        fwd_sl, back_sl = warp_jump_slopes(
            FPS, kappa=(args.warp_kappa if args.warp_kappa is not None else args.lam)
        )
        print(
            f"warp-jump ON: fwd_slope={fwd_sl:.4g} back_slope={back_sl:.4g} /frame",
            file=sys.stderr,
        )

    import yaml as _yaml
    from core.result import Err, Ok
    from workspaces.alignment_prototype.dataset import load_set
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    match load_set(args.gt):
        case Err(msg):
            sys.exit(f"GT load failed: {msg}")
        case Ok((gt, targets)):
            pass
    match load_bb12_mert(gt.set_id):
        case Err(msg):
            sys.exit(f"grid load failed: {msg}")
        case Ok((_sid, mix_series, ref_series)):
            pass

    raw = {
        (str(r.get("slot_label")), round(float(r.get("set_start_s", -1)), 2)): r
        for r in _yaml.safe_load(args.gt.read_text()).get("tracks", [])
    }

    set_dir = find_aligning_dir(gt.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    # precompute mix-channel features (serial; MPS not fork-safe for hubert)
    mix_npy: dict[str, Path] = {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        if stem not in want_stems:
            continue
        f = set_dir / fname
        if not f.is_file():
            continue
        print(f"{args.feature}({fname}) …", file=sys.stderr)
        mix_npy[stem] = _ensure_feat(
            f, f"{gt.set_id}_{stem}", args.feature, args.hubert_layer
        )

    jobs, meta, skipped = [], [], 0
    for i, t in enumerate(targets):
        if t.slot_label == "mix":
            continue
        stem = t.claimed_stem or "regular"
        if stem not in want_stems:
            continue
        row = raw.get((t.slot_label, round(t.set_start_s, 2)))
        if row is None or (row.get("ref_source") == "online_candidate"):
            continue
        track = by_tid.get(t.recording_id)
        if track is None:
            skipped += 1
            continue
        ref_path = None
        sk = _STEM_FILE.get(stem)
        if sk:
            sp = (track.get("stems") or {}).get(sk)
            if sp and Path(sp).is_file():
                ref_path = sp
        if ref_path is None:
            ref_path = track.get("local_path")
        if not ref_path or not Path(ref_path).is_file():
            skipped += 1
            continue
        mnpy = mix_npy.get(stem) or mix_npy.get("regular")
        if mnpy is None:
            skipped += 1
            continue
        ref_npy = _ensure_feat(ref_path, ref_path, args.feature, args.hubert_layer)
        a = int(t.set_start_s * FPS)
        n = int(max(0.0, t.set_end_s - t.set_start_s) * FPS)
        if n < 4:
            skipped += 1
            continue
        stretches = _stretch_band(t, mix_series, ref_series)
        wlen = int(args.window_s * FPS)
        hop = int(args.hop_s * FPS)
        lam_back = args.lam if args.lam_back is None else args.lam_back
        jobs.append(
            (
                i,
                str(mnpy),
                a,
                n,
                str(ref_npy),
                stretches,
                args.lam,
                wlen,
                hop,
                lam_back,
                fwd_sl,
                back_sl,
            )
        )
        meta.append((t, row, str(ref_npy), str(ref_path)))

    print(
        f"decoding {len(jobs)} spans ({skipped} no-audio) "
        f"feature={args.feature} lam={args.lam}…"
    )
    res: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_job, jobs, chunksize=2)):
            res[r["idx"]] = r
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    fiber_cache: dict[str, tuple] = {}

    def fibers_for(ref_audio: str):
        # Fibers ALWAYS on HuBERT + silence-gate — NOT the decode feature: chroma
        # is too self-similar and blobs the whole track into one fiber (fake 100%
        # fiber-aware). HuBERT (phonetic) + RMS gating gives real repeat classes.
        if ref_audio not in fiber_cache:
            from workspaces.alignment_prototype.ref_fibers import compute_fibers

            hf = np.load(
                _ensure_feat(ref_audio, ref_audio, "hubert", args.hubert_layer)
            )
            fiber_cache[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
        return fiber_cache[ref_audio]

    from workspaces.alignment_prototype.ref_fibers import fiber_ambiguity

    rows = []
    for (i, *_), (t, row, ref_npy, ref_audio) in zip(jobs, meta):
        r = res[i]
        fib = fibers_for(ref_audio) if args.fibers else None
        acc, n_pred, facc = trajectory_acc(r["segs"], row, fiber=fib)
        gt_n = len(row.get("ref_segments") or [1])
        # ref-instance ambiguity of THIS decode: does the predicted ref_start land
        # in a repeat class (>=2 instances)? — the signal fiber_gate keys on. amb
        # correlated with a strict MISS validates the gate (flagging = wrong-prone).
        ambiguous, n_inst = False, 1
        if fib is not None and r["segs"]:
            labels, hz = fib
            info = fiber_ambiguity(labels, hz, float(r["segs"][0][1]))
            ambiguous, n_inst = info["ambiguous"], info["n_instances"]
        rows.append(
            (
                _span_class(row),
                t.claimed_stem or "regular",
                acc,
                n_pred,
                gt_n,
                t.slot_label,
                t.label or "",
                facc,
                ambiguous,
                n_inst,
            )
        )

    if args.dump:
        payload = []
        for (i, *_), (t, row, _ref_npy, ref_audio) in zip(jobs, meta):
            payload.append(
                {
                    "slot_label": t.slot_label,
                    "track_id": row.get("track_id"),
                    "set_start_s": t.set_start_s,
                    "set_end_s": t.set_end_s,
                    "claimed_stem": t.claimed_stem or "regular",
                    "span_class": _span_class(row),
                    "ref_audio": ref_audio,
                    "segs": [list(s) for s in res[i]["segs"]],
                    "score": res[i]["score"],
                }
            )
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(
            json.dumps(
                {
                    "gt": str(args.gt),
                    "feature": args.feature,
                    "lam": args.lam,
                    "spans": payload,
                },
                indent=1,
            )
        )
        print(f"dumped {len(payload)} span predictions -> {args.dump}")

    def rep(name, sel):
        if not sel:
            return
        acc = np.array([r[2] for r in sel])
        extra = ""
        if args.fibers:
            facc = np.array([r[7] for r in sel])
            extra = f"   ||  fiber-aware mean {100 * facc.mean():3.0f}%"
        print(
            f"  {name:20} n={len(sel):3}  traj-acc(<2s) mean {100 * acc.mean():3.0f}%  "
            f">=80% covered: {100 * (acc >= 0.8).mean():3.0f}%{extra}"
        )

    print(f"\n=== path decode — trajectory accuracy ({args.feature}) ===")
    rep("ALL", rows)
    for cls in ("linear", "multiseg", "loop", "oddratio"):
        rep(cls, [r for r in rows if r[0] == cls])
    print("  by stem:")
    for st in ("regular", "acappella", "instrumental"):
        rep(f"  {st}", [r for r in rows if r[1] == st])

    if args.fibers:
        _instance_calibration(rows)
    return 0


def _instance_calibration(rows: list, tol: float = 0.5) -> None:
    """Ref-INSTANCE metric: is the predicted ref on the exact instance, and does
    the fiber-ambiguity flag predict when it isn't?

    A span is instance-correct if its strict traj-acc (exact <2s ref) covers
    >=``tol`` of the span; fiber-correct if the repeat-equivalent acc does. The
    gap = the repeat-ambiguity cost. Bucketing by the ambiguity FLAG (predicted
    ref falls in a >=2-instance class) tells whether the flag is a calibrated
    abstain signal: if flagged spans are much less instance-correct than
    unflagged, fiber_gate is right to down-weight them (and an instance selector
    has headroom exactly there). Reported per stem — acappella is the axis."""

    def _bucket(sel: list) -> str:
        if not sel:
            return "n=0"
        inst = np.array([r[2] >= tol for r in sel], float)
        fib = np.array([r[7] >= tol for r in sel], float)
        return (
            f"n={len(sel):3}  instance-acc={100 * inst.mean():3.0f}%  "
            f"fiber-acc={100 * fib.mean():3.0f}%  "
            f"gap(repeat-ambig)={100 * (fib.mean() - inst.mean()):+3.0f}pp"
        )

    print("\n=== ref-INSTANCE calibration (fiber_gate validation) ===")
    for st in ("acappella", "regular", "instrumental", None):
        sub = rows if st is None else [r for r in rows if r[1] == st]
        if not sub:
            continue
        flagged = [r for r in sub if r[8]]  # ambiguous flag
        clean = [r for r in sub if not r[8]]
        name = st or "ALL"
        print(f"  [{name}]")
        print(f"    flagged (fiber>=2) : {_bucket(flagged)}")
        print(f"    unflagged          : {_bucket(clean)}")
        if flagged and clean:
            lift = 100 * (
                np.mean([r[2] >= tol for r in clean])
                - np.mean([r[2] >= tol for r in flagged])
            )
            verdict = "GATE VALID" if lift > 5 else "gate weak/uninformative"
            print(
                f"    => unflagged − flagged instance-acc = {lift:+.0f}pp  ({verdict})"
            )


if __name__ == "__main__":
    sys.exit(main())

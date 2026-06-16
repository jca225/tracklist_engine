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
    reward: np.ndarray, lam_fwd: float, lam_back: float | None = None
) -> tuple[float, np.ndarray]:
    """Best piecewise-constant-offset path with a DIRECTIONAL jump penalty.

    reward[t, k] is the emission for window t at clip-start state k (larger k =
    later in the ref). Staying is free; a FORWARD jump (k increases — the DJ
    skips ahead, 51% of plays) costs lam_fwd; a BACKWARD jump (replay an earlier
    section) costs lam_back >> lam_fwd. This is the monotonic prior: among
    equivalent repeats (same fiber) the path prefers the forward-consistent
    instance, converting within-fiber ambiguity into correct placement.

    O(K) per step via exclusive prefix max (best forward source, j<k) and
    suffix max (best backward source, j>k)."""
    if lam_back is None:
        lam_back = lam_fwd
    tm, K = reward.shape
    dp = reward[0].astype(np.float64).copy()
    src = np.empty((tm, K), dtype=np.int32)
    ar = np.arange(K)
    for t in range(1, tm):
        # forward source: max over j<k (exclusive prefix)
        pm, pa = _cummax_arg(dp)
        pmv = np.empty(K)
        pmv[0] = -np.inf
        pmv[1:] = pm[:-1]
        pmi = np.empty(K, np.int32)
        pmi[0] = 0
        pmi[1:] = pa[:-1]
        # backward source: max over j>k (exclusive suffix) via reversed prefix
        rm, ra = _cummax_arg(dp[::-1])
        sm_inc = rm[::-1]
        sa_inc = (K - 1 - ra)[::-1]
        smv = np.empty(K)
        smv[-1] = -np.inf
        smv[:-1] = sm_inc[1:]
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


def _scores_at_stretch(win_f: np.ndarray, ref_f: np.ndarray, st: float) -> np.ndarray:
    """Normalized matched-filter curve of one window over the whole ref (the
    proven localizer — normalizes by each ref window's energy, so it localizes
    where a raw per-frame cosine wanders)."""
    from scipy.signal import fftconvolve

    n = win_f.shape[1]
    m = int(round(n * st))
    if m < 2 or ref_f.shape[1] <= m:
        return np.zeros(0, np.float32)
    idx = np.clip((np.arange(m) / st).astype(int), 0, n - 1)
    w = win_f[:, idx]
    w = w / (np.linalg.norm(w) + 1e-9)
    num = fftconvolve(ref_f, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
    e = np.concatenate([[0.0], np.cumsum((ref_f**2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    return (num / den).astype(np.float32)


def decode_path(
    M: np.ndarray,
    R: np.ndarray,
    stretches: tuple[float, ...],
    lam: float,
    wlen_frames: int = 516,  # ~12 s matched-filter window
    hop_frames: int = 86,  # ~2 s window hop
    lam_back: float | None = None,  # backward-jump penalty (monotonic prior)
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
            c = _scores_at_stretch(win, R, s)
            if c.size:
                curves.append(c)
                rel.append(ap)
        if not curves:
            continue
        # emission E[p, r0] = curve_p at clip-start r0 (= ref start minus the
        # window's own diagonal advance round(rel_p * s))
        shifts = [int(round(r * s)) for r in rel]
        valid = [(c, sh) for c, sh in zip(curves, shifts) if c.size - sh > 1]
        if not valid:
            continue
        lr0 = min(c.size - sh for c, sh in valid)
        e = np.stack([c[sh : sh + lr0] for c, sh in valid]).astype(np.float32)
        score, path_r0 = _viterbi(e, lam, lam_back)  # path over windows
        if best is None or score > best[0]:
            best = (score, s, np.asarray(rel), path_r0)
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
def _pieces(seg_list, span_start: float, span_end: float, default_slope: float):
    """Normalize a segment list to [(mix_lo, mix_hi, ref_at_lo, slope)] in
    absolute mix seconds, for piecewise-linear ref(mix_t) interpolation."""
    out = []
    for i, (ms, rs, re) in enumerate(seg_list):
        me = seg_list[i + 1][0] if i + 1 < len(seg_list) else span_end
        dur = max(me - ms, 1e-6)
        slope = (re - rs) / dur if (re - rs) else default_slope
        out.append((ms, me, rs, slope))
    return out


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
    return [(s0, s1, float(row["ref_start_s"]), slope)]


def _ref_at(pieces, t: float) -> float:
    for ms, me, rs, slope in pieces:
        if ms <= t <= me:
            return rs + (t - ms) * slope
    if t < pieces[0][0]:
        ms, _me, rs, slope = pieces[0]
        return rs + (t - ms) * slope
    ms, me, rs, slope = pieces[-1]
    return rs + (t - ms) * slope


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
    from workspaces.alignment_prototype.ref_fibers import fiber_at

    labels, hz = fiber
    eq = np.array(
        [
            fiber_at(labels, hz, float(p)) == fiber_at(labels, hz, float(gq))
            for p, gq in zip(pr, gr)
        ]
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
    idx, mix_npy, a, n, ref_npy, stretches, lam, wlen, hop, lam_back = args
    M = np.load(mix_npy, mmap_mode="r")[:, a : a + n]
    R = np.load(ref_npy, mmap_mode="r")
    M = np.ascontiguousarray(M, dtype=np.float32)
    R = np.ascontiguousarray(R, dtype=np.float32)
    segs, score = decode_path(M, R, tuple(stretches), lam, wlen, hop, lam_back)
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
    p.add_argument("--fiber-k", type=int, default=6, help="sections per ref")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    if not args.eval:
        p.error("only --eval is wired")
    want_stems = {s.strip() for s in args.stems.split(",") if s.strip()}

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
            )
        )
        meta.append((t, row, str(ref_npy)))

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

    def fibers_for(ref_npy: str):
        if ref_npy not in fiber_cache:
            from workspaces.alignment_prototype.ref_fibers import compute_fibers

            fiber_cache[ref_npy] = compute_fibers(np.load(ref_npy), FPS, k=args.fiber_k)
        return fiber_cache[ref_npy]

    rows = []
    for (i, *_), (t, row, ref_npy) in zip(jobs, meta):
        r = res[i]
        fib = fibers_for(ref_npy) if args.fibers else None
        acc, n_pred, facc = trajectory_acc(r["segs"], row, fiber=fib)
        gt_n = len(row.get("ref_segments") or [1])
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
            )
        )

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
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Oracle instance-selection separability + cross-set transfer (Task A).

Parameter-free gate: within the recoverable acappella population (spans placed in
the right fiber but the wrong instance), can {HuBERT diagonal evidence, fiber
μ/ambiguity, fp sharpness} rank the GT-correct fiber member above its same-fiber
rivals, and does that ranking transfer BB11<->BB12? Design + decision rule:
docs/superpowers/specs/2026-07-18-instance-separability-design.md.

This module MEASURES separability and emits the selector's future training dataset;
it fits no model claimed as a win (n=2 -> that waits for BB10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from workspaces.alignment_prototype.fibers.detect import fiber_at, fiber_intervals
from workspaces.alignment_prototype.landmark_fp import FHOP as _FHOP
from workspaces.alignment_prototype.landmark_fp import SR as _SR
from workspaces.alignment_prototype.path_decode import FPS as _FPS
from workspaces.alignment_prototype.path_decode import _scores_at_stretch

_FP_HZ = _SR / _FHOP  # fp offset bins per second

# Feature keys, in fixed order (used for the equal-weight combo and transfer).
# NB: the fp feature is per-instance vote MASS at the candidate's offset — global
# vote_sharpness is constant across a row's candidates and so cannot rank within it.
FEATURES: tuple[str, ...] = ("hubert", "mu", "fp")


@dataclass
class Candidate:
    """One ref instance within the GT fiber, a candidate for what the DJ played."""

    ref_start_s: float
    ref_end_s: float
    label: int
    is_gt_instance: bool
    features: dict[str, float | None] = field(default_factory=dict)


def enumerate_candidates(
    labels: np.ndarray,
    label_hz: float,
    gt_ref_start_s: float,
    *,
    min_len_s: float = 4.0,
) -> list[Candidate]:
    """Candidate ref instances = members of the fiber the GT placement lives in.

    The GT-correct instance is the interval whose [start, end) contains
    ``gt_ref_start_s``. Returns [] when the GT frame is not in any repeat class
    (fiber_id < 0) — there is then no instance to select.
    """
    fid = fiber_at(labels, label_hz, gt_ref_start_s)
    if fid < 0:
        return []
    out: list[Candidate] = []
    for s, e, lab in fiber_intervals(labels, label_hz, min_len_s):
        if lab != fid:
            continue
        is_gt = s <= gt_ref_start_s < e
        out.append(
            Candidate(ref_start_s=s, ref_end_s=e, label=int(lab), is_gt_instance=is_gt)
        )
    return out


def is_selectable(cands: list[Candidate]) -> bool:
    """A genuine instance choice exists: >= 2 members and the GT is among them."""
    return len(cands) >= 2 and any(c.is_gt_instance for c in cands)


# --------------------------------------------------------------------------
# Unit 3 — per-candidate feature extraction (pure over injected arrays)
# --------------------------------------------------------------------------


@dataclass
class FeatureContext:
    """Everything the three features read, pre-loaded once per population row.

    Making the arrays explicit keeps ``candidate_features`` pure and unit-testable;
    the I/O that fills this (``build_context``) is exercised by the integration run.
    """

    mix_win: np.ndarray  # (D, n) mix-vocal HuBERT window over the GT play interval
    ref_feat: np.ndarray  # (D, T) ref-vocal HuBERT features (whole track)
    stretch: float  # tempo_ratio (mix/ref), for the matched-filter resample
    labels: np.ndarray  # ref fiber labels
    label_hz: float
    mu: np.ndarray | None  # soft membership per ref frame (compute_fibers_soft)
    fp_votes: dict[int, int]  # landmark offset(bins) -> vote count, mix window vs ref
    fps: float = _FPS
    fp_hz: float = _FP_HZ
    fp_tol_s: float = 1.0  # +/- window (s) of fp offset mass credited to an instance


def candidate_features(cand: Candidate, ctx: FeatureContext) -> dict[str, float | None]:
    """The three named features for one candidate instance (``None`` if unscorable).

    - hubert: matched-filter score of the mix window at this instance's ref offset.
    - mu: fiber soft-membership at the instance's ref frame.
    - fp: landmark vote MASS at this instance's offset (per-instance, not global
      sharpness — sharpness is constant across a row and cannot rank within it).
    """
    feats: dict[str, float | None] = {}

    curve = _scores_at_stretch(ctx.mix_win, ctx.ref_feat, ctx.stretch)
    if curve.size == 0:
        feats["hubert"] = None
    else:
        fi = int(round(cand.ref_start_s * ctx.fps))
        feats["hubert"] = float(curve[fi]) if 0 <= fi < curve.size else None

    if ctx.mu is not None and ctx.mu.size:
        b = int(round(cand.ref_start_s * ctx.label_hz))
        feats["mu"] = float(ctx.mu[min(max(b, 0), ctx.mu.size - 1)])
    else:
        feats["mu"] = None

    if ctx.fp_votes:
        target = int(round(cand.ref_start_s * ctx.fp_hz))
        tol = int(round(ctx.fp_tol_s * ctx.fp_hz))
        feats["fp"] = float(
            sum(v for off, v in ctx.fp_votes.items() if abs(off - target) <= tol)
        )
    else:
        feats["fp"] = None

    return feats


# --------------------------------------------------------------------------
# Unit 4 — ranking + separability metrics (pure)
# --------------------------------------------------------------------------

_NEG_INF = float("-inf")
_TIE_EPS = 1e-9


def _val_or_neg_inf(v: float | None) -> float:
    return _NEG_INF if v is None else float(v)


def top1_score(cands: list[Candidate], feature_key: str) -> float | None:
    """Expected credit that the top-ranked-by-``feature_key`` candidate is the GT one,
    under a UNIFORM random tie-break among the argmax set.

    This makes the metric independent of candidate list order — the crucial guard
    against the earliest-instance confound (feature ties silently resolving to the
    first-enumerated instance, which is the earliest and often the GT one). ``None``
    feature values rank last; returns ``None`` if every candidate is missing it.
    """
    if not cands:
        return None
    vals = [_val_or_neg_inf(c.features.get(feature_key)) for c in cands]
    if all(v == _NEG_INF for v in vals):
        return None
    top = max(vals)
    arg = [c for c, v in zip(cands, vals) if abs(v - top) <= _TIE_EPS]
    n_gt = sum(1 for c in arg if c.is_gt_instance)
    return n_gt / len(arg)


def _score_from_values(
    cands: list[Candidate], values: "list[float] | np.ndarray"
) -> float:
    """Tie-fair expected GT credit for a precomputed per-candidate score vector."""
    top = max(float(v) for v in values)
    arg = [c for c, v in zip(cands, values) if abs(float(v) - top) <= _TIE_EPS]
    n_gt = sum(1 for c in arg if c.is_gt_instance)
    return n_gt / len(arg)


def position_baseline(rows: list[list[Candidate]], which: str) -> float:
    """Fraction recovered by a fixed positional rule over SELECTABLE rows.

    ``which='earliest'`` picks the lowest-ref_start instance, ``'latest'`` the
    highest — the confound baselines the content features must beat.
    """
    sel = [r for r in rows if is_selectable(r)]
    if not sel:
        return 0.0
    hit = 0.0
    for r in sel:
        pick = (
            min(r, key=lambda c: c.ref_start_s)
            if which == "earliest"
            else max(r, key=lambda c: c.ref_start_s)
        )
        hit += 1.0 if pick.is_gt_instance else 0.0
    return hit / len(sel)


def assign_combo(
    cands: list[Candidate], keys: tuple[str, ...] = FEATURES, out_key: str = "combo"
) -> None:
    """Set ``features[out_key]`` = sum of per-row z-scored features (equal weight).

    Z-scoring is within the row (across its candidates) so the combo is a fixed,
    unfitted aggregation — not a tuned model (which would be the n=2 overfit trap).
    A feature that is missing or constant across the row contributes 0.
    """
    z = _zscore_matrix(cands, keys)  # (n_cands, n_keys)
    for i, c in enumerate(cands):
        c.features[out_key] = float(z[i].sum())


def _zscore_matrix(cands: list[Candidate], keys: tuple[str, ...]) -> np.ndarray:
    n = len(cands)
    Z = np.zeros((n, len(keys)), dtype=float)
    for j, k in enumerate(keys):
        col = np.array(
            [
                c.features.get(k) if c.features.get(k) is not None else np.nan
                for c in cands
            ],
            dtype=float,
        )
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        mu = float(finite.mean())
        sd = float(finite.std())
        if sd < 1e-12:
            continue  # constant feature carries no ranking signal -> 0
        z = (col - mu) / sd
        z[~np.isfinite(z)] = 0.0  # missing -> neutral
        Z[:, j] = z
    return Z


def gap_recovered(
    rows: list[list[Candidate]], feature_key: str
) -> tuple[float, int, float]:
    """(expected_correct, n_selectable, fraction) over the SELECTABLE rows only.

    Fraction of the strict->fiber gap a top-1-by-``feature_key`` selector would
    recover, tie-fair (expected credit under random tie-break). Single-member /
    GT-absent rows are excluded from the denominator. ``expected_correct`` is a
    float because ties contribute fractional credit.
    """
    sel = [r for r in rows if is_selectable(r)]
    scored = [top1_score(r, feature_key) for r in sel]
    n_correct = sum(s for s in scored if s is not None)
    n = len(sel)
    return n_correct, n, (n_correct / n if n else 0.0)


def fit_ranker(
    rows: list[list[Candidate]],
    keys: tuple[str, ...] = FEATURES,
    *,
    lr: float = 0.5,
    iters: int = 2000,
) -> np.ndarray:
    """Fit the simplest ranker: logistic weights over per-row z-scored features.

    Label = is_gt_instance (1) vs rival (0), pooled over all selectable rows.
    No intercept (it cancels in the within-row argmax). Deterministic full-batch
    gradient descent — low capacity by design; a meaningful LOSO iff the oracle
    separability is real. Returns a length-``len(keys)`` weight vector.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for r in rows:
        if not is_selectable(r):
            continue
        X_parts.append(_zscore_matrix(r, keys))
        y_parts.append(np.array([1.0 if c.is_gt_instance else 0.0 for c in r]))
    if not X_parts:
        return np.zeros(len(keys))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    w = np.zeros(X.shape[1])
    n = X.shape[0]
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) / n
        w -= lr * grad
    return w


def _row_scores(cands: list[Candidate], weights: np.ndarray, keys: tuple[str, ...]):
    return _zscore_matrix(cands, keys) @ weights


def transfer_gap_recovered(
    rows: list[list[Candidate]], weights: np.ndarray, keys: tuple[str, ...] = FEATURES
) -> tuple[float, int, float]:
    """Apply fitted ``weights`` to score/rank each SELECTABLE row; tie-fair top-1."""
    sel = [r for r in rows if is_selectable(r)]
    n_correct = sum(_score_from_values(r, _row_scores(r, weights, keys)) for r in sel)
    n = len(sel)
    return n_correct, n, (n_correct / n if n else 0.0)


# --------------------------------------------------------------------------
# Unit 1 + I/O glue — recoverable population, context building, runner.
# Integration over the pure units above; exercised by the real BB11/BB12 run,
# not the unit suite. Heavy audio imports stay lazy.
# --------------------------------------------------------------------------

_HUBERT_LAYER = 9
_FP_SR = 22050  # landmark fingerprint sample rate (landmark_fp.SR)


@dataclass
class PopRow:
    """A recoverable GT acappella row: decoder placed it in-fiber but off-instance."""

    gt_row: dict
    ref_audio: str
    strict: float
    fiber: float


def build_population(
    set_id: str,
    gt_path,
    r0_timeline,
    *,
    recover_margin: float = 0.1,
) -> list[PopRow]:
    """GT acappella rows the frozen (R0) decoder placed in the right fiber but the
    wrong instance — the recoverable set the strict->fiber headroom lives in.

    A row qualifies when its R0 decode's fiber-aware accuracy exceeds strict by
    ``recover_margin`` (fiber recovers placement strict missed). Rows never decoded
    (no same-recording span) or without local ref audio are dropped (logged by the
    caller via the returned count vs the GT total).
    """
    import json

    from workspaces.alignment_prototype import score_timeline_vs_gt as sc
    from workspaces.alignment_prototype.evals import oracle_ladder as ol
    from workspaces.alignment_prototype.path_decode import trajectory_acc
    from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir

    gt_acap = ol._load_gt_acap(set_id, gt_path)
    set_dir = find_aligning_dir(set_id)
    by_tid = ol._by_tid(set_dir)
    r0_spans = json.loads(open(r0_timeline).read())["spans"]

    out: list[PopRow] = []
    for g in gt_acap:
        span = ol._nearest_same_recording(g, r0_spans)
        if span is None:
            continue
        ref_audio = sc._resolve_ref_audio(
            span, by_tid.get(str(span["recording_id"])), stem="acappella"
        )
        if ref_audio is None:
            continue
        labels, label_hz = _fibers_hard(ref_audio)
        pred = sc._pred_segs_from_span(span, anchor_s=float(g["set_start_s"]))
        strict, _n, facc = trajectory_acc(pred, g, fiber=(labels, label_hz))
        if facc - strict >= recover_margin:
            out.append(PopRow(g, ref_audio, float(strict), float(facc)))
    return out


_FIBER_CACHE: dict[str, tuple] = {}
_SOFT_CACHE: dict[str, tuple] = {}


def _fibers_hard(ref_audio: str) -> tuple:
    import numpy as _np

    from workspaces.alignment_prototype.fibers.detect import compute_fibers
    from workspaces.alignment_prototype.path_decode import FPS, _ensure_feat

    if ref_audio not in _FIBER_CACHE:
        hf = _np.load(_ensure_feat(ref_audio, ref_audio, "hubert", _HUBERT_LAYER))
        _FIBER_CACHE[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
    return _FIBER_CACHE[ref_audio]


def _fibers_soft(ref_audio: str) -> tuple:
    """(labels, label_hz, mu, per_fiber_conf) — cached per ref."""
    import numpy as _np

    from workspaces.alignment_prototype.fibers.detect import compute_fibers_soft
    from workspaces.alignment_prototype.path_decode import FPS, _ensure_feat

    if ref_audio not in _SOFT_CACHE:
        hf = _np.load(_ensure_feat(ref_audio, ref_audio, "hubert", _HUBERT_LAYER))
        _SOFT_CACHE[ref_audio] = compute_fibers_soft(hf, FPS, audio_path=ref_audio)
    return _SOFT_CACHE[ref_audio]


def build_context(pop: PopRow, mix_npy, mix_path: str) -> FeatureContext:
    """Load the arrays the three features read for one recoverable row."""
    import numpy as _np

    from workspaces.alignment_prototype.path_decode import FPS, _ensure_feat

    g = pop.gt_row
    s0, s1 = float(g["set_start_s"]), float(g["set_end_s"])
    a = int(s0 * FPS)
    n = int(max(0.0, s1 - s0) * FPS)
    mix_win = _np.load(mix_npy, mmap_mode="r")[:, a : a + n]
    ref_feat = _np.load(
        _ensure_feat(pop.ref_audio, pop.ref_audio, "hubert", _HUBERT_LAYER),
        mmap_mode="r",
    )
    labels, label_hz, mu, _conf = _fibers_soft(pop.ref_audio)
    stretch = float(g.get("tempo_ratio") or 1.0)
    fp_votes = _fp_votes(mix_path, pop.ref_audio, s0, s1 - s0, stretch)
    return FeatureContext(
        mix_win=_np.ascontiguousarray(mix_win, dtype=_np.float32),
        ref_feat=ref_feat,
        stretch=stretch,
        labels=labels,
        label_hz=label_hz,
        mu=mu,
        fp_votes=fp_votes,
    )


_REF_HASH_CACHE: dict[str, dict] = {}


def _ref_hashes(ref_audio: str) -> dict:
    """Landmark hashes of the whole ref vocal, cached (re-used across its rows)."""
    import warnings

    import librosa

    from workspaces.alignment_prototype.landmark_fp import constellation, hashes

    if ref_audio not in _REF_HASH_CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_y, _ = librosa.load(ref_audio, sr=_FP_SR, mono=True)
        _REF_HASH_CACHE[ref_audio] = hashes(*constellation(ref_y)) if ref_y.size else {}
    return _REF_HASH_CACHE[ref_audio]


def _fp_votes(
    mix_path: str, ref_audio: str, mix_off_s: float, mix_dur_s: float, stretch: float
) -> dict:
    """Landmark offset-vote histogram of the mix vocal window against the whole ref.

    Offsets are in FHOP bins of ref time, directly comparable to a candidate's
    ``ref_start_s * FP_HZ``. The mix window is stretched to the ref timebase (rate =
    tempo_ratio) so the peak lands at the true ref offset. NB: landmark evidence on
    Roformer-separated vocals is known to be noise-dominated (looptrace NOTES) — a
    weak/empty fp feature here is an expected outcome, not a bug.
    """
    import warnings

    import librosa

    from workspaces.alignment_prototype.landmark_fp import (
        _vote_histogram,
        constellation,
        hashes,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mix_y, _ = librosa.load(
            mix_path, sr=_FP_SR, mono=True, offset=mix_off_s, duration=mix_dur_s
        )
        if abs(stretch - 1.0) > 1e-3 and mix_y.size > 512:
            mix_y = librosa.effects.time_stretch(mix_y, rate=float(stretch))
    if mix_y.size < 512:
        return {}
    rh = _ref_hashes(ref_audio)
    if not rh:
        return {}
    mh = hashes(*constellation(mix_y))
    return _vote_histogram(mh, rh)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


@dataclass
class SetResult:
    set_id: str
    rows: list[list[Candidate]]  # candidates per SELECTABLE recoverable row
    n_gt_acap: int
    n_recoverable: int
    n_selectable: int
    n_single_member: int  # recoverable but GT fiber has one instance (no choice)


def run_set(set_id: str, gt_path, r0_timeline, out_dir) -> SetResult:
    """Build the recoverable population, enumerate + feature-score its candidates,
    persist the per-candidate dataset (the future selector's training input)."""
    from pathlib import Path

    from workspaces.alignment_prototype.path_decode import _ensure_feat
    from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_dir = find_aligning_dir(set_id)
    mix_path = str(set_dir / "mix_vocals.flac")
    print(f"[{set_id}] hubert(mix_vocals.flac) …")
    mix_npy = _ensure_feat(mix_path, f"{set_id}_acappella", "hubert", _HUBERT_LAYER)

    pop = build_population(set_id, gt_path, r0_timeline)
    n_gt = _gt_acap_count(set_id, gt_path)

    rows: list[list[Candidate]] = []
    n_single = 0
    dataset_path = out_dir / f"{set_id}_candidates.jsonl"
    with dataset_path.open("w") as fh:
        for p in pop:
            ctx = build_context(p, mix_npy, mix_path)
            cands = enumerate_candidates(
                ctx.labels, ctx.label_hz, float(p.gt_row["ref_start_s"])
            )
            if len(cands) < 2:
                n_single += 1
                continue
            for c in cands:
                c.features = candidate_features(c, ctx)
            assign_combo(cands)
            rows.append(cands)
            _dump_row(fh, set_id, p, cands)
    print(
        f"[{set_id}] gt_acap={n_gt} recoverable={len(pop)} "
        f"selectable={len(rows)} single_member={n_single} -> {dataset_path}"
    )
    return SetResult(set_id, rows, n_gt, len(pop), len(rows), n_single)


def _gt_acap_count(set_id: str, gt_path) -> int:
    from workspaces.alignment_prototype.evals import oracle_ladder as ol

    return len(ol._load_gt_acap(set_id, gt_path))


def _dump_row(fh, set_id: str, p: PopRow, cands: list[Candidate]) -> None:
    import json

    g = p.gt_row
    key = f"{g.get('slot_label')}@{round(float(g['set_start_s']), 2)}"
    for c in cands:
        fh.write(
            json.dumps(
                {
                    "set_id": set_id,
                    "gt_row_key": key,
                    "track_id": g.get("track_id"),
                    "ref_start_s": c.ref_start_s,
                    "ref_end_s": c.ref_end_s,
                    "is_gt_instance": c.is_gt_instance,
                    "features": {k: c.features.get(k) for k in (*FEATURES, "combo")},
                }
            )
            + "\n"
        )


def _separability_table(res: SetResult) -> dict:
    """Per-feature (and combo) fraction of the strict->fiber gap recovered, plus a
    random-among-members baseline (mean 1/n_members) to read the features against."""
    tbl = {}
    for key in (*FEATURES, "combo"):
        n_correct, n_sel, frac = gap_recovered(res.rows, key)
        tbl[key] = {"n_correct": n_correct, "n_selectable": n_sel, "frac": frac}
    sel = [r for r in res.rows if is_selectable(r)]
    tbl["random"] = {
        "n_correct": None,
        "n_selectable": len(sel),
        "frac": (sum(1.0 / len(r) for r in sel) / len(sel)) if sel else 0.0,
    }
    for which in ("earliest", "latest"):
        tbl[which] = {
            "n_correct": None,
            "n_selectable": len(sel),
            "frac": position_baseline(res.rows, which),
        }
    return tbl


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from workspaces.alignment_prototype.score_timeline_vs_gt import _REPO as repo

    default_out = (
        Path(__file__).resolve().parent.parent / "out" / "instance_separability"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sets",
        default="1fsnxchk,2nvzlh2k",
        help="comma list of set_ids (default BB12,BB11)",
    )
    p.add_argument("--out-dir", type=Path, default=default_out)
    a = p.parse_args(argv)

    gt_of = {
        "1fsnxchk": repo / "labeling/fixtures/bb12_ground_truth.yaml",
        "2nvzlh2k": repo / "labeling/fixtures/bb11_ground_truth.yaml",
    }
    r0_of = {
        "1fsnxchk": repo
        / "workspaces/alignment_prototype/out/1fsnxchk_predicted_timeline_lt.json",
        "2nvzlh2k": repo
        / "workspaces/alignment_prototype/out/2nvzlh2k_predicted_timeline_lt.json",
    }

    set_ids = [s.strip() for s in a.sets.split(",") if s.strip()]
    results: dict[str, SetResult] = {}
    tables: dict[str, dict] = {}
    for sid in set_ids:
        res = run_set(sid, gt_of[sid], r0_of[sid], a.out_dir)
        results[sid] = res
        tables[sid] = _separability_table(res)
        print(
            f"\n[{sid}] oracle separability (fraction of strict->fiber gap recovered):"
        )
        for key, t in tables[sid].items():
            print(
                f"  {key:6s}: {t['frac']:.3f}  ({t['n_correct']}/{t['n_selectable']})"
            )

    transfer: dict[str, dict] = {}
    if len(set_ids) == 2:
        s1, s2 = set_ids
        for train_id, test_id in ((s1, s2), (s2, s1)):
            w = fit_ranker(results[train_id].rows)
            n_c, n_s, frac = transfer_gap_recovered(results[test_id].rows, w)
            transfer[f"{train_id}->{test_id}"] = {
                "weights": {k: float(v) for k, v in zip(FEATURES, w)},
                "n_correct": n_c,
                "n_selectable": n_s,
                "frac": frac,
            }
            print(
                f"\n[transfer {train_id}->{test_id}] top-1 = {frac:.3f} "
                f"({n_c}/{n_s})  w={transfer[f'{train_id}->{test_id}']['weights']}"
            )

    summary = {
        "sets": {
            sid: {
                "n_gt_acap": r.n_gt_acap,
                "n_recoverable": r.n_recoverable,
                "n_selectable": r.n_selectable,
                "n_single_member": r.n_single_member,
                "separability": tables[sid],
            }
            for sid, r in results.items()
        },
        "transfer": transfer,
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {a.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

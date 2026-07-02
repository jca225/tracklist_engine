#!/usr/bin/env python3
"""Step-1 reconstruction-supervision probe (docs/reconstruction_supervision_plan.md).

Load-bearing question: does reconstruction error, measured in a time-aware spectral
space, actually track alignment correctness on BB12 (where we have GT)? If yes, the
mix audio is a free answer key for all ~20k sets and the whole pretrain plan is real.
If no, we stop here — half a day spent, not two months.

Two tests:
  A (validity)  : for GT spans, score the mix window vs the placed ref segment at the
                  TRUE placement vs deliberately PERTURBED ref offsets. If the mix
                  localizes content, the score curve peaks at offset 0.
  B (usability) : on the actual predicted timeline, correlate each span's match score
                  with whether that prediction was correct vs GT (AUC).

Match metric = per-frame cosine of L2-normalized mel-magnitude (time-aware: frame t of
the ref vs frame t of the mix). Stem-routed: acappella->vocal stem, instrumental->
instrumental stem, regular->full mix (headline rests on regular full-mix, which needs
no fragile stem matching).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.recon_probe --set-id 1fsnxchk
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
except Exception as e:  # pragma: no cover
    print("librosa required (venvs/audio):", e, file=sys.stderr)
    raise

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT_DIR = Path(__file__).resolve().parent / "out"

SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 64
MIN_DUR_S = 4.0
OFFSETS_S = [-60, -45, -30, -20, -12, -6, 6, 12, 20, 30, 45, 60]
_MEL_FB = None  # lazy


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def norm_slot(s: str) -> str:
    m = re.match(r"^0*(\d+)(w\d+)?$", str(s).strip())
    return f"{m.group(1)}{m.group(2) or ''}" if m else str(s).strip()


def find_aligning_dir(set_id: str) -> Path | None:
    base = Path.home() / "aligning"
    for d in base.iterdir() if base.exists() else []:
        if d.is_dir() and d.name.startswith(set_id):
            return d
    return None


# ---- audio + features -------------------------------------------------------

_AUDIO_CACHE: dict[str, np.ndarray] = {}


def load_audio(path: str | Path) -> np.ndarray | None:
    key = str(path)
    if key in _AUDIO_CACHE:
        return _AUDIO_CACHE[key]
    if not Path(path).exists():
        return None
    try:
        y, _ = librosa.load(key, sr=SR, mono=True)
    except Exception as e:
        print(f"  load fail {Path(path).name}: {e}", file=sys.stderr)
        return None
    _AUDIO_CACHE[key] = y
    return y


def _melmag(seg: np.ndarray) -> np.ndarray | None:
    """L2-normalized mel-magnitude, shape (N_MELS, T). None if too short."""
    global _MEL_FB
    if seg.size < N_FFT * 2:
        return None
    S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP)) ** 2
    if _MEL_FB is None:
        _MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    M = np.sqrt(_MEL_FB @ S)  # (N_MELS, T)
    if M.shape[1] < 4:
        return None
    n = np.linalg.norm(M, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def match_score(mix_seg: np.ndarray, ref_seg: np.ndarray) -> float | None:
    """Mean per-frame cosine of L2-normalized mel-magnitude, frame-truncated to min T."""
    a = _melmag(mix_seg)
    b = _melmag(ref_seg)
    if a is None or b is None:
        return None
    t = min(a.shape[1], b.shape[1])
    if t < 4:
        return None
    return float(np.mean(np.sum(a[:, :t] * b[:, :t], axis=0)))


def slice_s(y: np.ndarray, start_s: float, dur_s: float) -> np.ndarray:
    i0 = max(0, int(round(start_s * SR)))
    i1 = min(len(y), i0 + int(round(dur_s * SR)))
    return y[i0:i1]


# ---- stem routing -----------------------------------------------------------

_STEM_FILE = {"acappella": "vocals", "instrumental": "instrumental"}


def find_mix_stems(aligning_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for stem in ("vocals", "instrumental"):
        for pat in (f"mix_{stem}.flac", f"mix_{stem}.wav", f"stems_mix/{stem}.flac"):
            p = aligning_dir / pat
            if p.exists():
                out[stem] = p
                break
    return out


def ref_stem_path(aligning_dir: Path, local_path: str, stem: str) -> Path | None:
    key = _STEM_FILE.get(stem)
    if not key or not local_path:
        return None
    base = Path(local_path).stem  # filename without extension
    cand = aligning_dir / "stems" / base / f"{key}.flac"
    if cand.exists():
        return cand
    # glob fallback: stem dir sharing the leading label token
    label = base.split("__")[0] if "__" in base else base[:3]
    for d in (aligning_dir / "stems").glob(f"{label}*"):
        p = d / f"{key}.flac"
        if p.exists():
            return p
    return None


# ---- AUC (Mann-Whitney) -----------------------------------------------------


def auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rank_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n1 = len(pos)
    return (rank_pos - n1 * (n1 + 1) / 2.0) / (n1 * len(neg))


# ---- main -------------------------------------------------------------------


_HUB_CACHE: dict[str, tuple[np.ndarray, float]] = {}


def hubert_feat(path: str | Path) -> tuple[np.ndarray, float] | None:
    """(768, T) L2-normed HuBERT-L9 + fps, cached per file."""
    key = str(path)
    if key in _HUB_CACHE:
        return _HUB_CACHE[key]
    if not Path(path).exists():
        return None
    from workspaces.alignment_prototype.stem_placement import hubert_of

    f = hubert_of(key, layer=9)
    if f is None or f.shape[1] < 2:
        return None
    dur = librosa.get_duration(path=key)
    fps = f.shape[1] / dur if dur > 0 else 50.0
    _HUB_CACHE[key] = (f, fps)
    return _HUB_CACHE[key]


def _slice_feat(f: np.ndarray, fps: float, start_s: float, dur_s: float) -> np.ndarray:
    i0 = max(0, int(round(start_s * fps)))
    i1 = min(f.shape[1], i0 + int(round(dur_s * fps)))
    return f[:, i0:i1]


def _feat_cos(a: np.ndarray, b: np.ndarray) -> float | None:
    t = min(a.shape[1], b.shape[1])
    if t < 3:
        return None
    return float(np.mean(np.sum(a[:, :t] * b[:, :t], axis=0)))


def _warp_cos(mix_seg: np.ndarray, ref_seg: np.ndarray) -> float | None:
    """Cosine after linearly time-warping the ref window onto the mix window's frame
    grid (i.e. rendering with the placement's implied stretch), then re-normalizing."""
    tm = mix_seg.shape[1]
    if tm < 3 or ref_seg.shape[1] < 2:
        return None
    src = np.linspace(0.0, 1.0, ref_seg.shape[1])
    dst = np.linspace(0.0, 1.0, tm)
    r = np.stack([np.interp(dst, src, row) for row in ref_seg])
    r /= np.linalg.norm(r, axis=0, keepdims=True) + 1e-8
    m = mix_seg / (np.linalg.norm(mix_seg, axis=0, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(m * r, axis=0)))


def test_a_hubert_acappella(aligning, manifest, by_tid, mix_stems, gt_rows) -> None:
    """Re-run Test A on acappella spans with HuBERT (key/pitch-invariant) + fiber
    repeat-equivalence (an offset landing on an equivalent chorus is not counted as a
    competing peak). Isolates HuBERT vs mel and the effect of fibers."""
    from workspaces.alignment_prototype.ref_fibers import (
        compute_fibers_soft,
        same_fiber,
    )

    if "vocals" not in mix_stems:
        print("no mix_vocals stem — cannot run HuBERT acappella test", file=sys.stderr)
        return
    print("\n== HuBERT + fibers re-run (acappella) ==")
    print("loading mix_vocals HuBERT (once)...")
    mv = hubert_feat(mix_stems["vocals"])
    if mv is None:
        print("mix_vocals HuBERT failed", file=sys.stderr)
        return
    mix_f, mix_fps = mv

    def audible(r) -> bool:
        af = r.get("audible_frac")
        return af is None or float(af) > 0.1

    # every audible GT row that occupies mix time (for depth = concurrent-layer count)
    time_rows = [
        r
        for r in gt_rows
        if r.get("track_id")
        and audible(r)
        and r.get("set_start_s") is not None
        and r.get("set_end_s") is not None
    ]

    def depth_at(ss, se, self_slot) -> tuple[int, int]:
        """(total audible layers, overlaid ACAPPELLA layers) covering [ss,se], incl self."""
        tot = voc = 0
        for r in time_rows:
            if float(r["set_start_s"]) < se - 1 and float(r["set_end_s"]) > ss + 1:
                tot += 1
                if (r.get("claimed_stem") or "") == "acappella":
                    voc += 1
        return tot, voc

    # per-span records: (voc_depth, hit_nowarp, s0_nowarp)
    recs: list[tuple[int, int, float]] = []
    for row in gt_rows:
        if (row.get("claimed_stem") or "") != "acappella" or not row.get("track_id"):
            continue
        t = by_tid.get(str(row["track_id"]))
        if t is None:
            continue
        sp = ref_stem_path(aligning, t.get("local_path") or "", "acappella")
        if sp is None:
            continue
        try:
            ss, se = float(row["set_start_s"]), float(row["set_end_s"])
            rs, re_ = float(row["ref_start_s"]), float(row["ref_end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        dur = se - ss
        if dur < MIN_DUR_S:
            continue
        rf = hubert_feat(sp)
        if rf is None:
            continue
        ref_f, ref_fps = rf
        ref_dur = re_ - rs if (re_ - rs) >= MIN_DUR_S else dur
        mix_seg = _slice_feat(mix_f, mix_fps, ss, dur)
        # un-warped HuBERT cosine (warp-in-feature-space was proven invalid)
        s0 = _feat_cos(mix_seg, _slice_feat(ref_f, ref_fps, rs, ref_dur))
        if s0 is None:
            continue
        others = []
        for d in OFFSETS_S:
            sd = _feat_cos(mix_seg, _slice_feat(ref_f, ref_fps, rs + d, ref_dur))
            if sd is not None:
                others.append(sd)
        if not others:
            continue
        _tot, voc = depth_at(ss, se, row.get("slot_label"))
        recs.append((voc, int(s0 > max(others)), s0))

    if not recs:
        print("no scorable acappella spans", file=sys.stderr)
        return

    print(
        f"\n  acappella n={len(recs)}  (HuBERT-L9, un-warped; split by concurrent vocal depth)"
    )
    print(
        "  hypothesis: solo (1 vocal) reconstructs; deep stacks (2-3 overlaid) stay flat\n"
    )
    tiers = [
        ("solo (1 vocal)", lambda v: v <= 1),
        ("2 vocals", lambda v: v == 2),
        ("3+ vocals", lambda v: v >= 3),
    ]
    print(f"  {'tier':<18} {'n':>4} {'peak-at-0':>10} {'med true':>9}")
    for name, pred in tiers:
        g = [(h, s) for v, h, s in recs if pred(v)]
        if not g:
            print(f"  {name:<18} {'0':>4}      —")
            continue
        hits = [h for h, _ in g]
        ts = [s for _, s in g]
        print(f"  {name:<18} {len(g):>4} {np.mean(hits):>9.0%} {np.median(ts):>9.3f}")
    print(f"\n  chance peak-at-0 = {1 / (len(OFFSETS_S) + 1):.0%}")


_REF_AUDIO_HSR: dict[str, np.ndarray] = {}


def test_a_solo_warp(aligning, by_tid, mix_stems, gt_rows) -> None:
    """Decisive vocal test: truly-solo acappella spans (no overlapping vocal from any
    acappella OR host-regular row), with AUDIO-domain time-stretch (valid warp) before
    HuBERT. If solo jumps toward regular's 79%, warp was the confound; if it stays ~43%,
    vocal reconstruction is separation-limited and the host/vocal division is permanent."""
    from workspaces.section_hsmm.similarity_probe import SR as HSR, _hubert

    if "vocals" not in mix_stems:
        print("no mix_vocals stem", file=sys.stderr)
        return
    print("\n== SOLO acappella + AUDIO-domain warp (decisive) ==")
    print("loading mix_vocals HuBERT (once)...")
    mv = hubert_feat(mix_stems["vocals"])
    if mv is None:
        print("mix_vocals HuBERT failed", file=sys.stderr)
        return
    mix_f, mix_fps = mv

    def audible(r) -> bool:
        af = r.get("audible_frac")
        return af is None or float(af) > 0.1

    vocal_rows = [
        r
        for r in gt_rows
        if r.get("track_id")
        and audible(r)
        and (r.get("claimed_stem") or "") in ("acappella", "regular")
        and r.get("set_start_s") is not None
        and r.get("set_end_s") is not None
    ]

    def is_solo(row, ss, se) -> bool:
        for r in vocal_rows:
            if r is row:
                continue
            if float(r["set_start_s"]) < se - 1 and float(r["set_end_s"]) > ss + 1:
                return False
        return True

    def ref_hsr(path: str) -> np.ndarray | None:
        if path in _REF_AUDIO_HSR:
            return _REF_AUDIO_HSR[path]
        try:
            y, _ = librosa.load(path, sr=HSR, mono=True)
        except Exception:
            return None
        _REF_AUDIO_HSR[path] = y
        return y

    def stretched_hubert(y_hsr, r_start, r_dur, rate) -> np.ndarray | None:
        i0 = max(0, int(round(r_start * HSR)))
        i1 = min(len(y_hsr), i0 + int(round(r_dur * HSR)))
        seg = y_hsr[i0:i1]
        if len(seg) < HSR:
            return None
        try:
            seg = (
                librosa.effects.time_stretch(seg, rate=rate)
                if abs(rate - 1) > 1e-3
                else seg
            )
        except Exception:
            return None
        f = _hubert(seg, 9)  # (768, T) L2-normed on the SR/HOP grid
        return f if f.shape[1] >= 3 else None

    hits, trues, n_solo = [], [], 0
    for row in gt_rows:
        if (row.get("claimed_stem") or "") != "acappella" or not row.get("track_id"):
            continue
        try:
            ss, se = float(row["set_start_s"]), float(row["set_end_s"])
            rs, re_ = float(row["ref_start_s"]), float(row["ref_end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        dur = se - ss
        if dur < MIN_DUR_S or not is_solo(row, ss, se):
            continue
        t = by_tid.get(str(row["track_id"]))
        if t is None:
            continue
        sp = ref_stem_path(aligning, t.get("local_path") or "", "acappella")
        if sp is None:
            continue
        y = ref_hsr(str(sp))
        if y is None:
            continue
        n_solo += 1
        ref_dur = re_ - rs if (re_ - rs) >= MIN_DUR_S else dur
        rate = ref_dur / dur if dur > 0 else 1.0  # shrink ref_dur -> dur (mix timebase)
        mix_seg = _slice_feat(mix_f, mix_fps, ss, dur)
        rf0 = stretched_hubert(y, rs, ref_dur, rate)
        if rf0 is None:
            continue
        s0 = _feat_cos(mix_seg, rf0)
        if s0 is None:
            continue
        others = []
        for d in OFFSETS_S:
            rfd = stretched_hubert(y, rs + d, ref_dur, rate)
            if rfd is None:
                continue
            sd = _feat_cos(mix_seg, rfd)
            if sd is not None:
                others.append(sd)
        if not others:
            continue
        hits.append(int(s0 > max(others)))
        trues.append(s0)

    print(f"\n  truly-solo acappella spans found: {n_solo}; scorable: {len(hits)}")
    if hits:
        print(
            f"    solo + AUDIO warp : peak-at-0 = {np.mean(hits):.0%}  "
            f"median true = {np.median(trues):.3f}"
        )
    print(
        f"    reference points  : un-warped solo 43%/0.16 | regular 79%/0.74 | chance 8%"
    )
    if hits and np.mean(hits) >= 0.60:
        print("    => WARP was the confound; vocal reconstruction is salvageable.")
    elif hits:
        print(
            "    => stays weak; vocal reconstruction is separation-limited (division permanent)."
        )


_FEATS = [
    "recon_abs",
    "recon_margin",
    "recon_z",
    "recon_rank",
    "mert_conf",
    "path_conf",
]


def _topdecile_prec(scores, labels) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    k = max(5, len(order) // 10)
    top = order[:k]
    return sum(labels[i] for i in top) / k


def _tau_at_prec(scores, labels, target=0.90):
    """Return (recall, kept) at the highest-confidence prefix still >= target precision."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total = sum(labels)
    tp = best = 0
    best_rec = 0.0
    for rank, i in enumerate(order, 1):
        tp += labels[i]
        if tp / rank >= target and rank >= 5:
            best, best_rec = rank, tp / total
    return best_rec, best


def fusion_feasibility(feat_reg: list[tuple[dict, int]]) -> None:
    """Does a LEARNED fusion of label-free features beat the best single feature's
    ~78% top-decile precision? 5-fold CV logistic regression, out-of-fold scored."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    rows = [(f, l) for f, l in feat_reg if f is not None]
    if len(rows) < 30 or len({l for _, l in rows}) < 2:
        print("  insufficient host spans for CV")
        return
    # impute path_conf (missing on some spans) with the column median
    pvals = [f["path_conf"] for f, _ in rows if f.get("path_conf") is not None]
    pmed = float(np.median(pvals)) if pvals else 0.0
    X = np.array(
        [
            [float(f[k]) if f.get(k) is not None else pmed for k in _FEATS]
            for f, _ in rows
        ]
    )
    y = np.array([l for _, l in rows])
    print(f"  n={len(y)} correct={int(y.sum())}  features={_FEATS}")

    # single-feature baselines (top-decile precision + AUC)
    print(f"  {'feature':<14} {'AUC':>6} {'top-decile P':>13}")
    for j, name in enumerate(_FEATS):
        col = X[:, j].tolist()
        print(
            f"  {name:<14} {_f(auc(col, y.tolist())):>6} {_topdecile_prec(col, y):>12.0%}"
        )

    # 5-fold CV fused score (out-of-fold, no leakage)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    fa = auc(oof.tolist(), y.tolist())
    ftp = _topdecile_prec(oof.tolist(), y)
    rec, kept = _tau_at_prec(oof.tolist(), y.tolist(), 0.90)
    print(f"  {'FUSED (CV)':<14} {_f(fa):>6} {ftp:>12.0%}")
    if kept:
        print(
            f"  → fused τ@P≥90% keeps {kept}/{len(y)} ({rec:.0%} of correct) "
            f"→ ~{rec:.0%} of 20k host spans harvestable clean. FUSION CLEARS THE BAR."
        )
    else:
        print(
            f"  → fused still short of 90% precision (best top-decile {ftp:.0%}); "
            f"gate needs stronger features (fp-sharpness) or differentiable-render v2."
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-id", default="1fsnxchk")
    ap.add_argument("--gt", default="labeling/fixtures/bb12_ground_truth.yaml")
    ap.add_argument("--max-spans", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--hubert-acappella",
        action="store_true",
        help="run ONLY the HuBERT+fibers acappella Test A",
    )
    ap.add_argument(
        "--solo-warp",
        action="store_true",
        help="decisive: truly-solo acappella + audio-domain time-stretch",
    )
    args = ap.parse_args(argv)

    import yaml

    aligning = find_aligning_dir(args.set_id)
    if aligning is None:
        print(f"no aligning dir for {args.set_id}", file=sys.stderr)
        return 2
    manifest = json.load(open(aligning / "manifest.json"))
    mix_full_path = manifest.get("mix_local_path") or str(aligning / "mix.m4a")
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    mix_stems = find_mix_stems(aligning)

    gt = yaml.safe_load(open(_REPO / args.gt))
    gt_rows = gt["tracks"] if isinstance(gt, dict) else gt

    if args.hubert_acappella:
        test_a_hubert_acappella(aligning, manifest, by_tid, mix_stems, gt_rows)
        return 0

    if args.solo_warp:
        test_a_solo_warp(aligning, by_tid, mix_stems, gt_rows)
        return 0

    print(f"== recon probe {args.set_id} ==")
    print(
        f"mix: {Path(mix_full_path).name} | mix stems: {sorted(mix_stems)} | GT rows: {len(gt_rows)}"
    )
    print("loading mix (full)...")
    mix_full = load_audio(mix_full_path)
    if mix_full is None:
        print("mix load failed", file=sys.stderr)
        return 2
    mix_by_stem = {"full": mix_full}
    for stem, p in mix_stems.items():
        mix_by_stem[stem] = load_audio(p)

    def mix_for(claimed_stem: str) -> np.ndarray:
        key = _STEM_FILE.get(claimed_stem)  # vocals/instrumental
        if key and mix_by_stem.get(key) is not None:
            return mix_by_stem[key]
        return mix_full

    def ref_for(row) -> tuple[np.ndarray | None, bool]:
        """Return (ref audio, stem_routed?). Falls back to full ref if no stem."""
        tid = str(row.get("track_id") or "")
        t = by_tid.get(tid)
        if t is None:
            return None, False
        cs = row.get("claimed_stem") or "regular"
        sp = ref_stem_path(aligning, t.get("local_path") or "", cs)
        if sp is not None and mix_by_stem.get(_STEM_FILE.get(cs)) is not None:
            return load_audio(sp), True
        return load_audio(t.get("local_path")), False

    # ---------------- TEST A: validity (true vs perturbed) ------------------
    print("\n-- TEST A: does the match score peak at the TRUE placement? --")
    per_stem_curves: dict[str, list[list[float]]] = {}
    per_stem_hits: dict[str, list[int]] = {}
    per_stem_margin: dict[str, list[float]] = {}
    rows = gt_rows if not args.max_spans else gt_rows[: args.max_spans]
    for row in rows:
        if not row.get("track_id"):
            continue
        try:
            ss, se = float(row["set_start_s"]), float(row["set_end_s"])
            rs, re_ = float(row["ref_start_s"]), float(row["ref_end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        dur = se - ss
        if dur < MIN_DUR_S:
            continue
        cs = row.get("claimed_stem") or "regular"
        ref_audio, routed = ref_for(row)
        if ref_audio is None:
            continue
        mix_seg = slice_s(mix_for(cs), ss, dur)
        ref_dur = max(dur, re_ - rs)
        s0 = match_score(mix_seg, slice_s(ref_audio, rs, ref_dur))
        if s0 is None:
            continue
        curve = []
        for d in OFFSETS_S:
            sd = match_score(mix_seg, slice_s(ref_audio, rs + d, ref_dur))
            curve.append(sd if sd is not None else np.nan)
        others = [c for c in curve if not np.isnan(c)]
        if not others:
            continue
        tag = cs if not routed else f"{cs}*"  # * = stem-routed
        per_stem_curves.setdefault(tag, []).append([s0] + curve)
        per_stem_hits.setdefault(tag, []).append(int(s0 > max(others)))
        per_stem_margin.setdefault(tag, []).append(s0 - max(others))

    all_hits = [h for v in per_stem_hits.values() for h in v]
    print(f"\noffsets (s): 0(true) {OFFSETS_S}")
    for tag in sorted(per_stem_curves):
        arr = np.array(per_stem_curves[tag], dtype=float)
        mean_curve = np.nanmean(arr, axis=0)
        hit = np.mean(per_stem_hits[tag])
        marg = np.median(per_stem_margin[tag])
        print(
            f"\n  [{tag}] n={len(arr)}  peak-at-0 rate={hit:.0%}  median margin={marg:+.3f}"
        )
        print(
            "   true={:.3f} | curve@offset: {}".format(
                mean_curve[0],
                " ".join(f"{d:+d}:{v:.3f}" for d, v in zip(OFFSETS_S, mean_curve[1:])),
            )
        )
    if all_hits:
        print(
            f"\n  ALL spans peak-at-0 rate = {np.mean(all_hits):.0%} (n={len(all_hits)}); "
            f"chance = {1 / (len(OFFSETS_S) + 1):.0%}"
        )

    # ---------------- TEST B: usability (predicted correctness) -------------
    print("\n-- TEST B: does match score predict which predictions were CORRECT? --")
    pred_path = OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    aucs_line = ""
    if pred_path.exists():
        pred = json.load(open(pred_path))["spans"]
        # index GT by track_id for overlap matching
        scores: list[float] = []
        correct15: list[int] = []
        correct_id: list[int] = []
        by_stem: dict[str, tuple[list[float], list[int]]] = {}
        margin_reg: list[tuple[float, int]] = []
        feat_reg: list[tuple[dict, int]] = []
        for s in pred:
            try:
                ss, se = float(s["set_start_s"]), float(s["set_end_s"])
                rs = float(s["ref_start_s"])
            except (KeyError, TypeError, ValueError):
                continue
            dur = se - ss
            if dur < MIN_DUR_S:
                continue
            cs = s.get("claimed_stem") or "regular"
            tid = str(s.get("recording_id") or "")
            t = by_tid.get(tid)
            if t is None:
                continue
            sp = ref_stem_path(aligning, t.get("local_path") or "", cs)
            if sp is not None and mix_by_stem.get(_STEM_FILE.get(cs)) is not None:
                ref_audio = load_audio(sp)
            else:
                ref_audio = load_audio(t.get("local_path"))
            if ref_audio is None:
                continue
            mseg = slice_s(mix_for(cs), ss, dur)
            sc = match_score(mseg, slice_s(ref_audio, rs, dur))
            if sc is None:
                continue
            # reconstruction confidence features vs ALTERNATIVE placements
            # (perturb ref offset around predicted rs; all label-free, self-supervised)
            margin = None
            feat = None
            if cs == "regular":
                alts = []
                for d in OFFSETS_S:
                    md = match_score(mseg, slice_s(ref_audio, rs + d, dur))
                    if md is not None:
                        alts.append(md)
                if alts:
                    a = np.array(alts)
                    margin = sc - float(a.max())
                    z = (sc - float(a.mean())) / (float(a.std()) + 1e-6)
                    rank = float(np.mean(sc > a))
                    feat = {
                        "recon_abs": sc,
                        "recon_margin": margin,
                        "recon_z": z,
                        "recon_rank": rank,
                        "mert_conf": float(s.get("confidence") or 0.0),
                        "path_conf": s.get("ref_path_conf"),
                    }
            # GT rows overlapping this predicted span in time (+-5s)
            ov = [
                r
                for r in gt_rows
                if r.get("track_id")
                and float(r["set_start_s"]) < se + 5
                and float(r["set_end_s"]) > ss - 5
            ]
            id_ok = any(str(r["track_id"]) == tid for r in ov)
            same = [r for r in ov if str(r["track_id"]) == tid]
            place_ok = (
                bool(same)
                and min(abs(float(r["set_start_s"]) - ss) for r in same) <= 15
            )
            scores.append(sc)
            correct_id.append(int(id_ok))
            lab = int(id_ok and place_ok)
            correct15.append(lab)
            bs = by_stem.setdefault(cs, ([], []))
            bs[0].append(sc)
            bs[1].append(lab)
            if margin is not None:
                margin_reg.append((margin, lab))
            if feat is not None:
                feat_reg.append((feat, lab))
        a_id = auc(scores, correct_id)
        a_15 = auc(scores, correct15)

        def med(mask):
            v = [s for s, m in zip(scores, mask) if m]
            return np.median(v) if v else float("nan")

        print(
            f"  n scored predictions = {len(scores)} | correct(id&<15s) = {sum(correct15)}"
        )
        print(f"  AUC(match -> identity-correct)  ALL   = {_f(a_id)}")
        print(f"  AUC(match -> id & set_start<15s) ALL  = {_f(a_15)}")
        print(
            f"  median match ALL: correct={med(correct15):.3f}  wrong={med([1 - c for c in correct15]):.3f}"
        )
        for cs in sorted(by_stem):
            sc_, lb_ = by_stem[cs]
            a = auc(sc_, lb_)
            cp = [s for s, l in zip(sc_, lb_) if l]
            wp = [s for s, l in zip(sc_, lb_) if not l]
            print(
                f"    [{cs}] n={len(sc_)} correct={sum(lb_)}  AUC={_f(a)}  "
                f"median correct={np.median(cp) if cp else float('nan'):.3f} "
                f"wrong={np.median(wp) if wp else float('nan'):.3f}"
            )
        aucs_line = (
            f"Test B AUC(id&<15s) regular="
            f"{_f(auc(*by_stem.get('regular', ([], []))))} all={_f(a_15)}"
        )

        # ---- STEP 2a: pseudo-label harvest — precision/recall of a confidence gate ----
        print("\n-- STEP 2a: reconstruction as a pseudo-label GATE (regular/host) --")

        def harvest(pairs: list[tuple[float, int]], name: str) -> None:
            pairs = [p for p in pairs if p[0] is not None]
            if not pairs or not any(l for _, l in pairs) or all(l for _, l in pairs):
                print(f"  [{name}] insufficient spans / no class variety")
                return
            order = sorted(pairs, key=lambda p: -p[0])  # highest confidence first
            total_pos = sum(l for _, l in pairs)
            print(f"  [{name}] n={len(pairs)} correct={total_pos}")
            print(f"    {'gate τ':>7} {'kept':>5} {'prec':>6} {'recall':>7}")
            marks = {int(len(order) * f) for f in (0.1, 0.25, 0.5, 1.0)}
            best = None
            tp = 0
            for rank, (sc_, lb_) in enumerate(order, 1):
                tp += lb_
                prec, rec = tp / rank, tp / total_pos
                if prec >= 0.90 and best is None and rank >= 5:
                    best = (sc_, rank, prec, rec)
                if rank in marks:
                    print(f"    {sc_:>7.3f} {rank:>5} {prec:>5.0%} {rec:>7.0%}")
            if best:
                tau, k, p, r = best
                print(
                    f"    → τ@P≥90%: conf≥{tau:.3f} keeps {k}/{len(pairs)} ({r:.0%} of correct) "
                    f"→ ~{r:.0%} of 20k host spans harvestable clean"
                )
            else:
                top = order[: max(5, len(order) // 10)]
                tp10 = sum(l for _, l in top) / len(top)
                print(
                    f"    → no τ reaches 90% precision (top-decile precision {tp10:.0%})"
                )

        harvest(
            list(zip(*by_stem.get("regular", ([], []))))
            if by_stem.get("regular", ([], []))[0]
            else [],
            "absolute match",
        )
        harvest(margin_reg, "recon MARGIN (chosen vs best-alternative)")

        # ---- STEP 2b feasibility: LEARNED fusion of label-free features (k-fold CV) ----
        print(
            "\n-- STEP 2b feasibility: learned confidence FUSION (5-fold CV, host) --"
        )
        fusion_feasibility(feat_reg)
    else:
        print(f"  (no predicted timeline at {pred_path.name}; skipping Test B)")

    # ---------------- VERDICT ----------------------------------------------
    reg_hit = (
        np.mean(per_stem_hits.get("regular", [0]))
        if "regular" in per_stem_hits
        else 0.0
    )
    print("\n== VERDICT INPUTS ==")
    print(f"  regular peak-at-0 rate = {reg_hit:.0%} (headline validity)")
    print(f"  {aucs_line}")
    OUT_DIR.mkdir(exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

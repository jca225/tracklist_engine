#!/usr/bin/env python3
"""Phase 3 post-hoc analysis for the harmony_fibers SALAMI arm.

The main harness (``fiber_validation.py --harmony``) already scored the
harmony arm per-track (out/fiber_validation.json). Two things it does NOT
report, which the phase-3 verdict needs:

1. **Degenerate-collapse check.** Chroma's failure mode was ONE fiber covering
   (nearly) the whole track on 30/30. Per-track #fibers alone doesn't rule that
   out for harmony_fibers — a single giant fiber and a single small fiber both
   read "n=1". So: per-track **max-fiber-share** (largest fiber's frames as a
   share of the track, and as a share of the fibered frames), for all three
   detectors side by side.
2. **Pooled (micro) pair P/R/F.** The per-track MEANS in the JSON are distorted
   by the 22 tracks where harmony returned zero fibers (precision contributes
   0.0 there under ``_pr_f``, mir_eval contributes 1.0 — neither is "the
   precision of the merges it makes"). Micro-averaging the raw frame-pair
   counts over all tracks answers the gate question directly: *of the pairs the
   arm merges, how many are right* — and the same for HuBERT ∪ harmony.

Also supports a small, PRE-REGISTERED sensitivity sweep (``--sweep``): one knob
at a time around the shipped defaults (local_prom 0.04/0.10, glob_pct 85/97),
pooled-micro scored. This is reported as a robustness check only — SALAMI is
validation, not a tuning set; the shipped-defaults run is the result.

Reuses fiber_validation's loaders/caches read-only; recomputes labels only for
sweep configs (config-tagged cache keys, so the defaults cache is untouched).

Usage:
  venvs/audio/bin/python -m alignment.external.fiber_validation_phase3 [--sweep]

Writes out/fiber_validation_phase3.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alignment.continuity_refine import _compute_hubert  # noqa: E402
from alignment.external.fiber_validation import (  # noqa: E402
    _AUDIO_CACHE,
    _FEAT_CACHE,
    _OUT,
    FPS,
    _compute_chroma,
    _compute_harmony_labels,
    _fiber_int_on_grid,
    _gt_for,
    _gt_int_on_grid,
    _load_salami,
    _same_matrix,
)
from alignment.ref_fibers import compute_fibers  # noqa: E402

# pre-registered sweep: one knob at a time around the shipped defaults.
_SWEEP: dict[str, dict] = {
    "defaults": {},
    "local_prom=0.04": {"local_prom": 0.04},
    "local_prom=0.10": {"local_prom": 0.10},
    "glob_pct=85": {"glob_pct": 85.0},
    "glob_pct=97": {"glob_pct": 97.0},
}


def _harm_labels_cfg(path: str, cfg_name: str, kw: dict) -> tuple[np.ndarray, float]:
    """harmony_fibers labels for a sweep config, cached under a config tag so
    the shipped-defaults cache entries are never overwritten."""
    if not kw:  # defaults — reuse the main harness cache
        return _compute_harmony_labels(path)
    from alignment.harmony_fibers import harmony_fibers

    key = hashlib.md5(str(path).encode()).hexdigest()[:16]
    tag = cfg_name.replace("=", "").replace(".", "")
    cf = _FEAT_CACHE / f"{key}_harm_{tag}.npz"
    if cf.is_file():
        z = np.load(cf)
        return z["labels"], float(z["hz"])
    labels, hz = harmony_fibers(str(path), **kw)
    _FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cf, labels=labels, hz=np.float32(hz))
    return labels, float(hz)


def _fiber_stats(labels: np.ndarray) -> dict:
    """#fibers, coverage, max-fiber-share (of track / of fibered frames)."""
    pos = labels[labels >= 0]
    n_ids = int(np.unique(pos).size) if pos.size else 0
    cover = float(pos.size / labels.size) if labels.size else 0.0
    if pos.size == 0:
        return {"n": 0, "cover": 0.0, "max_share_track": 0.0, "max_share_fibered": 0.0}
    counts = np.bincount(pos)
    mx = int(counts.max())
    return {
        "n": n_ids,
        "cover": cover,
        "max_share_track": float(mx / labels.size),
        "max_share_fibered": float(mx / pos.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-hz", type=float, default=10.0)
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--sweep", action="store_true", help="run the pre-registered sweep")
    args = ap.parse_args()

    base = json.loads((_OUT / "fiber_validation.json").read_text())
    sids = [t["salami_id"] for t in base["tracks"]]
    d = _load_salami()

    cfgs = _SWEEP if args.sweep else {"defaults": {}}
    # pooled counters per config: [tp, pred_pos] for harm + union; hub shared.
    pool = {
        c: {"harm_tp": 0.0, "harm_pp": 0.0, "uni_tp": 0.0, "uni_pp": 0.0} for c in cfgs
    }
    fired = {
        c: {"harm_tp": 0.0, "harm_pp": 0.0, "gt": 0.0, "n_fired": 0, "cov": []}
        for c in cfgs
    }
    cfg_cov = {c: [] for c in cfgs}
    hub_tp = hub_pp = gt_tot = 0.0
    per_track: list[dict] = []

    for sid in sids:
        dest = _AUDIO_CACHE / f"{sid}.mp3"
        gt = _gt_for(d, sid)
        assert gt is not None and dest.is_file(), sid
        gt_intervals, gt_labels, track = gt
        gt_iv = np.asarray(gt_intervals, dtype=float)
        dur = float(track.duration or gt_iv[-1, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat = _compute_hubert(str(dest), args.layer)
            hub_lab, hub_hz = compute_fibers(feat, FPS, audio_path=str(dest))
            cfeat = _compute_chroma(str(dest))
            chr_lab, chr_hz = compute_fibers(cfeat, FPS, audio_path=str(dest))
        harm_by_cfg = {c: _harm_labels_cfg(str(dest), c, kw) for c, kw in cfgs.items()}

        # common grid (identical construction to _score_multimodal)
        ends = [dur, float(gt_iv[-1, 1]), hub_lab.size / hub_hz] + [
            lab.size / hz for lab, hz in harm_by_cfg.values()
        ]
        span = min(ends)
        n = int(span * args.grid_hz)
        centers = (np.arange(n) + 0.5) / args.grid_hz
        gt_i = _gt_int_on_grid(gt_iv, gt_labels, centers)
        gt_m = _same_matrix(gt_i)
        hub_i = _fiber_int_on_grid(hub_lab, hub_hz, centers)
        hub_m = _same_matrix(hub_i)
        g = float(gt_m.sum())
        gt_tot += g
        hub_tp += float((hub_m & gt_m).sum())
        hub_pp += float(hub_m.sum())

        row = {
            "sid": sid,
            "hub": _fiber_stats(hub_lab),
            "chroma": _fiber_stats(chr_lab),
        }
        for c, (hlab, hhz) in harm_by_cfg.items():
            hm = _same_matrix(_fiber_int_on_grid(hlab, hhz, centers))
            um = hub_m | hm
            st = _fiber_stats(hlab)
            pool[c]["harm_tp"] += float((hm & gt_m).sum())
            pool[c]["harm_pp"] += float(hm.sum())
            pool[c]["uni_tp"] += float((um & gt_m).sum())
            pool[c]["uni_pp"] += float(um.sum())
            cfg_cov[c].append(st["cover"])
            if st["n"] > 0:
                fired[c]["harm_tp"] += float((hm & gt_m).sum())
                fired[c]["harm_pp"] += float(hm.sum())
                fired[c]["gt"] += g
                fired[c]["n_fired"] += 1
                fired[c]["cov"].append(st["cover"])
            if c == "defaults":
                row["harm"] = st
        per_track.append(row)
        print(
            f"[{sid}] hub n={row['hub']['n']} maxshare={row['hub']['max_share_track']:.2f} | "
            f"chr n={row['chroma']['n']} maxshare={row['chroma']['max_share_track']:.2f} | "
            f"harm n={row['harm']['n']} cov={row['harm']['cover']:.2f} "
            f"maxshare={row['harm']['max_share_track']:.2f}",
            flush=True,
        )

    def _pr(tp: float, pp: float, g: float) -> tuple[float, float, float]:
        p = tp / pp if pp > 0 else 0.0
        r = tp / g if g > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    out: dict = {"n_tracks": len(sids), "grid_hz": args.grid_hz}
    hp, hr, hf = _pr(hub_tp, hub_pp, gt_tot)
    out["micro_hub"] = {"p": hp, "r": hr, "f": hf}
    print(f"\nPOOLED (micro) over {gt_tot:.0f} GT-true pairs, {len(sids)} tracks")
    print(f"  HuBERT           P={hp:.3f} R={hr:.3f} F={hf:.3f}")
    for c in cfgs:
        p, r, f = _pr(pool[c]["harm_tp"], pool[c]["harm_pp"], gt_tot)
        up, ur, uf = _pr(pool[c]["uni_tp"], pool[c]["uni_pp"], gt_tot)
        fp_, fr_, _ = _pr(fired[c]["harm_tp"], fired[c]["harm_pp"], fired[c]["gt"])
        cov = float(np.mean(cfg_cov[c]))
        out[f"cfg::{c}"] = {
            "micro_harm": {"p": p, "r": r, "f": f},
            "micro_union": {"p": up, "r": ur, "f": uf},
            "fired_tracks": fired[c]["n_fired"],
            "fired_micro_harm_p": fp_,
            "fired_micro_harm_r": fr_,
            "mean_cov": cov,
            "fired_mean_cov": float(np.mean(fired[c]["cov"]))
            if fired[c]["cov"]
            else 0.0,
        }
        print(
            f"  [{c}] harmony    P={p:.3f} R={r:.3f} F={f:.3f}  cov={cov:.2f}  "
            f"fired={fired[c]['n_fired']}/{len(sids)} (fired-only P={fp_:.3f} R={fr_:.3f})"
        )
        print(f"  [{c}] HUB∪HARM   P={up:.3f} R={ur:.3f} F={uf:.3f}")

    # collapse-check distributions (defaults config)
    n_harm = [t["harm"]["n"] for t in per_track]
    ms_harm = [t["harm"]["max_share_track"] for t in per_track]
    msf_harm = [t["harm"]["max_share_fibered"] for t in per_track if t["harm"]["n"] > 0]
    ms_chr = [t["chroma"]["max_share_track"] for t in per_track]
    out["collapse_check"] = {
        "harm_n_fibers_hist": {str(k): n_harm.count(k) for k in sorted(set(n_harm))},
        "harm_max_share_track": {
            "mean": float(np.mean(ms_harm)),
            "max": float(np.max(ms_harm)),
        },
        "harm_max_share_fibered_fired": {
            "mean": float(np.mean(msf_harm)) if msf_harm else 0.0,
            "values": [round(v, 2) for v in sorted(msf_harm)],
        },
        "chroma_max_share_track": {
            "mean": float(np.mean(ms_chr)),
            "min": float(np.min(ms_chr)),
        },
        "chroma_n_eq_1": sum(1 for t in per_track if t["chroma"]["n"] == 1),
    }
    out["per_track"] = per_track
    print("\nCOLLAPSE CHECK (defaults):")
    print(f"  harmony #fibers hist: {out['collapse_check']['harm_n_fibers_hist']}")
    print(
        f"  harmony max-fiber-share of track: mean {np.mean(ms_harm):.2f}, max {np.max(ms_harm):.2f}"
    )
    if msf_harm:
        print(
            f"  harmony max-fiber-share of FIBERED frames (fired tracks): {[round(v, 2) for v in sorted(msf_harm)]}"
        )
    print(
        f"  chroma (reference failure): n==1 on {out['collapse_check']['chroma_n_eq_1']}/{len(sids)}, "
        f"max-share-of-track mean {np.mean(ms_chr):.2f}"
    )

    (_OUT / "fiber_validation_phase3.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {_OUT / 'fiber_validation_phase3.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

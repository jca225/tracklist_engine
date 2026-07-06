"""Foote boundary novelty → the ``surprise`` action (order-free placement prior).

The info-dynamics study concluded that cheap unsupervised surprise localizes DJ
transitions above chance, and the D2 probe measured Foote checkerboard novelty
on the centered mix-MERT self-similarity at 64% GT-recall@4s on BB12. This
module is that signal as a harness action: the novelty curve of the whole mix
becomes per-span ``Observation``s via :func:`belief.surprise_prior`, banded
around the span's scraped cue anchor (the one prior that is independent of
every other probe). w-rows abstain — their scraped cue is 0.0 and layered
overlays don't show in full-mix novelty (the bed dominates).

Eval (measures P(correct | fired) vs GT, the validation-gate convention):

    venvs/audio/bin/python -m workspaces.alignment_prototype.agentic.novelty \
        --set-id 2nvzlh2k --gt labeling/fixtures/bb11_ground_truth.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.agentic.actions import REGISTRY
from workspaces.alignment_prototype.agentic.belief import Observation, surprise_prior

ANALYSIS_DIR = _REPO / "data" / "analysis"
KERNEL_BARS = 6  # D2's best setting (L=6: recall@4s 64%, L=9: 53%)
BAND_S = 45.0  # matches the infer coarse band


def foote_novelty(mert: np.ndarray, *, kernel_bars: int = KERNEL_BARS) -> np.ndarray:
    """Checkerboard novelty on the centered self-similarity of bar embeddings.

    Centering the SSM matters: MERT self-similarity has a ~0.9 floor, and the
    checkerboard contrast lives in the residual dynamic range.
    """
    x = mert.astype(np.float64)
    x[~np.isfinite(x).all(axis=1)] = 0.0  # a single NaN bar would poison the SSM
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    ssm = x @ x.T
    ssm -= ssm.mean()
    L = kernel_bars
    u = np.arange(-L, L) + 0.5
    kernel = np.outer(np.sign(u), np.sign(u)) * np.outer(
        np.exp(-(u**2) / (2 * (L / 2.0) ** 2)), np.exp(-(u**2) / (2 * (L / 2.0) ** 2))
    )
    n = x.shape[0]
    nov = np.zeros(n, dtype=np.float64)
    for t in range(L, n - L):
        nov[t] = float(np.sum(kernel * ssm[t - L : t + L, t - L : t + L]))
    return np.maximum(nov, 0.0)


def novelty_curve(artifact_path: Path) -> tuple[tuple[float, float], ...]:
    """(bar_time_s, novelty) curve from a cached mix-MERT artifact."""
    from eda.alignment.artifacts import load_mix_mert_artifact

    art = load_mix_mert_artifact(artifact_path)
    nov = foote_novelty(art.mert)
    return tuple(zip((float(t) for t in art.bar_start_s), (float(v) for v in nov)))


def _abstain(detail: str) -> Observation:
    spec = REGISTRY["surprise"]
    return Observation(
        probe="surprise",
        set_start_s=None,
        confidence=0.0,
        precision=spec.precision,
        detail=detail,
    )


def surprise_runner(curve: tuple[tuple[float, float], ...], *, band_s: float = BAND_S):
    """Runner: snap the span's cue anchor to the nearest novelty peak in band.

    The cue anchor is the band center because it is the only placement prior
    independent of the other probes — using the timeline's own set_start would
    let the winner vote for itself and inflate the measured precision.
    """
    spec = REGISTRY["surprise"]
    candidates = surprise_prior(curve, precision=spec.precision)

    def _run(data: dict) -> Observation:
        cue = data.get("cue_anchor_s")
        slot = str(data.get("slot_label") or "")
        if cue is None or (float(cue) == 0.0 and "w" in slot):
            return _abstain("no independent band center (w-row / missing cue)")
        cue = float(cue)
        in_band = [o for o in candidates if abs(o.set_start_s - cue) <= band_s]
        if not in_band:
            return _abstain(f"no novelty peak within ±{band_s:.0f}s of cue")
        nearest = min(in_band, key=lambda o: abs(o.set_start_s - cue))
        return replace(
            nearest,
            cost=spec.cost,
            detail=f"{nearest.detail} snap Δcue={nearest.set_start_s - cue:+.1f}s",
        )

    return _run


def surprise_runner_for_set(set_id: str, *, band_s: float = BAND_S):
    """Build the runner from the set's cached MERT artifact; None if absent."""
    path = ANALYSIS_DIR / f"{set_id}_mix_mert.npz"
    if not path.is_file():
        return None
    return surprise_runner(novelty_curve(path), band_s=band_s)


# ---------------------------------------------------------------------------
# Validation gate: measured P(correct | fired) vs GT, per stem
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import json

    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--band-s", type=float, default=BAND_S)
    args = p.parse_args(argv)

    run = surprise_runner_for_set(args.set_id, band_s=args.band_s)
    if run is None:
        sys.exit(f"no MERT artifact for {args.set_id} under {ANALYSIS_DIR}")

    tl_path = (
        Path(__file__).resolve().parents[1]
        / "out"
        / f"{args.set_id}_predicted_timeline.json"
    )
    spans = json.loads(tl_path.read_text())["spans"]

    # GT joined by normalized recording id (slot-label schemes differ per set)
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{args.set_id}.json"
    id_map = json.loads(id_map_path.read_text()) if id_map_path.exists() else {}
    gt_by_rec: dict[str, list[dict]] = {}
    for r in yaml.safe_load(args.gt.read_text())["tracks"]:
        tid = str(r.get("track_id") or "")
        if (
            tid
            and str(r.get("slot_label")) != "mix"
            and r.get("set_start_s") is not None
        ):
            gt_by_rec.setdefault(id_map.get(tid, tid), []).append(r)

    stats: dict[str, dict] = {}
    for s in spans:
        rows = gt_by_rec.get(str(s["recording_id"]))
        if not rows:
            continue
        stem = s.get("claimed_stem") or "regular"
        st = stats.setdefault(stem, {"n": 0, "fired": 0, "c15": 0, "c5": 0, "errs": []})
        st["n"] += 1
        obs = run(s)
        if obs.abstained:
            continue
        st["fired"] += 1
        err = min(abs(obs.set_start_s - r["set_start_s"]) for r in rows)
        st["errs"].append(err)
        st["c15"] += err <= 15
        st["c5"] += err <= 5

    print(f"surprise (Foote novelty, band ±{args.band_s:.0f}s) vs {args.gt.name}:")
    tot_fired = tot_c15 = 0
    for stem, st in sorted(stats.items()):
        e = sorted(st["errs"])
        med = f"{e[len(e) // 2]:.1f}s" if e else "—"
        p15 = f"{st['c15'] / st['fired']:.0%}" if st["fired"] else "—"
        p5 = f"{st['c5'] / st['fired']:.0%}" if st["fired"] else "—"
        print(
            f"  {stem:12} slots={st['n']:3}  fired={st['fired']:3}  "
            f"P(<=15s|fired)={p15:4}  <=5s={p5:4}  median={med}"
        )
        tot_fired += st["fired"]
        tot_c15 += st["c15"]
    if tot_fired:
        print(
            f"  → measured precision (all stems, <=15s): {tot_c15 / tot_fired:.2f} "
            f"on n={tot_fired} fires"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

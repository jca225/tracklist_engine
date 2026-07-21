"""P1: does RICH-EMBEDDING information dynamics predict mashup compatibility?

Follow-up to the chroma-surprise P0 experiment (``bb_mashup_surprise_p0*.py``),
which came back NULL and named its own bottleneck: *chroma is too coarse*. This
script swaps chroma for HuBERT tokens and computes the Abdallah & Plumbley
information-dynamic measures (``markov_infodyn.py``) — entropy rate, predictive
information rate (PIR), and surprise under a *subjective* BB-native prior.

Design
------
1. **BB-native observer** (the paper's subjectivity axis, §1.1 / §2.5):
   HuBERT-L9 frames of a HELD-OUT BB mix (BB10 = ``w1mgcjt``) → k-means codebook
   (a BB listener's "vocabulary") + a first-order transition matrix (its
   expectations). The observer is BB-fluent but has NOT heard BB11/BB12, so
   scoring those sets' pairings against it is leakage-free.

2. **Per candidate pairing** (real vocal+bed = positive; same vocal over a
   key/BPM-compatible but NON-co-occurring bed = negative — reused verbatim from
   ``bb_mashup_surprise_p0_v2``): render the combined audio on the set tempo
   grid, sum the stems, tokenize with the BB codebook, and compute:
     - naive per-stimulus measures:  entropy_rate, PIR, redundancy
     - subjective-under-BB-prior:    mean / max / var surprise

3. **Two readouts**
     - separation:  LOSO AUC (BB11 ↔ BB12) of each measure vs the key baseline.
     - inverted-U:  the paper's Fig.3 arch — do real mashups sit at HIGH PIR
       (structured *and* surprising) while negatives fall to the boring/random
       extremes? Reported as PIR-AUC + mean-PIR(pos) vs mean-PIR(neg).

4. **Interaction control**: the same measures for the BED ALONE and VOCAL ALONE.
   If combined-PIR only works because it tracks "is this a good bed", bed-alone
   AUC matches it. The *pairing* hypothesis survives only if COMBINED beats the
   marginals — that is the test that interaction carries signal.

Verdict (pre-registered)
------------------------
  GO   : PIR-AUC (or best combined measure) > 0.60, clears bed-alone by ≥0.05,
         and real mashups cluster at higher PIR than negatives.
  WEAK : 0.55–0.60 or interaction margin < 0.05.
  NULL : ≤ 0.55 and no arch clustering → rich embeddings null too; the lever is
         placement / a learned space / usage data, not intrinsic content IDyn.

Usage
-----
    venvs/audio/bin/python -m lab.information_dynamics.bb_mashup_pir_v1

Writes to data/analysis/aux.db :: analysis_results / bb_mashup_pir_v1 and prints
inverted-U scatter data. Small n — treat AUC diffs < ~0.1 as noise.
"""

from __future__ import annotations

import pathlib
import sqlite3
import sys
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from lab.information_dynamics import markov_infodyn as idyn
from lab.information_dynamics.bb_mashup_surprise_p0 import (
    AUX_DB,
    MAX_PAIR_S,
    MIN_PAIR_S,
    _import_sklearn,
    _key_baseline_auc,
    _loso_auc,
)
from lab.information_dynamics.bb_mashup_surprise_p0_v2 import (
    BB11_DIR,
    BB11_SET_ID,
    BB11_YAML,
    BB12_DIR,
    BB12_SET_ID,
    BB12_YAML,
    PairV2,
    _build_negative_pairs_clean,
    _build_positive_pairs_clean,
    _collect_bed_tracks,
    _load_set_data,
)

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# HuBERT loader (drop-in): _hubert(y@22050, layer) -> (768, T) L2-normed
from workspaces.section_hsmm.similarity_probe import _hubert  # noqa: E402

ALIGNING = pathlib.Path.home() / "aligning"
BB10_MIX_INSTR = (
    ALIGNING
    / "w1mgcjt__Two Friends - Big Bootie Mix Volume 10"
    / "mix_instrumental.flac"
)
BB10_MIX_VOCALS = (
    ALIGNING / "w1mgcjt__Two Friends - Big Bootie Mix Volume 10" / "mix_vocals.flac"
)

SR = 22050
HOP = 512  # librosa frame hop the HuBERT grid is resampled onto (43.07 fps)
HUBERT_LAYER = 9
K_STATES = 24  # codebook size (BB listener vocabulary), shared prior+pairs
PRIOR_ALPHA = 1.0  # Laplace smoothing on the BB prior transition matrix
PAIR_ALPHA = 0.2  # lighter smoothing so per-stimulus transitions aren't washed
PRIOR_SAMPLE_S = 1500.0  # cap BB10 audio used to fit the prior (seconds)
FRAME_SUBSAMPLE = 2  # keep every Nth HuBERT frame (43fps -> ~21.5fps token stream)
# Fixed analysis window: EVERY pair is truncated to exactly this many token
# frames so sequence length can't leak between classes (v2 makes negatives ≤
# positive duration). Pairs shorter than this are skipped.
ANALYSIS_S = 18.0
ANALYSIS_FRAMES = int(ANALYSIS_S * SR / HOP / FRAME_SUBSAMPLE)  # ~387 frames
ANALYSIS_NAME = "bb_mashup_pir_v1"


def _import_librosa():
    import librosa

    return librosa


# ---------------------------------------------------------------------------
# Audio rendering: stems on the SET tempo grid, summed into a mashup waveform
# ---------------------------------------------------------------------------


def _load_wave_on_set_grid(
    path: pathlib.Path,
    ref_start_s: float,
    dur_set_s: float,
    tempo_ratio: float,
    librosa,
) -> np.ndarray | None:
    """Return a mono 22.05k waveform of ``dur_set_s`` SET-seconds of ``path``.

    Reads ``dur_set_s * tempo_ratio`` ref-seconds then time-stretches by
    ``rate=tempo_ratio`` so the output plays at the set tempo (same convention
    as v2's ``_load_chroma_tempo``). Pitch preserved.
    """
    dur_ref_s = dur_set_s * tempo_ratio
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(
                str(path), sr=SR, mono=True, offset=ref_start_s, duration=dur_ref_s
            )
        if len(y) < SR * MIN_PAIR_S * 0.5:
            return None
        if abs(tempo_ratio - 1.0) > 1e-3:
            y = librosa.effects.time_stretch(y, rate=tempo_ratio)
        if len(y) < SR * MIN_PAIR_S:
            return None
        return y.astype(np.float32)
    except Exception as e:
        print(f"  [wave err] {path.name}: {e}", file=sys.stderr)
        return None


def _rms_normalise(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y**2))) + 1e-9
    return (y * (target_rms / rms)).astype(np.float32)


def _hubert_frames(y: np.ndarray) -> np.ndarray | None:
    """(T, 768) HuBERT-L9 frames for a waveform, subsampled in time, or None."""
    if y is None or len(y) < SR * MIN_PAIR_S:
        return None
    h = _hubert(y, HUBERT_LAYER)  # (768, T)
    if h.shape[1] < 4:
        return None
    frames = h.T[::FRAME_SUBSAMPLE]  # (T', 768)
    return frames.astype(np.float32)


# ---------------------------------------------------------------------------
# BB-native prior: codebook + transition model on a held-out BB mix
# ---------------------------------------------------------------------------


@dataclass
class BBPrior:
    kmeans: object  # fitted KMeans
    a_prior: np.ndarray  # (K, K) column-stochastic BB transition matrix
    n_frames: int


def _build_bb_prior(librosa) -> BBPrior:
    """Fit the BB-native observer on the held-out BB10 full mix (vocals+instr)."""
    from sklearn.cluster import KMeans

    print(
        f"Building BB-native prior from held-out BB10 mix (≤{PRIOR_SAMPLE_S:.0f}s)..."
    )
    if not BB10_MIX_INSTR.exists() or not BB10_MIX_VOCALS.exists():
        sys.exit(f"ERROR: BB10 mix stems missing at {BB10_MIX_INSTR.parent}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yi, _ = librosa.load(
            str(BB10_MIX_INSTR), sr=SR, mono=True, duration=PRIOR_SAMPLE_S
        )
        yv, _ = librosa.load(
            str(BB10_MIX_VOCALS), sr=SR, mono=True, duration=PRIOR_SAMPLE_S
        )
    n = min(len(yi), len(yv))
    full_mix = _rms_normalise(yi[:n]) + _rms_normalise(yv[:n])
    print(f"  BB10 sampled: {n / SR:.0f}s of full mix; extracting HuBERT...")

    frames = _hubert_frames(full_mix)
    if frames is None or len(frames) < K_STATES * 4:
        sys.exit("ERROR: too few BB10 frames to fit the prior")
    print(f"  {len(frames)} token frames; fitting K={K_STATES} codebook...")

    km = KMeans(n_clusters=K_STATES, n_init=4, random_state=0).fit(frames)
    seq = km.labels_.astype(np.int64)
    a_prior = idyn.fit_transition(seq, K_STATES, alpha=PRIOR_ALPHA)
    er = idyn.entropy_rate(a_prior)
    pr = idyn.pir(a_prior)
    print(f"  BB prior fitted: entropy_rate={er:.3f} nats  PIR={pr:.4f} nats")
    return BBPrior(kmeans=km, a_prior=a_prior, n_frames=len(frames))


# ---------------------------------------------------------------------------
# Per-pair information dynamics
# ---------------------------------------------------------------------------


class InfoDynRow(NamedTuple):
    label: str
    set_id: str
    is_positive: bool
    # combined-audio measures (naive per-stimulus)
    er_comb: float
    pir_comb: float
    redun_comb: float
    # combined-audio measures (subjective, under BB prior)
    surp_mean_comb: float
    surp_max_comb: float
    surp_var_comb: float
    # controls: bed alone / vocal alone
    pir_bed: float
    pir_vocal: float
    surp_mean_bed: float
    surp_mean_vocal: float
    # interaction deltas (combined minus marginal)
    pir_lift_vs_bed: float
    surp_lift_vs_bed: float
    duration_s: float


def _measures(frames: np.ndarray | None, prior: BBPrior) -> dict | None:
    """Naive + subjective info-dynamics of a HuBERT frame block."""
    if frames is None or len(frames) < K_STATES:
        return None
    seq = prior.kmeans.predict(frames).astype(np.int64)
    a = idyn.fit_transition(seq, K_STATES, alpha=PAIR_ALPHA)
    surp = idyn.surprise_under(seq, prior.a_prior)
    return {
        "seq": seq,
        "er": idyn.entropy_rate(a),
        "pir": idyn.pir(a),
        "redun": idyn.redundancy(a),
        **surp,
    }


def _pair_infodyn(pair: PairV2, prior: BBPrior, librosa) -> InfoDynRow | None:
    yv = _load_wave_on_set_grid(
        pair.vocal_path,
        pair.vocal_ref_start_s,
        pair.duration_set_s,
        pair.vocal_tempo_ratio,
        librosa,
    )
    yb = _load_wave_on_set_grid(
        pair.bed_path,
        pair.bed_ref_start_s,
        pair.duration_set_s,
        pair.bed_tempo_ratio,
        librosa,
    )
    if yv is None or yb is None:
        return None
    n = min(len(yv), len(yb))
    yv, yb = yv[:n], yb[:n]
    combined = _rms_normalise(yv) + _rms_normalise(yb)

    fc = _hubert_frames(combined)
    fb = _hubert_frames(yb)
    fv = _hubert_frames(yv)
    # fixed-length window so sequence length can't leak between classes
    if any(f is None or len(f) < ANALYSIS_FRAMES for f in (fc, fb, fv)):
        return None
    fc, fb, fv = fc[:ANALYSIS_FRAMES], fb[:ANALYSIS_FRAMES], fv[:ANALYSIS_FRAMES]

    m_comb = _measures(fc, prior)
    m_bed = _measures(fb, prior)
    m_voc = _measures(fv, prior)
    if m_comb is None or m_bed is None or m_voc is None:
        return None

    return InfoDynRow(
        label=pair.label,
        set_id=pair.set_id,
        is_positive=pair.is_positive,
        er_comb=m_comb["er"],
        pir_comb=m_comb["pir"],
        redun_comb=m_comb["redun"],
        surp_mean_comb=m_comb["mean_surprise"],
        surp_max_comb=m_comb["max_surprise"],
        surp_var_comb=m_comb["var_surprise"],
        pir_bed=m_bed["pir"],
        pir_vocal=m_voc["pir"],
        surp_mean_bed=m_bed["mean_surprise"],
        surp_mean_vocal=m_voc["mean_surprise"],
        pir_lift_vs_bed=m_comb["pir"] - m_bed["pir"],
        surp_lift_vs_bed=m_comb["mean_surprise"] - m_bed["mean_surprise"],
        duration_s=pair.duration_set_s,
    )


# ---------------------------------------------------------------------------
# Dataset assembly (verbatim from v2)
# ---------------------------------------------------------------------------


def _build_pairs(librosa) -> tuple[list[PairV2], list[PairV2]]:
    rng = np.random.default_rng(42)
    sd11 = _load_set_data(BB11_SET_ID, BB11_YAML, BB11_DIR)
    sd12 = _load_set_data(BB12_SET_ID, BB12_YAML, BB12_DIR)
    all_sets = [sd11, sd12]
    positives = _build_positive_pairs_clean(sd11) + _build_positive_pairs_clean(sd12)

    cooc_map: dict[str, set[str]] = {}
    for sd in all_sets:
        beds = _collect_bed_tracks(sd)
        for t in sd.gt_tracks:
            if t.claimed_stem != "acappella":
                continue
            key = f"{sd.set_id}/{t.slot_label}"
            cooc_map[key] = {
                f"{b.set_id}/{b.slot_label}"
                for b in beds
                if b.set_start_s < t.set_end_s and b.set_end_s > t.set_start_s
            }
    all_beds = _collect_bed_tracks(sd11) + _collect_bed_tracks(sd12)
    negatives = _build_negative_pairs_clean(
        positives, all_beds, cooc_map, rng, neg_per_pos=3
    )
    return positives, negatives


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save(metrics: dict[str, float], n_pos: int, n_neg: int) -> None:
    AUX_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(AUX_DB))
    con.execute("""CREATE TABLE IF NOT EXISTS analysis_results (
        analysis_name TEXT NOT NULL, metric TEXT NOT NULL, group_key TEXT NOT NULL,
        value REAL, unit TEXT, notes TEXT,
        computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (analysis_name, metric, group_key))""")
    rows = [
        (ANALYSIS_NAME, k, "all", float(v), "roc_auc_or_nats", "")
        for k, v in metrics.items()
    ]
    rows.append((ANALYSIS_NAME, "n_pos", "all", float(n_pos), "count", ""))
    rows.append((ANALYSIS_NAME, "n_neg", "all", float(n_neg), "count", ""))
    con.executemany(
        "INSERT OR REPLACE INTO analysis_results "
        "(analysis_name, metric, group_key, value, unit, notes) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    print(f"\nSaved to {AUX_DB} :: analysis_results / {ANALYSIS_NAME}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(limit: int | None = None) -> None:
    librosa = _import_librosa()
    print("=== P1: HuBERT-token information dynamics for mashup compatibility ===\n")

    prior = _build_bb_prior(librosa)

    print("\nBuilding pairs (clean bed, tempo-aligned; reused from v2)...")
    positives, negatives = _build_pairs(librosa)
    all_pairs = positives + negatives
    if limit:
        # keep class balance for a smoke test
        pos = [p for p in all_pairs if p.is_positive][: max(1, limit // 4)]
        neg = [p for p in all_pairs if not p.is_positive][: limit - len(pos)]
        all_pairs = pos + neg
    print(
        f"  {sum(p.is_positive for p in all_pairs)} pos + "
        f"{sum(not p.is_positive for p in all_pairs)} neg = {len(all_pairs)} pairs\n"
    )

    print("Computing per-pair info dynamics (HuBERT render → tokens → PIR)...")
    rows: list[InfoDynRow] = []
    n_failed = 0
    for i, p in enumerate(all_pairs):
        if i % 20 == 0:
            print(f"  {i}/{len(all_pairs)} ({len(rows)} ok, {n_failed} failed)...")
        r = _pair_infodyn(p, prior, librosa)
        if r is not None:
            rows.append(r)
        else:
            n_failed += 1

    n_pos = sum(1 for r in rows if r.is_positive)
    n_neg = sum(1 for r in rows if not r.is_positive)
    print(f"\nUsable: {n_pos} pos + {n_neg} neg ({n_failed} failed)")
    if n_pos < 5 or n_neg < 5:
        print("BLOCKED: too few usable pairs.")
        sys.exit(1)

    combined_feats = [
        "pir_comb",
        "er_comb",
        "redun_comb",
        "surp_mean_comb",
        "surp_max_comb",
        "surp_var_comb",
        "pir_lift_vs_bed",
        "surp_lift_vs_bed",
    ]
    control_feats = ["pir_bed", "pir_vocal", "surp_mean_bed", "surp_mean_vocal"]
    aucs = _loso_auc(rows, combined_feats + control_feats)
    key_baseline = _key_baseline_auc(rows)

    # inverted-U readout on PIR
    pir_pos = np.array([r.pir_comb for r in rows if r.is_positive])
    pir_neg = np.array([r.pir_comb for r in rows if not r.is_positive])
    er_pos = np.array([r.er_comb for r in rows if r.is_positive])
    er_neg = np.array([r.er_comb for r in rows if not r.is_positive])

    print("\n" + "=" * 64)
    print(f"RESULTS — LOSO AUC (BB11 ↔ BB12), key baseline = {key_baseline:.3f}")
    print("=" * 64)
    print("  COMBINED-audio measures:")
    for f in combined_feats:
        print(f"    {f:20s}: {aucs.get(f, float('nan')):.3f}")
    print("  CONTROL (marginal) measures:")
    for f in control_feats:
        print(f"    {f:20s}: {aucs.get(f, float('nan')):.3f}")
    print(f"    {'multi_lr(combined)':20s}: {aucs.get('multi_lr', float('nan')):.3f}")

    print("\n  Inverted-U readout (PIR of combined audio):")
    print(f"    mean PIR  positives: {pir_pos.mean():.4f}  (n={len(pir_pos)})")
    print(f"    mean PIR  negatives: {pir_neg.mean():.4f}  (n={len(pir_neg)})")
    print(f"    mean entropy_rate  pos: {er_pos.mean():.3f}  neg: {er_neg.mean():.3f}")
    pir_auc = aucs.get("pir_comb", float("nan"))
    interaction_margin = pir_auc - aucs.get("pir_bed", float("nan"))
    print(
        f"    PIR-AUC − bed-alone-PIR-AUC (interaction margin): {interaction_margin:+.3f}"
    )

    # verdict
    best_comb = max(
        (aucs.get(f, float("nan")) for f in combined_feats),
        default=float("nan"),
    )
    if best_comb > 0.60 and interaction_margin >= 0.05:
        verdict = "GO"
    elif best_comb > 0.55:
        verdict = "WEAK"
    else:
        verdict = "NULL"
    print("\n" + "=" * 64)
    print(
        f"VERDICT: {verdict}  (best combined AUC={best_comb:.3f}, "
        f"interaction margin={interaction_margin:+.3f})"
    )
    n = n_pos + n_neg
    ci = 1.96 * 0.5 / (n**0.5)
    print(f"  n={n}; 95% CI half-width ≈ ±{ci:.3f} — treat sub-CI diffs as noise.")
    print("=" * 64)

    metrics = {f"auc_{k}": v for k, v in aucs.items()}
    metrics["mean_pir_pos"] = float(pir_pos.mean())
    metrics["mean_pir_neg"] = float(pir_neg.mean())
    metrics["interaction_margin"] = float(interaction_margin)
    _save(metrics, n_pos, n_neg)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)

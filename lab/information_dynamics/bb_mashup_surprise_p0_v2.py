"""P0 v2: does a CLEANER measurement lift the chroma info-dynamic surprise
signal out of weak-GO territory?

Two variants layered on the v1 experiment
(``bb_mashup_surprise_p0.py``):

  variant A — CLEAN BED
      Replace the noisy Roformer ``mix_instrumental`` bed proxy with the
      co-occurring track's ISOLATED instrumental stem (individual-track stem,
      separated from the clean reference track, not from the DJ mix).
        positives = acappella vocal over its GT co-occurring track's clean instr
        negatives = same vocal over a NON-co-occurring, gate-passing track's
                    clean instr

  variant B — CLEAN BED + TEMPO-ALIGNED
      Same as A, but before extracting chroma, resample each stem's reference
      audio by its GT ``tempo_ratio`` (= ref_span_s / set_span_s) so that both
      the vocal and the bed are expressed on the SET (mix) time-grid — their
      beat-frames line up. Frame-local KL is then a fair harmonic comparison.

Surprise definition, features, KL, classifier, and key-baseline are UNCHANGED
from v1 (imported directly). The only change is HOW the (vocal, bed) chroma
pairs are built and time-aligned.

Time-alignment math
-------------------
GT gives, per span:  set_start_s, set_end_s, ref_start_s, tempo_ratio.
    tempo_ratio = ref_span_s / set_span_s   (how fast the ref plays vs the mix)
At a mix (set) time ``st`` inside a span, the reference offset is
    ref_time(st) = ref_start_s + (st - set_start_s) * tempo_ratio
For a positive, the shared window is the overlap of the acap span and the bed
span; both the vocal and bed ref offsets are computed from their own GT rows.

For variant B, we resample the sliced ref audio by 1/tempo_ratio so 1 set-second
== 1 output-second for both stems (librosa time-stretch via resample of the
raw signal; cheap and preserves chroma pitch classes because we resample the
*already-loaded* audio at a rate change that we then re-interpret at the SET
tempo — see ``_load_chroma_tempo``).

Usage
-----
    venvs/audio/bin/python -m lab.information_dynamics.bb_mashup_surprise_p0_v2

Persists headline metrics to aux.db under analysis_name='bb_mashup_surprise_p0_v2'
(does NOT overwrite v1). Updates FINDINGS.md is done by hand.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
import warnings
from dataclasses import dataclass, replace

import numpy as np

# Reuse everything from v1 --------------------------------------------------
from lab.information_dynamics.bb_mashup_surprise_p0 import (
    ALIGNING,
    AUX_DB,
    BB11_DIR,
    BB11_SET_ID,
    BB11_YAML,
    BB12_DIR,
    BB12_SET_ID,
    BB12_YAML,
    CHROMA_HOP_S,
    MAX_PAIR_S,
    MIN_PAIR_S,
    FeatureRow,
    Pair,
    SetData,
    _bpm_compatible,
    _bpm_from_path,
    _extract_features,
    _import_librosa,
    _import_sklearn,
    _key_baseline_auc,
    _key_compatible,
    _key_from_path,
    _kl_per_frame,
    _load_set_data,
    _loso_auc,
    _normalise_chroma,
    _smooth_chroma,
)

REPO = pathlib.Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Bed-candidate model (clean isolated instrumental stems)
# ---------------------------------------------------------------------------


@dataclass
class BedTrack:
    """A GT bed span whose ISOLATED instrumental stem is available on disk."""

    set_id: str
    slot_label: str
    track_id: str | None
    instr_path: pathlib.Path
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    tempo_ratio: float
    bpm: float | None
    key: str | None


def _bed_ref_time(bed: BedTrack, set_time_s: float) -> float:
    """Reference offset into the bed stem at a given mix (set) time."""
    return bed.ref_start_s + (set_time_s - bed.set_start_s) * bed.tempo_ratio


def _collect_bed_tracks(sd: SetData) -> list[BedTrack]:
    """All regular/instrumental GT spans with an isolated instr stem on disk."""
    beds: list[BedTrack] = []
    for t in sd.gt_tracks:
        if t.claimed_stem not in ("regular", "instrumental"):
            continue
        if not t.track_id or t.track_id not in sd.manifest:
            continue
        mt = sd.manifest[t.track_id]
        instr = mt.get("stems", {}).get("instrumental")
        if not instr:
            continue
        ip = pathlib.Path(instr)
        if not ip.exists():
            continue
        beds.append(
            BedTrack(
                set_id=sd.set_id,
                slot_label=t.slot_label,
                track_id=t.track_id,
                instr_path=ip,
                set_start_s=t.set_start_s,
                set_end_s=t.set_end_s,
                ref_start_s=t.ref_start_s,
                tempo_ratio=t.tempo_ratio or 1.0,
                bpm=_bpm_from_path(ip),
                key=_key_from_path(ip),
            )
        )
    return beds


# ---------------------------------------------------------------------------
# Pair with tempo-ratio metadata (extends v1 Pair with ratios)
# ---------------------------------------------------------------------------


@dataclass
class PairV2:
    label: str
    set_id: str
    is_positive: bool
    vocal_path: pathlib.Path
    vocal_ref_start_s: float
    vocal_tempo_ratio: float
    bed_path: pathlib.Path
    bed_ref_start_s: float
    bed_tempo_ratio: float
    duration_set_s: float  # window length in SET (mix) seconds
    vocal_bpm: float | None
    bed_bpm: float | None
    vocal_key: str | None
    bed_key: str | None


# ---------------------------------------------------------------------------
# Positive / negative construction (clean bed)
# ---------------------------------------------------------------------------


def _build_positive_pairs_clean(sd: SetData) -> list[PairV2]:
    """One pair per acappella span with a co-occurring clean-instr bed."""
    beds = _collect_bed_tracks(sd)
    pairs: list[PairV2] = []
    for t in sd.gt_tracks:
        if t.claimed_stem != "acappella":
            continue
        if not t.track_id or t.track_id not in sd.manifest:
            continue
        mt = sd.manifest[t.track_id]
        vocal_path = pathlib.Path(mt["local_path"])
        if not vocal_path.exists():
            continue
        acap_tr = t.tempo_ratio or 1.0
        # find co-occurring bed(s)
        cooc = [
            b
            for b in beds
            if b.set_start_s < t.set_end_s and b.set_end_s > t.set_start_s
        ]
        if not cooc:
            continue

        # pick the bed with the largest temporal overlap
        def _ovl(b: BedTrack) -> float:
            return min(t.set_end_s, b.set_end_s) - max(t.set_start_s, b.set_start_s)

        bed = max(cooc, key=_ovl)

        overlap_lo = max(t.set_start_s, bed.set_start_s)
        overlap_hi = min(t.set_end_s, bed.set_end_s)
        dur = min(overlap_hi - overlap_lo, MAX_PAIR_S)
        if dur < MIN_PAIR_S:
            continue

        vocal_ref0 = t.ref_start_s + (overlap_lo - t.set_start_s) * acap_tr
        bed_ref0 = _bed_ref_time(bed, overlap_lo)

        pairs.append(
            PairV2(
                label=f"{sd.set_id}/{t.slot_label}|bed={bed.slot_label}",
                set_id=sd.set_id,
                is_positive=True,
                vocal_path=vocal_path,
                vocal_ref_start_s=max(0.0, vocal_ref0),
                vocal_tempo_ratio=acap_tr,
                bed_path=bed.instr_path,
                bed_ref_start_s=max(0.0, bed_ref0),
                bed_tempo_ratio=bed.tempo_ratio,
                duration_set_s=dur,
                vocal_bpm=_bpm_from_path(vocal_path),
                bed_bpm=bed.bpm,
                vocal_key=_key_from_path(vocal_path),
                bed_key=bed.key,
            )
        )
    return pairs


def _build_negative_pairs_clean(
    positives: list[PairV2],
    all_beds: list[BedTrack],
    cooccurring_beds_by_vocal: dict[str, set[str]],
    rng: np.random.Generator,
    neg_per_pos: int = 3,
) -> list[PairV2]:
    """Same vocal over a NON-co-occurring, gate-passing clean-instr bed."""
    negatives: list[PairV2] = []
    for pos in positives:
        vocal_key = pos.label.split("|")[0]  # set_id/slot
        cooc_bed_ids = cooccurring_beds_by_vocal.get(vocal_key, set())
        cands = []
        for bed in all_beds:
            bed_id = f"{bed.set_id}/{bed.slot_label}"
            # must NOT be a co-occurring bed for this vocal (identity by set/slot)
            if bed_id in cooc_bed_ids:
                continue
            # also exclude the exact bed used in the positive
            if bed.instr_path == pos.bed_path:
                continue
            if not _key_compatible(pos.vocal_key, bed.key):
                continue
            if not _bpm_compatible(pos.vocal_bpm, bed.bpm):
                continue
            cands.append(bed)
        if not cands:
            # relax BPM if too few (BPM tags often missing on stems)
            cands = [
                b
                for b in all_beds
                if f"{b.set_id}/{b.slot_label}" not in cooc_bed_ids
                and b.instr_path != pos.bed_path
                and _key_compatible(pos.vocal_key, b.key)
            ]
        if not cands:
            continue
        chosen = rng.choice(
            len(cands), size=min(neg_per_pos, len(cands)), replace=False
        )
        for idx in chosen:
            bed = cands[int(idx)]
            # bed length in set seconds
            bed_set_span = bed.set_end_s - bed.set_start_s
            dur = min(pos.duration_set_s, bed_set_span, MAX_PAIR_S)
            if dur < MIN_PAIR_S:
                continue
            negatives.append(
                PairV2(
                    label=f"{pos.label.split('|')[0]}→{bed.set_id}/{bed.slot_label}",
                    set_id=pos.set_id,
                    is_positive=False,
                    vocal_path=pos.vocal_path,
                    vocal_ref_start_s=pos.vocal_ref_start_s,
                    vocal_tempo_ratio=pos.vocal_tempo_ratio,
                    bed_path=bed.instr_path,
                    bed_ref_start_s=bed.ref_start_s,
                    bed_tempo_ratio=bed.tempo_ratio,
                    duration_set_s=dur,
                    vocal_bpm=pos.vocal_bpm,
                    bed_bpm=bed.bpm,
                    vocal_key=pos.vocal_key,
                    bed_key=bed.key,
                )
            )
    return negatives


# ---------------------------------------------------------------------------
# Chroma loaders — plain (variant A) and tempo-aligned (variant B)
# ---------------------------------------------------------------------------


def _load_chroma_plain(
    path: pathlib.Path,
    ref_start_s: float,
    dur_ref_s: float,
    librosa,
) -> np.ndarray | None:
    """Load chroma over [ref_start_s, ref_start_s+dur_ref_s] of the audio.

    (variant A: no tempo correction — read the ref span as-is.)
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(
                str(path),
                sr=22050,
                mono=True,
                offset=ref_start_s,
                duration=dur_ref_s,
            )
        if len(y) < sr * MIN_PAIR_S:
            return None
        hop = int(CHROMA_HOP_S * sr)
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop, bins_per_octave=36
        )
        return chroma.astype(np.float32)
    except Exception as e:
        print(f"  [chroma err] {path.name}: {e}", file=sys.stderr)
        return None


def _load_chroma_tempo(
    path: pathlib.Path,
    ref_start_s: float,
    dur_set_s: float,
    tempo_ratio: float,
    librosa,
) -> np.ndarray | None:
    """Load chroma expressed on the SET time-grid (variant B).

    We read ``dur_set_s * tempo_ratio`` seconds of REF audio (that's the amount
    of reference material that plays during ``dur_set_s`` mix-seconds), then
    time-stretch by 1/tempo_ratio so the output spans ``dur_set_s`` output-
    seconds. Pitch classes are preserved (phase-vocoder time-stretch), so chroma
    is unaffected; only the time-axis is normalised to the set tempo.
    """
    dur_ref_s = dur_set_s * tempo_ratio
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(
                str(path),
                sr=22050,
                mono=True,
                offset=ref_start_s,
                duration=dur_ref_s,
            )
        if len(y) < sr * MIN_PAIR_S * 0.5:
            return None
        if abs(tempo_ratio - 1.0) > 1e-3:
            # stretch so ref-duration -> set-duration: rate = tempo_ratio
            # librosa.effects.time_stretch(y, rate) makes output len = len/rate
            # we want output_len = len / tempo_ratio  -> rate = tempo_ratio
            y = librosa.effects.time_stretch(y, rate=tempo_ratio)
        if len(y) < sr * MIN_PAIR_S:
            return None
        hop = int(CHROMA_HOP_S * sr)
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop, bins_per_octave=36
        )
        return chroma.astype(np.float32)
    except Exception as e:
        print(f"  [chroma-tempo err] {path.name}: {e}", file=sys.stderr)
        return None


def _extract_pair_features_v2(
    pair: PairV2,
    librosa,
    *,
    tempo_align: bool,
) -> FeatureRow | None:
    if tempo_align:
        cv = _load_chroma_tempo(
            pair.vocal_path,
            pair.vocal_ref_start_s,
            pair.duration_set_s,
            pair.vocal_tempo_ratio,
            librosa,
        )
        cb = _load_chroma_tempo(
            pair.bed_path,
            pair.bed_ref_start_s,
            pair.duration_set_s,
            pair.bed_tempo_ratio,
            librosa,
        )
    else:
        # variant A: read ref spans directly, length = set_dur * tempo_ratio
        cv = _load_chroma_plain(
            pair.vocal_path,
            pair.vocal_ref_start_s,
            pair.duration_set_s * pair.vocal_tempo_ratio,
            librosa,
        )
        cb = _load_chroma_plain(
            pair.bed_path,
            pair.bed_ref_start_s,
            pair.duration_set_s * pair.bed_tempo_ratio,
            librosa,
        )
    if cv is None or cb is None:
        return None
    n = min(cv.shape[1], cb.shape[1])
    if n < int(MIN_PAIR_S / CHROMA_HOP_S):
        return None
    cv = _normalise_chroma(_smooth_chroma(cv[:, :n]))
    cb = _normalise_chroma(_smooth_chroma(cb[:, :n]))
    kl = _kl_per_frame(cv, cb)
    feat = _extract_features(kl)
    return FeatureRow(
        label=pair.label,
        set_id=pair.set_id,
        is_positive=pair.is_positive,
        **feat,
        duration_s=pair.duration_set_s,
    )


# ---------------------------------------------------------------------------
# Runner for one variant
# ---------------------------------------------------------------------------


def _run_variant(
    positives: list[PairV2],
    negatives: list[PairV2],
    librosa,
    *,
    tempo_align: bool,
    variant_name: str,
) -> dict:
    all_pairs = positives + negatives
    print(f"\n--- Variant {variant_name} (tempo_align={tempo_align}) ---")
    print(f"  building features for {len(all_pairs)} pairs...")
    rows: list[FeatureRow] = []
    n_failed = 0
    for i, p in enumerate(all_pairs):
        if i % 40 == 0:
            print(f"    {i}/{len(all_pairs)} ({len(rows)} ok, {n_failed} failed)...")
        r = _extract_pair_features_v2(p, librosa, tempo_align=tempo_align)
        if r is not None:
            rows.append(r)
        else:
            n_failed += 1
    n_pos = sum(1 for r in rows if r.is_positive)
    n_neg = sum(1 for r in rows if not r.is_positive)
    print(f"  done: {n_pos} pos + {n_neg} neg ({n_failed} failed)")
    if n_pos < 5 or n_neg < 5:
        print("  BLOCKED: too few usable pairs")
        return {
            "variant": variant_name,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "blocked": True,
        }

    feature_names = [
        "mean_kl",
        "var_kl",
        "max_kl",
        "skew_kl",
        "invU_coeff",
        "kl_drop_spike",
    ]
    aucs = _loso_auc(rows, feature_names)
    key_baseline = _key_baseline_auc(rows)
    single = {f: aucs[f] for f in feature_names}
    best_feature = max(
        single, key=lambda f: single[f] if not np.isnan(single[f]) else -1
    )
    return {
        "variant": variant_name,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "aucs": aucs,
        "key_baseline": key_baseline,
        "best_feature": best_feature,
        "best_feature_auc": single[best_feature],
        "multi_lr": aucs.get("multi_lr", float("nan")),
        "blocked": False,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_v2(results: list[dict]) -> None:
    con = sqlite3.connect(str(AUX_DB))
    con.execute("""CREATE TABLE IF NOT EXISTS analysis_results (
        analysis_name TEXT NOT NULL, metric TEXT NOT NULL, group_key TEXT NOT NULL,
        value REAL, unit TEXT, notes TEXT,
        computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (analysis_name, metric, group_key))""")
    NAME = "bb_mashup_surprise_p0_v2"
    rows = []
    for res in results:
        if res.get("blocked"):
            continue
        g = res["variant"]
        rows.append(
            (
                NAME,
                "auc_multi_lr",
                g,
                res["multi_lr"],
                "roc_auc",
                "LOSO multi-feature LR",
            )
        )
        rows.append(
            (
                NAME,
                "auc_key_baseline",
                g,
                res["key_baseline"],
                "roc_auc",
                "random on key-matched",
            )
        )
        rows.append((NAME, "n_pos", g, float(res["n_pos"]), "count", ""))
        rows.append((NAME, "n_neg", g, float(res["n_neg"]), "count", ""))
        rows.append(
            (
                NAME,
                "best_feature_auc",
                f"{g}:{res['best_feature']}",
                res["best_feature_auc"],
                "roc_auc",
                "",
            )
        )
        for f, v in res["aucs"].items():
            rows.append((NAME, f"auc_{f}", g, v, "roc_auc", "LOSO AUC"))
    con.executemany(
        "INSERT OR REPLACE INTO analysis_results "
        "(analysis_name, metric, group_key, value, unit, notes) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    print(f"\nSaved {len(rows)} rows to {AUX_DB} :: {NAME}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    librosa = _import_librosa()
    rng = np.random.default_rng(42)

    print("=== P0 v2: clean-bed + tempo-aligned mashup surprise ===\n")
    sd11 = _load_set_data(BB11_SET_ID, BB11_YAML, BB11_DIR)
    sd12 = _load_set_data(BB12_SET_ID, BB12_YAML, BB12_DIR)
    all_sets = [sd11, sd12]

    # Build clean positives
    pos11 = _build_positive_pairs_clean(sd11)
    pos12 = _build_positive_pairs_clean(sd12)
    positives = pos11 + pos12
    print(
        f"Clean positives: BB11 {len(pos11)} | BB12 {len(pos12)} | total {len(positives)}"
    )

    # co-occurrence map so negatives never reuse a genuine co-occurring bed
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
    print(f"Total clean bed tracks available: {len(all_beds)}")

    negatives = _build_negative_pairs_clean(
        positives, all_beds, cooc_map, rng, neg_per_pos=3
    )
    print(f"Clean negatives: {len(negatives)}")

    # Variant A: clean bed, no tempo align
    resA = _run_variant(
        positives, negatives, librosa, tempo_align=False, variant_name="clean_bed"
    )
    # Variant B: clean bed + tempo align
    resB = _run_variant(
        positives, negatives, librosa, tempo_align=True, variant_name="clean_bed_tempo"
    )

    # Report
    print("\n" + "=" * 64)
    print("V2 RESULTS")
    print("=" * 64)
    for res in (resA, resB):
        if res.get("blocked"):
            print(
                f"\n{res['variant']}: BLOCKED (n_pos={res['n_pos']}, n_neg={res['n_neg']})"
            )
            continue
        print(f"\n{res['variant']}:")
        print(f"  n_pos={res['n_pos']}  n_neg={res['n_neg']}")
        print(f"  key_baseline AUC = {res['key_baseline']:.3f}")
        for f in [
            "mean_kl",
            "var_kl",
            "max_kl",
            "skew_kl",
            "invU_coeff",
            "kl_drop_spike",
        ]:
            print(f"    {f:16s}: {res['aucs'][f]:.3f}")
        print(f"    {'multi_lr':16s}: {res['multi_lr']:.3f}")
        print(
            f"  best single feature: {res['best_feature']} ({res['best_feature_auc']:.3f})"
        )

    _save_v2([resA, resB])

    # Decision
    print("\n" + "=" * 64)
    best_headline = max(
        (r for r in (resA, resB) if not r.get("blocked")),
        key=lambda r: max(r["multi_lr"], r["best_feature_auc"]),
        default=None,
    )
    if best_headline is None:
        print("DECISION: BLOCKED — insufficient clean pairs.")
        return
    headline = max(best_headline["multi_lr"], best_headline["best_feature_auc"])
    print(
        f"Best variant: {best_headline['variant']} "
        f"headline AUC {headline:.3f} vs key-baseline {best_headline['key_baseline']:.3f}"
    )
    if headline >= 0.60:
        print(
            "DECISION: GO — clean measurement lifts the signal to >=0.60. Build the ranker."
        )
    else:
        print(
            "DECISION: NULL-ish — signal stays ~0.565. Lean on the verb-log preference "
            "model; mean_kl is a weak feature, not a headline."
        )
    print("=" * 64)


if __name__ == "__main__":
    main()

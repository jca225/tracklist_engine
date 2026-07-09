"""Phase-0 follow-up: is the acappella warp DERIVED from the BPM gap?

Tests the low-rank claim: an acappella dropped at mix-time T is warped so its
tempo matches the bed it plays over. Formally

    tempo_ratio_acap  ==  mix_BPM(T) / acap_native_BPM
    mix_BPM(T)        ==  bed_native_BPM * bed_tempo_ratio   (bed covers T)

so predicted_acap_ratio = (bed_BPM * bed_tempo_ratio) / acap_BPM. If actual ≈
predicted, the acappella warp is not a free knob — it's pinned by the BPM gap,
and the synthetic generator should DERIVE it (sample a mix BPM, propagate),
never sample tempo_ratio directly.

Native BPM per recording comes from pi-storage (regular-stem Essentia rows,
median); acappella recordings inherit their parent song's regular BPM since GT
track_id == recording_id. Run extract_warp_table.py first is not required — this
reloads the fixtures directly. BPM file = /tmp/bpm.txt (recording_id|stem|bpm),
pulled via:
    ssh pi-storage sqlite3 ... "SELECT recording_id,stem,bpm ... bpm>0"
"""

from __future__ import annotations

import statistics
from pathlib import Path

from labeling.ground_truth.schema import load

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "labeling" / "fixtures"
SETS = {"bb11": FIX / "bb11_ground_truth.yaml", "bb12": FIX / "bb12_ground_truth.yaml"}
BPM_FILE = Path("/tmp/bpm.txt")


def load_bpm() -> dict[str, float]:
    """recording_id -> median regular-stem native BPM."""
    by_rid: dict[str, list[float]] = {}
    for line in BPM_FILE.read_text().splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        rid, stem, bpm = parts
        if stem != "regular":
            continue
        try:
            by_rid.setdefault(rid, []).append(float(bpm))
        except ValueError:
            continue
    return {r: statistics.median(v) for r, v in by_rid.items() if v}


def main() -> None:
    bpm = load_bpm()
    rows: list[dict] = []
    covered = 0
    for name, path in SETS.items():
        r = load(path)
        gt = r.value
        beds = [
            t
            for t in gt.tracks
            if t.claimed_stem in ("instrumental", "regular")
            and t.tempo_ratio
            and t.track_id in bpm
        ]
        for t in gt.tracks:
            if t.claimed_stem != "acappella" or not t.tempo_ratio:
                continue
            acap_bpm = bpm.get(t.track_id or "")
            if not acap_bpm:
                continue
            # bed covering this acappella's set_start
            T = t.set_start_s
            cov = [b for b in beds if b.set_start_s <= T <= b.set_end_s]
            if not cov:
                # nearest bed by start-time as fallback
                cov = sorted(beds, key=lambda b: abs(b.set_start_s - T))[:1]
                if not cov:
                    continue
            bed = min(cov, key=lambda b: abs(b.set_start_s - T))
            bed_bpm = bpm[bed.track_id]
            mix_bpm = bed_bpm * bed.tempo_ratio
            pred = mix_bpm / acap_bpm
            covered += 1
            rows.append(
                {
                    "set": name,
                    "slot": t.slot_label,
                    "acap_bpm": round(acap_bpm, 1),
                    "bed_bpm": round(bed_bpm, 1),
                    "mix_bpm": round(mix_bpm, 1),
                    "pred_ratio": round(pred, 4),
                    "actual_ratio": round(t.tempo_ratio, 4),
                    "err": round(t.tempo_ratio - pred, 4),
                    "abs_err": round(abs(t.tempo_ratio - pred), 4),
                }
            )

    print(f"acappella spans with acap+bed BPM available: {covered}")
    if not rows:
        return
    abserr = sorted(r["abs_err"] for r in rows)
    n = len(abserr)
    within = lambda thr: sum(1 for e in abserr if e <= thr)
    print("\n-- |actual - predicted| tempo_ratio --")
    print(
        f"  median={statistics.median(abserr):.4f}  mean={statistics.mean(abserr):.4f}"
    )
    print(f"  p90={abserr[int(0.9 * (n - 1))]:.4f}  max={abserr[-1]:.4f}")
    for thr in (0.02, 0.05, 0.10):
        print(f"  within ±{thr:.2f}: {within(thr)}/{n} ({100 * within(thr) // n}%)")

    # naive baseline: predicting ratio==1 (no warp) — how much better is BPM model?
    naive = sorted(abs(r["actual_ratio"] - 1.0) for r in rows)
    print("\n-- baseline |actual - 1.0| (predict no warp) --")
    print(
        f"  median={statistics.median(naive):.4f}  within ±0.05: "
        f"{sum(1 for e in naive if e <= 0.05)}/{n}"
    )

    # fold check: does allowing half/double time help the tail?
    folded = []
    for r in rows:
        cands = [r["pred_ratio"], r["pred_ratio"] * 2, r["pred_ratio"] / 2]
        folded.append(min(abs(r["actual_ratio"] - c) for c in cands))
    folded.sort()
    print("\n-- with half/double-time fold allowed --")
    print(
        f"  median={statistics.median(folded):.4f}  within ±0.05: "
        f"{sum(1 for e in folded if e <= 0.05)}/{n} ({100 * sum(1 for e in folded if e <= 0.05) // n}%)"
    )

    print("\nworst 8 residuals:")
    for r in sorted(rows, key=lambda r: -r["abs_err"])[:8]:
        print(
            f"  {r['set']} {r['slot']:6s} acap={r['acap_bpm']:6.1f} bed={r['bed_bpm']:6.1f} "
            f"pred={r['pred_ratio']:.3f} actual={r['actual_ratio']:.3f} err={r['err']:+.3f}"
        )


if __name__ == "__main__":
    main()

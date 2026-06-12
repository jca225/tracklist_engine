"""EDA: what harmonic relationship do BB12 DJs actually mash in?

The key/tempo baseline used raw semitone distance (AUC 0.63) — but DJs mix by the
CAMELOT WHEEL (harmonic mixing): same key, relative major/minor, or +-1 (perfect
fifth) are 'compatible'. This explores the real harmonic relationship between
mashed pairs vs random, to see if Camelot separates better than chromatic distance
— and lists the real pairs that are harmonically FAR (relative-key mashups, or
label noise / wrong downloads to investigate).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from workspaces.mashup_compat.baseline import _fetch_key_bpm, _key_dist
from workspaces.mashup_compat.pairs import extract_pairs

GT = "labeling/fixtures/bb12_ground_truth.yaml"

# pitch-class -> Camelot number (major = 'B' side; minor borrows relative major's number)
_MAJ = {0: 8, 7: 9, 2: 10, 9: 11, 4: 12, 11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7}


def camelot(pc: int, mode: str) -> tuple[int, str]:
    if mode == "minor":
        return (_MAJ[(pc + 3) % 12], "A")
    return (_MAJ[pc], "B")


def _num_dist(n1: int, n2: int) -> int:
    d = abs(n1 - n2) % 12
    return min(d, 12 - d)


def relationship(c1: tuple[int, str], c2: tuple[int, str]) -> str:
    if c1 == c2:
        return "same"
    nd = _num_dist(c1[0], c2[0])
    if c1[0] == c2[0]:
        return "relative"            # relative major/minor (same number, diff letter)
    if nd == 1 and c1[1] == c2[1]:
        return "fifth"               # +-1 same letter = perfect fifth/fourth
    if nd == 1:
        return "adj_diag"            # diagonal neighbor (energy/key change)
    return "far"


COMPATIBLE = {"same", "relative", "fifth"}     # textbook harmonic-mixing set


def main() -> int:
    pairs = extract_pairs(GT)
    feat = _fetch_key_bpm([p.bed.track_id for p in pairs] + [p.payload.track_id for p in pairs])

    rows = []
    for p in pairs:
        b, q = feat.get(p.bed.track_id), feat.get(p.payload.track_id)
        if not b or not q:
            continue
        cb, cq = camelot(b[0], b[1]), camelot(q[0], q[1])
        rows.append((p, relationship(cb, cq), _key_dist(b[0], q[0]),
                     relationship(cb, cq) in COMPATIBLE))
    pos = [r for r in rows if r[0].positive]
    neg = [r for r in rows if not r[0].positive]
    print(f"usable: pos={len(pos)} neg={len(neg)}\n")

    print(f"{'relationship':>10} {'%pos':>7} {'%neg':>7}")
    for rel in ["same", "relative", "fifth", "adj_diag", "far"]:
        fp = 100 * np.mean([r[1] == rel for r in pos])
        fn = 100 * np.mean([r[1] == rel for r in neg])
        print(f"{rel:>10} {fp:>6.0f}% {fn:>6.0f}%")
    pc = 100 * np.mean([r[3] for r in pos]); nc = 100 * np.mean([r[3] for r in neg])
    print(f"\nCamelot-compatible (same/relative/fifth):  pos {pc:.0f}%  vs  neg {nc:.0f}%")

    y = np.array([1 if r[0].positive else 0 for r in rows])
    auc_cam = roc_auc_score(y, np.array([1.0 if r[3] else 0.0 for r in rows]))
    auc_chrom = roc_auc_score(y, -np.array([r[2] for r in rows]))
    print(f"\nAUC  Camelot-compatible : {auc_cam:.3f}")
    print(f"AUC  chromatic distance : {auc_chrom:.3f}   (the baseline's feature)")

    print("\nreal mashups that are harmonically FAR (relative-key creativity, OR wrong-download noise?):")
    for r in sorted([r for r in pos if r[1] == "far"], key=lambda r: -r[2])[:10]:
        print(f"  keydist={r[2]}  BED {r[0].bed.label:34s} <- {r[0].payload.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

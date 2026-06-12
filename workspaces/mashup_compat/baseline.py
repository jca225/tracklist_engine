"""Key/tempo baseline: how far does explicit harmonic+tempo compatibility get?

Diagnostic for the mashup-compat bet. The GT says DJs mash songs <=1 semitone apart
in key and warp tempo to fit — so a rule over (key distance, bpm ratio) should
separate real BB12 mashups from random. This sets the CEILING a fine-tuned MERT
must beat to be worth it. Key/BPM read from each recording's FULL-track Essentia
features (a stem inherits its song's key), sidestepping no-Essentia-on-acappellas.

  venvs/audio/bin/python -m workspaces.mashup_compat.baseline
"""
from __future__ import annotations

import subprocess

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from workspaces.mashup_compat.pairs import PI, PI_DB, extract_pairs

GT = "labeling/fixtures/bb12_ground_truth.yaml"


def _fetch_key_bpm(rec_ids: list[str]) -> dict[str, tuple[int, str, float]]:
    ids = ",".join(f"'{r}'" for r in sorted(set(rec_ids)))
    sql = (
        "SELECT ta.recording_id, taf.key_pc, taf.key_mode, taf.bpm "
        "FROM track_audio ta JOIN track_audio_features taf ON taf.track_audio_id=ta.track_audio_id "
        f"WHERE ta.recording_id IN ({ids}) AND ta.stem='regular' AND taf.key_pc IS NOT NULL "
        "ORDER BY ta.is_reference DESC;"
    )
    out = subprocess.run(["ssh", PI, f"sqlite3 {PI_DB} \"{sql}\""],
                         capture_output=True, text=True, check=True).stdout
    m: dict[str, tuple[int, str, float]] = {}
    for line in out.strip().splitlines():
        rid, pc, mode, bpm = line.split("|")
        if rid not in m:                      # first = is_reference
            m[rid] = (int(pc), mode or "", float(bpm) if bpm else 0.0)
    return m


def _key_dist(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)                      # circular semitone distance 0..6


def _bpm_fold(ra: float, rb: float) -> float:
    if ra <= 0 or rb <= 0:
        return 0.5
    r = np.log2(ra / rb)
    return abs(r - round(r))                   # distance to nearest octave (half/double ok) 0..0.5


def main() -> int:
    pairs = extract_pairs(GT)
    feat = _fetch_key_bpm([p.bed.track_id for p in pairs] + [p.payload.track_id for p in pairs])

    rows, y, groups = [], [], []
    for p in pairs:
        b, q = feat.get(p.bed.track_id), feat.get(p.payload.track_id)
        if not b or not q:
            continue
        rows.append([_key_dist(b[0], q[0]), 1.0 if b[1] == q[1] else 0.0, _bpm_fold(b[2], q[2])])
        y.append(1 if p.positive else 0)
        groups.append(p.payload.track_id)
    X = np.array(rows); y = np.array(y); groups = np.array(groups)
    print(f"usable pairs with key/bpm: {len(y)} (pos={int(y.sum())}, neg={int((1-y).sum())}) "
          f"| key/bpm coverage of recordings: {len(feat)}\n")

    keyd, modem, bpmd = X[:, 0], X[:, 1], X[:, 2]
    print("single-feature AUC (compatible = small distance):")
    print(f"  key distance      : {roc_auc_score(y, -keyd):.3f}")
    print(f"  mode match        : {roc_auc_score(y, modem):.3f}")
    print(f"  bpm fold distance : {roc_auc_score(y, -bpmd):.3f}")

    # combined logistic, grouped CV by payload (no leakage)
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"\ncombined key+mode+bpm (grouped CV): AUC {auc:.3f}")
    print(f"\n>>> ceiling for explicit key/tempo = {auc:.3f}  "
          f"(MERT fine-tune must beat this to add value; section-MERT probe was 0.62)")
    # distribution sanity
    print(f"\nkey-distance: pos median={np.median(keyd[y==1]):.1f}  neg median={np.median(keyd[y==0]):.1f}"
          f"  (expect pos << neg if DJs mash in-key)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

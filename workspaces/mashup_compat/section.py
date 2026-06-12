"""Section-level mashup-compat: embed the EXACT mashed sections, not whole songs.

The whole-song proof was weak (cos AUC ~0.59) — averaging a 4-min song drowns the
specific 16-bar moment the DJ layered. Here we embed only the GT ref-span each
track actually played (per-segment), pair the segments that co-occur in mix-time,
and re-test per-layer separability. Reuses the locally-cached stems (no re-pull).

  venvs/audio/bin/python -m workspaces.mashup_compat.section
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from analysis.adapters import audio_io, mert_adapter
from labeling.ground_truth.schema import GroundTruthTrack, load
from workspaces.mashup_compat.embed import (CACHE_DIR, GRID_S, LOCAL_STEMS, _decode,
                                            _pull_stem, _resolve_track_audio_ids)
from workspaces.mashup_compat.pairs import BED_STEMS, MIN_OVERLAP_S, PAYLOAD_STEMS
from workspaces.mashup_compat.proof import _grouped_probe_auc, _l2n

SEC_EMB = CACHE_DIR / "bb12_section_embeds.pkl"
GT = "labeling/fixtures/bb12_ground_truth.yaml"


@dataclass(frozen=True)
class Section:
    track_id: str
    role: str            # bed | payload
    seg: int
    ref_start: float
    ref_end: float
    mix_start: float
    mix_end: float

    @property
    def key(self):
        return (self.track_id, self.role, self.seg)

    @property
    def stem_file(self):
        return "instrumental" if self.role == "bed" else "vocals"


def _sections(t: GroundTruthTrack, role: str) -> list[Section]:
    out: list[Section] = []
    if t.ref_segments:
        segs = sorted(t.ref_segments, key=lambda s: s.mix_start_s)
        for i, s in enumerate(segs):
            mix_end = segs[i + 1].mix_start_s if i + 1 < len(segs) else t.set_end_s
            if s.ref_end_s > s.ref_start_s:
                out.append(Section(t.track_id, role, i, s.ref_start_s, s.ref_end_s, s.mix_start_s, mix_end))
    elif t.ref_end_s and t.ref_end_s > t.ref_start_s:
        out.append(Section(t.track_id, role, 0, t.ref_start_s, t.ref_end_s, t.set_start_s, t.set_end_s))
    return out


def _mix_overlap(a: Section, b: Section) -> float:
    return max(0.0, min(a.mix_end, b.mix_end) - max(a.mix_start, b.mix_start))


def build_pairs():
    gt = load(GT).value
    elig = [t for t in gt.tracks if t.track_id and not t.unalignable]
    beds = [s for t in elig if t.claimed_stem in BED_STEMS for s in _sections(t, "bed")]
    pays = [s for t in elig if t.claimed_stem in PAYLOAD_STEMS for s in _sections(t, "payload")]

    pos, pos_keys = [], set()
    for p in pays:
        for b in beds:
            if b.track_id != p.track_id and _mix_overlap(b, p) >= MIN_OVERLAP_S:
                pos.append((b, p)); pos_keys.add((b.key, p.key))
    # negatives: bed/payload sections that never co-occur (deterministic stride)
    cand = [(b, p) for b in beds for p in pays
            if b.track_id != p.track_id and _mix_overlap(b, p) == 0.0 and (b.key, p.key) not in pos_keys]
    want = len(pos) * 3
    neg = cand[:: max(1, len(cand) // want)][:want]
    return pos, neg


def _section_token(h, samples: np.ndarray, s: Section) -> np.ndarray | None:
    a, b = int(s.ref_start * mert_adapter.MERT_SR), int(s.ref_end * mert_adapter.MERT_SR)
    seg = samples[a:b]
    if seg.size < mert_adapter.MERT_SR // 2:        # < 0.5s, too short
        return None
    dur = seg.size / mert_adapter.MERT_SR
    grid = tuple(float(x) for x in np.arange(0.0, dur, GRID_S)) + (float(dur),)
    if len(grid) < 2:
        grid = (0.0, float(dur))
    emb = mert_adapter.embed_track_per_measure(h, seg, track_audio_id=0, measure_times=grid)
    if not emb.is_ok() or not emb.value:
        return None
    return np.mean(np.stack([_decode(m) for m in emb.value]), axis=0).astype(np.float16)


def embed_sections(sections: list[Section]) -> dict:
    cache = pickle.loads(SEC_EMB.read_bytes()) if SEC_EMB.is_file() else {}
    todo = [s for s in sections if s.key not in cache]
    if not todo:
        return cache
    taids = _resolve_track_audio_ids([s.track_id for s in todo])
    h = mert_adapter.load().value
    by_stem = defaultdict(list)
    for s in todo:
        by_stem[(s.track_id, s.role)].append(s)

    for n, ((tid, role), segs) in enumerate(by_stem.items(), 1):
        taid = taids.get(tid)
        if taid is None:
            continue
        sp = _pull_stem(taid, segs[0].stem_file)        # local; --ignore-existing
        if sp is None:
            continue
        wf = audio_io.load_mono(sp, target_sr=mert_adapter.MERT_SR)
        if not wf.is_ok():
            continue
        for s in segs:
            tok = _section_token(h, wf.value.samples, s)
            if tok is not None:
                cache[s.key] = tok
        if n % 10 == 0:
            SEC_EMB.write_bytes(pickle.dumps(cache))
            print(f"  embedded {n}/{len(by_stem)} stems' sections")
    SEC_EMB.write_bytes(pickle.dumps(cache))
    return cache


def main() -> int:
    pos, neg = build_pairs()
    sections = list({s.key: s for b, p in (pos + neg) for s in (b, p)}.values())
    print(f"section pairs: pos={len(pos)} neg={len(neg)} | distinct sections={len(sections)}")
    cache = embed_sections(sections)

    rows, y, groups = [], [], []
    for b, p in pos + neg:
        if b.key in cache and p.key in cache:
            rows.append((cache[b.key].astype(np.float32), cache[p.key].astype(np.float32)))
            y.append(1 if (b, p) in pos else 0)
            groups.append(p.track_id)
    y = np.array(y); groups = np.array(groups)
    beds = np.stack([r[0] for r in rows]); pays = np.stack([r[1] for r in rows])
    n_layers = beds.shape[1]
    print(f"usable section pairs: {len(y)} (pos={int(y.sum())}, neg={int((1-y).sum())}) | "
          f"groups={len(np.unique(groups))}\n")

    bn = np.stack([[_l2n(beds[i, L]) for L in range(n_layers)] for i in range(len(beds))])
    pn = np.stack([[_l2n(pays[i, L]) for L in range(n_layers)] for i in range(len(pays))])

    print(f"{'layer':>5} {'cos_AUC':>8} {'probe_AUC':>10}")
    best = (0, 0.0, 0.0)
    for L in range(n_layers):
        cos = np.sum(bn[:, L] * pn[:, L], axis=1)
        auc_cos = roc_auc_score(y, cos)
        auc_probe = _grouped_probe_auc(bn[:, L], pn[:, L], y, groups)
        print(f"{L:>5} {auc_cos:>8.3f} {auc_probe:>10.3f}")
        if max(auc_cos, auc_probe if not np.isnan(auc_probe) else 0) > max(best[1], best[2]):
            best = (L, auc_cos, auc_probe)
    peak = max(best[1], best[2])
    print(f"\nbest: layer {best[0]}  cos={best[1]:.3f} probe={best[2]:.3f}")
    print("verdict:", "SIGNAL — section-level carries mashup compatibility" if peak >= 0.62
          else "still weak — pivot to learned compat head (corruption objective)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

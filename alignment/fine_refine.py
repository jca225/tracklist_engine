"""Decoupled per-span DTW fine-placement refinement.

The MERT monotonic decode (`predict_sequence`) places spans to ~37 s MAE — a
coarse tiling prior, because pooled MERT cannot localise within the mix. This
stage refines each start using actual audio: it takes the predicted source
section (at the predicted ref_start, routed to the mix's *instrumental* stem)
and runs a subsequence DTW of that query against a ±`band_s` slice of the mix
chromagram around the coarse start, snapping the start to the best local match.

Why DECOUPLED (per-span), not one global program-to-mix DTW:

  A single global warp couples every span, so neighbours pin locally-ambiguous
  drops — powerful when identity is perfect (BB12 oracle: median 21 s). But that
  same coupling is catastrophically fragile: a 4/147 (2.7 %) identity error on
  *training* spans mid-mix re-routes the global path and propagates 80 s errors
  to eval spans with perfect identity (slot 004: 0.1 s -> 80 s). With the
  aligner's real predicted identity the global method degrades to median 32 s.
  Placing each span independently in its own corridor removes the cascade — a
  wrong section can only hurt itself — and on BB12 with predicted identity beats
  both coarse (37 s) and the coupled global method (32 s): **median 26 s**,
  sub-8 s 3 -> 7, sub-16 s 6 -> 11. The cost is losing the global fix for spans
  whose *coarse* is off by more than `band_s` (slot 004 stays ~43 s); fixing the
  joint-decode within-slot identity swap (slots 058/059) is the path back to the
  coupled 21 s, tracked separately.

Two hard-won rules baked in (see docs/fine_placement_plan.md):

  * **Route by audio, not label.** Most BB12 "acappella" rows are full tracks
    (project_acappella_label_vs_audio), so everything routes to the instrumental
    mix stem. Trusting `claimed_stem` here regresses placement badly.
  * **Constrain to a corridor.** Unconstrained subsequence matching on the
    harmonically self-similar mix wanders; the ±band corridor around the coarse
    prior is what keeps each match local and honest.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .records import SpanPrediction, SpanTarget

log = logging.getLogger(__name__)

SR = 22050
HOP = 512
COL_S = 1.0
POOL = max(1, round(COL_S * SR / HOP))
_CORRIDOR_PENALTY = 10.0
_ALIGNING_ROOT = Path.home() / "aligning"


def _chroma_cols(y: np.ndarray) -> np.ndarray | None:
    """CQT chroma mean-pooled to ~COL_S columns, L2-normalised per column."""
    import librosa

    c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    n = c.shape[1] // POOL
    if n < 1:
        return None
    c = c[:, : n * POOL].reshape(12, n, POOL).mean(2)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


def _col_time(col: float) -> float:
    return col * POOL * HOP / SR


def find_aligning_dir(set_id: str) -> Path | None:
    """~/aligning/{set_id}__* — the manual-labeling folder for this set."""
    hits = sorted(_ALIGNING_ROOT.glob(f"{set_id}__*"))
    return hits[0] if hits else None


@dataclass
class AudioContext:
    """Mix instrumental chromagram + lazy per-recording source chroma."""

    mix_chroma: np.ndarray            # (12, Nmix)
    tid2path: dict[str, str]
    _cache: dict[tuple[str, int, int], np.ndarray | None]

    @property
    def n_cols(self) -> int:
        return self.mix_chroma.shape[1]

    @classmethod
    def from_set(cls, set_id: str) -> AudioContext | None:
        """Load mix_instrumental.flac + manifest from the aligning folder."""
        import librosa

        d = find_aligning_dir(set_id)
        if d is None:
            log.warning("fine_refine: no aligning dir for %s", set_id)
            return None
        stem = d / "mix_instrumental.flac"
        manifest = d / "manifest.json"
        if not stem.exists() or not manifest.exists():
            log.warning("fine_refine: missing mix_instrumental/manifest in %s", d)
            return None
        mix_y = librosa.load(str(stem), sr=SR, mono=True)[0]
        mix_chroma = _chroma_cols(mix_y)
        if mix_chroma is None:
            return None
        man = json.loads(manifest.read_text())
        tid2path = {t["track_id"]: t["local_path"] for t in man["tracks"]}
        return cls(mix_chroma=mix_chroma, tid2path=tid2path, _cache={})

    def source_chroma(self, rid: str, ref_start_s: float, dur_s: float) -> np.ndarray | None:
        import librosa

        key = (rid, int(round(ref_start_s)), int(round(dur_s)))
        if key in self._cache:
            return self._cache[key]
        path = self.tid2path.get(rid)
        seg = None
        if path:
            try:
                y, _ = librosa.load(str(path), sr=SR, mono=True, offset=max(0.0, ref_start_s), duration=dur_s)
                seg = _chroma_cols(y)
            except Exception as e:  # noqa: BLE001 — audio decode is best-effort
                log.warning("fine_refine: source load failed for %s: %s", rid, e)
        self._cache[key] = seg
        return seg


_MAX_QUERY_S = 60.0    # cap the per-span query length fed to the local DTW


def refine_placements(
    preds: tuple[SpanPrediction, ...],
    targets: tuple[SpanTarget, ...],
    ctx: AudioContext,
    *,
    band_s: float = 30.0,
    gate_z: float | None = None,
) -> tuple[SpanPrediction, ...]:
    """Refine each coarse start via a decoupled per-span local DTW.

    For span `i`: load its predicted source section, route to the instrumental
    mix stem, subsequence-DTW the query against the mix slice within ±`band_s`
    of the coarse start, and snap the start to the best local match. Spans are
    placed independently — no shared warp — so an identity error on one span
    cannot move another (see module docstring on why coupling is unsafe).

    `gate_z`: if set, keep the coarse start when the local match is not peaked
    by at least this many SDs over the corridor (flat match => self-similar =>
    trust coarse). None disables the gate.
    """
    import librosa

    col1 = _col_time(1)
    n_cols = ctx.n_cols
    out = list(preds)
    for i, p in enumerate(preds):
        rid = p.recording_id
        if not rid:
            continue
        q_dur = min(max(p.set_end_s - p.set_start_s, 8.0), _MAX_QUERY_S)
        q = ctx.source_chroma(rid, float(p.ref_start_s), q_dur)
        if q is None or q.shape[1] < 2:
            continue
        q_cols = q.shape[1]
        lo = max(0.0, p.set_start_s - band_s)
        c0 = int(lo / col1)
        c1 = min(n_cols, int((p.set_start_s + band_s) / col1) + q_cols + 1)
        if c1 - c0 < q_cols + 1:
            continue  # corridor too small to slide the query
        seg = ctx.mix_chroma[:, c0:c1]
        cost = 1.0 - q.T @ seg                                  # (q_cols, c1-c0)
        _D, wp = librosa.sequence.dtw(C=cost.astype(np.float64), subseq=True, backtrack=True)
        start_col = c0 + int(wp[-1, 1])                         # mix col where the match begins

        if gate_z is not None:
            # per-start cost across the corridor: how peaked is this match?
            row = cost[0]
            if len(row) >= 3:
                z = (row.mean() - row.min()) / (row.std() + 1e-9)
                if z < gate_z:
                    continue

        new_start = _col_time(start_col)
        delta = new_start - p.set_start_s
        out[i] = replace(p, set_start_s=new_start, set_end_s=p.set_end_s + delta)
    return tuple(out)

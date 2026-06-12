"""Shared data substrate for the information-dynamics study.

Loads the bar-synchronous mix MERT artifact and the YAML ground truth, fits a
single VQ codebook shared by every model so M0/M1/M2 speak the same alphabet,
and provides frame<->time helpers plus the labeled-region mask.

Tokenization caveat (read before judging prequentiality): the codebook is fit
on the *whole* mix. That is a fixed, unsupervised perceptual quantizer — it
never sees song boundaries or future targets — analogous to a fixed auditory
front-end. The thing that must be (and is) strictly prequential is the
*probabilistic sequence model*, not the quantizer. We keep the global codebook
to stay comparable to the existing findings.md result and the paper's spirit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eda.alignment.artifacts import MixMertArtifact, load_mix_mert_artifact
from eda.alignment.boundaries import gt_section_starts_s
from eda.alignment.mert_vectors import l2_normalize
from eda.alignment.tokenize import fit_vq_kmeans
from labeling.ground_truth.schema import load as load_gt


@dataclass(frozen=True)
class StudyData:
    set_id: str
    artifact: MixMertArtifact
    mert_clean: np.ndarray      # (n_frames, dim) float32 — non-finite bars repaired
    tokens: np.ndarray          # (n_frames,) int64 — shared codebook ids
    centroids: np.ndarray       # (K, dim) float32, L2-normalized
    n_tokens: int
    bar_start_s: np.ndarray     # (n_frames,) float64
    bar_end_s: np.ndarray       # (n_frames,)
    gt_boundary_s: np.ndarray   # (n_gt,) merged section-start times (seconds)
    labeled_lo_s: float         # first / last labeled times — scoring window
    labeled_hi_s: float

    @property
    def n_frames(self) -> int:
        return int(self.tokens.shape[0])

    def time_of_frame(self, idx: int) -> float:
        return float(self.bar_start_s[idx])

    def frames_to_times(self, frames: np.ndarray | tuple[int, ...]) -> np.ndarray:
        return self.bar_start_s[np.asarray(list(frames), dtype=int)]

    def labeled_frame_mask(self) -> np.ndarray:
        """Bool mask of frames inside the human-labeled span [lo, hi]."""
        return (self.bar_start_s >= self.labeled_lo_s) & (
            self.bar_start_s <= self.labeled_hi_s
        )


def _load_and_tokenize(
    artifact_path: Path | str, *, n_tokens: int, seed: int
) -> tuple[MixMertArtifact, np.ndarray, np.ndarray, np.ndarray]:
    """Load a mix MERT artifact, repair non-finite bars, fit the shared VQ."""
    art = load_mix_mert_artifact(Path(artifact_path))
    # Sanitize any non-finite MERT bars (float16 accumulate overflow, see
    # findings.md note on bar 36) before clustering.
    mert = np.asarray(art.mert, dtype=np.float32).copy()
    bad = ~np.isfinite(mert).all(axis=1)
    if bad.any():
        mert[bad] = np.nanmean(mert[~bad], axis=0)
    centroids, tokens = fit_vq_kmeans(mert, n_tokens, seed=seed)
    return art, mert, centroids, tokens


def _merge_boundaries(starts: np.ndarray, merge_tol_s: float) -> np.ndarray:
    """Collapse boundaries closer than ``merge_tol_s`` (keep the earlier)."""
    s = np.sort(np.asarray(starts, dtype=float))
    if s.size == 0:
        return s
    keep = [s[0]]
    for t in s[1:]:
        if t - keep[-1] >= merge_tol_s:
            keep.append(t)
    return np.asarray(keep, dtype=float)


def study_data_from_boundaries(
    artifact_path: Path | str,
    boundaries_s: np.ndarray | list[float] | tuple[float, ...],
    *,
    labeled_lo_s: float | None = None,
    labeled_hi_s: float | None = None,
    n_tokens: int = 24,
    seed: int = 0,
    merge_tol_s: float = 0.5,
) -> StudyData:
    """Build StudyData from a mix MERT artifact + an *explicit* boundary list.

    Decouples the study from the hand-labelled GT YAML so the same significance
    test can run against any boundary source — e.g. scraped *tracklist cue times*
    for sets that have no manual Ableton labelling. ``labeled_lo_s/hi_s`` default
    to the full bar grid (the tracklist spans the whole mix).
    """
    art, mert, centroids, tokens = _load_and_tokenize(
        artifact_path, n_tokens=n_tokens, seed=seed
    )
    starts = _merge_boundaries(np.asarray(boundaries_s, dtype=float), merge_tol_s)
    bar_start = np.asarray(art.bar_start_s, dtype=np.float64)
    bar_end = np.asarray(art.bar_end_s, dtype=np.float64)
    lo = float(bar_start[0]) if labeled_lo_s is None else float(labeled_lo_s)
    hi = float(bar_end[-1]) if labeled_hi_s is None else float(labeled_hi_s)
    return StudyData(
        set_id=art.set_id,
        artifact=art,
        mert_clean=mert,
        tokens=tokens.astype(np.int64),
        centroids=l2_normalize(centroids.astype(np.float32)),
        n_tokens=n_tokens,
        bar_start_s=bar_start,
        bar_end_s=bar_end,
        gt_boundary_s=starts,
        labeled_lo_s=lo,
        labeled_hi_s=hi,
    )


def load_study_data(
    artifact_path: Path | str,
    gt_path: Path | str,
    *,
    n_tokens: int = 24,
    seed: int = 0,
    merge_tol_s: float = 0.5,
) -> StudyData:
    art, mert, centroids, tokens = _load_and_tokenize(
        artifact_path, n_tokens=n_tokens, seed=seed
    )

    gt_res = load_gt(Path(gt_path))
    if not gt_res.is_ok():
        raise ValueError(f"failed to load GT: {gt_res.error}")
    gt = gt_res.value
    starts = np.asarray(gt_section_starts_s(gt, merge_tol_s=merge_tol_s), dtype=float)
    ends = [t.set_end_s for t in gt.tracks]
    labeled_lo = float(min(t.set_start_s for t in gt.tracks))
    labeled_hi = float(max(ends)) if ends else float(art.bar_end_s[-1])

    return StudyData(
        set_id=art.set_id,
        artifact=art,
        mert_clean=mert,
        tokens=tokens.astype(np.int64),
        centroids=l2_normalize(centroids.astype(np.float32)),
        n_tokens=n_tokens,
        bar_start_s=np.asarray(art.bar_start_s, dtype=np.float64),
        bar_end_s=np.asarray(art.bar_end_s, dtype=np.float64),
        gt_boundary_s=starts,
        labeled_lo_s=labeled_lo,
        labeled_hi_s=labeled_hi,
    )


def normalized_mert(data: StudyData) -> np.ndarray:
    """L2-normalized MERT rows (matches the codebook geometry)."""
    return l2_normalize(np.asarray(data.mert_clean, dtype=np.float32))

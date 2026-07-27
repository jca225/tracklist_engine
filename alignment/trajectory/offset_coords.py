"""Offset-coordinate answer encoding for the TRM decoder (bake-off §2.3).

The conv decoder emits a per-frame categorical over `Tr+1` ref bins — a
*variable*-size answer (Tr differs per span), which fights TRM's fixed-grid
assumption. Instead encode the answer as per-frame **clip-start offset** bins
over a FIXED vocabulary:

    offset_s(t) = ref_position(t) - mix_relative_time(t)

A straight clip is a run of constant offset (grid-lock); a tempo change bends
it; a section-cut is a discontinuity (the DJ's jump). Decoding inverts back to
ref bins and reuses `targets.frames_to_segments`, so the answer lands in the
exact coordinate `path_decode`/`trajectory_acc` already consume.

Vocabulary layout (fixed): offset classes `0 .. (hi_bin-lo_bin)` then one NULL
class at the end. `IGNORE_INDEX` (-100) marks frames excluded from the loss
(outside-ref / unalignable), matching torch's CE `ignore_index` default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE_INDEX = -100


@dataclass(frozen=True)
class OffsetVocab:
    """A fixed offset-bin vocabulary. `res_bin` feature-bins per offset class
    (1 = full resolution). Offset bin `b` -> class index `b - lo_bin`; the NULL
    class is the last index."""

    lo_bin: int
    hi_bin: int
    res_bin: int = 1

    @property
    def n_offsets(self) -> int:
        return (self.hi_bin - self.lo_bin) + 1

    @property
    def size(self) -> int:
        return self.n_offsets + 1  # + NULL

    @property
    def null_index(self) -> int:
        return self.size - 1

    @property
    def zero_index(self) -> int:
        """Class index of offset 0 (so `index - zero_index` recovers the bin)."""
        return -self.lo_bin

    def offset_bin_to_index(self, off_bin: np.ndarray) -> np.ndarray:
        idx = off_bin - self.lo_bin
        return np.clip(idx, 0, self.n_offsets - 1)

    def index_to_offset_bin(self, index: np.ndarray) -> np.ndarray:
        return index + self.lo_bin


def encode_offset_labels(
    ref_pos_s: np.ndarray,
    times_s: np.ndarray,
    kind: np.ndarray,
    bin_s: float,
    vocab: OffsetVocab,
) -> np.ndarray:
    """Per-frame offset-class labels. POSITION frames -> clamped offset index;
    NULL -> null_index; IGNORE -> IGNORE_INDEX. `times_s` is mix-relative (bin
    centers); `ref_pos_s` is absolute ref seconds."""
    from .targets import KIND_IGNORE, KIND_NULL, KIND_POSITION

    offset_s = ref_pos_s - times_s
    off_bin = np.rint(offset_s / (bin_s * vocab.res_bin)).astype(np.int64)
    labels = vocab.offset_bin_to_index(off_bin).astype(np.int64)
    labels[kind == KIND_NULL] = vocab.null_index
    labels[kind == KIND_IGNORE] = IGNORE_INDEX
    # defensive: any non-POSITION/NULL/IGNORE code is treated as ignore
    known = (kind == KIND_POSITION) | (kind == KIND_NULL) | (kind == KIND_IGNORE)
    labels[~known] = IGNORE_INDEX
    return labels


def offset_labels_to_ref_bins(
    labels: np.ndarray, vocab: OffsetVocab
) -> tuple[np.ndarray, np.ndarray]:
    """Invert offset labels to `(ref_bin, null_mask)` for
    `targets.frames_to_segments`. NULL and IGNORE frames terminate a segment
    (null_mask True); position frames map to `ref_bin[t] = t + offset_bin`."""
    labels = np.asarray(labels)
    null_mask = (labels == vocab.null_index) | (labels == IGNORE_INDEX)
    off_bin = vocab.index_to_offset_bin(labels)
    t = np.arange(len(labels))
    ref_bin = np.where(null_mask, 0, t + off_bin).astype(np.int64)
    return ref_bin, null_mask


__all__ = [
    "IGNORE_INDEX",
    "OffsetVocab",
    "encode_offset_labels",
    "offset_labels_to_ref_bins",
]

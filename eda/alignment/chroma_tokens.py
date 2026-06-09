"""Beat-sync chroma token stream — interpretable side-channel for info-dynamics."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eda.alignment.artifacts import MixMertArtifact
from eda.alignment.tokenize import fit_vq_kmeans


def _bar_chroma_vectors(
    audio_path: Path,
    artifact: MixMertArtifact,
    *,
    sr: int = 22050,
) -> np.ndarray:
    import librosa

    y, loaded_sr = librosa.load(str(audio_path), sr=sr, mono=True)
    out = np.zeros((artifact.n_bars, 12), dtype=np.float32)
    for i in range(artifact.n_bars):
        a = int(max(0.0, artifact.bar_start_s[i]) * loaded_sr)
        b = int(min(len(y), artifact.bar_end_s[i] * loaded_sr))
        if b - a < loaded_sr // 50:
            continue
        seg = y[a:b]
        chroma = librosa.feature.chroma_cqt(y=seg, sr=loaded_sr)
        out[i] = chroma.mean(axis=1).astype(np.float32)
    return out


def chroma_token_labels(
    audio_path: Path,
    artifact: MixMertArtifact,
    n_tokens: int,
) -> np.ndarray:
    """Per-bar chroma VQ labels aligned to the MERT artifact grid."""
    vectors = _bar_chroma_vectors(audio_path, artifact)
    if np.allclose(vectors, 0):
        raise ValueError(f"chroma extraction produced all zeros for {audio_path}")
    _, labels = fit_vq_kmeans(vectors, n_tokens)
    return labels

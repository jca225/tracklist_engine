"""HuBERT phonetic embeddings for vocal placement / verification.

Promoted out of the attic'd ``workspaces.section_hsmm.similarity_probe`` so the
aligner keeps a self-contained sensor after the focus cut.
"""

from __future__ import annotations

import numpy as np

from alignment.refine_ref_offsets import HOP, SR

_HUBERT_SR = 16000
_HUBERT_MODEL = "facebook/hubert-base-ls960"
_HUBERT_LAYER = 9  # mid layers carry the most phonetic content
_HUBERT_FPS = 50.0  # base model: 320-sample hop @ 16 kHz
_HUBERT_CHUNK_S = 30.0  # cap GPU memory on hour-long mixes
_hub_cache: dict = {}


def _hubert_model():
    if "m" not in _hub_cache:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        fe = AutoFeatureExtractor.from_pretrained(_HUBERT_MODEL)
        model = AutoModel.from_pretrained(_HUBERT_MODEL).to(dev).eval()
        _hub_cache["m"] = (model, fe, dev)
    return _hub_cache["m"]


def _resample_cols(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linear-interpolate a (D, T_in) feature onto T_out columns."""
    t_in = x.shape[1]
    if t_in == n_out or t_in < 2:
        return x
    src = np.linspace(0.0, 1.0, t_in)
    dst = np.linspace(0.0, 1.0, n_out)
    return np.stack([np.interp(dst, src, row) for row in x]).astype(np.float32)


def _hubert(y: np.ndarray, layer: int = _HUBERT_LAYER) -> np.ndarray:
    """(768, frames) L2-normed per frame, resampled onto the SR/HOP grid."""
    import librosa
    import torch

    model, fe, dev = _hubert_model()
    y16 = librosa.resample(y, orig_sr=SR, target_sr=_HUBERT_SR)
    step = int(_HUBERT_CHUNK_S * _HUBERT_SR)
    hs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(y16), step):
            chunk = y16[i : i + step]
            if len(chunk) < 400:  # < conv receptive field
                continue
            iv = fe(chunk, sampling_rate=_HUBERT_SR, return_tensors="pt")
            h = model(iv.input_values.to(dev), output_hidden_states=True).hidden_states[
                layer
            ][0]  # (T, 768)
            hs.append(h.float().cpu().numpy())
    if not hs:
        return np.zeros((768, 0), dtype=np.float32)
    h = np.concatenate(hs, axis=0).T  # (768, T@50fps)
    n_out = max(1, int(round(len(y) / HOP)))  # match librosa frame count @ SR/HOP
    h = _resample_cols(h, n_out)
    return (h / (np.linalg.norm(h, axis=0, keepdims=True) + 1e-8)).astype(np.float32)

"""Essentia worker — runs under venvs/essentia/ (Python 3.13).

Invoked by `essentia_adapter.py` via subprocess. Reads an audio file path
from argv, runs Essentia's signal-processing algorithms plus any TF
classifier heads whose .pb file is present under data/essentia_models/,
and prints one JSON blob to stdout. Errors go to stdout as
`{"error": ...}` with exit 1.

NEVER imported from the Py 3.14 audio venv. The adapter spawns it as a
subprocess to keep the Essentia install isolated. (Essentia ships wheels
for cp310-cp313 only; the rest of the pipeline runs on cp314.)

Two modes:
    python -m ... essentia_worker --ensure-models
        Downloads any missing .pb files into data/essentia_models/ and exits.
    python -m ... essentia_worker <audio_path>
        Analyzes the audio and prints features.
"""
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from . import essentia_models as em


# YAMNet AudioSet class indices (from audioset-yamnet-1.json).
_YAMNET_SPEECH = 0
_YAMNET_CONVERSATION = 2
_YAMNET_SINGING = 24
_YAMNET_CHEERING = 61
_YAMNET_APPLAUSE = 62
_YAMNET_CROWD = 64


def _ensure_models_cmd() -> int:
    report = em.ensure_downloaded()
    print(json.dumps({
        "downloaded": list(report.downloaded),
        "skipped": list(report.skipped),
        "failed": [asdict(f) for f in report.failed],
    }))
    return 0 if not report.failed else 1


def _run(audio_path: str) -> dict[str, object]:
    import essentia.standard as es
    import numpy as np

    models = em.by_name()
    present = em.which_present()
    mdir = em.models_dir()

    # Two loaders: 44.1k for the SP algos, 16k for every TF model.
    audio_44k = es.MonoLoader(filename=audio_path, sampleRate=44100)()
    audio_16k = es.MonoLoader(filename=audio_path, sampleRate=16000)()

    # --- Signal processing (no model files needed) ---
    key_tonic, key_mode, key_strength = es.KeyExtractor(profileType="edma")(audio_44k)
    bpm, beats, _, _, _ = es.RhythmExtractor2013(method="multifeature")(audio_44k)
    danceability_sp, _ = es.Danceability()(audio_44k)

    out: dict[str, object] = {
        "version": "essentia_v2",
        "models_present": sorted(present),
        "key": {
            "tonic": key_tonic,
            "mode": key_mode,
            "strength": float(key_strength),
            "profile": "edma",
        },
        "rhythm": {
            "bpm": float(bpm),
            "n_beats": int(len(beats)),
        },
        "danceability_sp": float(danceability_sp),
    }

    # --- TF: Discogs-EffNet embeddings + heads ---
    if "discogs_effnet" in present:
        effnet_path = str(em.model_path(models["discogs_effnet"], mdir))
        effnet = es.TensorflowPredictEffnetDiscogs(
            graphFilename=effnet_path,
            output=models["discogs_effnet"].output_node,
        )
        effnet_emb = np.asarray(effnet(audio_16k))   # [n_frames, 1280]

        def _effnet_head(name: str, positive_index: int) -> float | None:
            if name not in present:
                return None
            head = es.TensorflowPredict2D(
                graphFilename=str(em.model_path(models[name], mdir)),
                output=models[name].output_node,
            )
            preds = np.asarray(head(effnet_emb))     # [n_frames, 2]
            return float(preds.mean(axis=0)[positive_index])

        out["mood_happy"] = _effnet_head("mood_happy", 0)            # P(happy)
        out["mood_acoustic"] = _effnet_head("mood_acoustic", 0)      # P(acoustic)
        out["mood_aggressive"] = _effnet_head("mood_aggressive", 0)  # P(aggressive)
        # voice_instrumental classes: ['instrumental', 'voice']  → take index 1
        out["voice_prob"] = _effnet_head("voice_instrumental", 1)
        out["danceability_tf"] = _effnet_head("danceability_tf", 0)  # P(danceable)
    else:
        for k in ("mood_happy", "mood_acoustic", "mood_aggressive", "voice_prob", "danceability_tf"):
            out[k] = None

    # --- TF: MusiCNN embeddings + emomusic regressor ---
    if "msd_musicnn" in present and "emomusic" in present:
        musicnn = es.TensorflowPredictMusiCNN(
            graphFilename=str(em.model_path(models["msd_musicnn"], mdir)),
            output=models["msd_musicnn"].output_node,
        )
        musicnn_emb = np.asarray(musicnn(audio_16k))           # [n_segments, 200]

        emomusic = es.TensorflowPredict2D(
            graphFilename=str(em.model_path(models["emomusic"], mdir)),
            output=models["emomusic"].output_node,
        )
        va = np.asarray(emomusic(musicnn_emb))                 # [n_segments, 2]
        # Output order: (valence, arousal) — see emomusic-msd-musicnn-2.json description.
        # Range is the emoMusic 1..9 scale; rescale to 0..1.
        v_raw = float(va.mean(axis=0)[0])
        a_raw = float(va.mean(axis=0)[1])
        out["valence_raw"] = v_raw
        out["arousal_raw"] = a_raw
        out["valence"] = max(0.0, min(1.0, (v_raw - 1.0) / 8.0))
        out["arousal"] = max(0.0, min(1.0, (a_raw - 1.0) / 8.0))
    else:
        for k in ("valence", "arousal", "valence_raw", "arousal_raw"):
            out[k] = None

    # --- TF: YAMNet (audio event recognition) ---
    if "yamnet" in present:
        yamnet = es.TensorflowPredictVGGish(
            graphFilename=str(em.model_path(models["yamnet"], mdir)),
            input="melspectrogram",
            output=models["yamnet"].output_node,
        )
        acts = np.asarray(yamnet(audio_16k))                   # [n_frames, 521]
        mean = acts.mean(axis=0)
        peak = acts.max(axis=0)

        out["yamnet"] = {
            "speech_mean": float(mean[_YAMNET_SPEECH]),
            "conversation_mean": float(mean[_YAMNET_CONVERSATION]),
            "singing_mean": float(mean[_YAMNET_SINGING]),
            "cheering_max": float(peak[_YAMNET_CHEERING]),
            "applause_max": float(peak[_YAMNET_APPLAUSE]),
            "crowd_max": float(peak[_YAMNET_CROWD]),
        }
        # Spotify-shape derivations:
        # speechiness: speech-like vocal activity (excludes singing).
        out["speechiness"] = float(max(mean[_YAMNET_SPEECH], mean[_YAMNET_CONVERSATION]))
        # liveness: peak audience activity over the track.
        out["liveness"] = float(max(
            peak[_YAMNET_APPLAUSE], peak[_YAMNET_CHEERING], peak[_YAMNET_CROWD],
        ))
    else:
        out["yamnet"] = None
        out["speechiness"] = None
        out["liveness"] = None

    return out


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--ensure-models":
        return _ensure_models_cmd()
    if len(sys.argv) != 2:
        print(json.dumps({
            "error": "usage: essentia_worker.py [--ensure-models | <audio_path>]",
        }))
        return 1
    try:
        payload = _run(sys.argv[1])
    except Exception as e:
        print(json.dumps({
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }))
        return 1
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())

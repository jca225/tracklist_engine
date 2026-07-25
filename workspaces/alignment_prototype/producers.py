"""Bricks 7+8 — canonical feature producers over the provenance substrate.

Four REAL producers demonstrate the canonical feature-store path
(docs/engine/feature_store_and_registry.md):

- ``analyze.landmark_fp`` — fresh compute: audio bytes → landmark-fingerprint
  blob (the same `landmark_fp` serialization `track_fingerprints` stores),
  executed through ``Registry.run`` so the feature lands content-addressed with
  input/output/derivation edges and a deterministic ProcessSpec.
- ``migrate.mert_features`` — migration: an EXISTING cached MERT measure stack
  (``mert_store`` `.cache/mert/<set>_mert.npz`, computed on the cluster) is
  re-registered as a content-addressed ``mert_features`` artifact DERIVED FROM
  the audio artifact, with a ProcessSpec whose ``model_refs`` name the MERT
  model. Bytes are copied, never recomputed — this is how existing feature
  storage walks onto the canonical model.
- ``analyze.whisper`` (Brick 8) — fresh compute: audio bytes → word-timestamp
  transcript JSON via the REAL ``lyrics_align.transcribe_words`` Whisper path
  (openai/whisper-large-v3-turbo), wrapped as-is behind the pure bytes→bytes
  contract.
- ``analyze.hubert`` (Brick 8) — fresh compute: audio bytes → serialized
  HuBERT frame features via the REAL ``stem_placement.hubert_of`` path
  (facebook/hubert-base-ls960 layer 9, the mix_vocals placement features).

Plus ``register_trained_decoder``: the real trained checkpoint on disk
(`.cache/trajectory/decoder_slotsplit_seed0.pt`, the exact `trajectory/train.py:601`
payload the law-audit's law-14 row cites) becomes a content-addressed
MODEL_CHECKPOINT + TrainingSnapshot + ProcessSpec + FittedModel — grounding
laws 13/14 on real data. NOTE the honest naming: no MERT *identity head* is
persisted anywhere (`train.py --train-mert` trains in memory); the persisted
trained model in this repo is the trajectory decoder (structure axis), so that
is what gets registered.

Pi-storage access is READ-ONLY (scp one audio file down); if unreachable, pass
``--audio-file`` with a local file. Lives in the aligner package — imports
``core.provenance`` downward and aligner modules sideways; core imports none
of this.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from core.provenance import (
    REGISTRY,
    Artifact,
    ArtifactKind,
    ArtifactStore,
    Axis,
    FittedModel,
    ProvenanceRepository,
    artifact_type,
    capture_environment,
    make_fitted_model,
    make_parameter_set,
    make_process_spec,
    make_training_snapshot,
    transformation,
)
from workspaces.alignment_prototype.landmark_fp import (
    FHOP,
    NFFT,
    SR,
    fingerprint_from_file_streaming,
)
from workspaces.alignment_prototype.lyrics_align import MODEL as _WHISPER_MODEL_REF
from workspaces.alignment_prototype.mert_store import _PROBE_LAYER, _cache_path
from workspaces.alignment_prototype.refine_ref_offsets import (
    HOP as _GRID_HOP,
    SR as _GRID_SR,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_OUT_ROOT = Path(__file__).resolve().parent / "out" / "provenance" / "producers"
_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / ".cache"
    / "trajectory"
    / "decoder_slotsplit_seed0.pt"
)
_GT_FIXTURES = {
    "1fsnxchk": _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml",
    "2nvzlh2k": _REPO / "labeling" / "fixtures" / "bb11_ground_truth.yaml",
}
_MERT_MODEL_REF = "m-a-p/MERT-v1-330M"
# Mirrors workspaces/section_hsmm/similarity_probe.py (_HUBERT_MODEL/_HUBERT_LAYER),
# the module stem_placement.hubert_of stands on. Not imported: similarity_probe
# pulls sibling scorecard modules at import time; two constants beat that weight.
_HUBERT_MODEL_REF = "facebook/hubert-base-ls960"
_HUBERT_LAYER = 9
_PI_HOST = "pi-storage"
_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_MEDIA_TYPES = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
}


# --- declared artifact types -------------------------------------------------
@artifact_type(kind="landmark_fingerprint", media_type="application/json", version=1)
class LandmarkFingerprintArtifact:
    """Serialized `landmark_fp.LandmarkFingerprint.to_blob()` payload."""


@artifact_type(
    kind="mert_features", media_type="application/x-mert-measures", version=1
)
class MertFeaturesArtifact:
    """Per-measure MERT probe-layer vectors: JSON header + raw float arrays."""


# --- producer 1: fingerprint (fresh compute) ---------------------------------
def _sniff_suffix(content: bytes) -> str:
    head = content[:16]
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if head[:4] == b"RIFF":
        return ".wav"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:4] == b"OggS":
        return ".ogg"
    return ".audio"


@transformation(
    "analyze.landmark_fp",
    "1",
    inputs={"audio": "audio"},
    outputs={"features": "landmark_fingerprint"},
    stage="analysis",
    params={
        "sr": SR,
        "hop": FHOP,
        "n_fft": NFFT,
        "peak_size": 19,
        "db_floor": 60.0,
        "fan": 8,
        "dt_max": 80,
        "chunk_s": 120.0,
        "overlap_s": 3.0,
    },
)
def landmark_fp_producer(inputs: Mapping[str, bytes]) -> Mapping[str, bytes]:
    """Pure: audio bytes → serialized landmark fingerprint bytes."""
    content = inputs["audio"]
    with tempfile.NamedTemporaryFile(suffix=_sniff_suffix(content)) as tmp:
        tmp.write(content)
        tmp.flush()
        fp = fingerprint_from_file_streaming(tmp.name)
    return {"features": fp.to_blob()}


# --- producer 2: MERT (migrate existing cached features) ---------------------
def _serialize_mert_series(
    start: np.ndarray, end: np.ndarray, vec: np.ndarray, *, layer: int
) -> bytes:
    """Deterministic bytes for one recording's measure stack: one JSON header
    line, then the raw arrays. (np.savez zip timestamps are not deterministic;
    content-addressing wants byte-stable serialization.)"""
    header = {
        "v": 1,
        "kind": "mert_features",
        "model": _MERT_MODEL_REF,
        "layer": layer,
        "n_measures": int(vec.shape[0]),
        "dim": int(vec.shape[1]),
        "arrays": [
            {"name": "start_s", "dtype": str(start.dtype), "shape": list(start.shape)},
            {"name": "end_s", "dtype": str(end.dtype), "shape": list(end.shape)},
            {"name": "vectors", "dtype": str(vec.dtype), "shape": list(vec.shape)},
        ],
    }
    return (
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        + start.tobytes()
        + end.tobytes()
        + vec.tobytes()
    )


def migrate_cached_mert(
    recording_id: str,
    set_id: str,
    audio: Artifact,
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: str,
) -> Optional[Artifact]:
    """Register an EXISTING cached MERT measure stack as a canonical feature.

    Reads the local ``mert_store`` cache only (never triggers a pi export);
    returns None with a message when the cache or the recording is absent.
    This deliberately does NOT go through ``Registry.run``: migration is not a
    pure recompute from the input bytes — the feature bytes pre-exist — so it
    records its own ProcessSpec'd run with relation ``migrated_from_cache``.
    """
    cache = _cache_path(set_id)
    if not cache.is_file():
        print(f"  mert: no local cache for set {set_id} ({cache}) — skipped")
        return None
    with np.load(cache, allow_pickle=False) as z:
        ref_ids = json.loads(str(z["ref_ids"]))
        if recording_id not in ref_ids:
            print(f"  mert: recording {recording_id} not in {set_id} cache — skipped")
            return None
        start = np.asarray(z[f"ref_{recording_id}_start"])
        end = np.asarray(z[f"ref_{recording_id}_end"])
        vec = np.asarray(z[f"ref_{recording_id}_vec"])

    parameter_set = repo.record_parameter_set(
        make_parameter_set(
            "migrate.mert_features",
            {"layer": _PROBE_LAYER, "source_cache": cache.name, "set_id": set_id},
        )
    )
    environment = repo.record_environment_spec(capture_environment())
    spec = repo.record_process_spec(
        make_process_spec(
            name="migrate.mert_features",
            version="1",
            stage="analysis",
            code_commit=code_commit,
            parameter_set=parameter_set,
            environment=environment,
            model_refs=(_MERT_MODEL_REF, f"layer={_PROBE_LAYER}"),
            implementation_hash="",
        )
    )
    run = repo.begin_run_from_spec(spec)
    try:
        repo.add_input(run, audio, role="audio")
        feature = store.put_bytes(
            _serialize_mert_series(start, end, vec, layer=_PROBE_LAYER),
            kind="mert_features",
            media_type="application/x-mert-measures",
            source_uri=str(cache),
            source_system="mert_store_cache",
            metadata={
                "recording_id": recording_id,
                "set_id": set_id,
                "migrated": True,
                "artifact_type_version": 1,
            },
        )
        repo.add_output(run, feature, role="features")
        repo.derive(feature, audio, run, relation="migrated_from_cache")
        repo.succeed(run)
        return feature
    except Exception as exc:
        repo.fail(run, {"error": repr(exc)})
        raise


# --- producer 3: Whisper transcript (fresh compute, Brick 8) -----------------
@artifact_type(kind="whisper_transcript", media_type="application/json", version=1)
class WhisperTranscriptArtifact:
    """Word-timestamp ASR stream: JSON ``{model, n_words, words:[{w,s,e}]}`` —
    the exact ``lyrics_align.transcribe_words`` payload the lyrics channel
    consumes."""


@transformation(
    "analyze.whisper",
    "1",
    inputs={"audio": "audio"},
    outputs={"transcript": "whisper_transcript"},
    stage="analysis",
    params={
        "language": "en",
        "task": "transcribe",
        "block_s": 300.0,
        "chunk_length_s": 28,
        "stride_length_s": 4,
        "enhance_vocals": False,
    },
    reproducibility="best_effort",  # GPU/MPS ASR decode is not bit-stable
    model_refs=(_WHISPER_MODEL_REF,),
)
def whisper_producer(inputs: Mapping[str, bytes]) -> Mapping[str, bytes]:
    """Audio bytes → word-timestamp transcript JSON bytes.

    Wraps the REAL ``lyrics_align.transcribe_words`` Whisper path unchanged
    (enhance=False — VoiceFixer stays off, matching the raw-transcription
    default). The bytes land in a random temp file, so lyrics_align's
    mtime-keyed disk cache never aliases two different inputs."""
    from workspaces.alignment_prototype.lyrics_align import transcribe_words

    content = inputs["audio"]
    with tempfile.NamedTemporaryFile(suffix=_sniff_suffix(content)) as tmp:
        tmp.write(content)
        tmp.flush()
        words = transcribe_words(tmp.name, enhance=False)
    payload = {
        "v": 1,
        "kind": "whisper_transcript",
        "model": _WHISPER_MODEL_REF,
        "n_words": len(words),
        "words": words,
    }
    return {
        "transcript": json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    }


# --- producer 4: HuBERT features (fresh compute, Brick 8) --------------------
@artifact_type(
    kind="hubert_features", media_type="application/x-hubert-frames", version=1
)
class HubertFeaturesArtifact:
    """Phonetic frame features (D, T) on the SR/HOP grid: one JSON header
    line, then the raw float32 array — the ``stem_placement`` mix_vocals
    representation."""


def _serialize_hubert_features(feat: np.ndarray, *, layer: int) -> bytes:
    """Deterministic bytes for one HuBERT feature matrix (same header+raw
    scheme as ``_serialize_mert_series`` — byte-stable for content-addressing)."""
    feat = np.ascontiguousarray(feat, dtype=np.float32)
    header = {
        "v": 1,
        "kind": "hubert_features",
        "model": _HUBERT_MODEL_REF,
        "layer": layer,
        "dim": int(feat.shape[0]),
        "n_frames": int(feat.shape[1]),
        "grid": {"sr": _GRID_SR, "hop": _GRID_HOP},
        "arrays": [
            {"name": "features", "dtype": str(feat.dtype), "shape": list(feat.shape)}
        ],
    }
    return (
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        + feat.tobytes()
    )


@transformation(
    "analyze.hubert",
    "1",
    inputs={"audio": "audio"},
    outputs={"features": "hubert_features"},
    stage="analysis",
    params={"layer": _HUBERT_LAYER, "sr": _GRID_SR, "hop": _GRID_HOP},
    reproducibility="best_effort",  # accelerator inference is not bit-stable
    model_refs=(_HUBERT_MODEL_REF, f"layer={_HUBERT_LAYER}"),
)
def hubert_producer(inputs: Mapping[str, bytes]) -> Mapping[str, bytes]:
    """Audio bytes → serialized HuBERT feature-matrix bytes.

    Wraps the REAL ``stem_placement.hubert_of`` path unchanged (lazy import —
    stem_placement transitively loads torch/transformers)."""
    from workspaces.alignment_prototype.stem_placement import hubert_of

    content = inputs["audio"]
    with tempfile.NamedTemporaryFile(suffix=_sniff_suffix(content)) as tmp:
        tmp.write(content)
        tmp.flush()
        feat = hubert_of(tmp.name, layer=_HUBERT_LAYER)
    if feat is None:
        raise RuntimeError("hubert_of returned None for present audio bytes")
    return {"features": _serialize_hubert_features(feat, layer=_HUBERT_LAYER)}


# --- §8: register the real trained decoder checkpoint ------------------------
def register_trained_decoder(
    checkpoint: Path,
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: str,
) -> Optional[FittedModel]:
    """Real trained checkpoint → MODEL_CHECKPOINT artifact + TrainingSnapshot
    + training ProcessSpec + FittedModel (grounds laws 13/14).

    Honesty notes, recorded in the artifact metadata too: the registration is
    RETROSPECTIVE (the checkpoint predates this registry, so ``code_commit`` is
    the registering commit, not the training commit), and the snapshot holds
    the GT fixtures the slot-split protocol pools — its audio-derived features
    (chroma/HuBERT/mel FeatureBank) were computed on the fly and are not
    snapshotted.
    """
    if not checkpoint.is_file():
        print(f"  model: checkpoint {checkpoint} not found — skipped")
        return None
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        train_args = {k: v for k, v in payload.get("args", {}).items()}
    except Exception as exc:  # torch unavailable / unreadable payload
        print(f"  model: could not read checkpoint args ({exc!r}) — using file only")
        train_args = {"source_checkpoint": checkpoint.name}

    gt_artifacts = []
    for set_id, gt_path in sorted(_GT_FIXTURES.items()):
        if not gt_path.is_file():
            print(f"  model: GT fixture missing for {set_id} ({gt_path}) — skipped")
            return None
        gt_artifacts.append(
            store.put_file(
                gt_path,
                kind=ArtifactKind.LABEL_MANIFEST,
                media_type="application/yaml",
                metadata={"set_id": set_id, "role": "training_ground_truth"},
                source_uri=str(gt_path.relative_to(_REPO)),
            )
        )
    state = store.put_file(
        checkpoint,
        kind=ArtifactKind.MODEL_CHECKPOINT,
        media_type="application/octet-stream",
        metadata={
            "registered_retrospectively": True,
            "training_features": (
                "FeatureBank chroma/HuBERT/mel computed on the fly; not snapshotted"
            ),
            "payload": "trajectory/train.py torch.save({'model','args'})",
        },
        source_uri=str(checkpoint),
    )

    parameter_set = repo.record_parameter_set(
        make_parameter_set("train.trajectory_decoder", train_args)
    )
    environment = repo.record_environment_spec(
        capture_environment(_REPO / "requirements-audio.txt")
    )
    spec = repo.record_process_spec(
        make_process_spec(
            name="train.trajectory_decoder",
            version="1",
            stage="training",
            code_commit=code_commit,
            parameter_set=parameter_set,
            environment=environment,
            model_refs=(),
            implementation_hash="",
        )
    )
    run = repo.begin_run_from_spec(spec)
    try:
        for i, gt in enumerate(gt_artifacts):
            repo.add_input(run, gt, role="training_member", ordinal=i)
        repo.add_output(run, state, role="model_state")
        for gt in gt_artifacts:
            repo.derive(state, gt, run, relation="trained_on")
        repo.succeed(run)
    except Exception as exc:
        repo.fail(run, {"error": repr(exc)})
        raise

    snapshot = repo.record_training_snapshot(
        make_training_snapshot([a.content_sha256 for a in gt_artifacts])
    )
    model = repo.record_fitted_model(
        make_fitted_model(
            axis=Axis.STRUCTURE,  # trajectory decoder decodes ref trajectories
            process_spec=spec,
            serialized_state_sha256=state.content_sha256,
            training_snapshot=snapshot,
        )
    )
    return model


# --- audio acquisition (pi READ-ONLY, local fallback) ------------------------
def _pull_audio_from_pi(recording_id: str, dest_dir: Path) -> Optional[Path]:
    """scp ONE reference audio file down from pi-storage. Read-only: a SELECT
    over ssh + an scp; nothing on the pi is written or changed."""
    if not _ID_RE.match(recording_id):
        raise ValueError(f"recording_id {recording_id!r} is not a safe identifier")
    query = (
        "SELECT path FROM track_audio WHERE recording_id='" + recording_id + "' "
        "AND stem='regular' ORDER BY is_reference DESC, track_audio_id LIMIT 1"
    )
    try:
        out = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                _PI_HOST,
                f'sqlite3 -readonly /mnt/storage/data/db/music_database.db "{query}"',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  pi-storage unreachable ({exc}) — use --audio-file")
        return None
    if not out:
        print(f"  no regular-stem track_audio row on pi for {recording_id}")
        return None
    remote_path = out.splitlines()[0]
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / Path(remote_path).name
    if not local.is_file():
        try:
            subprocess.run(
                ["scp", "-q", f"{_PI_HOST}:{remote_path}", str(local)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"  scp failed ({exc}) — use --audio-file")
            return None
    print(f"  pulled (read-only) pi-storage:{remote_path}")
    return local


def _print_lineage(art: Artifact, repo: ProvenanceRepository) -> None:
    conn = repo._conn  # sibling module; read-only introspection
    for r in conn.execute(
        "SELECT d.parent_sha256, d.relation, r.process, r.process_spec_id "
        "FROM derivation d JOIN run r ON r.run_id=d.run_id WHERE d.child_sha256=?",
        (art.content_sha256,),
    ).fetchall():
        spec = str(r["process_spec_id"] or "")[:12]
        print(
            f"      derived from {r['parent_sha256'][:12]} "
            f"via {r['relation']} (process={r['process']}, spec={spec})"
        )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, help="recording_id to process")
    ap.add_argument(
        "--set-id",
        default=None,
        help="set whose MERT cache holds the recording (omit to skip MERT migration)",
    )
    ap.add_argument(
        "--audio-file", default=None, help="local audio file (fallback when pi is down)"
    )
    ap.add_argument(
        "--db-root", default=None, help=f"provenance root (default {_OUT_ROOT})"
    )
    ap.add_argument(
        "--model-checkpoint",
        default=str(_DEFAULT_CHECKPOINT),
        help="trained checkpoint to register as a FittedModel",
    )
    ap.add_argument(
        "--skip-model", action="store_true", help="skip FittedModel registration"
    )
    args = ap.parse_args()

    from core.provenance import connect

    db_root = Path(args.db_root or _OUT_ROOT)
    db_root.mkdir(parents=True, exist_ok=True)
    conn = connect(db_root)
    store = ArtifactStore(db_root, conn)
    repo = ProvenanceRepository(conn)
    try:
        code_commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO)
            .decode()
            .strip()
        )
    except Exception:
        code_commit = "unknown"

    # 1. one real track's audio — pi (read-only) or local fallback
    if args.audio_file:
        audio_path = Path(args.audio_file)
        source_system = "local_file"
        source_uri = str(audio_path)
    else:
        pulled = _pull_audio_from_pi(args.track, db_root / ".pulled")
        if pulled is None:
            print("no audio available; pass --audio-file <path>")
            return 2
        audio_path = pulled
        source_system = "pi-storage"
        source_uri = f"pi-storage:/mnt/storage/objects/{args.track}/{audio_path.name}"
    audio = store.put_file(
        audio_path,
        kind=ArtifactKind.AUDIO,
        media_type=_MEDIA_TYPES.get(
            audio_path.suffix.lower(), "application/octet-stream"
        ),
        metadata={"recording_id": args.track},
        source_uri=source_uri,
    )
    # source_system is not a put_file kwarg; keep provenance in source_uri.
    print(f"producers — track {args.track} (source: {source_system})")
    print(f"  audio artifact       : {audio.content_sha256}")

    # 2. fingerprint through the registry (fresh compute, full plumbing)
    outputs = REGISTRY.run(
        "analyze.landmark_fp",
        {"audio": audio},
        store=store,
        repo=repo,
        code_commit=code_commit,
    )
    fp_art = outputs["features"]
    print(f"  landmark_fingerprint : {fp_art.content_sha256}")
    _print_lineage(fp_art, repo)

    # 3. migrate the cached MERT stack (existing bytes -> canonical form)
    mert_art = None
    if args.set_id:
        mert_art = migrate_cached_mert(
            args.track,
            args.set_id,
            audio,
            store=store,
            repo=repo,
            code_commit=code_commit,
        )
    else:
        print("  mert: no --set-id given — migration skipped")
    if mert_art is not None:
        print(f"  mert_features        : {mert_art.content_sha256}")
        _print_lineage(mert_art, repo)

    # 4. register the real trained decoder (grounds laws 13/14)
    if not args.skip_model:
        model = register_trained_decoder(
            Path(args.model_checkpoint), store=store, repo=repo, code_commit=code_commit
        )
        if model is not None:
            print(
                f"  fitted_model         : {model.fitted_model_id[:16]} "
                f"(axis={model.axis.value}, state={model.serialized_state_sha256[:12]}, "
                f"snapshot={model.training_snapshot_id[:12]})"
            )

    print(f"  provenance DB        : {db_root}")
    print(f"  check laws           : python -m core.provenance.laws --db {db_root}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

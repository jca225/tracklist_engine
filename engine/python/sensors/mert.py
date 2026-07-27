"""analyze.mert — MERT (or deterministic stub) feature sensor.

Writes:
  - ``feature_path``: raw embedding bytes (float32 little-endian for stub)
  - ``mert_json`` sidecar for ``dj_migrate feature-register``

Default is **stub** mode (no torch): a content-derived pseudo-embedding so CI
and agents can exercise put_audio → FeatureBlob without GPUs. Pass
``--real`` to attempt the tracklist_engine MERT adapter when available.

LF emissions should reference the registered **feature ArtifactId**, not paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

from sensors.schemas import validate_payload

PROCESS_NAME = "analyze.mert"
PROCESS_VERSION = "0.1.0"
STUB_MODEL_ID = "stub-mert/deterministic-v1"
STUB_MODEL_REVISION = "0"
STUB_DIM = 64
REAL_MODEL_ID = "m-a-p/MERT-v1-330M"


def stub_embedding(audio_bytes: bytes, dim: int = STUB_DIM) -> bytes:
    """Deterministic float32 vector derived from audio sha256 (not musical)."""
    digest = hashlib.sha256(audio_bytes).digest()
    out = bytearray()
    seed = digest
    for i in range(dim):
        if i % 8 == 0 and i > 0:
            seed = hashlib.sha256(seed + digest).digest()
        b = seed[i % len(seed)]
        val = (b / 127.5) - 1.0
        out.extend(struct.pack("<f", val))
    return bytes(out)


def run_mert(
    audio_path: Path,
    out_dir: Path,
    *,
    real: bool = False,
    audio_artifact_id: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_bytes = audio_path.read_bytes()
    audio_sha = hashlib.sha256(audio_bytes).hexdigest()

    stub = True
    model_id = STUB_MODEL_ID
    model_revision = STUB_MODEL_REVISION
    dims = [STUB_DIM]
    feature_bytes: bytes
    error: str | None = None

    if real:
        try:
            feature_bytes, dims, model_id, model_revision = _try_real_mert(audio_path)
            stub = False
        except Exception as exc:  # noqa: BLE001 — surface as sensor error payload
            error = f"real mert failed, falling back to stub: {exc}"
            feature_bytes = stub_embedding(audio_bytes)
            stub = True
            model_id = STUB_MODEL_ID
            model_revision = STUB_MODEL_REVISION
            dims = [STUB_DIM]
    else:
        feature_bytes = stub_embedding(audio_bytes)

    feature_path = out_dir / f"{audio_sha[:16]}_mert.f32"
    feature_path.write_bytes(feature_bytes)
    feature_sha = hashlib.sha256(feature_bytes).hexdigest()

    result: dict[str, Any] = {
        "ok": True,
        "feature_kind": "mert_embedding",
        "feature_path": str(feature_path.resolve()),
        "media_type": "application/octet-stream",
        "model_id": model_id,
        "model_revision": model_revision,
        "dims": dims,
        "stub": stub,
        "content_sha256": feature_sha,
        "error": error,
        "process_name": PROCESS_NAME,
        "process_version": PROCESS_VERSION,
        "metadata": {
            "audio_path": str(audio_path.resolve()),
            "audio_sha256": audio_sha,
            "audio_artifact_id": audio_artifact_id,
            "coordinate_system": "stub_sha256_expanded" if stub else "mert_hidden_mean",
        },
    }
    # Boundary schema for the analyze family (subset fields).
    validate_payload(
        "analyze_response",
        {
            "ok": True,
            "feature_kind": "mert",
            "process_name": PROCESS_NAME,
            "process_version": PROCESS_VERSION,
            "content_sha256": feature_sha,
            "payload": {"dims": dims, "stub": stub},
            "error": error,
            "abstained": False,
        },
    )
    sidecar = out_dir / f"{audio_sha[:16]}_mert.json"
    sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["mert_json"] = str(sidecar.resolve())
    return result


def _try_real_mert(audio_path: Path) -> tuple[bytes, list[int], str, str]:
    """Optional import of tracklist_engine MERT — raises if unavailable."""
    _ = audio_path  # reserved for real embed_track path
    te = Path.home() / "Desktop" / "tracklist_engine"
    if not te.is_dir():
        raise RuntimeError(f"tracklist_engine not found at {te}")
    raise RuntimeError(
        "real MERT measure export not wired in this slice — use stub or extend "
        "sensors.mert to call analysis.adapters.mert_adapter and serialize layers"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True, help="path to audio bytes on disk")
    p.add_argument("--out-dir", type=Path, default=Path("staging/mert"))
    p.add_argument(
        "--audio-artifact-id",
        default=None,
        help="kernel ArtifactId of the parent audio (LF emissions should use this)",
    )
    p.add_argument(
        "--real",
        action="store_true",
        help="attempt real MERT via tracklist_engine (falls back to stub on failure)",
    )
    args = p.parse_args(argv)
    if not args.audio.is_file():
        print(f"audio not found: {args.audio}", file=sys.stderr)
        return 2
    result = run_mert(
        args.audio,
        args.out_dir,
        real=args.real,
        audio_artifact_id=args.audio_artifact_id,
    )
    print(json.dumps({"mert_json": result["mert_json"], "feature_path": result["feature_path"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Typed request/response models and stub sensors.

Stubs validate I/O against ``schemas/`` but do not run MIR or download.
Real adapters should:

1. Accept an audio ``ArtifactId`` / content URI from the kernel.
2. Write feature bytes back through the artifact store (sha256 identity).
3. Record model id + revision in metadata (see ``docs/storage.md``).
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from sensors.schemas import validate_payload


class SensorAdapter(Protocol):
    """Minimal process identity every sensor exposes to provenance."""

    process_name: str
    process_version: str


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_uri: str
    feature_kind: str
    process_name: str
    process_version: str
    seed: int | None = None


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    feature_kind: str
    process_name: str
    process_version: str
    content_sha256: str | None = None
    payload: dict[str, Any] | None = None
    error: str | None = None
    abstained: bool = False


class DownloadCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    rank: int | None = None
    source: str | None = None


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[DownloadCandidate] = Field(min_length=1)
    process_name: str
    process_version: str
    expected_duration_s: float | None = None


class DownloadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    process_name: str
    process_version: str
    path: str | None = None
    content_sha256: str | None = None
    source_url: str | None = None
    error: str | None = None
    attempts: int = 0


class AbletonParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    als_uri: str
    permitted_audio_uris: list[str] = Field(default_factory=list)
    process_name: str
    process_version: str


class AbletonClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_index: int
    audio_uri: str | None = None
    content_sha256: str | None = None
    mix_start_s: float | None = None
    mix_end_s: float | None = None
    unique_identity: bool = False
    abstained: bool = False
    abstain_reason: str | None = None


class AbletonParseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    process_name: str
    process_version: str
    clips: list[AbletonClip]
    error: str | None = None


class AnalyzeStub:
    """Stub analyzer — validates I/O; real MIR wraps monolith later."""

    process_name = "sensors.analyze.stub"
    process_version = "0.1.0"

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        validate_payload("analyze_request", request.model_dump())
        response = AnalyzeResponse(
            ok=True,
            feature_kind=request.feature_kind,
            process_name=self.process_name,
            process_version=self.process_version,
            payload={"stub": True, "artifact_uri": request.artifact_uri},
            abstained=False,
        )
        validate_payload("analyze_response", response.model_dump())
        return response


class DownloadStub:
    process_name = "sensors.download.stub"
    process_version = "0.1.0"

    def fetch(self, request: DownloadRequest) -> DownloadResponse:
        validate_payload("download_request", request.model_dump(exclude_none=True))
        response = DownloadResponse(
            ok=False,
            process_name=self.process_name,
            process_version=self.process_version,
            error="stub: no download performed",
            attempts=0,
        )
        validate_payload("download_response", response.model_dump())
        return response


class AbletonStub:
    process_name = "sensors.ableton.stub"
    process_version = "0.1.0"

    def parse(self, request: AbletonParseRequest) -> AbletonParseResponse:
        validate_payload("ableton_parse_request", request.model_dump())
        response = AbletonParseResponse(
            ok=True,
            process_name=self.process_name,
            process_version=self.process_version,
            clips=[
                AbletonClip(
                    clip_index=0,
                    abstained=True,
                    abstain_reason="stub: no .als bytes parsed",
                )
            ],
        )
        validate_payload("ableton_parse_response", response.model_dump())
        return response

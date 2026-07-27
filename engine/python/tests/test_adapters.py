from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sensors.adapters import (
    AbletonParseRequest,
    AbletonStub,
    AnalyzeRequest,
    AnalyzeStub,
    DownloadCandidate,
    DownloadRequest,
    DownloadStub,
)
from sensors.schemas import SchemaValidationError, validate_payload


def test_analyze_stub_roundtrip() -> None:
    stub = AnalyzeStub()
    req = AnalyzeRequest(
        artifact_uri="file:///tmp/x.wav",
        feature_kind="beats",
        process_name=stub.process_name,
        process_version=stub.process_version,
    )
    resp = stub.analyze(req)
    assert resp.ok is True
    assert resp.payload is not None


def test_download_stub_validates() -> None:
    stub = DownloadStub()
    req = DownloadRequest(
        candidates=[DownloadCandidate(url="https://example.com/a")],
        process_name=stub.process_name,
        process_version=stub.process_version,
    )
    resp = stub.fetch(req)
    assert resp.ok is False
    assert resp.error is not None


def test_ableton_stub_abstains() -> None:
    stub = AbletonStub()
    req = AbletonParseRequest(
        als_uri="file:///tmp/x.als",
        process_name=stub.process_name,
        process_version=stub.process_version,
    )
    resp = stub.parse(req)
    assert resp.clips[0].abstained is True


def test_schema_rejects_bad_feature_kind() -> None:
    with pytest.raises(SchemaValidationError):
        validate_payload(
            "analyze_request",
            {
                "artifact_uri": "x",
                "feature_kind": "not-a-kind",
                "process_name": "p",
                "process_version": "1",
            },
        )


@given(st.sampled_from(["beats", "cues", "mert", "fingerprint", "loudness", "diagnostics"]))
def test_analyze_request_accepts_known_kinds(kind: str) -> None:
    validate_payload(
        "analyze_request",
        {
            "artifact_uri": "file:///x",
            "feature_kind": kind,
            "process_name": "t",
            "process_version": "1",
        },
    )

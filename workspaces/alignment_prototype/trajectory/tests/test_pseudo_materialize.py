"""Audited pseudo-GT materializer (trm_flywheel_design.md §7.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from workspaces.alignment_prototype.trajectory.pseudo_materialize import (
    materialize_pseudo_gt,
)


def _span(
    slot: str,
    recording_id: str,
    *,
    driver_mode: str = "auto_commit",
    set_start_s: float = 10.0,
    set_end_s: float = 40.0,
    ref_start_s: float = 0.0,
    ref_end_s: float = 30.0,
    track_id: str | None = None,
    pseudo_gate_rejection: str | None = None,
) -> dict:
    out: dict = {
        "slot_label": slot,
        "recording_id": recording_id,
        "claimed_stem": "regular",
        "set_start_s": set_start_s,
        "set_end_s": set_end_s,
        "ref_start_s": ref_start_s,
        "ref_end_s": ref_end_s,
        "ref_stretch": 1.0,
        "ref_segments": [
            {
                "mix_start_s": set_start_s,
                "ref_start_s": ref_start_s,
                "ref_end_s": ref_end_s,
            }
        ],
        "driver_mode": driver_mode,
        "agentic_quality": 0.8,
    }
    if track_id is not None:
        out["track_id"] = track_id
    if pseudo_gate_rejection is not None:
        out["pseudo_gate_rejection"] = pseudo_gate_rejection
    return out


def _write_timeline(tmp_path: Path, spans: list[dict]) -> Path:
    path = tmp_path / "agentic_timeline.json"
    payload = {
        "set_id": "w1mgcjt",
        "spans": spans,
        "pseudo_safety": {
            "combine": True,
            "fiber_gate": True,
            "auto": 0.75,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_manifest(tmp_path: Path, tracks: list[dict]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "set_id": "w1mgcjt",
                "tracks": tracks,
            },
            indent=2,
        )
    )
    return path


def test_materialize_counts_each_rejection_and_resolves_track_id(tmp_path):
    timeline = _write_timeline(
        tmp_path,
        spans=[
            _span("1", "rec-a", driver_mode="auto_commit"),
            _span("2", "rec-b", driver_mode="review"),
            _span(
                "3",
                "rec-c",
                driver_mode="review",
                pseudo_gate_rejection="g2_independence",
            ),
            _span("4", "rec-d", driver_mode="auto_commit", set_end_s=0.0),
            _span("5", "missing", driver_mode="auto_commit"),
        ],
    )
    manifest = _write_manifest(
        tmp_path,
        tracks=[
            {"slot_label": "1", "recording_id": "rec-a", "track_id": "track-a"},
            {"slot_label": "4", "recording_id": "rec-d", "track_id": "track-d"},
        ],
    )

    report = materialize_pseudo_gt(timeline, manifest, tmp_path / "pseudo.yaml")

    assert report.accepted == 1
    assert report.total == 5
    assert report.dropped == {
        "g0_mode": 1,
        "g2_independence": 1,
        "g4_structure": 1,
        "audio_identity": 1,
    }
    doc = yaml.safe_load(report.output.read_text())
    assert doc["set_id"] == "w1mgcjt"
    assert len(doc["tracks"]) == 1
    assert doc["tracks"][0]["track_id"] == "track-a"
    assert doc["tracks"][0]["pseudo_label"] is True


def test_materialize_is_deterministic_and_atomic(tmp_path):
    timeline = _write_timeline(
        tmp_path,
        spans=[_span("1", "rec-a", driver_mode="auto_commit", track_id="track-a")],
    )
    manifest = _write_manifest(
        tmp_path,
        tracks=[{"slot_label": "1", "recording_id": "rec-a", "track_id": "track-a"}],
    )
    out = tmp_path / "pseudo.yaml"
    a = materialize_pseudo_gt(timeline, manifest, out)
    b = materialize_pseudo_gt(timeline, manifest, out)
    assert a.output.read_bytes() == b.output.read_bytes()
    doc = yaml.safe_load(a.output.read_text())
    assert "sha256" in doc["provenance"]
    assert doc["provenance"]["pseudo_safety"]["combine"] is True
    assert doc["provenance"]["pseudo_safety"]["fiber_gate"] is True
    assert all(row["track_id"] for row in doc["tracks"])


def test_materialize_exit_code_starvation_via_cli(tmp_path):
    from workspaces.alignment_prototype.trajectory.pseudo_materialize import main

    timeline = _write_timeline(
        tmp_path, spans=[_span("1", "rec-a", driver_mode="review")]
    )
    manifest = _write_manifest(
        tmp_path,
        tracks=[{"slot_label": "1", "recording_id": "rec-a", "track_id": "track-a"}],
    )
    code = main(
        [
            "--timeline",
            str(timeline),
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "empty.yaml"),
        ]
    )
    assert code == 2


def test_materialize_yaml_loads_into_dataset_when_audio_stubbed(tmp_path, monkeypatch):
    from workspaces.alignment_prototype.trajectory import data as traj_data
    from workspaces.alignment_prototype.trajectory.features import SpanAudio

    timeline = _write_timeline(
        tmp_path,
        spans=[_span("1", "rec-a", driver_mode="auto_commit")],
    )
    manifest = _write_manifest(
        tmp_path,
        tracks=[{"slot_label": "1", "recording_id": "rec-a", "track_id": "track-a"}],
    )
    report = materialize_pseudo_gt(timeline, manifest, tmp_path / "pseudo.yaml")

    aligning = tmp_path / "aligning"
    aligning.mkdir()
    (aligning / "manifest.json").write_text(
        json.dumps(
            {
                "set_id": "w1mgcjt",
                "mix_local_path": str(tmp_path / "mix.wav"),
                "tracks": [
                    {
                        "track_id": "track-a",
                        "recording_id": "rec-a",
                        "slot_label": "1",
                    }
                ],
            }
        )
    )

    stub = SpanAudio(
        mix_path=tmp_path / "mix.wav",
        ref_path=tmp_path / "ref.wav",
        match_feature="chroma",
        stem_routed=False,
    )
    monkeypatch.setattr(traj_data, "find_aligning_dir", lambda _sid: aligning)
    monkeypatch.setattr(traj_data, "find_mix_stems", lambda _d: {})
    monkeypatch.setattr(
        traj_data, "resolve_span_audio", lambda *_a, **_k: stub
    )

    ds = traj_data.TrajectorySpanDataset(
        [("w1mgcjt", report.output)],
        aligning_dirs={"w1mgcjt": aligning},
    )
    assert len(ds) == 1
    assert ds.specs[0].row["pseudo_label"] is True
    assert ds.specs[0].row["track_id"] == "track-a"

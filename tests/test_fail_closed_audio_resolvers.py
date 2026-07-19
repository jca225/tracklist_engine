"""Regression tests for fail-closed alignment audio resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import workspaces.alignment_prototype.refine_ref_offsets as refine_ref_offsets
from workspaces.alignment_prototype.infer_fused import _resolve_ref
from workspaces.alignment_prototype.joint_ref_decode import _ref_audio_for
from workspaces.alignment_prototype.refine_ref_offsets import ref_audio_for
from workspaces.alignment_prototype.stem_resolve import resolve_stem


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_resolve_stem_preserves_valid_manifest_path(tmp_path: Path) -> None:
    manifest_stem = _touch(tmp_path / "manifest" / "vocals.flac")
    track = {"track_audio_id": 42, "stems": {"vocals": str(manifest_stem)}}

    assert resolve_stem(tmp_path, "001w1", track, "vocals") == manifest_stem


def test_resolve_stem_accepts_one_fallback_candidate(tmp_path: Path) -> None:
    fallback = _touch(tmp_path / "stems" / "001w1__Artist - Title" / "vocals.flac")
    track = {"track_audio_id": 42, "stems": {}}

    assert resolve_stem(tmp_path, "001w1", track, "vocals") == fallback


def test_resolve_stem_abstains_on_ambiguous_fallback(tmp_path: Path) -> None:
    _touch(tmp_path / "stems" / "001w1__Artist - Title" / "vocals.flac")
    _touch(tmp_path / "stems" / "001w1__Artist - Title [128bpm 1B]" / "vocals.flac")
    track = {"track_audio_id": 42, "stems": {}}

    with pytest.warns(
        RuntimeWarning, match=r"track_audio_id=42.*slot=001w1.*stem=vocals"
    ):
        assert resolve_stem(tmp_path, "001w1", track, "vocals") is None


def test_stem_routed_callers_propagate_abstention(tmp_path: Path) -> None:
    local_path = _touch(tmp_path / "tracks" / "001w1__Artist - Title.m4a")
    _touch(tmp_path / "stems" / "001w1__Artist - Title" / "vocals.flac")
    _touch(tmp_path / "stems" / "001w1__Artist - Other Copy" / "vocals.flac")
    track = {
        "track_audio_id": 42,
        "local_path": str(local_path),
        "stems": {},
    }
    span = {"claimed_stem": "acappella", "slot_label": "001w1"}

    with pytest.warns(RuntimeWarning, match=r"track_audio_id=42"):
        assert ref_audio_for(span, track, tmp_path) is None
    with pytest.warns(RuntimeWarning, match=r"track_audio_id=42"):
        assert _ref_audio_for(span, track, tmp_path) is None


def test_stem_routed_callers_keep_regular_local_path(tmp_path: Path) -> None:
    local_path = _touch(tmp_path / "tracks" / "001__Artist - Title.m4a")
    track = {"track_audio_id": 42, "local_path": str(local_path), "stems": {}}
    span = {"claimed_stem": "regular", "slot_label": "001"}

    assert ref_audio_for(span, track, tmp_path) == local_path
    assert _ref_audio_for(span, track, tmp_path) == str(local_path)


def test_refine_main_supplies_set_dir_to_audio_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    set_dir = tmp_path / "set"
    _touch(set_dir / "mix.m4a")
    (set_dir / "manifest.json").write_text(
        '{"tracks":[{"track_id":"recording-1","local_path":"missing"}]}'
    )
    (out_dir / "set-1_predicted_timeline.json").write_text(
        '{"spans":[{"recording_id":"recording-1"}]}'
    )

    import librosa

    monkeypatch.setattr(refine_ref_offsets, "OUT_DIR", out_dir)
    monkeypatch.setattr(
        refine_ref_offsets, "find_aligning_dir", lambda _set_id: set_dir
    )
    monkeypatch.setattr(
        librosa, "load", lambda *_args, **_kwargs: (np.zeros(1024), 16000)
    )
    monkeypatch.setattr(refine_ref_offsets, "chroma", lambda _audio: np.zeros((12, 32)))

    class ResolverReached(Exception):
        pass

    def require_set_dir(_span: dict, _track: dict, actual_set_dir: Path | None = None):
        assert actual_set_dir == set_dir
        raise ResolverReached

    monkeypatch.setattr(refine_ref_offsets, "ref_audio_for", require_set_dir)

    with pytest.raises(ResolverReached):
        refine_ref_offsets.main(["--set-id", "set-1", "--no-fingerprint"])


def test_resolve_ref_preserves_valid_manifest_path(tmp_path: Path) -> None:
    manifest_ref = _touch(tmp_path / "manifest" / "001__Artist - Title.m4a")
    track = {"track_audio_id": 42, "local_path": str(manifest_ref)}

    assert _resolve_ref(track, tmp_path) == manifest_ref


def test_resolve_ref_accepts_one_fallback_candidate(tmp_path: Path) -> None:
    fallback = _touch(tmp_path / "tracks" / "001__Artist - Title.m4a")
    track = {
        "track_audio_id": 42,
        "local_path": str(tmp_path / "missing" / "001__Artist - Title.m4a"),
    }

    assert _resolve_ref(track, tmp_path) == fallback


def test_resolve_ref_abstains_on_ambiguous_fallback(tmp_path: Path) -> None:
    _touch(tmp_path / "tracks" / "001__Artist - Title.m4a")
    _touch(tmp_path / "tracks" / "001__Artist - Title [128bpm 1B].m4a")
    track = {
        "track_audio_id": 42,
        "local_path": str(tmp_path / "missing" / "001__Artist - Title.m4a"),
    }

    with pytest.warns(RuntimeWarning, match=r"track_audio_id=42.*slot=001"):
        assert _resolve_ref(track, tmp_path) is None

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from eda.alignment.spectrogram_review.source_audio import (
    _resolve_from_filesystem,
    _slot_aliases,
    preflight_aligning_audio,
    resolve_source_audio,
)


def test_slot_aliases_pad_and_unpad():
    assert "037" in _slot_aliases("37")
    assert "37" in _slot_aliases("037")
    assert "010w2" in _slot_aliases("10w2")


def test_filesystem_resolve_picks_named_folder_when_slot_ambiguous(tmp_path: Path):
    """Slot 037 with two stem folders — name hint must pick Heiress, not Blackout."""
    set_dir = tmp_path
    for folder, tone in (
        ("stems/037__Breathe Carolina - Blackout", 220.0),
        ("stems/037__Dune - Heiress Of Valentina (Alesso Remix)", 440.0),
    ):
        d = set_dir / folder
        d.mkdir(parents=True)
        t = np.linspace(0, 0.2, 4410, endpoint=False)
        y = (0.2 * np.sin(2 * np.pi * tone * t)).astype(np.float32)
        sf.write(str(d / "instrumental.flac"), y, 22050)

    hit = _resolve_from_filesystem(
        set_dir,
        "37",
        gt_stem="instrumental",
        name_hint="Dúné - Heiress Of Valentina (Alesso Remix)",
    )
    assert hit is not None
    assert "Heiress" in hit.as_posix()
    assert hit.name == "instrumental.flac"


def test_filesystem_resolve_fails_closed_when_ambiguous_and_no_name(tmp_path: Path):
    set_dir = tmp_path
    for folder in (
        "stems/037__Breathe Carolina - Blackout",
        "stems/037__Dune - Heiress Of Valentina",
    ):
        d = set_dir / folder
        d.mkdir(parents=True)
        sf.write(
            str(d / "instrumental.flac"),
            np.zeros(1000, dtype=np.float32),
            22050,
        )
    assert (
        _resolve_from_filesystem(set_dir, "037", gt_stem="instrumental", name_hint="")
        is None
    )


def test_unresolved_manifest_stub_does_not_slot_fallback(tmp_path: Path, monkeypatch):
    """Known missing recording_id must not pick a colliding slot folder."""
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    t = np.linspace(0, 0.2, 4410, endpoint=False)
    y = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(str(set_dir / "tracks" / "032__AFROJACK - Ten Feet Tall.wav"), y, 22050)

    rec_id = "tlp2853023"
    tracks = {
        rec_id: {
            "track_id": rec_id,
            "recording_id": rec_id,
            "slot_label": "032",
            "local_path": str(set_dir / "tracks" / "032__Lux Holm - Omega.wav"),
            "stems": {},
        }
    }
    monkeypatch.setattr(
        "eda.alignment.spectrogram_review.source_audio.find_aligning_dir",
        lambda _sid: set_dir,
    )

    hit = resolve_source_audio(
        "1fsnxchk",
        rec_id,
        gt_stem="instrumental",
        tracks=tracks,
        slot="032",
        name="Lux Holm - Omega",
    )
    assert hit is None


def test_preflight_skips_unalignable_and_counts_missing(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    sf.write(
        str(set_dir / "tracks" / "010__Good Track.wav"),
        np.zeros(1000, dtype=np.float32),
        22050,
    )
    good_id = "rec_good"
    bad_id = "rec_bad"
    tracks = {
        good_id: {
            "track_id": good_id,
            "local_path": str(set_dir / "tracks" / "010__Good Track.wav"),
            "stems": {},
        },
        bad_id: {
            "track_id": bad_id,
            "local_path": str(set_dir / "tracks" / "011__Missing.wav"),
            "stems": {},
        },
    }
    monkeypatch.setattr(
        "eda.alignment.spectrogram_review.source_audio.find_aligning_dir",
        lambda _sid: set_dir,
    )
    monkeypatch.setattr(
        "eda.alignment.spectrogram_review.source_audio.load_manifest_tracks",
        lambda _sid: tracks,
    )

    gt = [
        {"track_id": good_id, "slot_label": "010", "claimed_stem": "regular"},
        {"track_id": bad_id, "slot_label": "011", "claimed_stem": "regular"},
        {"track_id": "rec_skip", "slot_label": "012", "skip_training": True},
        {"track_id": "rec_unal", "slot_label": "013", "unalignable": True},
    ]
    missing = preflight_aligning_audio("1fsnxchk", gt)
    assert missing == [bad_id]

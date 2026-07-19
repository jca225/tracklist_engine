from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from eda.alignment.spectrogram_review.source_audio import (
    _resolve_from_filesystem,
    _slot_aliases,
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

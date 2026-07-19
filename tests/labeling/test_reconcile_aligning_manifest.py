from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from labeling.reconcile_aligning_manifest import reconcile_manifest


def _touch_audio(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(1000, dtype=np.float32), 22050)


def test_wires_missing_row_from_disk(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    (set_dir / "stems").mkdir()
    _touch_audio(set_dir / "tracks" / "037__Dune - Heiress.m4a")
    _touch_audio(set_dir / "stems" / "037__Dune - Heiress" / "instrumental.flac")
    man = {
        "set_id": "1fsnxchk",
        "title": "t",
        "mix_local_path": str(set_dir / "mix.m4a"),
        "tracks": [],
    }
    _touch_audio(set_dir / "mix.m4a")
    (set_dir / "manifest.json").write_text(json.dumps(man))

    def fake_slots(_sql: str):
        return [
            {
                "slot_label": "037",
                "recording_id": "94tc2y5",
                "claimed_stem": "instrumental",
                "claimed_variant": "regular",
                "name": "Dune - Heiress",
            }
        ]

    monkeypatch.setattr("labeling.reconcile_aligning_manifest.ssh_sqlite", fake_slots)
    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.fetch_tracks",
        lambda _sid: [],
    )

    report = reconcile_manifest(set_dir, dry_run=False)
    assert "037" in report.wired or "037" in report.added
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "94tc2y5")
    assert row["stem"] == "instrumental"
    assert Path(row["stems"]["instrumental"]).is_file()


def test_unresolved_slot_stays_pathless(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    # colliding orphan — must NOT be wired to Lux
    _touch_audio(set_dir / "tracks" / "032__AFROJACK - Ten Feet Tall.wav")
    (set_dir / "manifest.json").write_text(
        json.dumps({"set_id": "1fsnxchk", "tracks": []})
    )

    def fake_slots(_sql: str):
        return [
            {
                "slot_label": "032",
                "recording_id": "tlp2853023",
                "claimed_stem": "regular",
                "claimed_variant": "regular",
                "name": "Lux Holm - Omega",
            }
        ]

    monkeypatch.setattr("labeling.reconcile_aligning_manifest.ssh_sqlite", fake_slots)
    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.fetch_tracks",
        lambda _sid: [],
    )
    # name tokens for Lux must not match AFROJACK
    report = reconcile_manifest(set_dir, dry_run=False)
    assert "032" in report.unresolved
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "tlp2853023")
    assert row.get("local_path") in (None, "")

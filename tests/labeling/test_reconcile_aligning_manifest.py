from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from labeling.reconcile_aligning_manifest import reconcile_manifest


def _patch_pi(monkeypatch, slots: list[dict]) -> None:
    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.ssh_sqlite",
        lambda _sql: slots,
    )
    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.fetch_tracks",
        lambda _sid: [],
    )
    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.evaluate_set_inventory",
        lambda _sid, _ssh: [],
    )


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

    _patch_pi(monkeypatch, fake_slots(""))

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

    _patch_pi(monkeypatch, fake_slots(""))
    # name tokens for Lux must not match AFROJACK
    report = reconcile_manifest(set_dir, dry_run=False)
    assert "032" in report.unresolved
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "tlp2853023")
    assert row.get("local_path") in (None, "")


def test_clears_poisoned_existing_local_path(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    bad = set_dir / "tracks" / "032__AFROJACK - Ten Feet Tall.wav"
    _touch_audio(bad)
    (set_dir / "manifest.json").write_text(
        json.dumps(
            {
                "set_id": "1fsnxchk",
                "tracks": [
                    {
                        "track_id": "tlp2853023",
                        "slot_label": "032",
                        "label": "032",
                        "local_path": str(bad),
                        "stem": "regular",
                        "stems": {},
                    }
                ],
            }
        )
    )
    _patch_pi(
        monkeypatch,
        [
            {
                "slot_label": "032",
                "recording_id": "tlp2853023",
                "claimed_stem": "regular",
                "claimed_variant": "regular",
                "name": "Lux Holm - Omega",
            }
        ],
    )

    report = reconcile_manifest(set_dir, dry_run=False)
    assert "032" in report.unresolved
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "tlp2853023")
    assert row.get("local_path") in (None, "")
    assert "AFROJACK" not in (row.get("local_path") or "")


def test_lux_omega_proxy_uses_mix_instrumental(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    _touch_audio(set_dir / "mix_instrumental.flac")
    _touch_audio(set_dir / "tracks" / "032__AFROJACK - Ten Feet Tall.wav")
    (set_dir / "manifest.json").write_text(
        json.dumps({"set_id": "1fsnxchk", "tracks": []})
    )

    _patch_pi(
        monkeypatch,
        [
            {
                "slot_label": "032",
                "recording_id": "tlp2853023",
                "claimed_stem": "regular",
                "claimed_variant": "regular",
                "name": "Lux Holm - Omega",
            }
        ],
    )
    from labeling.reconcile_aligning_manifest import (
        PROXY_SLOT_AUDIO,
        reconcile_manifest,
    )

    assert PROXY_SLOT_AUDIO[("1fsnxchk", "032")] == "mix_instrumental.flac"
    report = reconcile_manifest(set_dir, dry_run=False)
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "tlp2853023")
    assert row["local_path"].endswith("mix_instrumental.flac")
    assert row["stem"] == "instrumental"
    assert row["satisfaction"] == "fallback"
    assert "proxy:mix_instrumental" in (row.get("gap") or "")
    assert "032" not in report.unresolved


def test_rewires_poisoned_local_path_to_proxy(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    bad = set_dir / "tracks" / "032__AFROJACK - Ten Feet Tall.wav"
    _touch_audio(bad)
    proxy = set_dir / "mix_instrumental.flac"
    _touch_audio(proxy)
    (set_dir / "manifest.json").write_text(
        json.dumps(
            {
                "set_id": "1fsnxchk",
                "tracks": [
                    {
                        "track_id": "tlp2853023",
                        "slot_label": "032",
                        "label": "032",
                        "local_path": str(bad),
                        "stem": "regular",
                        "stems": {},
                    }
                ],
            }
        )
    )
    _patch_pi(
        monkeypatch,
        [
            {
                "slot_label": "032",
                "recording_id": "tlp2853023",
                "claimed_stem": "instrumental",
                "claimed_variant": "regular",
                "name": "Lux Holm - Omega",
            }
        ],
    )

    report = reconcile_manifest(set_dir, dry_run=False)
    assert "032" in report.wired
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "tlp2853023")
    assert row["local_path"].endswith("mix_instrumental.flac")
    assert "AFROJACK" not in row["local_path"]


def test_inventory_enrichment_warning_on_ssh_failure(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    (set_dir / "manifest.json").write_text(
        json.dumps({"set_id": "1fsnxchk", "tracks": []})
    )
    _patch_pi(
        monkeypatch,
        [
            {
                "slot_label": "032",
                "recording_id": "tlp2853023",
                "claimed_stem": "regular",
                "claimed_variant": "regular",
                "name": "Lux Holm - Omega",
            }
        ],
    )

    def _ssh_down(_sid, _ssh):
        raise OSError("ssh down")

    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.evaluate_set_inventory",
        _ssh_down,
    )

    report = reconcile_manifest(set_dir, dry_run=True)
    assert any("inventory enrichment skipped" in w for w in report.warnings)

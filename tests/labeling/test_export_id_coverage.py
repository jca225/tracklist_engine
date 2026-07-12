"""GT export must refuse to write a fixture that resolved almost no recording_ids.

``resolve_identity`` fills ``track_id`` ONLY on an exact manifest match. A stale
``.als`` whose clip file-refs no longer match the manifest (e.g. not relinked
after a slot/tag rename) resolves ~0 ids; the exporter would otherwise write a
fixture that joins to nothing downstream and would corrupt canonical
``set_ground_truth`` on write-back. Regression for the BB12 re-export
(2026-07-12) which resolved 1/163 track_ids and was written silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from labeling.export_als_to_gt import ID_COVERAGE_MIN, id_coverage, main
from tests.labeling.synth_session import LayerSpec, session_als_file


@dataclass
class _Track:
    track_id: str | None


def test_id_coverage_counts_resolved():
    resolved, total, frac = id_coverage(
        [_Track("a"), _Track(None), _Track("b"), _Track(None)]
    )
    assert (resolved, total, frac) == (2, 4, 0.5)


def test_id_coverage_empty_is_full():
    # no tracks -> vacuously full coverage (don't refuse an empty session on this)
    assert id_coverage([]) == (0, 0, 1.0)


def _empty_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"set_id": "synth01", "mix_duration_s": 256.0, "tracks": []})
    )


def test_export_refuses_when_ids_unresolved(tmp_path, capsys):
    # An audible clip against an empty manifest -> track_id never resolves ->
    # coverage 0 < ID_COVERAGE_MIN -> refuse, and write nothing.
    als = session_als_file(tmp_path, layer_specs=(LayerSpec("layer-audible"),))
    _empty_manifest(tmp_path)
    out = tmp_path / "gt.yaml"

    rc = main(["--als", str(als), "--set-dir", str(tmp_path), "--out", str(out)])

    assert rc == 1
    assert not out.exists()
    assert "recording_id" in capsys.readouterr().err  # coverage-gate message


def test_export_allow_invalid_overrides_coverage(tmp_path):
    als = session_als_file(tmp_path, layer_specs=(LayerSpec("layer-audible"),))
    _empty_manifest(tmp_path)
    out = tmp_path / "gt.yaml"

    rc = main(
        [
            "--als",
            str(als),
            "--set-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--allow-invalid",
        ]
    )

    assert rc == 0
    assert out.exists()


def test_coverage_threshold_is_sane():
    assert 0.0 < ID_COVERAGE_MIN < 1.0

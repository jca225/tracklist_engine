"""GT export must refuse to write a fixture that content-bound almost no ids.

``_content_bind`` fills ``recording_id`` (and sets ``id_source="content"``) ONLY
on a content-catalog hash match (sha256 of the file, or the tag-invariant mdat
hash). A stale ``.als``/missing ``content_catalog.json`` resolves ~0 ids; the
exporter would otherwise write a fixture that joins to nothing downstream and
would corrupt canonical ``set_ground_truth`` on write-back. Regression for the
BB12 re-export (2026-07-12) which resolved 1/163 track_ids and was written
silently; re-based on content binding (not raw ``track_id``) so a stale/poisoned
id can no longer count as resolved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from labeling.extract.export_als_to_gt import ID_COVERAGE_MIN, id_coverage, main
from tests.labeling.synth_session import LayerSpec, session_als_file


@dataclass
class _Track:
    track_id: str | None
    id_source: str = ""


def test_id_coverage_counts_resolved():
    resolved, total, frac = id_coverage(
        [
            _Track("a", "content"),
            _Track(None, "abstain"),
            _Track("b", "content"),
            _Track(None, "abstain"),
            _Track("stale_c", "abstain"),  # non-null track_id but NOT content-bound
        ]
    )
    # old track_id-truthy code would give (3, 5, 0.6) — "stale_c" would wrongly count
    assert (resolved, total, frac) == (2, 5, 0.4)


def test_id_coverage_empty_is_full():
    # no tracks -> vacuously full coverage (don't refuse an empty session on this)
    assert id_coverage([]) == (0, 0, 1.0)


def _empty_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"set_id": "synth01", "mix_duration_s": 256.0, "tracks": []})
    )


def test_export_refuses_when_ids_unresolved(tmp_path, capsys):
    # An audible clip with no content_catalog.json -> never content-binds ->
    # coverage 0 < ID_COVERAGE_MIN -> refuse, and write nothing.
    als = session_als_file(tmp_path, layer_specs=(LayerSpec("layer-audible"),))
    _empty_manifest(tmp_path)
    out = tmp_path / "gt.yaml"

    rc = main(["--als", str(als), "--set-dir", str(tmp_path), "--out", str(out)])

    assert rc == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "content-bound" in err  # coverage-gate message
    assert "Abstained:" in err  # lists the abstained slots


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

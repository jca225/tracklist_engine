from __future__ import annotations

from core.acquisition_case import open_worklist, ProblemClass
from ingest.scan_source import open_cases_for_suspect, _klass_to_problem


def test_klass_mapping():
    assert _klass_to_problem("topic_original") is ProblemClass.WRONG_VERSION
    assert _klass_to_problem("mashup_metadata_gap") is ProblemClass.STRUCTURE
    assert _klass_to_problem("anything_else") is ProblemClass.WRONG_VERSION


def test_suspect_opens_case_per_placement(tmp_path):
    opened = open_cases_for_suspect(
        track_id="r1",
        klass="wrong_remix",
        detail="oEmbed lacks remixer",
        placements=[("1fsnxchk", "030"), ("2nvzlh2k", "045")],
        root=tmp_path,
    )
    assert len(opened) == 2
    wl = open_worklist(root=tmp_path)
    assert {c.set_id for c in wl} == {"1fsnxchk", "2nvzlh2k"}
    assert all(ProblemClass.WRONG_VERSION in c.problem_classes for c in wl)


def test_suspect_with_no_placements_opens_nothing(tmp_path):
    assert open_cases_for_suspect("r", "wrong_remix", "d", [], root=tmp_path) == []
    assert open_worklist(root=tmp_path) == []

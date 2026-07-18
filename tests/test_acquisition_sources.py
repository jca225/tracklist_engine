import json
from pathlib import Path
from core.acquisition_case import open_worklist, ProblemClass
from ingest.acquisition_sources import open_cases_from_never_matched


def test_residual_source_opens_one_case_per_entry(tmp_path):
    nm = tmp_path / "1fsnxchk_never_matched.json"
    nm.write_text(
        json.dumps(
            {
                "set_id": "1fsnxchk",
                "never_matched_recordings": [
                    {
                        "recording_id": "1jz334x5",
                        "slot_label": "097",
                        "claimed_stem": "acappella",
                        "reason": "x",
                    },
                    {
                        "recording_id": "1q9r2r8x",
                        "slot_label": "121",
                        "claimed_stem": "instrumental",
                        "reason": "y",
                    },
                ],
            }
        )
    )
    root = tmp_path / "cases"
    opened = open_cases_from_never_matched(nm, root=root)
    assert len(opened) == 2
    wl = open_worklist(root=root)
    assert len(wl) == 2
    assert all(ProblemClass.MISSING_ASSET in c.problem_classes for c in wl)


def test_residual_source_is_idempotent(tmp_path):
    nm = tmp_path / "s_never_matched.json"
    nm.write_text(
        json.dumps(
            {
                "set_id": "s",
                "never_matched_recordings": [
                    {
                        "recording_id": "r",
                        "slot_label": "1",
                        "claimed_stem": "regular",
                        "reason": "x",
                    }
                ],
            }
        )
    )
    root = tmp_path / "cases"
    open_cases_from_never_matched(nm, root=root)
    open_cases_from_never_matched(nm, root=root)
    assert len(open_worklist(root=root)) == 1  # no duplicate

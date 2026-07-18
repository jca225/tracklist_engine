"""Case-source adapters: turn detection signals into OPEN acquisition cases.

Imports ``core.acquisition_case`` (downward only). The residual source consumes
the scorer's ``never_matched.json``; the manual-scan source (Task 6) consumes
wrong-version suspects.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.acquisition_case import AcquisitionCase, CaseClaim, ProblemClass, open_case
from core.identity import normalize_stem


def open_cases_from_never_matched(
    json_path: str | Path,
    root: str | Path = "data/acquisition_cases",
) -> list[AcquisitionCase]:
    """Open one OPEN case per never-matched GT recording (deduped)."""
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    set_id = str(doc.get("set_id") or "")
    opened: list[AcquisitionCase] = []
    for entry in doc.get("never_matched_recordings", []):
        recording_id = str(entry.get("recording_id") or "")
        opened.append(
            open_case(
                set_id=set_id,
                slot_label=str(entry.get("slot_label") or ""),
                recording_id=recording_id,
                claim=CaseClaim(
                    recording_id=recording_id,
                    stem=normalize_stem(entry.get("claimed_stem")),
                ),
                problem_classes=(ProblemClass.MISSING_ASSET,),
                impact_score=1,
                notes=str(entry.get("reason") or "never matched by any predicted span"),
                root=root,
            )
        )
    return opened

"""Manual wrong-version scan → acquisition cases.

A suspect is keyed by track (corpus-wide); it maps to one case per placement in
``set_track_slots``. Pure mapping (``open_cases_for_suspect``) + a DB-join helper
(``placements_for_track``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.acquisition_case import AcquisitionCase, ProblemClass, open_case

_KLASS: dict[str, ProblemClass] = {
    "topic_original": ProblemClass.WRONG_VERSION,
    "wrong_remix": ProblemClass.WRONG_VERSION,
    "live_suspect": ProblemClass.WRONG_VERSION,
    "mashup_metadata_gap": ProblemClass.STRUCTURE,
}


def _klass_to_problem(klass: str) -> ProblemClass:
    return _KLASS.get(klass, ProblemClass.WRONG_VERSION)


def open_cases_for_suspect(
    track_id: str,
    klass: str,
    detail: str,
    placements: list[tuple[str, str]],
    root: str | Path = "data/acquisition_cases",
) -> list[AcquisitionCase]:
    """Open one case per ``(set_id, slot_label)`` placement of a suspect track."""
    problem = _klass_to_problem(klass)
    opened: list[AcquisitionCase] = []
    for set_id, slot_label in placements:
        opened.append(
            open_case(
                set_id=set_id,
                slot_label=slot_label,
                recording_id=track_id,
                problem_classes=(problem,),
                impact_score=1,
                notes=f"wrong-version scan ({klass}): {detail}",
                root=root,
            )
        )
    return opened


def placements_for_track(db_path: Path, track_id: str) -> list[tuple[str, str]]:
    """Every ``(set_id, slot_label)`` where ``track_id`` is claimed."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT set_id, slot_label FROM set_track_slots WHERE recording_id = ?",
            (track_id,),
        ).fetchall()
    finally:
        conn.close()
    return [(str(s), str(sl)) for s, sl in rows]

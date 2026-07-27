"""Title→stem heuristic LF.

Reads ``claimed_title`` only — never ``claimed_stem``. Canary gold must come
from ``canary_stem_gold.json``, not from this LF's votes.

TEMP: this LF is the *canary source* graded in Promote — keep it independent
of claimed_stem_lf. Inline comments are pedagogy.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from sensors.lfs.types import Emission, InferenceUnitOut, NativeScore
from sensors.schemas import validate_payload

SOURCE_NAME = "title_stem_heuristic_lf"
SOURCE_FAMILY = "symbolic"
SOURCE_SPEC_ID = "00000000-0000-4000-8000-000000000003"

# TEMP: measured on BB11/BB12 under abstain-if-no-keyword: 4 votes, 4 correct.
EXPECTED_CANARY_ACCURACY = 1.0
CANARY_TOLERANCE = 0.05

_ACAPELLA = re.compile(r"a\s*capp?ella|acapella", re.IGNORECASE)
_INSTRUMENTAL = re.compile(r"instrumental", re.IGNORECASE)


def guess_stem_from_title(title: str | None) -> str | None:
    """TEMP: keyword sniff only — None means abstain (no keyword found)."""
    if not title:
        return None
    # TEMP: check instrumental first so "instrumental acapella" edge cases are rare.
    if _INSTRUMENTAL.search(title):
        return "instrumental"
    if _ACAPELLA.search(title):
        return "acappella"
    return None


def _load_slots(fixtures_dir: Path) -> list[dict[str, Any]]:
    # TEMP: same slot loader as claimed_stem — must share unit_id scheme.
    inv_path = fixtures_dir / "inventory.json"
    if inv_path.exists():
        data = json.loads(inv_path.read_text(encoding="utf-8"))
        slots: list[dict[str, Any]] = data.get("slots", [])
        return slots
    slots: list[dict[str, Any]] = []
    for path in sorted(fixtures_dir.glob("*_slots.json")):
        slots.extend(json.loads(path.read_text(encoding="utf-8")))
    return slots


def run_title_stem_heuristic_lf(
    fixtures_dir: Path,
    *,
    producer_run_id: str | None = None,
) -> tuple[list[InferenceUnitOut], list[Emission]]:
    """TEMP: Propose from title keywords; Rust canaries this LF vs held-out gold."""
    run_id = producer_run_id or str(uuid.uuid4())
    units: list[InferenceUnitOut] = []
    emissions: list[Emission] = []

    for slot in _load_slots(fixtures_dir):
        # TEMP: MUST match claimed_stem_lf unit_id formula (set_id:row_index).
        unit_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{slot['set_id']}:{slot['row_index']}"))
        subject_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"slot:{slot['set_id']}:{slot['row_index']}")
        )
        units.append(
            InferenceUnitOut(
                unit_id=unit_id,
                set_id=slot["set_id"],
                axis="identity",
                subject_type="set-slot",
                subject_id=subject_id,
                generation_parameters_hash="title_stem_heuristic_lf:v0.1.0",
                split="development",  # TEMP: canary/dev split, not train
            )
        )
        # TEMP: title only — never read claimed_stem (independence for canary).
        guess = guess_stem_from_title(slot.get("claimed_title"))
        if guess is None:
            # TEMP: no keyword → abstain (still a row; never dropped).
            em = Emission(
                emission_id=str(uuid.uuid4()),
                unit_id=unit_id,
                source_spec_id=SOURCE_SPEC_ID,
                source_name=SOURCE_NAME,
                source_family=SOURCE_FAMILY,
                producer_run_id=run_id,
                value=None,
                abstained=True,
                native_score=NativeScore(
                    family=SOURCE_FAMILY, value=0.0, scale="abstain", support=None
                ),
            )
        else:
            em = Emission(
                emission_id=str(uuid.uuid4()),
                unit_id=unit_id,
                source_spec_id=SOURCE_SPEC_ID,
                source_name=SOURCE_NAME,
                source_family=SOURCE_FAMILY,
                producer_run_id=run_id,
                value=guess,
                abstained=False,
                native_score=NativeScore(family=SOURCE_FAMILY, value=1.0, scale="count", support=1),
            )
        validate_payload("evidence_emission", em.model_dump())
        emissions.append(em)

    return units, emissions

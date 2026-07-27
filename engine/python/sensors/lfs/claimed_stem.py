"""Claimed-stem labeling function (symbolic family).

Reads gold slot fixtures and emits ``EvidenceEmission`` rows for the Rust PWS
matrix. Abstains when ``claimed_stem`` is missing — never invents a stem.

Does **not** supply canary gold — use ``load_canary_stem_gold``. The Promote
canary source is ``title_stem_heuristic_lf`` (title keywords only).

TEMP: inline comments below are pedagogy — strip when comfortable.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from sensors.lfs.canary_gold import load_canary_stem_gold
from sensors.lfs.title_stem import (
    CANARY_TOLERANCE,
    EXPECTED_CANARY_ACCURACY,
    run_title_stem_heuristic_lf,
)
from sensors.lfs.title_stem import (
    SOURCE_NAME as TITLE_SOURCE_NAME,
)
from sensors.lfs.types import Emission, InferenceUnitOut, NativeScore
from sensors.schemas import validate_payload

SOURCE_NAME = "claimed_stem_lf"
SOURCE_FAMILY = "symbolic"
SOURCE_SPEC_ID = "00000000-0000-4000-8000-000000000001"
PROCESS_VERSION = "0.1.0"


def _load_slots(fixtures_dir: Path) -> list[dict[str, Any]]:
    # TEMP: prefer inventory.json (migrate shape); else glue *_slots.json files.
    inv_path = fixtures_dir / "inventory.json"
    if inv_path.exists():
        data = json.loads(inv_path.read_text(encoding="utf-8"))
        slots: list[dict[str, Any]] = data.get("slots", [])
        return slots
    slots = []
    for path in sorted(fixtures_dir.glob("*_slots.json")):
        slots.extend(json.loads(path.read_text(encoding="utf-8")))
    return slots


def run_claimed_stem_lf(
    fixtures_dir: Path,
    *,
    producer_run_id: str | None = None,
) -> tuple[list[InferenceUnitOut], list[Emission]]:
    """TEMP: Propose votes from the DJ's claimed_stem field on each slot."""
    run_id = producer_run_id or str(uuid.uuid4())
    units: list[InferenceUnitOut] = []
    emissions: list[Emission] = []

    for slot in _load_slots(fixtures_dir):
        # TEMP: stable unit_id so gold JSON and both LFs agree on the same cell.
        unit_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{slot['set_id']}:{slot['row_index']}"))
        subject_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"slot:{slot['set_id']}:{slot['row_index']}")
        )
        stem = slot.get("claimed_stem")

        # TEMP: one InferenceUnit = one question ("what stem is this slot?").
        units.append(
            InferenceUnitOut(
                unit_id=unit_id,
                set_id=slot["set_id"],
                axis="identity",
                subject_type="set-slot",
                subject_id=subject_id,
                generation_parameters_hash="claimed_stem_lf:v0.1.0",
                split="train",
            )
        )

        if stem:
            # TEMP: vote — family=symbolic so co-training can target opposite view.
            em = Emission(
                emission_id=str(uuid.uuid4()),
                unit_id=unit_id,
                source_spec_id=SOURCE_SPEC_ID,
                source_name=SOURCE_NAME,
                source_family=SOURCE_FAMILY,
                producer_run_id=run_id,
                value=stem,
                abstained=False,
                native_score=NativeScore(family=SOURCE_FAMILY, value=1.0, scale="count", support=1),
            )
        else:
            # TEMP: ABSTAIN — never invent a stem when the field is missing.
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
        # TEMP: catch shape bugs at Propose time (Rust will validate again).
        validate_payload("evidence_emission", em.model_dump())
        emissions.append(em)

    return units, emissions


def write_vertical_slice(fixtures_dir: Path, out_dir: Path) -> Path:
    """Propose-only: package fixtures into ``staging/vertical_lf_bundle.json``.

    Runs ``claimed_stem_lf`` + ``title_stem_heuristic_lf``, attaches independent
    canary gold (``canary_stem_gold.json``). Does not Decide/Promote — that is
    ``dj_migrate pws-fit`` / ``run-round`` in Rust.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # TEMP 1: symbolic LF votes (claimed_stem field).
    units, claimed = run_claimed_stem_lf(fixtures_dir)
    # TEMP 2: independent title heuristic (canary source — different signal).
    _title_units, title_ems = run_title_stem_heuristic_lf(fixtures_dir)
    # TEMP 3: held-out gold — NOT derived from either LF above.
    gold = load_canary_stem_gold(fixtures_dir)

    # TEMP: concatenate emissions; units come from claimed_stem (same unit_ids).
    emissions = list(claimed) + list(title_ems)
    bundle = {
        "process_version": PROCESS_VERSION,
        "units": [u.model_dump() for u in units],
        "emissions": [e.model_dump() for e in emissions],
        # TEMP: Promote grades TITLE lf, not claimed_stem_lf.
        "canaries": [
            {
                "source_name": TITLE_SOURCE_NAME,
                "known_accuracy": EXPECTED_CANARY_ACCURACY,
                "tolerance": CANARY_TOLERANCE,
            }
        ],
        # Singular kept for older readers; Prefer ``canaries``.
        "canary": {
            "source_name": TITLE_SOURCE_NAME,
            "known_accuracy": EXPECTED_CANARY_ACCURACY,
            "tolerance": CANARY_TOLERANCE,
        },
        "gold_by_unit": gold,
        "note": (
            "gold from canary_stem_gold.json; canary source is title_stem_heuristic_lf "
            "(not claimed_stem_lf)"
        ),
    }
    path = out_dir / "vertical_lf_bundle.json"
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path

"""Independent canary gold (not derived from LF emissions)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def slot_unit_id(set_id: str, row_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{set_id}:{row_index}"))


def load_canary_stem_gold(fixtures_dir: Path) -> dict[str, str]:
    """Load held-out stem labels keyed by InferenceUnit.unit_id (string UUID)."""
    path = fixtures_dir / "canary_stem_gold.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} — canary gold is required (do not use LF votes as gold)"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    gold: dict[str, str] = {}
    for row in data.get("labels", []):
        uid = slot_unit_id(row["set_id"], int(row["row_index"]))
        gold[uid] = str(row["stem"])
    if not gold:
        raise ValueError(f"empty canary gold in {path}")
    return gold

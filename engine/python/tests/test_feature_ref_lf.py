from __future__ import annotations

import json
from pathlib import Path

from sensors.lfs.feature_ref import emit_with_feature_ref, write_feature_ref_demo


def test_emission_cites_feature_artifact_id() -> None:
    em = emit_with_feature_ref(
        unit_id="00000000-0000-4000-8000-000000000010",
        feature_artifact_id="00000000-0000-4000-8000-000000000020",
        vote="acappella",
    )
    assert em["abstained"] is False
    assert em["evidence_ids"] == ["00000000-0000-4000-8000-000000000020"]
    assert em["feature_artifact_id"] == "00000000-0000-4000-8000-000000000020"


def test_write_feature_ref_demo(tmp_path: Path) -> None:
    report = tmp_path / "reg.json"
    report.write_text(
        json.dumps({"feature_artifact_id": "00000000-0000-4000-8000-000000000020"}),
        encoding="utf-8",
    )
    out = write_feature_ref_demo(report, tmp_path / "emissions.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["emissions"][0]["evidence_ids"][0].endswith("020")

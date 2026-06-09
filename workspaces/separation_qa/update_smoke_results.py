"""Merge smoke_report.json into docs/roformer_separation_plan.md Results table."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "workspaces/separation_qa/smoke_out/smoke_report.json"
DOC = REPO / "docs/roformer_separation_plan.md"


def main() -> int:
    if not REPORT.is_file():
        print(f"missing {REPORT}")
        return 1
    data = json.loads(REPORT.read_text())
    tracks = data.get("tracks", {})
    rof_bleeds = [t["bleed_rms"]["roformer_inst_ensemble"] for t in tracks.values()]
    dem_bleeds = [t["bleed_rms"]["demucs"] for t in tracks.values()]
    med_rof = sum(rof_bleeds) / len(rof_bleeds) if rof_bleeds else float("nan")
    med_dem = sum(dem_bleeds) / len(dem_bleeds) if dem_bleeds else float("nan")

    row = (
        f"| roformer ensemble (MPS smoke, {data.get('clip_s')}s clips) | — | — | — | "
        f"~{sum(sum(t['model_timings_s'].values()) for t in tracks.values()) / max(len(tracks),1):.0f}s/track | "
        f"median bleed {med_rof:.5f} vs demucs {med_dem:.5f} |"
    )
    text = DOC.read_text()
    marker = "| roformer ensemble | — | — | — | — | candidate default |"
    if marker not in text:
        print("results marker not found in doc")
        return 1
    DOC.write_text(text.replace(marker, row))
    print(f"updated {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

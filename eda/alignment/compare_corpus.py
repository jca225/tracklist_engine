"""Compare mix structure probes across sets (corpus baseline)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SetSpec:
    set_id: str
    artifact: Path
    audio: Path
    gt: Path | None = None


DEFAULT_SETS = (
    SetSpec(
        set_id="1fsnxchk",
        artifact=Path("data/analysis/1fsnxchk_mix_mert.npz"),
        audio=Path.home() / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/mix.m4a",
        gt=Path("labeling/fixtures/bb12_ground_truth.yaml"),
    ),
    SetSpec(
        set_id="2nvzlh2k",
        artifact=Path("data/analysis/2nvzlh2k_mix_mert.npz"),
        audio=Path.home() / "aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11/mix.m4a",
        gt=None,
    ),
)


def _f1(stream: dict, key: str = "mir_local") -> float | None:
    scores = stream.get("scores") or {}
    if key not in scores:
        return None
    return float(scores[key]["f1"])


def _mid_pir(stream: dict) -> float:
    return float(stream["information"]["trace_stats"]["fraction_bars_mid_pir_band"])


def _gt_lift(stream: dict) -> float | None:
    info = stream.get("information") or {}
    if "gt_vs_interior_mir_lift" not in info:
        return None
    return float(info["gt_vs_interior_mir_lift"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run dual-stream probe on multiple sets")
    p.add_argument("--out", type=Path, default=Path("data/analysis/mix_structure_corpus_compare.json"))
    args = p.parse_args(argv)

    rows: list[dict] = []
    for spec in DEFAULT_SETS:
        if not spec.artifact.is_file():
            print(f"skip {spec.set_id}: missing {spec.artifact}", file=sys.stderr)
            continue
        if not spec.audio.is_file():
            print(f"skip {spec.set_id}: missing {spec.audio}", file=sys.stderr)
            continue

        out_path = Path(f"data/analysis/{spec.set_id}_structure_probe_v2.json")
        cmd = [
            sys.executable,
            "-m",
            "eda.alignment.mix_structure_probe",
            "--artifact",
            str(spec.artifact),
            "--audio",
            str(spec.audio),
            "--out",
            str(out_path),
        ]
        if spec.gt is not None:
            cmd.extend(["--gt", str(spec.gt)])

        print(f"running {spec.set_id}...", file=sys.stderr)
        subprocess.run(cmd, check=True)
        result = json.loads(out_path.read_text())
        mert = result["streams"]["mert_vq"]
        chroma = result["streams"].get("chroma_vq")
        combined = result["streams"].get("combined_mir_local")

        row = {
            "set_id": spec.set_id,
            "n_bars": result["n_bars"],
            "mert_mid_pir_frac": _mid_pir(mert),
            "mert_f1_local": _f1(mert, "mir_local"),
            "mert_f1_global": _f1(mert, "mir_global"),
            "mert_gt_lift": _gt_lift(mert),
            "chroma_mid_pir_frac": _mid_pir(chroma) if chroma else None,
            "chroma_f1_local": _f1(chroma, "mir_local") if chroma else None,
            "combined_f1_local": _f1(combined, "mir_local") if combined else None,
        }
        rows.append(row)

    summary = {"sets": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

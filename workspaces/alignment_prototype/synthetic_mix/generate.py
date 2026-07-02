#!/usr/bin/env python3
"""Generate realistic synthetic DJ mashups with perfect GT labels.

Usage:
  venvs/audio/bin/python -m workspaces.alignment_prototype.synthetic_mix.generate \\
    --n 10 --curriculum medium --out data/synthetic_mixes --seed 0

  # Ear-check: list generated mixes + paths to mix.flac
  venvs/audio/bin/python -m workspaces.alignment_prototype.synthetic_mix.generate \\
    --list data/synthetic_mixes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.result import Err  # noqa: E402
from labeling.ground_truth.schema import save  # noqa: E402

from .catalog import load_catalog  # noqa: E402
from .labels import scenario_to_gt  # noqa: E402
from .render import render_scenario, write_flac  # noqa: E402
from .scenario import sample_scenario  # noqa: E402

DEFAULT_OUT = _REPO / "data" / "synthetic_mixes"
MANIFEST = "corpus_manifest.json"


def _write_mix(out_dir: Path, scenario, rendered) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = out_dir / "refs"
    refs.mkdir(exist_ok=True)

    write_flac(out_dir / "mix.flac", rendered.mix, rendered.sr)
    write_flac(out_dir / "mix_vocals.flac", rendered.mix_vocals, rendered.sr)
    write_flac(
        out_dir / "mix_instrumental.flac", rendered.mix_instrumental, rendered.sr
    )

    bed = scenario.bed
    shutil.copy2(bed.path, refs / f"{bed.recording_id}_instrumental.flac")
    seen = {bed.recording_id}
    for ov in scenario.overlays:
        rid = ov.payload.recording_id
        if rid in seen:
            continue
        seen.add(rid)
        shutil.copy2(ov.payload.path, refs / f"{rid}_vocals.flac")

    gt = scenario_to_gt(scenario)
    match save(gt, out_dir / "ground_truth.yaml"):
        case Err(e):
            raise RuntimeError(f"GT save failed: {e.detail}")


def generate(args: argparse.Namespace) -> int:
    catalog = load_catalog(require_key_bpm=not args.no_key_bpm)
    print(
        f"catalog: {len(catalog.beds)} beds, {len(catalog.payloads)} payloads "
        f"(key/bpm={'on' if not args.no_key_bpm else 'off'})"
    )
    if len(catalog.beds) < 1 or len(catalog.payloads) < 1:
        print("need ≥1 bed + ≥1 payload (check pi + local stems)", file=sys.stderr)
        return 1

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest: list[dict] = []

    made = 0
    attempts = 0
    max_attempts = args.n * 30
    while made < args.n and attempts < max_attempts:
        attempts += 1
        mix_id = f"synth_{made + 1:04d}"
        scenario = sample_scenario(
            catalog,
            mix_id=mix_id,
            curriculum=args.curriculum,
            rng=rng,
            mix_duration_s=args.duration_s,
        )
        if scenario is None:
            continue
        rendered = render_scenario(scenario)
        mix_dir = out_root / mix_id
        _write_mix(mix_dir, scenario, rendered)
        manifest.append(
            {
                "mix_id": mix_id,
                "dir": str(mix_dir.relative_to(out_root)),
                "curriculum": args.curriculum,
                "bed": scenario.bed.recording_id,
                "overlays": [o.payload.recording_id for o in scenario.overlays],
                "duration_s": scenario.mix_duration_s,
            }
        )
        made += 1
        bed = scenario.bed.label[:28]
        pays = ", ".join(o.payload.label[:20] for o in scenario.overlays)
        print(f"  [{made}/{args.n}] {mix_id}: {bed} + [{pays}]")

    (out_root / MANIFEST).write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {made} mixes → {out_root}  (attempts={attempts})")
    if made < args.n:
        print(
            f"warning: only {made}/{args.n} — relax curriculum or add stems",
            file=sys.stderr,
        )
    return 0 if made else 1


def list_corpus(out_root: Path) -> int:
    manifest_path = out_root / MANIFEST
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = [
            {"mix_id": d.name, "dir": d.name}
            for d in sorted(out_root.iterdir())
            if d.is_dir()
        ]
    print(f"{len(manifest)} mixes in {out_root}\n")
    for row in manifest:
        mix_dir = out_root / row["dir"]
        mix_flac = mix_dir / "mix.flac"
        gt = mix_dir / "ground_truth.yaml"
        print(
            f"  {row['mix_id']:12s}  mix={'OK' if mix_flac.is_file() else 'MISSING':7s}  "
            f"gt={'OK' if gt.is_file() else 'MISSING'}  {mix_flac}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=10, help="Number of mixes to generate")
    p.add_argument("--curriculum", choices=["easy", "medium", "hard"], default="medium")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--duration-s", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-key-bpm", action="store_true", help="Skip pi key/bpm (not recommended)"
    )
    p.add_argument(
        "--list", type=Path, default=None, metavar="DIR", help="List existing corpus"
    )
    args = p.parse_args(argv)

    if args.list is not None:
        return list_corpus(args.list)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())

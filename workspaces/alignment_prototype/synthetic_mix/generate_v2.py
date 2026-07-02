#!/usr/bin/env python3
"""Generate BB12-realistic synthetic windows (v2).

Usage:
  venvs/audio/bin/python -m workspaces.alignment_prototype.synthetic_mix.generate_v2 \\
    --n 20 --curriculum bb12-lite --out data/synthetic_mixes_v2 --seed 1

  venvs/audio/bin/python -m workspaces.alignment_prototype.synthetic_mix.generate_v2 \\
    --validate-only data/synthetic_mixes_v2
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

from core.result import Err, Ok  # noqa: E402
from labeling.ground_truth.schema import load, save  # noqa: E402

from .catalog import load_catalog  # noqa: E402
from .labels_v2 import window_to_gt  # noqa: E402
from .render_v2 import render_window_v2, write_flac  # noqa: E402
from .scenario_v2 import sample_window_v2  # noqa: E402
from .sections import get_curriculum  # noqa: E402
from .validate import bb12_reference_stats, format_stats, stats_from_gt, validate_window  # noqa: E402

DEFAULT_OUT = _REPO / "data" / "synthetic_mixes_v2"
MANIFEST = "corpus_manifest.json"


def _write_regular_ref(dest: Path, regular) -> None:
    """Write a full-song reference (instrumental + vocals summed)."""
    import numpy as np

    from workspaces.alignment_prototype.refine_ref_offsets import SR

    from .render import _load_mono

    inst = _load_mono(regular.instrumental_path)
    voc = _load_mono(regular.vocals_path)
    m = min(len(inst), len(voc))
    full = (inst[:m] + voc[:m]).astype(np.float32)
    peak = max(float(np.max(np.abs(full))), 1e-6)
    write_flac(dest, full * (0.95 / peak), SR)


def _write_window(out_dir: Path, window, rendered) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = out_dir / "refs"
    refs.mkdir(exist_ok=True)

    write_flac(out_dir / "mix.flac", rendered.mix, rendered.sr)
    write_flac(out_dir / "mix_vocals.flac", rendered.mix_vocals, rendered.sr)
    write_flac(out_dir / "mix_instrumental.flac", rendered.mix_instrumental, rendered.sr)

    seen: set[str] = set()
    for block in window.instrumentals:
        rid = block.bed.recording_id
        if rid not in seen:
            seen.add(rid)
            shutil.copy2(block.bed.path, refs / f"{rid}_instrumental.flac")
    for ac in window.acappellas:
        rid = ac.payload.recording_id
        if rid not in seen:
            seen.add(rid)
            shutil.copy2(ac.payload.path, refs / f"{rid}_vocals.flac")
    for reg in window.regulars:
        rid = reg.regular.recording_id
        ref_file = refs / f"{rid}_regular.flac"
        if not ref_file.is_file():
            _write_regular_ref(ref_file, reg.regular)

    gt = window_to_gt(window)
    match save(gt, out_dir / "ground_truth.yaml"):
        case Err(e):
            raise RuntimeError(f"GT save failed: {e.detail}")


def generate(args: argparse.Namespace) -> int:
    catalog = load_catalog(require_key_bpm=not args.no_key_bpm)
    cfg = get_curriculum(args.curriculum)
    print(
        f"catalog: {len(catalog.beds)} beds, {len(catalog.payloads)} payloads | "
        f"curriculum={args.curriculum} window={cfg.window_s}s"
    )
    ref = bb12_reference_stats()
    if ref:
        print(f"BB12 reference: {format_stats(ref)}")

    if len(catalog.beds) < cfg.n_instrumentals or len(catalog.payloads) < 4:
        print("insufficient stems", file=sys.stderr)
        return 1

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest: list[dict] = []
    made = 0
    attempts = 0
    max_attempts = args.n * 40

    while made < args.n and attempts < max_attempts:
        attempts += 1
        mix_id = f"synthv2_{made + 1:04d}"
        window = sample_window_v2(
            catalog,
            mix_id=mix_id,
            curriculum=args.curriculum,
            rng=rng,
        )
        if window is None:
            continue
        gt = window_to_gt(window)
        ok, issues = validate_window(gt, curriculum=args.curriculum)
        if not ok and not args.no_validate:
            if args.verbose:
                print(f"  reject {mix_id}: {'; '.join(issues)}")
            continue
        rendered = render_window_v2(
            window, crossfade_s=cfg.handoff_crossfade_s
        )
        mix_dir = out_root / mix_id
        _write_window(mix_dir, window, rendered)
        st = stats_from_gt(gt)
        manifest.append(
            {
                "mix_id": mix_id,
                "dir": str(mix_dir.relative_to(out_root)),
                "curriculum": args.curriculum,
                "duration_s": window.window_duration_s,
                "spans": st.n_spans,
                "loops": st.n_loops,
                "overlaps": st.overlap_pairs,
            }
        )
        made += 1
        print(f"  [{made}/{args.n}] {format_stats(st)}")

    (out_root / MANIFEST).write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {made} windows → {out_root}  (attempts={attempts})")
    return 0 if made else 1


def validate_corpus(root: Path, curriculum: str) -> int:
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("synthv2_"))
    if not dirs:
        print(f"no synthv2_* dirs in {root}", file=sys.stderr)
        return 1
    ok_n = 0
    for d in dirs:
        gt_path = d / "ground_truth.yaml"
        match load(gt_path):
            case Err(e):
                print(f"  FAIL {d.name}: {e.detail}")
                continue
            case Ok(gt):
                ok, issues = validate_window(gt, curriculum=curriculum)
                st = stats_from_gt(gt)
                flag = "OK" if ok else "WARN"
                print(f"  {flag} {format_stats(st)}")
                if issues:
                    for iss in issues:
                        print(f"       - {iss}")
                if ok:
                    ok_n += 1
    print(f"\n{ok_n}/{len(dirs)} passed validation")
    return 0 if ok_n == len(dirs) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument(
        "--curriculum",
        choices=["bb12-lite", "bb12-med", "bb12-full"],
        default="bb12-lite",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-key-bpm", action="store_true")
    p.add_argument("--no-validate", action="store_true", help="Accept windows failing stats gate")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--validate-only",
        type=Path,
        default=None,
        metavar="DIR",
        help="Validate existing v2 corpus",
    )
    args = p.parse_args(argv)
    if args.validate_only is not None:
        return validate_corpus(args.validate_only, args.curriculum)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())

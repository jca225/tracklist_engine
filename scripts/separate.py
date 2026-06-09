"""Standalone stem separation — drive a backend on one file for QA / A-B.

Wraps the same backends the analysis pipeline uses, via the project adapters
(Python API, reliable output tracking — not CLI globbing):
  - `uvr`    : the audio-separator vocal-cleanup chain (analysis/uvr_chain.yaml)
  - `demucs`   : htdemucs_ft 2-stem (the current pipeline default)
  - `roformer` : MSST RoFormer ensemble (see analysis/roformer_chain.yaml)
  - `both`     : run demucs + roformer into separate folders for A/B

Supersedes the old scripts/sota_stems.py.

Usage:
    ./venvs/audio/bin/python scripts/separate.py --input song.m4a
    ./venvs/audio/bin/python scripts/separate.py --input song.wav \\
        --separator both --byproducts --out-dir ./out

Output:
    <out-dir>/uvr/{vocals,instrumental}.<fmt>
    <out-dir>/uvr/byproducts/<stage>_<label>.<fmt>   (uvr + --byproducts)
    <out-dir>/demucs/{vocals,instrumental}.<fmt>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run from repo root: make the package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.adapters import demucs_adapter, roformer_chain_adapter, uvr_chain_adapter  # noqa: E402
from analysis.roformer_config import RoformerChainConfig  # noqa: E402
from analysis.separation_config import ChainConfig  # noqa: E402


def _flatten(backend_root: Path) -> None:
    """Adapters write to `<root>/<id>/`; we pass id=0, so lift those files up
    into `<root>/` and drop the numeric leaf for a clean standalone layout."""
    leaf = backend_root / "0"
    if not leaf.is_dir():
        return
    for f in leaf.iterdir():
        target = backend_root / f.name
        if target.exists():
            target.unlink()
        f.rename(target)
    leaf.rmdir()


def _run_uvr(input_path: Path, out_dir: Path, *, config: Path | None,
             device: str, byproducts: bool) -> int:
    cfg = ChainConfig.from_yaml(config) if config else ChainConfig.default()
    print(f">>> uvr chain: {[s.name for s in cfg.stages]} (device={device})", flush=True)
    load_r = uvr_chain_adapter.load(cfg, device=device)
    if not load_r.is_ok():
        print(f"ERROR uvr load: {load_r.error.kind} — {load_r.error.detail}", file=sys.stderr)
        return 1
    root = out_dir / "uvr"
    bp_dir = (root / "byproducts") if byproducts else None
    t0 = time.time()
    sep_r = uvr_chain_adapter.separate(
        load_r.value, input_path, root, track_audio_id=0, byproducts_dir=bp_dir,
    )
    if not sep_r.is_ok():
        print(f"ERROR uvr separate: {sep_r.error.kind} — {sep_r.error.detail}", file=sys.stderr)
        return 1
    _flatten(root)
    print(f"  uvr done in {time.time() - t0:.1f}s -> {root}/")
    return 0


def _run_roformer(input_path: Path, out_dir: Path, *, config: Path | None, device: str) -> int:
    cfg = RoformerChainConfig.from_yaml(config) if config else RoformerChainConfig.default()
    print(f">>> roformer {cfg.version} (device={device})", flush=True)
    load_r = roformer_chain_adapter.load(cfg, device=device)
    if not load_r.is_ok():
        print(f"ERROR roformer load: {load_r.error.kind} — {load_r.error.detail}", file=sys.stderr)
        return 1
    root = out_dir / "roformer"
    t0 = time.time()
    sep_r = roformer_chain_adapter.separate(load_r.value, input_path, root, track_audio_id=0)
    if not sep_r.is_ok():
        print(f"ERROR roformer separate: {sep_r.error.kind} — {sep_r.error.detail}", file=sys.stderr)
        return 1
    _flatten(root)
    print(f"  roformer done in {time.time() - t0:.1f}s -> {root}/")
    return 0


def _run_demucs(input_path: Path, out_dir: Path, *, device: str) -> int:
    print(f">>> demucs htdemucs_ft (device={device})", flush=True)
    load_r = demucs_adapter.load(device=device)
    if not load_r.is_ok():
        print(f"ERROR demucs load: {load_r.error.kind} — {load_r.error.detail}", file=sys.stderr)
        return 1
    root = out_dir / "demucs"
    t0 = time.time()
    sep_r = demucs_adapter.separate(load_r.value, input_path, root, 0)
    if not sep_r.is_ok():
        print(f"ERROR demucs separate: {sep_r.error.kind} — {sep_r.error.detail}", file=sys.stderr)
        return 1
    _flatten(root)
    print(f"  demucs done in {time.time() - t0:.1f}s -> {root}/")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="audio file to separate")
    ap.add_argument("--separator", choices=["uvr", "demucs", "roformer", "both"], default="uvr")
    ap.add_argument("--config", type=Path, default=None,
                    help="UVR chain yaml (uvr) or roformer chain yaml (roformer)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: ./separated/<input-stem>/)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--byproducts", action="store_true",
                    help="(uvr) also save chorus/reverb/echo/noise stems for QA")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        return 2

    out_dir = args.out_dir or (Path("separated") / args.input.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    if args.separator in ("uvr", "both"):
        rc |= _run_uvr(args.input, out_dir, config=args.config,
                       device=args.device, byproducts=args.byproducts)
    if args.separator in ("demucs", "both"):
        rc |= _run_demucs(args.input, out_dir, device=args.device)
    if args.separator in ("roformer", "both"):
        rc |= _run_roformer(args.input, out_dir, config=args.config, device=args.device)

    print(f"\nDone. Output under {out_dir}/")
    return rc


if __name__ == "__main__":
    sys.exit(main())

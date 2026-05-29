"""SOTA 2-stem separation: Mel-Band RoFormer (vocals) + MDX23C-InstVoc HQ (instrumental).

Runs each model independently and keeps the stem each is *best* at. This
beats `mix - vocals` subtraction (which inherits the vocal-extraction
model's noise into the instrumental) and beats picking a single model
that has to do both jobs.

Models (auto-downloaded on first use, ~500 MB each):
  - melband_roformer_instvox_duality_v2.ckpt   (Mel-Band RoFormer V2 by Unwa)
  - MDX23C-8KFFT-InstVoc_HQ_2.ckpt             (MDX23C InstVoc HQ 2)

Usage:
    ./venvs/audio/bin/python scripts/sota_stems.py <input.m4a> [--out-dir DIR]

Output:
    <out-dir>/vocals.flac
    <out-dir>/instrumental.flac
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VOCAL_MODEL = "melband_roformer_instvox_duality_v2.ckpt"
INST_MODEL = "MDX23C-8KFFT-InstVoc_HQ_2.ckpt"

# CLI binary lives next to the python interpreter we're using.
AS_BIN = Path(sys.executable).parent / "audio-separator"


def run_audio_separator(
    *,
    input_path: Path,
    model: str,
    output_dir: Path,
    keep_stem: str,  # "vocals" or "instrumental"
    out_name: str,   # final filename (e.g. "vocals.flac")
) -> Path:
    """Run audio-separator on `input_path` with `model`, copy the chosen
    stem to <output_dir>/<out_name>, return the final path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            str(AS_BIN),
            "--model_filename", model,
            "--output_dir", tmp,
            "--output_format", "flac",
            "--use_autocast",
            str(input_path),
        ]
        print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, check=True)

        # audio-separator names outputs like: <stem>_<input-stem>_<model-arch>.flac
        # find the file whose name contains the wanted stem (case-insensitive)
        candidates = list(Path(tmp).glob("*.flac"))
        match = next(
            (p for p in candidates if keep_stem.lower() in p.name.lower()),
            None,
        )
        if match is None:
            print(f"  produced files: {[p.name for p in candidates]}", file=sys.stderr)
            raise RuntimeError(f"no {keep_stem!r} output found")
        final = output_dir / out_name
        shutil.copy(match, final)
        print(f"  -> {final}")
        return final


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="audio file to separate")
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="output dir (default: ./sota_stems/<input-stem>/)",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (Path("sota_stems") / args.input.stem)

    run_audio_separator(
        input_path=args.input,
        model=VOCAL_MODEL,
        output_dir=out_dir,
        keep_stem="vocals",
        out_name="vocals.flac",
    )
    run_audio_separator(
        input_path=args.input,
        model=INST_MODEL,
        output_dir=out_dir,
        keep_stem="instrumental",
        out_name="instrumental.flac",
    )

    print(f"\nDone. Stems in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

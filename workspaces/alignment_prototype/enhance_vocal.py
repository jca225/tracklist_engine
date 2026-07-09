#!/usr/bin/env python3
"""Standalone vocal restorer — MUST run under the isolated ``venvs/voicefixer``.

Called file->file as a subprocess by ``vocal_enhance.maybe_enhance``. Kept
separate on purpose: VoiceFixer pins old torch/librosa that would break the
Whisper/transformers stack in ``venvs/audio`` (same reasoning as the Essentia
sandbox). Nothing here is imported by the main process.

VoiceFixer outputs 44.1 kHz; the lyrics channel resamples to 16 kHz for Whisper.

Setup (one time)::

    python3 -m venv venvs/voicefixer
    venvs/voicefixer/bin/pip install voicefixer
    # first run downloads the model weights (~hundreds of MB)

To swap in a different restorer (e.g. resemble-enhance), point $VOICEFIXER_PY
at that venv and give it a compatible --in/--out CLI, or edit ``restore`` here.
"""

from __future__ import annotations

import argparse
import sys


def restore(inp: str, out: str, mode: int, cuda: bool) -> None:
    from voicefixer import VoiceFixer

    vf = VoiceFixer()
    # mode 0 = original vocoder (safest for ASR); 1/2 are more aggressive.
    vf.restore(input=inp, output=out, cuda=cuda, mode=mode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--mode", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args(argv)
    restore(args.inp, args.out, args.mode, args.cuda)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

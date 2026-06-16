"""Bleed-residual: the principled, provenance-robust cleanliness signal.

For each candidate VOCAL we run a vocal/instrumental separator and measure the
RMS energy of the *instrumental* output — i.e. how much non-vocal content the
judge can still pull out of the "acappella". A true studio acappella yields
~nothing; a separated stem with drum/synth bleed yields measurable residual.

Unlike floor_db / lowend_ratio_db (which detect a separator's over-stripping
*fingerprint* and so only classify provenance), this measures actual
contamination and should rank monotonically even between two online sources or
two separators, because both candidates are re-judged by the *same* model.

Heavy: needs the MSST venv + BS-RoFormer checkpoint, ~100-300 s/file on MPS, so
we clip to 30 s and cache by (path, mtime).

Run (MSST venv, NOT venvs/audio):
  venvs/msst/bin/python workspaces/separation_qa/bleed_residual.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MSST = REPO / "workspaces" / "msst_webui"
CLIPS = HERE / "_bleed_clips"
CACHE = HERE / "_bleed_cache.json"
CLIP_S = 30
CLIP_SS = 20  # skip intros; take a 30 s window from 20 s in
DEVICE = "mps"
JUDGE_CKPT = "model_bs_roformer_ep_368_sdr_12.9628.ckpt"

# import pair discovery before we chdir into MSST
sys.path.insert(0, str(HERE))
from bb12_pair_eval import discover_pairs  # noqa: E402

os.chdir(MSST)
sys.path.insert(0, str(MSST))
from inference.msst_infer import MSSeparator  # noqa: E402
from utils.logger import get_logger  # noqa: E402


def _clip(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(CLIP_SS),
            "-t",
            str(CLIP_S),
            "-i",
            str(src),
            "-ar",
            "44100",
            "-ac",
            "2",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def _load_judge() -> MSSeparator:
    return MSSeparator(
        model_type="bs_roformer",
        config_path=str(MSST / "configs" / "vocal_models" / f"{JUDGE_CKPT}.yaml"),
        model_path=str(MSST / "pretrain" / "vocal_models" / JUDGE_CKPT),
        device=DEVICE,
        output_format="wav",
        store_dirs={"vocals": "", "instrumental": ""},
        logger=get_logger(),
        debug=False,
    )


def _instrumental_rms(judge: MSSeparator, wav: Path) -> float:
    """Peak-normalise, separate, return RMS of the instrumental (bleed) output."""
    import librosa

    mix, _ = librosa.load(str(wav), mono=False, sr=44100)
    if mix.ndim == 1:
        mix = np.stack([mix, mix])
    peak = float(np.max(np.abs(mix)))
    if peak > 1e-9:
        mix = mix / peak
    stems = judge.separate(mix)
    judge.del_cache()
    inst = stems.get("instrumental")
    if inst is None:
        inst = stems.get("other")
    if inst is None:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(inst))))


def _bleed_db(judge: MSSeparator, src: Path, cache: dict) -> float:
    key = str(src)
    mtime = src.stat().st_mtime
    hit = cache.get(key)
    if hit and hit.get("mtime") == mtime:
        return hit["bleed_db"]
    # parent dir disambiguates: produced stems are all named "vocals.flac"
    clip = CLIPS / f"{src.parent.name}__{src.stem}.wav"
    if not clip.exists():
        _clip(src, clip)
    rms = _instrumental_rms(judge, clip)
    bleed_db = 20.0 * float(np.log10(rms + 1e-10))  # dBFS; lower == cleaner
    cache[key] = {"mtime": mtime, "bleed_db": bleed_db}
    CACHE.write_text(json.dumps(cache))
    return bleed_db


def main() -> int:
    pairs = discover_pairs()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(
        f"== bleed-residual on {len(pairs)} pairs (lower dB == cleaner) ==\n",
        flush=True,
    )
    judge = _load_judge()
    wins = 0
    for key, win_path, lose_path in pairs:
        wb = _bleed_db(judge, win_path, cache)
        lb = _bleed_db(judge, lose_path, cache)
        online_cleaner = wb < lb  # online has less instrumental residual
        wins += int(online_cleaner)
        flag = "OK " if online_cleaner else "XX "
        print(
            f"{flag}{key:38s} online={wb:+6.1f}dB  produced={lb:+6.1f}dB  "
            f"Δ={wb - lb:+5.1f}",
            flush=True,
        )
    n = len(pairs)
    print(f"\n  online cleaner (less bleed): {wins}/{n}  ({wins / n:.0%})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

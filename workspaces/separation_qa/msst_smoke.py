"""MSST RoFormer MPS smoke: 3 BB12 clips, ensemble, demucs compare, bleed proxy.

Run from repo root:
  venvs/msst/bin/python workspaces/separation_qa/msst_smoke.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
MSST = REPO / "workspaces" / "msst_webui"
OUT = REPO / "workspaces" / "separation_qa" / "smoke_out"
BB = Path.home() / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/tracks"
CLIP_S = 30  # MPS is ~5× realtime; 30s keeps smoke under ~90 min
DEVICE = "mps"

# Pinned vocal_models (vocals + instrumental stems).
MODELS: tuple[tuple[str, str], ...] = (
    ("bs_roformer", "model_bs_roformer_ep_368_sdr_12.9628.ckpt"),
    ("mel_band_roformer", "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"),
    ("mel_band_roformer", "kimmel_unwa_ft.ckpt"),
)
VOCAL_MODELS = MODELS
INST_MODELS = MODELS[:2]  # BS + Mel only

SMOKE_TRACKS: tuple[tuple[str, str], ...] = (
    ("manse_freeze", "002__Manse - Freeze Time (AltVersion).m4a"),
    ("calvin_outside", "003__Calvin Harris - Outside.m4a"),
    ("kyle_ispy", "008__KYLE - iSpy (Remix).m4a"),
)

import os

os.chdir(MSST)
sys.path.insert(0, str(MSST))
from inference.msst_infer import MSSeparator  # noqa: E402
from utils.ensemble import ensemble_audios  # noqa: E402
from utils.logger import get_logger  # noqa: E402


def _ffmpeg_clip(src: Path, dst: Path, seconds: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-t", str(seconds), "-ar", "44100", "-ac", "2", str(dst)],
        check=True, capture_output=True,
    )


def _store_dirs(model_type: str) -> dict[str, str]:
    if model_type == "bs_roformer":
        return {"vocals": "", "instrumental": ""}
    return {"vocals": "", "other": ""}


def _normalize_stems(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(stems)
    if "instrumental" not in out and "other" in out:
        out["instrumental"] = out["other"]
    if "vocals" not in out or "instrumental" not in out:
        raise KeyError(f"expected vocals+instrumental, got {list(stems)}")
    return out


def _load_separator(model_type: str, ckpt: str) -> MSSeparator:
    return MSSeparator(
        model_type=model_type,
        config_path=str(MSST / "configs" / "vocal_models" / f"{ckpt}.yaml"),
        model_path=str(MSST / "pretrain" / "vocal_models" / ckpt),
        device=DEVICE,
        output_format="wav",
        store_dirs=_store_dirs(model_type),
        logger=get_logger(),
        debug=False,
    )


def _separate_file(sep: MSSeparator, wav: Path) -> dict[str, np.ndarray]:
    import librosa
    mix, _sr = librosa.load(str(wav), mono=False, sr=44100)
    if mix.ndim == 1:
        mix = np.stack([mix, mix])
    return _normalize_stems(sep.separate(mix))


def _ensemble(paths: list[Path], mode: str = "avg_fft") -> tuple[np.ndarray, int]:
    files = [str(p) for p in paths]
    audio, sr = ensemble_audios(files, mode, [1.0] * len(files))
    return audio, sr


def _bleed_score(inst_wav: Path, judge: MSSeparator) -> float:
    """Proxy bleed-judge: vocal energy when BS model runs on instrumental stem."""
    import librosa
    mix, _ = librosa.load(str(inst_wav), mono=False, sr=44100)
    if mix.ndim == 1:
        mix = np.stack([mix, mix])
    stems = _normalize_stems(judge.separate(mix))
    judge.del_cache()
    voc = stems.get("vocals")
    if voc is None:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(voc))))


def _run_demucs(wav: Path, out_dir: Path) -> None:
    subprocess.run(
        [str(REPO / "venvs/audio/bin/python"), str(REPO / "scripts/separate.py"),
         "--input", str(wav), "--separator", "demucs", "--device", DEVICE, "--out-dir", str(out_dir)],
        check=True,
    )


def main() -> int:
    if not (MSST / "configs").is_dir():
        subprocess.run(["cp", "-r", str(MSST / "configs_backup"), str(MSST / "configs")], check=True)

    clips_dir = OUT / "clips"
    report: dict = {"device": DEVICE, "clip_s": CLIP_S, "tracks": {}}

    print(f"== MSST smoke (device={DEVICE}) ==")
    judge = _load_separator("bs_roformer", "model_bs_roformer_ep_368_sdr_12.9628.ckpt")

    for slug, fname in SMOKE_TRACKS:
        src = BB / fname
        if not src.exists():
            print(f"SKIP missing {src}")
            continue
        clip = clips_dir / f"{slug}.wav"
        if not clip.exists():
            print(f"clip {slug} …")
            _ffmpeg_clip(src, clip, CLIP_S)

        track_out = OUT / slug
        track_out.mkdir(parents=True, exist_ok=True)
        per_model = track_out / "per_model"
        per_model.mkdir(exist_ok=True)

        vocal_paths: list[Path] = []
        inst_paths: list[Path] = []
        timings: dict[str, float] = {}

        for model_type, ckpt in MODELS:
            tag = ckpt.replace(".ckpt", "")
            v_path = per_model / f"{tag}_vocals.wav"
            i_path = per_model / f"{tag}_instrumental.wav"
            if v_path.exists() and i_path.exists():
                print(f"  {slug} / {ckpt} (cached)", flush=True)
                timings[ckpt] = 0.0
                if (model_type, ckpt) in VOCAL_MODELS:
                    vocal_paths.append(v_path)
                if (model_type, ckpt) in INST_MODELS:
                    inst_paths.append(i_path)
                continue
            print(f"  {slug} / {ckpt} …", flush=True)
            sep = _load_separator(model_type, ckpt)
            t0 = time.monotonic()
            stems = _separate_file(sep, clip)
            sep.del_cache()
            timings[ckpt] = time.monotonic() - t0
            sf.write(str(v_path), stems["vocals"], 44100)
            sf.write(str(i_path), stems["instrumental"], 44100)
            if (model_type, ckpt) in VOCAL_MODELS:
                vocal_paths.append(v_path)
            if (model_type, ckpt) in INST_MODELS:
                inst_paths.append(i_path)

        print(f"  {slug} / ensemble …", flush=True)
        v_ens, sr = _ensemble(vocal_paths, "avg_fft")
        i_ens, _ = _ensemble(inst_paths, "avg_fft")
        rof_v = track_out / "roformer_vocals.wav"
        rof_i = track_out / "roformer_instrumental.wav"
        sf.write(str(rof_v), v_ens, sr)
        sf.write(str(rof_i), i_ens, sr)

        print(f"  {slug} / demucs …", flush=True)
        demucs_dir = track_out / "demucs"
        t0 = time.monotonic()
        _run_demucs(clip, demucs_dir)
        demucs_s = time.monotonic() - t0
        demucs_i = demucs_dir / "demucs/instrumental.flac"
        if demucs_i.exists():
            # soundfile may not read flac from torchaudio; convert
            demucs_i_wav = track_out / "demucs_instrumental.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(demucs_i), str(demucs_i_wav)],
                check=True, capture_output=True,
            )
            demucs_bleed = _bleed_score(demucs_i_wav, judge)
        else:
            demucs_bleed = float("nan")

        rof_bleed = _bleed_score(rof_i, judge)

        report["tracks"][slug] = {
            "source": str(src),
            "model_timings_s": timings,
            "demucs_s": demucs_s,
            "bleed_rms": {"roformer_inst_ensemble": rof_bleed, "demucs": demucs_bleed},
            "bleed_delta": rof_bleed - demucs_bleed,
        }
        print(f"  {slug} bleed roformer={rof_bleed:.6f} demucs={demucs_bleed:.6f} "
              f"delta={rof_bleed - demucs_bleed:+.6f}")

    report_path = OUT / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

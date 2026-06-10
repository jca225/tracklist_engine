# Future work: `stemsep` — max-quality ensemble vocal separator

**Status: not started — spec + environment recon persisted 2026-06-10.**
Deferred in favor of the in-flight RoFormer/MSST work on `analysis/`
(`vast_worker` separator, commit `c635d5b`). This would be a *second*,
quality-over-speed separation path, incubating in `workspaces/stemsep/`.

## Goal

Installable package `stemsep` doing 2-stem separation (acappella +
instrumental) via an **ensemble of Roformer-family models** through the
`audio-separator` package. Quality is the only priority — time/VRAM/disk
explicitly unconstrained.

## Spec summary

- **Ensemble members** (run all, blend):
  1. BS-Roformer viperx `model_bs_roformer_ep_317_sdr_12.9755.ckpt`
  2. Best Mel-Band Roformer vocal model from the registry (by published SDR)
  3. Best MDX23C vocal model as an architecturally-diverse third voter
  4. SCNet XL vocal if the installed audio-separator supports it
- **Blending**: spectral-magnitude domain; modes `avg` (weighted, default
  0.35/0.35/0.3), `max_mag`, `min_mag`. Instrumental via
  `instrumental_mode: subtract | direct` (default `subtract` =
  `mix - ensembled_vocals` waveform subtraction).
- **Quality settings**: FP32 only (no autocast), largest segment + generous
  overlap, TTA (polarity-invert run, invert back, average — ON by default),
  native SR when ≥44.1 kHz else soxr-VHQ upsample, true-peak-safe
  normalization only on overflow (logged).
- **Interface**: Python API (`Separator.separate(path, **opts) ->
  SeparationResult`), CLI (`stemsep separate input.flac --out ./stems
  --blend avg --tta`), dataclass config + optional YAML,
  `stemsep doctor` (GPU / ffmpeg / model cache / version check).
- **Engineering**: py3.11+, fully typed, pyproject.toml, CUDA→MPS→CPU
  auto-detect, checkpoint cache + checksums, CUDA-OOM retry with smaller
  segment, structured per-model timing logs, unit tests for ensemble math
  (synthetic signals) + env-gated integration test on a 10 s clip, README
  documenting model choices.

## Environment recon (done — valid as of 2026-06-10)

- `audio-separator==0.44.1` installed in **`venvs/audio`** (Python 3.14;
  note spec said 3.11+, fine). Not in `venvs/essentia` / `venvs/msst`.
- **No SCNet support** in 0.44.1's registry (`list_supported_model_files()`
  keys: `VR`, `MDX`, `Demucs`, `MDXC` only) → ensemble is 3 members.
- **Top vocal-SDR registry picks** (Roformers live under the `MDXC` arch):
  | Member | Filename | Vocal SDR |
  |---|---|---|
  | Mel-Band Roformer (Kim) | `vocals_mel_band_roformer.ckpt` | 12.60 |
  | (runner-up: Big Beta 4 FT by unwa) | `melband_roformer_big_beta4.ckpt` | 12.52 |
  | BS-Roformer viperx (mandated) | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 11.77 |
  | MDX23C | `MDX23C-8KFFT-InstVoc_HQ.ckpt` | 10.56 |
  Many gabox/becruily Roformers have **no published SDR** in the registry —
  runtime "pick highest SDR" must tolerate `scores: None`/missing keys, and
  some entries have `stems: []` or `['vocals','other']` (the "other" *is*
  the instrumental — handle stem-name variance).
- **API surface** (`audio_separator.separator.Separator`):
  - `Separator(model_file_dir=…, output_dir=…, output_format='WAV',
    mdxc_params={'segment_size', 'override_model_segment_size', 'batch_size',
    'overlap', 'pitch_shift'}, use_autocast=False, …)` then
    `load_model(filename)` then `separate(path, custom_output_names={stem:
    name})` → returns written file paths. No in-memory array API — get
    float32 arrays by writing to a temp dir and reading back with soundfile.
  - **Output bit depth mirrors the input subtype**
    (`common_separator.py:232-251`): feed a FLOAT WAV → float32 stems out.
    So the pipeline should decode/resample input to a temp FLOAT32 WAV first.
  - **MDXC/Roformer `overlap` is an int divisor** (`hop = chunk_size //
    overlap`, default 8) — *not* the 0–1 fraction MDX uses. "Generous
    overlap" here = overlap 8–16, plus
    `override_model_segment_size`/`segment_size` for chunk length.
  - FP32: just leave `use_autocast=False` (default).
- TTA, magnitude-domain blending, and waveform subtraction are **not**
  provided by audio-separator (its built-in `ensemble_*` params are a
  separate file-level ensemble feature) — implement them in `stemsep` on the
  read-back arrays.

## Suggested build order (from spec)

1. Scaffold `workspaces/stemsep/` package (pyproject, config dataclass).
2. Single-model end-to-end (decode → temp FLOAT32 WAV → separate → read
   back arrays).
3. Ensemble layer (STFT magnitude blend, both instrumental modes).
4. TTA wrapper.
5. CLI (`separate`, `doctor`) + tests + README.

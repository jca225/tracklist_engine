#!/usr/bin/env bash
# MSST RoFormer backend — venvs/msst + model checkpoints for workspaces/msst_webui.
#
# Usage:
#   scripts/setup_roformer_separation.sh
#   MSST_VENV=venvs/msst scripts/setup_roformer_separation.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MSST_VENV:-$REPO_ROOT/venvs/msst}/bin/python"
PIP="$PY -m pip"
MSST="$REPO_ROOT/workspaces/msst_webui"

echo "== repo: $REPO_ROOT"
echo "== msst python: $PY"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "!! install ffmpeg (brew install ffmpeg)"
  exit 1
fi

if [[ ! -d "$MSST" ]]; then
  echo "== cloning MSST-WebUI"
  git clone --depth 1 https://github.com/SUC-DriverOld/MSST-WebUI.git "$MSST"
fi

if [[ ! -x "$PY" ]]; then
  echo "== creating venvs/msst (python3.13)"
  python3.13 -m venv "${MSST_VENV:-$REPO_ROOT/venvs/msst}"
fi

echo "== installing MSST inference deps"
$PIP install -q --upgrade pip wheel
$PIP install -q torch torchaudio soundfile librosa scipy einops rotary_embedding_torch \
  tqdm safetensors omegaconf pyyaml numpy beartype ml_collections timm huggingface-hub neuraloperator

for sub in configs data; do
  if [[ ! -d "$MSST/$sub" && -d "$MSST/${sub}_backup" ]]; then
    cp -r "$MSST/${sub}_backup" "$MSST/$sub"
  fi
done

echo "== downloading pinned RoFormer checkpoints (~2.5 GB)"
"$PY" "$REPO_ROOT/workspaces/separation_qa/download_msst_models.py"

echo "== MPS smoke (30s BS-RoFormer clip)"
mkdir -p "$REPO_ROOT/workspaces/separation_qa/smoke_out/clips"
BB="$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/tracks"
CLIP="$REPO_ROOT/workspaces/separation_qa/smoke_out/clips/setup_smoke.wav"
if [[ -f "$BB/002__Manse - Freeze Time (AltVersion).m4a" ]]; then
  ffmpeg -y -i "$BB/002__Manse - Freeze Time (AltVersion).m4a" -t 15 -ar 44100 -ac 2 "$CLIP" >/dev/null 2>&1
  cd "$MSST"
  "$PY" -c "
import sys, time
from pathlib import Path
import librosa
sys.path.insert(0, '.')
from inference.msst_infer import MSSeparator
from utils.logger import get_logger
sep = MSSeparator(
    model_type='bs_roformer',
    config_path='configs/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt.yaml',
    model_path='pretrain/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    device='mps', output_format='wav',
    store_dirs={'vocals':'', 'instrumental':''},
    logger=get_logger())
mix, _ = librosa.load('$CLIP', mono=False, sr=44100)
t0 = time.time()
out = sep.separate(mix)
sep.del_cache()
print(f'OK device={sep.device} elapsed={time.time()-t0:.1f}s stems={list(out)}')
"
  echo "== setup complete"
else
  echo "== models ready (skip clip smoke — no ~/aligning BB12 tracks)"
fi

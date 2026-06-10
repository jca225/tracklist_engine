#!/usr/bin/env bash
# MSST RoFormer backend — venvs/msst + model checkpoints for workspaces/msst_webui.
#
# Host-detecting, idempotent:
#   - Mac          : dedicated venvs/msst (python3.13, MPS torch wheels)
#   - Vast.ai CUDA : reuses /venv/main's pre-built CUDA torch (NEVER reinstall
#                    torch over it — same rule as vast_bootstrap.sh) and
#                    symlinks venvs/msst -> /venv/main so invocation is uniform
#
# IMPORTANT: the adapter imports MSST IN-PROCESS (sys.path insert), so the
# CALLING interpreter needs the MSST deps. Drive roformer runs with
# venvs/msst/bin/python, NOT venvs/audio:
#   venvs/msst/bin/python scripts/separate.py --input x.m4a --separator roformer
#
# Vast: run AFTER vast_bootstrap.sh (needs repo + /venv/main):
#   cd /workspace/tracklist_engine && bash scripts/setup_roformer_separation.sh
#
# Usage:
#   scripts/setup_roformer_separation.sh
#   MSST_VENV=venvs/msst scripts/setup_roformer_separation.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSST="$REPO_ROOT/workspaces/msst_webui"
MSST_COMMIT="${MSST_COMMIT:-a48a17e}"   # validated locally 2026-06 (Mac smoke)

echo "== repo: $REPO_ROOT"

HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_CUDA=1
fi
echo "== host: $(uname -s) cuda=$HAS_CUDA"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "== installing ffmpeg (apt)"
    apt-get update -qq && apt-get install -y -qq ffmpeg
  else
    echo "!! install ffmpeg (brew install ffmpeg)"
    exit 1
  fi
fi

if [[ ! -d "$MSST" ]]; then
  echo "== cloning MSST-WebUI"
  git clone --depth 1 https://github.com/SUC-DriverOld/MSST-WebUI.git "$MSST"
fi
# Pin to the validated commit (GitHub allows fetching reachable SHAs).
if [[ "$(git -C "$MSST" rev-parse --short HEAD)" != "$MSST_COMMIT" ]]; then
  git -C "$MSST" fetch --quiet --depth 1 origin "$MSST_COMMIT" \
    && git -C "$MSST" checkout --quiet "$MSST_COMMIT" \
    || echo "!! could not pin $MSST_COMMIT — staying on $(git -C "$MSST" rev-parse --short HEAD)"
fi
echo "== msst: $(git -C "$MSST" log -1 --oneline)"

# --- interpreter --------------------------------------------------------------
if [[ "$HAS_CUDA" == "1" && -d /venv/main ]]; then
  PY=/venv/main/bin/python
  mkdir -p "$REPO_ROOT/venvs"
  [[ -e "$REPO_ROOT/venvs/msst" ]] || ln -sfn /venv/main "$REPO_ROOT/venvs/msst"
  echo "== venvs/msst -> /venv/main (Vast CUDA torch reused; torch NOT reinstalled)"
else
  VENV_DIR="${MSST_VENV:-$REPO_ROOT/venvs/msst}"
  PY="$VENV_DIR/bin/python"
  if [[ ! -x "$PY" ]]; then
    echo "== creating venvs/msst"
    "$(command -v python3.13 || command -v python3)" -m venv "$VENV_DIR"
  fi
  "$PY" -m pip install -q --upgrade pip wheel
  "$PY" -m pip install -q torch torchaudio   # default wheels = MPS on Mac
fi

echo "== installing MSST inference deps (requirements-msst.txt)"
"$PY" -m pip install -q -r "$REPO_ROOT/requirements-msst.txt"

for sub in configs data; do
  if [[ ! -d "$MSST/$sub" && -d "$MSST/${sub}_backup" ]]; then
    cp -r "$MSST/${sub}_backup" "$MSST/$sub"
  fi
done

echo "== downloading pinned RoFormer checkpoints (~2.5 GB)"
"$PY" "$REPO_ROOT/workspaces/separation_qa/download_msst_models.py"

# --- smoke: 15s BS-RoFormer clip, device-appropriate ---------------------------
DEVICE=$([[ "$HAS_CUDA" == "1" ]] && echo cuda || echo mps)
mkdir -p "$REPO_ROOT/workspaces/separation_qa/smoke_out/clips"
CLIP="$REPO_ROOT/workspaces/separation_qa/smoke_out/clips/setup_smoke.wav"
SMOKE_SRC="${SMOKE_SRC:-$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/tracks/002__Manse - Freeze Time (AltVersion).m4a}"
if [[ -f "$SMOKE_SRC" ]]; then
  ffmpeg -y -i "$SMOKE_SRC" -t 15 -ar 44100 -ac 2 "$CLIP" >/dev/null 2>&1
  cd "$MSST"
  "$PY" -c "
import sys, time
import librosa
sys.path.insert(0, '.')
from inference.msst_infer import MSSeparator
from utils.logger import get_logger
sep = MSSeparator(
    model_type='bs_roformer',
    config_path='configs/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt.yaml',
    model_path='pretrain/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    device='$DEVICE', output_format='wav',
    store_dirs={'vocals':'', 'instrumental':''},
    logger=get_logger())
mix, _ = librosa.load('$CLIP', mono=False, sr=44100)
t0 = time.time()
out = sep.separate(mix)
sep.del_cache()
print(f'OK device={sep.device} elapsed={time.time()-t0:.1f}s stems={list(out)}')
"
  echo "== setup complete ($DEVICE smoke passed)"
else
  echo "== models ready (skip clip smoke — set SMOKE_SRC=/path/to/audio to enable)"
  echo "   full smoke: venvs/msst/bin/python scripts/separate.py --input x.m4a --separator roformer"
fi

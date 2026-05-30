#!/usr/bin/env bash
# Set up the UVR stem-separation backend (audio-separator) in venvs/audio.
#
# Host-agnostic: installs ffmpeg + audio-separator with the right onnxruntime
# variant for the host (CUDA hosts → onnxruntime-gpu; Mac/CPU → bundled CPU),
# pre-downloads the 6 chain models into the persistent cache, verifies the
# execution provider, and runs a tiny smoke separation.
#
# Usage:
#   scripts/setup_separation.sh                 # auto-detect host
#   MODEL_DIR=~/uvr-models scripts/setup_separation.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/venvs/audio/bin/python}"
PIP="$PY -m pip"
MODEL_DIR="${MODEL_DIR:-$HOME/uvr-models}"

echo "== repo: $REPO_ROOT"
echo "== python: $PY"
echo "== model cache: $MODEL_DIR"

# --- ffmpeg -----------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "== installing ffmpeg"
  if   command -v apt-get >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y ffmpeg
  elif command -v brew    >/dev/null 2>&1; then brew install ffmpeg
  else echo "!! install ffmpeg manually (no apt/brew found)"; exit 1
  fi
fi

# --- GPU detection ----------------------------------------------------------
HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_CUDA=1
  echo "== NVIDIA GPU detected → installing audio-separator[gpu] (onnxruntime-gpu)"
  $PIP install --upgrade "audio-separator[gpu]>=0.44.1"
else
  echo "== no CUDA GPU → installing audio-separator[cpu] (Mac uses MPS for torch models)"
  $PIP install --upgrade "audio-separator[cpu]>=0.44.1"
fi

# --- pre-download chain models ---------------------------------------------
echo "== pre-downloading chain models into $MODEL_DIR"
mkdir -p "$MODEL_DIR"
for m in Kim_Vocal_2.onnx UVR_MDXNET_KARA_2.onnx 5_HP-Karaoke-UVR.pth \
         Reverb_HQ_By_FoxJoy.onnx UVR-De-Echo-Aggressive.pth UVR-DeNoise.pth; do
  echo "  - $m"
  "$REPO_ROOT/venvs/audio/bin/audio-separator" \
    --model_file_dir "$MODEL_DIR" --download_model_only -m "$m" >/dev/null
done

# --- verify execution provider ---------------------------------------------
echo "== verifying onnxruntime execution provider"
$PY - "$HAS_CUDA" <<'PY'
import sys
has_cuda = sys.argv[1] == "1"
import onnxruntime as ort
providers = ort.get_available_providers()
print("   onnxruntime providers:", providers)
gpu_ok = "CUDAExecutionProvider" in providers
if has_cuda and not gpu_ok:
    print("!! CUDA GPU present but onnxruntime has no CUDAExecutionProvider — "
          "the UVR backend would silently run on CPU. Install onnxruntime-gpu "
          "matching your CUDA/cuDNN.", file=sys.stderr)
    sys.exit(1)
print("   provider check OK" + ("" if has_cuda else " (CPU/CoreML expected on this host)"))
PY

echo
echo "== done. Smoke-test a real file with:"
echo "   $PY scripts/separate.py --input <song> --separator uvr --byproducts"

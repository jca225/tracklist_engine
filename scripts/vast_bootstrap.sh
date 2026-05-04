#!/usr/bin/env bash
# Bootstrap a Vast.ai instance for Demucs / MERT / Essentia heavy analysis.
#
# Tailored for the PyTorch (Vast) image, which ships:
#   - /venv/main with Python 3.14 + torch + torchaudio + torchcodec
#     pre-built against CUDA 13.0
#   - apt-installed ffmpeg, but NOT libsndfile / build-essential
#   - empty /workspace (overlay disk, ~32 GB total)
#
# This script:
#   1. apt-installs missing system deps (libsndfile, build tools)
#   2. clones the repo into /workspace/tracklist_engine
#   3. installs the missing pip deps INTO /venv/main (don't make a new venv —
#      Vast's pre-installed torch+CUDA build is non-trivial to reproduce)
#   4. creates /venv/essentia (Py3.13) for the Essentia TF stack
#   5. fetches the 9 Essentia .pb model files (~40 MB)
#   6. installs Tailscale in userspace mode for pi-storage connectivity
#
# Run as root after `ssh vast` succeeds:
#   curl -fsSL https://raw.githubusercontent.com/jca225/tracklist_engine/main/scripts/vast_bootstrap.sh | bash
#
# Time budget: ~5 min for pip installs, ~30s for Tailscale, ~15s for models.
# Disk after run: ~3 GB used of 32 GB overlay (most is the /venv/main image).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jca225/tracklist_engine.git}"
REPO_DIR="${REPO_DIR:-/workspace/tracklist_engine}"
ESSENTIA_PYBIN="${ESSENTIA_PYBIN:-/usr/bin/python3.13}"

echo "==> [1/7] apt: ffmpeg + libsndfile + build tools"
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1 build-essential pkg-config \
    git rsync curl ca-certificates >/dev/null
echo "    ffmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"

echo "==> [2/7] Repo clone / update"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"
echo "    HEAD: $(git -C "$REPO_DIR" log -1 --oneline)"

echo "==> [3/7] Verify /venv/main GPU stack"
/venv/main/bin/python - <<'PY'
import torch, sys
print(f"    torch        : {torch.__version__}")
print(f"    cuda         : {torch.version.cuda}")
print(f"    device       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"    capability   : sm_{cap[0]}{cap[1]}")
assert torch.cuda.is_available(), "CUDA unavailable — wrong template?"
PY

echo "==> [4/7] Install audio pipeline deps INTO /venv/main"
# Don't reinstall torch/torchaudio/torchcodec — they're pre-built against
# this exact CUDA driver. Adding the rest on top.
/venv/main/bin/pip install --quiet --no-deps \
    "demucs>=4.0.1"
/venv/main/bin/pip install --quiet \
    "yt-dlp>=2026.3.17" "spotdl>=3.9.6" "spotipy>=2.26.0" \
    "librosa>=0.11" "pyloudnorm>=0.2" "soundfile>=0.13" \
    "beat-this>=1.1" "transformers>=4.57" "timm>=1.0"
echo "    /venv/main now has audio pipeline deps"

echo "==> [5/7] Create /venv/essentia (Py3.13, separate venv to isolate TF)"
if [ ! -d /venv/essentia ]; then
    "$ESSENTIA_PYBIN" -m venv /venv/essentia
fi
/venv/essentia/bin/pip install --quiet --upgrade pip
/venv/essentia/bin/pip install --quiet -r requirements-essentia.txt
echo "    essentia venv: $(/venv/essentia/bin/python --version 2>&1)"

echo "==> [6/7] Download Essentia .pb models (~40 MB)"
PYTHONPATH="$REPO_DIR" /venv/essentia/bin/python - <<'PY'
from audio_pipeline.analysis.adapters import essentia_models as em
report = em.ensure_downloaded()
print(f"    downloaded={len(report.downloaded)} skipped={len(report.skipped)} failed={len(report.failed)}")
if report.failed:
    for f in report.failed:
        print(f"      FAILED {f.name}: {f.reason}")
    raise SystemExit(1)
PY

echo "==> [7/7] Tailscale (userspace mode — no kernel/TUN required)"
if ! command -v tailscale >/dev/null 2>&1; then
    curl -fsSL https://tailscale.com/install.sh | sh >/dev/null
fi
# Userspace networking: Vast containers can't open /dev/net/tun, so we
# use socks5/http proxy mode. Tailscale routes through that.
mkdir -p /var/run/tailscale
nohup tailscaled \
    --tun=userspace-networking \
    --socks5-server=localhost:1055 \
    --outbound-http-proxy-listen=localhost:1055 \
    > /var/log/tailscaled.log 2>&1 &
sleep 2
echo "    tailscaled started in userspace mode (socks5: localhost:1055)"
echo "    *** RUN ON THIS BOX TO COMPLETE TAILSCALE SETUP ***"
echo "    tailscale up --auth-key=tskey-auth-... --hostname=vast-mert"
echo "    (auth key from https://login.tailscale.com/admin/settings/keys)"
echo "    After 'tailscale up', test: ALL_PROXY=socks5://localhost:1055 ssh pi-storage 'hostname'"

echo
echo "==> READY"
echo "    Repo      : $REPO_DIR"
echo "    Audio venv: /venv/main/bin/python  (torch + cuda + analysis deps)"
echo "    Essentia  : /venv/essentia/bin/python  (separate, no torch conflict)"
echo "    Models    : $REPO_DIR/data/essentia_models/  (9 .pb files, ~40 MB)"
echo
echo "Smoke test (no audio yet — just imports):"
echo "    cd $REPO_DIR && /venv/main/bin/pytest tests/test_essentia_adapter.py -v"

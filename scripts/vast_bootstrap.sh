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

# Essentia ships cp310-cp313 wheels for x86_64 Linux. Pick whatever Python
# the host actually has — Vast images vary (some have 3.13, some 3.12,
# some 3.11). Fail loudly if none is available.
if [ -z "${ESSENTIA_PYBIN:-}" ]; then
    for v in python3.13 python3.12 python3.11 python3.10; do
        if command -v "$v" >/dev/null 2>&1; then
            ESSENTIA_PYBIN="$(command -v "$v")"
            break
        fi
    done
fi
if [ -z "${ESSENTIA_PYBIN:-}" ]; then
    echo "ERROR: no python3.{10,11,12,13} found on this host. Essentia wheels"
    echo "       only exist for those interpreters. Install via apt then re-run."
    exit 1
fi

echo "==> [1/7] apt: ffmpeg + nodejs + libsndfile + build tools"
apt-get update -qq
# nodejs needed for yt-dlp's n-challenge JS runtime (see ingest/
# adapters/downloader.py — without it ~all YouTube videos return only
# image formats and downloads fail with "Signature solving failed").
apt-get install -y -qq ffmpeg nodejs libsndfile1 build-essential pkg-config \
    git rsync curl ca-certificates >/dev/null
echo "    ffmpeg : $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
echo "    node   : $(node --version 2>/dev/null || echo MISSING)"

echo "==> [2/7] Repo clone / update"
if [ "${SKIP_CLONE:-0}" = "1" ]; then
    echo "    SKIP_CLONE=1 — assuming repo is already at $REPO_DIR (e.g. rsync'd from Mac)"
elif [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"
if [ -d .git ]; then
    echo "    HEAD: $(git -C "$REPO_DIR" log -1 --oneline)"
else
    echo "    (no .git — running from rsync'd snapshot)"
fi

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
# this exact CUDA driver. Install demucs with --no-deps so its torch pin
# doesn't downgrade the CUDA build, then explicitly add demucs's other
# deps (dora-search, julius, lameenc, openunmix).
/venv/main/bin/pip install --quiet --no-deps "demucs>=4.0.1"
/venv/main/bin/pip install --quiet \
    "dora-search" "julius>=0.2.3" "lameenc>=1.2" "openunmix"
/venv/main/bin/pip install --quiet \
    "yt-dlp>=2026.5.0" "yt-dlp-ejs" "spotdl>=3.9.6" "spotipy>=2.26.0" \
    "librosa>=0.11" "pyloudnorm>=0.2" "soundfile>=0.13" \
    "beat-this>=1.1" "transformers>=4.57" "timm>=1.0" \
    "matplotlib"   # cue-detr/cue_points.py imports this at module level
echo "    /venv/main now has audio pipeline deps"

echo "==> [5/7] Create /venv/essentia (Py3.10-3.13, separate venv to isolate TF)"
if [ ! -d /venv/essentia ]; then
    "$ESSENTIA_PYBIN" -m venv /venv/essentia
fi
/venv/essentia/bin/pip install --quiet --upgrade pip
/venv/essentia/bin/pip install --quiet -r requirements-essentia.txt
echo "    essentia venv: $(/venv/essentia/bin/python --version 2>&1)"

# essentia_adapter.py looks at <repo>/venvs/essentia/bin/python; symlink so
# the path it expects resolves on Vast (where we keep envs at /venv/...).
mkdir -p "$REPO_DIR/venvs"
ln -sfn /venv/essentia "$REPO_DIR/venvs/essentia"
echo "    symlink: $REPO_DIR/venvs/essentia -> /venv/essentia"

echo "==> [6/7] Download Essentia .pb models (~40 MB)"
PYTHONPATH="$REPO_DIR" /venv/essentia/bin/python - <<'PY'
from analysis.adapters import essentia_models as em
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

# Userspace mode gives no direct outbound to tailnet IPs — plain
# `ssh pi-storage.tail116c2d.ts.net` fails on no-TUN boxes (hit 2026-06-10).
# Route ssh/sshfs through the socks5 proxy, pinned to pi's tailnet IP so
# MagicDNS isn't needed either. vast_run.sh's checks then pass unmodified.
apt-get install -y -qq netcat-openbsd >/dev/null
PI_TS_IP="${PI_TS_IP:-100.103.219.39}"
PI_TS_USER="${PI_TS_USER:-johncabrahams}"
mkdir -p ~/.ssh
if ! grep -q "Host pi-storage.tail116c2d.ts.net" ~/.ssh/config 2>/dev/null; then
    cat >> ~/.ssh/config <<EOF
Host pi-storage pi-storage.tail116c2d.ts.net
    HostName ${PI_TS_IP}
    User ${PI_TS_USER}
    ProxyCommand nc -X 5 -x localhost:1055 %h %p
    StrictHostKeyChecking accept-new
EOF
    chmod 600 ~/.ssh/config
    echo "    ssh config: pi-storage.tail116c2d.ts.net via socks5 proxy"
fi

echo "    *** RUN ON THIS BOX TO COMPLETE TAILSCALE SETUP ***"
echo "    tailscale up --hostname=vast-roformer   (prints a login URL to click)"
echo "    After 'tailscale up', test: ssh pi-storage.tail116c2d.ts.net 'hostname'"

echo
echo "==> READY"
echo "    Repo      : $REPO_DIR"
echo "    Audio venv: /venv/main/bin/python  (torch + cuda + analysis deps)"
echo "    Essentia  : /venv/essentia/bin/python  (separate, no torch conflict)"
echo "    Models    : $REPO_DIR/data/essentia_models/  (9 .pb files, ~40 MB)"
echo
echo "Smoke test (no audio yet — just imports):"
echo "    cd $REPO_DIR && /venv/main/bin/pytest tests/test_essentia_adapter.py -v"

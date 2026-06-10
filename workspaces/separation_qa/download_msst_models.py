"""Download pinned MSST RoFormer checkpoints into workspaces/msst_webui/pretrain/."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MSST = REPO / "workspaces" / "msst_webui"
MODELS_INFO = MSST / "data_backup" / "models_info.json"

# Pinned smoke/benchmark models (vocal_models, vocals+instrumental 2-stem).
PINNED = (
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "kimmel_unwa_ft.ckpt",  # third vocal-ensemble head (mel_band_roformer)
    "melband_roformer_inst_v2.ckpt",  # third inst-ensemble head, instrumental-specialized (unwa Inst V2)
)


def main() -> int:
    data = json.loads(MODELS_INFO.read_text())
    for name in PINNED:
        entry = data[name]
        dest = MSST / entry["target_position"].lstrip("./")
        if dest.exists() and dest.stat().st_size == entry["model_size"]:
            print(f"skip {name} (already present)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = entry["link"]
        print(f"download {name} ({entry['model_size'] / 1e6:.0f} MB) …")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

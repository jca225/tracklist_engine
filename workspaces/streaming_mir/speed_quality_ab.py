"""Separation speed/quality A/B: ensemble passes x precision (streaming_mir).

Motivated by the 2026-07-15 deep-research verdicts (sota_speed_research_20260715.md):
no published drop-in faster model exists at our quality bar, so the levers are
in-house — (a) fewer ensemble passes (the chain dedupes shared checkpoints, so
the shipped "5-model" config is really 3 forward passes; 2- and 1-pass variants
are pure config) and (b) bf16/fp16 autocast, which is also the only way SDPA's
flash-attention kernels can dispatch (they are half-precision-only).

Arms (reference first):
  p3_fp32   shipped chain (bs+mel+kimmel vocals / bs+mel instr), fp32  [REFERENCE]
  p2_fp32   drop kimmel — vocals bs+mel, instr bs+mel (2 passes)
  p1_fp32   bs_roformer only, both stems (1 pass)
  p3_bf16   shipped chain, bf16 autocast
  p3_fp16   shipped chain, fp16 autocast
  p1_fp16   max-speed combo

Gates — two different bars, do not conflate:
  * precision arms (same models): near-identical bar, >=38 dB SDR vs p3_fp32
    per stem on real music (same bar the batching change passed).
  * pass-reduction arms (different ensemble): expected to differ audibly-ish
    in SDR terms; the deploy gate is alignment-scorer invariance on BB11/BB12,
    NOT this script. This script only reports their SDR + speedup so we know
    whether the scorer test is worth running.

Run on a CUDA box (models under workspaces/msst_webui/pretrain):
  /venv/main/bin/python workspaces/streaming_mir/speed_quality_ab.py \
      --inputs /workspace/ab_tracks/*.m4a --out /workspace/speed_quality_ab.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.adapters import roformer_chain_adapter  # noqa: E402
from analysis.roformer_config import RoformerChainConfig  # noqa: E402
from workspaces.streaming_mir.seam_check import sdr  # noqa: E402

REFERENCE_ARM = "p3_fp32"


def build_arms(base: RoformerChainConfig) -> dict[str, RoformerChainConfig]:
    """Arm name -> config. base = the shipped chain (3 unique passes)."""
    bs_only = tuple(m for m in base.vocal_models if m.model_type == "bs_roformer")
    no_kimmel = tuple(m for m in base.vocal_models if "kimmel" not in m.ckpt)
    p2 = replace(base, vocal_models=no_kimmel)
    p1 = replace(base, vocal_models=bs_only, instrumental_models=bs_only)
    return {
        "p3_fp32": base,
        "p2_fp32": p2,
        "p1_fp32": p1,
        "p3_bf16": replace(base, precision="bf16"),
        "p3_fp16": replace(base, precision="fp16"),
        "p1_fp16": replace(p1, precision="fp16"),
    }


def run_arm(
    name: str,
    cfg: RoformerChainConfig,
    inputs: list[Path],
    out_root: Path,
    device: str,
) -> dict:
    load_r = roformer_chain_adapter.load(cfg, device=device)
    if not load_r.is_ok():
        return {"arm": name, "error": f"{load_r.error.kind}: {load_r.error.detail}"}
    handle = load_r.value
    tracks = []
    for i, audio in enumerate(inputs):
        dest = out_root / name
        t0 = time.monotonic()
        r = roformer_chain_adapter.separate(handle, audio, dest, track_audio_id=i)
        wall = time.monotonic() - t0
        if not r.is_ok():
            tracks.append({"input": audio.name, "error": r.error.detail})
            continue
        tracks.append(
            {"input": audio.name, "wall_s": round(wall, 1), "dir": str(dest / str(i))}
        )
        print(f"  {name} {audio.name}: {wall:.1f}s", flush=True)
    return {
        "arm": name,
        "version": cfg.version,
        "precision": cfg.precision,
        "tracks": tracks,
    }


def stem_sdr_vs_ref(arm_dir: Path, ref_dir: Path) -> dict[str, float]:
    out = {}
    for stem in ("vocals", "instrumental"):
        est, _ = sf.read(arm_dir / f"{stem}.flac")
        ref, _ = sf.read(ref_dir / f"{stem}.flac")
        n = min(len(est), len(ref))
        out[stem] = round(float(sdr(ref[:n], est[:n])), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("speed_quality_ab.json"))
    ap.add_argument("--scratch", type=Path, default=Path("/workspace/sq_ab_out"))
    args = ap.parse_args()

    base = RoformerChainConfig.default()
    arms = build_arms(base)
    results = {}
    for name, cfg in arms.items():
        print(f"== arm {name} ({cfg.version}, {cfg.precision})", flush=True)
        results[name] = run_arm(name, cfg, args.inputs, args.scratch, args.device)

    # SDR of every non-reference arm vs the reference, per input
    ref = results[REFERENCE_ARM]
    for name, res in results.items():
        if name == REFERENCE_ARM or "error" in res:
            continue
        for tr, rtr in zip(res["tracks"], ref["tracks"]):
            if "dir" in tr and "dir" in rtr:
                tr["sdr_vs_ref"] = stem_sdr_vs_ref(Path(tr["dir"]), Path(rtr["dir"]))

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")

    ref_wall = [t["wall_s"] for t in ref["tracks"] if "wall_s" in t]
    print(
        f"\n{'arm':10s} {'mean wall':>10s} {'speedup':>8s}  sdr_vs_ref (voc/inst per track)"
    )
    for name, res in results.items():
        walls = [t["wall_s"] for t in res.get("tracks", []) if "wall_s" in t]
        if not walls:
            print(f"{name:10s}  ERROR {res.get('error', '')}")
            continue
        mw = sum(walls) / len(walls)
        sp = (sum(ref_wall) / len(ref_wall)) / mw if ref_wall else float("nan")
        sdrs = " ".join(
            f"{t['sdr_vs_ref']['vocals']:.0f}/{t['sdr_vs_ref']['instrumental']:.0f}"
            for t in res["tracks"]
            if "sdr_vs_ref" in t
        )
        print(f"{name:10s} {mw:9.1f}s {sp:7.2f}x  {sdrs}")
    print(
        "\ngates: precision arms need >=38 dB/stem vs p3_fp32 (near-identical);"
        "\n       pass-reduction arms need the BB11/BB12 alignment-scorer check"
        " before any deploy decision."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

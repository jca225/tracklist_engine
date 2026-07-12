"""WS2 anchor experiment: how much block overlap recovers offline-SOTA SDR.

The corpus separates long DJ sets by hard-cutting the mix into fixed 360s chunks
and concatenating (scripts/render_set_stems.py) — no overlap, "faint seams at
joins". Those seams ARE the gap to offline SOTA. This script quantifies it:

  reference   = full-file RoFormer separation (the offline SOTA anchor)
  block-online = tile the file into `core`-length blocks, but feed each block
                 `margin` seconds of extra context on BOTH sides (lookahead +
                 look-back), keep only the core region, concatenate.

`margin=0` reproduces today's hard-cut behaviour. Sweeping `margin` upward and
scoring SDR-vs-reference gives the Pareto point: the smallest overlap that
recovers offline SDR (Δ≈0). Design law: the converged output must equal offline
SOTA — so we measure gap-to-offline, NOT gap-to-ground-truth-stems (which we
don't have). No external GT stems required.

Run on a separation-capable host (Mac MPS with venvs/msst, or a CUDA box):

    venvs/audio/bin/python workspaces/streaming_mir/block_overlap_sweep.py \
        --audio /path/to/reference_track.wav --device mps \
        --core-sec 30 --margins 0,0.5,1,2,5,10

A ~4-min reference TRACK (not a 62-min set) is the right subject: full-file fits
in memory, the sweep stays cheap, and the overlap finding transfers directly to
the long-set chunker.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import os  # noqa: E402

os.environ.setdefault("TRACKLIST_DISABLE_FK", "1")

from analysis.adapters import roformer_chain_adapter  # noqa: E402
from analysis.roformer_config import RoformerChainConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("block_overlap_sweep")

STEMS = ("vocals", "instrumental")


@dataclass(frozen=True)
class Stems:
    """Two same-length (samples, channels) float arrays at a common sample rate."""

    vocals: np.ndarray
    instrumental: np.ndarray
    sr: int

    def get(self, name: str) -> np.ndarray:
        return self.vocals if name == "vocals" else self.instrumental


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Global SDR in dB. Trims to common length; flattens channels."""
    n = min(len(ref), len(est))
    r = ref[:n].reshape(-1).astype(np.float64)
    e = est[:n].reshape(-1).astype(np.float64)
    num = float(np.sum(r * r)) + 1e-12
    den = float(np.sum((r - e) ** 2)) + 1e-12
    return 10.0 * np.log10(num / den)


def sdr_at_boundaries(ref: np.ndarray, est: np.ndarray, hop: int, window: int) -> float:
    """SDR computed ONLY within ±`window` samples of each interior core boundary.

    Global SDR is dominated by the non-seam bulk (near-identical to the offline
    reference), which masks the localized seam damage the overlap is meant to
    fix. Boundary-local SDR is the sensitive metric: it isolates exactly the
    regions where a zero-context block cut degrades quality. Boundaries sit at
    k*hop for k=1..n_blocks-1.
    """
    n = min(len(ref), len(est))
    if hop <= 0 or hop >= n:
        return float("nan")
    mask = np.zeros(n, dtype=bool)
    for b in range(hop, n, hop):
        mask[max(0, b - window) : min(n, b + window)] = True
    if not mask.any():
        return float("nan")
    r = ref[:n][mask].reshape(-1).astype(np.float64)
    e = est[:n][mask].reshape(-1).astype(np.float64)
    num = float(np.sum(r * r)) + 1e-12
    den = float(np.sum((r - e) ** 2)) + 1e-12
    return 10.0 * np.log10(num / den)


def _separate_array(handle, audio: np.ndarray, sr: int, tmp: Path, tag: int) -> Stems:
    """Write `audio` to a wav, run the production RoFormer adapter, read stems back."""
    in_path = tmp / f"in_{tag}.wav"
    sf.write(in_path, audio, sr, subtype="PCM_16")
    out_dir = tmp / f"out_{tag}"
    r = roformer_chain_adapter.separate(handle, in_path, out_dir, track_audio_id=tag)
    if not r.is_ok():
        raise RuntimeError(f"separate failed: {r.error.kind} — {r.error.detail}")
    leaf = out_dir / str(tag)
    voc, _ = sf.read(leaf / "vocals.flac", always_2d=True)
    inst, sr_o = sf.read(leaf / "instrumental.flac", always_2d=True)
    return Stems(vocals=voc, instrumental=inst, sr=sr_o)


def separate_full(handle, audio: np.ndarray, sr: int, tmp: Path) -> Stems:
    """Offline SOTA reference: separate the whole file in one pass."""
    return _separate_array(handle, audio, sr, tmp, tag=0)


def separate_blocked(
    handle, audio: np.ndarray, sr: int, core_sec: float, margin_sec: float, tmp: Path
) -> tuple[Stems, int]:
    """Block-online reassembly: core tiles with `margin`s of two-sided context.

    Each core region [k*hop, (k+1)*hop] is separated inside a padded window
    [k*hop - margin, (k+1)*hop + margin] (clamped at file edges), then only the
    core slice is kept. margin=0 == today's hard-cut chunking.
    """
    n = len(audio)
    hop = max(1, int(round(core_sec * sr)))
    margin = int(round(margin_sec * sr))
    ch = audio.shape[1] if audio.ndim == 2 else 1
    out_voc = np.zeros((n, ch), dtype=np.float32)
    out_inst = np.zeros((n, ch), dtype=np.float32)

    n_blocks = 0
    for start in range(0, n, hop):
        core_end = min(start + hop, n)
        win_start = max(0, start - margin)
        win_end = min(n, core_end + margin)
        block = audio[win_start:win_end]
        st = _separate_array(handle, block, sr, tmp, tag=1000 + n_blocks)
        off = start - win_start  # core offset inside the separated window
        length = core_end - start
        out_voc[start:core_end] = st.vocals[off : off + length]
        out_inst[start:core_end] = st.instrumental[off : off + length]
        n_blocks += 1

    return Stems(vocals=out_voc, instrumental=out_inst, sr=sr), n_blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audio", type=Path, required=True, help="reference track (wav/flac)"
    )
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--core-sec", type=float, default=30.0, help="core tile length (s)")
    ap.add_argument(
        "--margins",
        default="0,0.5,1,2,5,10",
        help="comma-separated overlap margins in seconds to sweep",
    )
    ap.add_argument(
        "--boundary-win-sec",
        type=float,
        default=2.0,
        help="±window (s) around each core cut for boundary-local SDR",
    )
    ap.add_argument(
        "--single-model",
        action="store_true",
        help="use only the strongest single model (bs_roformer) for BOTH stems "
        "— one model pass per block instead of the 3-model ensemble. ~3x faster; "
        "the overlap-heals-seams effect is model-agnostic so this is fine for the "
        "principle. Drop it for the full-ensemble confirmation run (on a real GPU).",
    )
    args = ap.parse_args()

    if not args.audio.exists():
        log.error("audio not found: %s", args.audio)
        return 1
    margins = [float(x) for x in args.margins.split(",")]

    audio, sr = sf.read(args.audio, always_2d=True)
    audio = audio.astype(np.float32)
    log.info(
        "loaded %s: %.1fs @ %dHz, %d ch",
        args.audio.name,
        len(audio) / sr,
        sr,
        audio.shape[1],
    )

    cfg = RoformerChainConfig.default()
    if args.single_model:
        import dataclasses

        top = cfg.vocal_models[:1]  # bs_roformer, SDR 12.96 — emits both stems
        cfg = dataclasses.replace(cfg, vocal_models=top, instrumental_models=top)
        log.info("single-model mode: %s (1 pass/block)", top[0].tag)

    log.info("loading RoFormer (device=%s)…", args.device)
    lr = roformer_chain_adapter.load(cfg, device=args.device)
    if not lr.is_ok():
        log.error("load failed: %s — %s", lr.error.kind, lr.error.detail)
        return 1
    handle = lr.value

    with tempfile.TemporaryDirectory(prefix="ws2_sweep_") as td:
        tmp = Path(td)

        log.info("=== reference: full-file separation (offline SOTA anchor) ===")
        t0 = time.monotonic()
        ref = separate_full(handle, audio, sr, tmp)
        log.info("reference separated in %.0fs", time.monotonic() - t0)

        hop = max(1, int(round(args.core_sec * sr)))
        bwin = int(round(args.boundary_win_sec * sr))
        print(
            f"\n{'margin_s':>9} {'n_blk':>6} "
            f"{'voc_glob':>9} {'inst_glob':>10} "
            f"{'voc_bnd':>8} {'inst_bnd':>9} {'wall_s':>7}"
        )
        print("-" * 62)
        for m in margins:
            t0 = time.monotonic()
            blocked, nb = separate_blocked(handle, audio, sr, args.core_sec, m, tmp)
            vg = sdr(ref.vocals, blocked.vocals)
            ig = sdr(ref.instrumental, blocked.instrumental)
            vb = sdr_at_boundaries(ref.vocals, blocked.vocals, hop, bwin)
            ib = sdr_at_boundaries(ref.instrumental, blocked.instrumental, hop, bwin)
            wall = time.monotonic() - t0
            print(
                f"{m:>9.2f} {nb:>6d} {vg:>9.2f} {ig:>10.2f} "
                f"{vb:>8.2f} {ib:>9.2f} {wall:>7.0f}"
            )

    print("\nInterpretation: SDR is vs the full-file offline output (higher = closer")
    print("to SOTA). The *_bnd columns (boundary-local) are the sensitive metric —")
    print("global SDR is diluted by the identical non-seam bulk. Find the smallest")
    print("margin where boundary-local SDR plateaus: that's the overlap the long-set")
    print("chunker (render_set_stems.py) needs to hit offline quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

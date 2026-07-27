"""Render a predicted timeline back to audio for perceptual A/B against the mix.

The aligner emits ``out/<set_id>_predicted_timeline.json`` (a list of span
predictions: which reference plays, where in the mix, where in the reference,
at what stretch). This module places the *clean reference tracks* at their
predicted positions and sums them into a single mono render, so a human can
A/B the reconstruction against the original set mix.

Validation principle (see the alignment plan): reconstruct from clean refs, NOT
from the mix's own stems -- otherwise the test is circular. If the reconstruction
is perceptually indistinguishable from the mix at the *transitions*, the
placement/warp is right regardless of exact label error. The ear is blind to two
things this renderer cannot fix: identity swaps that sound alike, and omitted
low-energy layers -- both minor for live sets, real for studio mashups.

Usage:
    python -m alignment.render_timeline --set-id 1fsnxchk
    python -m alignment.render_timeline --set-id 1fsnxchk --ab

The set's mix + reference tracks + stems must be staged in
``~/aligning/<set_id>__<title>/`` (via the alignment-pull skill), with a
``manifest.json`` mapping slot ``label`` -> local audio path.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 44_100
FADE_S = 0.010  # raised-cosine edge fade to suppress placement clicks
MIN_RATE, MAX_RATE = 0.25, 4.0
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- data model


@dataclass(frozen=True)
class Segment:
    """One contiguous stretch of reference audio placed in the mix."""

    mix_start_s: float
    mix_end_s: float
    ref_start_s: float
    ref_end_s: float


# ---------------------------------------------------------------- pure helpers


def _norm_slot(label: str) -> str:
    """'001' -> '1', '013w2' -> '13w2'. Joins timeline slot_label to manifest label."""
    mo = re.match(r"0*(\d+)(w\d+)?$", str(label))
    return (mo.group(1) + (mo.group(2) or "")) if mo else str(label)


def _stem_source(track: dict, claimed_stem: str | None) -> Path | None:
    """Locate the reference file for a span, stem-aware.

    acappella -> demucs/roformer vocals, instrumental -> instrumental stem,
    else the full track. Returns None if nothing on disk.
    """
    stems = track.get("stems") or {}
    key = {"acappella": "vocals", "instrumental": "instrumental"}.get(
        claimed_stem or ""
    )
    if key:
        p = stems.get(key)
        if p and Path(p).is_file():
            return Path(p)
    p = track.get("local_path")
    if p and Path(p).is_file():
        return Path(p)
    return None


def _segments_for(span: dict) -> list[Segment]:
    """Explode a span into placed segments (loops/multiseg -> many; else one)."""
    set_start = float(span["set_start_s"])
    set_end = float(span["set_end_s"])
    raw = span.get("ref_segments") or []
    if not raw:
        return [
            Segment(
                set_start, set_end, float(span["ref_start_s"]), float(span["ref_end_s"])
            )
        ]
    segs: list[Segment] = []
    for i, s in enumerate(raw):
        mix_start = float(s["mix_start_s"])
        # a segment runs until the next segment starts, or the span ends
        mix_end = float(raw[i + 1]["mix_start_s"]) if i + 1 < len(raw) else set_end
        if mix_end <= mix_start:
            continue
        segs.append(
            Segment(mix_start, mix_end, float(s["ref_start_s"]), float(s["ref_end_s"]))
        )
    return segs


def _fade(y: np.ndarray, sr: int = SR, fade_s: float = FADE_S) -> np.ndarray:
    """Raised-cosine fade in/out to avoid clicks at clip boundaries."""
    n = min(int(fade_s * sr), len(y) // 2)
    if n <= 0:
        return y
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    y = y.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def _stretch_to(src: np.ndarray, out_len: int, sr: int = SR) -> np.ndarray:
    """Time-stretch a mono ref slice to exactly out_len samples (phase vocoder)."""
    if len(src) == 0 or out_len <= 0:
        return np.zeros(max(out_len, 0), dtype=np.float32)
    rate = len(src) / out_len  # >1 speeds up (shorter), matches librosa
    if abs(rate - 1.0) < 0.01:
        y = src
    elif MIN_RATE <= rate <= MAX_RATE:
        y = librosa.effects.time_stretch(src, rate=rate)
    else:
        # out of sane range: fall back to naive resample so we still place *something*
        idx = np.clip((np.arange(out_len) * rate).astype(int), 0, len(src) - 1)
        y = src[idx]
    # phase vocoder length is approximate -> pad/trim to exact
    if len(y) < out_len:
        y = np.pad(y, (0, out_len - len(y)))
    return y[:out_len].astype(np.float32)


# ---------------------------------------------------------------- I/O at edges


def _find_set_dir(set_id: str) -> Path:
    hits = glob.glob(str(Path.home() / "aligning" / f"{set_id}*"))
    if not hits:
        sys.exit(f"no staged folder ~/aligning/{set_id}* -- pull the set first")
    return Path(hits[0])


def _load_ref(path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray:
    if path not in cache:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        cache[path] = y.astype(np.float32)
    return cache[path]


def render(
    set_id: str, timeline_path: Path, set_dir: Path, *, min_votes: int = 0
) -> tuple[np.ndarray, dict]:
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_slot = {_norm_slot(t.get("label", "")): t for t in manifest["tracks"]}
    timeline = json.loads(timeline_path.read_text())
    spans = timeline["spans"]
    if min_votes:
        spans = [s for s in spans if s.get("votes", 1 << 30) >= min_votes]

    total_s = float(
        manifest.get("mix_duration_s") or max(s["set_end_s"] for s in spans)
    )
    out = np.zeros(int(total_s * SR) + SR, dtype=np.float32)

    cache: dict[Path, np.ndarray] = {}
    stats = {"spans": len(spans), "rendered": 0, "no_ref": [], "segments": 0}

    for span in spans:
        rp = span.get("ref_path")  # fused timelines carry the resolved ref path
        if rp and Path(rp).is_file():
            src_path = Path(rp)
        else:
            track = by_slot.get(_norm_slot(span["slot_label"]))
            src_path = _stem_source(track, span.get("claimed_stem")) if track else None
        if src_path is None:
            stats["no_ref"].append(span["slot_label"])
            continue
        ref = _load_ref(src_path, cache)
        for seg in _segments_for(span):
            r0 = int(max(0.0, seg.ref_start_s) * SR)
            r1 = int(min(len(ref) / SR, seg.ref_end_s) * SR)
            if r1 <= r0:
                continue
            out_len = int((seg.mix_end_s - seg.mix_start_s) * SR)
            placed = _fade(_stretch_to(ref[r0:r1], out_len))
            at = int(seg.mix_start_s * SR)
            end = min(at + len(placed), len(out))
            out[at:end] += placed[: end - at]
            stats["segments"] += 1
        stats["rendered"] += 1

    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * 0.97  # normalize to just under full-scale
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", required=True)
    ap.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="defaults to out/<set_id>_predicted_timeline.json",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="defaults to out/<set_id>_render.wav"
    )
    ap.add_argument(
        "--ab",
        action="store_true",
        help="also write a stereo A/B: L=original mix, R=reconstruction",
    )
    ap.add_argument(
        "--min-votes",
        type=int,
        default=0,
        help="skip spans below this fp vote count (fused timelines only)",
    )
    args = ap.parse_args()

    timeline_path = args.timeline or (
        HERE / "out" / f"{args.set_id}_predicted_timeline.json"
    )
    if not timeline_path.is_file():
        sys.exit(f"no timeline at {timeline_path}")
    set_dir = _find_set_dir(args.set_id)

    print(f"rendering {args.set_id} from {timeline_path.name} ...", flush=True)
    out, stats = render(args.set_id, timeline_path, set_dir, min_votes=args.min_votes)
    print(
        f"  spans={stats['spans']} rendered={stats['rendered']} "
        f"segments={stats['segments']} no_ref={len(stats['no_ref'])} {stats['no_ref'][:8]}"
    )

    out_path = args.out or (HERE / "out" / f"{args.set_id}_render.wav")
    sf.write(str(out_path), out, SR)
    print(f"  wrote {out_path}")

    if args.ab:
        mix_hits = glob.glob(str(set_dir / "mix.*"))
        mix_hits = [m for m in mix_hits if not m.endswith(".asd")]
        if mix_hits:
            mix, _ = librosa.load(mix_hits[0], sr=SR, mono=True)
            n = min(len(mix), len(out))
            stereo = np.stack([mix[:n], out[:n]], axis=1).astype(np.float32)
            ab_path = out_path.with_name(f"{args.set_id}_ab.wav")
            sf.write(str(ab_path), stereo, SR)
            print(f"  wrote {ab_path}  (pan hard L=mix / R=render to A/B)")
        else:
            print("  --ab: no mix.* found, skipped")


if __name__ == "__main__":
    main()

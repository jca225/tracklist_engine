"""Prepare local hash caches for one strict fingerprint segment lane."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from workspaces.alignment_prototype.fp_index import (
    FpKey,
    compute_from_file,
    load_cached,
    save_cached,
)
from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir
from workspaces.alignment_prototype.infer import _manifest_by_tid, _vocal_ref_path
from workspaces.alignment_prototype.landmark_fp import constellation, hashes
from workspaces.alignment_prototype.mix_fp_hits import load_mix_mono
from core.result import Err, Ok

from .routes import lane
from .stem_overrides import norm_slot as _norm_slot
from .stem_overrides import timeline_stem_overrides


def prepare_lane(
    *,
    set_id: str,
    timeline_path: Path,
    lane_name: str,
    mix_hash_cache: Path,
    ref_fp_cache: Path,
    stem_override_path: Path | None = None,
) -> dict[str, int | str]:
    """Build local-only mix/reference hashes without touching canonical state."""
    route = lane(lane_name)
    set_dir = find_aligning_dir(set_id)
    if set_dir is None:
        raise FileNotFoundError(f"no aligning directory for {set_id}")
    mix_path = set_dir / route.mix_file
    if not mix_path.is_file():
        raise FileNotFoundError(f"missing lane mix audio: {mix_path}")
    mix_hash_cache.mkdir(parents=True, exist_ok=True)
    mix_cache_path = mix_hash_cache / f"{set_id}_{route.mix_cache_suffix}"
    if not mix_cache_path.is_file():
        mix_cache_path.write_bytes(
            pickle.dumps(hashes(*constellation(load_mix_mono(mix_path))))
        )

    timeline = json.loads(timeline_path.read_text())
    overrides = (
        timeline_stem_overrides(timeline["spans"], stem_override_path)
        if stem_override_path is not None
        else {}
    )
    by_tid = _manifest_by_tid(set_dir, set_id)
    counts = {"ready": 0, "built": 0, "missing_audio": 0, "failed": 0}
    for span in timeline["spans"]:
        slot = str(span["slot_label"])
        stem = overrides.get(
            _norm_slot(slot), str(span.get("claimed_stem") or "regular")
        )
        if stem not in route.claimed_stems:
            continue
        recording_id = str(span["recording_id"])
        key = FpKey(recording_id, route.reference_stem)
        if load_cached(key, ref_fp_cache) is not None:
            counts["ready"] += 1
            continue
        if route.name == "vocal":
            audio = _vocal_ref_path(by_tid.get(recording_id))
        else:
            audio = None
        if not audio or not Path(audio).is_file():
            counts["missing_audio"] += 1
            continue
        match compute_from_file(Path(audio)):
            case Ok(fp):
                save_cached(fp, key, ref_fp_cache)
                counts["built"] += 1
            case Err(_message):
                counts["failed"] += 1
    return {
        **counts,
        "mix_cache": str(mix_cache_path),
        "ref_fp_cache": str(ref_fp_cache),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--lane", choices=("instrumental", "vocal"), required=True)
    parser.add_argument("--mix-hash-cache", type=Path, required=True)
    parser.add_argument("--ref-fp-cache", type=Path, required=True)
    parser.add_argument("--stem-overrides", type=Path, default=None)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_lane(
                set_id=args.set_id,
                timeline_path=args.timeline,
                lane_name=args.lane,
                mix_hash_cache=args.mix_hash_cache,
                ref_fp_cache=args.ref_fp_cache,
                stem_override_path=args.stem_overrides,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

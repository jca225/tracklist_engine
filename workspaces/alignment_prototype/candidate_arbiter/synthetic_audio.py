"""Add pinned chroma verification evidence to synthetic instrumental banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workspaces.alignment_prototype.trajectory.features import BIN_S, FeatureBank

from .audio_enrich import with_verification
from .audio_verify import verify_alignment
from .io import read_candidate_bank, write_candidate_bank
from .schema import CandidateBankMetadata, PlacementCandidate


def enrich_synthetic_bank(
    window: Path,
    candidate_path: Path,
    *,
    window_s: float = 12.0,
) -> tuple[
    CandidateBankMetadata,
    tuple[PlacementCandidate, ...],
    dict[str, int],
]:
    metadata, candidates = read_candidate_bank(candidate_path)
    bank = FeatureBank()
    mix_features = bank.match(window / "mix_instrumental.flac", "chroma")
    window_bins = max(4, int(round(window_s / BIN_S)))
    ref_cache = {}
    out = []
    counts = {"enriched": 0, "missing_ref": 0, "short": 0}
    for candidate in candidates:
        ref_path = window / "refs" / f"{candidate.recording_id}_instrumental.flac"
        if not ref_path.is_file() or candidate.ref_start_s is None:
            counts["missing_ref"] += 1
            out.append(candidate)
            continue
        key = str(ref_path)
        if key not in ref_cache:
            ref_cache[key] = bank.match(ref_path, "chroma")
        verification = verify_alignment(
            mix_features,
            ref_cache[key],
            mix_start_bin=int(round(candidate.set_start_s / BIN_S)),
            ref_start_bin=int(round(candidate.ref_start_s / BIN_S)),
            window_bins=window_bins,
        )
        if verification is None:
            counts["short"] += 1
            out.append(candidate)
        else:
            out.append(with_verification(candidate, verification))
            counts["enriched"] += 1
    return metadata, tuple(out), counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args(argv)

    windows = sorted(path for path in args.root.glob("synthv2_*") if path.is_dir())
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    args.out.mkdir(parents=True, exist_ok=True)
    totals = {"enriched": 0, "missing_ref": 0, "short": 0}
    for window in windows:
        source = args.candidates / f"{window.name}.jsonl"
        if not source.is_file():
            continue
        metadata, candidates, counts = enrich_synthetic_bank(window, source)
        write_candidate_bank(args.out / source.name, metadata, candidates)
        for key, value in counts.items():
            totals[key] += value
        print(window.name, json.dumps(counts, sort_keys=True))
    print("total", json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build instrumental-FP candidate banks from exact synthetic mix labels."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from workspaces.alignment_prototype.landmark_fp import (
    constellation,
    fingerprint_from_audio,
    hashes,
)
from workspaces.alignment_prototype.mix_fp_hits import (
    fp_candidate_evidence,
    load_mix_mono,
    offset_candidates,
)

from .adapters import fp_candidates_for_span
from .io import write_candidate_bank
from .schema import CandidateBankMetadata, CandidateEvidence, PlacementCandidate

_PACKAGE = Path(__file__).resolve().parent
_REPO = _PACKAGE.parent.parent.parent


def _producer_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=_REPO,
    ).strip()


def _baseline_offset(rng: np.random.Generator) -> float:
    """Synthetic coarse-placement error with both local and tail examples."""
    if rng.random() < 0.75:
        return float(rng.normal(0.0, 18.0))
    return float(rng.normal(0.0, 75.0))


def build_synthetic_fp_bank(
    window: Path,
    *,
    seed: int,
    producer_sha: str,
) -> tuple[CandidateBankMetadata, tuple[PlacementCandidate, ...]]:
    """Materialize baseline + native top-K FP candidates for one synth window."""
    gt_path = window / "ground_truth.yaml"
    document = yaml.safe_load(gt_path.read_text())
    set_id = str(document["set_id"])
    rows = [
        row
        for row in document["tracks"]
        if (row.get("claimed_stem") or "regular") == "instrumental"
        and row.get("track_id")
    ]
    mix_path = window / "mix.flac"
    mix_y = load_mix_mono(mix_path)
    mix_hashes = hashes(*constellation(mix_y))
    rng = np.random.default_rng(seed)
    candidates: list[PlacementCandidate] = []

    for row in rows:
        recording_id = str(row["track_id"])
        gt_start = float(row["set_start_s"])
        baseline_start = max(0.0, gt_start + _baseline_offset(rng))
        span = {
            "slot_label": str(row["slot_label"]),
            "recording_id": recording_id,
            "claimed_stem": "instrumental",
            "set_start_s": baseline_start,
            "start_source": "synthetic_coarse",
        }
        candidates.append(
            PlacementCandidate(
                set_id=set_id,
                slot_label=span["slot_label"],
                recording_id=recording_id,
                claimed_stem="instrumental",
                source="baseline",
                rank=0,
                set_start_s=baseline_start,
                ref_start_s=(
                    float(row["ref_start_s"])
                    if row.get("ref_start_s") is not None
                    else None
                ),
                native_confidence=None,
                evidence=CandidateEvidence(
                    baseline_delta_s=0.0,
                    baseline_source="synthetic_coarse",
                ),
            )
        )
        ref_path = window / "refs" / f"{recording_id}_instrumental.flac"
        if not ref_path.is_file():
            continue
        ref_fp = fingerprint_from_audio(load_mix_mono(ref_path))
        raw = offset_candidates(mix_hashes, ref_fp, topk=6)
        candidates.extend(
            fp_candidates_for_span(
                set_id,
                span,
                fp_candidate_evidence(raw),
            )
        )

    return (
        CandidateBankMetadata(
            producer_sha=producer_sha,
            source_timeline_sha256=None,
        ),
        tuple(candidates),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    windows = sorted(
        path
        for path in args.root.glob("synthv2_*")
        if path.is_dir() and (path / "ground_truth.yaml").is_file()
    )
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    producer_sha = _producer_sha()
    manifest = []
    for index, window in enumerate(windows):
        metadata, candidates = build_synthetic_fp_bank(
            window,
            seed=args.seed + index,
            producer_sha=producer_sha,
        )
        output = args.out / f"{window.name}.jsonl"
        write_candidate_bank(output, metadata, candidates)
        manifest.append({"window": window.name, "candidates": len(candidates)})
        print(f"{window.name}: {len(candidates)} candidates")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidate_index.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

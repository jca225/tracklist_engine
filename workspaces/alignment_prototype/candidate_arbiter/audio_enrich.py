"""Stem-routed audio verification enrichment for placement candidate banks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from workspaces.alignment_prototype.candidate_arbiter.audio_verify import (
    PairVerification,
    verify_alignment,
)
from workspaces.alignment_prototype.drivers.base import gt_stem_by_slot, norm_slot
from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir
from workspaces.alignment_prototype.infer import _manifest_by_tid
from workspaces.alignment_prototype.recon_probe import find_mix_stems
from workspaces.alignment_prototype.trajectory.features import (
    BIN_S,
    FeatureBank,
    resolve_span_audio,
)

from .io import read_candidate_bank, write_candidate_bank
from .schema import CandidateBankMetadata, PlacementCandidate


def with_verification(
    candidate: PlacementCandidate,
    verification: PairVerification,
) -> PlacementCandidate:
    return replace(
        candidate,
        evidence=replace(
            candidate.evidence,
            verify_match_mean=verification.match_mean,
            verify_match_p10=verification.match_p10,
            verify_margin=verification.margin,
            verify_continuity=verification.continuity,
            verify_support_bins=verification.support_bins,
            verify_background_count=verification.background_count,
        ),
    )


def enrich_real_bank(
    timeline_path: Path,
    candidate_path: Path,
    *,
    gt_path: Path,
    stems: frozenset[str],
    window_s: float = 12.0,
) -> tuple[
    CandidateBankMetadata,
    tuple[PlacementCandidate, ...],
    dict[str, int],
]:
    timeline = json.loads(timeline_path.read_text())
    set_id = str(timeline["set_id"])
    set_dir = find_aligning_dir(set_id)
    if set_dir is None:
        raise FileNotFoundError(f"no aligning directory for {set_id}")
    metadata, candidates = read_candidate_bank(candidate_path)
    spans = {
        (norm_slot(span["slot_label"]), str(span["recording_id"])): span
        for span in timeline["spans"]
    }
    gt_stems = gt_stem_by_slot(gt_path)
    by_tid = _manifest_by_tid(set_dir, set_id)
    mix_stems = find_mix_stems(set_dir)
    mix_full = set_dir / "mix.m4a"
    bank = FeatureBank()
    window_bins = max(4, int(round(window_s / BIN_S)))
    enriched: list[PlacementCandidate] = []
    counts = {"enriched": 0, "missing_span": 0, "missing_audio": 0, "short": 0}

    for candidate in candidates:
        key = (norm_slot(candidate.slot_label), candidate.recording_id)
        span = spans.get(key)
        if span is None:
            counts["missing_span"] += 1
            enriched.append(candidate)
            continue
        stem = gt_stems.get(key[0]) or candidate.claimed_stem
        routed = resolve_span_audio(
            set_dir,
            by_tid,
            mix_stems,
            mix_full,
            {"track_id": candidate.recording_id, "claimed_stem": stem},
        )
        if isinstance(routed, str) or stem not in stems:
            counts["missing_audio"] += isinstance(routed, str)
            enriched.append(replace(candidate, claimed_stem=stem))
            continue
        baseline_ref = span.get("ref_start_s")
        ref_start = (
            candidate.ref_start_s
            if candidate.ref_start_s is not None
            else baseline_ref
        )
        if ref_start is None:
            counts["missing_audio"] += 1
            enriched.append(replace(candidate, claimed_stem=stem))
            continue
        mix_features = bank.match(routed.mix_path, routed.match_feature)
        ref_features = bank.match(routed.ref_path, routed.match_feature)
        verification = verify_alignment(
            mix_features,
            ref_features,
            mix_start_bin=int(round(candidate.set_start_s / BIN_S)),
            ref_start_bin=int(round(float(ref_start) / BIN_S)),
            window_bins=window_bins,
        )
        updated = replace(candidate, claimed_stem=stem)
        if verification is None:
            counts["short"] += 1
        else:
            updated = with_verification(updated, verification)
            counts["enriched"] += 1
        enriched.append(updated)
    return metadata, tuple(enriched), counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stems", default="regular,instrumental")
    args = parser.parse_args(argv)

    metadata, candidates, counts = enrich_real_bank(
        args.timeline,
        args.candidates,
        gt_path=args.gt,
        stems=frozenset(part for part in args.stems.split(",") if part),
    )
    write_candidate_bank(args.output, metadata, candidates)
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

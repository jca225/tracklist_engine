"""Run collision-aware fingerprint / HuBERT segment decoding in shadow mode.

Example (instrumental landmark):

    venvs/audio/bin/python -m workspaces.alignment_prototype.fp_segments.run \
      --set-id 2nvzlh2k \
      --timeline workspaces/alignment_prototype/out/2nvzlh2k_agentic_baseline_gtstem.json \
      --lane instrumental \
      --mix-hash-cache eda/alignment/ridge_diagnostic/out/stem_mix_hash_cache \
      --output workspaces/alignment_prototype/out/fp_segments/2nvzlh2k.json

Example (vocal HuBERT):

    venvs/audio/bin/python -m workspaces.alignment_prototype.fp_segments.run \
      --set-id 2nvzlh2k \
      --timeline workspaces/alignment_prototype/out/2nvzlh2k_agentic_baseline_gtstem.json \
      --lane vocal \
      --observation hubert \
      --stem-overrides labeling/fixtures/bb11_ground_truth.yaml \
      --output workspaces/alignment_prototype/out/fp_segments/2nvzlh2k_vocal_hubert.json

The output is a diagnostic segment bank, not a PredictedTimeline. This command
cannot change the default driver or canonical database.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from workspaces.alignment_prototype.fp_index import DEFAULT_CACHE_DIR, FpKey
from workspaces.alignment_prototype.fp_index import load as load_ref_fp
from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir
from workspaces.alignment_prototype.infer import _manifest_by_tid, _vocal_ref_path
from workspaces.alignment_prototype.instrumental_probe import (
    _instr_stem,
    _resolve_stem_by_slot,
)

from .chroma_retrieve import load_chroma_feat, retrieve_chroma_matches
from .hubert_retrieve import load_hubert_feat, retrieve_hubert_matches
from .local_decode import decode_constituent
from .retrieve import retrieve_matches
from .routes import lane
from .schema import LandmarkMatch
from .stem_overrides import norm_slot as _norm_slot
from .stem_overrides import timeline_stem_overrides

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_OBSERVATIONS = frozenset({"landmark", "hubert", "chroma"})
_LANE_OBSERVATIONS = {
    "instrumental": frozenset({"landmark", "chroma"}),
    "vocal": frozenset({"hubert", "landmark"}),
}


def _producer_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()


def _stem_overrides(path: Path | None, spans: list[dict[str, Any]]) -> dict[str, str]:
    """Evaluation-only stem routing via GT track_id → timeline recording_id."""
    if path is None:
        return {}
    return timeline_stem_overrides(spans, path)


def _reference_fp(recording_id: str, stem: str, *, cache_dir: Path):
    # Segment lanes are strictly like-for-like. Missing stem fingerprints
    # abstain; falling back to regular audio would silently violate the route.
    return load_ref_fp(FpKey(recording_id, stem), cache_dir=cache_dir)


def _default_observation(lane_name: str) -> str:
    return "hubert" if lane_name == "vocal" else "landmark"


def _resolve_observation(lane_name: str, observation: str | None) -> str:
    chosen = observation or _default_observation(lane_name)
    allowed = _LANE_OBSERVATIONS.get(lane_name)
    if allowed is None:
        raise ValueError(f"unknown lane {lane_name!r}")
    if chosen not in allowed:
        raise ValueError(
            f"lane {lane_name!r} does not support --observation {chosen!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return chosen


def _resolve_ref_instrumental(
    set_dir: Path,
    by_tid: dict[str, dict],
    recording_id: str,
    slot_label: str,
) -> Path | None:
    path = _instr_stem(by_tid.get(recording_id))
    if path and Path(path).is_file():
        return Path(path)
    # Timeline slots are unpadded; stem dirs on disk are often zero-padded.
    candidates = [slot_label]
    match = re.match(r"^(\d+)(.*)$", slot_label)
    if match:
        candidates.append(f"{int(match.group(1)):03d}{match.group(2)}")
    for candidate in candidates:
        resolved = _resolve_stem_by_slot(set_dir, candidate)
        if resolved and Path(resolved).is_file():
            return Path(resolved)
    return None


def _segment_rows(segments) -> list[dict[str, float]]:
    return [
        {
            "mix_start_s": segment.mix_start_s,
            "mix_end_s": segment.mix_end_s,
            "ref_start_s": segment.ref_start_s,
            "ref_end_s": segment.ref_end_s,
            "slope": segment.slope,
            "evidence": segment.evidence,
            "confidence": segment.confidence,
        }
        for segment in segments
    ]


def _row(
    *,
    slot: str,
    recording_id: str,
    stem: str,
    status: str,
    matches: tuple[LandmarkMatch, ...] = (),
    segments=(),
) -> dict[str, Any]:
    return {
        "slot_label": slot,
        "recording_id": recording_id,
        "stem": stem,
        "status": status,
        "match_count": len(matches),
        "segments": _segment_rows(segments),
    }


def _decode_landmark_lane(
    *,
    route,
    timeline: dict[str, Any],
    overrides: dict[str, str],
    mix_hashes: dict,
    ref_fp_cache: Path,
    pair_cap: int,
    slopes: tuple[float, ...],
    mix_duration_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for span in timeline["spans"]:
        slot = str(span["slot_label"])
        stem = overrides.get(
            _norm_slot(slot), str(span.get("claimed_stem") or "regular")
        )
        if stem not in route.claimed_stems:
            continue
        recording_id = str(span["recording_id"])
        ref_fp = _reference_fp(
            recording_id, route.reference_stem, cache_dir=ref_fp_cache
        )
        if ref_fp is None:
            rows.append(
                _row(
                    slot=slot,
                    recording_id=recording_id,
                    stem=stem,
                    status="no_reference_fingerprint",
                )
            )
            continue
        matches = retrieve_matches(
            mix_hashes,
            ref_fp,
            recording_id=recording_id,
            ref_stem=route.reference_stem,
            mix_channel=route.mix_channel,
            pair_cap=pair_cap,
        )
        segments = decode_constituent(
            matches,
            mix_duration_s=mix_duration_s,
            allowed_slopes=slopes,
        )
        rows.append(
            _row(
                slot=slot,
                recording_id=recording_id,
                stem=stem,
                status="decoded" if segments else "abstained",
                matches=matches,
                segments=segments,
            )
        )
    return rows


def _decode_chroma_instrumental_lane(
    *,
    set_id: str,
    route,
    timeline: dict[str, Any],
    overrides: dict[str, str],
    slopes: tuple[float, ...],
    mix_duration_s: float,
) -> list[dict[str, Any]]:
    set_dir = find_aligning_dir(set_id)
    if set_dir is None:
        raise FileNotFoundError(f"no aligning directory for {set_id}")
    mix_path = set_dir / route.mix_file
    if not mix_path.is_file():
        raise FileNotFoundError(f"missing lane mix audio: {mix_path}")
    mix_feat = load_chroma_feat(mix_path)
    by_tid = _manifest_by_tid(set_dir, set_id)
    ref_feat_cache: dict[str, np.ndarray] = {}

    rows: list[dict[str, Any]] = []
    for span in timeline["spans"]:
        slot = str(span["slot_label"])
        stem = overrides.get(
            _norm_slot(slot), str(span.get("claimed_stem") or "regular")
        )
        if stem not in route.claimed_stems:
            continue
        recording_id = str(span["recording_id"])
        audio = _resolve_ref_instrumental(set_dir, by_tid, recording_id, slot)
        if audio is None:
            rows.append(
                _row(
                    slot=slot,
                    recording_id=recording_id,
                    stem=stem,
                    status="missing_instrumental_audio",
                )
            )
            continue
        if recording_id not in ref_feat_cache:
            ref_feat_cache[recording_id] = load_chroma_feat(audio)
        matches = retrieve_chroma_matches(
            mix_feat,
            ref_feat_cache[recording_id],
            recording_id=recording_id,
            ref_stem=route.reference_stem,
            mix_channel=route.mix_channel,
        )
        segments = decode_constituent(
            matches,
            mix_duration_s=mix_duration_s,
            allowed_slopes=slopes,
        )
        rows.append(
            _row(
                slot=slot,
                recording_id=recording_id,
                stem=stem,
                status="decoded" if segments else "abstained",
                matches=matches,
                segments=segments,
            )
        )
    return rows


def _decode_hubert_vocal_lane(
    *,
    set_id: str,
    route,
    timeline: dict[str, Any],
    overrides: dict[str, str],
    slopes: tuple[float, ...],
    mix_duration_s: float,
) -> list[dict[str, Any]]:
    set_dir = find_aligning_dir(set_id)
    if set_dir is None:
        raise FileNotFoundError(f"no aligning directory for {set_id}")
    mix_path = set_dir / route.mix_file
    if not mix_path.is_file():
        raise FileNotFoundError(f"missing lane mix audio: {mix_path}")
    mix_feat = load_hubert_feat(mix_path)
    by_tid = _manifest_by_tid(set_dir, set_id)
    ref_feat_cache: dict[str, np.ndarray] = {}

    rows: list[dict[str, Any]] = []
    for span in timeline["spans"]:
        slot = str(span["slot_label"])
        stem = overrides.get(
            _norm_slot(slot), str(span.get("claimed_stem") or "regular")
        )
        if stem not in route.claimed_stems:
            continue
        recording_id = str(span["recording_id"])
        audio = _vocal_ref_path(by_tid.get(recording_id))
        if not audio or not Path(audio).is_file():
            rows.append(
                _row(
                    slot=slot,
                    recording_id=recording_id,
                    stem=stem,
                    status="missing_vocal_audio",
                )
            )
            continue
        if recording_id not in ref_feat_cache:
            ref_feat_cache[recording_id] = load_hubert_feat(audio)
        matches = retrieve_hubert_matches(
            mix_feat,
            ref_feat_cache[recording_id],
            recording_id=recording_id,
            ref_stem=route.reference_stem,
            mix_channel=route.mix_channel,
        )
        segments = decode_constituent(
            matches,
            mix_duration_s=mix_duration_s,
            allowed_slopes=slopes,
        )
        rows.append(
            _row(
                slot=slot,
                recording_id=recording_id,
                stem=stem,
                status="decoded" if segments else "abstained",
                matches=matches,
                segments=segments,
            )
        )
    return rows


def run_shadow(
    *,
    set_id: str,
    timeline_path: Path,
    output_path: Path,
    mix_hash_cache: Path | None = None,
    stem_override_path: Path | None = None,
    pair_cap: int = 64,
    slopes: tuple[float, ...] = (0.94, 0.97, 1.0, 1.03, 1.06),
    lane_name: str = "instrumental",
    observation: str | None = None,
    ref_fp_cache: Path = DEFAULT_CACHE_DIR,
) -> Path:
    timeline = json.loads(timeline_path.read_text())
    if str(timeline.get("set_id")) != set_id:
        raise ValueError("timeline set_id does not match --set-id")
    route = lane(lane_name)
    chosen = _resolve_observation(lane_name, observation)
    overrides = _stem_overrides(stem_override_path, timeline["spans"])
    mix_duration_s = max(float(span["set_end_s"]) for span in timeline["spans"])

    cache_path: Path | None = None
    if chosen == "landmark":
        if mix_hash_cache is None:
            raise ValueError("--mix-hash-cache is required for landmark observation")
        cache_path = mix_hash_cache / f"{set_id}_{route.mix_cache_suffix}"
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing mix hash cache: {cache_path}")
        mix_hashes = pickle.loads(cache_path.read_bytes())
        rows = _decode_landmark_lane(
            route=route,
            timeline=timeline,
            overrides=overrides,
            mix_hashes=mix_hashes,
            ref_fp_cache=ref_fp_cache,
            pair_cap=pair_cap,
            slopes=slopes,
            mix_duration_s=mix_duration_s,
        )
    elif chosen == "chroma":
        rows = _decode_chroma_instrumental_lane(
            set_id=set_id,
            route=route,
            timeline=timeline,
            overrides=overrides,
            slopes=slopes,
            mix_duration_s=mix_duration_s,
        )
    else:
        rows = _decode_hubert_vocal_lane(
            set_id=set_id,
            route=route,
            timeline=timeline,
            overrides=overrides,
            slopes=slopes,
            mix_duration_s=mix_duration_s,
        )

    payload = {
        "schema_version": 1,
        "producer_sha": _producer_sha(),
        "set_id": set_id,
        "source_timeline": str(timeline_path),
        "mix_hash_cache": str(cache_path) if cache_path is not None else None,
        "pair_cap": pair_cap,
        "slopes": list(slopes),
        "lane": route.name,
        "observation": chosen,
        "channel_route": (f"mix_{route.mix_channel}->reference_{route.reference_stem}"),
        "shadow_only": True,
        "spans": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument(
        "--mix-hash-cache",
        type=Path,
        default=None,
        help="required for landmark observation; unused for hubert",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stem-overrides",
        type=Path,
        default=None,
        help="evaluation-only GT YAML; routes stems via track_id→recording_id",
    )
    parser.add_argument("--pair-cap", type=int, default=64)
    parser.add_argument(
        "--lane", choices=("instrumental", "vocal"), default="instrumental"
    )
    parser.add_argument(
        "--observation",
        choices=("landmark", "hubert", "chroma"),
        default=None,
        help="default: landmark for instrumental, hubert for vocal; "
        "chroma is instrumental-only",
    )
    parser.add_argument("--ref-fp-cache", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    path = run_shadow(
        set_id=args.set_id,
        timeline_path=args.timeline,
        mix_hash_cache=args.mix_hash_cache,
        output_path=args.output,
        stem_override_path=args.stem_overrides,
        pair_cap=args.pair_cap,
        lane_name=args.lane,
        observation=args.observation,
        ref_fp_cache=args.ref_fp_cache,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run collision-aware instrumental fingerprint segment decoding in shadow mode.

Example:

    venvs/audio/bin/python -m workspaces.alignment_prototype.fp_segments.run \
      --set-id 2nvzlh2k \
      --timeline workspaces/alignment_prototype/out/2nvzlh2k_agentic_baseline_gtstem.json \
      --mix-hash-cache eda/alignment/ridge_diagnostic/out/stem_mix_hash_cache \
      --output workspaces/alignment_prototype/out/fp_segments/2nvzlh2k.json

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

import yaml

from workspaces.alignment_prototype.fp_index import DEFAULT_CACHE_DIR, FpKey
from workspaces.alignment_prototype.fp_index import load as load_ref_fp

from .local_decode import decode_constituent
from .retrieve import retrieve_matches
from .routes import lane

_REPO = Path(__file__).resolve().parent.parent.parent.parent


def _producer_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()


def _norm_slot(value: object) -> str:
    match = re.match(r"^0*(\d+)(.*)$", str(value))
    return match.group(1) + match.group(2) if match else str(value)


def _stem_overrides(path: Path | None) -> dict[str, str]:
    """Evaluation-only stem routing for stale historical timelines."""
    if path is None:
        return {}
    rows = yaml.safe_load(path.read_text())["tracks"]
    return {
        _norm_slot(row["slot_label"]): str(row.get("claimed_stem") or "regular")
        for row in rows
        if row.get("slot_label") and str(row.get("slot_label")) != "mix"
    }


def _reference_fp(recording_id: str, stem: str, *, cache_dir: Path):
    # Segment lanes are strictly like-for-like. Missing stem fingerprints
    # abstain; falling back to regular audio would silently violate the route.
    return load_ref_fp(FpKey(recording_id, stem), cache_dir=cache_dir)


def run_shadow(
    *,
    set_id: str,
    timeline_path: Path,
    mix_hash_cache: Path,
    output_path: Path,
    stem_override_path: Path | None = None,
    pair_cap: int = 64,
    slopes: tuple[float, ...] = (0.94, 0.97, 1.0, 1.03, 1.06),
    lane_name: str = "instrumental",
    ref_fp_cache: Path = DEFAULT_CACHE_DIR,
) -> Path:
    timeline = json.loads(timeline_path.read_text())
    if str(timeline.get("set_id")) != set_id:
        raise ValueError("timeline set_id does not match --set-id")
    route = lane(lane_name)
    cache_name = f"{set_id}_{route.mix_cache_suffix}"
    cache_path = mix_hash_cache / cache_name
    if not cache_path.is_file():
        raise FileNotFoundError(f"missing instrumental mix hash cache: {cache_path}")
    mix_hashes = pickle.loads(cache_path.read_bytes())
    overrides = _stem_overrides(stem_override_path)
    mix_duration_s = max(float(span["set_end_s"]) for span in timeline["spans"])

    rows = []
    for span in timeline["spans"]:
        slot = str(span["slot_label"])
        stem = overrides.get(_norm_slot(slot), str(span.get("claimed_stem") or "regular"))
        if stem not in route.claimed_stems:
            continue
        recording_id = str(span["recording_id"])
        ref_fp = _reference_fp(
            recording_id, route.reference_stem, cache_dir=ref_fp_cache
        )
        if ref_fp is None:
            rows.append(
                {
                    "slot_label": slot,
                    "recording_id": recording_id,
                    "stem": stem,
                    "status": "no_reference_fingerprint",
                    "match_count": 0,
                    "segments": [],
                }
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
            {
                "slot_label": slot,
                "recording_id": recording_id,
                "stem": stem,
                "status": "decoded" if segments else "abstained",
                "match_count": len(matches),
                "segments": [
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
                ],
            }
        )

    payload = {
        "schema_version": 1,
        "producer_sha": _producer_sha(),
        "set_id": set_id,
        "source_timeline": str(timeline_path),
        "mix_hash_cache": str(cache_path),
        "pair_cap": pair_cap,
        "slopes": list(slopes),
        "lane": route.name,
        "channel_route": (
            f"mix_{route.mix_channel}->reference_{route.reference_stem}"
        ),
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
    parser.add_argument("--mix-hash-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stem-overrides",
        type=Path,
        default=None,
        help="evaluation-only GT YAML used solely to correct stale claimed_stem",
    )
    parser.add_argument("--pair-cap", type=int, default=64)
    parser.add_argument("--lane", choices=("instrumental", "vocal"), default="instrumental")
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
        ref_fp_cache=args.ref_fp_cache,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

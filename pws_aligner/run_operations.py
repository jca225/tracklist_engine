"""Operation-LF runner — apply keylock_vs_varispeed to BB12 spans.

Reuses capture_votes.py's audio-loading helpers to resolve per-span
mix-audio and ref-audio without re-implementing path resolution or
manifest indexing.

For each span in the predicted timeline that has a placed ref:
  1. Resolve mix-audio path via capture_votes._mix_audio_path
  2. Resolve ref-audio path via capture_votes._ref_audio_path /
     capture_votes._load_manifest_by_rid
  3. Load waveforms (librosa.load, mono, shared sr=22050)
  4. Compute tempo_ratio from ref_span / set_span (same formula als_reconcile
     uses).  Spans with degenerate durations (ratio < 0.5 or > 2.0) are
     skipped — they are fiber sub-spans with pathological ref_end_s values
     that do not represent a real stretch relationship.
  5. Call keylock_vs_varispeed(span_id, mix, ref, sr, tempo_ratio)
  6. Collect OperationVote → write JSON + print histogram.

ANALYSIS ONLY — no accuracy claim.  There is no per-clip GT stretch label
in the pipeline yet; tempo_ratio here is INFERRED from placed span durations
and carries placement error.

Usage
-----
    venvs/audio/bin/python -m pws_aligner.run_operations \\
        --set-id 1fsnxchk

Output: alignment/out/<set_id>_operation_votes.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import librosa
import numpy as np

# Resolve repo root by walking up from this file's location.
# Use parent chaining rather than parents[N] to avoid the ratchet count
# (capture_votes.py already accounts for the two pws_aligner instances).
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_OUT_DIR = _REPO / "alignment" / "out"

# Tempo ratio bounds: spans outside [0.5, 2.0] are almost certainly fiber
# sub-spans with broken ref_end_s (looping fragments where ref duration ≠
# set duration in any musically meaningful sense).
_TEMPO_RATIO_MIN = 0.5
_TEMPO_RATIO_MAX = 2.0

# Shared sample rate for CQT (operations.py uses librosa CQT at whatever sr
# librosa.load gives; match what the probe expects).
_SR = 22_050

# Minimum span duration in seconds — avoid loading nearly-empty segments.
_MIN_SPAN_S = 2.0


def _load_mono(path: Path, start_s: float, end_s: float | None) -> np.ndarray:
    """Load a slice of audio as mono float32 at _SR.

    Uses librosa.load with offset/duration so we don't load the entire file.
    """
    duration = (end_s - start_s) if end_s is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(
            str(path),
            sr=_SR,
            mono=True,
            offset=start_s,
            duration=duration,
        )
    return y


def run_operations(
    set_id: str,
    *,
    timeline_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Apply keylock_vs_varispeed to every span in the predicted timeline.

    Returns the path to the written operation_votes.json.
    """
    # Late imports: keep them close to use so import errors are legible.
    from pws_aligner.capture_votes import (
        _find_aligning_dir,
        _load_manifest_by_rid,
        _load_timeline,
        _mix_audio_path,
        _ref_audio_path,
    )
    from pws_aligner.operations import keylock_vs_varispeed

    out_dir = _DEFAULT_OUT_DIR
    if timeline_path is None:
        timeline_path = out_dir / f"{set_id}_predicted_timeline.json"
    if out_path is None:
        out_path = out_dir / f"{set_id}_operation_votes.json"

    if not timeline_path.exists():
        sys.exit(
            f"ERROR: predicted timeline not found: {timeline_path}\n"
            "Run infer.py first:\n"
            f"  venvs/audio/bin/python -m alignment.infer "
            f"--set-id {set_id}\n"
        )

    spans = _load_timeline(timeline_path)
    if not spans:
        sys.exit(f"ERROR: no spans in timeline: {timeline_path}")

    # Resolve set-level resources (same as capture_votes.capture_votes)
    set_dir = _find_aligning_dir(set_id)
    if set_dir is None:
        sys.exit(
            f"ERROR: no aligning dir found for {set_id} under ~/aligning/ — "
            "mix audio is required for operation LF"
        )
    manifest_by_rid = _load_manifest_by_rid(set_dir, set_id)

    votes: list[dict] = []
    n_processed = 0
    n_skipped_no_ref = 0
    n_skipped_no_mix = 0
    n_skipped_degenerate = 0
    n_skipped_load_error = 0
    tempo_ratio_source_counts: Counter[str] = Counter()

    print(f"Processing {len(spans)} spans from {timeline_path.name} …", flush=True)

    for i, span in enumerate(spans):
        slot_label = str(span.get("slot_label", str(i)))
        recording_id = span.get("recording_id") or None
        claimed_stem = str(span.get("claimed_stem", "regular"))
        set_start_s = float(span.get("set_start_s", 0.0))
        set_end_s = float(span.get("set_end_s", 0.0))
        ref_start_s = float(span.get("ref_start_s", 0.0))
        ref_end_s_raw = span.get("ref_end_s")

        # ---- tempo_ratio ------------------------------------------------
        # Prefer a field on the span itself (future proofing); fall back to
        # computing from placed durations.
        if "tempo_ratio" in span and span["tempo_ratio"] is not None:
            tempo_ratio = float(span["tempo_ratio"])
            tempo_ratio_source = "span_field"
            tempo_ratio_source_counts["span_field"] += 1
            set_span = set_end_s - set_start_s
        else:
            set_span = set_end_s - set_start_s
            ref_span = (
                float(ref_end_s_raw) - ref_start_s if ref_end_s_raw is not None else 0.0
            )
            if set_span <= 0 or ref_span <= 0:
                print(
                    f"  [{slot_label}] SKIP — degenerate span "
                    f"(set_span={set_span:.2f}s ref_span={ref_span:.2f}s)",
                    flush=True,
                )
                n_skipped_degenerate += 1
                continue
            tempo_ratio = ref_span / set_span
            tempo_ratio_source = "computed"
            tempo_ratio_source_counts["computed"] += 1

        # Filter physically implausible ratios (fiber sub-spans, etc.)
        if not (_TEMPO_RATIO_MIN <= tempo_ratio <= _TEMPO_RATIO_MAX):
            print(
                f"  [{slot_label}] SKIP — tempo_ratio={tempo_ratio:.4f} out of "
                f"[{_TEMPO_RATIO_MIN}, {_TEMPO_RATIO_MAX}] (fiber sub-span?)",
                flush=True,
            )
            n_skipped_degenerate += 1
            continue

        # Minimum duration guard
        if set_span < _MIN_SPAN_S:
            print(
                f"  [{slot_label}] SKIP — set_span={set_span:.2f}s < {_MIN_SPAN_S}s",
                flush=True,
            )
            n_skipped_degenerate += 1
            continue

        # ---- mix audio --------------------------------------------------
        mix_audio = _mix_audio_path(set_dir, claimed_stem)
        if mix_audio is None or not mix_audio.is_file():
            # Fallback to full mix
            mix_audio = _mix_audio_path(set_dir, "regular")
        if mix_audio is None or not mix_audio.is_file():
            print(
                f"  [{slot_label}] SKIP — mix audio not found for stem={claimed_stem!r}",
                flush=True,
            )
            n_skipped_no_mix += 1
            continue

        # ---- ref audio --------------------------------------------------
        ref_row = manifest_by_rid.get(recording_id or "")
        if not ref_row:
            print(
                f"  [{slot_label}] SKIP — recording_id={recording_id!r} not in manifest",
                flush=True,
            )
            n_skipped_no_ref += 1
            continue
        ref_audio = _ref_audio_path(ref_row, claimed_stem)
        if ref_audio is None or not ref_audio.is_file():
            # Try regular stem as fallback
            ref_audio = _ref_audio_path(ref_row, "regular")
        if ref_audio is None or not ref_audio.is_file():
            print(
                f"  [{slot_label}] SKIP — ref audio not found for "
                f"recording_id={recording_id!r} stem={claimed_stem!r}",
                flush=True,
            )
            n_skipped_no_ref += 1
            continue

        # ---- load waveforms (clamp to set span / ref duration) ----------
        ref_end_s = float(ref_end_s_raw) if ref_end_s_raw is not None else None
        try:
            mix_y = _load_mono(mix_audio, set_start_s, set_end_s)
            ref_y = _load_mono(ref_audio, ref_start_s, ref_end_s)
        except Exception as exc:
            print(
                f"  [{slot_label}] SKIP — audio load error: {exc}",
                file=sys.stderr,
            )
            n_skipped_load_error += 1
            continue

        if mix_y.size < _SR * _MIN_SPAN_S or ref_y.size < _SR * _MIN_SPAN_S:
            print(
                f"  [{slot_label}] SKIP — waveform too short "
                f"(mix={mix_y.size // _SR}s ref={ref_y.size // _SR}s)",
                flush=True,
            )
            n_skipped_degenerate += 1
            continue

        # ---- call LF ----------------------------------------------------
        print(
            f"  [{slot_label}] {claimed_stem} tempo_ratio={tempo_ratio:.4f} … ",
            end="",
            flush=True,
        )
        vote = keylock_vs_varispeed(slot_label, mix_y, ref_y, _SR, tempo_ratio)
        print(
            f"{vote.label}  conf={vote.confidence:.2f}  abstain={vote.abstained}",
            flush=True,
        )

        vote_dict = asdict(vote)
        # AbstainReason is an enum — serialize to its .value string
        if hasattr(vote_dict.get("reason"), "value"):
            vote_dict["reason"] = vote_dict["reason"].value
        elif isinstance(vote_dict.get("reason"), str):
            pass  # already a string (e.g. if asdict resolved it)
        else:
            vote_dict["reason"] = str(vote_dict.get("reason", ""))
        votes.append(
            {
                **vote_dict,
                # Attach diagnostic context — source captured at compute time,
                # not re-inferred from span dict (which may carry the field even
                # if the computed path was actually taken, or vice-versa in future).
                "tempo_ratio": tempo_ratio,
                "tempo_ratio_source": tempo_ratio_source,
                "claimed_stem": claimed_stem,
                "recording_id": recording_id,
            }
        )
        n_processed += 1

    # ---- write JSON -----------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(votes, indent=2))

    # ---- histogram ------------------------------------------------------
    label_counts: Counter[str] = Counter()
    n_abstain = 0
    for v in votes:
        if v["abstained"]:
            n_abstain += 1
        else:
            label_counts[v["label"]] += 1

    print()
    print("=" * 60)
    print(f"Operation-LF histogram (analysis only — no accuracy claim)")
    print(f"  N spans processed: {n_processed}")
    print(
        f"  Skipped: {n_skipped_no_ref} no-ref  "
        f"{n_skipped_no_mix} no-mix  "
        f"{n_skipped_degenerate} degenerate  "
        f"{n_skipped_load_error} load-error"
    )
    print(
        f"  tempo_ratio source: "
        f"{tempo_ratio_source_counts['computed']} computed(ref/set span)  "
        f"{tempo_ratio_source_counts['span_field']} from span field"
    )
    print()
    for label in ("keylock", "varispeed", "no_tempo_change", "unknown"):
        print(f"  {label:20s}: {label_counts[label]}")
    print(f"  {'abstained':20s}: {n_abstain}")
    print(f"  {'TOTAL':20s}: {n_processed}")
    print("=" * 60)
    print(f"wrote {out_path}  ({len(votes)} votes)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Apply keylock_vs_varispeed LF to every span in a predicted "
            "timeline and report the label distribution. "
            "ANALYSIS ONLY — no accuracy claim."
        )
    )
    p.add_argument(
        "--set-id",
        required=True,
        help="DJ-set identifier (e.g. 1fsnxchk for BB12)",
    )
    p.add_argument(
        "--timeline",
        type=Path,
        default=None,
        dest="timeline_path",
        help=(
            "path to <set_id>_predicted_timeline.json "
            "(default: alignment/out/<set_id>_predicted_timeline.json)"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        dest="out_path",
        help=(
            "output path for operation_votes.json "
            "(default: alignment/out/<set_id>_operation_votes.json)"
        ),
    )
    args = p.parse_args(argv)

    run_operations(
        args.set_id,
        timeline_path=args.timeline_path,
        out_path=args.out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Emit warp_prior.json — the empirical warp model the synthetic generator reads.

Validated by extract_warp_table.py (two channel-locked axes) + decompose_bpm.py
(acappella warp is derived from the BPM gap, 70-75% within ±5%). The generator
should NOT sample tempo_ratio directly; it samples BPMs and derives.

Artifact schema (consumed by workspaces/alignment_prototype/synthetic_mix/):
  bed_tempo:      near-native jitter for instrumental/regular beds
  overlay_warp:   DERIVED model for acappella/regular overlays + octave-fold prob
  cut_up:         segment/jump model for the instrumental bed channel (axis 2)
  pitch_shift:    discrete semitone distribution
  bed_bpm_sample: percentiles to draw a per-window master/bed BPM from
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from labeling.ground_truth.schema import load

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "labeling" / "fixtures"
SETS = {"bb11": FIX / "bb11_ground_truth.yaml", "bb12": FIX / "bb12_ground_truth.yaml"}
OUT = Path(__file__).resolve().parent / "warp_prior.json"
BPM_FILE = Path("/tmp/bpm.txt")


def load_bpm() -> dict[str, float]:
    by: dict[str, list[float]] = {}
    if not BPM_FILE.exists():
        return {}
    for line in BPM_FILE.read_text().splitlines():
        p = line.split("|")
        if len(p) == 3 and p[1] == "regular":
            try:
                by.setdefault(p[0], []).append(float(p[2]))
            except ValueError:
                pass
    return {r: statistics.median(v) for r, v in by.items()}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[int(q * (len(xs) - 1))], 4)


def main() -> None:
    tracks = []
    for name, path in SETS.items():
        for t in load(path).value.tracks:
            tracks.append((name, t))
    bpm = load_bpm()

    # --- axis 1: bed tempo jitter (near-native) ---
    bed_ratios = [
        t.tempo_ratio
        for _, t in tracks
        if t.claimed_stem in ("instrumental", "regular") and t.tempo_ratio
    ]
    bed_dev = [r - 1.0 for r in bed_ratios]

    # --- pitch shift distribution ---
    from collections import Counter

    pc = Counter(t.pitch_shift_semi for _, t in tracks)
    tot = sum(pc.values())
    pitch_dist = {str(k): round(v / tot, 4) for k, v in sorted(pc.items())}

    # --- axis 2: instrumental cut-up (segments + jumps) ---
    instr = [t for _, t in tracks if t.claimed_stem == "instrumental"]
    instr_multiseg = [t for t in instr if len(t.ref_segments) > 1]
    seg_counts = [len(t.ref_segments) for t in instr_multiseg]
    jumps = []
    for t in instr_multiseg:
        seg = sorted(t.ref_segments, key=lambda s: s.mix_start_s)
        for i in range(1, len(seg)):
            j = seg[i].ref_start_s - seg[i - 1].ref_end_s
            if abs(j) < 400:  # drop degenerate slivers
                jumps.append(round(j, 1))

    # --- bed BPM sampling pool (draw a master tempo per window) ---
    bed_bpms = [
        bpm[t.track_id]
        for _, t in tracks
        if t.claimed_stem in ("instrumental", "regular") and t.track_id in bpm
    ]

    prior = {
        "_provenance": {
            "sets": ["bb11", "bb12"],
            "n_spans": len(tracks),
            "validated_by": [
                "extract_warp_table.py (2 channel-locked axes)",
                "decompose_bpm.py (acap warp = mix_BPM/acap_BPM, 70-75% within ±5%)",
            ],
            "bpm_source": "pi-storage regular-stem Essentia (median per recording)",
        },
        "bed_tempo": {
            "comment": "instrumental/regular beds play near-native; sample tempo_ratio ~ N(1, sigma). Do NOT apply the old ±6% master_tempo_jitter. sigma is robust (from p10-p90 band); a few bed outliers make the raw stdev misleadingly large.",
            "model": "gaussian",
            "mean": 1.0,
            "sigma": round((pct(bed_ratios, 0.9) - pct(bed_ratios, 0.1)) / 2.563, 4),
            "sigma_raw": round(statistics.pstdev(bed_dev), 4),
            "p10_p90": [pct(bed_ratios, 0.1), pct(bed_ratios, 0.9)],
            "frac_within_pm05": round(
                sum(1 for r in bed_ratios if abs(r - 1) <= 0.05) / len(bed_ratios), 3
            ),
            "n": len(bed_ratios),
        },
        "overlay_warp": {
            "comment": "acappella/regular overlays: DERIVE, never sample. tempo_ratio = mix_BPM / payload_native_BPM. mix_BPM = current bed_BPM * bed_tempo_ratio. This formula already exists at scenario_v2.py:198 (tempo_ratio(host,payload)) but is dead (master_bpm=None).",
            "model": "derived",
            "formula": "tempo_ratio = mix_BPM / payload_native_BPM",
            "octave_fold_prob": 0.10,
            "octave_fold_comment": "with prob ~0.10, DJ picks half/double time to keep the vocal in a natural register — multiply/divide the derived ratio by 2. (Also masks Essentia octave errors; treat as approximate.)",
            "empirical_fit": {"within_pm05": 0.70, "within_pm05_with_fold": 0.75},
        },
        "cut_up": {
            "comment": "axis 2 — instrumental beds get chopped & jump-arranged (warp-marker-heavy at tempo_ratio~1). Generator's instr_jump_prob is directionally right; tune to these.",
            "multiseg_prob": round(len(instr_multiseg) / len(instr), 3),
            "segments_when_multiseg": {
                "min": min(seg_counts) if seg_counts else None,
                "median": statistics.median(seg_counts) if seg_counts else None,
                "max": max(seg_counts) if seg_counts else None,
            },
            "ref_jump_s": {
                "p10": pct(jumps, 0.1),
                "median": round(statistics.median(jumps), 1),
                "p90": pct(jumps, 0.9),
                "n": len(jumps),
            },
        },
        "pitch_shift_semi": {
            "comment": "small integer semitone; mixing-in-key keeps it tight",
            "dist": pitch_dist,
        },
        "bed_bpm_sample": {
            "comment": "draw a per-window master/bed BPM from these percentiles (empirical bed native BPMs across BB11+BB12)",
            "p10": pct(bed_bpms, 0.1) if bed_bpms else None,
            "median": round(statistics.median(bed_bpms), 1) if bed_bpms else None,
            "p90": pct(bed_bpms, 0.9) if bed_bpms else None,
            "n": len(bed_bpms),
        },
    }

    OUT.write_text(json.dumps(prior, indent=2))
    print(json.dumps(prior, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

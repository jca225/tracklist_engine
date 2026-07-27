# warp_analysis — empirical warp prior for synthetic mashup generation

Purpose: make the parallel agent's synthetic mashups
(`alignment/synthetic_mix/`) warp tracks the way real DJs
do, so the trajectory decoder trains across a smaller domain gap. Handoff is the
config artifact **`warp_prior.json`** — this module does NOT edit the generator.

## The finding (BB11 + BB12, n=316 GT spans)

Warp is **low-rank** and splits into two orthogonal, channel-locked axes — the
same direction in both sets (no BB11↔BB12 inversion):

1. **Tempo-stretch → the acappella channel.** Beds (instrumental/regular) play
   near-native (93% within ±5%, σ≈0.012). Acappellas carry essentially all the
   stretch (only 31% within ±5%). And that stretch is **derived, not free**:
   `tempo_ratio = mix_BPM / acap_native_BPM` holds for 70% of acappellas within
   ±5% (75% allowing a half/double-time octave fold); the residual tail is
   dominated by Essentia BPM octave-detection errors, not DJ behaviour. Baseline
   "predict no warp" gets only 31%.
2. **Cut/rearrange → the instrumental channel.** Beds get chopped into segments
   and jump-arranged (65% multiseg, median 3 segments) even at `tempo_ratio≈1` —
   warp-marker-heavy but not tempo-stretched.

Pitch shift is nearly discrete-trivial: `{0: 77%, ±1: 22%, ±2: <1%}`.

## Consuming `warp_prior.json`

The generator should **sample BPMs and derive warp**, never sample `tempo_ratio`
directly:

- `bed_bpm_sample` → draw a per-window master/bed BPM.
- `bed_tempo` → beds play at `tempo_ratio ~ N(1, 0.012)`. (Do **not** use the old
  ±6% `master_tempo_jitter` — beds barely move.)
- `overlay_warp` → each acappella/regular overlay: `tempo_ratio = mix_BPM /
  payload_native_BPM`, where `mix_BPM = bed_BPM * bed_tempo_ratio`. With prob
  ~0.10, apply a half/double octave fold. **The formula already exists at
  `synthetic_mix/scenario_v2.py:198` (`tempo_ratio(host, payload)`) — it's just
  dead because `_schedule_instrumentals` is called with `master_bpm=None`.**
- `cut_up` → instrumental bed segment-count and ref-jump distributions (axis 2).
- `pitch_shift_semi` → discrete semitone distribution.

## Scripts

- `extract_warp_table.py` — builds `out/warp_spans.csv` + `out/warp_segments.csv`
  from the fixtures (local, no pi-storage). Prints per-stem / per-span-type warp
  distributions.
- `decompose_bpm.py` — tests `tempo_ratio == mix_BPM/acap_BPM`. Needs
  `/tmp/bpm.txt` (`recording_id|stem|bpm`) pulled from pi-storage
  `track_audio_features` (regular stem).
- `build_prior.py` — fits and writes `warp_prior.json`.

## Caveats

- n=316 spans across 2 sets; marginals are robust, tail behaviour less so.
- `octave_fold_prob` ~0.10 is approximate — it's entangled with Essentia BPM
  octave errors and cannot be cleanly separated at this n.
- Re-fit when BB10 / Murph GT lands.

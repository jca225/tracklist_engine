# Sparse Multi-Stem Fingerprint Segment Aligner

**Status:** Phase 2 synthetic milestone complete; later phases proposed
**Objective:** turn whole-mix, stem-matched landmark correspondences into
piecewise constituent timelines with explicit starts, ends, jumps, overlaps,
and re-entries.

## Execution checkpoint

Phase 2 synthetic proof completed on `fp-hit-decoder-clean`:

- added typed raw landmark-match and bounded constituent-segment contracts;
- refactored the existing looptrace cover DP to expose explicit mix end
  boundaries while preserving its legacy return behavior;
- added a whole-mix local decoder with Hough candidates, a NULL state,
  multi-run output, frozen slope selection, and weak collision-run rejection;
- proved linear boundaries, NULL gaps, reference jumps, same-diagonal
  re-entries, tempo-slope selection, short repeated-pattern rejection, and
  no-match abstention on deterministic point clouds;
- proved synthetic audio through actual landmark hashing and matching recovers
  two separated, reference-jumped appearances.

This earns Phase 3 multi-channel corroboration. It does **not** yet establish a
real-set improvement or authorize tracklist attribution/default integration.

The collision-aware retrieval path and instrumental shadow CLI are also
implemented. Real-set shadow runs preserve the asymmetric diagnostic seen
before cleanup: BB11 contains strong recoverable paths and fewer false runs
after weighting, while BB12 remains weak and collision-prone. The segment bank
is therefore runnable but remains non-production.

Phase 3's channel-safe fusion contract is implemented: instrumental paths stay
geometrically fixed while an independently decoded full-mix/full-reference
path marks them corroborated, contradicted, or missing. Synthetic agreement,
corruption, and missing-channel tests pass. The real-set shadow gate fails,
however: full-channel agreement is useful on BB11 but does not distinguish its
false paths sharply, and on BB12 it corroborates false paths while missing the
few correct instrumental paths. Vocal evidence is correctly treated as
missing for this instrumental-only lane. No confidence acceptance rule is
therefore earned, and tracklist attribution remains blocked.

**Architecture correction:** full-channel corroboration is not part of the
intended aligner. The only legal primary lanes are independent
`mix_instrumental -> reference_instrumental` and
`mix_vocals -> reference_vocals`. They may occur at different times and are
never required to corroborate one another. `fp_segments.routes` enforces these
pairings and rejects full or cross-stem routes. The fusion code remains only as
reusable experiment output; the primary runner cannot select it.

Both strict lanes have now been executed independently on both complete-GT
sets. Instrumental landmark paths retain the previously recorded asymmetric
BB11/BB12 behavior. Vocal landmark paths are much weaker: the lane correctly
uses only `mix_vocals.flac` and reference `vocals.flac`, but sparse
constellation hashes do not survive vocal separation/processing reliably.
This rejects landmark FP as the vocal lane's representation, not the
vocal-to-vocal architecture. The vocal lane should keep the same routing and
segment contract while replacing landmark correspondences with HuBERT/phonetic
anchors.

**HuBERT peak follow-up (same day):**
`--lane vocal --observation hubert` builds whole-mix HuBERT-L9 cosine matrices,
sparsifies local peaks, and feeds `decode_constituent`. Synthetic peak gates
pass. Real BB11/BB12 shadow banks decode more often than landmark vocal but
still fail placement (false paths dominate). Ledgered as representation
NO-GO for *peak-sparsified* HuBERT; do not retune peak thresholds on these
sets. Phonetic/lyric anchors or a non-peak read of dense \(M\) remain open.

**Instrumental BB12 coverage fix + re-autopsy (same day):**
BB12 GT uses Ableton clip labels; the agentic timeline uses tracklist slots.
Naive zero-strip stem overrides silently mapped the wrong spans (Ableton
`007` → timeline `7`). `fp_segments.stem_overrides.timeline_stem_overrides`
now bridges `GT.track_id → timeline.recording_id` (nearest `set_start_s` on
ties). After the fix:

1. **Coverage:** BB12 instrumental override slots **20** (was ~5 false
   collisions); 1 GT instrumental remains a true inventory gap (no timeline
   recording). BB11 stays 22/22.
2. **BB12 scored recall@15:** **12/20 (0.60)**. Misses split into low
   GT-band support (ridge-absent / false decode) vs a few near-misses with
   support (e.g. ~18s).
3. **BB11:** unchanged **17/22 (0.77)** — misses still look like
   decoder/boundary/false-extra with strong GT-band support.

Next instrumental lever (honest slice): second observation (chroma) for
ridge-absent BB12 misses; false-run rejection for BB11-style near-misses.
Do not re-introduce slot-norm GT overrides for BB12.

**Instrumental chroma + secondary-run filter (same day):**
`--observation chroma` on the instrumental lane (mix_instrumental ↔ ref
instrumental chroma peaks → same decoder) is wired. Shadow recall@15 with
recording-id overrides: BB12 landmark 0.60 → chroma **0.65**; BB11 landmark
0.77 → chroma **0.64**. Mixed — chroma helps the weaker set, regresses the
stronger; not earned as the default observation. Raising
`min_run_evidence_fraction` from 0.05 → 0.25 (false-run rejection) regressed
both sets (BB12 0.45 / BB11 0.64); default stays 0.05 with an opt-in fraction
tested synthetically. Canonical SOTA unchanged.

**Shadow timeline materialize (same day):**
`fp_segments.materialize` writes decoded instrumental banks onto the agentic
baseline (`start_source=fp_segment_dp`). Real `score_timeline_vs_gt` vs that
baseline: instrumental traj can improve, but **overall placement regresses on
both sets** (ungated overwrite of good baselines). Ledgered NO-GO for ungated
promotion; needs an acceptance gate before any driver wiring.

**Gated materialize (`--gated` / `--gate-s 90`):** baseline-consistency filter
on segment `mix_start` (frozen 90s window). Reduces BB12 teleport damage vs
ungated but **still regresses overall set_start on both sets** (near-wrong
overwrites). Instr traj up; board promotion not earned. Do not retune gate_s
on BB11/BB12.

## Runnable shadow command

```bash
venvs/audio/bin/python \
  -m workspaces.alignment_prototype.fp_segments.run \
  --set-id 2nvzlh2k \
  --timeline workspaces/alignment_prototype/out/2nvzlh2k_agentic_baseline_gtstem.json \
  --mix-hash-cache eda/alignment/ridge_diagnostic/out/stem_mix_hash_cache \
  --stem-overrides labeling/fixtures/bb11_ground_truth.yaml \
  --output workspaces/alignment_prototype/out/fp_segments/2nvzlh2k.json
```

`--stem-overrides` is evaluation-only and compensates for historical timelines
whose `claimed_stem` field predates the materialization repair. Omit it for a
current correctly routed timeline. Output is a shadow segment bank and cannot
mutate a timeline or canonical state.

The independent vocal lane uses the same runner after preparing local-only
vocal hashes:

```bash
venvs/audio/bin/python \
  -m workspaces.alignment_prototype.fp_segments.prepare \
  --set-id 2nvzlh2k \
  --timeline workspaces/alignment_prototype/out/2nvzlh2k_agentic_baseline_gtstem.json \
  --lane vocal \
  --mix-hash-cache workspaces/alignment_prototype/out/fp_segments/cache/mix \
  --ref-fp-cache workspaces/alignment_prototype/out/fp_segments/cache/ref \
  --stem-overrides labeling/fixtures/bb11_ground_truth.yaml

venvs/audio/bin/python \
  -m workspaces.alignment_prototype.fp_segments.run \
  --set-id 2nvzlh2k \
  --timeline workspaces/alignment_prototype/out/2nvzlh2k_agentic_baseline_gtstem.json \
  --lane vocal \
  --mix-hash-cache workspaces/alignment_prototype/out/fp_segments/cache/mix \
  --ref-fp-cache workspaces/alignment_prototype/out/fp_segments/cache/ref \
  --stem-overrides labeling/fixtures/bb11_ground_truth.yaml \
  --output workspaces/alignment_prototype/out/fp_segments/2nvzlh2k_vocal.json
```

The operative pseudocode is:

```python
mix_hashes = fingerprint_whole_mix("mix_instrumental.flac")

for slot in instrumental_slots:
    ref_hashes = fp_index.load(slot.recording_id, stem="instrumental")

    matches = []
    for key shared by mix_hashes and ref_hashes:
        pair_count = len(mix_hashes[key]) * len(ref_hashes[key])
        if pair_count > PAIR_CAP:
            continue
        weight = 1 / log2(2 + pair_count)
        for mix_frame in mix_hashes[key]:
            for ref_frame in ref_hashes[key]:
                matches.append((mix_frame, ref_frame, weight))

    for slope in ALLOWED_SLOPES:
        diagonals = hough(matches, intercept=ref_time - slope * mix_time)
        support = local_weighted_support(diagonals, matches)
        support -= random_diagonal_background_floor(matches)
        path = viterbi(states=[*diagonals, NULL], emissions=support)
        runs = explicit_non_null_runs(path)

    emit(best_slope_runs_or_abstain)
```

Canonical benchmark values remain exclusively in
[`docs/alignment_status.md`](../../alignment_status.md). This plan defines gates
and artifacts, not new headline numbers.

## Problem statement

The current fingerprint path performs the right retrieval step but collapses
too early:

```text
whole separated mix + one reference stem
    -> matching landmark pairs
    -> offset histogram
    -> densest contiguous cluster
    -> one placement per tracklist span
```

That loses the evidence needed to decide where an appearance begins and ends,
whether the reference jumps, and whether the same constituent re-enters later.
It also lets a short repeated drum or synth pattern win as a globally strong
but musically false offset.

The target path is:

```text
whole mix channels + indexed reference channels
    -> sparse (mix_time, ref_time, channel, weight) correspondences
    -> candidate diagonal segments
    -> per-constituent NULL-aware local path decode
    -> tracklist-aware structured selection
    -> piecewise timeline
```

Fingerprinting remains the observation function. Dynamic programming becomes
the actor that decides which observations form real appearances.

## Existing code to reuse

- `landmark_fp.py`: legacy constellation and exact landmark hashes.
- `fp_index.py`: `(recording_id, stem)` reference fingerprint index.
- `mix_fp_hits.py::_vote_pairs`: raw exact-hash correspondence generation.
- `looptrace/landmarks.py`: pitch/tempo-tolerant landmark point clouds.
- `looptrace/segments.py`: Hough diagonal extraction, background subtraction,
  NULL state, segment-cover DP, and segment materialization.
- `harness/contract.py::RefSegment`: normalized piecewise alignment output.
- `candidate_arbiter/schema.py`: immutable placement/candidate provenance.
- `drivers/` and `score_timeline_vs_gt.py`: end-to-end race and canonical
  scorer.

Do not create a second fingerprint implementation or a second timeline schema.

## Load-bearing decisions

1. **Fingerprint the whole mix once per channel.** Never repeatedly fingerprint
   candidate windows during retrieval.
2. **Compare like with like, in exactly two independent lanes.**
   - separated instrumental mix against instrumental references;
   - separated vocals against vocal references.
   Full-audio and cross-stem routes are forbidden in this segment pipeline.
3. **Preserve raw correspondences.** Offset histograms may propose diagonals,
   but may not discard the underlying `(mix_time, ref_time)` points.
4. **Decode NULL explicitly.** No evidence must mean “not playing,” not the
   least-bad track or diagonal.
5. **Allow multiple local paths per recording.** A constituent may stop,
   re-enter, loop, or jump within its reference.
6. **Allow simultaneous constituents and independent stem appearances.** Track
   decodes are not mutually exclusive; vocals and beds may overlap or appear
   at completely different times.
7. **Use tracklist order as a soft global factor.** It corroborates and labels
   audio evidence, but cannot manufacture an appearance with no audio support.
8. **Fail closed.** The new decoder runs shadow-only until it improves both
   held-out sets without regressing protected slices.
9. **No BB11/BB12 threshold mining.** Configuration is set from synthetic data
   and frozen before the bidirectional real-set evaluation.

## Public contracts

Create `workspaces/alignment_prototype/fp_segments/`.

### `schema.py`

```python
@dataclass(frozen=True)
class LandmarkMatch:
    recording_id: str
    ref_stem: str
    mix_channel: str
    mix_time_s: float
    ref_time_s: float
    weight: float
    hash_frequency: int


@dataclass(frozen=True)
class DiagonalCandidate:
    recording_id: str
    ref_stem: str
    mix_channel: str
    slope: float
    intercept_s: float
    votes: int
    support_s: float
    background_margin: float


@dataclass(frozen=True)
class ConstituentSegment:
    recording_id: str
    slot_label: str | None
    ref_stem: str
    mix_channel: str
    mix_start_s: float
    mix_end_s: float
    ref_start_s: float
    ref_end_s: float
    slope: float
    evidence: float
    confidence: float
```

All values are finite and time bounds are ordered. `slot_label` remains
optional until tracklist attribution.

### `retrieve.py`

```python
def index_mix_channels(paths: MixChannelPaths) -> MixFingerprintIndex:
    """Fingerprint each available whole-mix channel exactly once."""


def retrieve_matches(
    mix_index: MixFingerprintIndex,
    reference: ReferenceFingerprint,
    *,
    channel_route: ChannelRoute,
) -> tuple[LandmarkMatch, ...]:
    """Return sparse time correspondences; do not choose an offset."""
```

Each hash match receives inverse-frequency weighting so ubiquitous landmarks
contribute less than rare landmarks. Keys whose Cartesian product exceeds a
fixed cap abstain, following `looptrace.landmarks.match_points`.

### `local_decode.py`

```python
def decode_constituent(
    matches: Sequence[LandmarkMatch],
    *,
    mix_duration_s: float,
    allowed_slopes: Sequence[float],
    config: SegmentDecodeConfig,
) -> tuple[ConstituentSegment, ...]:
    """Decode diagonal runs and NULL gaps across the whole mix."""
```

For each allowed slope:

1. Compute intercepts `b = ref_time - slope * mix_time`.
2. Extract Hough peaks as candidate diagonals.
3. Compute local support on a coarse mix-time grid.
4. Subtract a random-intercept background floor.
5. Run Viterbi over `{candidate diagonals, NULL}`.
6. Convert non-NULL runs into segments.
7. Split on reference jumps or slope changes.
8. Keep the slope/path with the strongest normalized inlier evidence.

Unlike the current `looptrace` call, the grid spans the complete set and can
emit several disjoint appearances.

### `tracklist_decode.py`

```python
def attribute_segments(
    segments_by_recording: Mapping[str, Sequence[ConstituentSegment]],
    slots: Sequence[TracklistSlot],
    *,
    config: TracklistDecodeConfig,
) -> TracklistAlignment:
    """Assign audio-backed segments to slots with soft order constraints."""
```

This is a second, sparse DP over slot order and segment proposals. Its state is
`(slot_index, proposal_index | SKIP)`. It rewards:

- local audio evidence and duration;
- agreement between full, instrumental, and vocal channels;
- plausible ordering of first appearances;
- compatibility with cue/MERT placement priors;
- adjacent-slot transition consistency.

It penalizes:

- assigning the same proposal to incompatible slots;
- large backward first-appearance jumps;
- unsupported slot assignments;
- implausibly short isolated fragments.

It must permit:

- `SKIP` for tracklist rows not audibly recoverable;
- multiple proposals for one slot;
- overlapping time ranges across different slots;
- zero ordering penalty for explicitly simultaneous/constituent rows.

### `materialize.py`

Convert attributed segments to the existing timeline contract:

```python
span["set_start_s"] = min(segment.mix_start_s)
span["set_end_s"] = max(segment.mix_end_s)
span["ref_segments"] = [
    {
        "mix_start_s": segment.mix_start_s,
        "mix_end_s": segment.mix_end_s,
        "ref_start_s": segment.ref_start_s,
        "ref_end_s": segment.ref_end_s,
    }
]
span["start_source"] = "fp_segment_dp"
```

If no accepted segment exists, preserve the baseline span byte-for-byte.

## Phase 0 — freeze fixtures and parity

**Files**

- Create: `fp_segments/schema.py`
- Create: `fp_segments/routes.py`
- Create: `tests/alignment_prototype/test_fp_segment_schema.py`
- Create: `tests/alignment_prototype/test_fp_segment_routes.py`

**Tasks**

- Define immutable contracts and validation.
- Encode the three like-for-like channel routes.
- Add explicit missing-channel abstention.
- Create tiny deterministic point-cloud fixtures for:
  linear play, two re-entries, reference jump, overlap, repeated-pattern
  distractor, and no-match.
- Prove exact-hash correspondences are identical to `_vote_pairs` before any
  weighting or filtering.

**Gate**

- Contract and route tests pass.
- Whole-mix hashing occurs once per available channel.
- Missing stems cannot silently fall back to an unlike channel.

**Commit**

`feat(fp-segments): add correspondence and routing contracts`

## Phase 1 — expose raw whole-mix correspondences

**Files**

- Create: `fp_segments/retrieve.py`
- Modify: `mix_fp_hits.py` only to expose a public, typed wrapper around raw
  pairs; retain legacy behavior unchanged.
- Modify: `fp_index.py` only if bulk/index iteration is missing.
- Create: `tests/alignment_prototype/test_fp_segment_retrieve.py`

**Tasks**

- Build one mix fingerprint index per channel.
- Retrieve matches for every tracklist reference on its routed channel.
- Attach inverse hash-frequency weights.
- Cap combinatorial/common-key explosions.
- Serialize correspondence banks under ignored
  `out/fp_segments/<set_id>/matches/`.
- Stamp schema version, producer SHA, audio identity, fingerprint mode, and
  channel route.

**Gate**

- Deterministic byte-for-byte correspondence banks.
- Legacy `offset_candidates` and `decode_placements` parity tests remain green.
- Runtime and memory are measured on both complete mixes before proceeding.

**Commit**

`feat(fp-segments): preserve whole-mix landmark correspondences`

## Phase 2 — local segment-cover decoder

**Files**

- Create: `fp_segments/local_decode.py`
- Reuse/refactor shared primitives from `looptrace/segments.py`; do not copy
  their implementation.
- Create: `tests/alignment_prototype/test_fp_segment_local_decode.py`

**Tasks**

- Generalize Hough and support computation to absolute whole-mix time.
- Add the explicit NULL state.
- Decode multiple non-contiguous runs.
- Recover start/end from support boundaries rather than the first/last raw hit.
- Support a frozen tempo-slope grid.
- Merge adjacent runs only when their diagonal and gap are compatible.
- Emit path evidence, background margin, coverage, and boundary confidence.

**Synthetic gates**

- Recover each fixture’s segment count and boundaries within fixed tolerances.
- Reject the repeated-pattern distractor when sustained context favors the
  true path.
- Return no segments for the no-match fixture.
- Preserve two simultaneous constituents when decoded independently.

**Commit**

`feat(fp-segments): decode local diagonal paths with null gaps`

## Phase 3 — multi-channel corroboration

**Files**

- Create: `fp_segments/fuse.py`
- Create: `tests/alignment_prototype/test_fp_segment_fuse.py`

**Tasks**

- Keep each stem channel’s path independent.
- Merge only time-compatible segments for the same recording/slot.
- Increase confidence for independent full/instrumental/vocal agreement.
- Preserve a strong single-channel path; absence of another stem is not
  disagreement.
- Penalize contradictory channel paths without averaging their locations.

**Gate**

- Synthetic full+stem agreement improves confidence without shifting correct
  boundaries.
- A corrupted channel cannot overturn two agreeing channels.
- A missing channel and a negative channel remain distinguishable.

**Commit**

`feat(fp-segments): fuse corroborating stem paths`

## Phase 4 — tracklist attribution DP

**Files**

- Create: `fp_segments/tracklist_decode.py`
- Create: `tests/alignment_prototype/test_fp_segment_tracklist_decode.py`

**Tasks**

- Implement `SKIP`, assignment, re-entry, and simultaneous-layer transitions.
- Use tracklist order only on first appearances and only as a soft factor.
- Add cue/MERT priors as optional emissions, never hard windows.
- Preserve multiple segments for one slot.
- Produce a trace explaining every reward, penalty, skip, and attribution.

**Gates**

- An out-of-order high-vote distractor loses to a sustained in-order path.
- Legitimate overlaps survive.
- A later re-entry does not get mislabeled as a later track.
- With all global-factor weights set to zero, output equals independent local
  decodes.

**Commit**

`feat(fp-segments): add tracklist-aware segment attribution`

## Phase 5 — shadow driver and scorer integration

**Files**

- Create: `drivers/fp_segment.py`
- Create: `fp_segments/materialize.py`
- Modify: `drivers/race.py` to register an explicit shadow driver.
- Create: `tests/alignment_prototype/test_fp_segment_materialize.py`

**Tasks**

- Materialize existing `ref_segments` without changing identity.
- Preserve the baseline byte-for-byte for abstained slots.
- Record per-span provenance and decoder trace path.
- Add an oracle report that separates:
  candidate recall, boundary recovery, attribution, and acceptance failures.
- Score strict and fiber-aware using the existing scorer.

**Gate**

- Output validates through `core.contracts.load_timeline`.
- Identity is unchanged from the input baseline.
- No accepted segment may be supported solely by a tracklist prior.
- Shadow mode cannot affect the default driver.

**Commit**

`feat(drivers): add shadow fingerprint segment aligner`

## Phase 6 — honest evaluation and promotion

### Configuration protocol

1. Tune decoder constants only on exact-label synthetic mixtures.
2. Freeze the configuration and artifact SHA.
3. Run bidirectional held-out evaluation on the two complete real GT sets.
4. Inspect failures only after the pass/fail board is written.
5. Do not change thresholds and rerun the same board as “validation.”

### Required reports

- candidate recall before decoding;
- segment boundary precision/recall;
- placement and trajectory results through the canonical scorer;
- results by `regular`, `instrumental`, and `acappella`;
- loops, jumps, overlaps, and re-entry slices;
- false-positive duration in NULL regions;
- baseline-regression list with provenance traces;
- runtime, peak memory, and cache size.

### Promotion gate

Promote only if the frozen decoder:

- improves the canonical board on both held-out sets;
- does not regress protected identity or strong baseline slices;
- improves or preserves instrumental and acappella slices independently;
- keeps false-positive segments below the predeclared synthetic threshold;
- passes `make check`.

If the gate fails, record the result in `attic/EXPERIMENTS.md`, retain reusable
correspondence/path artifacts, and leave the driver shadow-only.

## Execution order

```text
P0 contracts + fixtures
  -> P1 whole-mix correspondence banks
  -> P2 per-recording segment DP
  -> P3 multi-channel corroboration
  -> P4 tracklist attribution DP
  -> P5 shadow timeline driver
  -> P6 frozen held-out race
```

The first decisive checkpoint is the end of Phase 2. If the local decoder
cannot recover synthetic re-entries, jumps, boundaries, and NULL gaps from
known correspondences, global tracklist logic is not yet earned.

# Synthetic mix plan v2 — BB12-realistic generator (2026-06-25)

Phase 1 proved the **pipeline** (generate → MERT pretrain → BB12 ablation). It did **not**
prove transfer: 100-mix pretrain on Vast 4090 was flat on BB12 held-out (identity +0.0%,
MAE set_start +0.056s). Root cause is structural — we trained on the wrong distribution.

## Phase 1 result (baseline)

| Metric | Finetune-only | Pretrain→finetune | Delta |
|--------|---------------|-------------------|-------|
| identity_acc | 100% | 100% | +0.0% |
| MAE set_start | 33.251s | 33.307s | +0.056s |

Artifacts pulled locally:
- `workspaces/alignment_prototype/.cache/pretrain_synthetic_mert.pt` (100-mix MERT)
- `data/synthetic_mixes/synth_pretrain.log`

Vast instance `42571634` (`synth-pretrain`) destroyed.

## Distribution gap (measured)

| Property | Phase 1 synthetic (medium) | BB12 |
|----------|---------------------------|------|
| Mix duration | 90 s | ~61 min |
| Spans / mix | 2 | 166 |
| Instrumental slots | 1 continuous bed | 25 (19 multi-segment) |
| Acappella slots | 1 overlay | 100 |
| `is_loop` | 0 | 10 |
| `ref_segments` | 0 | 83 spans |
| Overlapping span pairs | 0–1 | 314 |
| `regular` stems | 0 | 41 |

Phase 1 = *one static bed + one vocal phrase*. BB12 = *switching host instrumentals,
stacked acappellas, loops, piecewise ref mappings, dense overlap*.

Pretrain loader (`synthetic_mix/corpus.py`) also ignores `ref_segments` and `is_loop` —
even perfect labels for loops would not train segment-aware heads today.

## v2 hypothesis

If synthetic pretrain matches **BB12 topology** (not just stem types), transfer to BB12
held-out improves. Minimum viable realism:

1. **Multiple instrumental slots** that hand off (not one 90 s bed)
2. **Overlapping acappellas** on shared/adjacent mix regions
3. **Loops** with repeated ref windows + `is_loop: true`
4. **`ref_segments`** on instrumentals (and loops) — same YAML schema as real GT
5. **Window length** aligned with real section scale (3–8 min windows sampled from
   longer timelines, or full 15–20 min mini-sets)

We do **not** need to synthesize a full 61-minute set on day one. We need windows whose
**span statistics** match BB12 (overlap rate, segment count, loop fraction, instrumental
switch rate).

## Target span statistics (from BB12)

Use these as acceptance gates for generated YAML (per window):

| Stat | BB12 | v2 target (per 5-min window) |
|------|------|--------------------------------|
| Spans | 166 / 61 min ≈ 2.7/min | 12–18 |
| acappella : instrumental | 100 : 25 | ~4 : 1 |
| Spans with `ref_segments` | 83 / 166 ≈ 50% | ≥40% of instrumentals + all loops |
| `is_loop` fraction | 10 / 166 ≈ 6% | ≥5% of acappellas |
| Overlapping pairs | 314 / C(166,2) ≈ 2.3% | match within 2× |
| Span duration median | 39 s | 25–50 s (acap), 60–120 s (instr) |

## Architecture (new modules)

Extend `workspaces/alignment_prototype/synthetic_mix/`:

```
timeline.py      # Slot timeline: ordered sections with start/end, overlap policy
sections.py      # Sample section types: intro, build, drop, loop-block, handoff
scenario_v2.py   # Build MashupScenarioV2 from timeline + catalog picks
render_v2.py     # Piecewise bed composite, loop repeat, stacked vocals, gain curves
labels_v2.py     # Emit ref_segments, is_loop, gain_curve (subset of BB12)
validate.py      # Compare window stats vs BB12 targets; fail generation if off
generate_v2.py   # CLI: --window-min, --spans-target, --curriculum bb12-lite|bb12-full
```

Reuse unchanged: `catalog.py`, stem paths, pi key/BPM fetch, output layout (`mix.flac`,
`mix_vocals.flac`, `mix_instrumental.flac`, `refs/`).

### Timeline model

```
Section 0: instrumental A  [0, 90s)     ref_segments: [A:0-45], [A:120-165]  # jump cut
Section 1: acappella X     [20, 55s)     linear ref
Section 2: acappella Y     [45, 80s)     overlaps X; is_loop + 3 ref_segments
Section 3: instrumental B  [80, 150s)    handoff from A; new bed
Section 4: acappella Z     [100, 140s)   on top of B
```

Rules sampled from BB12 empirical priors:

- **Instrumental handoff:** end previous bed with fade; start next at compatible BPM/key
  (reuse `compatible()` from `scenario.py`).
- **Overlap:** 30–70% of acappella spans overlap another active span (BB12 dense).
- **Loop block:** pick 4–8 s acap phrase; repeat 3–6 times with small ref_start jitter;
  emit one GT row with `is_loop: true` + N `ref_segments`.
- **Instrumental jump cut:** 2–4 `ref_segments` per instrumental slot (non-contiguous
  ref timeline — matches BB12 slots like 003, 155).
- **Slot labels:** mirror real convention — base slot for instrumental, `NNNw1` for
  acappella overlays on that section.

### Renderer changes

| Feature | Phase 1 | v2 |
|---------|---------|-----|
| Bed | single clip | piecewise segments + crossfade at handoffs |
| Vocals | 1–2 overlays | 3–8 with independent gain curves |
| Loops | none | repeat same ref slice with fade boundaries |
| Gain | fixed 4 s fade | simple 3-point gain_curve per span (match schema) |
| Channels | mix / vocals / instrumental | same; instrumental = sum of active beds only |

### Pretrain / loader changes (required for v2 to matter)

1. **`corpus.py`:** when `ref_segments` present, emit **one training example per segment**
   (or one multi-example span — match how BB12 finetune will eventually consume them).
2. **`pretrain.py` / `mert_features.py`:** document whether loss is per-segment or
   per-span; align with `alignment_prototype/CLAUDE.md` segment-list roadmap.
3. **Ablation unchanged** — same `pretrain.py --ablation` harness.

Without loader support, v2 labels are audit-only. **Phase 2b (loader)** is a hard
dependency before re-running ablation.

## Phased execution

### Phase 2a — Generator + validation (no GPU) — **DONE 2026-06-25**

**Goal:** 20 windows that pass `validate.py` stats + 5-ear human pass.

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.synthetic_mix.generate_v2 \
  --n 20 --window-min 5 --curriculum bb12-lite --out data/synthetic_mixes_v2
```

**Result:** 20/20 pass validation at `data/synthetic_mixes_v2/`. Typical window:
2 instrumentals (handoff + jump-cut `ref_segments`), 5–7 acappellas, 1 loop,
12–22 overlap pairs. Pretrain dry-run (20 mixes): 154 span rows → **265 train rows**
(segment expansion via `corpus.targets_for_mix`).

**Fixes applied (2026-06-25):**
- Block-level instrumental crossfade (no double-loud beds at handoffs).
- Loop training rows end at phrase length, not the parent fade tail.
- `iter_mixes` discovers `synthv2_*` (not just `synth_*`) when no manifest.
- `instr_jump_prob` 0.6→0.85 (lower reject rate: 107 attempts/20 vs 168).

**`regular` (full-song) stems:** capability wired end-to-end (catalog dual-stem
detection → `RegularSpan` sampling → render of summed inst+voc into all three
channels → `claimed_stem: regular` GT + `{rid}_regular.flac` ref → `corpus._ref_path`).
**Data-limited:** only 3 local tracks have BOTH stems, so regulars rarely pass
key/BPM compatibility and 0 landed in this corpus. Activates automatically as the
dual-stem cache grows (549 sets on pi). Curriculum knob: `n_regulars`.

**Gate remaining:** listen to 5 windows (`mix.flac` under `synthv2_0001` …).

### Phase 2b — Segment-aware pretrain — **DONE 2026-06-25**

`corpus.track_to_targets` expands each `ref_segments` row into one `SpanTarget`
per segment (loop iteration / instrumental jump-cut), so `build_examples` trains
the head on each (mix-window, ref-window) pair. Loop rows clamp to phrase length.
Verified: 154 spans → 265 examples. No decode change needed — the BB12 ablation
harness compares pretrain→finetune vs finetune-only on the same decoder.

### Phase 2b — Loader + segment-aware pretrain

**Goal:** pretrain consumes `ref_segments`; smoke on 20 windows locally.

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain \
  --synthetic-root data/synthetic_mixes_v2 --features mert --max-mixes 20 \
  --out workspaces/alignment_prototype/.cache/pretrain_synthetic_v2_smoke.pt
```

**Gate:** `examples=` count scales with segments (not just span rows).

### Phase 2c — Scale + ablation

**Goal:** 100–200 v2 windows, MERT pretrain on Vast, BB12 ablation.

Success criterion (unchanged): ≥5% identity lift **or** ≥2s MAE set_start improvement
on BB12 held-out.

**Kill criterion:** flat after 200 v2 windows → pivot away from synthetic pretrain;
document gap; invest in real GT / fingerprint / stem-wise channels instead.

### Phase 2d — Optional: BB12 template replay

Highest-fidelity mode: parse BB12 YAML → **replace** each span's audio with a different
catalog stem but **keep slot topology, overlap graph, and segment boundaries**. Labels
stay structurally identical; only recording_ids change. Proves topology hypothesis in
isolation.

```bash
# future CLI sketch
generate_v2 --template labeling/fixtures/bb12_ground_truth.yaml \
  --resample-ids --out data/synthetic_bb12_topology/
```

## Curriculum levels (v2)

| Level | Window | Spans | Loops | Instr handoffs | Overlap |
|-------|--------|-------|-------|----------------|---------|
| `bb12-lite` | 3 min | 8–10 | 1 | 1 | moderate |
| `bb12-med` | 5 min | 12–18 | 1–2 | 2 | high |
| `bb12-full` | 8 min | 20–30 | 2–3 | 3 | BB12-matched |

Start with `bb12-lite`; only scale after ear-check passes.

## Raw material (unchanged)

- 57 instrumental beds + 82 vocal payloads (local cache)
- 549 pi stem sets (scale)
- BB12 YAML as topology template + stats oracle
- Grids / BPM / key from pi (`catalog.py`)

## Risks

| Risk | Mitigation |
|------|------------|
| Renderer complexity | implement handoffs before loops; validate each layer |
| Disk / Vast sync | window-based corpus (~5 min × 200 ≈ similar to Phase 1 11 GB) |
| Segment pretrain not wired | Phase 2b before any GPU spend |
| Topology alone insufficient | Phase 2d template replay isolates variable |
| MERT still weak on placement | synthetic pretrain may only help identity; placement needs fingerprint/stem channels per CLAUDE.md |

## Related

- `docs/synthetic_mix_plan.md` — Phase 1 plan + kill criteria
- `labeling/fixtures/bb12_ground_truth.yaml` — topology oracle
- `workspaces/alignment_prototype/synthetic_mix/` — Phase 1 implementation
- `.cursor/skills/vast-jobs/SKILL.md` — Vast orchestration

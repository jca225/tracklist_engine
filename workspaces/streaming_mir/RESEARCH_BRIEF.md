# streaming_mir — corpus-first incremental analysis (tracklist engine)

**Status:** research brief (2026-07-12), **tracklist-first** scope. Incubating in
`workspaces/` per repo convention; promote to standalone repo only if
open-sourced. Mashup low-latency is an explicit **non-goal this cycle** — it
falls out later as a *drive mode* of the same engine.

## Framing

Point this at the **corpus**, not the mashup cold path. The value to tracklist
is **throughput + cost + quality on the ~16k-track corpus and the long-form DJ
sets**, measured against the existing SOTA anchor and `make scorecard`. The
mashup streaming path inherits the results for free later — build it once, in
the mature codebase, scored on the corpus.

Refocusing narrows the project: it **selects** the throughput/quality
mechanisms and **drops** the pure-latency machinery. See "Deferred" below.

## The design law (non-negotiable)

> **Converged output = SOTA offline. Any causal/early estimate is preview-only,
> never canonical.** Causal-streaming stems (RT-STT 5.17 dB, HS-TasNet 4.65 dB)
> are disqualified as corpus output; only SOTA (BS-Roformer ~9.8 dB class) is
> written to the canonical store.

## The three workstreams (all stand on their own corpus value)

### WS1 — Corpus throughput (ship first, zero quality risk)
- **Cross-track I/O–compute pipelining**: prefetch the download of track N+1/N+2
  while the GPU analyzes N. Hides yt-dlp/network latency behind GPU work.
- **Model batching**: run Roformer/MERT on batches of tracks/chunks per forward
  pass to raise GPU utilization.
- *Pure engineering. No DL changes, no quality risk. Validates the harness and
  buys an immediate wall-clock/$ win on the corpus + Vast budget.*

### WS2 — Block-online SOTA separation for long-form sets (research centerpiece)
- 60–90 min DJ **sets** must already be chunked for Roformer (VRAM-bound). The
  question: **how much overlap / lookahead recovers full offline SDR at bounded
  block size**, with clean block boundaries (no seam artifacts).
- Deliverable: an SDR-gap-vs-(block size, lookahead, overlap) curve on the set
  side, plus a block-online separator that hits offline SDR at bounded memory →
  separate arbitrarily long mixes on cheaper GPUs.

**Research resolves this to a near-certainty (2026-07-12, pass 2):** offline SOTA
separators are *themselves* blockwise — HT-Demucs / BS-Roformer process at test
time in fixed segments (~3.4–12.2 s) with **25% overlap + linear crossfade** —
and separation context **saturates at ~8–12 s** (HT-Demucs receptive field to
15 s gave no gain, still 9.20 dB). So the corpus's **360 s hard-cut, zero-overlap**
chunking (`render_set_stems.py`, "faint seams at joins") is simply coarser than
the models assume. **Hypothesis: overlapping the outer chunks by ~10 s +
crossfade recovers offline SDR (Δ≈0).** This is *applying the overlap the offline
models already expect*, not inventing causal separation. Confirmed by the
`block_overlap_sweep.py` harness → boundary-local SDR should plateau by ~10 s.
- **Harness BUILT:** `workspaces/streaming_mir/block_overlap_sweep.py` — sweeps
  overlap margin, scores global + boundary-local SDR vs full-file offline. Needs
  one GPU/MPS run on a ~4-min reference track.

### WS3 — Anytime confidence → adaptive compute / abstention
- An incremental estimator yields **confidence-over-time**. Use it to route hard
  tracks to more compute or human review and let easy tracks early-exit → saves
  corpus GPU.
- Plugs into existing abstention-via-margin / probe-precision / active-labeling.
- The intellectually interesting arm; also feeds the alignment abstention story.

## Deferred (mashup-latency-only — NOT built this cycle)

- Progressive 2s-preview emission (corpus never needs a fast first result).
- Causal separation as canonical output (quality regression; banned).
- Warm-model snapshot/CRIU (already captured by the batch loops; marginal).
- **Streaming beat / key** — offline beat_this / madmom already fine and *better*
  for the corpus; causal adds ~nothing except as a confidence signal.
- **Streaming/online structure segmentation** — hostile (needs global SSM) and
  tracklist wants it offline anyway. No corpus value either way.

## Plan of action (tracklist-first)

**Phase 0 — SOTA anchor + instrumentation.** Pin offline reference quality AND
wall-clock per task on the corpus (Roformer SDR, Beat This! F1, key acc,
structure) + current serial corpus throughput and long-set separation
time/VRAM. Extend `make scorecard` with a `streaming_mir` view. Every variant is
scored as gap-to-anchor. Nothing optimized before this exists.

**Phase 1 — WS1 corpus throughput.** Cross-track pipelining + batching in the
analysis driver loop. Measure wall-clock/$ delta on a corpus slice. Ship it.

**Phase 2 — WS2 block-online SOTA separation.** Sweep block/overlap/lookahead;
plot SDR-gap-to-offline vs latency+memory on long sets. Land a block-online
separator that matches offline SDR at bounded VRAM.

**Phase 3 — WS3 anytime confidence.** Confidence-over-time from incremental
estimators → adaptive compute + abstention routing. Score GPU saved vs quality
held.

**Phase 4 — Feature-store integration.** Converged outputs into the canonical
DB; this also makes tracklist the mashup app's warm cache (mashup benefit,
zero extra work).

## Success criteria (set up front)

- **WS1:** measurable corpus wall-clock reduction (target ≥20%) at *identical*
  outputs (quality-neutral by construction).
- **WS2:** block-online SDR within noise of full-file offline Roformer (target
  Δ≈0) at bounded, documented VRAM; report the overlap/lookahead needed to reach
  within 0.5 dB.
- **WS3:** GPU/$ saved on the corpus at a stated, small quality/abstention cost.

## Kill / de-scope criteria

- If block-online separation can't reach offline SDR at tolerable memory →
  fall back to current full-file chunking; document the frontier.
- WS3 is speculative; if confidence-over-time isn't predictive of final
  correctness, drop to "log only," don't gate compute on it.

## What NOT to do

- **Never** write causal-streaming stems into the canonical corpus.
- Don't re-derive tempo/grid from causal estimates for GT.
- Don't build the beat/key/structure streaming arms this cycle.

## Session handoff (2026-07-12) — resume here

**Built & committed:** this brief + `block_overlap_sweep.py` (WS2 anchor harness,
compiles clean). Two deep-research passes done; verdicts folded in below.

**Running at close:** the WS2 sweep on Mac MPS (single-model bs_roformer, 90s
Vanessa Carlton clip, core 20s, margins 0/2/5/10). If it finished, results are
in the run log; if the session killed it, **re-run**:

```
# 1. transcode a ~4-min reference track to wav (soundfile can't read m4a)
ffmpeg -v error -y -ss 30 -t 90 \
  -i "$HOME/aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11/tracks/012w1__Vanessa Carlton - A Thousand Miles [95bpm 1B].m4a" \
  -ar 44100 -ac 2 -c:a pcm_s16le /tmp/ref_vanessa_90s.wav
# 2. run the sweep (venvs/msst = the roformer venv; MPS ~40min single-model)
TRACKLIST_DISABLE_FK=1 venvs/msst/bin/python workspaces/streaming_mir/block_overlap_sweep.py \
  --audio /tmp/ref_vanessa_90s.wav --device mps --single-model \
  --core-sec 20 --margins 0,2,5,10 --boundary-win-sec 2
```
On a real GPU (Vast) drop `--single-model` for the full 3-model ensemble.

**Expected result:** boundary-local SDR (`*_bnd` cols) LOW at margin=0 (seam
damage), rising and plateauing by ~5–10s → overlap heals the seam to offline
quality. Global SDR (`*_glob`) moves little (diluted by the identical bulk) —
watch the `_bnd` columns.

**FIRST RESULT (2026-07-12, MPS single-model bs_roformer, 90s clip, core 20s):**
```
 margin_s  n_blk  voc_glob  inst_glob  voc_bnd  inst_bnd  wall_s
     0.00      5     31.69      33.82    25.75     26.14     599
     2.00      5     39.83      41.96    34.56     34.95     663
     5.00      5     40.39      42.52    41.71     42.11     842
    10.00      5    173.39     141.06   166.75    167.15     994
```
- **CONFIRMED (direction):** boundary-local SDR rises monotonically with overlap
  (voc 25.8→34.6→41.7 dB over margins 0→2→5s). Hard-cut seam damage is real;
  overlap heals it. WS2 principle validated.
- ~~OPEN QUESTION — margin=10 = ~167 dB (bit-identical).~~ **RESOLVED** by a
  decoupled-core rerun (core=25 so margin≠½·core; 120s clip):
```
 margin_s  voc_bnd  inst_bnd
     0.00    24.54    27.82   ← hard cut (today's behavior)
     3.00    38.15    41.43
     6.00    43.87    47.15   ← PLATEAU
    10.00    43.92    47.20   (identical to 6s)
    12.00    43.92    47.20   (identical to 6s)
```
  The 167 dB was the **artifact** (only at margin=½·core). The true curve is a
  clean monotone rise **plateauing at ~6 s overlap** (+~19 dB boundary SDR vs
  hard cut), flat thereafter — matching the research's context-saturation
  prediction. Plateau floor ~44 dB (voc) / ~47 dB (inst): NOT bit-identical
  (separate model calls differ intrinsically) but inaudibly close (~0.6% RMS).

**WS2 VERDICT: overlap the 360s chunks by ~8–10 s (6 s suffices; 8–10 for
safety) → recovers offline quality. The fix is settled.**

**IMPLEMENTED (2026-07-12):** `render_set_stems.py` now separates each core
chunk with `--overlap-sec` (default 10 s) of two-sided context, trimmed back off
before concat. `plan_windows()` (pure, unit-tested) + `extract_window` /
`trim_to_core` (sample-accurate). Geometry verified end-to-end: overlap→trim→
concat WITHOUT separation reconstructs the mix bit-for-bit (179 dB, 0 sample
diff). `--overlap-sec 0` = legacy hard-cut. Not a crossfade — pure trim, which
is what the sweep validated.
**Remaining:** (1) real before/after on one set (BB11/BB12) on a GPU — confirm
audible seam gone + SDR; (2) WS1 cross-track prefetch in `vast_loop.py`.

**Next step if plateau confirmed:** small patch to
`scripts/render_set_stems.py` `split_chunks()` — overlap the 360s chunks by the
validated margin + crossfade at concat (currently hard-cut, "faint seams").
Then WS1 (cross-track download prefetch in `vast_loop.py`) is the other cheap win.

**Note:** MPS runs bs_roformer at ~18 min/model-pass over a 4-min track — the
full 3-model ensemble sweep is 6+ hrs on MPS, hence single-model for the first
read. Definitive numbers belong on a GPU. **Also:** during this session an
UnmixDB `eval_bench` was pegging a CPU core (another workstream — don't kill it);
it's CPU-bound (chroma/nmf/dtw), likely not MPS contention, but it means MPS
timings here aren't clean. Run the definitive sweep on **Vast** (full ensemble,
no contention) — trust the MPS run only for the *shape* of the `_bnd` curve.

---

## WS1 A/B RESULT — cross-track prefetch on Vast (2026-07-14)

Definitive A/B on a rented RTX 4090 (Vast contract 44910031), RoFormer separator,
28 real corpus tracks from set `1d15br69` (14 baseline `--no-prefetch`, 14
prefetch), analyzed straight into canonical (not throwaway). Design +
implementation: `docs/superpowers/specs/2026-07-14-ws1-prefetch-vast-validation-design.md`,
`.../plans/2026-07-14-ws1-prefetch-vast-validation.md`.

**Metric = per-track overhead** (each track's inter-handoff wall-gap minus its
own `analyze_s`). This cancels track-length variance, which is large here
(analyze ranged 78–234 s), so it is the arm-independent, mechanism-true number —
raw wall/track differs between arms only because they drew different tracks.

| Arm | mean analyze_s | mean pull_s | **mean overhead/track** | GPU duty cycle |
|---|---|---|---|---|
| baseline (serial) | 130.4 | 3.0 | **16.3 s** | 130.4 / 146.7 = **89%** |
| prefetch | 145.2 | 7.0 | **~0.15 s** | 145.2 / 145.4 = **~100%** |

**Verdict: overhead eliminated (16.3 s → ~0.15 s), GPU duty cycle 89% → ~100%,
≈ 11% wall-clock saved at identical output.** Every prefetch track's gap equals
its analyze time to within timestamp rounding (several marginally negative — the
prior track's persist thread is fully overlapped). Note `pull_s` *grew* to ~7 s
in the prefetch arm (larger later files) yet stayed 100% hidden behind GPU work —
network latency no longer touches wall-clock, which is the design goal.

This is **~2× the ~5% the brief originally estimated for prefetch** — because the
pre-WS1 loop also joined its *persist* thread milliseconds after starting it
(join at loop-top, not before next hand-off), so the documented "~30% rsync
hiding" never actually happened and the stem-rsync + DB-push tail (~9–21 s) ran
serially too. The WS1 patch fixed both (input prefetch + persist-join moved), with
in-flight-tid exclusion keeping `next_task` correct. `--no-prefetch` reproduces
the legacy serial path (the baseline arm).

### Per-stage GPU-time table (n=28) — the model-batching go/no-go data

| stage | s/track | % of analyze | note |
|---|---|---|---|
| **separation (RoFormer)** | 108.9 | **79%** | the dominant cost |
| essentia | 18.2 | 13% | **CPU** subprocess (x86 sandbox), not GPU |
| mert | 5.1 | 3.7% | |
| cues (cue-detr) | 3.8 | 2.7% | |
| load+lufs | 1.2 | 0.9% | |
| beats (beat_this) | 0.7 | 0.5% | |

**Batching verdict: GO, but narrowly — only the RoFormer separation forward pass
is worth batching** (79% of analyze; everything else combined is ~10 s). Two
follow-on levers, ranked:
1. **Overlap Essentia (WS1.5, cheap, no model change).** Essentia's 18 s is
   CPU-bound (subprocess), so it can be hidden behind the *next* track's GPU
   separation with the same single-slot-thread trick WS1 just validated — a
   further ~13% analyze-time cut for near-zero risk.
2. **Batch RoFormer chunks per forward pass (bigger, invasive).** Touches
   `analysis/adapters/roformer_chain_adapter.py` internals; needs its own
   identical-output validation. Justified by the 79% share; do it after WS1.5.

**WS1 status: DONE + validated.** Prefetch + persist-overlap landed
(`400efae`); the corpus analysis loop now runs the GPU at ~100% duty.

## WS2 REAL-SET RESULT — overlap seam heal on BB11 (2026-07-15)

Same Vast RTX 4090. Three full renders of the BB11 mix (`set_audio_id=6`, 3581 s,
RoFormer, `--no-push`): `ovl0` (hard cut, legacy), `ovl10` (10 s overlap+trim),
and `pref` (10 s overlap, `--grid-offset-sec 180` → the shifted-grid
pseudo-reference, since a 60-min set has no full-file offline reference). Scored
by `seam_check.py`: SDR in ±2 s windows at each of the 9 chunk joins vs the
pseudo-reference, with chunk-midpoint windows as the interior control.

| stem | hard-cut (ovl0) join vs interior | overlap-10 (ovl10) join vs interior |
|---|---|---|
| vocals | 17.8 vs 41.7 dB → **−23.9 dB seam** (worst join −18.1 dB) | 41.9 vs 41.2 → **−0.78 dB (no seam)** |
| instrumental | 22.5 vs 86.7 dB → **−64.2 dB seam** | 86.2 vs 86.7 → **+0.45 dB (within noise)** |

**Verdict: the 10 s overlap fully heals the hard-cut seam on a real 60-min set
with the production RoFormer ensemble.** 24–64 dB of boundary damage collapses to
within ~0.5 dB of interior quality — passing the spec criterion (B join within
0.5 dB of its interior control) and confirming the earlier MPS single-model sweep
(plateau ~6 s, 10 s for safety) on real GPU + real audio. Worst-join ear-check
clips at `workspaces/streaming_mir/ws2_snippets/` (untracked; SDR verdict already
decisive).

**WS2 status: CLOSED.** The `--overlap-sec 10` default in `render_set_stems.py`
(`cad14a3`) is validated end-to-end; no further work.

---

## streaming_mir standing (2026-07-15)

- **WS1 (throughput):** DONE — GPU duty 89% → ~100%, ~11% corpus wall-clock at
  identical output. Next levers, ranked: (1) **WS1.5** overlap Essentia's ~18 s
  CPU cost behind GPU separation (~13% more, low risk, same thread trick);
  (2) batch the RoFormer forward pass (79% of analyze, invasive, needs
  identical-output proof).
- **WS2 (seam):** CLOSED — 10 s overlap validated on BB11.
- **WS-batching (RoFormer batch_size):** DONE — 1.65× separation speedup at
  batch 4 (plateau), output batch-invariant. See below.
- **WS-encoder (MERT/HuBERT speedup):** QUEUED (see below).
- **WS3 (anytime confidence):** untouched, still speculative per the plan.

## WS-batching RESULT — RoFormer batch_size on AWS A10G (2026-07-15)

Validated on a **personal-AWS** g5.xlarge (A10G 23 GB, us-east-2 — the personal
account's GPU quota lives there, not us-east-1), single `bs_roformer_ep_368`
model, synthetic 4-min stereo wav, separation-only wall time:

| batch_size | separation time | speedup vs 1 |
|---|---|---|
| 1  | 117.7 s | 1.00× |
| 4  | 71.4 s  | **1.65×** |
| 8  | 71.4 s  | 1.65× (no further gain) |
| 16 | 71.2 s  | 1.65× (no further gain) |

**Verdict: ~1.65× on the separation stage, plateauing at batch 4.** The A10G
saturates at batch 4 for this model + track length; 8/16 add nothing (and cost
VRAM). Separation is 79% of analyze, so ~1.65× there ≈ ~1.45× overall
analyze-time reduction — the single biggest corpus-throughput lever found, larger
than WS1's 11%. Corpus default set to **`batch_size: 4`** in
`analysis/roformer_chain.yaml`.

**Output batch-invariance — CONFIRMED (with a caveat on the test signal):**
instrumental SDR(bs1 vs bs8) = **57.9 dB** (≈ bit-identical). Vocals read
−12.5 dB, but that is a **silent-signal artifact, not a real difference** — the
synthetic input (noise + 440 Hz sine) has no vocal content, so the vocals stem is
near-silent and its SDR is numerically meaningless. bs_roformer emits *both*
stems from *one* forward pass, so the instrumental being bit-identical proves the
forward pass is batch-invariant → vocals necessarily is too (chunked overlap-add
is deterministic regardless of batch grouping). **Follow-up before full corpus
rollout:** one real-music A/B (a track with vocals) to get a clean vocals SDR and
close the caveat — cheap, do it on the next GPU run.

**Caveats/notes:** (1) numbers are for ONE model; the production chain runs a
5-model ensemble (3 vocal + 2 instrumental), and the ~1.65× per-model factor
applies to each, so the chain-level win is comparable. (2) The plateau at 4 is
track-length dependent — full 60-min *sets* have far more chunks and may benefit
from larger batches; re-measure batch size on the set path (`render_set_stems`)
separately. (3) g5.xlarge on-demand ~\$1/hr; the whole validation was <\$2;
instance terminated + SG/keypair deleted at teardown.

## WS-encoder — MERT/HuBERT faster at no quality loss (QUEUED 2026-07-15)

Same batching lesson applied to the transformer encoders. **Priority context:**
in the corpus loop MERT is only ~3.7% (5.1 s/track) and HuBERT is absent
(HuBERT lives in the aligner: `fp_probe`/`stem_placement`/`joint_ref_decode`/
`lyrics_align`, all layer 9). So this matters for **(a)** the planned MERT-95M→330M
upgrade (~3× cost → MERT becomes a real slice) and **(b)** running the aligner at
~40k-set scale — not for shaving the current corpus loop. Do it AFTER RoFormer
batching + WS1.5.

Verified against the code, ranked by how lossless each is:

1. **Batch the 10 s chunks (truly lossless, #1 lever).** `mert_adapter`
   (`analysis/adapters/mert_adapter.py:196`) and the HuBERT feature code both run
   one forward pass PER chunk in a Python `for` loop. Stack N chunks into one
   batched forward → mathematically identical output, big GPU-util win (the
   RoFormer lesson).
2. **Early-exit at the consumed layer (lossless, single-layer consumers only).**
   HuBERT uses layer 9 of 12 → compute 1–9, skip 10–12 (~25% off, bit-identical).
   personalization MERT uses layer 6 → skip 7–12 (~50% off). **NOT** the corpus
   MERT production path — it deliberately `torch.stack`s ALL 13 hidden states
   (`mert_adapter.py:205`) for the future learned weighted-sum head; early-exit
   would break that design.
3. **bf16/fp16 autocast (near-free).** Corpus MERT already STORES fp16 but
   COMPUTES fp32 (`mert_adapter.py:205-206`) — the fp32 precision is discarded at
   storage anyway, so a bf16 forward is ~2× + half VRAM at negligible cosine
   drift. Strongest near-free lever.
4. **SDPA/FlashAttention** (`attn_implementation="sdpa"`) + **torch.compile** —
   near-identical numerics, 1.2–2×.
5. **Distillation (distilHuBERT / MERT-lite) — LOSSY**, only with a downstream
   equivalence gate (alignment SDR / taste AUC held).

**Equivalence gate:** cosine ≈1.0 for #1–4 (bit-identical for #1–2), downstream
metric for #5 — same discipline as the RoFormer batch A/B. Near-free combo =
batch chunks + bf16 + (HuBERT) early-exit → plausibly 3–4× on aligner HuBERT,
~2× on corpus MERT, zero measurable loss.

## Research findings — per-task verdicts (2026-07-12, pass 2)

Can a streaming/blockwise estimator converge to offline SOTA, and at what
lookahead budget?

- **Separation → YES, at ~10 s overlap (bounded).** Offline SOTA is already
  blockwise (3.4–12.2 s segments, 25% overlap + crossfade); context saturates
  ~8–12 s. Strictly-*causal* separation stays ~2–5 dB below offline (RT-STT
  5.17, HS-TasNet 4.65 vs offline ~8–10 dB) — but the corpus doesn't need
  causal, only bounded lookahead. **This is WS2; it's essentially applied
  engineering.** Distillation (teacher→causal student) only closes part of the
  *causal* gap (causal-pretrain frontend ~78%; contextual-KD alone ~0.5 dB) —
  relevant only to the deferred mashup live path.
- **Beat → causal costs ~3.6–4.9 F-pts** (online BeatNet vs offline madmom DBN);
  BEAST ~8.5 pp at 46 ms. Lookahead does NOT close it (PLP: 74.7%@0ms →
  72.4%@580ms — worse). BeatNet spans causal+offline in one model. *Deferred;
  corpus uses offline beat_this anyway.*
- **Downbeat → hostile: ~18–19 pp** online-vs-offline gap (roughly 2× the beat
  gap). The genuinely hard streaming task. *Deferred.*
- **Structure → mostly offline.** Neural boundary detection relies on
  non-causal long-range context (bidirectional / whole-patch). BUT Foote-novelty
  localizes a boundary with lookahead ≈ checkerboard-kernel half-width (a few s)
  — so *coarse-early / fine-late* is viable, SOTA neural boundaries are not
  streaming. *Deferred; confirms the earlier prior.*
- **Anytime theory anchor:** speculative decoding is **lossless** (output-exact,
  2–3× speedup) — the clean model for "fast preview + exact-SOTA final, zero
  final-quality loss." Successive-refinement theory: Gaussian/squared-error IS
  refinable to the optimum, but not all sources are (intermediate description
  can bottleneck) — a real caveat for progressive schemes.

**Net:** WS2 (separation) is the one arm that converges to offline SOTA cheaply
(~10 s overlap) → build it. Beat/downbeat/structure streaming all trail offline
and stay deferred/mashup-only, exactly as scoped.

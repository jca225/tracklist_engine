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

**Next step if plateau confirmed:** small patch to
`scripts/render_set_stems.py` `split_chunks()` — overlap the 360s chunks by the
validated margin + crossfade at concat (currently hard-cut, "faint seams").
Then WS1 (cross-track download prefetch in `vast_loop.py`) is the other cheap win.

**Note:** MPS runs bs_roformer at ~18 min/model-pass over a 4-min track — the
full 3-model ensemble sweep is 6+ hrs on MPS, hence single-model for the first
read. Definitive numbers belong on a GPU.

---

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

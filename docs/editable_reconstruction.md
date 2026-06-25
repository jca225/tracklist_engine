# Editable mix reconstruction — the trio (placement · gain · EQ)

Recover, from a recorded DJ mix + its source tracks, the *editable* parameters a
DJ applied: **when** each track plays, **how loud** (gain/fade), and **EQ**. This
is the "reverse-engineering" half of the alignment objective. Built and measured
this session (2026-06-25); see [alignment_research_plan.md](alignment_research_plan.md).

## TL;DR — what works, what's validated

| ingredient | method | code | result (UnmixDB) | validated? |
|---|---|---|---|---|
| identity (which track) | landmark fingerprint votes | `landmark_fp.py`, `eval_bench.py --identity` | 82.6% rank@1 | ✅ |
| placement (when) | fingerprint offset + matched-filter agreement-gate | `eval_bench.py` method `fused`; `infer_fused.py` | median ~2.4 s | ✅ |
| **gain (how loud / fades)** | fingerprint-banded NMF | `nmf_baseline.recover_banded` | onset 3.8 s, **2% leakage** | ✅ (vs cue GT) |
| **EQ (per-band)** | banded-recon ratio in solo frames | `nmf_baseline.recover_editable` | bass-boost detected (low/high 0.74→1.08) | 🟡 direction only |

## How to run

```bash
# benchmark on UnmixDB (placement + warp + identity)
venvs/audio/bin/python -m workspaces.alignment_prototype.eval_bench \
    --unmixdb-root ~/data/unmixdb-v1.1 --max-mixes 150 \
    --methods fused,grid_mf,nmf --n-distractors 10 --identity

# fused inference on a REAL pulled set (no pi/MERT) -> timeline JSON + abstention
venvs/audio/bin/python -m workspaces.alignment_prototype.infer_fused --set-id 1fsnxchk

# gain + EQ per track: recover_banded(V, dicts, anchors) / recover_editable(...)
# anchors = {track_idx: (set_start_s, stretch)} from fp_offset (landmark_fp).
```

## Why fingerprint-banded NMF (the key idea)

A plain reference-conditioned NMF (model the mix as a non-negative sum of the
known source tracks) **fails on real audio** — spectrally-similar tracks splatter
their activation onto each other's regions (~22 s placement error). The fix is
**temporal continuity**: a track plays as one continuous diagonal streak. We get
that streak *for free* from the fingerprint (it tells us each track's start), so
we **band** the NMF activation to that diagonal. Cross-talk dies (2% leakage), and
the activation along the band **is** the gain curve; the per-band mix/reconstruction
ratio in solo frames **is** the EQ.

## Relation to the SOTA (André, Schwarz, Fourer 2024, arXiv 2410.04198)

Their code is **gone** — the paper links `github.com/etiandre/icassp2025-dj-transcription`
but the repo and the author's entire GitHub account 404 (verified 2026-06-25; never
mirrored, never archived by Software Heritage/Wayback). The only routes to their exact
method are emailing IRCAM (andre@ircam.fr / schwarz@ircam.fr) or reimplementing from the
paper. The dataset generator survives: `github.com/Ircam-RnD/unmixdb-creation`.
What their paper actually does, vs us:

- **Identity:** they *don't* — they punt to fingerprinting (out of scope). We do it.
- **Warp + gain:** their core. They use **IS-NMF** (Itakura–Saito, not our KL),
  a **multi-pass coarse→fine** scheme (halve the hop each pass) for memory + refine,
  and **inter-pass morphological line-filtering** (keep diagonal streaks, blur,
  threshold) for continuity — handling loops/jumps natively + learned noise (X̄H̄).
  We have a **simpler** banded KL-NMF; we *approximate* their continuity filter with
  the fingerprint band, and don't yet have multi-pass / IS-NMF / morphological filter /
  noise model. Loops/jumps we handle separately (`path_decode`), not in the NMF.
- **EQ:** their paper **explicitly disregards equalization** (lumps it into additive
  noise). We *attempt* per-band EQ — so on EQ we're ahead of the paper (but ours is
  direction-validated only; UnmixDB has no per-track EQ GT).

**Honest standing:** running their code = the SOTA baseline to *match* on warp+gain.
Matching it does **not** make us SOTA — to *beat* it we need our additions: the
**stem axis** (vocal/instrumental — their whole line is full-track-only), fused
identity, EQ, and open-set on real scraped sets.

## Validation limits (read before trusting numbers)

- Placement/identity validated on UnmixDB (synthetic, real benchmark). Real scraped
  sets (BB12/BB11) have no per-span GT except hand-labeled BB12.
- Gain validated against UnmixDB's linear-crossfade cue regions.
- **EQ not calibrated** — UnmixDB effects are global (no per-track EQ GT); we only
  show the method *detects* a band change in the right direction.
- Numbers are n≈150 mixes / 423 spans; small-n runs were optimistic.

## Next steps (to actually become SOTA, not just match)

1. **Reimplement their multi-pass NMF** (repo deleted — see above) on UnmixDB → the
   true SOTA baseline number, in-repo. Note our local UnmixDB is the *-excerpts* subset
   (6 base mixes, 40 s refsongs), not the full 1931-mix benchmark.
2. **Add the stem axis** — separate each recovered track into vocal/instrumental
   and recover gain/EQ per stem (the unclaimed differentiator).
3. **Adopt their wins where cheap** — IS-NMF + a morphological line-filter (drop-in
   for our band) + multi-pass, so we're not strictly behind on warp.
4. **Make it learned** (the §0 north star) once the signal-processing floor is solid.

## Code map

`landmark_fp.py` (fingerprint), `eval_bench.py` (harness + `fused`/`nmf` methods +
identity), `nmf_baseline.py` (`recover` / `recover_banded` / `recover_editable`),
`infer_fused.py` (real-set inference), `external/unmixdb.py` (loader).

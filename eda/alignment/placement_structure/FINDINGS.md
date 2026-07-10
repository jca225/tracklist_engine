# Placement objectivity, tempo drift, cue points — findings (2026-07-10)

Three questions, answered against BB11+BB12 hand-labeled GT plus a
verified-where-possible literature sweep (deep-research run, sources cited
inline; ⚠ marks claims whose adversarial verification was cut short — sourced
and quoted, but single-extraction):

1. Is there an **objective optimal placement** for overlaying an acappella on
   an instrumental?
2. How non-constant is **tempo** — in reference tracks and in mixes — and how
   much does the global-BPM assumption cost?
3. Are **cue-detr cue points** empirically useful, and what does the
   literature say about cue/section detection generally?

Reproduce:

```bash
PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/placement_grid.py
PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/tempo_drift.py
PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/cue_eval.py
```

JSON artifacts land in `out/`. Coverage caveat everywhere below: the local dev
DB is a BB11-era snapshot (~120 analyzed reference tracks), so ref-side tests
are BB11-weighted; mix-side tests cover both GT sets fully (315 usable spans).

---

## 1. Placement IS lattice-structured — but the lattice is the *grid*, not the entry time

**Setup** (`placement_grid.py`): every GT event (entry `set_start_s`, exit
`set_end_s`, loop-jump `ref_segments[i>0].mix_start_s`) is phased against the
mix bar grid (`data/analysis/<set_id>_measure_times.json`). Circular
statistics: resultant length R (0 = uniform/free placement, 1 = perfectly
grid-locked), Rayleigh p.

| Event class (pooled BB11+BB12) | n | beat-R | p | median dist to beat |
|---|---|---|---|---|
| regular entries | 71 | **0.41** | 4.6e-06 | 0.080 s |
| instrumental entries | 50 | **0.35** | 2.0e-03 | 0.093 s |
| loop jumps | 220 | **0.29–0.42** | ~1e-10 (per set) | 0.070–0.090 s |
| **acappella entries** | 192 | **0.08** | 0.32 (n.s.) | 0.100 s |

Three results, one picture:

- **Beds and cut points are grid events.** Regular/instrumental entries and
  especially loop-jump points concentrate on the beat lattice. Loop jumps are
  the most quantized event class in the data — when the DJ cuts, the cut is on
  the grid.
- **Acappella *entry times* are NOT grid events** — statistically
  indistinguishable from uniform phase. This is not evidence against objective
  placement; it's the vocal pickup (anacrusis): the clip starts where the
  *phrase* starts, which is routinely a fraction of a bar before the downbeat.
- **What is locked for acappellas is the grid itself, not the entry.** The
  phase-transfer test (mix-beat-phase minus ref-beat-phase at the entry
  instant, n=39 spans with local ref grids) shows instrumentals phase-lock
  hard (R=0.72, 67% within 0.1 beat, p=0.04); acappella n=11 is too small to
  conclude (R=0.31, n.s.) — but the warp side is already settled:
  `warp_analysis/decompose_bpm.py` showed acappella `tempo_ratio` is *derived*
  from the bed BPM. The acappella's beat grid is laid onto the mix grid; its
  clip boundary floats ahead of it.
- **Phrase lattice (acappella entry bar minus bed entry bar):** mod-4 residue 0
  at 37% (uniform 25%), mod-8 residue 4 at 23% (uniform 12.5%), n=188. Real
  but moderate concentration — smeared by exactly the pickup effect above
  (nearest-bar of an off-grid entry jitters ±1 bar).

**Literature agreement:** DJ practice codifies this. Zehren et al. (CMJ 2022):
viable EDM switch points lie on downbeats opening 4-bar periods ⚠. An aligned
DJ-mix corpus study found cue-point pair deviations peaking at every 32-beat
phrase, 40.4% within one measure, 73.6% within 8 measures ⚠. Practitioner
tooling (CueGen) snaps cues to bars with a 4-bar minimum spacing by default ⚠.
For mashups specifically, Lee et al. (ISMIR 2015) quantize placement to
2^κ-beat units on musical grounds ⚠; Huang et al. (AAAI 2021) treat ±1 s
downbeat misalignment as a *negative training example* — misplacement =
incompatibility ⚠.

**So does an "objective optimal placement" exist?** The evidence supports a
two-level answer:

- **Where** (fine): objective and grid-determined. Candidate placements live
  on the mix bar/phrase lattice; for vocals the constraint is grid-phase
  alignment (warp), with the audio onset floating a pickup ahead of the
  downbeat. This is a search-space collapse from ℝ to a lattice — and it is
  exactly why grid-lock was THE placement lever in the collapse-ladder
  experiments.
- **Which lattice point** (coarse): a *ranking* problem, not a constraint.
  The literature's compatibility scoring (AutoMashUpper's harmonic
  mashability; Lee's harmonic-change *balance* — complementary textures beat
  raw chroma similarity ⚠; Huang's learned stem compatibility, which beats
  AutoMashUpper in listening tests ⚠) says multiple lattice points are
  viable and DJs choose among them. GT shows the same: the mod-8 lattice has
  a mode, not a spike.

**Aligner implication:** placement decoding should (a) snap candidates to the
mix bar grid and penalize off-grid hypotheses for beds and cut points, (b) for
acappellas, score *grid-phase* alignment (warped-grid onto mix-grid) rather
than entry-time snap, and (c) treat the residual choice among lattice points
as the identity/evidence problem it already is. Windowing at ±(4·bar) around a
phrase hypothesis is empirically justified; sub-beat placement search is not
where the information is.

---

## 2. Tempo: individual tracks are nearly constant; the *mix* is not; the tracker is the third problem

**Setup** (`tempo_drift.py`): fold-corrected instantaneous BPM curves
(half/double-time tracker flips folded back; fold fraction reported
separately) from `track_analysis.beat_times_json` (119 tracks) and the 23
cached mix bar grids.

**Reference tracks (BB11-era sample, n=119):**

| Metric | Value |
|---|---|
| true BPM spread (p95−p5)/median, median track | **2.4%** |
| tracks with spread >5% | 9% |
| tracker fold fraction, median / p90 | 0.8% / **22%** |
| tracks with >5% of intervals fold-corrected | **29%** |
| anchored constant-BPM end-of-track error, median | 3.2 beats (1.8 s) |
| tracks desynced >0.5 beat by end | **76%** |

Reading: true within-track tempo drift is *small* for this corpus (studio
EDM/pop is DAW-rendered) — the user-observed "drift after the drop" on
constant-BPM assumptions is real but is dominated by (a) a minority of
genuinely variable tracks (live versions, older recordings: tail up to ~115
beats of accumulated error) and (b) **beat-grid measurement error**, which
also accumulates when you anchor once and extrapolate. Even a 2 ms anchor
error compounds to ~1 beat over a 6-minute track. The median 3.2-beat end
error mixes both causes; treat it as an upper bound on true drift but an
*operational* measurement of what a global BPM costs you.

**Mixes (n=23):** fold-corrected spread is 39–55% (p5–p95 ≈ 100→150 BPM) with
12–29 sustained tempo plateaus per set, and 11–20% of bar intervals still
needed fold correction. Both GT sets sit at median 127.7 with plateau counts
15–18. The mix timeline is **fundamentally a tempo map**, not a BPM — which
the repo already encodes (BB11 master-tempo export convention).

**Literature agreement:** rekordbox's ANLZ beat grid stores tempo **per beat**
(BPM×100 per entry) — the industry-native representation is already a
variable-tempo map ⚠. madmom's DBN makes tempo flexibility an explicit knob
(`transition_lambda`); constant-BPM is a modeling *choice* ⚠. beat_this
removes DBN postprocessing precisely to express variable tempo, but is
documented weaker on continuity metrics (CMLt/AMLt) than on F1 ⚠ — which is
exactly the instability our fold_frac measures (29% of tracks >5% flips), and
the known madmom/EDM failure of beat-1-vs-beat-3 downbeat confusion forced
prior cue work down to half-bar quantization ⚠. And the DJ-mix corpus prior:
86.1% of tracks in real mixes are tempo-adjusted <5%, 94.5% <10% ⚠ —
consistent with our warp prior (n=316).

**Aligner implication:** (a) per-track global BPM is fine as a *prior* for
studio tracks but must never be integrated over minutes — anchor+extrapolate
is guaranteed desync; (b) the mix side must stay a tempo map (it already
does); (c) beat-grid *post-processing* (fold-correction / continuity
enforcement) is cheap and removes most apparent "drift" — worth wiring into
`track_measures` derivation, since a quarter of raw grids carry half/double
flips that would poison any bar-indexed feature built on them.

---

## 3. cue-detr: weak-but-real entry prior; useless for loop points; unsnapped

**Setup** (`cue_eval.py`): 120 tracks with stored cues (median 4/track,
1.2/min). Downbeat coherence vs the beat_this grid; GT-entry utility vs a
seeded Monte-Carlo null (random position in the same track, same cue set).

| Test | Result | Null |
|---|---|---|
| cue within 0.25 s of a downbeat | **66%** | 26% |
| GT entry within 2 s of a cue — acappella | 40% (n=15) | 7% (lift 5.4×) |
| — instrumental | 40% (n=10) | 6% (lift 6.3×) |
| — regular | 31% (n=29) | 10% (lift 3.1×) |
| median entry→cue distance | 4–12 s | 15–22 s |
| loop-segment ref_starts near cues | **no lift** (0.9–1.1× @8 s) | — |

The memory "cue-detr = weak ref-offset prior; soft channel not gate" is now
quantified: a 3–6× lift at ±2 s on entries is genuinely informative, but with
40% hit rate and 4–12 s median miss it can only ever be a soft channel. For
loop-jump source points it carries no information at all.

The 34% of cues that sit off-downbeat are consistent with the pipeline having
**no grid snapping** — cue-detr postprocesses with plain `find_peaks`, no
beat/bar quantization ⚠, and its scores are min-max normalized per track, so
low-confidence tracks still emit peaks ⚠ (we also don't persist per-cue
confidence — only the global sensitivity).

**Literature context (the robustness question):** CUE-DETR's published
absolute accuracy is modest — F1 ≈ 0.36–0.39 at one-beat tolerance (verified
3-0), rising to precision 0.62 only at 16-bar phrase granularity (verified
3-0). It beats Automix (0.13) and Mixed In Key (0.22) on that benchmark
(verified 3-0), but its training GT is the cue conventions of just 4 DJs'
libraries, EDM-only, no audio shipped (verified 2-0). Meanwhile a *rule-based*
detector (novelty + 4-bar-period + salience, Zehren CMJ 2022) reports 86%
precision / 49% recall at 0.5 s on expert-annotated EDM ⚠ — with the caveat
that its rules break on non-4-bar-period tracks (36% of its false negatives)
⚠. Generic structure segmenters are poor cue proxies (MSAF ~30% precision on
the same data ⚠); the strongest current section models are All-In-One (joint
beat/downbeat/section, HR.5F 0.660 on Harmonix, demucs-stem input
load-bearing ⚠) and EDMFormer (2026; EDM-98 dataset, beats SongFormer on
EDM boundaries ⚠). rekordbox ships beat-indexed phrase labels (PSSI tag,
XOR-obfuscated but extractable) — a practitioner-scale EDM section labeler ⚠.

**Aligner implication:** keep cue-detr wired exactly as it is (soft entry
prior), but (a) snap stored cues to the fold-corrected downbeat grid at
persist time — free precision, since DJ-usable cues are grid events by
definition; (b) don't spend on cue-detr for loop/jump decoding; (c) if section
structure becomes load-bearing, the upgrade path is All-In-One/EDMFormer-style
joint models (or harvesting rekordbox PSSI as weak GT), not more cue-detr —
and our Foote-novelty `surprise` channel is the same family AutoMashUpper used
for phrase segmentation, so it's already the right rule-based complement.

---

## Verified vs unverified sources

Deep-research verification was rate-limited mid-run: 5 claims survived 3-vote
adversarial verification (all CUE-DETR benchmark numbers, marked "verified"
above); the ⚠ claims are single-extraction with source+quote but no
adversarial pass. Key sources: arXiv:2407.06823 (CUE-DETR), Zehren et al. CMJ
46(3) 2022 + arXiv:2007.08411 (rule-based cues, M-DJCUE), arXiv:2307.16425
(All-In-One), arXiv:2603.08759 (EDMFormer, EDM-98), Davies et al. 2014
(AutoMashUpper), ISMIR 2015 #302 (Lee, vocal-over-instrumental), AAAI 2021
arXiv:2103.14208 (Huang, learned stem compatibility), rekordbox ANLZ format
(crate-digger/pyrekordbox), madmom DBN docs, arXiv:2407.21658 (DJ-mix corpus
statistics). Full claim dump: deep-research workflow `wf_879bcfae-545`.

# A Re-characterization of DJ-Mix Alignment

**Date:** 2026-07-12
**Status:** framing document — the conceptual spine for the paper *and* the project's
roadmap. Numbers are harness-regenerated (see [alignment_status.md](alignment_status.md)
for provenance); every figure here traces to a scorer run, not prose.

---

## 0. The claim, in one sentence

> DJ-mix alignment is not a scalar task but a product of **near-orthogonal
> sub-problems — identity, placement, and structure** — that require different
> invariances, differ in difficulty between synthetic and real data, and
> generalize differently across mixes. Measuring alignment as one number on a
> synthetic benchmark **systematically overstates progress and hides the binding
> constraint.**

This is a claim about alignment *algorithms in general*, not about our system.
Our aligner is the **instrument** that produces the evidence; the **object** of
the paper is the problem's structure. (Framing decision — see §6.)

---

## 1. Why a re-characterization is needed

Prior work on recorded-DJ-mix alignment (André 2024 and the UnmixDB line) reports
alignment as essentially **one scalar** — placement/warp error — on **one
synthetic benchmark** (UnmixDB, algorithmically generated mixes). Two things
follow that we believe are wrong, and can now show are wrong:

1. **A single scalar conflates sub-problems that behave differently.** Getting the
   *right song* is a different task, with a different solution and a different
   difficulty, than getting the *right place* or the *right internal structure*.
2. **A synthetic-only benchmark stresses the easy axis.** UnmixDB models linear
   time-warp of single-instance spans. That is the sub-problem that is nearly
   solved. It does not synthesize the dimensions where real mixes are hard.

The result is a field that looks further along than it is: progress accrues on the
axis the benchmark measures, while the axis that actually blocks real-mix
alignment is neither exercised nor scored.

---

## 2. The three axes

A mix moment is a **sum of layers**; a song's appearance in a mix is described by
three near-orthogonal questions:

| axis | question | nuisance-invariance it needs | our primary channel |
|---|---|---|---|
| **Identity** | *which* recording is playing? | invariant to key, tempo, EQ, crosstalk | MERT (timbre); fingerprint; HuBERT for vocals |
| **Placement** | *where* in the mix does it start, and at what tempo? | invariant to the song's internal content | fingerprint diagonal + warp; HuBERT stem-placement |
| **Structure** | *which internal parts*, in what order, incl. repeats/loops/jumps? | invariant to *which instance* of a repeated section | looptrace / learned trajectory decoder |

They are near-orthogonal because they factor along the classic
timbre × harmony × language decomposition: **identity is a timbre/fingerprint
question, placement is a diagonal-offset question, structure is a
sequence question.** Solving one does not solve another — and, critically, they
do not share generalization behavior (§4c).

---

## 3. Two measurement failures this exposes — and the fixes

**(a) Span-exact scoring conflates "wrong content" with "wrong instance."**
On a repeated chorus, a decoder that lands on the *right chorus content* but the
*wrong occurrence* is scored identically to one that landed on unrelated audio.
That is a structure-axis error being counted as a total miss.

- **Fix — the fiber-aware metric.** Score within self-repeat classes ("fibers"):
  credit content-correct-instance-wrong picks. The gap this opens is the
  **structure residual**, and it is large: **+19 to +27 pp across all axes and
  both real mixes** (strict → fiber-aware). Roughly *half* of what span-exact
  scoring calls an acappella "error" is actually the right content at the wrong
  repeat — arguably correct for a mashup. The fiber metric is externally
  precision-validated (SALAMI P .88).

**(b) Synthetic-only benchmarks measure the easy axis.**
The same placement machinery that stalls on real mixes is near-solved on UnmixDB.
This is not a contradiction; it is the point — see §4b.

---

## 4. The evidence

All numbers harness-regenerated 2026-07-12. Two real mixes (BB11 = Episode 11 /
`2nvzlh2k`; BB12 = Volume 12 / `1fsnxchk`) + UnmixDB (141 mixes, 423 GT spans).

### 4a. Identity is solved-ish and is the easy axis

- BB11 **85%** (127/150), BB12 **84%** (128/152) span identity.
- On UnmixDB, fingerprint identity **73%** rank@1 against 50 distractors
  (chroma 38%).

### 4b. Placement is near-solved on synthetic, stalls on real — the punchline

UnmixDB placement (set_start MAE), ours vs reproduced baselines:

| method | set_start MAE | median | tempo MAE |
|---|---|---|---|
| **fused_resample (ours)** | **5.4 s** | 2.2 s | **0.045** |
| fused (ours) | 6.7 s | 2.4 s | 0.076 |
| dtw | 6.9 s | 2.6 s | 0.396 |
| no_warp | 9.1 s | 3.0 s | 0.050 |
| **nmf (André, reproduced)** | **20.2 s** | 13.6 s | 0.104 |

On real mixes the *same* placement stack gives median **4.8–5.3 s** but a heavy
tail (**p90 48–51 s**, only 66–76% within 15 s). Synthetic placement error is
small and *smooth*; real placement error is small-median but *heavy-tailed*,
because real DJs enter mid-song over dense medley beds — a regime UnmixDB does not
synthesize. **Caveat:** the NMF/DTW numbers are *our reproduction*, not André's
published protocol — so this is "beats strong reproduced baselines," not a SOTA
banner (see §6).

### 4c. Structure is the binding constraint AND does not generalize

Impact-weighted failure on real mixes: **placement 37% ≈ structure/decode-residual
38%** of all lost GT-seconds — co-equal walls; identity/tempo/route are minor.
Per-axis trajectory (strict → fiber-aware):

| set | acappella | regular | instrumental |
|---|---|---|---|
| BB11 | 14 → 34% | 25 → 51% | 32 → 55% |
| BB12 | 21 → 47% | 32 → 57% | 23 → 43% |

The learned structure decoder, evaluated **leave-one-set-out** (train on the other
mix, decode this one — zero in-set leakage):

| direction | Δ vs classical decode (strict) | 95% CI |
|---|---|---|
| train BB12 → decode BB11 | **+4.7 pp** | [+2.1, +7.4] (significant) |
| train BB11 → decode BB12 | **−2.1 pp** | [−6.5, +2.1] (negative, n.s.) |

The lever *works* in one direction and *hurts* in the other. Identity transfers
cross-set; structure does not. **This leg is n=2 — suggestive, not decisive.** We
present it as *"generalization is a necessary evaluation axis that current
benchmarks ignore, with preliminary evidence,"* not as a load-bearing claim (§6).

---

## 5. Applying it to the project

The re-characterization is not just paper framing; it reorganizes what we build
and measure.

**Evaluation becomes decomposed by construction.** The standard scorecard reports
per-axis (identity / placement / structure), **strict AND fiber-aware**, with the
oracle-vs-e2e decomposition (isolate placement from structure) and the
synthetic-vs-real contrast (UnmixDB vs BB). No single scalar. This is already what
the ablation harness (`experiments/`, `make align-ablate`) emits — the harness is
the operationalization of this document.

**Priorities fall out of the axes:**
- **Identity — solved; stop investing.** It generalizes and is ~84%. Remaining
  identity gaps are ingest/inventory, not the aligner.
- **Placement — small-median, heavy-tail.** The lever is the tail (mid-song
  entries over medley beds), not the median. Synthetic benchmarks will *not* show
  this; only real mixes will.
- **Structure — the binding constraint, and it does not generalize.** Therefore
  the lever is **not another hand-built probe** (sensor phase is frozen —
  consistent with this framing) but **the learned trajectory decoder + more
  ground-truth via the labeling flywheel.** The n=2 generalization asymmetry *is*
  the argument for scaling GT (BB10 and beyond).

**North-star restatement.** "audio + tracklist → Ableton-round-trippable
structure" is unchanged, but its progress is now read as *three curves, not one*:
identity (near-flat, high), placement (median-good, tail-open), structure
(the frontier). "99%" is an aspiration on the **structure** curve specifically,
gated on GT scale.

**What this predicts / falsifies.** If the re-characterization is right: (i) more
GT sets stabilize the structure decoder (falsifiable once BB10 lands); (ii)
methods that top UnmixDB will *not* move the real-mix structure wall; (iii) the
fiber-aware−strict gap persists until an instance-selection model closes it. Each
is a concrete experiment, not a vibe.

---

## 6. Open framing decisions (flag before the paper locks)

1. **How hard to lean on generalization (leg 3).** Default: keep as
   honest-preliminary motivation (n=2), submit on legs 1–2. Alternative: hold for
   BB10 and make generalization a full third leg. *This decides submit-now vs
   submit-after-third-set.*
2. **Object = problem, instrument = our system.** Default: yes (more general, ages
   better). Alternative: system front-and-center.
3. **Title-level claim (placeholder, wants a better one):**
   *"Synthetic benchmarks have made DJ-mix alignment look solved; decomposed,
   real-mix evaluation shows the hard problem is structural, not spectral."*

---

## Appendix — provenance

- BB ablation + per-axis + LOSO: `experiments/results/scores.db` via
  `make align-ablate` (2026-07-12); regenerate with
  `python -m workspaces.alignment_prototype.experiments.cli --fibers`.
- UnmixDB: `python -m workspaces.alignment_prototype.external.eval_bench
  --unmixdb-root ~/data/unmixdb-v1.1 --methods grid_mf,no_warp,nmf,dtw,fused,fused_resample
  --stratified --identity --n-distractors 50 --max-mixes 150`.
- Headline/loss-decomposition/fiber provenance: [alignment_status.md](alignment_status.md).
- Prior verdict this supersedes: the project memory once read *"SOTA = André;
  we're NOT SOTA"* — this document keeps that honesty (§4b, §6) while reframing
  the contribution away from the SOTA axis entirely.

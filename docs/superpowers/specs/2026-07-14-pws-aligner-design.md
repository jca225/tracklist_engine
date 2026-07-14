# Alignment as Open-Vocabulary Programmatic Weak Supervision — Research Synthesis + Design Spec

**Date:** 2026-07-14
**Status:** Design (approved for spec; no code this cycle)
**Branch:** `pws-alignment-reframe`
**Author:** John Abrahams (w/ Claude)

---

## 0. Thesis in one paragraph

Stop treating a recorded DJ mix as a *natural object* to be segmented and matched. Treat it as the
**output of a generative process**: `mix = {source tracks} × {a finite-but-open grammar of
tool-afforded operations}` (loop, key-lock tempo change, pitched acappella-over-instrumental,
spinback, sidechain, double-drop, …). Alignment is then **approximate Bayesian inversion** of that
process. The maturation move is to rebuild the aligner as a **programmatic weak-supervision (PWS)**
system: many small, abstaining **labeling functions** (one per operation-type, seeded from the
history of DJ/DAW tools) whose accuracies are **learned from unlabeled data without ground truth**
by an instance-feature-conditioned **label model**, audited by a **verifier**, and extended by an
**open-vocabulary discovery loop** that surfaces operation-types we have never seen. Manual Ableton
ground truth (GT) is **demoted from training target to grader/validation set**. This is why the path
forward is *not* "label Big Bootie 10" — one more labeled set helps one set; one more labeling
function helps all ~40,000.

The formal spine (Solms–Friston active inference): **precision-weighted prediction error ≡ per-LF
accuracy ≡ the repo's existing inverse-variance fusion** (`neuro/`). We are already doing PWS by
hand; the upgrade is to *learn* the precisions instead of hand-setting them.

---

# PART I — RESEARCH SYNTHESIS

Reading list (all in iCloud `AI/`, added 2026-07-13/14), grouped by role.

## I.1 Programmatic weak supervision — the training regime

### Data Programming (`1605.07723`, Ratner/De Sa/Wu/Selsam/Ré, NeurIPS 2016)
Model the *labeling process* generatively. Each labeling function `λ_i : X → {−1,0,1}` (0 = abstain)
has propensity `β_i` (prob. it votes) and accuracy `α_i` (prob. correct given it votes). Fit `(α,β)`
by **marginal MLE over the unlabeled set** (no `Y` observed), then train a **noise-aware** end model
against the resulting soft labels.

**The load-bearing guarantee (Thm 1).** Under LF conditional independence given `Y`, reaching
expected end-model loss `ε` needs

> `|S| ≥ (356 / ε²) · log(m / 3ε)`  ⇒  `|S| = Õ(ε⁻²)`

— the **same asymptotic sample complexity as fully-supervised learning**, except the `ε⁻²` *labeled*
examples are replaced by **O(1) labeling functions + `ε⁻²` unlabeled points**. §4 generalizes the
naive-Bayes network to a **factor graph** with user/learned dependency edges (similar/fixing/
reinforcing/exclusive); the same `Õ(ε⁻²)` scaling survives — richer structure is a *compute*, not a
*statistical*, cost. **This theorem is the formal justification for "add LFs, not GT sets."**

### Snorkel (`1711.10160`, Ratner et al., VLDB 2018)
Operationalizes Data Programming and answers *when the generative label model is worth it* vs a plain
majority vote (MV), via **label density** `d` and **modeling advantage** `A`:
- Low density: `E[A*] = O(d̄²)` → advantage vanishes → use MV.
- High density: `E[A*] ≤ e^{−2 p_t (2ᾱ*−1)² d̄}` → MV → optimal → use MV.
- **Mid density: the learned label model wins.** Snorkel's optimizer estimates this from label
  statistics and only switches on the generative model when the predicted advantage clears a
  tolerance. **Structure learning** selects LF correlations from the label matrix alone.

### FABLE — Leveraging Instance Features (`2210.02724`, Zhang/Song/Ratner, NeurIPS 2022) — **key paper**
Every prior label model aggregates *only* the LF votes, implicitly assuming LF correctness is
**instance-independent**. FABLE makes each LF's confusion pattern a **function of the instance
features** `f(x)`: it routes the EBCC subtype-mixture coefficient through a **Gaussian-process
classifier over features** — `π_{ikm} = softmax(σ(GP(f(x_i))))` — so nearby instances share
subtype/correlation patterns. Non-conjugacy (logistic-softmax link) is handled with a
λ/Poisson/**Pólya-Gamma** augmentation stack → closed-form variational updates; Lanczos low-rank
gives ~10× speedup.

**Diagnostic law (the design constraint):** instance-conditioning helps *iff* the LF confusion is
genuinely feature-dependent — the gain over EBCC is positively correlated (Pearson r = 0.469,
p < 0.01) with `Corr(X, LF-correctness)`. If LF accuracy is feature-flat, FABLE reduces to EBCC and
gains nothing. Beats neural feature-aware methods (Denoise/WeaSEL) precisely because those need a
gold validation set FABLE (and PWS in general) assumes absent.

### Confident Learning (`1911.00068`, Northcutt/Jiang/Chuang, JAIR 2022) — the auditor
Given noisy labels `ỹ` + out-of-sample predicted probabilities, directly estimate the **joint**
`Q_{ỹ,y*}` between noisy and true labels (per-class thresholds + collision handling), then rank &
prune the off-diagonal (mislabeled) mass and reweight the loss. **Consistent even under per-example
miscalibration** (Thm 2) because it depends on the *ranking* of probabilities relative to class
thresholds, not their calibration. Role here: a **quality gate on weak labels** before the end model
trains, and a calibration check on the label model using held-out GT.

### Weaver — Weak Verifiers (`2506.18203`, Saad-Falcon/Chen/Huang/Sala/Ré, 2025) — the selector
Same latent-variable-over-binary-votes skeleton as the label model, but the "LFs" are fixed noisy
**verifiers scoring candidate outputs**, and the target is **selection** (pick the best of K). Recovers
unobservable verifier accuracies by **method-of-moments** under `S_i ⊥ S_j | Y`, using a ~1% dev set
only to pin the class prior `Pr(Y=1)`. Closes 12–14.5% of a generation–verification gap with no GT.
Role here: the template for GT-as-a-tiny-dev-set, and for a verifier that *selects* among competing
alignment parses.

**The learning-theory throughline.** Model the noisy labeling process generatively → recover source
accuracies without GT → the substitution for hand labels is `Õ(ε⁻²)`-sound **whenever sources are
conditionally independent (or their dependence is modeled) and collectively informative.** FABLE
loosens "accuracy is a scalar" to "accuracy is a function of `x`"; Confident Learning certifies the
resulting joint; Weaver ports the apparatus from input-labeling to output-verification.

## I.2 Open vocabulary — discovering operation-types we don't know exist

### OVR-CNN (`2011.10678`) & Segment Anything (`2304.02643`) — the two mechanisms
1. **Open vocabulary = a shared embedding space.** Replace a fixed closed-set classifier head with
   **similarity to text/name embeddings of arbitrary classes**. A category never box-trained is
   handled because its *name* embeds into the same space the features were aligned to. Vocabulary is
   bounded by *language*, not by the *labeled* set — naming a new class is writing a string.
2. **Class-agnostic localize-then-name.** Localization ("a thing is *here*") is learned once,
   category-free, and generalizes to unseen classes; naming is deferred. SAM only proposes coherent
   masks; semantics come from the prompt. **A rare/unknown instance still gets localized — it just
   isn't named yet.** SAM also emits **K candidate masks per ambiguous prompt** (valid, not averaged)
   and bootstraps 1.1B masks via a **data engine** (assisted → semi-automatic → fully-automatic),
   amortizing a heavy image encode across cheap prompts.

### Mcity Data Engine (`2504.21614`) & Waymo Rare Example Mining (`2210.08375`) — the mining loop
Both find & prioritize **rare / long-tail / unknown** examples so you never inspect the whole dataset:
- Mcity: open-vocab NL query + ensemble **consensus filter** + **instance-count filter** → cut
  99.34% of the stream, label the 0.66% that matters → +17.45% mAP in one iteration.
- Waymo REM: **rareness ≠ difficulty.** Rare = epistemic (low density in learned feature space);
  hard = aleatoric (occlusion). Train a **normalizing flow** over per-object features; rareness score
  `r = −log p_θ(x)`; subtract a hard/degenerate filter (`r_i = h_i · v_i`) to mine *rare-not-garbage*.
  Mining 3% more data → +30.97% on rare intra-class subcategories.

**The reusable data-engine pattern:** bootstrap a small model → run it over the ocean cheaply →
score/filter by a tractability signal (confidence / consensus / feature-space rareness) → route only
the top slice to humans → merge → retrain → loop, with a self-reinforcing quality ratchet.

## I.3 The ontological spine (Solms–Friston, `Friston_Paper`)
Active inference: a self-organizing system models its sensory data as generated by hidden causes,
inverts that generative model to infer the causes, and acts to reduce **free energy** (= negative log
model-evidence = surprise). Prediction errors are weighted by **precision** (= inverse variance =
channel reliability). This yields the exact identity we exploit:

> **precision-weighting (Friston) ≡ per-LF accuracy (Snorkel/FABLE) ≡ inverse-variance fusion
> (`neuro/`).**

The reframe — "a mix is not observed, it is *generated by hidden causes we infer*" — is not a
metaphor; it is the same object the PWS label model estimates.

## I.4 The operation ontology (tool history → labeling functions)
A recorded mix is the output of a finite menu of tool-afforded operations. History yields ~40
primitives across 6 families, each with a DSP-detectable **signature** precise enough to seed an LF.

**Keystone finding — key-lock is the pivotal operation.** The CDJ **Master Tempo** (2001+) decoupled
tempo from key, breaking the vinyl *varispeed* weld. Clean discriminator:
- **varispeed** → pitch offset `= 12·log₂(r)` **predicted by** tempo ratio `r` (tempo & pitch coupled);
- **key-lock** → `r ≠ 1` but **chroma/pitch preserved** (transient smearing at large `|r−1|`).

This formalizes the repo's `key_change_breaks_chroma` and `warp_prior` memories. Second finding: a
large share of modern "sets" (Big-Bootie archetype) are **studio-produced Ableton mashups, not live**
— Ableton **Repitch** re-introduces varispeed while **Beats/Complex** preserve pitch, so one tool
spans both invariants; **live-vs-studio provenance is a latent, unlabeled variable that shifts the
operation prior.** Third: the audio→operation map is an **under-determined inverse problem** (backspin
vs FX spin-back; hand cue-jump vs quantized edit) → detectors must emit **soft multi-label +
abstention**, matching the repo's Fellegi–Sunter accept/review/abstain bands.

**Operation families (seed LF catalog):**

| Family | Example operations | Example audio signature (LF seed) |
|---|---|---|
| **A. Tempo/Time** | beatmatch, varispeed, key-lock tempo, warp, half/double, backspin, tape-stop | `r≠1` w/ vs w/o coupled pitch; sub-second negative-rate glide (backspin); monotonic pitch→0 (tape-stop) |
| **B. Pitch/Key** | key-shift, harmonic mix, pitched acappella-to-host | chroma rotated by integer `n`, grid unchanged; Camelot-adjacent keys at blend; integer-semitone offset vs acappella's own master |
| **C. Structure** | cue-jump, loop, loop-roll/slip, beat-jump, instant-double, medley reorder | exact bar-boundary self-similarity (loop); slip discontinuity on release; segment sequence = permutation of source arrangement |
| **D. Layering/Source** | acappella-over-instrumental, mashup (≥2 concurrent), bass-swap, stem-swap | two source identities concurrent; sub-band identity ≠ mid/high-band identity; single-stem identity diverges for a span |
| **E. Mix/Dynamics** | crossfade, EQ-kill, filter fade, fader ride, sidechain, LUFS-match | complementary amplitude envelopes; sharp band drop; centroid glide; **periodic kick-locked ducking** (sidechain) |
| **F. FX/Transition** | echo/delay throw, reverb tail, gate/stutter, noise riser/downlifter, drum-roll, vocal chop | beat-synced decaying repeats; exponential wet tail; square-wave gating; rising broadband centroid → downbeat |

**Open-vocabulary is mandatory**, for four structural reasons: (1) the tool space is non-stationary
(real-time neural stems, 2020–22, spawned a whole family un-enumerable in 2015); (2) operations
**compose** combinatorially with emergent signatures; (3) the inverse problem is ambiguous; (4)
live-vs-studio provenance is latent. Therefore: treat the ~40 primitives as a **strong seed set of
LFs**, and pair them with a novelty/unseen-mass channel (Good-Turing / capture–recapture, per the
`benchmark_certification` research) + a discovery loop that grows the vocabulary.

---

# PART II — DESIGN SPEC

## II.1 Decisions (fixed)
- **Phasing:** label-model-first (Phase 1), then open-vocab discovery (Phase 2). *(Chosen 2026-07-14.)*
- **Home:** new fork `workspaces/pws_aligner/` that **imports** (does not copy) `alignment_prototype`'s
  `harness/`, `agentic/`, `neuro/`, `fibers/` and rewires them as a PWS pipeline. The working aligner
  keeps producing `docs/alignment_status.md`; the fork must **beat it on the scorecard before
  promotion** (`workspaces_dir` convention).
- **No code this cycle.** Terminal step is `writing-plans`, not implementation.

## II.2 Architecture — three layers + interfaces

### Layer 1 — Labeling Functions (LFs)
One uniform contract extending the existing `Probe` ABC (`harness/contract.py:112`). Per mix span an
LF emits:
- a **distribution over `(offset/trajectory, operation-type)`** (not a single hard label);
- an **`abstain` flag with a *typed reason*** (`no_data` vs `low_margin` vs `out_of_domain`) — today
  abstention is lossy (`abstain=True` drops the "why"); the reason is needed for the label model and
  the discovery loop;
- an **instance-feature vector `f(x)`**: span embedding (MERT/HuBERT, cached once per set) + the
  matched-filter sharpness proxies already in `neuro/precision.py` (`margin`, `z`, `prominence`).

Existing probes (fp, HuBERT, chroma, lyrics, novelty, path-decode, fibers) are wrapped as LFs.
`fiber_gate`/`masking_gate` are **pulled out of fusion** (`agentic/belief.py:169–237`) — they are not
aggregators; they become either LF-internal refinements or verifier-layer signals.

**Open-vocab structure (staged, mostly Phase 2):** *localize* = class-agnostic "an edit happened
here" (Foote novelty / fiber onsets / fp-diagonal breaks — already exist); *name* = score the
localized span against **name-embeddings of operation descriptions**. Far-from-everything ⇒
**abstain-and-surface** (candidate unknown operation), never silent drop.

### Layer 2 — Label Model (Phase 1 core)
FABLE-style instance-feature-conditioned label model that **estimates each LF's accuracy conditioned
on `f(x)` with no GT** and outputs probabilistic labels + calibrated per-span confidence. It
**replaces hand-set precisions as the default** — critically, `harness/merge.py` currently ignores
precision entirely; that is the hole. It **models LF dependencies** (fp and chroma are both "content"
→ not conditionally independent) via Snorkel-style structure learning, else correlated evidence is
double-counted. The existing Beta-bandit (`agentic/learning.py`) is subsumed: it approximated per-LF
accuracy online; the label model does it offline, without GT, instance-conditioned.

### Layer 3 — Verifier + GT-as-validation
Confident-Learning auditor estimates the noisy→true joint over the weak labels and prunes confidently
mislabeled spans before the end model trains. **GT (BB11/BB12) is demoted to the dev/validation set**:
pins class balance (Weaver's ~1% dev-set role), calibrates the verifier, and **grades** the label
model. Never the training target.

## II.3 Data flow (Phase 1)
```
per set:  audio ──encode-once──▶ span embeddings (cache; reuse streaming_mir / render_set_stems)
spans ──▶ [LF_1 … LF_m] ──(vote dist + abstain-reason + f(x))──▶ label matrix Λ, features F
Λ, F ──▶ FABLE label model (accuracies learned on UNLABELED sets, no GT) ──▶ soft labels ỹ
ỹ ──▶ Confident-Learning verifier (prune/reweight) ──▶ training labels
GT(BB11,BB12) ──▶ validation only: class-balance prior + calibration + scorecard grade
```

## II.4 Phasing

**Phase 1 — label-model-first (build now).**
1. Uniform LF interface + typed abstention; wrap existing probes.
2. Span-embedding cache (encode-once amortization).
3. FABLE label model; wire as **default** fusion, learned on unlabeled sets; include **dependency
   structure learning** (not optional — see risks).
4. Demote GT to validation; add dev-set class-balance estimate.
5. **Density gate first:** run Snorkel's density/modeling-advantage check per span-population; where
   LF density is low, ship majority-vote (provably near-optimal) — do **not** deploy the label model
   there.
6. **Instance-conditioning gate:** for each LF, condition its accuracy on `f(x)` only when
   `Corr(X, LF-correctness)` is non-trivial (FABLE law). Otherwise keep it feature-flat.

**Phase 2 — open-vocab discovery (specced, deferred).**
Normalizing flow over span embeddings → rareness `r = −log p` × degenerate-span filter
(voiced-frac / gain gates) → mine rare-not-garbage. Ensemble **agreement-on-location +
disagreement-on-name** flags candidate unknown operations → top ~1% routed to `LISTENING QUEUE.als` →
human names → **vocabulary grows** → reflow. Good-Turing estimates remaining unseen-operation mass.
Emit **K candidate operation-parses** for ambiguous spans (SAM-style), keep the human-confirmed one.

## II.5 Learning-theory guardrails (operationalized as go/no-go)
- **`Õ(ε⁻²)` substitution** → prioritize adding LFs over adding GT sets. (Directional rule.)
- **Snorkel density gate** → label model only in the mid-density regime it provably beats MV.
- **FABLE `Corr(X,LF)` gate** → instance-condition an LF only where its confusion is feature-dependent.
- **CL consistency** → trustworthy weak-label auditing even under miscalibration.
- **Good-Turing / capture–recapture** → quantified unknown-operation mass, not hand-waving.

## II.6 Success criteria (Phase 1 — the falsifiable claim)
**Primary:** the learned label model **beats today's hand-tuned fusion on held-out BB11/BB12** on the
existing scorecard (trajectory + placement), **adding zero new GT.** Measured via `make scorecard` /
`score_timeline_vs_gt.py`.
**Secondary (calibration sanity):** the per-LF accuracies the label model learns **track the measured
`probe_precision_transfer` values** (e.g. fp's feature-dependent 0.90→0.53 swing) — catches
right-answer-for-wrong-reasons.
**Null result is informative:** if LF density is low enough that MV ≈ label model, the Snorkel regime
*predicts* no gain — that confirms the theory rather than refuting the approach; the lever is then
*more/denser LFs*, not more GT.

## II.7 Risks & costs of this recommendation
- **Conditional-independence is violated** by correlated content probes (fp+chroma); skipping
  dependency modeling → overconfident label model. *Mitigation:* Snorkel structure learning is a
  Phase-1 line item, not optional.
- **Thin LF density on hard spans** (e.g. heavy acappella) → label model degenerates to MV there; the
  density gate makes this explicit rather than silently wrong.
- **Latent live-vs-studio provenance** shifts the operation prior. Phase 1 treats it as an unmodeled
  nuisance; Phase 2 can surface it as a covariate.
- **FABLE cost:** GP + Pólya-Gamma VI is heavier than the Beta-bandit; Lanczos low-rank keeps it ~10×
  cheaper but it is the biggest new dependency. If it proves too heavy, fall back to
  EBCC/MeTaL (feature-flat) where the `Corr(X,LF)` gate says features don't help anyway.

## II.8 Repo integration map
| PWS primitive | Reuse | Build |
|---|---|---|
| LFs | `harness/contract.py` `Probe`, existing probes, `fibers/`, `neuro/precision.py` | uniform emit `(dist, typed-abstain, f(x))`; wrap gates |
| Label model | `agentic/belief.py` noisy-OR, `agentic/learning.py` Beta-bandit (subsumed) | FABLE (EBCC+GP+PG-VI); dependency structure learning; make it the default in place of `harness/merge.py` priority-arbiter |
| Verifier | margin/z/prominence gates, `unalignable` GT flag | Confident-Learning joint estimator + prune/reweight |
| Eval / GT | `score_timeline_vs_gt.py`, `make scorecard`, `set_ground_truth`, event log | GT→validation wiring; dev-set class-balance prior |
| Discovery (P2) | Foote novelty, fibers, `LISTENING QUEUE.als`, `benchmark_certification` unseen-mass | normalizing-flow rareness; open-vocab name-embedding; mining loop |

## II.9 Out of scope (YAGNI)
- No new GT authoring this effort (the whole point).
- No aligner-model architecture change beyond the LF/label-model/verifier wrapping.
- Phase 2 discovery is specced but not built this cycle.
- No changes to the shipped `alignment_prototype` until the fork beats it on the scorecard.

## II.10 Open question carried into planning
Whether the label model's end product feeds the **existing decode** (soft labels as priors into
`path_decode` / agentic belief) or trains a **separate noise-aware end model**. Data Programming's
value comes largely from the end model *generalizing beyond the LFs*; the plan should decide which,
and the density-gate result informs it.

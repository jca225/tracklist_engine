# PWS scale cohorts + the Big-Bootie ablation

Cohorts are set-lists that parameterize the **fit-on-unlabeled harness**: the label
model is fit on the *pooled votes* of a cohort's sets (unsupervised — no GT labels),
then graded on the GT sets (BB11 `2nvzlh2k`, BB12 `1fsnxchk`). Because the fit is
set-list-driven, an ablation is just two fits with different cohorts.

## The ablation (design: John, 2026-07-15)

Tests whether pooling across DJ archetypes helps or hurts the PWS label model.

- **Arm A — BB-only:** fit on the `big_bootie` cohort (27 sets, homogeneous
  Two-Friends studio-DAW-mashup archetype; `big_bootie.json`).
- **Arm B — BB + others:** fit on `big_bootie` + a sample of the ~582 other runnable
  non-BB sets (heterogeneous — live CDJ sets, different operation distribution).
- **Ablate** A vs B on:
  1. **σ-identifiability** (the leakage-free criterion) — does the calibration
     tripwire's rank-inversion clear, and does learned per-LF σ ordering recover
     (fp tightest)? This is the Gate-v3 blocker; more sets ⇒ denser co-voting ⇒ σ
     should become identifiable.
  2. **Held-out grade on BB11/BB12** (identity / placement / trajectory).
  3. **Per-LF accuracy drift** between arms — if a live set's LF accuracies differ
     from a mashup's, pooling degrades the homogeneous estimate → direct evidence
     the label model needs **FABLE instance-conditioning** (condition per-LF accuracy
     on archetype/rig features). If pooling *helps*, evidence for "one LF helps all
     40k."

**Predicted forks:** B-helps ⇒ scale to 609 with a flat label model. B-hurts ⇒
build the FABLE instance-conditioned lane before scaling.

## Sub-ablation (within BB)

The `big_bootie.json` `live: true` sets (`@ Big Bootie Land` — MSG, KIA Forum,
Boston, Chicago) are LIVE recordings of the mixes vs the studio versions. Studio-vs-
live within one artist isolates the live-vs-studio provenance covariate (the spec's
latent nuisance) with the source material held constant.

## Status

Deferred: the actual fits need per-set votes from `capture_votes`, which needs the
corpus-native capture path + a compute campaign (parked for the parallel speed
agent's pipeline work). The cohort roster + this design are locked so the campaign
runs exactly this experiment. BB11/BB12 votes already exist locally
(`1fsnxchk_probe_votes.json`) for harness smoke-testing.

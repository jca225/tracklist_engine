# Alignment — canonical status (single source of truth)

> **Numbers regenerated 2026-07-11 at commit `eb21a5e`** (§1–§3 re-run and verified
> unchanged from the initial `bd44417` regeneration) via the command block in
> §Regeneration below. **This doc owns every current alignment headline number.**
> Other docs cite it; they do not re-state numbers. If a number here is stale,
> re-run §Regeneration — do not hand-edit. Drift found while building this doc is
> logged in [alignment_status_corrections_20260711.md](alignment_status_corrections_20260711.md).

**Set ids:** **BB11 = `2nvzlh2k`** (Two Friends – Big Bootie Mix *Episode 11*),
**BB12 = `1fsnxchk`** (*Volume 12*). Only two GT sets exist; generalization
evidence is LOSO (below), never a cross-set CI from n=2.

**Metric definition (stated once):** `traj-acc` = mean fraction of a span's
GT-seconds whose decoded ref-time lands within **±2 s** of truth, seconds-weighted.
**strict** = each occurrence must be the exact GT one; **fiber-aware** = credits a
decode that lands on the *right repeated content* (chorus/loop class) even if the
occurrence index differs — see §The strict→fiber gap. Unless noted, all numbers
are on the **`_lt` (looptrace) timeline** scored by `make scorecard` /
`score_timeline_vs_gt`, the module's declared source of truth.

**Interpretive frame:** these numbers are read through the decomposed account in
[alignment_recharacterization.md](alignment_recharacterization.md) — alignment as
three near-orthogonal axes (identity / placement / structure) with different
difficulty on synthetic vs real data and different generalization. Report per-axis,
strict **and** fiber-aware; never a single scalar.

---

## 1. Headline (regenerated on `_lt`, 2026-07-11)

| | BB11 (`2nvzlh2k`) | BB12 (`1fsnxchk`) |
|---|---|---|
| **Identity** (span recording correct) | 124/150 (83%) | 128/152 (84%) |
| **set_start** placement, median / <15 s | 7.9 s / 63% | 6.3 s / 68% |
| ref-offset MAE, straight clips (median / p90) | 8.7 s / 143.6 s | 22.9 s / 123.4 s |
| **Trajectory — multiseg+loop headline** (strict → fiber-aware) | **16% → 37%** | **19% → 38%** |
| &nbsp;&nbsp;stem: acappella | 12% → 30% | 10% → 33% |
| &nbsp;&nbsp;stem: regular | 23% → 51% | 31% → 57% |
| &nbsp;&nbsp;stem: instrumental | 20% → 48% | 16% → 36% |
| GT-seconds lost (corpus, both sets combined) | **85%** (13176 / 15433 s) | ← both sets |

**Loss attribution** (binding cause, seconds-weighted, both sets, `_lt`):

| cause | % of loss | note |
|---|---|---|
| decode-residual | **38%** | "which chorus" repeat-instance wall |
| placement | **37%** | co-equal binding wall (was mis-typed as ~31%, see corrections C2) |
| mis-route | 9% | stale `set_track_slots` stem axis (score from GT stem, not timeline) |
| identity | 6% | |
| tempo/octave | 4% | |
| instance-ambiguity | 4% | |
| loop-instance | 2% | |

Acappella is **51%** of corpus mix-seconds and the worst axis; acappella-multiseg
alone is **34%** of all loss.

---

## 2. The strict → fiber-aware gap (named contribution, not a caveat)

Fiber-aware − strict ≈ **+19 to +22 pp** on the multiseg+loop headline, holding
across **both sets, all three drivers, and both timelines**. This gap *is* the
"which-instance" residual: the decoder lands on the correct repeated content
(right chorus) but not always the exact occurrence. It is externally
precision-validated (SALAMI **P .88**; low recall R .06 is precision-first-by-
design on a jam-band pessimistic floor, **not** a limitation verdict). Fibers v4
(2026-07-09) fixed the acappella recall hole (vocal coverage 0.06–0.28 → 0.33–0.73,
ear-validated by John); the phase-cancel clone certificate is wired. **This +20pp
is a finding, not a footnote** — under-crediting fibers as a "scoring util" was
the original trigger for this overhaul (corrections C5).

**Timeline caveat (corrections C4):** the fiber headline is **45 (BB12) / 40
(BB11) on the base classical timeline** (race board) but **38 / 37 on `_lt`**.
Looptrace `_lt` does *not* uniformly beat base classical on fiber-aware
trajectory (it regresses BB12 45→38). Trust the **lift** (+20pp), not any single
absolute. Which composition is "current best" for fiber trajectory is unresolved
— flagged, not fixed, in this docs pass.

---

## 3. Oracle placement ceiling (isolates placement from decode)

`path_decode --eval` on acappella (HuBERT decode, GT placement given), n=21:
**strict 37% / fiber-aware 61%.** Real end-to-end acappella (`_lt`) is 10–12%
strict / 30–33% fiber-aware → a **~+28 pp** placement gap. Placement is a real
binding wall, consistent with the 37%-of-loss attribution. The `fiber_gate`
validated in the same run (unflagged − flagged instance-acc = +17 pp, GATE VALID).

---

## 4. Driver race board  ⚠ carried 2026-07-10 (NOT regenerated — see below)

From [agent_handoff_fibers_20260710.md](agent_handoff_fibers_20260710.md); base
classical timeline. A fresh `make race` re-runs `infer` (expensive) and mutates
timeline files, out-of-scope for a docs pass.

| set | driver | place (s) | ref (s) | strict% | fiber% |
|---|---|---|---|---|---|
| BB12 | classical | 4.8 | 15.6 | 21 | 45 |
| BB12 | **agentic** | **3.3** | 15.6 | 21 | 45 |
| BB12 | ml (ungated) | 4.8 | **9.6** | 19 | 39↓ |
| BB11 | classical | 6.8 | 6.2 | 20 | 40 |
| BB11 | **agentic** | **1.9** | 7.8 | 20 | 40 |
| BB11 | **ml** | 6.8 | **2.9** | **23** | **41** |

Reading: **agentic = placement champion** both sets; **ml = ref-offset champion**
where classical is weak (BB11), but regresses BB12 without `--ml-gate`. Composed
default target = agentic placement + gated-ml decode.

---

## 5. Method / component registry

| method | status | role | current signal | deep doc |
|---|---|---|---|---|
| landmark fingerprint | wired | placement (diagonal offset) | localizes diagonal 0.2 s / 76%; gate `--fp-placement-gate-s 90` | module CLAUDE.md |
| HuBERT stem-placement | wired | acappella set_start | `--stem-placement`; BB12 <15 s 61→76% | [project_per_stem_hubert_setstart] |
| MERT | wired | **identity only** (cannot localize) | 83–84% span identity | module CLAUDE.md |
| chroma matched-filter | wired | instrumental ref-offset | weak axis (set_start-under-crosstalk) | — |
| lyrics-align | wired | acappella ref-decode | ~50% coverage; loses loops; fuse not replace | [project_lyrics_ref_decode] |
| looptrace (`_lt`) | wired | acappella loop-collapse decode | source-of-truth timeline; **regresses BB12 fiber headline (C4)** | looptrace/NOTES.md |
| fibers v4 | wired (scoring) | which-instance credit | **+20 pp fiber-aware**; recall hole fixed; SALAMI P .88 | [fiber_validation_findings] |
| phase-cancel clone cert | wired | clone detection | BB12/BB11 clone pairs flagged | fibers/NOTES.md |
| agentic driver | wired | end-to-end (placement) | placement champion (BB11 1.9 s, BB12 3.3 s) | drivers/ |
| ml driver | wired (gated) | end-to-end (ref-offset) | ref champion on BB11 (2.9 s); needs `--ml-gate` | drivers/ |
| co-train / LOSO | built (n=2) | cross-set head | **identity transfers 100%, placement does NOT** | [cotrain_loso_findings] |
| trajectory decoder | in-progress | the current lever | learned segment-trajectory; the actor, not perception | trajectory/ |

---

## 6. Binding walls & current lever

Two co-equal walls: **placement (37% of loss)** and **which-instance
decode-residual (38%)**. LOSO proof (n=2): identity transfers 100% cross-set both
directions; **placement does not** (the MERT head memorizes placement per-set).
The lever for transferable placement is therefore **not** the MERT head — it is
the **learned trajectory decoder + the agentic pseudo-label flywheel** (+ a third
GT set to unlock the learned instance selector). Sensor phase is **frozen
(2026-07-09)**: the channel inventory is rich enough; the wall is the actor. Dead
ends live in [attic/EXPERIMENTS.md](../workspaces/alignment_prototype/attic/EXPERIMENTS.md)
— read the verdict before re-testing.

---

## 7. Paper framing (decided 2026-07-11)

"99% accuracy" is an **aspirational north star** (design target via the
trajectory decoder + flywheel), **not** the paper's empirical claim. The paper
reports the real numbers above + the methodology, and positions 99% as the
target. Recorded here so it stops being re-litigated.

---

## Regeneration (copy-paste; re-stamp the header when you run it)

```bash
# repo root, venvs/audio/bin/python. BB11=2nvzlh2k, BB12=1fsnxchk.
make scorecard                       # attribution + per-axis + identity + set_start (_lt)
for sid in 1fsnxchk 2nvzlh2k; do
  tl=workspaces/alignment_prototype/out/${sid}_predicted_timeline_lt.json
  venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \
      --set-id $sid --timeline $tl --decompose            # strict
  venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \
      --set-id $sid --timeline $tl --fibers --decompose   # fiber-aware
done
venvs/audio/bin/python -m workspaces.alignment_prototype.path_decode --eval \
    --feature hubert --stems acappella --fibers --workers 8   # oracle ceiling
make race                            # driver board (re-runs infer; mutates timelines)
```

After running: update the header's date + `git rev-parse --short HEAD`, and
diff against §1–§4. Any number you cannot regenerate → mark it
`⚠ unverified — carried from <file>`, never silently.

## Appendix — deep-doc index

- `eda/alignment/failure_analysis/FINDINGS.md` — full failure taxonomy, one
  binding cause per span. Owns the *detail*; this doc owns the headline.
- `workspaces/alignment_prototype/looptrace/NOTES.md` — looptrace phase log +
  dead decode threads.
- `workspaces/alignment_prototype/cotrain_loso_findings.md` — LOSO write-up + caveats.
- `workspaces/alignment_prototype/external/fiber_validation_findings.md` — SALAMI validation.
- `docs/agent_handoff_fibers_20260710.md` — race board source (carried §4).
- `workspaces/alignment_prototype/attic/EXPERIMENTS.md` — closed-experiment verdicts.

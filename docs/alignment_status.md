# Alignment — canonical status (single source of truth)

> **Numbers regenerated 2026-07-23 (RT1) at commit `37e1d8a`** on the canonical
> `_lt` timelines scored against the **de-poisoned (post-Crush) GT**, via
> §Regeneration. §3–§4 carried, not rerun. **This doc owns every current
> alignment headline number.** Other docs cite it; they do not re-state numbers.
> If a number here is stale, re-run §Regeneration — do not hand-edit.
>
> **⚠ NO IDENTITY NUMBER IN THIS DOC IS TRUSTWORTHY YET — GT is incomplete at the
> export layer.** Root cause (proven 2026-07-23 via `als_audit`): the hand-session
> `.als` files are audio-SOUND — BB12 **281/290**, BB11 **264/271** clips
> audio-verify against the mix, and **every** clip resolves identity from its file
> reference. But `export_als_to_gt` **abstained on 34% of BB12 / 43% of BB11
> appearances** (`track_id=None`) — **all 57/57 BB12 abstained rows sit on OK
> audio-verified clips**, i.e. the identity was present and verifiable and the
> export threw it away. So every identity number computed so far is an artifact of
> a one-third-blank GT:
> - **82%/84% (pre-2026-07-19, poisoned GT):** inflated — wrong `track_id`s that
>   matched predictions scored as hits.
> - **51%/61% (per-span, de-poisoned GT):** deflated — correct predictions on
>   abstained appearances scored as misses (57/57 BB12 + 66 BB11 "misses" predict a
>   real recording; 53/57 + 61/66 overlap an abstained row).
> - **89%/82% (recall over adjudicable):** measured over only the ~66% that survived
>   export — the abstained third is non-random, so this is not the honest whole.
>
> **The fix is GT completion**: re-bind the abstained appearances from the
> audio-verified `.als` identity (local, sound — audio verification is a *stronger*
> gate than payload-hash). Until then, treat all identity figures below as
> provisional. Placement/trajectory are time-based and less abstention-sensitive
> but still measured over bound appearances only.
>
> **RT1 methodology (independent of the above, valid):** the scorecard is now
> form-centric — a GT appearance no span covers counts as full recall loss
> (`unrecovered_form`), totals audible-weighted (old 11959 s was envelope-inflated).
> Prior drift log: [alignment_status_corrections_20260711.md](alignment_status_corrections_20260711.md).

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

## 1. Headline (regenerated on `_lt`, 2026-07-23 — de-poisoned GT + RT1)

| | BB11 (`2nvzlh2k`) | BB12 (`1fsnxchk`) |
|---|---|---|
| **Identity** ⚠ PROVISIONAL (GT incomplete — see header) | per-span 51% · recall-over-adjudicable 82% | per-span 61% · recall-over-adjudicable 89% |
| **set_start** placement, median / <15 s | 6.8 s / 62% | 5.1 s / 67% |
| ref-offset MAE, straight clips (median / p90) | 8.0 s / 107.5 s | 26.3 s / 123.4 s |
| **Trajectory — multiseg+loop headline** (strict → fiber-aware) | **31% → 48%** | **26% → 46%** |
| &nbsp;&nbsp;stem: acappella | 20% → 42% | 12% → 33% |
| &nbsp;&nbsp;stem: regular | 27% → 51% | 49% → 66% |
| &nbsp;&nbsp;stem: instrumental | 28% → 51% | 21% → 48% |
| GT-seconds lost (corpus, both sets combined) | **76%** (5896 / 7764 s) | ← both sets |
| GT-form coverage (unmatched appearances, RT1) | 5 forms / 187 s | 12 forms / 215 s |

**Identity is the headline mover:** the prior 82%/84% was measured against poisoned
GT and credited the aligner for matching *wrong* bindings. On de-poisoned GT the
July `_lt` predictions land at **51%/61%** — the honest ceiling for those
predictions. Placement and trajectory shifted only modestly (both directions).

**Loss attribution** (binding cause, seconds-weighted, both sets, `_lt`, RT1 form-centric):

| cause | % of loss | note |
|---|---|---|
| decode-residual | **36%** | "which chorus" repeat-instance wall |
| placement | **30%** | co-equal binding wall |
| mis-route | 11% | stale `set_track_slots` stem axis (score from GT stem, not timeline) |
| identity | 7% | |
| **unrecovered_form** | **6%** | RT1: GT appearance no predicted span covers (form-centric recall loss) |
| instance-ambiguity | 4% | |
| loop-instance | 3% | |
| tempo/octave | 2% | |

Acappella is **44%** of corpus mix-seconds and the worst axis; acappella-multiseg
is **19%** of all loss. The new `unrecovered_form` bucket (6%) is acappella-heavy —
chopped/reprised vocal appearances the one-span-per-slot timeline never represented.

---

## 2. The strict → fiber-aware gap (named contribution, not a caveat)

Fiber-aware − strict is **+17 pp (BB11) / +20 pp (BB12)** on the regenerated
multiseg+loop headline. This gap *is* the
"which-instance" residual: the decoder lands on the correct repeated content
(right chorus) but not always the exact occurrence. It is externally
precision-validated (SALAMI **P .88**; low recall R .06 is precision-first-by-
design on a jam-band pessimistic floor, **not** a limitation verdict). Fibers v4
(2026-07-09) fixed the acappella recall hole (vocal coverage 0.06–0.28 → 0.33–0.73,
ear-validated by John); the phase-cancel clone certificate is wired. **This fiber
lift is a finding, not a footnote** — under-crediting fibers as a "scoring util" was
the original trigger for this overhaul (corrections C5).

**Timeline caveat:** the carried base-classical race board reports **45 (BB12) /
40 (BB11)**, while regenerated `_lt` reports **46 / 48**. Because the race board
was not rerun on the corrected GT, treat this cross-timeline comparison as
provisional; trust the within-run fiber lift above.

---

## 3. Oracle placement ceiling (isolates placement from decode)

⚠ **Carried from the 2026-07-11 run; not regenerated on the 2026-07-19 GT.**
`path_decode --eval` on acappella (HuBERT decode, GT placement given), n=21:
**strict 37% / fiber-aware 61%.** Regenerated real end-to-end acappella (`_lt`)
is 14–15% strict / 32–37% fiber-aware, leaving a **~24–29 pp** fiber-aware
placement gap. Placement is a real
binding wall, consistent with the regenerated loss attribution. The `fiber_gate`
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
| MERT | wired | **identity only** (cannot localize) | 51–61% span identity (honest, de-poisoned GT; was 82–84% on poisoned GT) | module CLAUDE.md |
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

Two co-equal walls: **placement (30% of loss)** and **which-instance
decode-residual (39%)**. LOSO proof (n=2): identity transfers 100% cross-set both
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

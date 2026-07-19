# Acappella oracle→e2e gap decomposition (the "oracle ladder")

**Date:** 2026-07-18
**Status:** design — approved at the brainstorming design gate (2026-07-18).
**Owner lane:** learned trajectory decoder / ML driver (sanctioned lane 2). Analysis
experiment; **no production code change**.
**Branch:** `worktree-acap-oracle-ladder` (fresh off `origin/main` @ f9c9fe4).

---

## 0. One-sentence goal

Empirically attribute the acappella **fiber-aware** gap between end-to-end
(~30%) and oracle-placement (~61%) to `{routing, identity, placement}`, and
isolate the **decode-instance headroom** (the strict→fiber gap), so we know
whether a perfect instance selector would actually move the scorecard — *before*
building one.

## 1. Why this, why now

The scoped "biggest modelling prize" is a **learned acappella instance selector**
over `{HuBERT diagonal evidence, fiber μ/ambiguity, fp sharpness}`. Two facts make
"just train it" the wrong first move:

1. **It is unfalsifiable at n=2.** The selector is explicitly gated on a third GT
   set for LOSO. BB10 is *initialized, not labeled* (confirmed 2026-07-18), so we
   remain at n=2. A fitted selector at n=2 violates the project's own
   overfitting bar.
2. **Six decode-layer instance-selection threads already died** (looptrace Phase 4
   discrim, Phase 5 residual tiebreak, DP-support reselection, PWS Dawid–Skene),
   each concluding "needs the learned arbiter + a third GT set."

Meanwhile the **strict→fiber gap on the current timeline (~+20 pp for acappella)
already is** the "perfect instance selector at current placement" headroom — the
scorecard reports it. So the genuinely *un-decomposed* quantity is the
**fiber-aware e2e (~30%) → oracle fiber-aware (~61%)** gap: ~30 pp that is
placement + routing + identity + span/GT correspondence, **not decode**. The
looptrace notes repeatedly flag this gap as living "in span/GT correspondence"
and never pinned it, because the two endpoints were measured on *different
populations with different tooling* (oracle `path_decode --eval` n≈21 vs scorecard
n=174). This experiment pins it with a single scorer on a single population.

**Baseline (regenerated 2026-07-18, `_lt`, this machine):** acappella traj **14%**
overall, **42%** of corpus GT-seconds, **~47% of all alignment loss**; ref-offset
MAE on straight acap clips BB12 43.4 s / BB11 38.4 s; 40% of GT-acap spans
mis-routed (traj 4% vs 20%); placement ≤15 s → traj 25% vs >15 s → 11%. Canonical
numbers: [docs/alignment_status.md](../../alignment_status.md).

## 2. Design — a layered oracle-substitution ladder

One eval harness. For each set (BB11 `2nvzlh2k`, BB12 `1fsnxchk`) it builds four
timelines and scores each with the **canonical** `score_timeline_vs_gt --fibers
--decompose` — same scorer, same matching internals, same population — so the
lifts between rungs are clean.

| rung | recording_id | placement (`set_start`) | routing (decode feature) | `ref_segments` |
|---|---|---|---|---|
| **R0** e2e | predicted | predicted | stale `claimed_stem` | from current `_lt` timeline (unchanged) |
| **R1** +routing | predicted | predicted | **GT stem → HuBERT** for GT-acap spans | re-decoded |
| **R2** +identity | **GT recording** | predicted | GT stem | re-decoded |
| **R3** +placement | GT recording | **GT `set_start`** | GT stem | re-decoded (≈ `path_decode --eval`) |

**Invariants.**
- Each rung reuses the existing per-span decode (`joint_ref_decode` →
  `path_decode.decode_path`) with oracle *inputs* overridden. The **decode-instance
  choice always stays the real decoder's output** — that is the quantity we are
  bounding, never oracle-substituted.
- All oracle substitutions are **eval-only measurement** (hand the decoder GT to
  isolate one layer). No production routing/ingest/identity change. Routing and
  identity levers live on the *other session's turf*; we only *measure* their
  slice, we do not act on them here.
- **Population held identical:** the report is restricted to GT-acappella rows,
  matched to predicted spans by the scorer's own logic on every rung. GT-acap rows
  with no matching R0 span (coverage/never-matched gaps) score 0 at R0 and are
  recovered at R2 (oracle identity) — correctly attributed to the identity rung.
- **n=2 is fine:** this is *attribution, not a fitted model*. Report **per-set**
  (BB11, BB12 separately). Never pool into a cross-set CI.

## 3. Outputs

A per-set table of acappella traj at each rung × `{strict, fiber-aware}`, plus:
- **Lifts** attributed to routing (R1−R0), identity (R2−R1), placement (R3−R2).
- **strict→fiber gap at R0 and R3** = instance-selection headroom at the current
  vs the clean (all-nuisances-oracle) operating point.

Written to a findings note (`looptrace/NOTES.md` append or a new
`trajectory/ORACLE_LADDER.md`) + committed. Machine-readable per-rung JSON kept
under `out/` (gitignored) for reproducibility.

## 4. Decision rule (threshold-free, honest at n=2)

Rank the reachable slices **per set**. Then:

- **Build the selector** iff the decode-instance slice (strict→fiber gap) is the
  **largest single reachable slice AND positive in both sets AND survives placement
  being fixed** — i.e. the **R3 strict→fiber gap is also large**, meaning the
  instance ambiguity is real, not an artifact of bad placement.
- **Redirect to placement/routing** if those slices dominate the fiber gap — then a
  perfect selector caps acappella e2e near ~30% and the leverage is elsewhere
  (placement is partly mine via the decoder's placement side; routing is the other
  session's).

The rule is deliberately a *ranking*, not a numeric cutoff, because n=2 cannot
support a calibrated threshold.

## 5. Correctness gate (the harness's own test — TDD anchor)

The harness is trusted only if its **endpoints reproduce independently-known
numbers** on the acappella subset:

- **R0** must reproduce the scorecard's acappella e2e (~**14% strict / ~30%
  fiber**).
- **R3** must reproduce the independently-computed oracle (`path_decode --eval
  --feature hubert --stems acappella --fibers`, ~**37% strict / ~61% fiber**).

If either anchor is off by more than a few pp, the harness is wrong — that is the
build gate. These two checks are written first (TDD) and block the rung logic.

## 6. Reuse map (no new decoders)

| need | reuse |
|---|---|
| per-span decode → `ref_segments` | `joint_ref_decode` + `path_decode.decode_path` |
| oracle-placement decode (R3) | `path_decode --eval` semantics (GT `set_start` crop) |
| feature routing (acap→HuBERT) | `harness/axes.py` |
| scoring + matching + fiber-aware | `score_timeline_vs_gt.py --fibers --decompose` |
| GT rows / spans | `labeling/fixtures/bb1{1,2}_ground_truth.yaml` |
| audio (mix/stems/manifest) | `~/aligning/<set>__*/` (shared) |

## 7. Scope / non-goals (YAGNI)

- **No** learned model, **no** production timeline/routing/ingest change, **no**
  new probe/channel/prior (sensor phase is frozen).
- **No** BB10 labeling (out of scope; if the decision is "build the selector," the
  n=3 unlock becomes a *separate* follow-up).
- Acappella only. Regular/instrumental are not the prize and are already
  understood (regular solved by legacy; instrumental on looptrace).

## 8. Risks

- **Cost:** ~174 acap GT spans × 3 re-decode rungs × 2 sets; HuBERT features build
  on first use (`.feat_cache` is currently empty — the "666 files" in old docs is
  stale). One-time, cached thereafter; runs off the scorecard, backgroundable.
- **The gap may be genuinely ambiguous:** the audit found spans where both
  evidence channels agree on the "wrong" answer (GT picks one of two equally-valid
  images). The R3 strict→fiber gap measures exactly this ceiling — a low value is
  itself the finding (selector is bounded).
- **R3 ≠ path_decode --eval exactly:** `--eval` uses its own `trajectory_acc` on
  n≈21; R3 routes decodes through the full-population scorer. Small deltas are
  expected; the anchor check tolerates a few pp, not exact identity.

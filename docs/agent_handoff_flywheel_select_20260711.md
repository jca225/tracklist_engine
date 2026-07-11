# Handoff — build the flywheel escalate→labeling-queue (batch selection)

**Date:** 2026-07-11 · **Repo HEAD at handoff:** `4051273` on `main` (clean).
**Spec (read first):** [docs/superpowers/specs/2026-07-11-flywheel-escalate-select-design.md](superpowers/specs/2026-07-11-flywheel-escalate-select-design.md)

## Why this exists (one paragraph)

North star: **SOTA alignment, closest to GT, on all 1001tracklists data.** The
critical path is **GT scale**. Gear 1 (multi-set co-train + LOSO) is built and
merged — labeling more sets now improves the model. This cycle builds **batch
selection**: turn your Ableton labeling time toward the spans the aligner most
needs. The agentic harness's `escalate` rung already identifies exactly those
spans ("cannot resolve confidently"), so batch selection = **wire `escalate` → the
labeling seeder.** Retrain half already exists (`export_als_to_gt` + `cotrain`).

## Load-bearing findings from this session (do not relearn)

- **Co-train LOSO (bb11↔bb12, merged):** identity transfers cross-set **100%**;
  **placement does NOT** (bb11 18.6 s vs bb12 1436 s, unstable). The MERT head
  memorizes placement per-set. Consequence for this cycle: on a *new* set many
  spans will escalate (weak placement) — good for coverage, but **cap the batch
  with `TOP`** so a labeling session is human-sized.
  Write-up: `workspaces/alignment_prototype/cotrain_loso_findings.md`.
- **Agentic probes are `validated=False`** (provisional precisions,
  `agentic/actions.py`). That's fine here: **escalation/ranking-what-to-label does
  NOT need validated auto-commit precision.** Do not try to flip `validated=True`
  in this cycle — n=2 GT is too thin (separate, later).
- **Slot-label normalization is a repeated footgun.** `slot_priors.normalize_slot`
  strips 3-digit zero-padding (`'038'`→`'38'`, `'026w1'`→`'026w1'`). GT fixtures use
  padded labels; scraped/other sources use unpadded. **When matching slot labels
  across sources, normalize both sides** — a raw-string compare gives 0 overlap
  (this bit the co-train anchor twice this session). Verify against real output,
  never assume a key name/format.

## Verified code anchors (grepped this session)

- **Agentic resolve:** `agentic/loop.py` — `resolve(spans, runners, EventLog, ladder)`
  returns a frozen `Resolution` with `.committed / .review / .suggested / .escalated`,
  **each `dict[str, SpanBelief]`** keyed by slot label. `escalated` is your target.
- **How spans + runners are built:** `agentic/__main__.py` (~lines 120-230):
  `--set-id --gt --timeline --live`. Builds `spans` (list of dicts, `data={**s,
  'claimed_stem': stem}`), `runners` = `build_live_runners(ctx)` when `--live` else
  replay runners, then `resolve(...)`. **Reuse this construction verbatim** in the
  new `escalated_slots(...)`; don't reinvent it.
- **Seeder to generalize:** `review/seed_worst_spans_als.py` — `worst_slots(set_id,
  top)` ranks by GT-seconds-lost from the scorecard `SPAN_TABLE`; `main` reads
  `OUT_DIR/{set_id}_predicted_timeline.json`, `pred_by_slot`, seeds via
  `review/seed_als_from_timeline.py`. **Factor the render core** (predicted-timeline
  clips → A/B `.als`) out of the GT-ranking, so the new `seed_slots_als(set_id,
  slots, ...)` reuses rendering with an **explicit slot list and NO GT dependency**
  (escalated spans have no GT — that's the point).
- **Retrain half (exists, out of scope here):** `labeling/export_als_to_gt.py`,
  `cotrain.py` / `train.py --loso`.
- **Set ids:** bb11 = `2nvzlh2k`, bb12 = `1fsnxchk` (corrected 2026-07-11 — the
  original handoff had these swapped; see `docs/alignment_status_corrections_20260711.md` C6).
  Run from repo root with
  `venvs/audio/bin/python`.

## Things I INFERRED in the spec — VERIFY before coding

1. `SpanBelief`'s margin/uncertainty field name (for ordering escalated spans) —
   read `agentic/belief.py` (`SpanBelief`); use what `escalated` values actually
   carry (belief margin vs share-of-mass). Spec left this open.
2. `seed_als_from_timeline`'s exact function signature for the render core — read it
   before extracting, so `seed_slots_als` and `seed_worst_spans_als` share it cleanly.
3. Whether every escalated slot has a predicted-timeline entry to render (it should —
   escalation falls back to the classical span, so the timeline is complete — but
   confirm `pred_by_slot` covers the escalated keys).

## How to execute

1. `superpowers:writing-plans` on the spec → a 3-task plan:
   (T1) `agentic/select.py` `escalated_slots(...)` [unit test, monkeypatch `resolve`];
   (T2) extract render core + `review/seed_slots_als.py` [unit test = GT-independence;
   refactor guard = `seed_worst_spans_als` unchanged output];
   (T3) `flywheel_seed(...)` + `make flywheel-seed SET=<id> [LIVE=1] [TOP=N]` +
   offline deliverable run on bb11.
2. `superpowers:subagent-driven-development` to build. Cheap model for the
   monkeypatched unit tasks; the offline agentic run (like this session's LOSO)
   is subagent-death-prone — **run it inline, capped with `TOP`**, don't fabricate.
3. Guardrail: `make check` (guardrails.py + pytest subset) passes before push;
   end commit messages with the Co-Authored-By line.

## Definition of done

`make flywheel-seed SET=1fsnxchk TOP=15` produces a labeling `.als` of bb12's
(`1fsnxchk`) top-15 escalated spans (no GT dependency), ready to hand-correct in Ableton →
`export_als_to_gt` → `cotrain` retrain. That closes the flywheel's front half.

## Session context (already merged to main this session)

André-absorption reduction table (Phase 0) · resample-ratio arm (Phase 2) ·
multi-set co-train + LOSO (flywheel gear 1). Banked specs not yet built:
FX-ladder robustness benchmark (`docs/superpowers/specs/2026-07-11-fx-ladder-benchmark-design.md`)
and this one. Ledger of the SDD runs: `.superpowers/sdd/progress.md`.

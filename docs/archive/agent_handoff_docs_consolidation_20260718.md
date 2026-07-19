# Handoff — memory/doc consolidation cleanup (2026-07-18)

**Transient work-order, not a standing doc.** Delete or let `docs_gc` archive this
once the task below is done. It exists only to resume a scoped cleanup across a
session boundary.

**Trigger:** velocity concern — sessions burn budget re-establishing state from a
149-line memory index + a growing pile of dated docs. Diagnosis this session:
codebase sprawl is *not* why alignment is stuck (that's 2 GT sets + the sim2real
transfer wall), but it *is* a real per-session tax. This cleanup attacks the tax,
not the research wall. Do not let it get miscoded as "refactor to unblock alignment."

---

## The load-bearing finding (fix this first)

**`docs/alignment_state_of_record.md` — the "read this FIRST" living doc that
CLAUDE.md and memory `project_alignment_state_of_record` both point every session
to — DOES NOT EXIST on `trm-ablation-framework` (current branch) or main.** It
exists ONLY in the sibling worktree:

```
.claude/worktrees/cotrain-accept-precision/docs/alignment_state_of_record.md  (17k, 2026-07-17)
```

Same stranding for `scripts/docs_gc.py` and `docs/archive/` — referenced in that
worktree's CLAUDE.md, absent on this branch. So the consolidation *target already
exists*; the job is mostly **propagate + prune**, not write-from-scratch.

Inventory (2026-07-18, this branch):
- Memory: **175 files, 149-line index** loaded every session (150 `type: project`).
- Docs: **69 `docs/*.md`, 16 dated** handoff/bearings/snapshot docs; no archive, no GC.

---

## Scoped plan (3 phases)

**Phase 0 — un-strand the infra (~30 min, highest leverage).**
Bring onto the live branch (→ main): `docs/alignment_state_of_record.md`,
`scripts/docs_gc.py`, `make docs-gc`. **Then update the doc**: it's stamped
2026-07-17 but the **TRM sim2real gap landed 2026-07-18** (v0 overfit 0.95;
synthetic→real flat ~0.09 < 0.306 control — architecture works, data/sim2real is
the wall). Doc must reflect that before it's trusted. After this, the "read first"
instruction resolves to a real, current file.

**Phase 1 — collapse the alignment memory cluster (~1–2 hrs).**
~60 of the 149 index lines are alignment-core verdicts (TRM, PWS-dead, fibers,
placement, identity, cotrain/LOSO, sensor-freeze, etc.). Their content now belongs
in the state-of-record doc + the attic ledger
(`workspaces/alignment_prototype/attic/EXPERIMENTS.md`). Fold content in, then cut
the index entries to a few pointers (state-of-record, status SSOT, closed-experiments
ledger). **Knowledge is not deleted — it stops being loaded every session.** Target:
alignment index footprint ~60 → ~6 lines.

**Phase 2 — archive dated docs (~30 min, mechanical).**
Once `docs_gc.py` is ported, run it: 16 dated docs → `docs/archive/`, keeping live
ops playbooks (esp. `docs/handoff_pws_cotrain_20260716.md` — the Vast gap-fill box
may still be draining; verify before archiving). This work-order archives itself here.

**Out of scope:** the ~90 non-alignment memories (soundcloud, vast, ingest, corpus,
appleseed, personalization). Different domain; not the alignment-session tax. Touching
them is scope creep.

---

## Two decisions the user still owes (asked, not yet answered)

1. **Branch.** Do this on `trm-ablation-framework`, or reconcile with main /
   `cotrain-accept-precision` first? Check what's actually merged before moving the
   stranded files — risk of conflict with the worktree copy.
2. **Memory-cut depth.** Aggressive (~60 → 6 pointers, content only in doc/ledger)
   vs conservative (keep one-liners for load-bearing verdicts — PWS-dead,
   TRM-sim2real — so a session sees them without opening the doc).

Recommended entry point: **Phase 0 only**, report back before touching memories.

---

## Session context (why the scope looks like this)

Canonical sources confirmed this session (cite, don't re-derive):
- Alignment status SSOT: `docs/alignment_status.md` (stamped 2026-07-11, `eb21a5e`).
  Identity 83–84% (transfers 100% cross-set, LOSO n=2); placement + which-instance
  decode = co-equal walls (37%/38% of loss); 85% GT-seconds lost.
- TRM bake-off: `workspaces/alignment_prototype/docs/trm_decoder_bakeoff.md` — verdict
  box 2026-07-18: architecture works, sim2real is the wall; lever = synthetic REALISM
  or the real pseudo-label flywheel, NOT more GPU.
- PWS label model REFUTED twice (attic ledger, 2026-07-14): categorical offset-bin DS
  breaks on continuous offsets + heterogeneous probe precision. Co-training is *not*
  dead — it's the active plan (`docs/handoff_pws_cotrain_20260716.md`), but the current
  MERT head transfers identity, not placement (LOSO: `cotrain_loso_findings.md`).

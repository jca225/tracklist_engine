# Operation Crush — TLDR for humans

> Human-readable digest of the program. **Operative plan of record:**
> [operation_crush_assault_plan.md](operation_crush_assault_plan.md).
> **Live status + every headline number:** [alignment_status.md](alignment_status.md)
> (the SSOT — numbers live only there). **Research mapping:**
> [operation_crush_research_synthesis.md](operation_crush_research_synthesis.md).

## What is this?

Operation Crush is the infrastructure project whose only job is to get the
alignment algorithm from "kind of works" to "actually works" by **August 1**. It
is not a new feature. It is the cleanup of the data layer so every other feature
can stop fighting bad inputs.

The core insight: the alignment algorithm has been underperforming because it was
being fed **wrong audio, wrong identity labels, or wrong ground-truth boundaries**.
Crush exists to make that impossible by design.

## The four problems we are crushing

| # | Problem | Why it matters | Where to track |
|---|---|---|---|
| 1 | **Wrong audio resolved** | A track is labeled "acappella" but the downloaded file is the full mix, or vice versa. The model learns nonsense. | #40 |
| 2 | **Wrong GT captured** | Ableton clips are arranged one way but exported as ground truth another way (e.g. deactivated acappella clips exported as instrumental). | #40 |
| 3 | **Identity taxonomy is fragile** | Remix vs original, extended vs regular, acappella vs instrumental, remixer name — currently not robustly represented or verified. | #40 |
| 4 | **No automated acquisition gate** | When the agentic pipeline downloads songs from the internet, there is no rigorous way to prove it got the right file. | #41 |

## The three deliverables

1. **Identity / GT capture hardening** — use stable IDs, audio fingerprints,
   provenance logging, and human review queues so every GT row and every resolved
   audio file can be audited. (#40)
2. **Transition physics model** — implement DJtransGAN's fade/EQ model to estimate
   `gain_curve` and `audible_*` windows from the actual mix audio. (#42)
3. **Agentic acquisition gate** — add audio-text verification, reasoning traces,
   authentication, and hallucination checks so the flywheel can safely download
   and verify its own training data. (#41)

## Research we are folding in

See [operation_crush_research_synthesis.md](operation_crush_research_synthesis.md)
for the full paper mapping. The highest-leverage findings:

- **MERT 330M + frame-level MERT** for precise localization and identity verification.
- **DJtransGAN** (Chen et al., ICASSP 2022) for modeling crossfades and audible windows.
- **FIGMA / MULTI-SCORE** (ACL 2026) for audio-text verification and two-stage retrieval.
- **2026 LALM / audio-reasoning surveys** for trustworthiness: authentication,
  hallucination checks, and reasoning traces.

## Milestone

[Operation Crush](https://github.com/jca225/tracklist_engine/milestone/1)

## The operative plan

[operation_crush_assault_plan.md](operation_crush_assault_plan.md) (2026-07-20) is
the phased plan of record: Phase 0 consolidate scattered fixes (#48) → Phase 1 GT
de-poisoning (#47) → Phase 2 audio truth (#41, #45) → Phase 3 canonical
re-measurement (#44, #48) → Phase 4 SOTA offensive (#46, #42, flywheel). It carries
the full discrepancy register (D1–D15).

## Sub-issues

**Phase 0 — consolidate**
- #48 — consolidate scattered remediation (unmerged branches, unpushed commits, two SSOTs), then one canonical re-measurement

**Phase 1 — GT de-poisoning (critical path)**
- #40 — robust GT capture and audio resolution (identity taxonomy, provenance, MERT upgrade)
- #47 — BB12 GT fixture stale track_ids — .als relocation decision + gated re-export

**Phase 2 — audio truth**
- #41 — agentic acquisition / verification gate
- #45 — Calvin Harris "I Need Your Love" work_id grouping investigation

**Phase 3 — re-measure (on clean GT only)**
- #44 — re-run TRM diagnostics on clean GT and latest scorer/referee
- #2, #3, #4 — legacy aligner issues; evidence re-scored on clean GT before any decoder work

**Phase 4 — SOTA offensive**
- #46 — cache vocal-enhance step (iteration-speed unlock)
- #42 — DJtransGAN transition model for GT gain_curve / audible windows

## What this doc is not

This is the human digest. No code or numbers live here — implementation happens in
the sub-issues above, and every metric lives only in
[alignment_status.md](alignment_status.md).

# Checked System Contract — design spec (2026-07-21)

> **Status:** design approved 2026-07-21. Next step: implementation plan
> (writing-plans). Sequenced into Operation Crush **Phase 0** (issue #49) and the
> [master plan](../../operation_crush_master_plan.md) §3.

## 1. Problem this solves

The GT poisoning was a **Type-II failure**: the poison existed and nothing
detected it. Coding agents moved ~10× faster but no single place held "what is
true," and the failure lived in *data* (stale `track_id`s, path divergence), not
in code. A human-readable codebase snapshot would **not** have caught it — the
code did what it said; the data flowing through it was wrong.

The repo already had desired-state docs (`alignment_status.md`, per-module
`CLAUDE.md`, state-of-record). Their failure mode was **silent divergence**
(two `alignment_status.md` versions on two branches = D11). More hand-maintained
prose would drift the same way.

**Conclusion:** the counter to "you don't know what you don't know" is not a doc
a human reads periodically — it is a set of **data-truth invariants that fail the
build**. This spec defines a small, machine-checked contract of exactly those
invariants, and binds them so the contract *cannot say something the checks don't
enforce*.

## 2. Prior art (folded in)

- **Data contracts / validation** — Pandera, Great Expectations, dbt tests, Soda.
  Core idea = "define + enforce assumptions at runtime to prevent silent data
  corruption propagating downstream." Two techniques adopted directly:
  **row-count/span delta vs a trailing baseline** (a dropped row never shows up
  in NULL counts → catches the silent slot-111 span shift) and **freshness
  checks** (a pipeline succeeds while its source went stale → the D1 stale
  fixture).
- **Architecture fitness functions / ArchUnit** — constraints as test-like
  assertions wired into CI. Adoption law: *introduce gradually, keep the suite
  green* — which is the **ratchet** pattern `guardrails.py` already uses.
- **Executable-docs / agent-instruction enforcement** — ContextCov
  (arXiv 2603.00822) derives + enforces executable constraints from agent
  instruction files by blocking violating commits; "Steerability via constraints"
  (arXiv 2607.02389) frames constraints as scalable oversight of coding agents.
  This is our claim↔check meta-check, aimed at the Type-II-from-agents root cause.
- **QUITO-X** (arXiv 2408.10497) — LLM context compression; **explicitly out of
  scope** (no model in the check loop; compression helps an agent *hold* context,
  not *verify truth*).

## 3. Design

**One line:** a machine-readable **registry** of data-truth invariants, each bound
to exactly one check, rendered into a human-readable `SYSTEM_CONTRACT.md`, riding
the existing `guardrails.py` harness — so the doc cannot claim what the checks
don't enforce.

### 3.1 The invariants (C-series), two planes

**Static plane** — runs on any checkout (pre-commit + CI), hard-fail immediately:

| id | invariant | bound check | status |
|----|-----------|-------------|--------|
| **C1** | No identity-by-mutable-string, enforced **by construction**: (a) every committed GT row carries `id_source: content \| abstain` — no row is stamped from a filename/slot guess; (b) **GT export is invariant under rename/renumber/relocation** of local audio | (a) new `guardrails._check_id_source` (static); (b) new `tests/labeling/test_export_metamorphic.py` (the renumber-metamorphic test) | new — see "C1 re-specified" below |
| **C2** | One metric SSOT: no alignment-metric strings outside `docs/alignment_status.md` (**detection pattern + path exemptions — `docs/archive/`, `attic/EXPERIMENTS.md`, corrections logs, memory, discrepancy quotes — spec'd in #53 before implementation**) | new `guardrails._check_ssot_fence` (= #53) | new |
| **C3** | Contract integrity meta-check: every claim → exactly one live check, every registered check → exactly one claim (bidirectional; orphan either way = red) | new `guardrails._check_contract_registry` | new |

**Data plane** — runs where GT + DB exist (`make verify-contract`, gt-gate,
weekly audit); ratcheted, not a wall of red:

| id | invariant | bound check | status |
|----|-----------|-------------|--------|
| **C0** *(top priority)* | `.als` reference integrity: every audio ref resolves **after `html.unescape`**, matches the `.als`'s own `OriginalFileSize`/`OriginalCrc`, and binds to a `track_audio_id`/`recording_id` by content — **the D1/D20 poison-catcher** (see §6b for detail) | new `verify_als_refs` | new |
| **C4** | GT round-trips: `.als` → export → re-derive within tolerance | audio round-trip law (PR #37) | exists / wire |
| **C5** | Zero stale ids: every GT `track_id` resolves to the same song the slot claims | `scripts/audit_gt_recording_ids.py` | exists / gate |
| **C6** | GT freshness/provenance: every committed fixture is sha-proven-derived from the *current* `.als`, never a stale-manifest fallback | new `verify_gt_provenance` (from dbt freshness) — **kills D1 as a class** | new |
| **C7** | Span-count/duration delta guard: a re-export within trailing tolerance of the last stamped export; a silent span shift fails | new `verify_gt_delta` (from Pandera row-count delta) — **kills the slot-111 shift as a class** | new |
| **C8** | No ungated audio consumed: every `track_audio` alignment reads carries a passing gate verdict or explicit quarantine | Phase-2 acquisition gate 1+2 | partial |

C6 and C7 are the direct products of the prior-art scan and map one-to-one onto
the two concrete poison mechanisms from the July handoffs.

### 3.2 Binding — registry is the source of truth

- A small `contract/registry.py`: a list of `Claim(id, statement, plane,
  check=<callable>, status)`. This is the **single source of truth**.
- `SYSTEM_CONTRACT.md` is **rendered from the registry** (griffe-style), never
  hand-edited — so the human doc cannot drift from the registry.
- **C3** (the meta-check) enforces the bidirectional mapping: registry claims ↔
  registered checks, no orphans; and that the rendered doc matches the registry.
- A claim with `status="planned"` is allowed (its check may be a stub asserting
  `xfail`), but it must still appear on both sides — planned ≠ orphan.

### 3.3 Enforcement + adoption (ratcheted, green-first)

- **Static plane (C1–C3):** hard-fail in pre-commit + CI immediately — cheap,
  always runnable on a clean checkout.
- **Data plane (C0, C4–C8):** a new `make verify-contract` target, also invoked by
  gt-gate at write-back and by the weekly audit. Introduced **ratcheted**: each
  invariant starts as warn + baseline; it flips to hard-fail **per-set** as that set
  passes it. This honors the ArchUnit "keep it green" lesson and matches the repo's
  existing entropy ratchet — no wall of red on day one. **DB access:** `verify-contract`
  must query pi over SSH (or accept a local DB snapshot **with a freshness stamp**) —
  the local `data/db/music_database.db` is never truth (CLAUDE.md); state which per run.
  **Ratchet baselines** live in a `contract_ratchet.json` (precedent:
  `guardrails_ratchet.json` / `entropy_ratchet.json`); the first *de-poisoned*
  re-export is a deliberate baseline reset, not an alarm.

### 3.4 Placement

- `contract/registry.py` + `SYSTEM_CONTRACT.md` live at **repo root / `docs/`** —
  canonical, discoverable. **Not** in `alignment` (an
  experimental, currently frozen fork; a repo-truth artifact cannot live there).
- Archiving stale aligner scaffolding into `alignment/attic/`
  is a *separate* Phase-0 cleanup, not part of this contract.

## 4. Sequencing into Operation Crush Phase 0

1. After the worktree census (#49), scaffold `contract/registry.py` + the
   `SYSTEM_CONTRACT.md` renderer + **C1–C3** (static plane). These go green
   immediately and start guarding on day one.
2. Wire the data-plane checks **C4–C7** as Phase 1's gt-gate and de-poisoned GT
   land — they *activate* the moment there is clean GT to pass them, making
   Phases 1–2 self-verifying rather than trusting a one-time manual pass.
3. C8 lands with the Phase-2 acquisition gates.

The contract is the connective tissue that turns the Crush phases from
"a human verified it once" into "the build verifies it every time."

## 5. Non-goals (YAGNI)

- Not an architecture map (identity axes, module DAG, storage topology) — the
  approved scope is data-truth invariants only. Architecture drift is a real but
  *different* problem; do not fold it in here.
- Not an LLM/agent context system — no compression, no retrieval.
- Not a replacement for the per-module `CLAUDE.md` files — those stay as human
  orientation; this contract is the *enforced* subset.

## 6b. Revision note (2026-07-21, post-Fable adversarial review)

An adversarial review found the C-series as first drafted **would not catch the
one poison mechanism actually demonstrated this session** — `.als` GT linked to
audio by mutable relative filename, with filenames/positions drifting (`117 Mode
(Remix)` → `033 Mode (Jay Hardway Remix)`; `.m4a` → `.flac`). C1 fences *code*
joins; nothing bound the `.als`'s own reference layer. Corrections:

- **C0 (NEW, data-plane, top priority) — `.als` reference integrity.** For every
  GT `.als`: every audio `RelativePath` (a) resolves on disk **after HTML-unescape**
  — the review caught that the un-unescaped parse mis-flags every `&`-containing
  track; (b) the resolved file **matches the `.als`'s own recorded `OriginalFileSize`
  + `OriginalCrc`** for that `FileRef` (embedded ground truth about the exact bytes
  labeled against — mechanical, no external data, and the check that caught the Manse
  34/14 MB near-miss); and (c) binds to a `track_audio_id`/`recording_id` **by content**
  (hash/fingerprint), failing on unresolved refs *or* refs whose content identity
  differs from the slot's claim. (a)+(b) are cheap and belong in the relink gate
  itself; (c) is the full identity bind. **Binding source for (c):** compare a
  chromaprint fingerprint (`track_fingerprints`), *not* raw sha256 — local
  `~/aligning` files legitimately differ byte-wise from canonical pi bytes (stems,
  transcodes); (a)+(b) already pin exact-bytes identity locally. This is the check
  that would have caught D1/D20.
- **C7 scope corrected.** Span-count/duration deltas **cannot** catch an identity
  swap (same spans, wrong song) — so C7 is demoted to a *drop/shift* guard only;
  identity is C0's job. **Provisional tolerance:** total audible-span duration within
  **±2%** and row-count exactly equal vs the last stamp; baseline in
  `contract_ratchet.json`; the first *de-poisoned* export is a deliberate reset, not
  an alarm.
- **C6 made implementable.** "current `.als`" was undefined (4 siblings exist; sha
  churns per save). C6 reads a **canonical-als registry `contract/canonical_als.yaml`**
  (`set_id → {als_path, content_sha}`) and checks provenance against *that*, not "the
  newest". This registry is also where the §4 naming convention and the D22
  convention/path choice get encoded and enforced.
- **C3 honesty caveat.** The meta-check enforces 1:1 claim↔check bookkeeping but a
  stubbed/`planned` check still renders into the doc — so the doc can still state
  not-yet-enforced things. `SYSTEM_CONTRACT.md` must visibly mark each claim's
  enforcement status (enforced / ratcheting / planned).
- **CI-reach caveat.** C0/C4–C6 need Mac-side GT + pi DB and cannot run in
  clean-checkout CI; their tooling currently lives only on an unpushed branch.
  "The poison would fail the build" is the target once these land + a named
  weekly-audit owner runs the data plane — not true on day one.
- **C1 re-specified (2026-07-21, post root-cause review — supersedes the table's
  original binding).** A second adversarial review (Fable) found the original C1
  binding — "extend `entropy_audit` fail-closed resolver fences (mostly exists)"
  — is a **paper tiger**: `scripts/entropy_audit.py` fences exactly three classes
  (`net_subprocess_no_timeout`, `_no_encoding`, `bare_except`); there is **no
  resolver fence at all**, on `main` or any branch. "No identity-by-mutable-string"
  is a *semantic* property — an AST fence for it is either vacuous or a false-positive
  flood, and C3's meta-check would happily certify it as "enforced." Replaced with two
  checks that can actually exist:
  - **C1a — `id_source` stamp (static, cheap, CI-runnable on a clean checkout).**
    The exporter stamps every GT row `id_source: content | abstain`.
    `guardrails._check_id_source` fails on any committed fixture row that is neither
    — i.e. no row may be stamped from a filename/slot guess.
  - **C1b — the renumber-metamorphic test (the real class-killer).**
    `tests/labeling/test_export_metamorphic.py` takes a small synthetic set from the
    existing `tests/labeling/fixtures/als/` matrix, **renames/renumbers every audio
    file and moves the `.als` one folder deeper**, re-exports, and asserts the GT is
    **byte-identical**. Pre-registered invariant: *GT export is invariant under any
    rename, renumber, or relocation of local audio.* If green, the location-as-identity
    and guess-ladder diseases are dead in the export path **by construction** — no
    vigilance. This is only reachable once the guess-ladder is removed: **deleting
    `slot_id_map` + the weak tiers of `match_manifest_for_path` is a prerequisite**
    (tracked as a Phase-1 exit criterion in the master plan), because the metamorphic
    test cannot pass while a renumber can still silently re-key identity. C3 binds to
    C1a+C1b, not to the non-existent fence. Root cause + full design:
    memory `project_path_identity_root_cause`; master-plan Root-cause respec.

## 6. Definition of done

- `contract/registry.py` exists; `SYSTEM_CONTRACT.md` renders from it; C3 fails
  the build on any claim↔check orphan or doc drift.
- C1–C3 hard-fail in `make check` / pre-commit / CI.
- `make verify-contract` runs **C0, C4–C8** against GT + DB, ratcheted per-set;
  **C0's (a)+(b) also run inline in the Phase-1 relink/re-export gate.**
- `contract/canonical_als.yaml` exists and pins each GT set's canonical `.als` +
  content sha (encoding the naming convention / D22 choice).
- The master plan §3 and issue #49 reference this spec; it lands in PR #54.

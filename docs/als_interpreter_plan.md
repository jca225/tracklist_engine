# Plan: `labeling/als/` as a robust `.als` interpreter — architecture, testing, and the open-source seam

**Status:** §6 steps 1–5 EXECUTED 2026-07-02 on `als-codec-extraction`
(cst/semantics split, validate.py, roundtrip.py laws, Hypothesis+fuzz,
`als_core_boundary` guardrail). Step 6 (repo split/publish) remains gated on
the open questions below. One deviation: read APIs kept their signatures
(no Result retrofit — house style); validation is a separate total pass.
Companion to
[als_codec_subpackage_plan.md](als_codec_subpackage_plan.md) (the mechanical
extraction, DONE) — this doc covers what comes next: interpreter-grade
structure, a serious test regime, and an open-sourceable core.

---

## 0. Survey: does an OSS repo already do this better? (No.)

Checked 2026-07-02. Everything public is a **read-only extractor**; none is a
bidirectional codec with defined timeline semantics:

| Project | What it does | Gap vs ours |
|---|---|---|
| [dawtool](https://github.com/offlinemark/dawtool) | Reverse-engineered Live/FL parsers; markers + tempo map → real time | Closest semantic overlap, but read-only, and its docs say tempo automation is "linear only; nonlinear may cause inaccuracies" — our `tempo_beat_to_sec` computes the **exact log-integral of linear BPM ramps** |
| [pyableton](https://github.com/maranedah/pyableton) | Parse .als → tracks/clips, `to_midi()` | Read-only, pre-release, no warp/tempo/volume semantics |
| [AbletonParsing (DBraun)](https://github.com/DBraun/AbletonParsing) | `.asd` clip files: warp markers | `.asd` not `.als`; Live 12 format unsupported |
| [alsd](https://github.com/andrewcb/alsd), [loive](https://github.com/naglalakk/loive), [als-tools](https://github.com/luizen/als-tools), [MTL analyser](https://www.musictechlab.io/blog/software-development/extracting-data-from-ableton-als-asd-files) | Dump/catalog utilities | Inspection only |
| [als2cue_web](https://github.com/LucaTNT/als2cue_web) | Locators → .cue | Single-purpose |

**Nobody has:** (a) write-side session mutation that Live reliably reopens
(PointeeId discipline), (b) a round-trip law as a tested invariant, (c) both
warped-mix *and* unwarped/master-tempo conventions behind one interface,
(d) volume-automation audibility semantics. Conclusion: don't adopt — our core
is ahead. Open-sourcing it fills a real gap (and dawtool's README is evidence
of demand). It also means we maintain it alone; that's the cost.

## 1. Framing: which kind of "interpreter" this is

Dragon-book pipeline mapped onto the codec (Aho–Lam–Sethi–Ullman,
[user-linked PDF](https://faculty.sist.shanghaitech.edu.cn/faculty/songfu/cav/Dragon-book.pdf)):

| Compiler phase (Dragon ch.) | `.als` codec equivalent | Where |
|---|---|---|
| Lexing / parsing (3–4) | gzip + XML — **delegated to `gzip` + `lxml`**; we do not hand-roll a grammar | `read.load_als_xml` |
| Concrete syntax tree | The **lxml tree is our lossless CST** (full fidelity, like Roslyn red-green trees / rust-analyzer's rowan / [LibCST](https://github.com/Instagram/LibCST)) | kept in memory per run |
| Abstract syntax (4.x, "syntax vs semantics") | Frozen dataclasses = **AST**, a typed projection of the CST | `models.py` |
| Syntax-directed translation (5) | CST → AST extraction | `read.parse_layer_clips` etc. |
| Semantic analysis / symbol tables (5–6) | **Validation pass** (new, §3) + identity resolution (manifest = symbol table) | `validate.py` (new), `identity.py` (private) |
| Evaluation (interpretation proper) | **Denotational semantics of the timeline**: ⟦session⟧ : arrangement-beat → mix-second, gain, audibility | mappers + envelope fns (today in `read.py`; split to `semantics.py`) |
| Code generation / pretty-printing | **In-place CST mutation, never regeneration** — this is how fidelity survives and why Live doesn't crash | `write.py` |

Two deliberate **non-goals** (over-engineering guards): no parser/lexer
generators, no IR/bytecode, no optimizer — the input language is XML and the
"program" is a timeline; the value is in the *semantics*, not the parsing. And
CST-edit-in-place stays the write strategy; a from-scratch pretty-printer that
re-emits whole sessions is exactly the thing that crashes Live (see the
`feedback_als_seed_strip_automation` history).

## 2. Target module layout

Reorganize `labeling/als/` (mostly file splits of `read.py`; models/tags/write
already right-sized):

```
labeling/als/
  __init__.py     # curated public API (exists)
  cst.py          # load/save gzipped XML; byte-fidelity helpers   [from read.load_als_xml + new save]
  models.py       # AST records (exists)
  read.py         # CST → AST extraction only
  semantics.py    # beat↔sec evaluators: mappers, tempo integral,
                  # envelope/audibility, clip splitting             [split out of read.py]
  validate.py     # NEW — well-formedness diagnostics (§3)
  write.py        # in-place CST mutation (exists)
  roundtrip.py    # NEW — executable laws harness (§4)
  identity.py     # PRIVATE — manifest/slot/stem resolution (exists)
  tags.py         # PRIVATE-ish — annotator tag stripping (exists)
```

Errors as values per house style: library functions return
`Result` (`core/result.py`) with a small diagnostic type carrying **source
location** = (track name, clip name, XPath) so a bad session tells you *which
clip* — the interpreter-world equivalent of line/column in error messages.
CLI edges keep `sys.exit`.

## 3. `validate.py` — semantic well-formedness ("parse, don't validate" at the boundary)

One pass, run before extraction; every invariant is a named diagnostic
(warning or error), never a silent fixup:

- warp markers: sorted, finite, distinct-beat pairs exist (the Aftershock
  duplicated-marker case becomes a *warning*, since `beat_to_sec` handles it)
- loop domain units: unwarped clips carry seconds, warped carry beats — flag
  suspicious magnitudes (the exact confusion that caused the −430 s bug)
- tempo envelope: FloatEvents sorted, positive BPM, sentinel times clamped;
  duplicate-Time steps recognized as steps
- PointeeId uniqueness across AutomationTargets (the Live-crash class)
- schema version: read `<Ableton MinorVersion>`; unknown major versions →
  explicit `Err`, not best-effort parsing
- volume automation: envelope targets resolve; values in [0, ~2]

## 4. Test regime (the "current standards" part)

What robust parser/printer projects actually do, applied here:

1. **Golden corpus tests** (have the seeds already): BB12 (152 clips), BB11,
   the clean seed template. Snapshot the extracted GT projection; any diff is
   a review event. *OSS caveat: real sessions embed artist names/local paths —
   ship synthetic fixtures, keep real sets private (§5).*
2. **Round-trip laws as executable properties** (`roundtrip.py`):
   - **Law A** `parse ∘ print = id` — write-side ops (tempo envelope,
     locators) then re-parse recovers exactly what was written; untouched
     XML regions byte-identical (assert on the gzip-decompressed bytes).
   - **Law B** `print ∘ parse = id` on the projection — export GT, re-read,
     structurally equal (this is `anchor_check` generalized and moved in).
3. **Property-based tests** ([Hypothesis](https://hypothesis.readthedocs.io/)) —
   generate inputs, assert laws, shrink counterexamples:
   - random monotonic warp markers ⇒ `beat_to_sec` monotonic, and
     `sec_to_beat`-free invariants like span positivity
   - random piecewise-linear tempo curves ⇒ `tempo_beat_to_sec` ≈ numeric
     quadrature of 60/bpm (the exact-integral claim becomes a *tested* claim)
   - random gain curves ⇒ `audible_from_curve` fraction ∈ [0,1], start ≤ end,
     agreement of the three audibility fields (the 066/112 bug class)
   - `split_clip_at_mix_span_edges` ⇒ parts tile the original span, each part
     monotonic in mix-seconds
4. **Differential tests**: (a) new package vs the `als_io` shim on the golden
   corpus (already implicitly true — same code — but pin it so future edits
   can't fork behavior silently); (b) tempo maps vs **dawtool** on shared
   fixtures where both support the feature.
5. **Corpus fuzzing** (cheap tier): mutate golden-session XML (drop
   attributes, reorder elements, corrupt floats) ⇒ codec must return `Err`
   diagnostics, never raise or hang. Hypothesis handles this too
   (`st.sampled_from(corpus) → mutations`); no need for a coverage-guided
   fuzzer at this scale.
6. **Version matrix**: fixture per supported Live version (11, 12); explicit
   unsupported-version test.

CI: all of the above join `make check` (guardrails already run pytest; add the
`hypothesis` dep to the audio venv requirements).

## 5. Open-source seam — publish the codec, keep the project

The Phase-1 module split is already ~the OSS boundary:

| | Modules | Why |
|---|---|---|
| **Open-source core** ("alscodec" / "liveset-codec", MIT) | `cst` `models` `read` `semantics` `validate` `write` `roundtrip` + synthetic fixtures | Generic Ableton knowledge only — nothing about DJ sets, GT, our DB, or the pipeline |
| **Stays private** | `identity.py` (manifest/slot/claimed_stem taxonomy, aligning-folder layout), `tags.py` annotator convention, every consumer (`export_als_to_gt`, seeder, …), GT schema, all real-set fixtures | This is the actual moat: the labeling workflow, identity axes, and the GT corpus — none of it needs to leak |

Leak audit before any publish: no `~/aligning` paths, no set IDs/artist names
in fixtures or docstrings (several current docstrings name BB12/Aftershock —
scrub or generalize), no `core.identity` import in the core (the *one* current
cross-dependency: `identity.py` imports `normalize_stem` — already on the
private side, fine), no pi-storage/DB references.

Phasing (don't split repos prematurely):

1. **Now:** enforce the seam in-repo — guardrail rule: `models/cst/read/
   semantics/validate/write/roundtrip` must not import `identity`, `tags`,
   `core.*`, or anything project-side. Costs nothing, keeps the option open.
2. **When the test regime (§4) is green and the API has stopped moving:** lift
   the core to its own repo, publish to PyPI; `labeling/als/` keeps `identity`/
   `tags` and re-exports the dependency. History: fresh repo (no
   `git filter-repo` archaeology — our history names sets everywhere).
3. Versioning discipline from day one: semver, CHANGELOG, the round-trip laws
   documented as the package's contract.

## 6. Execution order

1. Split `semantics.py` out of `read.py`; add `cst.py` (mechanical, shim-safe;
   refactor-safety skill again). **Gate:** `make check` green, goldens byte-stable.
2. `validate.py` + diagnostic type + `Result`-ification of the public read
   API (behavior change — separate commit; consumers updated).
3. `roundtrip.py` (absorb/generalize `anchor_check`) + golden corpus wiring.
4. Hypothesis properties + corpus fuzzing + differential-vs-dawtool.
5. Import-boundary guardrail (the §5 rule) into `scripts/guardrails.py`.
6. (Later, deliberate) repo split + PyPI publish + synthetic-fixture pass.

Steps 1–5 are all safe on the current branch cadence; step 6 is a
human-decision gate (naming, license, maintenance commitment).

## Open questions for John

- Package name for the OSS core (`alscodec`? `liveset`? check PyPI squatting).
- MIT vs Apache-2.0 (Apache's patent grant is the usual argument; MIT matches
  the survey's ecosystem).
- Is differential-vs-dawtool worth the dependency, or golden-only? (Cheap to
  add, easy to drop.)
- Live 12 `.asd`/format drift: do we commit to Live 12 fixtures now or pin
  "Live 11-era .als" as the supported surface until the corpus moves?

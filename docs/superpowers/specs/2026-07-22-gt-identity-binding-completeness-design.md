# GT Identity Binding — soundness kept, completeness proved

**Date:** 2026-07-22
**Status:** design, pending Fable review + implementation plan
**Context:** Operation Crush killed the `slot_id_map` poison by binding GT clip → recording *only* by audio content. That is **sound** (no wrong labels) but **incomplete**: BB12 66% / BB11 57% bound, the rest abstain. This doc formalizes the problem, proves why the current scheme is sound-but-incomplete, and specifies a composed resolver that is **provably sound and completeness-maximal** — recovering the abstains that are recoverable, and abstaining on exactly those that are not.

---

## 0. Why this matters (the stakes, stated once)

The GT is the answer key for every alignment number. A *wrong* label is worse than a *missing* one — it silently poisons training and evaluation (that was Crush). But a GT that abstains on 40% of clips is a crippled answer key: half the aligner's regime is unmeasured. The requirement is therefore two-sided and non-negotiable:

- **(S) Soundness** — never emit a wrong identity. `B(c) ≠ ⊥ ⟹ B(c)` is the true identity of clip `c`.
- **(K) Completeness** — abstain *only* when no reliable channel can identify `c`. `B(c) = ⊥ ⟹` `c` is genuinely unrecoverable.

Crush achieved (S). This doc is about achieving (K) **without sacrificing (S)** — and proving both.

---

## 1. Formal model

**Objects.**
- `C` — the set of GT clips in a set's `.als`. Each clip `c` references a local file `f(c)` with byte-content `b(c) ∈ {0,1}*`, sits at placement index `σ(c) ∈ ℕ` (the `.als` clip order), and carries a referenced path `p(c)` (e.g. `stems/003__…/instrumental.flac`).
- `A` — canonical audio artifacts on pi: every `track_audio` row and every `track_stems` (demucs/roformer) file. Each `a ∈ A` has byte-content `β(a)`, a **recording identity** `ρ(a) ∈ R`, and a **track_audio id** `τ(a)`.
- `H` — a cryptographic hash (`sha256`, or the tag-invariant `mdat` payload hash for mp4). `H` is injective up to negligible collision probability.

**The true identity** of a clip, `id*(c) ∈ R ∪ {⊥}`, is what a human annotator would assign: the recording the clip's audio *is*, or ⊥ if it is not a recording (a mix-extract, a bespoke edit with no catalog counterpart).

**The binding function** `B : C → R ∪ {⊥}` is what the exporter emits. We want `B ⪯ id*` (sound) and `B⁻¹(⊥)` minimal (complete).

**The transformation group `T`.** Audio identity is invariant under a set of byte-level transformations that preserve *what song it is*:
```
T = { re-encode (codec/bitrate), re-separate (Demucs→Roformer), container-retag,
      resample, trim/pad silence, loudness-normalize, … }
```
Crucially: `t ∈ T, t ≠ id ⟹ H(t(x)) ≠ H(x)` in general. **`H` is not `T`-invariant.** This single fact is the source of all incompleteness (§3).

---

## 2. The current scheme and its soundness proof

**Content binding:**
```
B_content(c) = ρ(a)   if ∃ a ∈ A : H(b(c)) = H(β(a))
             = ⊥       otherwise
```
Implemented by `_content_bind` ([export_als_to_gt.py:226](../../labeling/export_als_to_gt.py#L226)) against `content_catalog.json`, whose entries are `{content_sha256, payload_sha256, recording_id, track_audio_id, stem}` for every `track_audio` row **and** every `track_stems` file of the set's recordings ([build_content_catalog.py](../../labeling/build_content_catalog.py)).

**Theorem 1 (Soundness of `B_content`).** If `B_content(c) = ρ(a) ≠ ⊥`, then `id*(c) = ρ(a)`, assuming (i) `H` collision-free and (ii) `ρ` correct on `A`.

*Proof.* `B_content(c) = ρ(a)` requires `H(b(c)) = H(β(a))`. By (i), `b(c) = β(a)` — the clip references a byte-identical copy of canonical artifact `a`. Byte-identical audio *is* the same recording, so `id*(c) = ρ(a)` by (ii). ∎

**Assumption (ii) is not free** — it is exactly the wrong-recording mis-attach class (`track_audio` row with a wrong `recording_id`, e.g. the `20911` case). It is discharged by: PR #72's same-song guard (prevents new mis-attaches), the `audit-gt` identity pass (0 identity mismatches on BB11/BB12), and the corpus mis-link audit (6 rows, 5 remediated). **`B_content` is only as sound as `ρ|_A`; the guard + audit are what make (ii) hold.**

**Why the poison is dead.** The `slot_id_map`/manifest channel bound via `σ(c)` (or the scraped tracklist row order). Placement order `σ` and tracklist row order are *different coordinate systems* (§4), so any `B_σ` is **unsound** — proved live: clip at `σ=002` has `id* =` Post Malone (Congratulations acappella, audio-verified), but tracklist row 002 = Manse (Freeze Time). `B_content` ignores `σ` entirely, so Theorem 1 holds regardless of the coordinate mismatch. **Soundness is achieved by construction: never read `σ`.**

---

## 3. The incompleteness, exactly characterized

**Theorem 2 (Incompleteness of `B_content`).** `B_content(c) = ⊥` whenever the clip's referenced audio is a non-trivial `T`-image of the true canonical artifact: `b(c) = t(β(a)), t ∈ T, t ≠ id`, even though `id*(c) = ρ(a)`.

*Proof.* `t ≠ id ⟹ H(b(c)) = H(t(β(a))) ≠ H(β(a))` (generically), and if no *other* catalog artifact happens to be byte-equal to `b(c)`, no hash match exists, so `B_content(c) = ⊥`. Yet `t ∈ T` preserves identity, so `id*(c) = ρ(a) ≠ ⊥`. ∎

**Empirical confirmation (BB12, 2026-07-22).** Of 739 local stem files, **467 hash into the catalog, 272 do not.** The 272 are `T`-images: the same recording has multiple local stem dirs (`Circle Of Life`, `Circle Of Life (…)`, `Circle Of Life [126bpm 8B]`) from (a) annotator `[NNNbpm KK]` renames, (b) no-`--prune` re-pulls stacking un-tagged fresh dirs beside tagged old ones, and (c) **re-separation drift** (canonical stems regenerated Demucs→Roformer after the annotator aligned). The `.als` clip references one specific dir; if it is a `T`-image of the current canonical, `B_content` abstains despite known identity.

**The redownload/replace churn is a `T`-source, not a separate bug.** `replace_track_audio` deletes `a` and inserts `a'` with `β(a') ≠ β(a)` (a fresh download of the same song = a `T`-image). Every GT clip referencing the pre-replace bytes `β(a)` now abstains, because the catalog only holds `H(β(a'))`. So **the acquisition pipeline continuously manufactures incompleteness**: each replace/redownload silently un-binds any clip pinned to the old bytes. This is why "how did we even get this issue" — it is structural, not a one-off.

**The abstain set decomposes (BB12, 57 abstains):**
```
26  demucs/roformer stems   — T-images (re-separation / tag-dup drift)   → RECOVERABLE (§5)
 5  online_candidate        — re-encoded/replaced masters, T-images       → RECOVERABLE (§5)
26  ref_source=None / mix   — mix-extracts, bespoke edits, no counterpart → genuinely ⊥ (id*=⊥)
```
So ~31/57 are *false abstains* (identity known, `B_content` blind); ~26/57 are *true abstains* (`id*=⊥`). Completeness means recovering the first group and only the first group.

---

## 4. Why `σ` is unsound (the poison, formal)

Let `σ(c)` be the `.als` placement index and `rank(t)` the scraped-tracklist row of track `t`. The poison assumed `σ(c) = rank(id*(c))` — a bijection between placement order and tracklist order. **This is false:** the mix contains overlays, acappella throws, and mashups that have no tracklist row, and tracklist tracks that span multiple placement clips. So `σ` and `rank` are incomparable orderings; any map `slot_id_map : σ → R` fit on one set is a memorized permutation with no cross-artifact validity. Binding via `σ` therefore has unbounded error (measured: 30/31 disagreements with audio-verified content in §agreement-test). **Formal rule: no sound channel may read `σ(c)` or `rank`.** Every channel below reads only `b(c)` (bytes) or `π(c)` (physical provenance).

---

## 5. The composed resolver (sound + completeness-maximal)

Add two channels, each provably sound, ordered by soundness strength. Both are **`σ`-free**.

### 5.1 Provenance channel `B_prov` — `T`-invariant by construction

Every file the **pull** places into `~/aligning/<set>/…` is `rsync`-copied from one specific canonical artifact: `stems/<slot>__…/<stem>.flac ← /mnt/storage/stems/{τ}/<stem>.flac`, `tracks/… ← /objects/{τ}/…`. The pull *chooses* the source `τ` per destination path — a **physical copy-fact**, recorded in `manifest.json` as `path → τ`. Define:
```
B_prov(c) = ρ(a)   where τ(a) = prov(p(c)),  prov = the pull's path→τ map
          = ⊥       if p(c) has no pull-provenance (annotator-added / old dir)
```

**Theorem 3 (Soundness of `B_prov`).** If `B_prov(c) = ρ(a)` via `prov(p(c)) = τ(a)`, then `id*(c) = ρ(a)`.

*Proof.* `prov(p(c)) = τ(a)` means the pull wrote `f(c)` by copying `β(a)`. At copy time `b(f(c)) = β(a)`, so `id*(c) = ρ(a)`. Subsequent `T`-drift of `b(f(c))` (retag, re-separate) does **not** change `prov`, which records the *source*, not the current bytes — so `B_prov` is stable under exactly the `T` that breaks `B_content`. ∎

**`B_prov` is not `B_σ`.** `prov` is keyed on the *physical file path* the clip references (which canonical file was copied there), never on the placement index `σ` or tracklist rank. It survives the §4 coordinate mismatch because it never consults an ordering — only "which bytes were copied into this path." This is the crux distinction that makes it sound where `slot_id_map` was poison.

**Consistency certificate (Theorem 4).** `B_prov` must be *certified* against `B_content` before trust: `∀c : B_content(c) ≠ ⊥ ∧ B_prov(c) ≠ ⊥ ⟹ B_content(c) = B_prov(c)`. This is machine-checkable on every export (it is the agreement test already run). A single violation falsifies the pull's `prov` map and blocks the channel. (Contrast: the manifest-`σ` channel *fails* this test 30:1 — which is how we know it is poison and `prov` is not.)

### 5.2 Fuzzy channel `B_fuzz` — perceptual, gated

For clips with neither content match nor pull-provenance (annotator-added files, phase-cancel outputs, foreign dirs), the last sound-up-to-a-gate channel is a `T`-invariant *perceptual* fingerprint (chromaprint for masters; HuBERT-L9 for vocals, already the stem-identity lever):
```
B_fuzz(c) = ρ(a*)   where a* = argmax_a sim(b(c), β(a)),
                    if sim(b(c),β(a*)) ≥ τ  AND  margin(a*, a₂) ≥ δ
          = ⊥        otherwise
```
`sim` is `T`-invariant (fingerprints survive re-encode/re-separate). Soundness is **probabilistic**: bounded false-positive rate under the gate `(τ, δ)`, calibrated on GT so that measured precision ≥ target (e.g. 0.99). Uses the existing abstention-by-*margin* rule (state-of-record §1) — never absolute cosine.

### 5.3 The composition

```
B(c) = B_content(c)              if ≠ ⊥        # exact, Thm 1
       else B_prov(c)            if ≠ ⊥        # copy-fact, Thm 3 (certified by Thm 4)
       else B_fuzz(c)            if passes gate # perceptual, sound w.h.p.
       else ⊥                                   # id*(c)=⊥, genuinely unrecoverable
```
Each row is stamped in `id_source ∈ {content, provenance, fuzzy, abstain}` for auditability.

**Theorem 5 (Soundness of `B`).** `B` is sound. *Proof.* Each channel is individually sound (Thms 1, 3; `B_fuzz` sound w.h.p. under its gate). Ordering cannot introduce error because a channel only fires when the stronger ones abstain, and where two channels are both defined they agree (Thm 4 certifies content/prov; the gate bounds fuzz). ∎

**Theorem 6 (Completeness-maximality of `B`).** `B(c) = ⊥ ⟹` `c` has (a) no byte-equal canonical artifact, (b) no pull-provenance, and (c) no canonical artifact within perceptual `(τ,δ)`. Under the assumption that the reliable channels are {byte-identity, copy-provenance, perceptual-fingerprint}, this is the minimal abstain set: any binding of such a `c` would require reading `σ` (unsound, §4) or a sub-gate perceptual guess (violates the precision target). ∎

The residual `⊥` set is then, by Theorem 6, exactly the `id*=⊥` clips (mix-extracts / bespoke edits) plus any clip whose true source audio is *absent from `A` entirely* (never downloaded) — which is an **acquisition** gap, not a binding gap, and routes to the acquisition worklist (closing the loop with the redownload/replace flow: acquire the missing master → it enters `A` → `B_content`/`B_prov` bind it next export).

---

## 6. Pain points, exposed (summary table)

| # | Pain point | Formal statement | Consequence | Fix |
|---|---|---|---|---|
| P1 | Coordinate mismatch (the poison) | `σ ≠ rank`, incomparable orderings | `B_σ` unsound, unbounded error | Never read `σ` (done in Crush) |
| P2 | Content drift | `H` not `T`-invariant (Thm 2) | `B_content` false-abstains on `T`-images | Add `B_prov` (Thm 3) + `B_fuzz` |
| P3 | Redownload/replace churn | `replace` applies `t∈T` to `β(a)`, mutates `H` | every clip on old bytes silently un-binds | `B_prov` (survives `T`) + re-export after replace |
| P4 | Mis-attach (assumption ii) | `ρ|_A` may be wrong (the `20911` class) | `B_content` sound-*looking* but wrong | PR #72 guard + `audit-gt` + mis-link audit |
| P5 | No-prune dir stacking | multiple `T`-image dirs per song | clip may reference the non-canonical copy | `B_prov` keys on path→τ, dir-name-agnostic |
| P6 | Missing master | `id*(c) ∉ ρ(A)` (never acquired) | genuine ⊥, but recoverable by acquisition | route ⊥ to acquisition worklist |

---

## 7. Implementation shape (for the follow-on plan)

1. **`prov` map surfacing.** Have the pull emit `path → track_audio_id` (+ stem) into `manifest.json` (it already chooses the source; record it). `B_prov` reads it.
2. **Wire `B_prov` into the exporter** as the second channel; stamp `id_source='provenance'`.
3. **Consistency gate (Thm 4)** as a hard export check: content vs prov agreement = 100% on overlap or the export fails. (This is the safety proof that `prov` is not poison.)
4. **`B_fuzz`** reusing the fingerprint/HuBERT lane, gate `(τ,δ)` calibrated on the content∩prov-certified subset as ground truth.
5. **`⊥ → acquisition worklist`** emitter, closing the loop with `acquire_variant`/`replace_track_audio`.
6. **Re-export both GT sets**, expect coverage 66%→≈82% (BB12) with **zero** new wrong labels (Thm 5), then re-write-back.

## 8. Out of scope
- The acquisition/download policy itself (what to fetch) — separate engine.
- Placement/timing correctness (this doc is identity only).
- Fingerprint model choice — reuse the settled HuBERT-L9 / chromaprint lanes.

---

# v2 — corrections after Fable adversarial review (2026-07-22)

Fable reviewed §§1–7 against the code and found the **architecture correct but three proofs are theater**. The load-bearing adjudication first, then the fixes.

## v2.0 — The one thing that had to be true, is true
**`B_prov` is NOT a relabeled `slot_id_map`.** Fable traced `pull_set_for_alignment.py` (`fetch_tracks` L282–454, `resolve_slot_audio` L378): the scraped tracklist *does* decide which `τ`'s audio is copied into which slot path. But the poison's inference was **ordinal** (clip-at-placement-k ↔ tracklist-row-k, no physical link), whereas prov's inference is **physical at every hop** (clip → the file it literally references and *plays* → the bytes rsync'd there → `τ`). If the tracklist wrongly put recording R at slot 003, the file there still *contains R's audio*, the annotator aligned *that audio*, so `id*(c)=ρ(R)` regardless of whether row 003 "should" be R. **The tracklist governs what is *available*, not what the clip *is*.** §5.1's "never consults an ordering" survives. Everything below is about making the *proofs* honest, not about a broken architecture.

## v2.1 — Per-theorem verdicts (Fable)
| Thm | Claim | Verdict | Core gap |
|---|---|---|---|
| 1 | content sound | HOLDS-WITH-CAVEAT | export-time vs **annotation-time** bytes (P8); mdat = *codec-payload* equality, not presentation |
| 2 | incompleteness = T-images | HOLDS in effect, **wrong algebra** | Demucs vs Roformer stems are **siblings under a common ancestor**, not `t`-images; T is a lossy **monoid**, not a group |
| 3 | prov sound (copy-fact) | **FALSE as stated** | "copy-fact" elides assumption **(N)** = no non-T local write to a pull path since copy; `manifest.json` is overwritable last-writer state, no epoch; name-match ≠ copy-fact |
| 4 | consistency certificate | HOLDS as *falsification*, **FALSE as certification** | **selection bias**: overlap = untouched files (prov trivially right); prov fires only on the *touched* disjoint set it never certifies |
| 5 | composed sound | **OVERCLAIMS** | `B_fuzz` FP rate ε>0 ⟹ `B` is ε-sound, not (S)-sound; the "∎" hides the downgrade §0 forbids |
| 6 | completeness-maximal | **VACUOUS** | channel set assumed exhaustive = the conclusion; ≥2 cheap **sound** channels missing |

## v2.2 — The decisive correction: a content-history **hash ledger** (new primary lever)
The dominant recoverable classes — 26 demucs/roformer stems + 5 replaced masters (31/57 BB12 abstains) — are **not** T-images; they are **siblings of a common ancestor**. `replace_track_audio` and stem regeneration *delete the only sound evidence* (the old sha256). **That deletion, not hash non-invariance, is the incompleteness factory (P9).**

**Fix (strictly sound, no new trust assumption — strictly stronger than prov):** an append-only `content_history(recording_id, track_audio_id, sha256, payload_sha256, kind, generation, ts)` on pi. Never drop a sha256 on replace; hash stems *before* regeneration. The catalog emits historical hashes with a `generation` flag. Then a clip referencing *any prior generation's bytes* content-binds **exactly** (Theorem 1 applies verbatim to historical hashes — no (N), no fuzz, no perceptual gate). This converts the doc's entire "recoverable" set into sound byte-level binds *going forward*; it cannot resurrect already-discarded generations (prov/fuzz still earn their keep on the legacy backlog). **This demotes the provenance channel from primary completeness lever to a legacy-backlog fallback.**

## v2.3 — A free second sound channel: FLAC decoded-PCM MD5
FLAC's STREAMINFO stores an MD5 of the *decoded PCM* — tag-invariant **and** lossless-re-encode-invariant, the exact `mdat_sha256` analogue for the FLAC stem population (mdat is mp4-only; stems are FLAC). It sits in the first ~42 bytes of every stem file and is currently unused. Add it as a payload key in `content_hash.py` + `build_content_catalog.py` (verify non-null; some encoders skip it). Recovers re-encoded/re-containered stems soundly.

## v2.4 — Composition unsoundness bug to fix now (P7, ambiguity leak)
`_load_content_catalog` ([export_als_to_gt.py:184-223](../../labeling/export_als_to_gt.py#L184)) **deliberately drops** hash keys mapping to >1 `recording_id` (same bytes under two recordings — an active assumption-(ii) violation) so content **fail-closes to abstain**. In the naïve composition that abstain **falls through to `B_prov`, which binds the manifest `τ` anyway** — resolving by fiat the exact ambiguity content refused, at ~coin-flip odds. **Rule:** an ambiguous-key hit **hard-abstains across ALL channels**, and `B_prov` abstains if its `τ` participates in any ambiguous key. Promote this catalog ambiguity-drop into a **named lemma** — it is load-bearing for Theorem 1 and the v1 draft omitted it.

## v2.5 — Corrected theorems (superseding §§2–5)
- **Thm 1′ (content, corrected).** Sound **under (i) collision-free, (ii) `ρ|_A` correct, (iii) `id*` defined w.r.t. *annotation-time* bytes AND the file is unmutated since annotation** (staleness check: `.als` mtime vs file mtime / ledgered hash; else emit `content_mutated_since_annotation`). mdat/FLAC-PCM keys give *codec-payload* equality (identity-preserving up to container edits), not raw byte equality — restate accordingly.
- **Thm 3′ (prov, corrected).** Sound under (ii) **+ (N)** *made checkable*: replace `manifest.json` as the prov source with an **append-only pull ledger** `(path, τ, sha_written, inode, mtime, pull_generation)`; the exporter trusts prov only when the current file is byte- or payload-equal to the ledgered write (else abstain `modified_since_pull`). Tag-insensitive name-matching is an inference layer *on top of* the copy-fact, sound only when the tagged path is a same-inode rename — check it, don't assume it. (Note: `manifest.json` *already* records `local_path`+`track_audio_id`+stems at L818–836; the missing piece is the **copy-time hash + generation stamp**, so §7.1 is mis-scoped.)
- **Thm 4′ (certificate, corrected).** *Necessary, not sufficient.* Passing ⟹ "the map isn't scrambled," NOT "prov is right on the abstain set" (selection bias: it's certified only on the untouched stratum). Add: a **minimum overlap** requirement (a 3-file overlap certifies vacuously); a **shadow-`B_fuzz`** run over the content-bound set every export to measure fuzz FP in deployment-adjacent conditions; a per-channel agreement matrix. A single violation still kills the channel (keep that).
- **Thm 5′ (composition, corrected).** **Stratified soundness**, not blanket. `{content, historical-content, provenance}` rows sound under their hypotheses; `fuzzy` rows **ε-sound with measured ε**. (S)-critical consumers — **write-back and eval denominators** — consume only the sound strata via the `id_source` stamp. Delete the unqualified "B is sound ∎."
- **Thm 6′ (completeness, corrected).** Only *relative* maximality holds for {byte, prov, perceptual}. True maximality requires the missing sound channels: **hash ledger, FLAC-PCM-MD5, human attestation** (the annotator is the identity authority — the exporter itself asks for a manual map at [export_als_to_gt.py:528-530](../../labeling/export_als_to_gt.py#L528)), **derivation records** (Mac-side re-separations of a content-bound master). Add these and re-argue exhaustiveness, or weaken the claim honestly.

## v2.6 — Added pain points (extend §6 table)
| # | Pain point | Consequence | Fix |
|---|---|---|---|
| P7 | Ambiguity leak (content fail-closed drop falls through to prov) | prov binds an ambiguous key content refused | hard-abstain across all channels (v2.4) |
| P8 | Annotation-time vs export-time decoherence (`rsync --inplace` between label & export) | binds bytes the human never labeled; wrong-version-fixed-later ⟹ wrong identity | annotation-time `id*` + staleness check |
| P9 | Churn deletes evidence (replace/re-sep drops old hashes) | manufactures incompleteness at the source | content-history ledger (v2.2) |
| P10 | Catalog scope gap: catalog keys on `recording_id IS NOT NULL`, pull resolves via `COALESCE(recording_id, track_id)` | slots with only legacy `track_id` are pulled+aligned but **catalog-invisible** ⟹ guaranteed abstain on pristine bytes | widen catalog key to match pull resolution; it's a **4th abstain class** (not `id*=⊥`, not acquisition) |
| P11 | No FLAC payload key (mdat is mp4-only) | re-encoded FLAC stems needlessly abstain | STREAMINFO PCM-MD5 (v2.3) |
| P12 | Gate drift: `id_coverage` counts only `id_source=='content'` ([L769](../../labeling/export_als_to_gt.py#L769)); `.als` refs not `html.unescape`d before path match | ≥50% export gate misfires once prov lands; `&`-titled tracks silently mis-join | fix coverage to count sound strata; unescape refs |

## v2.7 — Corrected implementation priority (supersedes §7)
1. **P7 hard-abstain** (unsoundness bug — do first, cheap).
2. **Content-history ledger** on pi + catalog `generation` (biggest sound completeness win; kills the churn factory).
3. **FLAC PCM-MD5** payload key (free sound channel for the stem population).
4. **P10** catalog-scope widen to `COALESCE(recording_id, track_id)`.
5. **Pull ledger** `(path, τ, sha_written, inode, mtime, generation)` → sound `B_prov` (Thm 3′) for the legacy backlog the ledger can't retro-fill.
6. **Staleness check** (P8) + fix `id_coverage` (P12) + `.als` unescape.
7. **`B_fuzz`** last, **calibrated per-T-subclass** (master-re-encode vs cross-backend-stem vs phase-cancel — pooled precision does not transfer); HuBERT cross-backend (Demucs↔Roformer) similarity on the 26-stem class is an **unmeasured empirical claim** — measure before trusting.
8. Re-export both sets; expect 66%→~82%+ (BB12) with **zero** new wrong labels; re-write-back consuming sound strata only.

**Net:** content-first + historical-content + physically-grounded prov + free FLAC/mdat payload keys + gated perceptual, all σ-free, with soundness **stratified and honest** rather than blanket-claimed. The hash ledger + PCM-MD5 alone recover most of the recoverable set with **no new trust assumptions** — the cheapest sound path, missed by v1.

---

# v3 — identity is a product of axes, and acquisition is a separate space

v1/v2 wrote the binding codomain as a scalar `ρ(a) ∈ R`. That is wrong: the repo already keys playable identity on **three orthogonal axes** (`core/identity.py`, `RecordingAxes`), and *how the audio was obtained* is a **fourth, orthogonal** space that drives the entire drift/churn problem. Making both explicit changes what "sound" and "complete" must mean.

## v3.1 — The real identity codomain (WHAT it is)

```
id*(c) ∈  Work × Version × Stem × Variant × (Remixer?)
Version ∈ {original, remix, rework, altversion, edit, bootleg, mashup}   # track_metadata.version
Stem    ∈ {regular, acappella, instrumental}                            # track_audio.stem
Variant ∈ {regular, extended}                                           # track_audio.variant
key = version__stem__variant   (remixer → recording.version_artist)
```
A binding is **correct only if correct on every axis.** "Right work, wrong stem" (bound the instrumental when the clip is the acappella) and "right work, wrong variant" (bound the radio edit when the clip is the extended) are **wrong labels**, not partial credit. So Soundness (S) must be *per-axis*.

**Key structural fact (discharges most of the risk):** each `(work, version, stem, variant)` point is a **distinct `track_audio` row with its own `sha256`** (acappella/instrumental variants get their own `track_audio_id` — memory `project_variant_mert`). Therefore:

- **Content and provenance are axis-exact for free.** A byte-match (Thm 1) or a copy-fact (Thm 3′) resolves to *one specific `track_audio` row*, which *is* a single point `(work, version, stem, variant)`. Content/prov **cannot** confuse stem or variant — the row carries them. This is a strength, now stated.
- **The hash ledger must be keyed at axis granularity.** `content_history` keys on `track_audio_id` (which pins all axes), *not* bare `recording_id`. A clip of the *acappella* must bind only to acappella-row generations; a naïve `recording_id`-keyed ledger would let it bind a `regular`-row generation of the same work — a stem-axis error. **Ledger key = (recording_id, stem, variant) generation-chain**, and demucs/roformer stem entries carry the derived `stem` axis (`vocals→acappella`, `instrumental→instrumental`) exactly as `build_content_catalog` already does.

**`B_fuzz` is axis-LOSSY — the sharp caveat.** A perceptual fingerprint matches the *work/version* but is weak-to-blind on the other axes:
- **Stem:** chromaprint/HuBERT of an acappella vs the instrumental of the *same work* can score high (shared harmony/lyrics) — a fuzzy match does **not** certify the stem axis.
- **Variant:** a 3-min radio edit vs a 6-min extended of the same work partial-matches — a fuzzy match does **not** certify the variant axis.

So the `B_fuzz` gate (v2, §5.2) must be **per-axis**, not a single similarity: (a) work/version by fingerprint sim + margin, (b) **stem** by an independent vocal-presence/instrumental-null test (the HuBERT-L9 vocal gate already does this — `candidate_vocal_gate.py`), (c) **variant** by duration/coverage ratio. A clip binds by fuzz only if *all three* pass; otherwise abstain or bind only the axes that pass and mark the rest `⊥` (partial identity is honest; a guessed stem is not).

## v3.2 — The acquisition-provenance space (HOW it was obtained) — orthogonal to identity

Your "how was it downloaded / needed to retry / choose between candidates / regular that became extended" is a **separate axis from what the audio is.** It is the *dynamics* that generate the `content_history` generations and, crucially, decides **which acquisition operations preserve `id*` (the ledger may bind across them) vs move `id*` (the ledger must NOT).** Proposed taxonomy, grounded in the actual flows:

| Acquisition op | Repo flow | Effect on `id*` axes | Ledger rule |
|---|---|---|---|
| **fetch** (initial) | `ingest.main` (yt-dlp/SC/YTM), `spotdl` | establishes the point | new generation @ these axes |
| **retry / rescue** | `redownload_via_ytmusic/spotdl`, `mac_push_acquire` | **preserves** axes, new bytes (better/other rip) | same-axis generation — **bind across** ✓ |
| **re-separate** | Demucs→Roformer, Mac re-stem | **preserves** (work,version,variant); realizes stem | same (recording,stem) generation — **bind across** ✓ (the 26-stem class) |
| **retag / re-encode** | `tag_aligning_folder`, codec change | **preserves** all axes (T) | same-axis generation — **bind across** ✓ (mdat/FLAC-PCM key) |
| **candidate-select** | `candidate_vocal_gate` (HuBERT winner of N acappellas) | **picks one** of several *same-work* alternatives; losers may be **different recordings** (covers, live, wrong source) | bind **only the chosen** `track_audio_id`; **never** a rejected alternative — see v3.3 |
| **extend-derive** | acquire/construct extended | **MOVES** variant `regular→extended` | **different axis point** — a `regular` generation is **NOT** a valid bind for an `extended` clip ✗ |
| **version-correct** | `replace_track_audio --axis version` | **MOVES** version (`original↔remix`, wrong-remix→right) | old generation is a **different point** — **do not bind across** ✗ |
| **stem-correct** | `acquire_variant`, `replace_stem_audio` | establishes/moves stem | axis-scoped generation |

The `track_audio_correction` ledger already speaks this language: `axis ∈ {version, variant, stem, recording}`, `action ∈ {replace, add, relink, detach}`. **The content-history ledger is the byte-level shadow of the correction ledger**: every correction/acquisition event appends a generation stamped with `(track_audio_id, recording_id, version, stem, variant, sha256, payload_sha256, op, source, generation, ts)`. Binding to a historical generation is sound iff that generation's axes equal the clip's true axes — which the axis-granular key (v3.1) enforces mechanically.

**This is the answer to "how did we even get this issue," at the axis level:** the churn that manufactures incompleteness (P9) is *identity-preserving* acquisition ops (retry, re-separate, retag) whose only sin is changing the bytes — exactly the ops the ledger should let us bind across, and currently can't because it discards the old hash. Whereas the ops that *move* the axis (extend, version-correct, candidate-swap) **should** cause a mismatch — abstaining there is correct, not a bug.

## v3.3 — The candidate-selection sub-problem (choosing between acappellas/instrumentals)

When a slot has N candidate acappellas (`stems/<slot>/candidates/vocals/…`), `candidate_vocal_gate` scores each against the studio recording's *own separated vocals* (HuBERT-L9 matched filter) and emits a winner (high score + margin) or abstains (`WINNER.txt` → `ingest_candidate_winners`). The GT clip references the **chosen** candidate. Binding must therefore:

1. **Prefer content/prov/ledger** — they resolve the *exact chosen* `track_audio_id` byte-for-byte; they cannot drift to a rejected candidate. (This is why content bound 84/84 candidates cleanly — candidates are pristine downloads with stable hashes.)
2. **Constrain `B_fuzz`** — a fingerprint of the chosen acappella will *also* score high against **rejected same-work candidates** (they are the same song). So fuzz must match against the **specific chosen artifact's** fingerprint (or the recording's own separated vocals, as the gate does), **never** the work generically — else it can bind a cover/live/wrong-source loser. Formally: `B_fuzz` for a candidate-slot is gated on similarity to *the ingested winner's* embedding, with the gate's `⊥` inherited from `candidate_vocal_gate`'s own abstain (low margin / all-bad, e.g. Mumford "The Cave" all-choir-covers).
3. **Record the alternatives.** The acquisition ledger stores the losing candidates + why (score, margin) — both as training signal and so a later re-selection (better candidate arrives) is a *new generation*, not a silent overwrite that un-binds the old GT.

## v3.4 — Consequences for the theorems
- **Thm 1/3 (content/prov):** strengthen — they are **axis-exact** because they resolve a specific `track_audio_id`. State this; it is a soundness *bonus*, not a caveat.
- **Thm 5′ (composition):** the `id_source` stratification becomes **per-axis**: `content`/`provenance`/`historical-content` rows are sound on all four axes; `fuzzy` rows are sound only on the axes their per-axis sub-gate passed — the stamp must record *which axes* fuzz certified.
- **Thm 6′ (completeness):** the residual `⊥` now includes a legitimate **partial-identity** state: "work known, stem/variant unverifiable" is a *more complete and still sound* label than full `⊥`. Completeness-maximality is defined per-axis.
- **Ledger (v2.2) key correction:** `(recording_id, stem, variant)` generation-chains, not `recording_id` — otherwise the ledger itself introduces stem/variant-axis errors.

## v3.5 — Added implementation items (extend v2.7)
9. **Axis-granular `content_history` key** `(recording_id, stem, variant, generation)`; each generation stamps `(version, stem, variant, op, source)` mirroring `track_audio_correction`.
10. **Per-axis `B_fuzz` gate:** work/version (fingerprint+margin) ∧ stem (vocal-presence / instrumental-null) ∧ variant (duration ratio); emit partial identity + per-axis `id_source`.
11. **Candidate binding:** fuzz for candidate slots gated on the *ingested winner's* embedding (reuse `candidate_vocal_gate`), inheriting its abstain; never bind a rejected same-work alternative.
12. **Ledger records losing candidates** (score, margin, reason) so re-selection is a new generation, not a silent un-bind.

**Recommend Fable re-review v3** — the per-axis soundness split and the acquisition-op partition (preserve vs move `id*`) are new proof obligations that deserve the same adversarial pass v1 got.

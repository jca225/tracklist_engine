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

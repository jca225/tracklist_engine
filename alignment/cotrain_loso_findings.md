# Multi-set co-train + LOSO — first cross-set result (2026-07-11)

> **Headline numbers: [docs/alignment_status.md](../../docs/alignment_status.md)** (canonical).
> This doc owns the LOSO detail; the "identity transfers 100% / placement does
> not" result is folded into canonical §6. BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`.

**Gear:** `cotrain.py` (`SetStores`, `cotrain`, `run_loso`) + `train.py --loso`.
Co-train works because the head trains on set-agnostic materialized examples
(`build_examples`), so we concat per-set examples and train one head; LOSO wraps
that head around the held-out set with a scraped-cue placement anchor
(`anchor_sigma_s`, cues are aligner input not GT — leakage-free).

## Run: `train.py --loso --sets bb11,bb12` (MPS, ~4 min)

| held-out | trained on | spans | identity_acc | set_start MAE |
|---|---|---|---|---|
| bb11 (2nvzlh2k) | bb12 | 150 | **100%** | **18.6 s** |
| bb12 (1fsnxchk) | bb11 | 166 | **100%** | **1436 s** |

## The finding (the honest read)

- **Identity transfers cross-set — 100% both directions.** The co-trained head
  correctly names which song plays on a set it never trained on. This is
  anchor-independent (identity does not use the placement anchor) and is the
  robust, trustworthy signal here.
- **Placement does NOT transfer, and is wildly unstable across the two
  directions — bb11 18.6 s vs bb12 1436 s (>75×).** The learned MERT head carries
  no dependable cross-set placement signal; on an unseen set placement leans
  entirely on the scraped-cue anchor, which rescues bb11 but not bb12. This is
  consistent with the documented behavior (`mert_model.py:239-246`): on an unseen
  set the curves carry no placement signal and the decode collapses toward
  front-of-mix. **The head memorizes placement per-set; it does not learn
  transferable placement.**
- **What drives the bb11/bb12 asymmetry is NOT resolved (and it is NOT anchor
  coverage — bb12 has *more* real cues, 149 nonzero vs bb11's 41, yet places far
  worse).** Candidate causes: bb12's scraped cues may be in a different frame /
  less correct, or its decode is harder — n=2 cannot disambiguate. Do not
  over-explain this; the defensible claim is only "identity transfers, placement
  does not, unstably."

## Why it matters (north star: SOTA GT-closeness on all 1001tl data)

The SOTA-placement lever is **not** the current MERT head. To place tracks on
unseen sets we need either (a) the learned **trajectory decoder** (a placement
representation that generalizes), or (b) denser per-set **cue/GT** coverage. The
flywheel's value is exactly this: each newly labeled set now (i) trains the head
(identity already generalizes; more sets should sharpen it) and (ii) is measurable
held-out via `--loso`, so we see transfer improve set by set.

## Bugs the LOSO harness surfaced

The identity number was always valid (it ignores the anchor). One real wiring bug
corrupted the *placement* number:
- **`_cue_anchor` read cue key `cue_seconds`; `fetch_slot_rows` emits `cue_s`** →
  empty anchor → floor mode (fixed, commit e84877b). This was the actual unblocker.

A second "fix" I made was a **wrong turn, reverted**: I added `_pad_slot` to
zero-pad cue labels, believing GT/cue labels mismatched. They don't in the real
lookup — `normalize_slot` strips padding on *both* sides, so unpadded cue keys
already exact-match. `_pad_slot` re-padded them and *forced the coarser
interpolation fallback*, degrading bb11 (18.6 s → 32.7 s). The final-review caught
this; `_pad_slot` was removed and bb11's 18.6 s restored. Lesson: the "0 overlap"
I measured was an artifact of comparing raw labels instead of normalized ones.

## Caveats

- **n = 2.** One number per direction — directional, not a benchmark. Its value is
  proving the gear + the robust identity-transfers/placement-doesn't split, and
  estimating that a 3rd labeled set is worth training on. The bb11/bb12 placement
  asymmetry is a lead to investigate, not a measured law.
- Requires bb11+bb12 MERT stores (local cache / pi). Cross-set anchor uses scraped
  cues (`fetch_slot_rows`), never held-out GT — leakage-free (final-review confirmed).

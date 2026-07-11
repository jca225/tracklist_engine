# Multi-set co-train + LOSO — first cross-set result (2026-07-11)

**Gear:** `cotrain.py` (`SetStores`, `cotrain`, `run_loso`) + `train.py --loso`.
Co-train works because the head trains on set-agnostic materialized examples
(`build_examples`), so we concat per-set examples and train one head; LOSO wraps
that head around the held-out set with a scraped-cue placement anchor
(`anchor_sigma_s`, cues are aligner input not GT — leakage-free).

## Run: `train.py --loso --sets bb11,bb12` (MPS, ~4 min)

| held-out | trained on | spans | identity_acc | set_start MAE | cue-anchor coverage |
|---|---|---|---|---|---|
| bb11 (1fsnxchk) | bb12 | 150 | **100%** | **32.7 s** | 116/143 (81%) |
| bb12 (2nvzlh2k) | bb11 | 166 | **100%** | **1416 s** | 39/155 (25%) |

## The finding (the honest read)

- **Identity transfers cross-set — 100% both directions.** The co-trained head
  correctly names which song plays on a set it never trained on. This result is
  anchor-independent (identity does not use the placement anchor) and is the
  robust, trustworthy signal here.
- **Placement does NOT transfer.** set_start MAE tracks *anchor coverage*, not
  training: well-anchored bb11 (81%) → 33 s; poorly-anchored bb12 (25%) → 1416 s,
  because its 75% unanchored spans have no placement signal on an unseen set and
  collapse toward front-of-mix. This confirms, at the model level, the documented
  behavior (`mert_model.py:239-246`): "on an unseen set the curves carry no
  placement signal and the DP collapses to the front of the mix." The learned MERT
  head **memorizes placement per-set**; it does not learn transferable placement.

## Why it matters (north star: SOTA GT-closeness on all 1001tl data)

The SOTA-placement lever is **not** the current MERT head. To place tracks on
unseen sets we need either (a) the learned **trajectory decoder** (a placement
representation that generalizes), or (b) denser per-set **cue/GT** coverage. The
flywheel's value is exactly this: each newly labeled set now (i) trains the head
(identity already generalizes; more sets should sharpen it) and (ii) is measurable
held-out via `--loso`, so we see transfer improve set by set.

## Bugs the LOSO harness surfaced (all fixed)

Two stacked wiring bugs corrupted the *placement* numbers before this run (the
identity number was always valid — it ignores the anchor):
1. `_cue_anchor` read cue key `cue_seconds`; `fetch_slot_rows` emits `cue_s`
   → empty anchor → floor mode (commit e84877b).
2. GT slot labels are 3-digit zero-padded (`038`, `026w1`); scraped cues are
   unpadded (`38`, `26w1`) → **0 anchor overlap** → the "anchored" numbers were
   garbage until `_pad_slot` normalized them (commit 3dedf87).

## Caveats

- **n = 2.** One number per direction — directional, not a benchmark. Its value is
  proving the gear + revealing identity-transfers / placement-doesn't, and
  estimating that a 3rd labeled set is worth training on.
- bb12's low cue coverage (25%) inflates its MAE; even fully anchored it would
  show placement *rescued by cues*, not placement *transferred* — the finding holds.
- Requires bb11+bb12 MERT stores (local cache / pi). Cross-set anchor uses scraped
  cues (`fetch_slot_rows`), never held-out GT.

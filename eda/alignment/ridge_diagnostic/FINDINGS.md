# Ridge diagnostic findings

Date: 2026-07-18
SHA: 2d83ea2
Timelines used: `workspaces/alignment_prototype/out/1fsnxchk_agentic_timeline.json` (BB12), `workspaces/alignment_prototype/out/2nvzlh2k_agentic_timeline.json` (BB11)
n cases: 12
Contrast threshold (aid only): 2.0

## Decision rule
- ridge present in ≥1 channel → decoder/voting wall
- ridge absent in all channels → representation wall (encoder earned)

## Per-case table
| case_id | stem | span_class | place_err_s | hubert | chroma | fp_hit | instr_stem | verdict |
|---|---|---|---|---|---|---|---|---|
| bb12_42w1 | acappella | multiseg | 116.5 | absent | absent | absent | absent | representation_wall |
| bb11_23w2 | acappella | multiseg | 106.1 | absent | absent | absent | absent | representation_wall |
| bb11_39w3 | acappella | multiseg | 82.9 | absent | absent | absent | absent | representation_wall |
| bb11_39 | instrumental | multiseg | 82.1 | absent | absent | present | absent | decoder_wall |
| bb11_34 | instrumental | multiseg | 80.4 | absent | absent | present | absent | decoder_wall |
| bb12_42w2 | acappella | multiseg | 68.6 | absent | absent | absent | absent | representation_wall |
| bb12_10w3 | acappella | linear | 61.0 | absent | absent | absent | absent | representation_wall |
| bb12_42w3 | instrumental | multiseg | 60.8 | absent | absent | absent | absent | representation_wall |
| bb12_42w5 | regular | linear | 55.3 | absent | absent | present | absent | decoder_wall |
| bb12_41 | regular | linear | 53.4 | absent | absent | absent | absent | representation_wall |
| bb12_3w2 | regular | multiseg | 52.0 | absent | absent | present | absent | decoder_wall |
| bb12_39 | instrumental | multiseg | 46.3 | absent | absent | present | absent | decoder_wall |

No eye-check overrides: threshold labels matched all inspected heatmaps (bb12_42w5, bb11_39, bb12_42w1, bb11_23w2 fp_hit + hubert; spot-check bb11_34, bb12_3w2, bb11_39w3).

## Aggregate
- decoder_wall: 5/12
- representation_wall: 7/12

## What we saw (qualitative, ≤10 bullets)
- All five decoder_wall cases are **fp_hit-only**: HuBERT, chroma, and instr_stem never reach contrast ≥2.0 on any case.
- fp_hit ridges in decoder_wall are visually sharp (bb12_42w5 contrast 9.0: bright cluster ~ref 125–175; bb12_3w2 contrast 7.1: continuous diagonal under cyan overlay) but often **offset or on a parallel branch** from the GT cyan diagonal (bb12_42w5, bb11_39 three-segment multiseg).
- bb11_34 fp_hit shows two segment-scale diagonal bands with sparse hits; ridge_present at 3.1 is credible though weaker than bb12_42w5/bb12_3w2.
- Representation_wall acappella multiseg (bb12_42w1, bb11_23w2, bb11_39w3, bb12_42w2) is uniformly flat: fp_hit heatmaps are mostly black with scattered noise; HuBERT is textured but no dominant GT-band ridge.
- bb11_23w2 acappella fp_hit contrast 0.71 — visually the weakest fp_hit panel in the sample; no override warranted.
- bb12_10w3 (acappella linear, 61 s err) and bb12_41 (regular linear, 53 s err) also lack any channel ridge despite simpler span_class.
- Instrumental multiseg splits: bb11_39/bb11_34/bb12_39 → decoder_wall (fp_hit); bb12_42w3 → representation_wall (all channels flat).
- Largest placement errors (116 s, 106 s) are representation_wall acappella multiseg — superposition appears to destroy signal in every owned channel.
- HuBERT panels on decoder_wall cases (e.g. bb11_39, bb12_42w5) show mid-level texture but no GT-aligned bright diagonal; confirms contrast scores ~1.1–1.4 are not false negatives.
- Cases drawn from agentic timelines on identity-correct / placement-wrong spans; see [docs/alignment_status.md](../../docs/alignment_status.md) for headline placement metrics on BB11/BB12.

## Recommendation
The immediate next lever is **decoder/voting on fp_hit**, not a new encoder in this package: five of twelve hard cases show a usable fp_hit diagonal ridge (contrast 3.0–9.0, confirmed by eye) while HuBERT, chroma, and instr_stem stay flat — placement errors of 46–82 s with visible fingerprint signal implicate path selection, branch disambiguation, or fp_hit weighting in the existing stack rather than missing embeddings. The other seven representation_wall cases (five acappella multiseg, one acappella linear, one regular linear, one instrumental multiseg) have no ridge in any channel; freeze `out/cases.json` as the eval holdout for a future mashup-invariant encoder elsewhere, but **do not start encoder work here** until fp_hit-aware decoding is tried on the decoder_wall subset.

## Non-claims
No statistical generalization from n<50. No new aligner channel shipped.
Headline alignment numbers: see [docs/alignment_status.md](../../docs/alignment_status.md) only.

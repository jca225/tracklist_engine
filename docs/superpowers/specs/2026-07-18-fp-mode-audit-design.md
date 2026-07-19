# FP Mode-Audit Design (decoder_wall follow-up)

**Date:** 2026-07-18  
**Status:** locked  
**Home:** `eda/alignment/ridge_diagnostic/` (EDA; worktree `fp-hit-decoder-wall`)  
**Upstream:** [placement ridge diagnostic](2026-07-18-placement-ridge-diagnostic-design.md) FINDINGS — next lever = fp_hit decoder/voting

## Goal

On the 5 `decoder_wall` cases only, determine whether the GT placement sits in the
existing top‑K fingerprint offset candidates and was mis-selected, or never
appears as a candidate. Decides the next code change — no `infer` edits in this study.

## Method

Reuse `mix_fp_hits.offset_candidates` with infer defaults (`topk=6`, `gap_s=6`).
Per case compare candidates to GT audible onset and agentic `pred_set_start_s`.

### Verdicts

| Label | Criterion | Implication |
|---|---|---|
| `gt_in_topk_missed` | nearest top‑K `set_start` within 6s of GT; agentic pred farther | mode/selection bug |
| `gt_is_argmax` | argmax (or selected) candidate already within 6s of GT | gate/refine bug |
| `gt_absent` | no top‑K candidate within 6s of GT | hash/candidate generation gap |

## Deliverables

- `mode_audit.py` CLI
- `out/mode_audit.tsv`
- `MODE_AUDIT.md` recommendation

## Non-goals

No encoder, no `infer`/`harness` changes, no headline metric edits to `alignment_status.md`.

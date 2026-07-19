# FP mode audit — decoder_wall follow-up

Date: 2026-07-18  
Upstream: [FINDINGS.md](FINDINGS.md) (ridge diagnostic)  
Spec: [docs/superpowers/specs/2026-07-18-fp-mode-audit-design.md](../../../docs/superpowers/specs/2026-07-18-fp-mode-audit-design.md)  
Method: `mix_fp_hits.offset_candidates` with infer defaults (`topk=6`, `gap_s=6`); near threshold 6s.  
Artifact: `out/mode_audit.tsv`

## Aggregate (n=5)

| Verdict | n | Meaning |
|---|---|---|
| `gt_is_argmax` | **2** | Dominant fp candidate already ≈ GT; agentic still 55–82s off |
| `gt_in_topk_missed` | **2** | GT in top‑K (ranks 2 and 5); wrong mode selected |
| `gt_absent` | **1** | No candidate within 6s (nearest ~10s) |

## Per case

| case_id | stem | place_err_s | verdict | argmax_err_s | nearest_rank | nearest_err_s |
|---|---|---|---|---|---|---|
| bb11_39 | instrumental | 82.1 | `gt_is_argmax` | 0.9 | 1 | 0.9 |
| bb11_34 | instrumental | 80.4 | `gt_in_topk_missed` | 33.2 | 2 | 1.9 |
| bb12_42w5 | regular | 55.3 | `gt_is_argmax` | 0.2 | 1 | 0.2 |
| bb12_3w2 | regular | 52.0 | `gt_absent` | 10.2 | — | — |
| bb12_39 | instrumental | 46.3 | `gt_in_topk_missed` | 29.5 | 5 | 0.3 |

## What this means

The ridge study was right that fp signal exists — and for **4/5** cases the live
`offset_candidates` stack already has a near-GT mode (argmax or lower rank).
Agentic placement is not limited by missing landmarks on this subset.

Split next lever (both earned; neither is an encoder):

1. **Gate / overwrite (priority — `gt_is_argmax`)** — On bb11_39 and bb12_42w5 the
   fp argmax is within 1s of GT while agentic sits 55–82s away. Something after
   (or instead of) raw fp argmax — the 90s MERT consistency gate, agentic
   refine, or stem routing — discarded a correct diagonal. First code dig:
   for these two slots, did classical `--fp-placement` place correctly and
   agentic move it, or did the gate keep MERT?

2. **Mode selection (`gt_in_topk_missed`)** — bb11_34 (rank 2) and bb12_39 (rank 5)
   need better disambiguation among top‑K (monotonic decode / vote sharpness /
   prior), not new features.

3. **Holdout** — bb12_3w2 is a soft miss (10s vs 6s threshold); revisit after (1)–(2),
   not with an encoder.

## Non-claims

n=5 looking exercise. No `infer` change shipped here. Headline metrics:
[docs/alignment_status.md](../../../docs/alignment_status.md) only.

## Run

```bash
venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.mode_audit
```

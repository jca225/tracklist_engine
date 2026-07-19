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

### Dig: `gt_is_argmax` overwrite (2026-07-18)

| case | classical / `_lt` | agentic | smoking gun |
|---|---|---|---|
| bb12_42w5 | fp @ 3519.5 (err 1.9s) | `agentic:surprise` @ 3466 (err 55s) | Events show fp proposed 3519.5 (conf 0.8) then surprise z=5.1 snapped to mert band and **won the belief vote** |
| bb11_39 | `_lt` fp @ 3399.8 (err 0.9s); race classical kept mert | `agentic:cue_prior` @ 3481 (err 82s) | cue_prior stamped 3481; fp not in that agentic `probe_proposals` (race classical stale vs `_lt`) |

Root cause on bb12_42w5: `SpanBelief.best()` took the heaviest cluster; mert+surprise co-cluster (same band, correlated independence group) outvoted lone fp even though fp was GT-correct.

### Fix landed (this branch)

`belief.best()` now prefers a competitive fp-containing cluster when its weight is
≥ `FP_CLUSTER_MARGIN` (0.5) × the heaviest cluster — see
`test_fp_cluster_preferred_over_mert_surprise_pileup`. Weak stray fp still loses
(`test_weak_fp_cluster_does_not_steal_strong_mert_pileup`).

### Dig: `gt_in_topk_missed` (reclassified)

| case | mode_audit | actual stack | implication |
|---|---|---|---|
| **bb12_39** | nearest = vote-rank 5 | classical / `_lt` / predicted **already place fp @ GT (0.3s)**; agentic overwrote with `cue_prior+surprise` @ +46s | Same class as bb12_42w5 — covered by fp-cluster preference (`test_fp_preferred_over_cue_prior_surprise_overwrite`) |
| **bb11_34** | nearest = vote-rank 2 (163 votes) vs argmax 413 @ +33s | classical/`instr_fp`/`_lt` all pick the **wrong** diagonal (~3023); mert prior is also nearer the wrong one | True remaining mode-selection hard case — not an agentic overwrite |

So the belief fix targets **3/5** decoder_wall cases (2× `gt_is_argmax` + bb12_39). Only bb11_34 needs a decode-side candidate re-rank; bb12_3w2 remains a soft holdout.

**Still open:**

1. **bb11_34 wrong-diagonal** — high-vote false mode beats GT (2.5× votes); mert prior does not disambiguate toward GT. Needs a new signal (sharpness / extent / neighbor monotonicity), not belief fusion.
2. **bb11_39 race path** — ensure agentic actually *runs* fp for that slot (cue_prior-only belief can't be rescued by the tie-break).
3. **Holdout** — bb12_3w2 soft miss (~10s).
4. **Re-race** — agentic on BB11/BB12 when parallel agents are clear; expect movement on bb12_42w5, bb12_39, maybe bb11_39 if fp fires.

## Non-claims

n=5 looking exercise. No `infer` change shipped here. Headline metrics:
[docs/alignment_status.md](../../../docs/alignment_status.md) only.

## Run

```bash
venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.mode_audit
```

# pws_aligner — Programmatic Weak Supervision Aligner

This fork rebuilds the alignment model as programmatic weak supervision per [docs/superpowers/specs/2026-07-14-pws-aligner-design.md](../../docs/superpowers/specs/2026-07-14-pws-aligner-design.md).

## Sensor phase frozen

No new probes (chroma, fp, HuBERT, etc.) are developed here. The package aggregates evidence from the existing `alignment_prototype` channel inventory via `Vote` records and a label model (Dawid–Skene fusion). All innovation is in the weak-supervision aggregation layer, not in new sensors.

## Kill-gate

**If Task 5's Dawid–Skene fusion does not beat `source_priority` on the BB11 + BB12 scorecard, stop before FABLE.** The programmatic weak-supervision approach must show measurable improvement over the existing axis-priority fusion to justify the added complexity. If the label model does not lift performance, pivot back to `alignment_prototype`'s classical driver.

# ridge_diagnostic — placement ridge diagnostic EDA

Answers one binary question for identity-correct / placement-wrong spans on BB11
(`2nvzlh2k`) and BB12 (`1fsnxchk`):

- **Decoder/voting wall** — a usable diagonal ridge already exists in ≥1 existing
  representation channel, but the aligner missed placement; or
- **Representation wall** — superposition destroyed the ridge in every channel we
  own, so a mashup-invariant encoder would be *earned*, not assumed.

Full design: [docs/superpowers/specs/2026-07-18-placement-ridge-diagnostic-design.md](../../docs/superpowers/specs/2026-07-18-placement-ridge-diagnostic-design.md).

## Decision rule

Per (case, channel): `ridge_contrast = mean(M[GT band]) / mean(M[background])`.
If contrast ≥ threshold (default 2.0), label **ridge_present** — a *looking aid*,
not a statistical claim.

Per case aggregate:

- **ridge present** in ≥1 of four channels → **decoder_wall**
- **ridge absent** in all available channels → **representation_wall**

## Run

From repo root:

```bash
venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.run \
  --n 12 \
  --min-place-err-s 15 \
  --bin-s 0.5 \
  --contrast-threshold 2.0
```

Useful flags:

- `--cases-json eda/alignment/ridge_diagnostic/out/cases.json` — skip selection; replot from a saved case list
- `--sets 1fsnxchk,2nvzlh2k` — limit which sets contribute candidates
- `--timeline SET=PATH` — override agentic/predicted timeline (repeatable)

Dry run (2 cases):

```bash
venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.run --n 2 --min-place-err-s 15
```

## Inputs

- **Timelines:** `alignment/out/<set_id>_agentic_timeline.json`
  (preferred) or `*_predicted_timeline*.json`, scored via `score_timeline_vs_gt`.
- **Ground truth:** `labeling/fixtures/bb11_ground_truth.yaml`,
  `labeling/fixtures/bb12_ground_truth.yaml`.
- **Audio:** pulled aligning folders under `~/aligning/<set_id>__*/` (mix, ref
  tracks, optional `mix_instrumental.flac` + ref instrumental stems).
- **Features:** HuBERT-L9 / chroma via existing `.feat_cache/` helpers; first
  extract can be slow.

## Outputs (`out/` — gitignored)

| Artifact | Description |
|---|---|
| `cases.json` | Selected hard cases with GT provenance |
| `contrast_table.tsv` | Ridge contrast + per-channel verdict |
| `heatmaps/<case_id>__<channel>.png` | Similarity matrix with GT diagonal overlay |

The CLI prints a one-screen summary (per-case channel verdicts + decoder vs
representation counts). Human-written `FINDINGS.md` is a separate step.

## Channels

| Channel | Pair |
|---|---|
| `hubert` | mix crop ↔ ref crop (HuBERT-L9 pooled) |
| `chroma` | same |
| `fp_hit` | landmark co-occurrence density |
| `instr_stem` | mix_instrumental ↔ ref instrumental (skipped if missing) |

Reference audio is tempo-stretched by GT `tempo_ratio` before comparison. GT
diagonal overlay uses crop origins from the same pad-based bounds as
`compute_panel`.

### Mix side: full mix vs `mix_vocals`

For `hubert`, `chroma`, and `fp_hit`, the mix crop is taken from the first
`mix.*` file in the aligning folder (full mix), **not** `mix_vocals.flac` —
even when the case is acappella. The ref side still follows `ref_audio_for`
(vocal stem for acappella, full ref for regular). `instr_stem` alone uses
`mix_instrumental.flac` ↔ ref instrumental.

This matches current `features._mix_full_path` behavior. Acappella
representation_wall verdicts may be biased toward "no ridge" when vocals are
buried in the full mix; a follow-up pass could re-run those channels on
`mix_vocals` without changing the encoder/feature math here.

## Non-goals

- **Do not** build or train a mashup-invariant encoder from this package.
- **Do not** add probes/channels to `alignment/{infer,harness,drivers}`.
- **Do not** claim statistical generalization from n&lt;50.

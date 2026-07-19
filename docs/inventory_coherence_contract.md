# Inventory coherence contract

Status: 2026-06-23. Consumed by alignment training export and ingest QA.

## Invariants

1. **SlotClaim** — every playable row in `set_track_slots` has `slot_label`, `recording_id`, `claimed_stem`, `claimed_variant`, and derived `layer_role`.
2. **AssetCandidate** — every satisfied slot has a `track_audio` row whose file exists on pi-storage at `path`.
3. **ReferenceChoice** — ground truth `ref_source` documents how the annotator obtained usable audio when it differs from canonical reference.
4. **Write-back** — GT export + `reconcile_gt_inventory.py` close the loop from labeling to inventory.

## Satisfaction statuses

| Status | Meaning |
|--------|---------|
| `satisfied` | Exact stem/variant match |
| `fallback` | Substituted tier (e.g. regular for extended) |
| `missing` | No track_audio |
| `wrong_stem` | Payload slot has only regular full track |
| `wrong_recording` | Identity mismatch (future fingerprint gate) |

## Layer roles

| Role | Typical slot | Expected asset |
|------|--------------|----------------|
| `bed` | primary in mashup section | full mashup / instrumental bed |
| `payload` | `NNNw1` | acapella or vocal stem |
| `constituent` | other w/ rows | source stems |
| `solo` | non-concurrent primary | regular reference |

## Tools

| Command | Purpose |
|---------|---------|
| `make check-inventory SET=…` | Pre-pull gate |
| `python -m labeling.reconcile_aligning_manifest <set_id> [--apply]` | Mac manifest repair from pi slots + disk |
| `scripts/reconcile_gt_inventory.py --yaml …` | GT → action CSV |
| `scripts/apply_stem_matches.py` | Discord → canonical |
| `scripts/ingest_candidate_winners.py` | candidates/ → canonical |
| `scripts/aligning_refresh.py` | Post-pull ALS tagging chain |

## Proxy / unalignable

Some slots have **no commercial release** (e.g. BB12 slot `032` — Lux Holm - Omega).
Labeling uses the set's own **`mix_instrumental.flac`** as a host/proxy bed, not a
fake `track_audio` row on pi-storage.

| Layer | Contract |
|-------|----------|
| **Mac manifest** | `reconcile_aligning_manifest` wires `local_path` to `<set_dir>/mix_instrumental.flac`, `stem="instrumental"`, `satisfaction="fallback"`, `gap="proxy:mix_instrumental — original unavailable (…)"` via `PROXY_SLOT_AUDIO`. |
| **Ableton / GT** | Clip references `mix_instrumental.flac`; `export_als_to_gt._PLACEHOLDER_RE` marks it `unalignable: true` with `source_note` containing `mix_instrumental` (SSOT for placeholder detection). |
| **Scoring / training** | Honor `unalignable: true` / `skip_training` — do **not** score proxy spans as normal identity+trajectory targets. |

If an ALS clip still points at a missing Lux file, relink once to `mix_instrumental.flac`
and re-export GT. Do not invent pi `track_audio` for unavailable originals.

## Training bundle fields (Phase 8)

GT export should include per play: `layer_role`, `satisfaction`, `ref_source`, `resolve_tier`.

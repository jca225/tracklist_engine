# Stem discovery playbook — acappella / instrumental re-source

**Status:** 2026-05-30  
**Coordination:** read [agent_handoff_identity_rollout_20260530.md](agent_handoff_identity_rollout_20260530.md) before pi-storage DB work. Do not start a second `tokenizer.materialize` or re-run migrations.

**Related:** [alignment_review_20260530.md](alignment_review_20260530.md), [alignment_objective.md](alignment_objective.md) (stem discovery = ingest, not aligner).

---

## Baby rule

- One file per slot under `~/aligning/<set>/tracks/`.
- If the DJ played an **acappella** or **instrumental**, that full file should be the reference audio (`track_audio.stem` + `is_reference=1`), not a muddy YT Music auto-download.
- Demucs `stems/vocals` etc. come from the analyzed reference `track_audio_id`.

---

## Three places audio lives

| Place | Role |
|-------|------|
| Downloads | Scratch — not registered |
| pi-storage `objects/` + DB | **Canonical library** |
| `~/aligning/<set>/` | Ableton working copy (from `pull_set_for_alignment.py`) |

---

## Two QA layers

### Layer A — Identity (semi-automatic)

After ingest, chromaprint vs the `regular` sibling ([ingest/adapters/fingerprint.py](../ingest/adapters/fingerprint.py)):

- `FALLBACK_TO_ORIGINAL` — got the full master, not a stem cut
- `WRONG_SONG` — likely different song (instrumental)
- `DURATION_MISMATCH` — wrong edit length
- `WEAK_SIGNAL` — acappella; confirm by ear

Printed by `acquire_variant` and `replace_stem_audio.py`.

### Layer B — Quality (human)

Listen. Log structured notes in `track_audio_correction.reason`:

```
quality:good|identity:OK|note:YouTube studio acapella vs YT Music muddy
```

Prefixes: `quality:`, `identity:`, `note:`

---

## Tools

| Task | Command |
|------|---------|
| First stem for a track | `scripts/acquire_variant.py --track-id … --role acappella\|instrumental --url …` |
| Replace bad stem row | `scripts/replace_stem_audio.py --track-audio-id OLD --url … --reason '…'` |
| Wrong full song / remix | `scripts/replace_track_audio.py` (`--axis version`) |
| Manual ledger row | `python -m ingest.corrections` |
| Pull for Ableton | `labeling/pull_set_for_alignment.py <set_id>` |
| GT timings | `python -m labeling.write_back_ground_truth --yaml …` |

`replace_track_audio.py` now supports `--stem` and inherits stem from `--track-audio-id` when omitted.

---

## YouTube search (not YT Music auto-query)

[ingest/search_query.py](../ingest/search_query.py) strips `(Acappella)` / `(Instrumental)` from YT Music search on purpose. For stem cuts, search YouTube explicitly:

- `{artist} {title} acapella`
- `{artist} {title} instrumental`

Shortlist 2–3 links → fingerprint Layer A → listen Layer B → ingest winner.

Use distinct `--player-id` per attempt (`yt-acap-v1`, `yt-acap-v2`).

---

## After materialize completes

1. Audit stem slots: `set_track_slots.claimed_stem` for your `set_id`.
2. Re-source flagged tracks on pi-storage (no second materialize).
3. Fresh pull → new Ableton project.
4. Write GT YAML with `claimed_stem` → `write_back_ground_truth`.
5. Keep old aligning folder **un-pruned** if a legacy `.als` still references it.

---

## Do not (pi-storage)

- Second `tokenizer.materialize`
- `reconcile_orphans.py --apply` (done)
- `reconcile_orphans.py --apply-promotions` (blocked on `5uzdn35`)
- Re-run identity migrations
- Replace `9hp84x` acappella ref (`taid=758`) without explicit intent

---

## Verify materialize done

```bash
ssh pi-storage 'tail -3 /tmp/materialize.log'
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db \
  "SELECT COUNT(*) FROM track_metadata;"'
```

Expect `DONE —` in log and `track_metadata` count > 0.

# Audio, embeddings, and feature storage

How durable bytes live in this engine. Behavioral contract:
[`../dj_engine_pseudocode.md`](../dj_engine_pseudocode.md) §§2 and 6.
Implementation: `dj_kernel::artifact`, `ArtifactKind`, migrate `apply`.

## Rules of thumb

1. **Content address is identity for bytes.** `sha256(payload)` keys the object
   store. Two identical rips share one object; a re-encode is a new object.
2. **Paths and legacy `track_id`s are locators**, never identity. They may appear
   on `Artifact.source_uri` or in observations; they must not mint `RecordingId`.
3. **Features are derived artifacts**, not columns that overwrite audio rows.
   Every embedding / fingerprint / stem blob points at a producing `Run` and a
   parent audio (or session) artifact via `Derivation`.
4. **Migrate audio + GT + claims; regenerate MIR.** Stems, MERT, cues, and
   aligner outputs are cheap relative to human labels and scarce rips — and they
   must be recomputed under a pinned `ProcessSpec` anyway.

## Audio layout

```
{artifact_root}/
  objects/{sha256[0..2]}/{sha256}   # raw bytes (mp3, wav, als gzip, …)
  meta/{artifact_id}.json           # Artifact record (kind, media_type, hashes, uri)
```

| Kind | Use |
|------|-----|
| `Audio` | Track or mix rip |
| `AbletonSession` | `.als` / session archive |
| `FeatureBlob` | Embeddings, fingerprints, beat grids, **open-vocab OD JSON**, … |
| `ModelCheckpoint` / `FittedModel` | Trained heads (versioned) |
| `HtmlPage` / screenshots | Treat browser/Ableton captures as content-addressed images; OD is a derived FeatureBlob |

### Open-vocabulary object detection

Image bytes → `Artifact` (screenshot / spectrogram render). Detector run
(OWL-ViT, Grounding DINO, YOLO-World, …) writes a `FeatureBlob` of boxes under
a pinned `ProcessSpec` + model card. The Python `OpenVocabOdGatherer` turns
those detections into `EvidenceEmission` rows (`source_family=vision`) that
**cite** `image_artifact_id` / detection feature ids — paths are locators only.

Open vocab matters because Ableton UI and tracklist pages are not a fixed COCO
class set: queries like `"acappella clip"` or `"tracklist row"` are round-time
parameters, not a frozen taxonomy.


Cold storage (pi disk today, S3/R2 later) should use the **same sha256 keys** so
mirrors do not fork identity.

## Embeddings and features

```
audio artifact (sha256)
  │  Run ← ProcessSpec(name, version, code hash, params, env, model_refs)
  ▼
feature artifact (sha256)   # e.g. mert_measures.npz, clap.npy, chromaprint.json
```

Pin in metadata / `ProcessSpec`:

- Model id + revision (e.g. `m-a-p/MERT-v1-330M`, commit or HF revision)
- Layer / hop / sample rate / windowing
- Code `implementation_hash` (computed, not hand-authored)

Changing any of those produces a **new** feature artifact; old ones remain for
audit and co-training lineage.

Prefer blob files (npz, Arrow, msgpack) over float arrays stuffed into
SQLite. Provenance DB holds ids and edges; object store holds bulk.

## Online services

Dense music embeddings for **arbitrary local rips** are not reliably outsourced:

| Source | Role here |
|--------|-----------|
| Self-hosted MERT / CLAP / similar | Primary embedding sensors (Python) |
| AcoustID / Chromaprint lookup | Identity LF when the recording is known |
| Hugging Face Inference | Optional; pin revision + hash the response as an external artifact |
| Spotify / MB / Discogs | Metadata locators, not file embeddings |
| Commercial MIR (e.g. Cyanite) | Optional `EvidenceSource` family; store raw JSON as artifact |

**Online search agents** download *candidate audio*; once on disk, candidates
enter the same content-addressed pipeline. Do not treat a vendor embedding URL
as durable identity.

## Practical default (this repo)

1. Audio → `ArtifactKind::Audio` via `dj_migrate audio-put` (`put_audio`).
2. Embeddings → `analyze.mert` sensor (stub by default) → `feature-register`
   → `ArtifactKind::FeatureBlob` + `analyzed_from` + model card on
   `ProcessSpec.model_artifact_ids`.
3. LFs cite `feature_artifact_id` in `evidence_ids` (`lfs.feature_ref`).
4. Later: mirror `objects/{aa}/{sha256}` to S3/R2 with the same keys.

```bash
make features-demo    # synthetic wav, no live DB, stub MERT
```

`--real` on the Python sensor attempts tracklist_engine MERT; not required for
the wiring demo.

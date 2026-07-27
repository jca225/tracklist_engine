# Open-vocabulary object detection (Propose plug)

Tracklist labeling is multi-modal. Humans already stare at **frames**:

- Ableton arrangement / session screenshots
- 1001tracklists / browser captures
- Spectrograms rendered as images
- Set posters / YouTube keyframes (OCR hand-off)

Closed-set detectors (COCO) are a poor fit — we need round-time queries like
`"acappella clip"` or `"tracklist row"`. That is open-vocabulary OD
(OWL-ViT, Grounding DINO, YOLO-World, …).

## Contract fit

```text
image bytes  →  Artifact (content-addressed; Rust ArtifactStore)
     │
     │  Run + ProcessSpec + model card
     ▼
DetectionSet JSON  →  FeatureBlob
     │
     ▼
OpenVocabOdGatherer  →  EvidenceEmission (family=vision)
     │
     ▼
dj_migrate run-round / pws-fit   (Rust Decide → Promote → Act)
```

- Emission **value** is the categorical label (matrix-friendly).
- Boxes live on the `DetectionSet` FeatureBlob cited via `evidence_ids`.
- Missing image or no hit above threshold → **abstain**.
- OD does **not** mint `RecordingId`.
- Decide / Promote stay in Rust — this plug only Proposes.

## Usage

```python
from uuid import uuid4
from sensors.plugs import (
    ABLETON_QUERIES,
    EvidenceContext,
    OpenVocabOdGatherer,
)

screenshot_id = uuid4()  # already deposited image artifact
gatherer = OpenVocabOdGatherer(
    default_queries=ABLETON_QUERIES,
    detect=my_grounding_dino_fn,  # optional; stub if omitted
    min_score=0.25,
)
ctx = EvidenceContext(
    image_artifact_ids=[screenshot_id],
    od_queries=["acappella clip", "cue marker"],
)
# emissions → append to vertical LF bundle → Rust Decide
```

Related: [storage.md](storage.md), agentic Propose in `sensors.plugs.agentic`.

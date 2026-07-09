"""core.contracts — typed records for artifacts that cross stage boundaries.

See README.md (inventory) and docs/entropy_reduction_plan.md (workstream A).
Load artifacts ONLY through these loaders; join set-scoped artifacts ONLY
through `join_guard`.
"""

from core.contracts.guard import SetMismatchError, join_guard
from core.contracts.ids import (
    RecordingId,
    SetId,
    SlotLabel,
    TlpId,
    TrackAudioId,
    normalize_slot_label,
)
from core.contracts.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestRow,
    load_manifest,
)
from core.contracts.timeline import (
    PredictedTimeline,
    RefSegment,
    SkippedSlot,
    TimelineSpan,
    load_timeline,
)

__all__ = [
    "Manifest",
    "ManifestRow",
    "PredictedTimeline",
    "RecordingId",
    "RefSegment",
    "SetId",
    "SetMismatchError",
    "SkippedSlot",
    "SlotLabel",
    "TimelineSpan",
    "TlpId",
    "TrackAudioId",
    "join_guard",
    "MANIFEST_FILENAME",
    "load_manifest",
    "load_timeline",
    "normalize_slot_label",
]

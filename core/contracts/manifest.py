"""Manifest — the typed record for `~/aligning/<set>/manifest.json`.

Writer: `labeling/pull_set_for_alignment.py` (the ONLY sanctioned writer — the
folder is a read-replica of pi-storage). Readers: infer's stem/lyrics ref
resolution, joint_ref_decode, the scorer's fiber ref-audio lookup, the als
identity layer, enrich_gt_track_ids.

Schema eras tolerated on load:
* current (2026-07): label + slot_label + identity axes + satisfaction fields
* pre-slot_label (BB10/BB12 pulls before 43c24a6): label only
* pre-identity-axes (Disco Lines): `version_tag`, no label at all

Fields consumers key on are validated; unknown extras ignored until the
writer emits through this module.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from core.contracts.ids import SetId, SlotLabel, normalize_slot_label


class ManifestRow(msgspec.Struct, frozen=True):
    track_id: str
    track_audio_id: int | None = None
    recording_id: str | None = None
    label: str | None = None
    slot_label: str | None = None
    artist: str = ""
    title: str = ""
    version: str | None = None
    stem: str | None = None
    variant: str | None = None
    axes_key: str | None = None
    local_path: str | None = None
    pi_path: str | None = None
    duration_s: float | None = None
    stems: dict[str, str] = {}
    layer_role: str | None = None
    satisfaction: str | None = None
    gap: str | None = None
    bed_slot: str | None = None
    resolve_tier: int | None = None
    version_tag: str | None = None  # pre-identity-axes era

    @property
    def slot(self) -> SlotLabel | None:
        return normalize_slot_label(self.slot_label or self.label)


class Manifest(msgspec.Struct, frozen=True):
    set_id: str
    tracks: tuple[ManifestRow, ...]
    title: str = ""
    mix_local_path: str | None = None
    mix_pi_path: str | None = None
    mix_duration_s: float | None = None

    @property
    def sid(self) -> SetId:
        return SetId(self.set_id)

    def by_track_id(self) -> dict[str, ManifestRow]:
        """Rows keyed by track_id, plus recording_id where present.

        Namespace bridging via labeling/fixtures/id_maps is the CALLER's
        concern until the entity_ids crosswalk lands (plan A3).
        """
        out: dict[str, ManifestRow] = {}
        for r in self.tracks:
            out[r.track_id] = r
            if r.recording_id:
                out.setdefault(r.recording_id, r)
        return out


_DECODER = msgspec.json.Decoder(Manifest)


def load_manifest(path: Path | str) -> Manifest:
    """Load + validate a pull manifest; raises with field detail on mismatch."""
    m = _DECODER.decode(Path(path).read_bytes())
    if not m.set_id:
        raise msgspec.ValidationError(f"{path}: manifest has empty set_id")
    return m

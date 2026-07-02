"""Local stem catalog with recording_id and key/BPM metadata."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from workspaces.mashup_compat.baseline import _bpm_fold, _fetch_key_bpm, _key_dist
from workspaces.mashup_compat.pairs import PI, PI_DB

_REPO = Path(__file__).resolve().parents[3]
STEM_ROOT = _REPO / "data" / "mashup_compat" / "stems"
TAID_CACHE = _REPO / "data" / "mashup_compat"


@dataclass(frozen=True)
class BedEntry:
    """Instrumental host bed."""

    recording_id: str
    track_audio_id: str
    label: str
    path: Path
    key_pc: int | None
    key_mode: str
    bpm: float


@dataclass(frozen=True)
class PayloadEntry:
    """Vocal overlay (acappella)."""

    recording_id: str
    track_audio_id: str
    label: str
    path: Path
    key_pc: int | None
    key_mode: str
    bpm: float


@dataclass(frozen=True)
class RegularEntry:
    """Full-song play (instrumental + vocals of the same recording)."""

    recording_id: str
    track_audio_id: str
    label: str
    instrumental_path: Path
    vocals_path: Path
    key_pc: int | None
    key_mode: str
    bpm: float


@dataclass(frozen=True)
class StemCatalog:
    beds: tuple[BedEntry, ...]
    payloads: tuple[PayloadEntry, ...]
    regulars: tuple[RegularEntry, ...] = ()


def _load_taid_map() -> dict[str, str]:
    """recording_id -> track_audio_id (cached from pi)."""
    cache_f = TAID_CACHE / "taid_map_vocals.json"
    if cache_f.is_file():
        return json.loads(cache_f.read_text())
    have = {p.name for p in STEM_ROOT.iterdir() if p.is_dir()}
    sql = (
        "SELECT recording_id, track_audio_id FROM track_audio "
        "WHERE stem='regular' ORDER BY is_reference DESC, track_audio_id;"
    )
    try:
        out = subprocess.run(
            ["ssh", PI, f'sqlite3 {PI_DB} "{sql}"'],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    raw: dict[str, str] = {}
    for line in out.strip().splitlines():
        rid, taid = line.split("|")
        if taid in have:
            raw.setdefault(rid, taid)
    cache_f.write_text(json.dumps(raw))
    return raw


def _fetch_features(recording_ids: list[str]) -> dict[str, tuple[int, str, float]]:
    feat: dict[str, tuple[int, str, float]] = {}
    batch = 80
    for i in range(0, len(recording_ids), batch):
        feat.update(_fetch_key_bpm(recording_ids[i : i + batch]))
    return feat


def load_catalog(*, require_key_bpm: bool = True) -> StemCatalog:
    """Beds (instrumental) and payloads (vocals) from local stem cache."""
    taid_map = _load_taid_map()
    if not taid_map:
        return StemCatalog(beds=(), payloads=())

    beds_raw: list[tuple[str, str, Path]] = []
    payloads_raw: list[tuple[str, str, Path]] = []
    regulars_raw: list[tuple[str, str, Path, Path]] = []
    for rid, taid in taid_map.items():
        inst = STEM_ROOT / taid / "instrumental.flac"
        voc = STEM_ROOT / taid / "vocals.flac"
        if inst.is_file():
            beds_raw.append((rid, taid, inst))
        if voc.is_file():
            payloads_raw.append((rid, taid, voc))
        if inst.is_file() and voc.is_file():
            regulars_raw.append((rid, taid, inst, voc))

    all_rids = sorted({r for r, _, _ in beds_raw} | {r for r, _, _ in payloads_raw})
    feat = _fetch_features(all_rids) if require_key_bpm and all_rids else {}

    beds: list[BedEntry] = []
    for rid, taid, path in beds_raw:
        kb = feat.get(rid)
        if require_key_bpm and not kb:
            continue
        key_pc, key_mode, bpm = kb if kb else (None, "", 0.0)
        beds.append(
            BedEntry(
                recording_id=rid,
                track_audio_id=taid,
                label=rid[:8],
                path=path,
                key_pc=key_pc,
                key_mode=key_mode or "",
                bpm=bpm,
            )
        )

    payloads: list[PayloadEntry] = []
    for rid, taid, path in payloads_raw:
        kb = feat.get(rid)
        if require_key_bpm and not kb:
            continue
        key_pc, key_mode, bpm = kb if kb else (None, "", 0.0)
        payloads.append(
            PayloadEntry(
                recording_id=rid,
                track_audio_id=taid,
                label=rid[:8],
                path=path,
                key_pc=key_pc,
                key_mode=key_mode or "",
                bpm=bpm,
            )
        )

    regulars: list[RegularEntry] = []
    for rid, taid, inst, voc in regulars_raw:
        kb = feat.get(rid)
        if require_key_bpm and not kb:
            continue
        key_pc, key_mode, bpm = kb if kb else (None, "", 0.0)
        regulars.append(
            RegularEntry(
                recording_id=rid,
                track_audio_id=taid,
                label=rid[:8],
                instrumental_path=inst,
                vocals_path=voc,
                key_pc=key_pc,
                key_mode=key_mode or "",
                bpm=bpm,
            )
        )

    return StemCatalog(
        beds=tuple(beds), payloads=tuple(payloads), regulars=tuple(regulars)
    )


def compatible(
    bed: BedEntry | PayloadEntry,
    payload: PayloadEntry,
    *,
    max_key_dist: int,
    max_bpm_fold: float,
) -> bool:
    if bed.key_pc is None or payload.key_pc is None:
        return False
    if bed.bpm <= 0 or payload.bpm <= 0:
        return False
    if bed.recording_id == payload.recording_id:
        return False
    if _key_dist(bed.key_pc, payload.key_pc) > max_key_dist:
        return False
    if _bpm_fold(bed.bpm, payload.bpm) > max_bpm_fold:
        return False
    return True


def pitch_shift_semi(bed: BedEntry, payload: PayloadEntry) -> int:
    if bed.key_pc is None or payload.key_pc is None:
        return 0
    delta = (bed.key_pc - payload.key_pc) % 12
    if delta > 6:
        delta -= 12
    return int(delta)


def tempo_ratio(bed: BedEntry, payload: PayloadEntry) -> float:
    if bed.bpm <= 0 or payload.bpm <= 0:
        return 1.0
    return payload.bpm / bed.bpm

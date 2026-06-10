"""Set-mix analysis: beat_this + Demucs on the full DJ mix.

Skips cue-detr (its training distribution is track-level EDM structure,
not 60-min mixes) and MERT (full-mix embeddings are heterogeneous, not
useful as-is). What we actually need from the set mix is:

1. A downbeat / measure grid on the mix axis, so stage-5 measure refinement
   can snap the Stage-1 warping path to mix measures.
2. Demucs stems of the mix, so stage-4 can compare `set.vocals` against
   `ref.vocals` per aligned measure to decide acappella / instrumental / full.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapters import beat_this_adapter

from core.models import SetAudioAsset
from core.result import Err, Ok, Result
from .errors import AnalysisError
from .models import BeatGrid, StemSet
from .pipeline import Analyzers, run_separation


@dataclass(frozen=True)
class SetAnalysisResult:
    set_audio_id: int
    beats: BeatGrid
    stems: StemSet
    analyzer_versions: dict[str, str]


def analyze_set(
    analyzers: Analyzers,
    asset: SetAudioAsset,
    stems_dir: Path,
    *,
    skip_stems: bool = False,
) -> Result[SetAnalysisResult, AnalysisError]:
    """Run beat_this + the selected stem backend on the full mix. `stems_dir`
    is the parent; output goes into
    `stems_dir/set/<set_audio_id>/{vocals,instrumental}.flac` to keep set stems
    clearly separated from track stems on disk.

    ``skip_stems=True`` runs the beat grid only (empty StemSet) — for when
    separation is scheduled on another host (e.g. roformer on Vast) and this
    host only needs ``measure_times_json`` for set-side MERT."""
    assert asset.set_audio_id is not None, "SetAudioAsset must be persisted before analysis"
    mix_path = Path(asset.path)
    set_stems_dir = stems_dir / "set"

    beats_r = beat_this_adapter.predict(analyzers.beats, mix_path)
    if not beats_r.is_ok():
        return beats_r
    beat_times, downbeat_times = beats_r.value
    bpm = beat_this_adapter.estimate_bpm(beat_times)
    measures = beat_this_adapter.measure_times(downbeat_times)

    if skip_stems:
        stems = StemSet(track_audio_id=asset.set_audio_id, stems=())
        versions = {"beat_this": analyzers.beats.version}
    else:
        # Reuse the selected stem backend — it already writes to stems_dir/<id>/.
        # Pass `set_audio_id` as the id; the "set/" parent keeps namespaces clear.
        stems_r = run_separation(
            analyzers, mix_path, set_stems_dir, asset.set_audio_id,
        )
        if not stems_r.is_ok():
            return stems_r
        stems = stems_r.value
        versions = {
            analyzers.separator: analyzers.stems_version,
            "beat_this": analyzers.beats.version,
        }

    return Ok(SetAnalysisResult(
        set_audio_id=asset.set_audio_id,
        beats=BeatGrid(
            track_audio_id=asset.set_audio_id,   # StemSet/BeatGrid reuse the field name
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            measure_times=measures,
            bpm=bpm,
        ),
        stems=stems,
        analyzer_versions=versions,
    ))

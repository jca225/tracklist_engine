# Mashup Compiler Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap a separate private product repo (`~/Desktop/mashup_compiler`, working title) that compiles a hard-coded mashup intent — *vocals of song A over instrumental of song B* — into rendered audio + an openable Ableton `.als`, consuming frozen engine modules copied from `tracklist_engine`.

**Architecture:** A pure-function compiler pipeline: `analyze` (per-song profile via copied adapters: beat_this grid, Essentia BPM/key, Roformer stems) → `gate` (BPM-ratio ±6% + key ±1 semitone, per the DJ-agent spec's locked v1 decisions) → `compile` (placement math → `MashupTimeline` IR) → two backends off the same IR: `render` (pyrubberband stretch/pitch + mix → wav) and `emit_als` (template-seeded clip construction, reusing the proven crash-safe deep-copy path from `seed_als_from_timeline.py`). No NL parsing in this plan — intent is CLI args. NL is a later plan.

**Tech Stack:** Python 3.14, lxml, numpy, soundfile, librosa, torch (MPS), beat-this, pyrubberband (brew `rubberband`), pytest. Essentia via the existing Py3.13 subprocess sandbox. Roformer via the existing MSST install (referenced in place, not copied).

## Global Constraints

- **Source repo:** `/Users/johnnycabrahams/Desktop/tracklist_engine` (called `$TLE` below). New repo: `/Users/johnnycabrahams/Desktop/mashup_compiler` (called `$MC`).
- **NEVER copy** anything from `web_crawler/`, `ingest/`, or scrape/download code. No `yt-dlp`/`spotdl` in this repo. The product operates on user-supplied audio only.
- **No audio files committed** except the `.als` template fixture and generated sine-tone test fixtures. `.gitignore` blocks `*.mp3 *.m4a *.wav *.flac` outside `tests/fixtures/`.
- **Copied engine code is FROZEN** — bugs get fixed upstream in tracklist_engine, then re-copied. `ENGINE_MANIFEST.md` records source commit + file list.
- **Style:** same as tracklist_engine — `from __future__ import annotations`, frozen dataclasses, full type hints, `Result` in library code, `sys.exit` only in `main.py`.
- **Heavyweight model installs are referenced in place via env vars** (defaults point at tracklist_engine): `MC_MSST_ROOT`, `MC_ESSENTIA_PYTHON`, `MC_ESSENTIA_MODELS`. Only small models (beat_this) download into this repo's cache.
- **Tests:** `venv/bin/python -m pytest tests/ -m "not integration"` must pass offline with no models. Model-dependent tests are `@pytest.mark.integration`.
- **Import rewrites on copy:** `core.result` → `engine.result`; `core.identity` → `engine.identity`; `analysis.errors` → `engine.errors`; `analysis.adapters.X` → `engine.X`; `labeling.als` → `engine.als`.
- **Every task ends with a commit in `$MC`** (new repo history starts clean — no tracklist_engine history).

---

### Task 1: Repo scaffold

**Files:**
- Create: `$MC/.gitignore`, `$MC/README.md`, `$MC/requirements.txt`, `$MC/pytest.ini`, `$MC/ENGINE_MANIFEST.md`, `$MC/engine/__init__.py`, `$MC/compiler/__init__.py`, `$MC/tests/__init__.py`

**Interfaces:**
- Produces: repo layout + venv every later task assumes; `pytest.ini` defining the `integration` marker.

- [ ] **Step 1: Create repo + git init**

```bash
mkdir -p ~/Desktop/mashup_compiler/{engine,compiler,tests/fixtures,out}
cd ~/Desktop/mashup_compiler && git init
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
venv/
__pycache__/
*.pyc
out/
.pytest_cache/
*.mp3
*.m4a
*.wav
*.flac
!tests/fixtures/*.wav
*.mashup_profile.json
.stems/
```

- [ ] **Step 3: Write `requirements.txt`**

```
lxml>=5.0
numpy>=2.0
soundfile>=0.13
librosa>=0.11
torch>=2.11
torchaudio>=2.11
transformers>=4.57
beat-this>=1.1
pyrubberband>=0.4
pyloudnorm>=0.2
pytest>=9.0
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
markers =
    integration: needs models/venvs from tracklist_engine (deselect with -m "not integration")
```

- [ ] **Step 5: Write `README.md`**

```markdown
# mashup_compiler (working title)

Compiles a mashup intent — vocals of song A over instrumental of song B —
into rendered audio + an editable Ableton Live set (.als).

User brings their own audio files. This repo contains no downloaders, no
scrapers, and no corpus audio.

## Setup
    brew install rubberband ffmpeg
    python3.14 -m venv venv && venv/bin/pip install -r requirements.txt

## Usage
    venv/bin/python -m compiler.main VOCAL_SONG.mp3 INSTR_SONG.mp3 \
        --drop-bar 16 --out out/demo

Produces `out/demo.wav` and `out/demo.als`.

`engine/` is frozen code copied from the private research repo — see
ENGINE_MANIFEST.md. Do not edit in place; fix upstream and re-copy.
```

- [ ] **Step 6: Write `ENGINE_MANIFEST.md`** (commit sha filled in Task 2)

```markdown
# Engine manifest

Frozen copies from tracklist_engine (private research repo).
Source commit: <FILLED-IN-TASK-2>
Copied files: see Task 2 of the bootstrap plan.
Rule: never edit engine/ in place — fix upstream, re-copy, update the sha.
Heavy installs referenced in place via env vars:
  MC_MSST_ROOT        (default: $TLE/workspaces/msst_webui)
  MC_ESSENTIA_PYTHON  (default: $TLE/venvs/essentia/bin/python)
  MC_ESSENTIA_MODELS  (default: $TLE/data/essentia_models)
```

- [ ] **Step 7: Create empty `__init__.py` files, venv, install**

```bash
touch engine/__init__.py compiler/__init__.py tests/__init__.py
python3.14 -m venv venv && venv/bin/pip install -r requirements.txt
brew list rubberband >/dev/null 2>&1 || brew install rubberband
```

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "chore: repo scaffold (working title: mashup_compiler)"
```

---

### Task 2: Copy frozen engine modules

**Files:**
- Create: `$MC/engine/result.py`, `engine/identity.py`, `engine/errors.py`, `engine/audio_io.py`, `engine/loudness.py`, `engine/beat_this_adapter.py`, `engine/essentia_adapter.py`, `engine/essentia_worker.py`, `engine/essentia_models.py`, `engine/roformer_chain_adapter.py`, `engine/roformer_config.py` (if it is a separate file upstream — check; it may live inside the adapter), `engine/paths.py` (new), `engine/als/` (whole package), `engine/als_assets/seed_template.als`
- Test: `$MC/tests/test_engine_smoke.py`

**Interfaces:**
- Produces: `engine.result.{Result, Ok, Err}`; `engine.audio_io.load_mono(path, target_sr) -> Result[Waveform, AudioIoError]`; `engine.beat_this_adapter.{load, predict, estimate_bpm}`; `engine.essentia_adapter.analyze(audio_path, track_audio_id, timeout_s) -> Result[EssentiaFeatures, EssentiaError]`; `engine.roformer_chain_adapter.{load, separate}` returning `StemSet` of `StemAsset(stem_name, path, ...)`; `engine.als.*` (full codec API); `engine.paths.{msst_root, essentia_python, essentia_models}`.

- [ ] **Step 1: Copy files and record provenance**

```bash
TLE=~/Desktop/tracklist_engine; MC=~/Desktop/mashup_compiler
cp $TLE/core/result.py                                  $MC/engine/result.py
cp $TLE/core/identity.py                                $MC/engine/identity.py
cp $TLE/analysis/errors.py                              $MC/engine/errors.py
cp $TLE/analysis/adapters/audio_io.py                   $MC/engine/audio_io.py
cp $TLE/analysis/adapters/loudness.py                   $MC/engine/loudness.py
cp $TLE/analysis/adapters/beat_this_adapter.py          $MC/engine/beat_this_adapter.py
cp $TLE/analysis/adapters/essentia_adapter.py           $MC/engine/essentia_adapter.py
cp $TLE/analysis/adapters/essentia_worker.py            $MC/engine/essentia_worker.py
cp $TLE/analysis/adapters/essentia_models.py            $MC/engine/essentia_models.py
cp $TLE/analysis/adapters/roformer_chain_adapter.py     $MC/engine/roformer_chain_adapter.py
cp -R $TLE/labeling/als                                 $MC/engine/als
mkdir -p $MC/engine/als_assets
cp $TLE/tests/labeling/fixtures/als/seed_template.als   $MC/engine/als_assets/seed_template.als
git -C $TLE rev-parse HEAD   # paste into ENGINE_MANIFEST.md "Source commit:"
```

Note: if `RoformerChainConfig` lives in a separate `roformer_config.py` / `separation_config.py` upstream, copy that file too and rewrite its import the same way. Also copy `engine/als`'s upstream `models.py` dependencies if `grep -r "from core" $MC/engine/als` reveals more than `normalize_stem`.

- [ ] **Step 2: Rewrite imports across `engine/`**

```bash
cd $MC
LC_ALL=C find engine -name '*.py' -exec sed -i '' \
  -e 's/from core\.result/from engine.result/g' \
  -e 's/from core\.identity/from engine.identity/g' \
  -e 's/from core import identity/from engine import identity/g' \
  -e 's/from analysis\.errors/from engine.errors/g' \
  -e 's/from analysis\.adapters\.audio_io/from engine.audio_io/g' \
  -e 's/from analysis\.adapters\.essentia_models/from engine.essentia_models/g' \
  -e 's/from analysis\.adapters import essentia_models/from engine import essentia_models/g' \
  -e 's/from labeling\.als/from engine.als/g' \
  -e 's/from labeling import als/from engine import als/g' {} +
grep -rn "from core\|from analysis\|from labeling\|import core\|import analysis\|import labeling" engine/ && echo "UNREWRITTEN IMPORTS — fix by hand" || echo OK
```

Also copy `engine/als`'s upstream `__init__.py` re-exports as-is; only its internal imports needed rewriting.

- [ ] **Step 3: Write `engine/paths.py` and repoint hardcoded paths**

```python
"""Locations of heavyweight installs referenced in place (not copied).

Defaults point at the sibling research repo on this machine; override via
env vars when the installs move."""
from __future__ import annotations

import os
from pathlib import Path

_TLE = Path.home() / "Desktop" / "tracklist_engine"


def msst_root() -> Path:
    return Path(os.environ.get("MC_MSST_ROOT", _TLE / "workspaces" / "msst_webui"))


def essentia_python() -> Path:
    return Path(
        os.environ.get("MC_ESSENTIA_PYTHON", _TLE / "venvs" / "essentia" / "bin" / "python")
    )


def essentia_models() -> Path:
    return Path(os.environ.get("MC_ESSENTIA_MODELS", _TLE / "data" / "essentia_models"))
```

Then in the copied adapters, find every repo-relative path constant and replace it with the matching `engine.paths` call:

```bash
grep -n "parents\[\|__file__\|venvs/\|essentia_models\|msst" engine/essentia_adapter.py engine/essentia_models.py engine/roformer_chain_adapter.py
```

- In `essentia_adapter.py`: the subprocess python must come from `paths.essentia_python()`. **Important:** the worker it invokes must be `$MC/engine/essentia_worker.py` (invoke by file path, not `-m analysis.adapters.essentia_worker`).
- In `essentia_models.py`: `models_dir()` returns `paths.essentia_models()`.
- In `roformer_chain_adapter.py`: `RoformerChainConfig.default()`'s `msst_root` comes from `paths.msst_root()`.

- [ ] **Step 4: Write the smoke test**

```python
# tests/test_engine_smoke.py
from __future__ import annotations

from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "engine" / "als_assets" / "seed_template.als"


def test_engine_imports() -> None:
    from engine import als, audio_io, beat_this_adapter, essentia_adapter  # noqa: F401
    from engine.result import Err, Ok  # noqa: F401

    assert Ok(1).is_ok() and not Err("x").is_ok()


def test_template_parses() -> None:
    from engine.als import load_als_xml, parse_layer_clips

    root = load_als_xml(TEMPLATE)
    parse_layer_clips(root)  # must not raise; template may have 0 layer clips
```

- [ ] **Step 5: Run tests**

Run: `venv/bin/python -m pytest tests/test_engine_smoke.py -v`
Expected: 2 PASS. If imports fail, the grep in Step 2 missed a rewrite — fix and re-run.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: frozen engine modules copied from tracklist_engine (see ENGINE_MANIFEST.md)"
```

---

### Task 3: Extract the crash-safe .als clip builder (`engine/als_seed.py`)

The clip-construction code lives in `$TLE/workspaces/alignment_prototype/review/seed_als_from_timeline.py` — it embeds three hard-won Ableton-crash fixes (`strip_automation`, `renumber_pointee_ids` incl. `<Pointee>`, `NextPointeeId` bump). Extract the reusable functions verbatim, with one extension: `rewrite_clip` gains a `pitch_coarse` parameter (upstream hard-codes `PitchCoarse=0`; mashups transpose ±1).

**Files:**
- Create: `$MC/engine/als_seed.py`
- Test: `$MC/tests/test_als_seed.py`

**Interfaces:**
- Consumes: `engine.als.{load_als_xml, save_als_xml, parse_layer_clips, write_tempo_envelope}`
- Produces: `ffprobe_audio(path) -> tuple[float, int]`; `find_template_track(root) -> _Element`; `build_track(template, *, track_id, track_name, name, color, arr_start, arr_end, file_path, file_dur_s, sample_rate, ref_start_s, ref_end_s, is_warped=True, pitch_coarse=0) -> _Element`; `strip_automation(track)`; `renumber_pointee_ids(track, alloc)`; `doc_max_id(root) -> int`

- [ ] **Step 1: Create `engine/als_seed.py`** by copying, from the source file above, the functions `ffprobe_audio`, `_set_value`, `find_template_track`, `rewrite_clip`, `strip_automation`, `build_track`, `doc_max_id`, `renumber_pointee_ids` — verbatim, except:
  - imports become `from lxml import etree` only (drop repo-path shims and `labeling.als` imports; none of these eight functions need them),
  - `find_template_track` raises `ValueError("no single-clip warped AudioTrack found in template .als")` instead of `sys.exit` (library code — errors as values/exceptions, exit stays in `main.py`),
  - `rewrite_clip` signature gains `pitch_coarse: int = 0` and the line `_set_value(clip, "PitchCoarse", "0")` becomes `_set_value(clip, "PitchCoarse", str(pitch_coarse))`.
  - module docstring: `"""Crash-safe Ableton track/clip construction, extracted frozen from tracklist_engine's seed_als_from_timeline.py (see ENGINE_MANIFEST.md). The strip_automation / renumber_pointee_ids / NextPointeeId trio is load-bearing: skipping any of them makes Live offer to "fix" the file and then crash."""`

- [ ] **Step 2: Write the failing test**

```python
# tests/test_als_seed.py
from __future__ import annotations

import copy
import itertools
from pathlib import Path

from engine.als import load_als_xml, parse_layer_clips, save_als_xml, write_tempo_envelope
from engine.als_seed import build_track, doc_max_id, find_template_track, renumber_pointee_ids

TEMPLATE = Path(__file__).resolve().parents[1] / "engine" / "als_assets" / "seed_template.als"


def test_build_one_clip_roundtrips(tmp_path: Path) -> None:
    root = load_als_xml(TEMPLATE)
    template_track = copy.deepcopy(find_template_track(root))
    tracks_node = root.find(".//LiveSet/Tracks")
    for t in list(tracks_node):
        if t.tag in ("AudioTrack", "GroupTrack", "MidiTrack"):
            tracks_node.remove(t)
    write_tempo_envelope(root, [(0.0, 128.0)])
    alloc = itertools.count(doc_max_id(root) + 2000)
    track = build_track(
        template_track,
        track_id=1000,
        track_name="1-acappella",
        name="acappella",
        color=5,
        arr_start=64.0,
        arr_end=192.0,
        file_path=Path("/tmp/vocals.wav"),
        file_dur_s=180.0,
        sample_rate=44100,
        ref_start_s=12.5,
        ref_end_s=72.5,
        pitch_coarse=1,
    )
    renumber_pointee_ids(track, alloc)
    tracks_node.insert(0, track)
    for npi in root.findall(".//NextPointeeId"):
        npi.set("Value", str(next(alloc)))
    out = tmp_path / "one_clip.als"
    save_als_xml(root, out)

    clips = parse_layer_clips(load_als_xml(out))
    assert len(clips) == 1
    c = clips[0]
    assert c.arr_start == 64.0 and c.arr_end == 192.0
    assert c.pitch_coarse == 1
    assert abs(c.ref_start_s() - 12.5) < 1e-6
    assert c.path.endswith("vocals.wav")
```

- [ ] **Step 3: Run test to verify it fails** — `venv/bin/python -m pytest tests/test_als_seed.py -v` → FAIL (`ModuleNotFoundError: engine.als_seed`) before Step 1 is done; after extraction it should PASS. If `parse_layer_clips` returns 0 clips, check the track-name prefix rules in `engine/als/read.py` (the parser may skip tracks named like the mix — `1-mix`/`2-mix` — the name `1-acappella` avoids that).

- [ ] **Step 4: Run test to verify it passes** — same command, expected PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: crash-safe als clip builder extracted from seeder (pitch_coarse extension)"`

---

### Task 4: Compiler IR (`compiler/models.py`)

**Files:**
- Create: `$MC/compiler/models.py`
- Test: `$MC/tests/test_models.py`

**Interfaces:**
- Produces (consumed by every later task):

```python
MashupIntent(vocal_song: Path, instrumental_song: Path, drop_bar: int = 16, vocal_gain_db: float = 0.0)
SongProfile(path, bpm: float, key_tonic: str, key_mode: str,
            beat_times: tuple[float, ...], downbeat_times: tuple[float, ...],
            duration_s: float, vocal_stem: Path | None, instrumental_stem: Path | None,
            vocal_onset_s: float | None)
ClipSpec(source: Path, role: str, arr_start_beats: float, arr_len_beats: float,
         content_start_s: float, content_end_s: float, pitch_semitones: int, gain_db: float)
MashupTimeline(tempo_bpm: float, clips: tuple[ClipSpec, ...])
```

- [ ] **Step 1: Write `compiler/models.py`**

```python
"""The compiler IR. One timeline drives both backends (render + .als), the
same single-IR convention as the GT schema upstream.

Warp convention (matches the .als two-marker linear clip): a clip occupies
[arr_start_beats, arr_start_beats + arr_len_beats] in arrangement beats and
plays [content_start_s, content_end_s] of its source file linearly across
that span. Stretch is implicit: arr_len_beats = content_len_s * source_bpm/60
means the source's beats map 1:1 onto arrangement beats."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MashupIntent:
    vocal_song: Path
    instrumental_song: Path
    drop_bar: int = 16          # instrumental bar (4/4) where the vocal enters
    vocal_gain_db: float = 0.0


@dataclass(frozen=True)
class SongProfile:
    path: Path
    bpm: float
    key_tonic: str              # 'C', 'C#', ... (Essentia edma profile)
    key_mode: str               # 'major' | 'minor'
    beat_times: tuple[float, ...]
    downbeat_times: tuple[float, ...]
    duration_s: float
    vocal_stem: Path | None
    instrumental_stem: Path | None
    vocal_onset_s: float | None  # first vocal activity in the vocal stem


@dataclass(frozen=True)
class ClipSpec:
    source: Path
    role: str                   # 'instrumental' | 'acappella'
    arr_start_beats: float
    arr_len_beats: float
    content_start_s: float
    content_end_s: float
    pitch_semitones: int
    gain_db: float

    @property
    def content_len_s(self) -> float:
        return self.content_end_s - self.content_start_s


@dataclass(frozen=True)
class MashupTimeline:
    tempo_bpm: float
    clips: tuple[ClipSpec, ...]

    @property
    def duration_beats(self) -> float:
        return max((c.arr_start_beats + c.arr_len_beats for c in self.clips), default=0.0)

    @property
    def duration_s(self) -> float:
        return self.duration_beats * 60.0 / self.tempo_bpm
```

- [ ] **Step 2: Write and run the test**

```python
# tests/test_models.py
from __future__ import annotations

from pathlib import Path

from compiler.models import ClipSpec, MashupTimeline


def test_timeline_durations() -> None:
    c = ClipSpec(Path("x.wav"), "instrumental", 0.0, 240.0, 10.0, 130.0, 0, 0.0)
    t = MashupTimeline(tempo_bpm=120.0, clips=(c,))
    assert c.content_len_s == 120.0
    assert t.duration_beats == 240.0
    assert t.duration_s == 120.0
```

Run: `venv/bin/python -m pytest tests/test_models.py -v` → PASS.

- [ ] **Step 3: Commit** — `git commit -am "feat: compiler IR (intent, profile, timeline)"` (add new files first).

---

### Task 5: Compatibility gate (`compiler/gate.py`)

Implements the spec's locked v1 rule: vocals accept |stretch ratio − 1| ≤ 6%; key compatible within ±1 semitone transposition (relative major/minor counted as equal).

**Files:**
- Create: `$MC/compiler/gate.py`
- Test: `$MC/tests/test_gate.py`

**Interfaces:**
- Consumes: `SongProfile` (only `.bpm`, `.key_tonic`, `.key_mode`), `engine.result.{Result, Ok, Err}`
- Produces: `check(vocal: SongProfile, instrumental: SongProfile) -> Result[GatePass, GateFailure]`; `GatePass(stretch_ratio: float, transpose_semitones: int)`; `GateFailure(kind: str, detail: str)`; `canonical_pitch_class(tonic: str, mode: str) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gate.py
from __future__ import annotations

from pathlib import Path

from compiler.gate import GateFailure, canonical_pitch_class, check
from compiler.models import SongProfile


def _profile(bpm: float, tonic: str, mode: str) -> SongProfile:
    return SongProfile(Path("x"), bpm, tonic, mode, (), (), 180.0, None, None, None)


def test_relative_minor_equals_major() -> None:
    assert canonical_pitch_class("A", "minor") == canonical_pitch_class("C", "major")


def test_pass_same_key_close_bpm() -> None:
    r = check(_profile(126.0, "A", "minor"), _profile(128.0, "C", "major"))
    assert r.is_ok()
    assert abs(r.value.stretch_ratio - 128.0 / 126.0) < 1e-9
    assert r.value.transpose_semitones == 0


def test_pass_one_semitone_up() -> None:
    r = check(_profile(128.0, "B", "major"), _profile(128.0, "C", "major"))
    assert r.is_ok() and r.value.transpose_semitones == 1


def test_fail_stretch() -> None:
    r = check(_profile(100.0, "C", "major"), _profile(128.0, "C", "major"))
    assert not r.is_ok() and r.error.kind == "stretch"


def test_fail_key_tritone() -> None:
    r = check(_profile(128.0, "C", "major"), _profile(128.0, "F#", "major"))
    assert not r.is_ok() and r.error.kind == "key"


def test_transpose_wraps_around_octave() -> None:
    r = check(_profile(128.0, "C", "major"), _profile(128.0, "B", "major"))
    assert r.is_ok() and r.value.transpose_semitones == -1
```

- [ ] **Step 2: Run to verify FAIL** — `venv/bin/python -m pytest tests/test_gate.py -v` → `ModuleNotFoundError: compiler.gate`.

- [ ] **Step 3: Write `compiler/gate.py`**

```python
"""Compatibility gate — the v1 rule from the DJ-agent spec: no warbly
renders. Out-of-tolerance pairs are rejected with a reason, not stretched."""
from __future__ import annotations

from dataclasses import dataclass

from compiler.models import SongProfile
from engine.result import Err, Ok, Result

MAX_VOCAL_STRETCH = 0.06  # |ratio - 1| tolerance for vocals
MAX_TRANSPOSE = 1         # semitones

_PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


@dataclass(frozen=True)
class GatePass:
    stretch_ratio: float        # instrumental_bpm / vocal_bpm
    transpose_semitones: int    # applied to the vocal


@dataclass(frozen=True)
class GateFailure:
    kind: str                   # 'stretch' | 'key'
    detail: str


def canonical_pitch_class(tonic: str, mode: str) -> int:
    """Pitch class of the relative-major tonic (Am == C)."""
    pc = _PITCH_CLASS[tonic]
    return pc if mode == "major" else (pc + 3) % 12


def check(vocal: SongProfile, instrumental: SongProfile) -> Result[GatePass, GateFailure]:
    ratio = instrumental.bpm / vocal.bpm
    if abs(ratio - 1.0) > MAX_VOCAL_STRETCH:
        return Err(GateFailure(
            "stretch",
            f"vocal {vocal.bpm:.1f} bpm vs instrumental {instrumental.bpm:.1f} bpm: "
            f"stretch ratio {ratio:.3f} outside ±{MAX_VOCAL_STRETCH:.0%}",
        ))
    delta = (canonical_pitch_class(instrumental.key_tonic, instrumental.key_mode)
             - canonical_pitch_class(vocal.key_tonic, vocal.key_mode)) % 12
    if delta > 6:
        delta -= 12
    if abs(delta) > MAX_TRANSPOSE:
        return Err(GateFailure(
            "key",
            f"{vocal.key_tonic} {vocal.key_mode} onto {instrumental.key_tonic} "
            f"{instrumental.key_mode} needs {delta:+d} semitones (max ±{MAX_TRANSPOSE})",
        ))
    return Ok(GatePass(stretch_ratio=ratio, transpose_semitones=delta))
```

- [ ] **Step 4: Run to verify PASS** — same command, 6 PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: compatibility gate (bpm ratio + key transposition)"`

---

### Task 6: Placement math (`compiler/placement.py`)

**Files:**
- Create: `$MC/compiler/placement.py`
- Test: `$MC/tests/test_placement.py`

**Interfaces:**
- Consumes: numpy only (pure functions over arrays/tuples)
- Produces: `first_active_time(samples: np.ndarray, sr: int, frame_s=0.05, threshold_db=-35.0) -> float`; `snap_to_grid(t: float, grid: tuple[float, ...]) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_placement.py
from __future__ import annotations

import numpy as np

from compiler.placement import first_active_time, snap_to_grid


def test_first_active_time_finds_late_onset() -> None:
    sr = 22050
    y = np.zeros(sr * 4, dtype=np.float32)
    t = np.arange(sr * 2) / sr
    y[sr * 2:] = 0.5 * np.sin(2 * np.pi * 440 * t)
    assert abs(first_active_time(y, sr) - 2.0) < 0.1


def test_first_active_time_all_silence() -> None:
    assert first_active_time(np.zeros(1000, dtype=np.float32), 22050) == 0.0


def test_snap_to_grid() -> None:
    grid = (0.0, 1.9, 3.8, 5.7)
    assert snap_to_grid(2.1, grid) == 1.9
    assert snap_to_grid(3.0, grid) == 3.8
    assert snap_to_grid(-1.0, grid) == 0.0
```

- [ ] **Step 2: Run to verify FAIL** — `ModuleNotFoundError: compiler.placement`.

- [ ] **Step 3: Write `compiler/placement.py`**

```python
"""Placement primitives: onset detection on a stem + grid snapping.

v1 limitation (accepted): snapping the vocal onset to the NEAREST downbeat
cuts pickups that start just before the bar. Fine for the demo; phrase-level
placement is a later lever."""
from __future__ import annotations

import numpy as np


def first_active_time(
    samples: np.ndarray, sr: int, frame_s: float = 0.05, threshold_db: float = -35.0
) -> float:
    """First time the frame RMS rises above threshold_db relative to peak RMS."""
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    n = max(1, int(sr * frame_s))
    n_frames = len(mono) // n
    if n_frames == 0:
        return 0.0
    rms = np.sqrt((mono[: n_frames * n].reshape(n_frames, n) ** 2).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return 0.0
    db = 20.0 * np.log10(rms / peak + 1e-12)
    active = np.flatnonzero(db > threshold_db)
    return float(active[0] * frame_s) if active.size else 0.0


def snap_to_grid(t: float, grid: tuple[float, ...]) -> float:
    if not grid:
        return t
    arr = np.asarray(grid)
    return float(arr[int(np.argmin(np.abs(arr - t)))])
```

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: placement primitives (onset + grid snap)"`

---

### Task 7: Timeline compilation (`compiler/compile.py`)

Pure arithmetic from two `SongProfile`s + intent to a `MashupTimeline`. Master tempo = instrumental BPM; instrumental's first downbeat lands on arrangement beat 0; vocal enters at `drop_bar * 4` beats, its onset snapped to its own downbeat grid; vocal trimmed so it never outlasts the instrumental.

**Files:**
- Create: `$MC/compiler/compile.py`
- Test: `$MC/tests/test_compile.py`

**Interfaces:**
- Consumes: `compiler.gate.check`, `compiler.placement.snap_to_grid`, `compiler.models.*`, `engine.result.*`
- Produces: `compile_mashup(intent: MashupIntent, vocal: SongProfile, instrumental: SongProfile) -> Result[MashupTimeline, GateFailure]` — clip[0] is always the instrumental, clip[1] the acappella.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_compile.py
from __future__ import annotations

from pathlib import Path

from compiler.compile import compile_mashup
from compiler.models import MashupIntent, SongProfile


def _instr() -> SongProfile:
    # 120 bpm, first downbeat at 0.5 s, 240.5 s long => 480 beats of content
    return SongProfile(Path("instr.mp3"), 120.0, "C", "major",
                       (), tuple(0.5 + i * 2.0 for i in range(120)),
                       240.5, None, Path("instr_stem.wav"), None)


def _vocal(bpm: float = 120.0) -> SongProfile:
    # onset at 10.3 s, downbeats every 2 s from 0.4 => snaps to 10.4
    return SongProfile(Path("voc.mp3"), bpm, "A", "minor",
                       (), tuple(0.4 + i * 2.0 for i in range(80)),
                       160.4, Path("voc_stem.wav"), None, 10.3)


def test_compile_basic_layout() -> None:
    r = compile_mashup(MashupIntent(Path("voc.mp3"), Path("instr.mp3"), drop_bar=16), _vocal(), _instr())
    assert r.is_ok()
    t = r.value
    assert t.tempo_bpm == 120.0
    instr, voc = t.clips
    assert instr.role == "instrumental" and voc.role == "acappella"
    assert instr.arr_start_beats == 0.0 and instr.content_start_s == 0.5
    assert instr.arr_len_beats == 480.0            # (240.5-0.5) * 120/60
    assert voc.arr_start_beats == 64.0             # bar 16 * 4
    assert voc.content_start_s == 10.4             # onset snapped to vocal grid
    assert voc.content_end_s == 160.4              # to end of vocal
    assert voc.arr_len_beats == 300.0              # 150 s * 120/60
    assert voc.source == Path("voc_stem.wav")      # the STEM, not the full song
    assert instr.source == Path("instr_stem.wav")


def test_vocal_trimmed_to_instrumental_end() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i"), drop_bar=100), _vocal(), _instr())
    assert r.is_ok()
    voc = r.value.clips[1]
    assert voc.arr_start_beats == 400.0
    assert voc.arr_start_beats + voc.arr_len_beats == 480.0   # capped
    assert abs(voc.content_len_s - 40.0) < 1e-9               # 80 beats at 120bpm


def test_drop_past_end_fails() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i"), drop_bar=121), _vocal(), _instr())
    assert not r.is_ok() and r.error.kind == "placement"


def test_gate_failure_propagates() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i")), _vocal(bpm=100.0), _instr())
    assert not r.is_ok() and r.error.kind == "stretch"
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Write `compiler/compile.py`**

```python
"""intent + profiles -> MashupTimeline. Pure; no I/O."""
from __future__ import annotations

from compiler.gate import GateFailure, check
from compiler.models import ClipSpec, MashupIntent, MashupTimeline, SongProfile
from compiler.placement import snap_to_grid
from engine.result import Err, Ok, Result

BEATS_PER_BAR = 4  # 4/4 EDM assumption, same as beat_this measure derivation


def compile_mashup(
    intent: MashupIntent, vocal: SongProfile, instrumental: SongProfile
) -> Result[MashupTimeline, GateFailure]:
    gate = check(vocal, instrumental)
    if not gate.is_ok():
        return gate

    tempo = instrumental.bpm
    i_start = instrumental.downbeat_times[0] if instrumental.downbeat_times else 0.0
    i_len_s = instrumental.duration_s - i_start
    instr_clip = ClipSpec(
        source=instrumental.instrumental_stem or instrumental.path,
        role="instrumental",
        arr_start_beats=0.0,
        arr_len_beats=i_len_s * instrumental.bpm / 60.0,
        content_start_s=i_start,
        content_end_s=instrumental.duration_s,
        pitch_semitones=0,
        gain_db=0.0,
    )

    drop_beats = float(intent.drop_bar * BEATS_PER_BAR)
    avail_beats = instr_clip.arr_len_beats - drop_beats
    if avail_beats <= 0:
        return Err(GateFailure(
            "placement",
            f"drop bar {intent.drop_bar} is past the instrumental's "
            f"{instr_clip.arr_len_beats / BEATS_PER_BAR:.0f} bars",
        ))

    anchor_s = snap_to_grid(vocal.vocal_onset_s or 0.0, vocal.downbeat_times)
    # vocal beats map 1:1 onto arrangement beats (that IS the linear stretch)
    vocal_beats = (vocal.duration_s - anchor_s) * vocal.bpm / 60.0
    vocal_beats = min(vocal_beats, avail_beats)
    vocal_clip = ClipSpec(
        source=vocal.vocal_stem or vocal.path,
        role="acappella",
        arr_start_beats=drop_beats,
        arr_len_beats=vocal_beats,
        content_start_s=anchor_s,
        content_end_s=anchor_s + vocal_beats * 60.0 / vocal.bpm,
        pitch_semitones=gate.value.transpose_semitones,
        gain_db=intent.vocal_gain_db,
    )
    return Ok(MashupTimeline(tempo_bpm=tempo, clips=(instr_clip, vocal_clip)))
```

- [ ] **Step 4: Run to verify PASS** (4 tests).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: timeline compilation (placement + trim + gate propagation)"`

---

### Task 8: Song analysis (`compiler/analyze.py`)

Orchestrates the copied adapters into a `SongProfile`, with a JSON sidecar cache (stems + analysis take minutes; the demo loop must not repeat them).

**Files:**
- Create: `$MC/compiler/analyze.py`
- Test: `$MC/tests/test_analyze.py` (cache logic offline; full path `@pytest.mark.integration`)

**Interfaces:**
- Consumes: `engine.beat_this_adapter.{load, predict}`, `engine.essentia_adapter.analyze`, `engine.roformer_chain_adapter.{load, separate}`, `engine.audio_io.load_mono`, `compiler.placement.first_active_time`
- Produces: `analyze_song(path: Path, need_vocal_stem: bool, need_instrumental_stem: bool) -> Result[SongProfile, str]`; `_profile_to_json(p: SongProfile) -> dict`; `_profile_from_json(d: dict) -> SongProfile`. Cache file: `<song>.mashup_profile.json` next to the source. Stems land in `<song dir>/.stems/<song stem>/{vocals,instrumental}.*`.

- [ ] **Step 1: Write the failing cache tests**

```python
# tests/test_analyze.py
from __future__ import annotations

from pathlib import Path

from compiler.analyze import _profile_from_json, _profile_to_json
from compiler.models import SongProfile


def test_profile_json_roundtrip(tmp_path: Path) -> None:
    p = SongProfile(tmp_path / "a.mp3", 128.0, "F#", "minor",
                    (0.1, 0.5), (0.1,), 200.0, tmp_path / "v.wav", None, 3.2)
    q = _profile_from_json(_profile_to_json(p))
    assert q == p


def test_profile_json_none_fields(tmp_path: Path) -> None:
    p = SongProfile(tmp_path / "a.mp3", 128.0, "C", "major", (), (), 10.0, None, None, None)
    assert _profile_from_json(_profile_to_json(p)) == p
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Write `compiler/analyze.py`**

```python
"""Song file -> SongProfile via the frozen engine adapters, with a JSON
sidecar cache. Essentia runs on the FULL song only, never on stems
(vocals-only audio breaks Essentia — upstream house rule)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from compiler.models import SongProfile
from compiler.placement import first_active_time
from engine.result import Err, Ok, Result

_CACHE_SUFFIX = ".mashup_profile.json"
_PROFILE_VERSION = 1


def _profile_to_json(p: SongProfile) -> dict:
    d = asdict(p)
    d["version"] = _PROFILE_VERSION
    for k in ("path", "vocal_stem", "instrumental_stem"):
        d[k] = str(d[k]) if d[k] is not None else None
    return d


def _profile_from_json(d: dict) -> SongProfile:
    return SongProfile(
        path=Path(d["path"]),
        bpm=float(d["bpm"]),
        key_tonic=d["key_tonic"],
        key_mode=d["key_mode"],
        beat_times=tuple(d["beat_times"]),
        downbeat_times=tuple(d["downbeat_times"]),
        duration_s=float(d["duration_s"]),
        vocal_stem=Path(d["vocal_stem"]) if d["vocal_stem"] else None,
        instrumental_stem=Path(d["instrumental_stem"]) if d["instrumental_stem"] else None,
        vocal_onset_s=d["vocal_onset_s"],
    )


def _cache_path(song: Path) -> Path:
    return song.with_name(song.name + _CACHE_SUFFIX)


def _load_cache(song: Path, need_vocal: bool, need_instr: bool) -> SongProfile | None:
    cp = _cache_path(song)
    if not cp.is_file():
        return None
    d = json.loads(cp.read_text())
    if d.get("version") != _PROFILE_VERSION:
        return None
    p = _profile_from_json(d)
    if need_vocal and (p.vocal_stem is None or not p.vocal_stem.is_file()):
        return None
    if need_instr and (p.instrumental_stem is None or not p.instrumental_stem.is_file()):
        return None
    return p


def analyze_song(
    path: Path, need_vocal_stem: bool, need_instrumental_stem: bool
) -> Result[SongProfile, str]:
    cached = _load_cache(path, need_vocal_stem, need_instrumental_stem)
    if cached is not None:
        return Ok(cached)

    from engine import audio_io, beat_this_adapter, essentia_adapter

    bt = beat_this_adapter.load(device="auto")
    if not bt.is_ok():
        return Err(f"beat_this load: {bt.error}")
    grid = beat_this_adapter.predict(bt.value, path)
    if not grid.is_ok():
        return Err(f"beat grid: {grid.error}")
    beats, downbeats = grid.value

    ess = essentia_adapter.analyze(path, track_audio_id=0)
    if not ess.is_ok():
        return Err(f"essentia: {ess.error}")
    f = ess.value

    wav = audio_io.load_mono(path)
    if not wav.is_ok():
        return Err(f"audio load: {wav.error}")
    duration_s = len(wav.value.samples) / wav.value.sample_rate

    vocal_stem: Path | None = None
    instrumental_stem: Path | None = None
    vocal_onset_s: float | None = None
    if need_vocal_stem or need_instrumental_stem:
        from engine import roformer_chain_adapter

        h = roformer_chain_adapter.load(device="auto")
        if not h.is_ok():
            return Err(f"separator load: {h.error}")
        stems = roformer_chain_adapter.separate(h.value, path)
        if not stems.is_ok():
            return Err(f"separation: {stems.error}")
        by_name = {s.stem_name: Path(s.path) for s in stems.value.stems}
        vocal_stem = by_name.get("vocals")
        instrumental_stem = by_name.get("instrumental")
        if need_vocal_stem and vocal_stem is None:
            return Err("separator produced no vocals stem")
        if need_instrumental_stem and instrumental_stem is None:
            return Err("separator produced no instrumental stem")
        if vocal_stem is not None:
            vw = audio_io.load_mono(vocal_stem)
            if vw.is_ok():
                vocal_onset_s = first_active_time(vw.value.samples, vw.value.sample_rate)

    profile = SongProfile(
        path=path, bpm=float(f.bpm), key_tonic=f.key_tonic, key_mode=f.key_mode,
        beat_times=beats, downbeat_times=downbeats, duration_s=duration_s,
        vocal_stem=vocal_stem, instrumental_stem=instrumental_stem,
        vocal_onset_s=vocal_onset_s,
    )
    _cache_path(path).write_text(json.dumps(_profile_to_json(profile), indent=1))
    return Ok(profile)
```

- [ ] **Step 4: Run offline tests** — `venv/bin/python -m pytest tests/test_analyze.py -v` → 2 PASS.

- [ ] **Step 5: Add the integration test (appended to `tests/test_analyze.py`)**

```python
import pytest


@pytest.mark.integration
def test_analyze_real_song() -> None:
    """Needs: a real song at ~/Desktop/mashup_demo/instr.mp3, MSST + essentia
    installs reachable via engine.paths defaults."""
    from compiler.analyze import analyze_song

    song = Path.home() / "Desktop" / "mashup_demo" / "instr.mp3"
    if not song.is_file():
        pytest.skip("no demo song staged")
    r = analyze_song(song, need_vocal_stem=False, need_instrumental_stem=True)
    assert r.is_ok(), r.error
    p = r.value
    assert 60 < p.bpm < 200 and len(p.downbeat_times) > 10
    assert p.instrumental_stem and p.instrumental_stem.is_file()
```

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: song analysis orchestration with sidecar cache"`

---

### Task 9: Audio renderer (`compiler/render.py`)

**Files:**
- Create: `$MC/compiler/render.py`
- Test: `$MC/tests/test_render.py` (synthetic sine fixtures; skipped if `rubberband` CLI missing)

**Interfaces:**
- Consumes: `MashupTimeline`, `ClipSpec`; `soundfile`, `librosa`, `pyrubberband`
- Produces: `render(timeline: MashupTimeline, out_path: Path, sr: int = 44100) -> Path` — writes a stereo wav, peak-normalized to −1 dBFS.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render.py
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from compiler.models import ClipSpec, MashupTimeline
from compiler.render import render

needs_rubberband = pytest.mark.skipif(
    shutil.which("rubberband") is None, reason="rubberband CLI not installed"
)


def _tone(path: Path, freq: float, dur_s: float, sr: int = 44100) -> Path:
    t = np.arange(int(dur_s * sr)) / sr
    sf.write(path, 0.4 * np.sin(2 * np.pi * freq * t), sr)
    return path


@needs_rubberband
def test_render_places_and_stretches(tmp_path: Path) -> None:
    instr = _tone(tmp_path / "instr.wav", 220.0, 8.0)
    voc = _tone(tmp_path / "voc.wav", 440.0, 4.0)
    # tempo 120: 1 beat = 0.5 s. Instrumental spans 16 beats (8 s at ratio 1).
    # Vocal content is 4 s but occupies 9 beats = 4.5 s -> stretch ratio 1.125.
    t = MashupTimeline(tempo_bpm=120.0, clips=(
        ClipSpec(instr, "instrumental", 0.0, 16.0, 0.0, 8.0, 0, 0.0),
        ClipSpec(voc, "acappella", 8.0, 9.0, 0.0, 4.0, 1, 0.0),
    ))
    out = render(t, tmp_path / "mix.wav")
    y, sr = sf.read(out)
    assert y.ndim == 2 and y.shape[1] == 2
    assert abs(len(y) / sr - 8.5) < 0.1          # 17 beats at 120 bpm
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-3   # -1 dBFS ceiling
    # energy present before AND after the vocal entry at 4.0 s
    assert np.abs(y[: int(3.5 * sr)]).max() > 0.01
    assert np.abs(y[int(4.2 * sr): int(8.0 * sr)]).max() > 0.01
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Write `compiler/render.py`**

```python
"""MashupTimeline -> stereo wav. Rubberband for stretch/pitch (what every
generator in the prior-art scan uses); the .als backend shares the same IR
so the render and the session always agree."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pyrubberband
import soundfile as sf

from compiler.models import ClipSpec, MashupTimeline

_PEAK_DBFS = -1.0


def _load_segment(clip: ClipSpec, sr: int) -> np.ndarray:
    """(n, 2) float64 of the clip's content span, resampled to sr."""
    info = sf.info(str(clip.source))
    start = int(clip.content_start_s * info.samplerate)
    stop = int(clip.content_end_s * info.samplerate)
    y, file_sr = sf.read(str(clip.source), start=start, stop=stop, always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    if file_sr != sr:
        y = librosa.resample(y.T, orig_sr=file_sr, target_sr=sr).T
    return y


def _process(clip: ClipSpec, y: np.ndarray, sr: int, spb: float) -> np.ndarray:
    target_s = clip.arr_len_beats * spb
    actual_s = len(y) / sr
    rate = actual_s / target_s          # >1 speeds up
    if abs(rate - 1.0) > 1e-4:
        y = pyrubberband.time_stretch(y, sr, rate)
    if clip.pitch_semitones != 0:
        y = pyrubberband.pitch_shift(y, sr, clip.pitch_semitones)
    if clip.gain_db != 0.0:
        y = y * (10.0 ** (clip.gain_db / 20.0))
    return y


def render(timeline: MashupTimeline, out_path: Path, sr: int = 44100) -> Path:
    spb = 60.0 / timeline.tempo_bpm
    total = int(np.ceil(timeline.duration_s * sr)) + sr
    mix = np.zeros((total, 2), dtype=np.float64)
    for clip in timeline.clips:
        y = _process(clip, _load_segment(clip, sr), sr, spb)
        at = int(clip.arr_start_beats * spb * sr)
        end = min(at + len(y), total)
        mix[at:end] += y[: end - at]
    # trim trailing silence pad, then peak-normalize to -1 dBFS
    last = int(np.ceil(timeline.duration_s * sr))
    mix = mix[:last]
    peak = np.abs(mix).max()
    if peak > 0:
        mix *= (10.0 ** (_PEAK_DBFS / 20.0)) / peak
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mix, sr)
    return out_path
```

- [ ] **Step 4: Run to verify PASS** (or SKIP if rubberband missing — then `brew install rubberband` and re-run; the test must PASS on this machine).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: audio renderer (rubberband stretch/pitch, peak-normalized)"`

---

### Task 10: .als emission (`compiler/emit_als.py`)

**Files:**
- Create: `$MC/compiler/emit_als.py`
- Test: `$MC/tests/test_emit_als.py`

**Interfaces:**
- Consumes: `engine.als.{load_als_xml, save_als_xml, parse_layer_clips, write_tempo_envelope, write_locators}`, `engine.als_seed.*`, `MashupTimeline`
- Produces: `emit_als(timeline: MashupTimeline, out_path: Path, template: Path = DEFAULT_TEMPLATE) -> Path` — self-validates by re-parsing its own output (same pattern as the upstream seeder). `DEFAULT_TEMPLATE = engine/als_assets/seed_template.als`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emit_als.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from compiler.emit_als import emit_als
from compiler.models import ClipSpec, MashupTimeline
from engine.als import load_als_xml, parse_layer_clips


def _tone(path: Path, dur_s: float, sr: int = 44100) -> Path:
    t = np.arange(int(dur_s * sr)) / sr
    sf.write(path, 0.4 * np.sin(2 * np.pi * 220 * t), sr)
    return path


def test_emit_two_clips_roundtrip(tmp_path: Path) -> None:
    instr = _tone(tmp_path / "instr.wav", 8.0)
    voc = _tone(tmp_path / "voc.wav", 4.0)
    t = MashupTimeline(tempo_bpm=128.0, clips=(
        ClipSpec(instr, "instrumental", 0.0, 16.0, 0.5, 8.0, 0, 0.0),
        ClipSpec(voc, "acappella", 8.0, 8.0, 1.0, 4.0, -1, 0.0),
    ))
    out = emit_als(t, tmp_path / "demo.als")
    clips = sorted(parse_layer_clips(load_als_xml(out)), key=lambda c: c.arr_start)
    assert len(clips) == 2
    assert clips[0].arr_start == 0.0 and clips[0].arr_end == 16.0
    assert abs(clips[0].ref_start_s() - 0.5) < 1e-6
    assert clips[1].arr_start == 8.0
    assert clips[1].pitch_coarse == -1
    assert clips[1].path.endswith("voc.wav")
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Write `compiler/emit_als.py`**

```python
"""MashupTimeline -> .als. Master tempo = timeline tempo (constant), every
clip warped with exactly two markers (the linear-stretch convention shared
with the render backend). Self-validates by re-parsing its own output."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

from compiler.models import MashupTimeline
from engine.als import load_als_xml, parse_layer_clips, save_als_xml, write_tempo_envelope
from engine.als_seed import (
    build_track,
    doc_max_id,
    ffprobe_audio,
    find_template_track,
    renumber_pointee_ids,
)

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[1] / "engine" / "als_assets" / "seed_template.als"
_COLOR = {"instrumental": 9, "acappella": 5}


def emit_als(
    timeline: MashupTimeline, out_path: Path, template: Path = DEFAULT_TEMPLATE
) -> Path:
    root = load_als_xml(template)
    template_track = copy.deepcopy(find_template_track(root))
    tracks_node = root.find(".//LiveSet/Tracks")
    for t in list(tracks_node):
        if t.tag in ("AudioTrack", "GroupTrack", "MidiTrack"):
            tracks_node.remove(t)
    write_tempo_envelope(root, [(0.0, timeline.tempo_bpm)])

    alloc = itertools.count(doc_max_id(root) + len(timeline.clips) * 2 + 2000)
    for i, clip in enumerate(timeline.clips):
        dur_s, sr = ffprobe_audio(clip.source)
        track = build_track(
            template_track,
            track_id=1000 + i,
            track_name=f"{i + 1}-{clip.role}",
            name=clip.role,
            color=_COLOR.get(clip.role, 2),
            arr_start=clip.arr_start_beats,
            arr_end=clip.arr_start_beats + clip.arr_len_beats,
            file_path=clip.source,
            file_dur_s=dur_s,
            sample_rate=sr,
            ref_start_s=clip.content_start_s,
            ref_end_s=clip.content_end_s,
            pitch_coarse=clip.pitch_semitones,
        )
        renumber_pointee_ids(track, alloc)
        tracks_node.insert(i, track)
    for npi in root.findall(".//NextPointeeId"):
        npi.set("Value", str(next(alloc)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_als_xml(root, out_path)

    reparsed = parse_layer_clips(load_als_xml(out_path))
    if len(reparsed) != len(timeline.clips):
        raise RuntimeError(
            f"als self-validation failed: {len(reparsed)} clips parsed, "
            f"{len(timeline.clips)} placed"
        )
    return out_path
```

- [ ] **Step 4: Run to verify PASS.** If `parse_layer_clips` drops a track, check the name-prefix skip rules in `engine/als/read.py` and adjust `track_name` (never name a track `1-mix`).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: als emission from timeline (self-validating roundtrip)"`

---

### Task 11: CLI + end-to-end demo run

**Files:**
- Create: `$MC/compiler/main.py`
- Modify: `$MC/README.md` (demo walkthrough)

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m compiler.main VOCAL INSTR [--drop-bar N] [--vocal-gain-db G] [--out out/NAME]` → `NAME.wav` + `NAME.als`.

- [ ] **Step 1: Write `compiler/main.py`**

```python
"""CLI edge: fail-fast, human-readable. All library errors surface here."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compiler.analyze import analyze_song
from compiler.compile import compile_mashup
from compiler.emit_als import emit_als
from compiler.models import MashupIntent
from compiler.render import render


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="vocals of SONG_A over instrumental of SONG_B")
    p.add_argument("vocal_song", type=Path)
    p.add_argument("instrumental_song", type=Path)
    p.add_argument("--drop-bar", type=int, default=16)
    p.add_argument("--vocal-gain-db", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=Path("out/mashup"))
    args = p.parse_args(argv)

    for f in (args.vocal_song, args.instrumental_song):
        if not f.is_file():
            sys.exit(f"no such file: {f}")

    print(f"analyzing {args.vocal_song.name} (vocals)…")
    vocal = analyze_song(args.vocal_song, need_vocal_stem=True, need_instrumental_stem=False)
    if not vocal.is_ok():
        sys.exit(f"analysis failed: {vocal.error}")
    print(f"analyzing {args.instrumental_song.name} (instrumental)…")
    instr = analyze_song(args.instrumental_song, need_vocal_stem=False, need_instrumental_stem=True)
    if not instr.is_ok():
        sys.exit(f"analysis failed: {instr.error}")

    v, i = vocal.value, instr.value
    print(f"  vocal: {v.bpm:.1f} bpm, {v.key_tonic} {v.key_mode}")
    print(f"  instr: {i.bpm:.1f} bpm, {i.key_tonic} {i.key_mode}")

    timeline = compile_mashup(
        MashupIntent(args.vocal_song, args.instrumental_song, args.drop_bar, args.vocal_gain_db),
        v, i,
    )
    if not timeline.is_ok():
        sys.exit(f"incompatible pair ({timeline.error.kind}): {timeline.error.detail}")

    wav = render(timeline.value, args.out.with_suffix(".wav"))
    als = emit_als(timeline.value, args.out.with_suffix(".als"))
    print(f"\nwrote {wav}\nwrote {als}\n\nopen the .als in Ableton; play the .wav.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Full offline test suite green**

Run: `venv/bin/python -m pytest tests/ -m "not integration" -v`
Expected: all PASS.

- [ ] **Step 3: End-to-end demo run (manual, this machine)**

Stage two real, key/BPM-compatible songs the user owns at `~/Desktop/mashup_demo/{vocal.mp3,instr.mp3}` (ask John to pick the pair — ideally one of the five friend ideas). Then:

```bash
venv/bin/python -m compiler.main ~/Desktop/mashup_demo/vocal.mp3 \
    ~/Desktop/mashup_demo/instr.mp3 --drop-bar 16 --out out/demo
```

Expected: prints both profiles, writes `out/demo.wav` + `out/demo.als`. First run takes minutes (separation); reruns are seconds (sidecar cache). If the gate rejects the pair, it must print *why* (bpm ratio or key delta) — that's correct behavior, pick a compatible pair or try `--drop-bar`.

- [ ] **Step 4: Verification (human ears + Ableton)**

1. Play `out/demo.wav` — vocal enters on the bar, in key, no warble. **This is gate #1; if it sounds dead, that's a finding, not a failure of the plan.**
2. Open `out/demo.als` in Ableton Live — must open with **no "fix file" dialog** (the crash trio worked), show 2 tracks, tempo = instrumental BPM, vocal clip transposed.

- [ ] **Step 5: Update README with the demo walkthrough + limitations** (mono→stereo handled; pickups cut by nearest-downbeat snap; constant-tempo assumption; per-clip gain not yet in .als) and commit:

```bash
git add -A && git commit -m "feat: CLI + end-to-end demo (render + als from one timeline)"
```

---

### Task 12: Interpreter rigor — crash-invariant validator + property-based roundtrip

The .als is the product artifact: if Live shows a "fix file" dialog, the user is gone. Upstream, crash-safety lives in a convention (always call the `strip_automation`/`renumber_pointee_ids`/`NextPointeeId` trio). This task turns the convention into a checked law: a mechanical validator for the known Live-crash invariants, wired into every emission, plus a Hypothesis property test that fuzzes timelines through emit→parse.

**Files:**
- Create: `$MC/engine/als_lint.py`, `$MC/tests/test_als_lint.py`, `$MC/tests/test_emit_property.py`
- Modify: `$MC/compiler/emit_als.py` (call the validator before returning), `$MC/requirements.txt` (add `hypothesis>=6.100`)

**Interfaces:**
- Consumes: lxml tree (post-emission `root`), `emit_als`, `parse_layer_clips`
- Produces: `validate_live_invariants(root: etree._Element) -> list[str]` — empty list = safe; each string one violated invariant.

- [ ] **Step 1: Write the failing lint tests**

```python
# tests/test_als_lint.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from lxml import etree

from compiler.emit_als import emit_als
from compiler.models import ClipSpec, MashupTimeline
from engine.als import load_als_xml
from engine.als_lint import validate_live_invariants


def _emitted_root(tmp_path: Path) -> etree._Element:
    tone = tmp_path / "t.wav"
    sf.write(tone, 0.3 * np.sin(np.arange(44100 * 4) / 44100 * 2 * np.pi * 220), 44100)
    t = MashupTimeline(tempo_bpm=124.0, clips=(
        ClipSpec(tone, "instrumental", 0.0, 8.0, 0.0, 4.0, 0, 0.0),
        ClipSpec(tone, "acappella", 4.0, 4.0, 0.0, 2.0, 1, 0.0),
    ))
    return load_als_xml(emit_als(t, tmp_path / "lint.als"))


def test_emitted_als_passes_lint(tmp_path: Path) -> None:
    assert validate_live_invariants(_emitted_root(tmp_path)) == []


def test_lint_catches_duplicate_pointee(tmp_path: Path) -> None:
    root = _emitted_root(tmp_path)
    pointees = [el for el in root.iter() if el.tag == "Pointee" or el.tag.endswith("Target")]
    assert len(pointees) >= 2
    pointees[1].set("Id", pointees[0].get("Id"))          # forge the crash condition
    errs = validate_live_invariants(root)
    assert any("duplicate" in e for e in errs)


def test_lint_catches_stale_next_pointee_id(tmp_path: Path) -> None:
    root = _emitted_root(tmp_path)
    for npi in root.findall(".//NextPointeeId"):
        npi.set("Value", "1")
    errs = validate_live_invariants(root)
    assert any("NextPointeeId" in e for e in errs)
```

- [ ] **Step 2: Run to verify FAIL** — `ModuleNotFoundError: engine.als_lint`.

- [ ] **Step 3: Write `engine/als_lint.py`**

```python
"""Mechanical checks for the known Ableton-crash invariants (learned the hard
way upstream — see als_seed.py docstring). An emitted document violating any
of these makes Live offer to "fix" the file and then crash during migration."""
from __future__ import annotations

from collections import Counter

from lxml import etree


def _pointee_ids(root: etree._Element) -> list[int]:
    out: list[int] = []
    for el in root.iter():
        v = el.get("Id")
        if v is None or not v.lstrip("-").isdigit():
            continue
        if el.tag == "Pointee" or el.tag.endswith("Target"):
            out.append(int(v))
    return out


def validate_live_invariants(root: etree._Element) -> list[str]:
    errs: list[str] = []
    ids = _pointee_ids(root)
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        errs.append(f"duplicate pointee ids (Live crash): {dupes[:8]}")
    top = max(ids, default=0)
    for npi in root.findall(".//NextPointeeId"):
        v = npi.get("Value")
        if v is None or not v.isdigit() or int(v) <= top:
            errs.append(f"NextPointeeId {v!r} not above max pointee id {top}")
    for clip in root.iter("AudioClip"):
        warped = clip.find("IsWarped")
        if warped is None or warped.get("Value") != "true":
            continue
        beats = [float(w.get("BeatTime")) for w in clip.findall(".//WarpMarker")]
        if len(beats) < 2:
            errs.append("warped clip with <2 warp markers")
        elif any(b2 <= b1 for b1, b2 in zip(beats, beats[1:])):
            errs.append(f"warp marker beats not strictly increasing: {beats}")
    return errs
```

- [ ] **Step 4: Wire into `compiler/emit_als.py`** — after the existing clip-count self-validation, add:

```python
    from engine.als_lint import validate_live_invariants

    problems = validate_live_invariants(load_als_xml(out_path))
    if problems:
        raise RuntimeError("als crash-invariant lint failed: " + "; ".join(problems))
```

(move the import to the top of the file with the others).

- [ ] **Step 5: Run lint tests to verify PASS** — `venv/bin/python -m pytest tests/test_als_lint.py -v` → 3 PASS.

- [ ] **Step 6: Add the Hypothesis property test**

```python
# tests/test_emit_property.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from compiler.emit_als import emit_als
from compiler.models import ClipSpec, MashupTimeline
from engine.als import load_als_xml, parse_layer_clips

_TONE: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _tone(tmp_path_factory: pytest.TempPathFactory) -> None:
    global _TONE
    _TONE = tmp_path_factory.mktemp("prop") / "tone.wav"
    sf.write(_TONE, 0.3 * np.sin(np.arange(44100 * 10) / 44100 * 2 * np.pi * 220), 44100)


clip_st = st.builds(
    lambda arr, ln, cs, cl, pitch: ClipSpec(
        _TONE, "acappella", arr, ln, cs, cs + cl, pitch, 0.0
    ),
    arr=st.floats(0.0, 512.0),
    ln=st.floats(1.0, 256.0),
    cs=st.floats(0.0, 5.0),
    cl=st.floats(0.5, 4.9),
    pitch=st.integers(-1, 1),
)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(tempo=st.floats(70.0, 180.0), clips=st.lists(clip_st, min_size=1, max_size=4))
def test_emit_parse_roundtrip(tmp_path: Path, tempo: float, clips: list[ClipSpec]) -> None:
    t = MashupTimeline(tempo_bpm=tempo, clips=tuple(clips))
    out = emit_als(t, tmp_path / "prop.als")     # raises if lint fails
    parsed = sorted(parse_layer_clips(load_als_xml(out)), key=lambda c: c.arr_start)
    original = sorted(t.clips, key=lambda c: c.arr_start_beats)
    assert len(parsed) == len(original)
    for p, o in zip(parsed, original):
        assert abs(p.arr_start - o.arr_start_beats) < 1e-4
        assert abs(p.ref_start_s() - o.content_start_s) < 1e-4
        assert p.pitch_coarse == o.pitch_semitones
```

Add `hypothesis>=6.100` to `requirements.txt` and `venv/bin/pip install hypothesis`.

- [ ] **Step 7: Run the full offline suite** — `venv/bin/python -m pytest tests/ -m "not integration"` → all PASS. Property shrinking failures here are real emitter bugs — fix the emitter, never loosen the tolerance.

- [ ] **Step 8: Commit** — `git add -A && git commit -m "feat: als crash-invariant lint + hypothesis emit/parse roundtrip law"`

---

## Deferred (explicitly NOT in this plan)

- NL intent parsing (next plan — the "compiler front-end"; this plan builds the back-end).
- Phrase-aware placement, section selection (chorus-picking), gain curves from `warp_prior.json`.
- Per-clip gain in the .als (render-only for now).
- Micro-pitch detune correction (estimator not yet implemented upstream — `pitch_fine` stays 0).
- Corpus lookup / catalog anything. User files only.

## Self-review notes

- Spec coverage: locked decisions 1–3 of the DJ-agent spec are implemented (compiler-not-collaborator = deterministic CLI; single mashup; linear-warp + gate). The spec's "resolution from corpus" is intentionally out (door-one product: user brings files).
- Type consistency: `ClipSpec` beats↔seconds convention is stated once in `models.py` and used identically by `compile`, `render` (`rate = actual_s / target_s`), and `emit_als` (2 warp markers `(0, content_start)…(arr_len, content_end)`).
- Known risk: exact upstream file names for the Roformer config (`roformer_config.py` vs inline) and the als package's internal import graph — Task 2 Steps 1–2 include the greps that catch both.

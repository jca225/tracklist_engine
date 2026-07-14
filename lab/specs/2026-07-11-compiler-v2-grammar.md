# Compiler v2 (Grammar Rulebook) + Refinement Verbs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the mashup compiler obey the empirically mined DJ grammar (`bb_mashup_grammar_v1`) — 16-bar first-chorus hook, pickup-led entry, LUFS-matched levels, tight 40-bar structure — and add refinement verbs to the seam web app so every "try again" becomes logged preference data (P0-3 of the decision-model plan).

**Architecture:** Extend `SongProfile` (v2) with a vocal-energy envelope + per-stem LUFS computed at analyze time (with a stem-reuse guard so the version bump does NOT re-run separation). A new pure module `compiler/hook.py` ranks 16-bar hook windows from the envelope. `compile_mashup` consumes the top-ranked hook (or `hook_rank` for re-rolls), leads it with a pickup, auto-gains from the LUFS delta, and emits a 40-bar intro/hook/outro structure by default. The server gains `/api/refine` with four verbs, each deriving a child mash from its parent (the DPO-pair log IS the `mashes` table).

**Tech Stack:** Existing repo stack (numpy, soundfile, pyrubberband, FastAPI). New test dep: `httpx` (starlette TestClient).

## Global Constraints

- Repo: `/Users/johnnycabrahams/Desktop/mashup_compiler` ($MC). Every task commits there. Work from $MC.
- `engine/` is FROZEN (Task-2 contract of the bootstrap plan) — never edit; consume `engine.loudness.integrated_lufs(samples, sr) -> Result[float, LoudnessError]` as-is.
- Offline suite must stay green after every task: `venv/bin/python -m pytest tests/ -m "not integration"` (currently 27 passed, 1 deselected; count grows per task).
- A global PostToolUse hook ruff-formats .py writes — acceptable; semantics must be unchanged.
- Grammar constants come verbatim from `bb_mashup_grammar_v1` (tracklist_engine `lab/corpus_empirics/findings.md`): HOOK_BARS=16, hook search restricted to first 60% of song, OUTRO_BARS=8, default drop bar 16, LUFS auto-gain clamp ±9 dB, pickup lead ≤ 1 bar.
- `_PROFILE_VERSION` bumps 1→2; old sidecar caches invalidate by design. The stem-reuse guard (Task 1) MUST land in the same commit so re-analysis skips separation.
- The demo pair with warm stems exists at `~/Desktop/mashup_demo/{vocal,instr}.wav` (stems under `~/Desktop/mashup_demo/.stems/<name>/`).

---

### Task 1: Profile v2 — energy envelope, stem LUFS, stem-reuse guard

**Files:**
- Modify: `$MC/compiler/models.py` (SongProfile gains 3 optional fields)
- Modify: `$MC/compiler/analyze.py` (version bump, envelope+LUFS computation, stem-reuse guard, json round-trip of new fields)
- Test: `$MC/tests/test_analyze_v2.py`

**Interfaces:**
- Produces: `SongProfile.vocal_energy: tuple[float, ...] | None` (RMS envelope of the vocal stem at `ENERGY_HZ = 2.0` frames/sec, normalized to peak=1.0); `SongProfile.vocal_lufs: float | None` (integrated LUFS over ACTIVE vocal frames only); `SongProfile.instrumental_lufs: float | None`; `compiler.analyze.ENERGY_HZ = 2.0`; `_existing_stems(song: Path) -> dict[str, Path]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analyze_v2.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from compiler.analyze import (
    ENERGY_HZ,
    _energy_envelope,
    _existing_stems,
    _lufs_active,
    _profile_from_json,
    _profile_to_json,
)
from compiler.models import SongProfile


def test_energy_envelope_peaks_where_loud() -> None:
    sr = 22050
    y = np.zeros(sr * 4, dtype=np.float64)
    y[sr * 2 : sr * 3] = 0.8 * np.sin(np.arange(sr) / sr * 2 * np.pi * 440)
    env = _energy_envelope(y, sr)
    assert len(env) == int(4 * ENERGY_HZ)
    assert max(env) == 1.0
    assert env[int(2.5 * ENERGY_HZ)] > 0.9
    assert env[int(0.5 * ENERGY_HZ)] < 0.05


def test_lufs_active_ignores_silence() -> None:
    sr = 44100
    tone = 0.3 * np.sin(np.arange(sr * 2) / sr * 2 * np.pi * 220)
    padded = np.concatenate([np.zeros(sr * 8), tone, np.zeros(sr * 8)])
    full = _lufs_active(tone, sr)
    pad = _lufs_active(padded, sr)
    assert full is not None and pad is not None
    assert abs(full - pad) < 1.5  # silence-gated: padding barely moves it


def test_profile_v2_json_roundtrip(tmp_path: Path) -> None:
    p = SongProfile(
        tmp_path / "a.mp3", 128.0, "C", "minor", (0.1,), (0.1,), 200.0,
        tmp_path / "v.wav", None, 3.2,
        vocal_energy=(0.0, 0.5, 1.0), vocal_lufs=-18.5, instrumental_lufs=-12.0,
    )
    assert _profile_from_json(_profile_to_json(p)) == p


def test_profile_v1_fields_default_none(tmp_path: Path) -> None:
    p = SongProfile(tmp_path / "a.mp3", 128.0, "C", "major", (), (), 10.0, None, None, None)
    assert p.vocal_energy is None and p.vocal_lufs is None and p.instrumental_lufs is None


def test_existing_stems_found(tmp_path: Path) -> None:
    song = tmp_path / "song.wav"
    d = tmp_path / ".stems" / "song"
    d.mkdir(parents=True)
    (d / "vocals.flac").write_bytes(b"x")
    (d / "instrumental.wav").write_bytes(b"x")
    found = _existing_stems(song)
    assert set(found) == {"vocals", "instrumental"}
```

- [ ] **Step 2: Run to verify FAIL** — `venv/bin/python -m pytest tests/test_analyze_v2.py -v` → ImportError (`ENERGY_HZ`).

- [ ] **Step 3: Extend `compiler/models.py`** — append to `SongProfile` (after `vocal_onset_s`), keeping every existing field untouched:

```python
    # v2 (grammar rulebook): all optional so v1 constructors keep working.
    vocal_energy: tuple[float, ...] | None = None   # vocal-stem RMS @ ENERGY_HZ, peak-normalized
    vocal_lufs: float | None = None                 # integrated LUFS over ACTIVE vocal frames
    instrumental_lufs: float | None = None
```

- [ ] **Step 4: Extend `compiler/analyze.py`**

At top: `import numpy as np`, `ENERGY_HZ = 2.0`, and `_PROFILE_VERSION = 2`.

New helpers:

```python
def _energy_envelope(samples: np.ndarray, sr: int, hz: float = ENERGY_HZ) -> tuple[float, ...]:
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    n = max(1, int(sr / hz))
    n_frames = len(mono) // n
    if n_frames == 0:
        return ()
    rms = np.sqrt((mono[: n_frames * n].reshape(n_frames, n) ** 2).mean(axis=1))
    peak = float(rms.max())
    return tuple(float(v) for v in (rms / peak if peak > 0 else rms))


def _lufs_active(samples: np.ndarray, sr: int, gate_db: float = -40.0) -> float | None:
    """Integrated LUFS over frames above gate_db rel peak — silence-gated so a
    sparse vocal stem isn't dragged down by its rests."""
    from engine.loudness import integrated_lufs

    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    n = max(1, int(sr / ENERGY_HZ))
    n_frames = len(mono) // n
    if n_frames == 0:
        return None
    frames = mono[: n_frames * n].reshape(n_frames, n)
    rms = np.sqrt((frames**2).mean(axis=1))
    peak = rms.max()
    if peak <= 0:
        return None
    active = frames[20.0 * np.log10(rms / peak + 1e-12) > gate_db]
    if len(active) == 0:
        return None
    r = integrated_lufs(active.reshape(-1), sr)
    return float(r.value) if r.is_ok() else None


def _existing_stems(song: Path) -> dict[str, Path]:
    d = _stems_dir(song)
    if not d.is_dir():
        return {}
    return {p.stem: p for p in d.iterdir() if p.stem in ("vocals", "instrumental")}
```

In `analyze_song`, replace the separation block's entry with the reuse guard, and compute the new fields:

```python
    vocal_stem: Path | None = None
    instrumental_stem: Path | None = None
    vocal_onset_s: float | None = None
    vocal_energy: tuple[float, ...] | None = None
    vocal_lufs: float | None = None
    instrumental_lufs: float | None = None
    if need_vocal_stem or need_instrumental_stem:
        by_name = _existing_stems(path)
        need_sep = (need_vocal_stem and "vocals" not in by_name) or (
            need_instrumental_stem and "instrumental" not in by_name
        )
        if need_sep:
            # ... existing separator load / separate / hoist block, unchanged,
            # ending with by_name = {hoisted stems} ...
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
                vocal_energy = _energy_envelope(vw.value.samples, vw.value.sample_rate)
                vocal_lufs = _lufs_active(vw.value.samples, vw.value.sample_rate)
        if instrumental_stem is not None:
            iw = audio_io.load_mono(instrumental_stem)
            if iw.is_ok():
                from engine.loudness import integrated_lufs

                r = integrated_lufs(iw.value.samples, iw.value.sample_rate)
                instrumental_lufs = float(r.value) if r.is_ok() else None
```

`_profile_from_json` gains (using `.get` so v2 files without a field still load):

```python
        vocal_energy=tuple(d["vocal_energy"]) if d.get("vocal_energy") else None,
        vocal_lufs=d.get("vocal_lufs"),
        instrumental_lufs=d.get("instrumental_lufs"),
```

(`_profile_to_json` already serializes via `asdict`; only Path fields need the existing str() loop — the new fields are JSON-native.)

- [ ] **Step 5: Run to verify PASS** — 5 new tests + full offline suite green (32 passed, 1 deselected; the property/als tests are unaffected).

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: profile v2 — vocal energy envelope, active-gated LUFS, stem-reuse guard"`

---

### Task 2: Hook selection (`compiler/hook.py`)

**Files:**
- Create: `$MC/compiler/hook.py`
- Test: `$MC/tests/test_hook.py`

**Interfaces:**
- Consumes: `SongProfile.{vocal_energy, downbeat_times, bpm, duration_s, vocal_onset_s}`, `compiler.analyze.ENERGY_HZ`
- Produces: `HookWindow(anchor_s: float, content_start_s: float, end_s: float, score: float)` (frozen dataclass; `anchor_s` = the downbeat that must land on the bed's grid; `content_start_s ≤ anchor_s` includes the pickup); `select_hooks(profile: SongProfile, bars: int = 16, max_rank: int = 4) -> tuple[HookWindow, ...]` (ranked best-first, non-overlapping, never empty for a profile with downbeats).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hook.py
from __future__ import annotations

from pathlib import Path

from compiler.hook import HookWindow, select_hooks
from compiler.models import SongProfile


def _profile(env: tuple[float, ...], bpm: float = 120.0, dur: float = 120.0) -> SongProfile:
    downbeats = tuple(i * 2.0 for i in range(int(dur // 2)))  # 120bpm -> bar = 2s
    return SongProfile(Path("v"), bpm, "C", "minor", (), downbeats, dur,
                       Path("v.wav"), None, 0.0, vocal_energy=env)


def test_picks_hottest_16_bars() -> None:
    # 120s song @2Hz -> 240 frames. Hot region 40-72s (frames 80-144).
    env = tuple(1.0 if 80 <= i < 144 else 0.05 for i in range(240))
    hooks = select_hooks(_profile(env))
    # 16 bars @120bpm = 32s. Best window starts at the bar closest to 40s.
    assert abs(hooks[0].anchor_s - 40.0) <= 2.0
    assert abs((hooks[0].end_s - hooks[0].anchor_s) - 32.0) < 1e-6


def test_search_restricted_to_first_60_percent() -> None:
    env = tuple(1.0 if i >= 200 else 0.05 for i in range(240))  # hot only at the end
    hooks = select_hooks(_profile(env))
    assert hooks[0].anchor_s <= 0.6 * 120.0


def test_ranked_hooks_do_not_overlap() -> None:
    env = tuple(1.0 if (40 <= i < 104 or 120 <= i < 184) else 0.05 for i in range(240))
    hooks = select_hooks(_profile(env))
    assert len(hooks) >= 2
    assert abs(hooks[0].anchor_s - hooks[1].anchor_s) >= 16.0  # >= half a window


def test_pickup_lead_when_energy_before_anchor() -> None:
    # energy starts 1s BEFORE the 40s bar line (frame 78)
    env = tuple(1.0 if 78 <= i < 144 else 0.0 for i in range(240))
    hooks = select_hooks(_profile(env))
    h = hooks[0]
    assert h.content_start_s < h.anchor_s          # pickup included
    assert h.anchor_s - h.content_start_s <= 2.0   # at most one bar


def test_no_envelope_falls_back_to_onset() -> None:
    p = SongProfile(Path("v"), 120.0, "C", "minor", (),
                    tuple(i * 2.0 for i in range(60)), 120.0,
                    Path("v.wav"), None, 10.3)
    hooks = select_hooks(p)
    assert len(hooks) == 1
    assert abs(hooks[0].anchor_s - 10.0) <= 2.0    # nearest downbeat to onset
```

- [ ] **Step 2: Run to verify FAIL** — ModuleNotFoundError.

- [ ] **Step 3: Write `compiler/hook.py`**

```python
"""Hook selection — the grammar's 'which part of the song' rule.

bb_mashup_grammar_v1: DJs take ~16-bar slots (median 29 s), from early in the
song (normalized entry median 0.11), with pickup-led entries (the warped grid
locks; the clip edge leads it by up to a bar)."""
from __future__ import annotations

from dataclasses import dataclass

from compiler.analyze import ENERGY_HZ
from compiler.models import SongProfile
from compiler.placement import snap_to_grid

SEARCH_FRACTION = 0.6      # hooks live in the first 60% of the song
PICKUP_GATE = 0.15         # pre-bar mean energy (rel peak) that earns a pickup
BEATS_PER_BAR = 4


@dataclass(frozen=True)
class HookWindow:
    anchor_s: float          # the downbeat that lands on the bed's grid
    content_start_s: float   # <= anchor_s; includes the pickup
    end_s: float
    score: float


def _mean(env: tuple[float, ...], t0: float, t1: float) -> float:
    i0, i1 = max(0, int(t0 * ENERGY_HZ)), min(len(env), int(t1 * ENERGY_HZ))
    if i1 <= i0:
        return 0.0
    return sum(env[i0:i1]) / (i1 - i0)


def select_hooks(profile: SongProfile, bars: int = 16, max_rank: int = 4) -> tuple[HookWindow, ...]:
    bar_s = BEATS_PER_BAR * 60.0 / profile.bpm
    window_s = bars * bar_s
    env = profile.vocal_energy
    if not env:
        anchor = snap_to_grid(profile.vocal_onset_s or 0.0, profile.downbeat_times)
        end = min(anchor + window_s, profile.duration_s)
        return (HookWindow(anchor, anchor, end, 0.0),)

    limit = SEARCH_FRACTION * profile.duration_s
    scored = [
        (a, _mean(env, a, a + window_s))
        for a in profile.downbeat_times
        if a <= limit and a + window_s <= profile.duration_s
    ]
    if not scored:
        scored = [(profile.downbeat_times[0] if profile.downbeat_times else 0.0, 0.0)]
    scored.sort(key=lambda t: -t[1])

    kept: list[HookWindow] = []
    for anchor, score in scored:
        if any(abs(anchor - k.anchor_s) < window_s / 2 for k in kept):
            continue
        lead = bar_s if _mean(env, anchor - bar_s, anchor) > PICKUP_GATE else 0.0
        kept.append(HookWindow(
            anchor_s=anchor,
            content_start_s=max(0.0, anchor - lead),
            end_s=anchor + window_s,
            score=score,
        ))
        if len(kept) >= max_rank:
            break
    return tuple(kept)
```

- [ ] **Step 4: Run to verify PASS** (5 tests; full suite 37 passed).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: ranked 16-bar hook selection with pickup lead (grammar rule 1+3)"`

---

### Task 3: Compile v2 — structure, pickup placement, LUFS auto-gain

**Files:**
- Modify: `$MC/compiler/models.py` (MashupIntent gains `hook_rank: int = 0`, `full_bed: bool = False` — appended after `vocal_gain_db`)
- Modify: `$MC/compiler/compile.py` (rewrite `compile_mashup` body)
- Test: rewrite `$MC/tests/test_compile.py` (v1 expectations are obsolete BY DESIGN — whole-vocal behavior was the bug)

**Interfaces:**
- Consumes: `select_hooks`, `HookWindow`, profile v2 fields
- Produces: same signature `compile_mashup(intent, vocal, instrumental) -> Result[MashupTimeline, GateFailure]`; clip[0]=bed (trimmed to `drop + hook + OUTRO_BARS` bars unless `intent.full_bed`), clip[1]=vocal hook (arr_start may be < drop_beats by the pickup lead; `pitch_semitones` from gate; `gain_db` = LUFS delta clamp ±9 when intent.vocal_gain_db == 0 and both LUFS present, else intent value + LUFS delta is skipped). Constants: `HOOK_BARS = 16`, `OUTRO_BARS = 8`, `GAIN_CLAMP_DB = 9.0`.

- [ ] **Step 1: Rewrite `tests/test_compile.py`**

```python
# tests/test_compile.py — v2 grammar expectations
from __future__ import annotations

from pathlib import Path

from compiler.compile import HOOK_BARS, OUTRO_BARS, compile_mashup
from compiler.models import MashupIntent, SongProfile


def _instr(lufs: float | None = -10.0) -> SongProfile:
    return SongProfile(Path("instr.mp3"), 120.0, "C", "major",
                       (), tuple(0.5 + i * 2.0 for i in range(120)), 240.5,
                       None, Path("instr_stem.wav"), None,
                       instrumental_lufs=lufs)


def _vocal(bpm: float = 120.0, lufs: float | None = -16.0) -> SongProfile:
    # hot vocal region frames 80-144 (40-72s); downbeats every 2s from 0.4
    env = tuple(1.0 if 80 <= i < 144 else 0.02 for i in range(int(160.4 * 2)))
    return SongProfile(Path("voc.mp3"), bpm, "A", "minor",
                       (), tuple(0.4 + i * 2.0 for i in range(80)), 160.4,
                       Path("voc_stem.wav"), None, 10.3,
                       vocal_energy=env, vocal_lufs=lufs)


def test_v2_structure_and_hook() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i"), drop_bar=16), _vocal(), _instr())
    assert r.is_ok()
    t = r.value
    bed, voc = t.clips
    # bed trimmed: drop(16) + hook(16) + outro(8) = 40 bars = 160 beats
    assert bed.arr_len_beats == (16 + HOOK_BARS + OUTRO_BARS) * 4
    # hook body = 16 bars = 64 beats anchored at drop; anchor near 40.4s
    assert voc.arr_start_beats <= 64.0                       # pickup may lead
    anchor_beats = 64.0
    body_beats = (voc.arr_start_beats + voc.arr_len_beats) - anchor_beats
    assert abs(body_beats - HOOK_BARS * 4) < 1e-6
    assert 38.0 <= voc.content_start_s <= 42.5               # hook region, maybe pickup


def test_lufs_auto_gain() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i")), _vocal(lufs=-16.0), _instr(lufs=-10.0))
    assert r.is_ok()
    assert r.value.clips[1].gain_db == 6.0                   # -10 - (-16), within clamp


def test_lufs_gain_clamped() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i")), _vocal(lufs=-30.0), _instr(lufs=-10.0))
    assert r.is_ok()
    assert r.value.clips[1].gain_db == 9.0


def test_explicit_gain_wins_over_auto() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i"), vocal_gain_db=-2.0), _vocal(), _instr())
    assert r.is_ok()
    assert r.value.clips[1].gain_db == -2.0


def test_full_bed_keeps_whole_instrumental() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i"), full_bed=True), _vocal(), _instr())
    assert r.is_ok()
    assert r.value.clips[0].arr_len_beats == 480.0           # v1 behavior preserved


def test_hook_rank_picks_second_region() -> None:
    env = list(0.02 for _ in range(321))
    for i in range(80, 144):
        env[i] = 1.0
    for i in range(180, 244):
        env[i] = 0.8
    v = SongProfile(Path("voc.mp3"), 120.0, "A", "minor",
                    (), tuple(0.4 + i * 2.0 for i in range(80)), 160.4,
                    Path("voc_stem.wav"), None, 10.3,
                    vocal_energy=tuple(env), vocal_lufs=-16.0)
    r0 = compile_mashup(MashupIntent(Path("v"), Path("i"), hook_rank=0), v, _instr())
    r1 = compile_mashup(MashupIntent(Path("v"), Path("i"), hook_rank=1), v, _instr())
    assert r0.is_ok() and r1.is_ok()
    assert r1.value.clips[1].content_start_s > r0.value.clips[1].content_start_s + 10


def test_drop_past_bed_end_fails() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i"), drop_bar=121), _vocal(), _instr())
    assert not r.is_ok() and r.error.kind == "placement"


def test_gate_failure_propagates() -> None:
    r = compile_mashup(MashupIntent(Path("v"), Path("i")), _vocal(bpm=100.0), _instr())
    assert not r.is_ok() and r.error.kind == "stretch"
```

- [ ] **Step 2: Run to verify FAIL** (ImportError on HOOK_BARS; old behavior fails new assertions).

- [ ] **Step 3: Extend `MashupIntent` in `compiler/models.py`**

```python
    hook_rank: int = 0        # refinement: 0 = best hook, 1 = next region, ...
    full_bed: bool = False    # keep the whole instrumental (v1 behavior)
```

- [ ] **Step 4: Rewrite `compiler/compile.py`**

```python
"""intent + profiles -> MashupTimeline. Pure; no I/O.

v2 implements bb_mashup_grammar_v1: 16-bar hook from the hottest early vocal
region (pickup-led), LUFS-matched gain, and a tight intro/hook/outro bed."""
from __future__ import annotations

from compiler.gate import GateFailure, check
from compiler.hook import select_hooks
from compiler.models import ClipSpec, MashupIntent, MashupTimeline, SongProfile
from engine.result import Err, Ok, Result

BEATS_PER_BAR = 4
HOOK_BARS = 16
OUTRO_BARS = 8
GAIN_CLAMP_DB = 9.0


def _auto_gain(intent: MashupIntent, vocal: SongProfile, instr: SongProfile) -> float:
    if intent.vocal_gain_db != 0.0:
        return intent.vocal_gain_db
    if vocal.vocal_lufs is None or instr.instrumental_lufs is None:
        return 0.0
    delta = instr.instrumental_lufs - vocal.vocal_lufs
    return max(-GAIN_CLAMP_DB, min(GAIN_CLAMP_DB, delta))


def compile_mashup(
    intent: MashupIntent, vocal: SongProfile, instrumental: SongProfile
) -> Result[MashupTimeline, GateFailure]:
    gate = check(vocal, instrumental)
    if not gate.is_ok():
        return gate

    tempo = instrumental.bpm
    i_start = instrumental.downbeat_times[0] if instrumental.downbeat_times else 0.0
    bed_avail_beats = (instrumental.duration_s - i_start) * instrumental.bpm / 60.0

    drop_beats = float(intent.drop_bar * BEATS_PER_BAR)
    if bed_avail_beats - drop_beats <= 0:
        return Err(GateFailure(
            "placement",
            f"drop bar {intent.drop_bar} is past the instrumental's "
            f"{bed_avail_beats / BEATS_PER_BAR:.0f} bars",
        ))

    hooks = select_hooks(vocal, bars=HOOK_BARS)
    hw = hooks[min(max(intent.hook_rank, 0), len(hooks) - 1)]

    lead_beats = (hw.anchor_s - hw.content_start_s) * vocal.bpm / 60.0
    lead_beats = min(lead_beats, drop_beats)          # never before arrangement 0
    body_beats = (hw.end_s - hw.anchor_s) * vocal.bpm / 60.0
    body_beats = min(body_beats, bed_avail_beats - drop_beats)
    content_start = hw.anchor_s - lead_beats * 60.0 / vocal.bpm
    content_end = hw.anchor_s + body_beats * 60.0 / vocal.bpm

    vocal_clip = ClipSpec(
        source=vocal.vocal_stem or vocal.path,
        role="acappella",
        arr_start_beats=drop_beats - lead_beats,
        arr_len_beats=lead_beats + body_beats,
        content_start_s=content_start,
        content_end_s=content_end,
        pitch_semitones=gate.value.transpose_semitones,
        gain_db=_auto_gain(intent, vocal, instrumental),
    )

    if intent.full_bed:
        bed_len_beats = bed_avail_beats
    else:
        bed_len_beats = min(bed_avail_beats, drop_beats + body_beats + OUTRO_BARS * BEATS_PER_BAR)
    bed_clip = ClipSpec(
        source=instrumental.instrumental_stem or instrumental.path,
        role="instrumental",
        arr_start_beats=0.0,
        arr_len_beats=bed_len_beats,
        content_start_s=i_start,
        content_end_s=i_start + bed_len_beats * 60.0 / instrumental.bpm,
        pitch_semitones=0,
        gain_db=0.0,
    )
    return Ok(MashupTimeline(tempo_bpm=tempo, clips=(bed_clip, vocal_clip)))
```

- [ ] **Step 5: Run to verify PASS** (8 tests; full offline suite green — the composition/property tests hand-build timelines so are unaffected; `tests/test_pipeline_composition.py` calls `compile_mashup` with profiles lacking `vocal_energy` → exercises the onset fallback; if its assertions on `ref_start_s` fail, update THAT test's expectation to the fallback anchor, not the compiler).

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: compiler v2 — hook placement, pickup lead, LUFS auto-gain, 40-bar structure"`

---

### Task 4: Render micro-fades

**Files:**
- Modify: `$MC/compiler/render.py` (`_process` gains edge fades)
- Test: append to `$MC/tests/test_render.py`

**Interfaces:**
- Produces: every rendered clip gets `FADE_S = 0.01` linear fade-in/out (kills clicks at hook boundaries — v2 clips start/end mid-song).

- [ ] **Step 1: Failing test (append)**

```python
@needs_rubberband
def test_clip_edges_are_faded(tmp_path: Path) -> None:
    src = _tone(tmp_path / "t.wav", 220.0, 4.0)
    t = MashupTimeline(tempo_bpm=120.0, clips=(
        ClipSpec(src, "acappella", 0.0, 8.0, 1.0, 3.0, 0, 0.0),  # mid-song slice
    ))
    y, sr = sf.read(render(t, tmp_path / "f.wav"))
    assert abs(y[0]).max() < 0.02          # faded in from ~zero
    head = abs(y[: int(0.001 * sr)]).max()
    body = abs(y[int(0.5 * sr) : int(1.0 * sr)]).max()
    assert head < body * 0.2
```

- [ ] **Step 2: Verify FAIL** (first sample is full-amplitude mid-sine today).

- [ ] **Step 3: Implement** — in `compiler/render.py`, add `FADE_S = 0.01` beside `_PEAK_DBFS`, and at the END of `_process` (after gain):

```python
    edge = min(int(FADE_S * sr), len(y) // 2)
    if edge > 0:
        ramp = np.linspace(0.0, 1.0, edge)[:, None]
        y[:edge] *= ramp
        y[-edge:] *= ramp[::-1]
    return y
```

(`y` may be 1-D after pyrubberband on mono input — guard: `ramp = ramp if y.ndim == 2 else ramp[:, 0]`.)

- [ ] **Step 4: Verify PASS**; full suite green.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: 10ms clip-edge fades in renderer"`

---

### Task 5: CLI flags + demo re-render (the A/B)

**Files:**
- Modify: `$MC/compiler/main.py`

**Interfaces:**
- Produces: `--hook-rank N` (default 0), `--full-bed` flags; stdout reports the chosen hook (`hook: 40.4s–72.4s of the vocal, gain +6.0 dB`).

- [ ] **Step 1: Extend `main.py`** — add args and thread them into `MashupIntent`:

```python
    p.add_argument("--hook-rank", type=int, default=0)
    p.add_argument("--full-bed", action="store_true")
```

```python
    intent = MashupIntent(
        args.vocal_song, args.instrumental_song, args.drop_bar,
        args.vocal_gain_db, hook_rank=args.hook_rank, full_bed=args.full_bed,
    )
    timeline = compile_mashup(intent, v, i)
```

After a successful compile, before rendering, print the vocal clip's plan:

```python
    voc = timeline.value.clips[1]
    print(f"  hook: {voc.content_start_s:.1f}s–{voc.content_end_s:.1f}s of the vocal, "
          f"gain {voc.gain_db:+.1f} dB, enters beat {voc.arr_start_beats:.1f}")
```

- [ ] **Step 2: Full offline suite green; CLI smoke** — `venv/bin/python -m compiler.main --help` exits 0 showing the new flags.

- [ ] **Step 3: Re-render the demo pair (cache-invalidation check rides along)**

```bash
venv/bin/python -m compiler.main ~/Desktop/mashup_demo/vocal.wav \
    ~/Desktop/mashup_demo/instr.wav --out out/demo_v2
```

Expected: re-analysis runs (profile v1 caches invalid) but **finishes in ~1–2 min, NOT ~30 min** — the stem-reuse guard must skip separation (stems exist on disk). If it starts a Roformer run, STOP and fix Task 1's guard. Output: `out/demo_v2.wav` (~80s long, not 3+ min) + `out/demo_v2.als`.

- [ ] **Step 4: Human ear test (report, don't gate)** — note in the report that `out/demo.wav` (v1, awful) vs `out/demo_v2.wav` is John's A/B; the implementer only verifies duration ≈ 40 bars and non-silence in the first 5s and at the drop.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: CLI hook-rank/full-bed flags + v2 demo render"`

---

### Task 6: Refinement verbs in seam (P0-3 — the preference log)

**Files:**
- Modify: `$MC/server/db.py` (columns + migration), `$MC/server/jobs.py` (thread params into intent), `$MC/server/app.py` (`/api/refine`), `$MC/server/templates/feed.html` (verb buttons + lineage label), `$MC/server/static/app.js` (verb POST), `$MC/requirements.txt` (`httpx>=0.27` for TestClient)
- Test: `$MC/tests/test_refine.py`

**Interfaces:**
- Produces: `mashes` gains columns `hook_rank INTEGER NOT NULL DEFAULT 0`, `vocal_gain_db REAL NOT NULL DEFAULT 0`, `parent_mash_id INTEGER`, `verb TEXT`; `POST /api/refine` with form `mash_id`, `verb ∈ {different_hook, drop_earlier, drop_later, more_beat}` → inserts child mash derived from parent (`different_hook`: hook_rank+1; `drop_earlier/later`: drop_bar∓8 floor 8; `more_beat`: vocal_gain_db−3), enqueues, returns `{"id": N}`. The parent link + verb IS the preference-pair log.

- [ ] **Step 1: Failing tests**

```python
# tests/test_refine.py
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from server import db as sdb

    monkeypatch.setattr(sdb, "DB_PATH", tmp_path / "state.db")
    sdb.init()
    con = sdb.connect()
    con.execute("INSERT INTO songs(id,title,path,status,bpm,key_tonic,key_mode,key_pc,duration_s,created_at)"
                " VALUES(1,'A','/a.wav','ready',123,'C','minor',3,200,0),"
                "       (2,'B','/b.wav','ready',123,'C','minor',3,200,0)")
    con.execute("INSERT INTO mashes(id,vocal_song_id,instr_song_id,drop_bar,hook_rank,vocal_gain_db,"
                " status,created_by,created_at) VALUES(1,1,2,16,0,0,'done','nick',0)")
    con.commit(); con.close()

    from server import app as sapp, jobs

    monkeypatch.setattr(jobs, "enqueue", lambda kind, rid: None)
    monkeypatch.setattr(sapp, "_scan", lambda: None)
    from fastapi.testclient import TestClient

    return TestClient(sapp.app)


def _refine(client, verb: str) -> dict:
    r = client.post("/api/refine", data={"mash_id": 1, "verb": verb})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("verb,col,expected", [
    ("different_hook", "hook_rank", 1),
    ("drop_later", "drop_bar", 24),
    ("drop_earlier", "drop_bar", 8),
    ("more_beat", "vocal_gain_db", -3.0),
])
def test_verbs_derive_child(client, verb, col, expected) -> None:
    from server import db as sdb

    child = _refine(client, verb)["id"]
    row = sdb.connect().execute("SELECT * FROM mashes WHERE id=?", (child,)).fetchone()
    assert row["parent_mash_id"] == 1 and row["verb"] == verb
    assert row[col] == expected
    assert row["status"] == "queued"


def test_unknown_verb_rejected(client) -> None:
    r = client.post("/api/refine", data={"mash_id": 1, "verb": "make_it_pop"})
    assert r.status_code == 400
```

- [ ] **Step 2: Verify FAIL.** (`venv/bin/pip install httpx` first; add `httpx>=0.27` to requirements.txt.)

- [ ] **Step 3: Implement**

`server/db.py` — extend `_SCHEMA`'s mashes CREATE with the four columns, and add migration for existing DBs at the end of `init()`:

```python
def _migrate(con: sqlite3.Connection) -> None:
    have = {r[1] for r in con.execute("PRAGMA table_info(mashes)")}
    for col, ddl in (
        ("hook_rank", "INTEGER NOT NULL DEFAULT 0"),
        ("vocal_gain_db", "REAL NOT NULL DEFAULT 0"),
        ("parent_mash_id", "INTEGER"),
        ("verb", "TEXT"),
    ):
        if col not in have:
            con.execute(f"ALTER TABLE mashes ADD COLUMN {col} {ddl}")
```

(call `_migrate(con)` inside `init()` before `commit`).

`server/jobs.py` — in `_mash`, build the intent from the row:

```python
    intent = MashupIntent(
        Path(voc["path"]), Path(ins["path"]), drop_bar=int(m["drop_bar"]),
        vocal_gain_db=float(m["vocal_gain_db"]), hook_rank=int(m["hook_rank"]),
    )
```

`server/app.py`:

```python
_VERBS = {
    "different_hook": lambda p: {"hook_rank": p["hook_rank"] + 1},
    "drop_earlier":   lambda p: {"drop_bar": max(8, p["drop_bar"] - 8)},
    "drop_later":     lambda p: {"drop_bar": p["drop_bar"] + 8},
    "more_beat":      lambda p: {"vocal_gain_db": p["vocal_gain_db"] - 3.0},
}


@app.post("/api/refine")
def refine(request: Request, mash_id: int = Form(...), verb: str = Form(...)):
    fn = _VERBS.get(verb)
    if fn is None:
        return JSONResponse({"error": f"unknown verb {verb!r}"}, status_code=400)
    con = db.connect()
    parent = con.execute("SELECT * FROM mashes WHERE id=?", (mash_id,)).fetchone()
    if parent is None:
        con.close()
        return JSONResponse({"error": "no such mash"}, status_code=404)
    params = {
        "drop_bar": int(parent["drop_bar"]),
        "hook_rank": int(parent["hook_rank"]),
        "vocal_gain_db": float(parent["vocal_gain_db"]),
    }
    params.update(fn(params))
    cur = con.execute(
        "INSERT INTO mashes(vocal_song_id, instr_song_id, drop_bar, hook_rank,"
        " vocal_gain_db, status, parent_mash_id, verb, created_by, created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (parent["vocal_song_id"], parent["instr_song_id"], params["drop_bar"],
         params["hook_rank"], params["vocal_gain_db"], "queued", mash_id, verb,
         _name(request) or "someone", db.now()),
    )
    con.commit(); child = cur.lastrowid; con.close()
    jobs.enqueue("mash", child)
    return {"id": child}
```

`feed.html` — inside the `status == 'done'` branch, after the als link:

```html
      <div class="verbs">
        {% for v, label in [("different_hook","different hook"), ("drop_earlier","drop earlier"),
                            ("drop_later","drop later"), ("more_beat","more beat")] %}
        <button class="verb" data-verb="{{ v }}" data-mash="{{ m['id'] }}">{{ label }}</button>
        {% endfor %}
      </div>
```

and in the `.meta` block, a lineage line when refined:

```html
    {% if m['verb'] %}<span class="lineage">↳ {{ m['verb'].replace('_',' ') }} of #{{ m['parent_mash_id'] }}</span>{% endif %}
```

(feed query already `SELECT m.*` so the columns flow through.)

`app.js` — append:

```js
for (const b of document.querySelectorAll(".verb")) {
  b.addEventListener("click", async () => {
    b.disabled = true;
    const body = new URLSearchParams({ mash_id: b.dataset.mash, verb: b.dataset.verb });
    const r = await fetch("/api/refine", { method: "POST", body });
    if (r.ok) location.reload();
    else b.disabled = false;
  });
}
```

`style.css` — append:

```css
.verbs { display: flex; flex-wrap: wrap; gap: 6px; }
.verb {
  background: var(--panel2); color: var(--paper); border: 1px solid var(--line);
  border-radius: 999px; padding: 6px 12px; font: 500 12.5px var(--body); cursor: pointer;
}
.verb:disabled { opacity: 0.4; }
.lineage { font-family: var(--mono); font-size: 11px; color: var(--dim); }
```

- [ ] **Step 4: Verify PASS** (5 refine tests; full offline suite green).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(server): refinement verbs — child mashes with parent/verb lineage (preference log)"`

---

### Task 7: Docs

**Files:**
- Modify: `$MC/README.md`, `$MC/BACKLOG.md`

- [ ] **Step 1: README** — replace the limitations list entries that v2 fixed (whole-vocal, no level matching) with the v2 behavior (16-bar hook from the hottest early region, pickup-led entry, LUFS auto-gain, 40-bar structure, `--full-bed` escape hatch), and document the four refinement verbs + the fact that refinements record lineage.
- [ ] **Step 2: BACKLOG** — mark LUFS-matching and hook-selection DONE; add: `loop_hook` verb (grammar: 19% of acap spans loop), chorus detection upgrade (energy proxy → section model), bed-structure variants (double drop), verb-log export for the decision model (P0 of docs/mashup_decision_model_plan.md in the research repo).
- [ ] **Step 3: Run full offline suite one final time; commit** — `git add -A && git commit -m "docs: v2 grammar behavior + verbs; backlog refresh"`

---

## Self-review notes

- Grammar coverage: rule 1 (span ~30 s → HOOK_BARS=16) Task 2/3; rule 2 (first-chorus source → SEARCH_FRACTION 0.6 + energy argmax) Task 2; rule 3 (pickup-led entry) Task 2/3; rule 4 (flat ducking → LUFS match once) Task 1/3; bed ≈36–40 bars Task 3; loops deferred to backlog (explicitly).
- Type consistency: `HookWindow` fields used identically in Tasks 2/3; `MashupIntent.hook_rank/full_bed` threaded Task 3→5→6; mashes columns Task 6 match jobs' intent construction.
- Known ripple: `tests/test_pipeline_composition.py` exercises the no-envelope fallback after Task 3 — its vocal-clip assertions may need the fallback-anchor expectation (called out in Task 3 Step 5).

# Real-Live `.als` test fixtures

Hand-authored **in Ableton Live** (not synthesized) so the codec is tested
against Live's real serialization, not our approximation of it. Any `*.als`
committed to this directory is auto-discovered by
[test_als_roundtrip.py](../../test_als_roundtrip.py) and put through the full
law set: validate-clean, reparse stability, tempo/locator write round-trips.

## How to author one

1. New Live set. Use **tiny audio** (a few seconds of silence/tone is fine —
   parsing never opens the media, and short clips keep the `.als` small).
2. Build ONE variation from the matrix below. One behavior per fixture — small
   sets with a single twist beat one big set with everything.
3. Save as `<name>.als` here. Live's save is already gzipped XML; a minimal
   set is ~50–150 KB — fine to commit.
4. Optional: add a `<name>.expect.json` sidecar pinning extraction facts, e.g.
   `{"layer_clips": 3, "tempo_points": 4}` — the test asserts them when present.
   (`layer_clips` counts AudioClips on non-`1-mix`/`2-mix` audio tracks;
   `tempo_points` counts master-tempo breakpoints incl. the writer's clamped
   sentinel.)

## Fixture matrix (each row = one small set)

| Fixture name | What to build | Code path it pins |
|---|---|---|
| `warped_basic` | 1-mix track with a **warped** clip + 2 layer clips, flat tempo | `ArrangementMapper` (BB12 convention) |
| `master_tempo` | **unwarped** 1-mix stub + tempo automation with a ramp AND a step (two nodes at one beat) | `TempoArrangementMapper` + exact ramp integral (BB11 convention) |
| `pitch` | clips with PitchCoarse ±N and a **fractional** PitchFine (e.g. 25.5¢) | pitch parsing (detune rounding) |
| `volume_rides` | fader automation riding a clip, one clip fully muted (slider at -inf), one clip with **no** automation | envelope/audibility semantics + the muted-but-"playing" class |
| `group_tracks` | layer tracks inside a Group track | `group_name` attribution |
| `locators` | several arrangement markers, incl. non-ASCII names | locator read/write round-trip |
| `live12_save` | any of the above re-saved in **Live 12** (when available) | version drift (`version-unknown` gate) |

Name variations `<base>_v2.als` etc. Live-version coverage matters more than
quantity: re-saving the same set in a different Live version is a cheap,
high-value fixture.

## Bundled audio

`audio/` mirrors the aligning-folder layout (`tracks/`, `stems/<song>/`,
`.../candidates/vocals/`) so `classify_path` semantics stay realistic, and the
fixtures' `Path` refs point at it — the sets stay openable in Live even after
the source `~/aligning/` folder is deleted. When authoring a new fixture, use
files already in `audio/` where possible; otherwise copy the new file in
(mirroring its subpath) and keep it small.

## What does NOT belong here

Real DJ-set sessions (BB11/BB12 …) — those stay Mac-local and are pinned by
the skip-if-missing goldens in the same test file. This directory is for
minimal, committable sessions only.

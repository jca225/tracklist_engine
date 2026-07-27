# Ableton ReAct harness — design

**Date:** 2026-07-19  
**Status:** approved for implementation  
**Package:** `alignment/daw_env/`

## Problem

Human labeling became tractable as a ReAct loop in Ableton: listen → place/adjust → listen. The agentic loop is ReAct over **probes only**; Ableton seeding is a one-shot write. The agent never gets DAW actions + listen observations in the same belief loop.

## Goals

| Mode | Purpose |
|------|---------|
| **A** | Autonomous alignment: agent mutates session geometry, listens, senses, commits a timeline |
| **B** | Labeling assist: same loop, but `escalate_human` leaves a Live-openable `.als` for correction |

## Non-goals (v1)

- Live OSC / MIDI remote / freeze-bounce automation  
- New MIR sensors (sensor phase stays frozen)  
- Promoting daw_env into default `make align` / race board before a GO ledger  

## Architecture (ALS-first)

```
SpanBelief → DAW action → .als / session mutate → offline render window
         → existing sensors → Observation → SpanBelief
```

- **Act:** mutate gzipped `.als` (and/or in-memory span geometry) via seeder clip primitives.  
- **Listen:** crop mix (or stem bus) to WAV under `out/daw_env/<set_id>/`.  
- **Sense:** emit `Observation`s (`daw_onset`, optional fp/HuBERT when caches exist).  
- **Session name:** `* DAW REACT.als` — never `* align.als`.

## Action API (frozen)

- `place_span(slot, set_start_s, set_end_s, ref_start_s)`  
- `nudge_set_start(slot, delta_s)` / `nudge_ref_start(slot, delta_s)`  
- `solo_layer(bus)` — `mix` | `mix_instrumental` | `mix_vocals` | slot  
- `render_listen(t0, t1)`  
- `commit_span(slot)` / `escalate_human(slot)`  

## Success gate (Mode A)

Shadow scorecard vs agentic baseline on BB11/BB12: no regression on placement median / fiber-aware traj; aim for a win on ridge `decoder_wall` hard-placement spans. Ledger GO/NO-GO in `attic/EXPERIMENTS.md`.

## CLI

```bash
venvs/audio/bin/python -m alignment.daw_env \
  --set-id 1fsnxchk --mode a|b [--timeline PATH] [--max-steps N]
```

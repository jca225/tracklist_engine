# daw_env — Ableton ReAct harness (ALS-first)

Place → listen → sense → iterate. Spec:
[`docs/superpowers/specs/2026-07-19-ableton-react-harness-design.md`](../../../../docs/superpowers/specs/2026-07-19-ableton-react-harness-design.md).

## Mode A (autonomous shadow)

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.daw_env \
  --set-id 1fsnxchk --mode a \
  --timeline workspaces/alignment_prototype/out/1fsnxchk_predicted_timeline_lt.json \
  --set-dir ~/aligning/<bb12-folder>
```

Writes `out/daw_env/<set_id>/` timeline + `events.jsonl` + `action_history.json`.
Does **not** change default `make align` until EXPERIMENTS.md GO.

## Mode B (labeling assist)

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.daw_env \
  --set-id 1fsnxchk --mode b \
  --als ~/aligning/.../BB12\ SEEDED.als \
  --set-dir ~/aligning/<bb12-folder>
```

Copies the seed to `* DAW REACT.als` (never `* align.als`), runs the loop,
stamps `AGENT PROPOSE <slot>` locators on escalated spans.

**Human handoff:** open the `DAW REACT.als` in Live → correct clips →

```bash
venvs/audio/bin/python -m labeling.export_als_to_gt \
  --als <path-to-DAW-REACT.als> --set-dir <aligning-folder>
```

## Scorecard (operator)

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \
  --set-id 1fsnxchk \
  --timeline workspaces/alignment_prototype/out/daw_env/1fsnxchk/1fsnxchk_daw_react_timeline.json \
  --fibers --decompose
```

Compare to agentic `_lt` baseline; ledger verdict in `attic/EXPERIMENTS.md`.

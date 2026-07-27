# Plan — Ableton ReAct harness

Implements [`docs/superpowers/specs/2026-07-19-ableton-react-harness-design.md`](../specs/2026-07-19-ableton-react-harness-design.md).

## Delivered

| Phase | Status |
|-------|--------|
| P0 env kernel | `daw_env/{actions,session,als_mutate,render}.py` + `tests/alignment_prototype/test_daw_env.py` |
| P1 sense loop | `sense.py` + `loop.resolve_daw` + event log |
| P2 Mode A CLI | `python -m alignment.daw_env --mode a` → shadow timeline; scorecard operator recipe in README; EXPERIMENTS ledger **SHADOW** pending BB audio run |
| P3 Mode B | `--mode b` + `AGENT PROPOSE` locators + `* DAW REACT.als` + export handoff in README |

## Operator scorecard (GO/NO-GO)

When BB11/BB12 mix audio + `_lt` timeline are present:

```bash
venvs/audio/bin/python -m alignment.daw_env --set-id 1fsnxchk --mode a --set-dir ~/aligning/<bb12>
venvs/audio/bin/python -m alignment.score_timeline_vs_gt \
  --set-id 1fsnxchk \
  --timeline alignment/out/daw_env/1fsnxchk/1fsnxchk_daw_react_timeline.json \
  --fibers --decompose
```

Promote to race board only on EXPERIMENTS.md **GO**.

# Session checkpoint — AUTO_COMMIT / gpubox flywheel (2026-07-20)

**Closed:** 2026-07-20 morning. Resume from this doc; do not re-litigate closed paths.

**SSOT headlines:** unchanged — cite [docs/alignment_status.md](../../docs/alignment_status.md) only.
Identity ~solved; binding walls still **placement + decode-residual (structure)**.

---

## What landed (code)

| Piece | Path |
|---|---|
| Aligning path remap (Mac abs → set_dir) | `alignment/aligning_paths.py` + `_vocal_ref_path(..., set_dir=)` in `infer.py` / `live_runners.py` |
| Tests | `alignment/tests/test_aligning_paths.py` (6 green) |
| gpubox driver | `scripts/gpubox_agentic_both.py` — allowlist push, `--bb10-only`, py3.12 conda `align`, vocal preflight |
| Bedtime watchdog | `scripts/gpubox_bb10_bedtime_watchdog.py` |
| gpubox mandate | `AGENTS.md`, `docs/vast_coordination.md`, `.cursor/rules/vast-gpubox.mdc` |
| Spec / plan | `docs/superpowers/specs/2026-07-19-auto-commit-sensor-restore-design.md`, `docs/superpowers/plans/2026-07-19-auto-commit-sensor-restore.md` |

Allowlist must include: `core`, `analysis`, `labeling`, `eda/{__init__,alignment/__init__,mert_vectors}`, **`workspaces/section_hsmm`**, `alignment`. Never full-repo rsync (`.claude` ~40G).

---

## Harvest results (diagnostics)

| Run | BB10 AUTO | Notes |
|---|---|---|
| Mac E1 (earlier) | **16/113** | all `agentic:lyrics` @ q=0.76 |
| gpubox first both (`45324301`) | **0** | lyrics/hubert abstain — Mac absolute paths |
| gpubox lyrics-only (`45334205`) | **10/113** | path remap worked; hubert skipped (`section_hsmm` missing) |
| gpubox hubert+lyrics (`45344153`) | **none** | died mid-Whisper twice; bedtime watchdog destroyed at +90m |

**Artifacts:** `alignment/out/vast_agentic/`
- `w1mgcjt_agentic_timeline.json` / `w1mgcjt_pseudo_gt.yaml` = **10 AUTO** harvest
- `agentic_both_bb10_hubert_deadline.log` = failed hubert attempt

**Do not train** on the 0-AUTO starvation yaml. 10-AUTO yaml is train-eligible only as a diagnostic (prior 16-label TRM already under control — see E1 write-up).

---

## Vast / money

- **No** `align-bb10-flywheel` / `align-agentic-both` attached.
- Do **not** destroy other live instances (A100s / `alignment-race` unless you own them).
- Always **gpubox** for rent/attach/destroy.

---

## Resume checklist (next session)

1. Optional: Mac BB10 agentic with hubert available locally — confirm ~16 AUTO without GPU rent.
2. Or gpubox `--bb10-only` with allowlist including `section_hsmm`; watch Whisper OOM/silent death; keep bedtime watchdog.
3. If AUTO ≥ ~Mac 16: rematerialize → short TRM train BB10-pseudo → eval BB11 (**diagnostic only**).
4. Else / after: pivot to placement/structure actor walls (north star), not more GPU on thin AUTO pools.
5. Phase-2 ladder retune only if coverage stays &lt;8/113 after sensors work — **not** triggered after 10/113.

---

## Explicit non-goals left open

- Regenerating `alignment_status.md` from this session (no scorecard move).
- Committing the WIP branch (dirty; ratchet debt may block gate — fix/raise separately).

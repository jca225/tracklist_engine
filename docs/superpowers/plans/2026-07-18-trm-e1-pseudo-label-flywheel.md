# TRM E1 Real Pseudo-Label Flywheel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize safety-gated pseudo-labels from unlabeled BB10, train the TRM on those real-distribution labels, and compare it honestly against BB11 GT and the existing controls.

**Architecture:** Use explicit artifacts between stages: base timeline → pseudo-safe agentic timeline → pseudo-GT YAML → trajectory dataset → model runs. Refactor the existing agentic driver around a shared refinement function so supervised behavior stays unchanged while an unlabeled caller supplies timeline/manifest stem claims instead of GT.

**Tech Stack:** Python 3.14, frozen dataclasses, PyYAML, msgspec timeline contracts, NumPy, PyTorch, pytest, existing agentic probes and trajectory training stack.

## Global Constraints

- Work only in the isolated `e1-flywheel` worktree; do not edit the active shared `trm-ablation-framework` checkout.
- Do not add BB10 to `GT_BY_SET`; BB10 is deliberately unlabeled.
- Never train on the eval mix: pseudo-train set ID must differ from `--eval-set`.
- Enable `Ladder(combine=True)`, the explicit G2 independence predicate, and `LiveContext.apply_fiber_gate=True` for pseudo-label production.
- Only `AUTO_COMMIT` spans surviving G0–G4 may enter the YAML.
- Do not write to the canonical DB or mutate pi-storage; all artifacts stay under local `out/` and model cache paths.
- Do not commit generated timelines, pseudo-GT YAML, event logs, model checkpoints, or experiment outputs.
- The strict referee remains `path_decode.trajectory_acc`; do not change scoring to improve the result.
- Alignment headline numbers belong only in `docs/alignment_status.md`; the experiment ledger records the verdict and artifact pointer without duplicating metrics.

---

### Task 1: Shared pseudo-safe agentic refinement

**Files:**
- Modify: `workspaces/alignment_prototype/drivers/agentic.py`
- Modify: `workspaces/alignment_prototype/agentic/live_runners.py`
- Test: `tests/alignment_prototype/test_agentic.py`

**Interfaces:**
- Consumes: `timeline: dict`, `spans_ctx: list[SpanCtx]`, `runners: dict[str, Runner]`, `EventLog`, `Ladder`.
- Produces: `refine_agentic_spans(...) -> tuple[list[dict], Resolution]`.
- Produces: `pseudo_acceptance(belief: SpanBelief) -> tuple[bool, str | None]`.
- Preserves: `AgenticDriver.align_set(ctx)` output semantics when `pseudo_safe=False`.

- [ ] **Step 1: Write failing tests for the explicit G2 predicate**

Add tests that distinguish independent agreement, correlated agreement, and one genuinely high-precision probe:

```python
from workspaces.alignment_prototype.drivers.agentic import pseudo_acceptance


def test_pseudo_acceptance_requires_independence_or_high_precision():
    independent = _belief(
        _obs("mert_decode", 100.0, prec=0.70),
        _obs("lyrics", 101.0, prec=0.70),
    )
    correlated = _belief(
        _obs("fp", 100.0, prec=0.80),
        _obs("chroma_refine", 101.0, prec=0.80),
    )
    single_high = _belief(_obs("lyrics", 100.0, prec=0.90))
    fiber_ambiguous = _belief(
        Observation(
            "lyrics", 100.0, confidence=1.0, precision=0.90,
            ref_start_s=20.0, detail="[fiber×2 amb]",
        )
    )

    assert pseudo_acceptance(independent) == (True, None)
    assert pseudo_acceptance(correlated) == (False, "g2_independence")
    assert pseudo_acceptance(single_high) == (True, None)
    assert pseudo_acceptance(fiber_ambiguous) == (False, "g3_fiber")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
venvs/audio/bin/python -m pytest \
  tests/alignment_prototype/test_agentic.py::test_pseudo_acceptance_requires_independence_or_high_precision -q
```

Expected: collection fails because `pseudo_acceptance` does not exist.

- [ ] **Step 3: Implement the pure G2 predicate**

In `drivers/agentic.py`, derive members of the winning cluster from the belief’s observations and group them with `INDEPENDENCE_GROUP`:

```python
def pseudo_acceptance(belief: SpanBelief) -> tuple[bool, str | None]:
    top = belief.best()
    if top is None:
        return False, "g0_mode"
    members = tuple(
        obs
        for obs in belief.observations
        if not obs.abstained and obs.probe in top.probes
    )
    if any("[fiber" in obs.detail for obs in members):
        return False, "g3_fiber"
    groups = {
        INDEPENDENCE_GROUP.get(obs.probe, obs.probe)
        for obs in members
    }
    if len(groups) >= 2 or max((obs.precision for obs in members), default=0.0) >= 0.9:
        return True, None
    return False, "g2_independence"
```

- [ ] **Step 4: Write failing tests for shared refinement and pseudo metadata**

Use two synthetic spans and stub runners. Assert general mode preserves current `driver_mode`, while pseudo-safe mode demotes an AUTO_COMMIT belief that fails G2:

```python
def test_refine_agentic_spans_demotes_unsafe_pseudo_commit(tmp_path):
    timeline = {"set_id": "pool", "spans": [_timeline_span("1", "r1")]}
    spans = [SpanCtx("1", "r1", "regular", timeline["spans"][0])]
    runners = {"fp": lambda _: _obs("fp", 25.0, prec=0.80)}

    out, _ = refine_agentic_spans(
        timeline,
        spans,
        runners,
        EventLog(tmp_path / "events.jsonl"),
        ladder=Ladder(auto=0.75, combine=True),
        pseudo_safe=True,
    )

    assert out[0]["driver_mode"] == "review"
    assert out[0]["pseudo_gate_rejection"] == "g2_independence"
```

- [ ] **Step 5: Extract `refine_agentic_spans` without changing the supervised driver**

Move the `resolve` call and output-span construction from `AgenticDriver.align_set` into:

```python
def refine_agentic_spans(
    timeline: dict,
    spans_ctx: list[SpanCtx],
    runners: dict,
    log: EventLog,
    *,
    ladder: Ladder,
    pseudo_safe: bool = False,
) -> tuple[list[dict], Resolution]:
    ...
```

When `pseudo_safe=True`, call `pseudo_acceptance` after the ladder chooses `AUTO_COMMIT`. A failure changes the serialized mode to `review` and adds `pseudo_gate_rejection`. Preserve `agentic_quality`, `start_source`, placement translation, and unchanged `ref_segments`. When `pseudo_safe=False`, emit byte-equivalent semantic fields to today’s driver.

- [ ] **Step 6: Add an explicit fiber-gate constructor option**

Change `LiveContext.from_set` to:

```python
@classmethod
def from_set(
    cls,
    set_id: str,
    spans: list[dict],
    *,
    apply_fiber_gate: bool = False,
) -> LiveContext | None:
    ctx = cls(set_id=set_id, apply_fiber_gate=apply_fiber_gate)
    ...
```

Existing callers retain `False`; the E1 caller passes `True`.

- [ ] **Step 7: Run agentic tests**

Run:

```bash
venvs/audio/bin/python -m pytest tests/alignment_prototype/test_agentic.py -q
```

Expected: all tests pass, including the new G2 and shared-refinement cases.

- [ ] **Step 8: Commit Task 1**

```bash
git add workspaces/alignment_prototype/drivers/agentic.py \
  workspaces/alignment_prototype/agentic/live_runners.py \
  tests/alignment_prototype/test_agentic.py
git commit -m "feat(agentic): add pseudo-safe unlabeled refinement"
```

---

### Task 2: Auditable pseudo-GT materializer

**Files:**
- Create: `workspaces/alignment_prototype/trajectory/pseudo_materialize.py`
- Create: `workspaces/alignment_prototype/trajectory/tests/test_pseudo_materialize.py`
- Modify: `workspaces/alignment_prototype/trajectory/pseudo_labels.py`

**Interfaces:**
- Consumes: validated agentic timeline JSON and pulled-set manifest JSON.
- Produces: `MaterializeReport(total: int, accepted: int, dropped: dict[str, int], output: Path)`.
- Produces: `materialize_pseudo_gt(timeline_path: Path, manifest_path: Path, output_path: Path) -> MaterializeReport`.
- Produces: YAML `{set_id, provenance, gate_counts, tracks}` written atomically.

- [ ] **Step 1: Write failing tests for ID resolution and gate accounting**

Create fixtures with one accepted span, one review span, one G2-demoted span, one structurally invalid span, and one unresolved recording. Assert exact counts:

```python
def test_materialize_counts_each_rejection_and_resolves_track_id(tmp_path):
    timeline = _write_timeline(
        tmp_path,
        spans=[
            _span("1", "rec-a", driver_mode="auto_commit"),
            _span("2", "rec-b", driver_mode="review"),
            _span("3", "rec-c", driver_mode="review",
                  pseudo_gate_rejection="g2_independence"),
            _span("4", "rec-d", driver_mode="auto_commit", set_end_s=0.0),
            _span("5", "missing", driver_mode="auto_commit"),
        ],
    )
    manifest = _write_manifest(tmp_path, tracks=[
        {"slot_label": "1", "recording_id": "rec-a", "track_id": "track-a"},
        {"slot_label": "4", "recording_id": "rec-d", "track_id": "track-d"},
    ])

    report = materialize_pseudo_gt(timeline, manifest, tmp_path / "pseudo.yaml")

    assert report.accepted == 1
    assert report.dropped == {
        "g0_mode": 1,
        "g2_independence": 1,
        "g4_structure": 1,
        "audio_identity": 1,
    }
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
venvs/audio/bin/python -m pytest \
  workspaces/alignment_prototype/trajectory/tests/test_pseudo_materialize.py -q
```

Expected: collection fails because `pseudo_materialize` does not exist.

- [ ] **Step 3: Harden structural validation in `pseudo_gt_row`**

Make `_segments_ok` validate `mix_start_s`, finite numeric values, and mix bounds; reject non-finite slope/timing values. Keep the public return type `dict | None`.

```python
if not all(math.isfinite(v) for v in (mix_start, ref_start, ref_end)):
    return False
if not s0 <= mix_start < s1 or ref_end < ref_start:
    return False
```

- [ ] **Step 4: Implement deterministic manifest resolution**

Index manifest tracks by normalized slot, `recording_id`, and legacy `track_id`. Resolve in this order: existing span `track_id`, recording ID, slot label. Reject ambiguity instead of choosing the first candidate.

```python
def resolve_track_id(span: dict, tracks: list[dict]) -> str | None:
    ...
```

- [ ] **Step 5: Implement materialization and atomic output**

Use `core.contracts.load_timeline` for validation, then load the raw JSON only after validation to retain extension fields. Compute the source SHA-256 from timeline bytes. Write sorted-key YAML to a sibling temporary file, `flush` + `os.fsync`, then `os.replace`.

```python
@dataclass(frozen=True)
class MaterializeReport:
    total: int
    accepted: int
    dropped: dict[str, int]
    output: Path
```

Reject `driver_mode != "auto_commit"` as `g0_mode` unless `pseudo_gate_rejection` names G2/G3. Reject `pseudo_gt_row(...) is None` as `g4_structure`. Reject missing `track_id` as `audio_identity`.

- [ ] **Step 6: Add the CLI**

Support:

```bash
python -m workspaces.alignment_prototype.trajectory.pseudo_materialize \
  --timeline PATH --manifest PATH --output PATH
```

Print `accepted/total` and sorted drop counts. Exit `2` when `accepted == 0` so orchestration can identify starvation distinctly.

- [ ] **Step 7: Test atomic determinism and YAML round-trip**

Assert two runs produce identical bytes, provenance includes SHA-256 and safety settings, every row has `pseudo_label: true` and non-empty `track_id`, and `TrajectorySpanDataset` can construct when audio resolution is stubbed.

- [ ] **Step 8: Run pseudo-label tests**

Run:

```bash
venvs/audio/bin/python -m pytest \
  workspaces/alignment_prototype/trajectory/tests/test_pseudo_labels.py \
  workspaces/alignment_prototype/trajectory/tests/test_pseudo_materialize.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add workspaces/alignment_prototype/trajectory/pseudo_labels.py \
  workspaces/alignment_prototype/trajectory/pseudo_materialize.py \
  workspaces/alignment_prototype/trajectory/tests/test_pseudo_materialize.py
git commit -m "feat(trajectory): materialize audited pseudo ground truth"
```

---

### Task 3: Pseudo-training input and leakage guard

**Files:**
- Modify: `workspaces/alignment_prototype/trajectory/train.py`
- Create: `workspaces/alignment_prototype/trajectory/tests/test_train_pseudo.py`

**Interfaces:**
- Produces: `load_pseudo_train_yaml(path: Path, eval_set: str) -> tuple[str, Path]`.
- Adds CLI: `--train-yaml PATH`.
- Preserves: existing hand-GT, synthetic augmentation, and synthetic-only behavior.

- [ ] **Step 1: Write failing pure validation tests**

```python
def test_load_pseudo_train_yaml_rejects_eval_leakage(tmp_path):
    path = _pseudo_yaml(tmp_path, set_id="2nvzlh2k")
    with pytest.raises(ValueError, match="same set"):
        load_pseudo_train_yaml(path, eval_set="2nvzlh2k")


def test_load_pseudo_train_yaml_requires_provenance_and_rows(tmp_path):
    path = _pseudo_yaml(tmp_path, provenance=None, tracks=[])
    with pytest.raises(ValueError, match="provenance|tracks"):
        load_pseudo_train_yaml(path, eval_set="2nvzlh2k")
```

- [ ] **Step 2: Run validation tests and verify RED**

Run:

```bash
venvs/audio/bin/python -m pytest \
  workspaces/alignment_prototype/trajectory/tests/test_train_pseudo.py -q
```

Expected: import fails because `load_pseudo_train_yaml` does not exist.

- [ ] **Step 3: Add parser and validation**

Make `--train-set` optional at parser level, then require exactly one usable training source for set split:

```python
ap.add_argument("--train-set")
ap.add_argument("--train-yaml", type=Path)

if args.split == "set" and not args.train_set and not args.train_yaml:
    ap.error("--split set requires --train-set or --train-yaml")
if args.train_yaml and args.synthetic_only:
    ap.error("--train-yaml cannot be combined with --synthetic-only")
```

`load_pseudo_train_yaml` verifies set inequality, top-level provenance, non-empty tracks, and `pseudo_label is True` on every row before constructing a dataset.

- [ ] **Step 4: Wire pseudo YAML into set split**

```python
if args.train_yaml:
    train_set_id, train_yaml = load_pseudo_train_yaml(args.train_yaml, args.eval_set)
else:
    train_set_id = args.train_set
    train_yaml = GT_FIXTURES[train_set_id]

train_ds = TrajectorySpanDataset([(train_set_id, train_yaml)], bin_s=args.bin_s)
if not train_ds.specs:
    raise ValueError(f"no trainable spans in {train_yaml}")
```

Use `train_set_id` in checkpoint naming. Eval remains `GT_FIXTURES[args.eval_set]`.

- [ ] **Step 5: Add a CLI smoke test with the dataset stubbed**

Monkeypatch `TrajectorySpanDataset`, `evaluate`, and the short training loop dependencies. Assert a pseudo YAML reaches the train dataset while BB11 fixture reaches eval, and same-set input fails before any feature access.

- [ ] **Step 6: Run trajectory tests**

Run:

```bash
venvs/audio/bin/python -m pytest \
  workspaces/alignment_prototype/trajectory/tests/ -q
```

Expected: all trajectory tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add workspaces/alignment_prototype/trajectory/train.py \
  workspaces/alignment_prototype/trajectory/tests/test_train_pseudo.py
git commit -m "feat(trajectory): train from leakage-guarded pseudo labels"
```

---

### Task 4: E1 orchestration command

**Files:**
- Create: `workspaces/alignment_prototype/trajectory/e1.py`
- Create: `workspaces/alignment_prototype/trajectory/tests/test_e1.py`
- Modify: `Makefile`

**Interfaces:**
- Adds CLI: `python -m workspaces.alignment_prototype.trajectory.e1`.
- Adds Make target: `make trm-e1`.
- Inputs: pool set ID, eval set ID, base timeline, synthetic root, output directory.
- Outputs: agentic timeline, pseudo YAML, subprocess logs, checkpoints, and `e1_result.json`.

- [ ] **Step 1: Write failing command-construction tests**

Test that full mode builds, in order: pseudo materialization, pseudo conv, synthetic-only TRM control, and pseudo TRM. Verify every learned run uses BB11 eval and only the pseudo runs use `--train-yaml`.

```python
def test_e1_commands_keep_pool_and_eval_disjoint(tmp_path):
    cfg = E1Config(
        pool_set="w1mgcjt",
        eval_set="2nvzlh2k",
        base_timeline=tmp_path / "base.json",
        synthetic_root=tmp_path / "synthetic",
        out_dir=tmp_path / "out",
    )
    commands = build_train_commands(cfg, tmp_path / "pseudo.yaml")
    assert all(["--eval-set", "2nvzlh2k"] == cmd[cmd.index("--eval-set"):][:2]
               for cmd in commands)
    assert sum("--train-yaml" in cmd for cmd in commands) == 2
    assert any("--synthetic-only" in cmd for cmd in commands)
```

- [ ] **Step 2: Run the E1 tests and verify RED**

Run:

```bash
venvs/audio/bin/python -m pytest \
  workspaces/alignment_prototype/trajectory/tests/test_e1.py -q
```

Expected: collection fails because `trajectory.e1` does not exist.

- [ ] **Step 3: Implement `E1Config` and preflight**

```python
@dataclass(frozen=True)
class E1Config:
    pool_set: str
    eval_set: str
    base_timeline: Path
    synthetic_root: Path
    out_dir: Path
    smoke_only: bool = False
    reuse_agentic: bool = False
```

Reject equal pool/eval IDs. Call `preflight_set(pool_set)`. Validate the base timeline set ID. Build `LiveContext.from_set(..., apply_fiber_gate=True)` and fail with the existing actionable missing-MERT message if unavailable.

- [ ] **Step 4: Implement unlabeled agentic artifact production**

Build `SpanCtx` rows from the base timeline using timeline/manifest stem claims,
call `refine_agentic_spans` with `Ladder(combine=True)` and
`pseudo_safe=True`, then write and validate
`<pool>_agentic_pseudo_timeline.json` with `finalize`. Add top-level
`pseudo_safety` metadata containing the exact ladder thresholds,
`combine: true`, and `fiber_gate: true`; the materializer copies this into YAML
provenance.

- [ ] **Step 5: Call materialization and stop cleanly on starvation**

Invoke `materialize_pseudo_gt` directly. If `accepted == 0`, write `e1_result.json` with status `"starved"` and return exit code `2`; do not launch training.

- [ ] **Step 6: Build and execute reproducible model commands**

Use `sys.executable -m workspaces.alignment_prototype.trajectory.train`. Run a two-epoch pseudo-TRM smoke first. Full mode then runs:

1. conv + Viterbi on pseudo-BB10 → BB11;
2. synthetic-only TRM → BB11 using `--synthetic-root`;
3. pseudo-BB10 TRM → BB11.

Stream output to the terminal and tee each command to a distinct log under `out/e1/`. Stop on the first non-zero subprocess exit while preserving prior logs and artifacts.

- [ ] **Step 7: Add CLI options and Make target**

CLI:

```text
--pool-set w1mgcjt
--eval-set 2nvzlh2k
--base-timeline PATH
--synthetic-root PATH
--out-dir PATH
--smoke-only
--reuse-agentic
```

Make target:

```make
trm-e1:
	venvs/audio/bin/python -m workspaces.alignment_prototype.trajectory.e1 \
		--pool-set $(POOL) --eval-set $(EVAL) \
		--base-timeline $(TIMELINE) --synthetic-root $(SYNTH)
```

- [ ] **Step 8: Test orchestration failure and resume behavior**

Stub expensive calls. Assert starvation launches no subprocess, `--reuse-agentic` skips probes but validates the artifact, failed training leaves pseudo YAML intact, and `e1_result.json` records command exit state without embedding hand-written headline metrics.

- [ ] **Step 9: Run all focused tests**

Run:

```bash
venvs/audio/bin/python -m pytest \
  tests/alignment_prototype/test_agentic.py \
  workspaces/alignment_prototype/trajectory/tests/ -q
```

Expected: all focused tests pass.

- [ ] **Step 10: Commit Task 4**

```bash
git add Makefile \
  workspaces/alignment_prototype/trajectory/e1.py \
  workspaces/alignment_prototype/trajectory/tests/test_e1.py
git commit -m "feat(trajectory): orchestrate the E1 pseudo-label experiment"
```

---

### Task 5: Verify and run E1

**Files:**
- Modify only after a completed experiment: `workspaces/alignment_prototype/attic/EXPERIMENTS.md`
- Generated and untracked: `workspaces/alignment_prototype/out/e1/**`

**Interfaces:**
- Consumes the completed code from Tasks 1–4 and local pulled/cached assets.
- Produces a reproducible runtime verdict and artifact paths.

- [ ] **Step 1: Run the full repository gate**

Run:

```bash
make check
```

Expected: guardrails, typecheck, and full tests pass.

- [ ] **Step 2: Verify local BB10 prerequisites without mutation**

Run:

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.trajectory.e1 \
  --pool-set w1mgcjt \
  --eval-set 2nvzlh2k \
  --base-timeline workspaces/alignment_prototype/out/w1mgcjt_predicted_timeline.json \
  --synthetic-root data/synthetic_mixes_v2 \
  --out-dir workspaces/alignment_prototype/out/e1 \
  --smoke-only
```

Expected: preflight names any missing local mix, manifest, MERT, fingerprint, or feature prerequisite; otherwise it materializes non-zero pseudo-label coverage and completes the smoke fit. Do not fetch or write canonical state automatically.

- [ ] **Step 3: Resolve only local/cache prerequisites**

If preflight reports a missing cache that can be exported read-only from pi-storage, use the existing export/cache command documented by the failing message. If it requires a canonical write or service restart, stop and report the blocker rather than mutating shared state.

- [ ] **Step 4: Run the full experiment**

Run the same command without `--smoke-only` and capture its generated `e1_result.json` plus logs. Expected terminal state is one of:

- `"completed"`: all raw, conv, synthetic-only TRM, and pseudo-TRM evaluations ran;
- `"starved"`: zero pseudo labels survived, no training ran;
- `"failed"`: a named stage exited non-zero and prior artifacts remain reusable.

- [ ] **Step 5: Record the verdict without duplicating metrics**

Append one concise ledger entry naming the commit, pool/eval split, artifact path, and verdict (`viable`, `starved`, or `noise-floor`). Do not copy metric values into the ledger; if the result changes canonical alignment status, regenerate `docs/alignment_status.md` through its scorer workflow.

- [ ] **Step 6: Run final verification**

Run:

```bash
make check
git status --short
```

Expected: `make check` passes; only the intended ledger change is tracked, while generated E1 artifacts remain ignored/untracked as designed.

- [ ] **Step 7: Commit the measured verdict**

```bash
git add workspaces/alignment_prototype/attic/EXPERIMENTS.md
git commit -m "docs(trm): record E1 pseudo-label verdict"
```

Do not create this commit if the experiment was blocked before producing a verdict; report the blocker instead.


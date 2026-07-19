# Fail-Closed Audio Resolvers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** stop alignment code from silently choosing the first slot-named audio file when multiple candidates exist.

**Architecture:** Preserve existing manifest paths and unambiguous post-pull fallback files. Central stem resolution and fused-reference resolution collect all candidates, accept exactly one, and warn/abstain on ambiguity; stem-routed callers propagate abstention rather than substituting the full track.

**Tech Stack:** Python 3, `pathlib`, standard-library `warnings`, pytest.

## Global Constraints

- Keep existing valid manifest-path behavior unchanged.
- Keep unambiguous post-pull fallback behavior unchanged.
- Warnings must identify `track_audio_id` and slot label.
- Do not add a stable-ID index or modify audio inventory in this patch.
- Do not fall through to full-track audio after an ambiguous stem lookup.

---

### Task 1: Make stem resolution unique-or-abstain

**Files:**
- Create: `tests/test_fail_closed_audio_resolvers.py`
- Modify: `workspaces/alignment_prototype/stem_resolve.py`
- Modify: `workspaces/alignment_prototype/refine_ref_offsets.py`
- Modify: `workspaces/alignment_prototype/joint_ref_decode.py`

**Interfaces:**
- Consumes: `resolve_stem(set_dir: Path | None, slot_label: str | None, track: dict | None, stem_name: str) -> Path | None`
- Produces: the same signature, with exactly-one fallback semantics and a `RuntimeWarning` on ambiguity.
- Produces: stem-routed callers return `None` when `resolve_stem` abstains.

- [ ] **Step 1: Write failing stem resolver and caller tests**

Create temporary valid manifest paths, one-candidate fallback trees, and
two-candidate fallback trees. Assert:

```python
assert resolve_stem(set_dir, "001w1", track, "vocals") == manifest_stem
assert resolve_stem(set_dir, "001w1", stale_track, "vocals") == unique_stem

with pytest.warns(RuntimeWarning, match=r"track_audio_id=42.*slot=001w1"):
    assert resolve_stem(set_dir, "001w1", stale_track, "vocals") is None

with pytest.warns(RuntimeWarning):
    assert ref_audio_for(span, stale_track, set_dir) is None

with pytest.warns(RuntimeWarning):
    assert _ref_audio_for(span, stale_track, set_dir) is None
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
venvs/audio/bin/python -m pytest tests/test_fail_closed_audio_resolvers.py -q
```

Expected: ambiguous stem resolution returns the first candidate, and both
stem-routed callers return the full `local_path`.

- [ ] **Step 3: Implement exactly-one stem fallback**

In `stem_resolve.py`, replace tagged-first iteration with candidate collection:

```python
candidates: dict[Path, Path] = {}
for slot in _slot_variants(slot_label):
    for directory in sorted(stems_root.glob(f"{slot}__*")):
        candidate = directory / f"{stem_name}.flac"
        if candidate.is_file():
            candidates[candidate.resolve()] = candidate

if len(candidates) == 1:
    return next(iter(candidates.values()))
if len(candidates) > 1:
    warnings.warn(
        "ambiguous stem fallback: "
        f"track_audio_id={(track or {}).get('track_audio_id', '?')} "
        f"slot={slot_label} stem={stem_name} candidates={len(candidates)}",
        RuntimeWarning,
        stacklevel=2,
    )
return None
```

Remove `_tagged_first`, update the module and function docstrings, and import
`warnings`.

- [ ] **Step 4: Propagate stem abstention**

In `refine_ref_offsets.ref_audio_for`, return the resolved `Path` or `None`
immediately for claimed acappella/instrumental spans:

```python
if stem_key:
    hit = resolve_stem(set_dir, span.get("slot_label"), track, stem_key)
    return hit
```

In `joint_ref_decode._ref_audio_for`, preserve its `str | None` return type:

```python
if stem_key:
    hit = resolve_stem(set_dir, span.get("slot_label"), track, stem_key)
    return str(hit) if hit is not None else None
```

Keep `local_path` fallback only for regular spans.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
venvs/audio/bin/python -m pytest tests/test_fail_closed_audio_resolvers.py -q
```

Expected: stem tests pass with no unexpected warnings.

### Task 2: Make fused reference resolution unique-or-abstain

**Files:**
- Modify: `tests/test_fail_closed_audio_resolvers.py`
- Modify: `workspaces/alignment_prototype/infer_fused.py`

**Interfaces:**
- Consumes: `_resolve_ref(track: dict, set_dir: Path) -> Path | None`
- Produces: the same signature, returning a fallback only for exactly one non-ASD file and warning on ambiguity.

- [ ] **Step 1: Write failing fused resolver tests**

Assert valid path and one-candidate behavior remain unchanged, then reproduce
the first-hit bug:

```python
assert _resolve_ref(track, set_dir) == manifest_ref
assert _resolve_ref(stale_track, set_dir) == unique_ref

with pytest.warns(RuntimeWarning, match=r"track_audio_id=42.*slot=001"):
    assert _resolve_ref(stale_track, set_dir) is None
```

- [ ] **Step 2: Run the fused tests and verify RED**

Run:

```bash
venvs/audio/bin/python -m pytest tests/test_fail_closed_audio_resolvers.py -q
```

Expected: `_resolve_ref` returns the lexicographically first ambiguous file.

- [ ] **Step 3: Implement exactly-one reference fallback**

Replace first-hit iteration with:

```python
candidates = [
    hit
    for hit in sorted((set_dir / "tracks").glob(f"{slot}__*"))
    if hit.suffix != ".asd" and hit.is_file()
]
if len(candidates) == 1:
    return candidates[0]
if len(candidates) > 1:
    warnings.warn(
        "ambiguous reference fallback: "
        f"track_audio_id={track.get('track_audio_id', '?')} "
        f"slot={slot} candidates={len(candidates)}",
        RuntimeWarning,
        stacklevel=2,
    )
return None
```

Update `_resolve_ref`'s docstring to state unique-or-abstain behavior.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
venvs/audio/bin/python -m pytest tests/test_fail_closed_audio_resolvers.py -q
```

Expected: all resolver tests pass.

### Task 3: Verify repository integration

**Files:**
- Verify only; no additional production files.

**Interfaces:**
- Consumes: completed resolver behavior from Tasks 1 and 2.
- Produces: a gate-clean branch ready for review.

- [ ] **Step 1: Run lint diagnostics on edited files**

Check:

```text
workspaces/alignment_prototype/stem_resolve.py
workspaces/alignment_prototype/refine_ref_offsets.py
workspaces/alignment_prototype/joint_ref_decode.py
workspaces/alignment_prototype/infer_fused.py
tests/test_fail_closed_audio_resolvers.py
```

Expected: no newly introduced diagnostics.

- [ ] **Step 2: Run the repository gate**

Run:

```bash
make check
```

Expected: guardrails, typecheck, and tests all pass.

- [ ] **Step 3: Review the final diff**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: only the approved spec, plan, resolver code, and resolver tests are
changed.

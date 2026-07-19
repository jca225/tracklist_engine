# BB12 Inventory + Audio Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BB12 (`1fsnxchk`) alignment inputs inventory-coherent — every GT/span recording either has the correct stem audio on disk + in `manifest.json` + (where required) on pi, or is an explicit labeled proxy/abstain — so the aligner stops training and scoring on silent misses, stem mis-routes, and colliding slot folders.

**Architecture:** Treat this as three layers that must agree: (1) **pi inventory** (`set_track_slots` + `track_audio`), (2) **Mac aligning replica** (`~/aligning/…/manifest.json` + `tracks/`/`stems/`), (3) **GT fixture** (`labeling/fixtures/bb12_ground_truth.yaml`). Add a durable `labeling/reconcile_aligning_manifest.py` (replaces one-off shell repairs), formalize the existing Lux/`mix_instrumental` proxy path, acquire or Demucs-promote the remaining wrong-stem payloads, remap drifted GT `slot_label`s to pi labels, then rebuild span table + spectrogram review and re-score BB12 so we can measure whether inventory debt was the silent killer.

**Tech Stack:** Python 3.12+, `venvs/audio/bin/python`, pi-storage SQLite over SSH (`labeling/pull_set_for_alignment.py` / `inventory_check.py`), pytest, existing ingest helpers (`scripts/ingest_stem_url.py`, `scripts/ingest_candidate_winners.py`, Demucs on Mac/Vast).

## Global Constraints

- **Set ids:** BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`. This plan is BB12-first; BB11 is a follow-on once the tooling exists.
- **Canonical DB is shared/live:** do not `--apply` non-idempotent migrations or delete `track_audio` rows without a dry-run + operator OK. Code + Mac `~/aligning/` edits are safe; pi writes go through existing ingest CLIs.
- **Baby rule:** one full mix file in `tracks/`; acappella/instrumental claims prefer Demucs `stems/{vocals,instrumental}` when a dedicated stem download does not exist — do not invent a second full file.
- **Annotator territory:** files/dirs tagged `[NNNbpm KK]` / `[no-features]` are never pruned. Manifest may *point at* them; pull `--prune` must not delete them.
- **Numbers:** post-repair alignment headline metrics go only in `docs/alignment_status.md` (regenerated). This plan cites inventory counts, not aligner SOTA.
- **Style:** `from __future__ import annotations`, typed, frozen dataclasses for records; I/O at edges.
- **Proxy policy (Lux Holm – Omega):** original recording does not exist for our purposes. Labeling uses the set’s `mix_instrumental.flac` (already at `~/aligning/1fsnxchk__…/mix_instrumental.flac`) as host/proxy, with `unalignable: true` + `source_note` (pattern already in `bb12_ground_truth.yaml` for “Lux x Spaceman”). Do **not** invent a fake `track_audio` that pretends to be Lux Holm.

## Why this likely hurts the aligner

| Failure mode | What the model sees | BB12 scale (2026-07-19 audit) |
|---|---|---|
| Manifest missing GT `track_id` | EDA/aligner “audio not found” or wrong slot fallback | 6 wired (Heiress class); pattern may recur after re-pull |
| Manifest `stem=regular` while claim is acap/instr | Router loads full mix; Demucs channel ignored | 25 stem-field rewrites done locally; pi still `wrong_stem` for ~20 payloads |
| Pi `missing` / no `track_audio` | Unresolved stub; filesystem slot collision → **wrong song** | 8 missing; 7 had disk orphans; **032 Lux** is the true empty |
| GT `slot_label` drift (old ALS numbers) | Humans + slot-keyed tools look at wrong folders | ~155 GT rows; resolve-by-`track_id` works, slot UX lies |
| Span `stem_mismatch` | Aligner *routed* regular for instrumental/acap GT | 62 in stale span_table (mix of inventory + algo) |

Hypothesis to validate in Task 8: after inventory coherence, BB12 identity stays high and **traj / stem-mismatch / “src=no”** drop on a fresh score — if they do not, the residual is algo, not data.

## File map

| Path | Role |
|---|---|
| `labeling/reconcile_aligning_manifest.py` | **New.** Idempotent Mac-side manifest repair from pi slots + disk |
| `labeling/inventory_check.py` | Already evaluates satisfaction; reuse, do not fork |
| `labeling/pull_set_for_alignment.py` | Source of `fetch_tracks` / rsync; call, don’t reimplement |
| `eda/alignment/spectrogram_review/source_audio.py` | Fail-closed on unresolved stubs (partially done) |
| `labeling/fixtures/bb12_ground_truth.yaml` | Slot remap + Lux proxy notes |
| `labeling/export_als_to_gt.py` | Already encodes `mix_instrumental` placeholder notes |
| `scripts/reconcile_gt_inventory.py` | GT → action CSV (existing) |
| `docs/inventory_coherence_contract.md` | Update with proxy + reconcile CLI |
| `tests/labeling/test_reconcile_aligning_manifest.py` | **New** |
| `tests/eda/test_spectrogram_source_audio.py` | Extend fail-closed coverage |

---

### Task 1: Durable `reconcile_aligning_manifest` CLI

Replace one-off repair scripts with an idempotent tool that: (a) loads pi `set_track_slots` + `fetch_tracks`, (b) ensures every slot has a manifest row, (c) points `local_path`/`stems` at existing disk files (name-disambiguated), (d) sets `stem` from pi `claimed_stem` when Demucs or native stem files exist, (e) leaves explicit unresolved stubs for true empties (no slot filesystem guess).

**Files:**
- Create: `labeling/reconcile_aligning_manifest.py`
- Create: `tests/labeling/test_reconcile_aligning_manifest.py`
- Modify: `docs/inventory_coherence_contract.md` (Tools table)

**Interfaces:**
- Consumes: `pull_set_for_alignment.fetch_tracks`, `pull_set_for_alignment.ssh_sqlite`, `inventory_check.evaluate_set_inventory` / `satisfaction_to_manifest_fields`
- Produces: `reconcile_manifest(set_dir: Path, *, dry_run: bool = True) -> ReconcileReport` with fields `added: list[str]`, `stem_rewrites: list[tuple[str,str,str]]`, `wired: list[str]`, `unresolved: list[str]`, `backed_up: Path | None`

- [ ] **Step 1: Write failing tests for report shape + fail-closed stub**

```python
# tests/labeling/test_reconcile_aligning_manifest.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from labeling.reconcile_aligning_manifest import reconcile_manifest


def _touch_audio(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(1000, dtype=np.float32), 22050)


def test_wires_missing_row_from_disk(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    (set_dir / "stems").mkdir()
    _touch_audio(set_dir / "tracks" / "037__Dune - Heiress.m4a")
    _touch_audio(set_dir / "stems" / "037__Dune - Heiress" / "instrumental.flac")
    man = {
        "set_id": "1fsnxchk",
        "title": "t",
        "mix_local_path": str(set_dir / "mix.m4a"),
        "tracks": [],
    }
    _touch_audio(set_dir / "mix.m4a")
    (set_dir / "manifest.json").write_text(json.dumps(man))

    def fake_slots(_sql: str):
        return [
            {
                "slot_label": "037",
                "recording_id": "94tc2y5",
                "claimed_stem": "instrumental",
                "claimed_variant": "regular",
                "name": "Dune - Heiress",
            }
        ]

    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.ssh_sqlite", fake_slots
    )
    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.fetch_tracks",
        lambda _sid: [],
    )

    report = reconcile_manifest(set_dir, dry_run=False)
    assert "037" in report.wired or "037" in report.added
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "94tc2y5")
    assert row["stem"] == "instrumental"
    assert Path(row["stems"]["instrumental"]).is_file()


def test_unresolved_slot_stays_pathless(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    # colliding orphan — must NOT be wired to Lux
    _touch_audio(set_dir / "tracks" / "032__AFROJACK - Ten Feet Tall.wav")
    (set_dir / "manifest.json").write_text(
        json.dumps({"set_id": "1fsnxchk", "tracks": []})
    )

    def fake_slots(_sql: str):
        return [
            {
                "slot_label": "032",
                "recording_id": "tlp2853023",
                "claimed_stem": "regular",
                "claimed_variant": "regular",
                "name": "Lux Holm - Omega",
            }
        ]

    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.ssh_sqlite", fake_slots
    )
    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.fetch_tracks",
        lambda _sid: [],
    )
    # name tokens for Lux must not match AFROJACK
    report = reconcile_manifest(set_dir, dry_run=False)
    assert "032" in report.unresolved
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "tlp2853023")
    assert row.get("local_path") in (None, "")
```

- [ ] **Step 2: Run tests — expect FAIL (module missing)**

```bash
venvs/audio/bin/python -m pytest tests/labeling/test_reconcile_aligning_manifest.py -v
```

Expected: `ModuleNotFoundError` or import error for `labeling.reconcile_aligning_manifest`.

- [ ] **Step 3: Implement `reconcile_manifest`**

Create `labeling/reconcile_aligning_manifest.py` with:

- CLI: `python -m labeling.reconcile_aligning_manifest <set_id> [--dest ~/aligning] [--apply]`
- Default **dry-run** (print report only); `--apply` writes after backup `manifest.json.bak_reconcile_<utc>`
- Name-scoring: require ≥1 strong title token match before wiring a colliding `NNN__*` file (Lux must not bind AFROJACK)
- Special-case registry (module-level dict) for known proxy homes, e.g. `029w1` → prefer `101__…I Just Had Sex (Acappella)…` — keep small and documented

- [ ] **Step 4: Run tests — expect PASS**

```bash
venvs/audio/bin/python -m pytest tests/labeling/test_reconcile_aligning_manifest.py -v
```

- [ ] **Step 5: Apply once on BB12 (Mac)**

```bash
venvs/audio/bin/python -m labeling.reconcile_aligning_manifest 1fsnxchk --apply
venvs/audio/bin/python labeling/pull_set_for_alignment.py 1fsnxchk --check 2>&1 | head -5
```

Expected: dry inventory still shows pi `blocking` (canonical unchanged); manifest has no silent pathless GT ids except Lux stub + any new true empties.

- [ ] **Step 6: Commit**

```bash
git add labeling/reconcile_aligning_manifest.py tests/labeling/test_reconcile_aligning_manifest.py docs/inventory_coherence_contract.md
git commit -m "$(cat <<'EOF'
feat(labeling): reconcile aligning manifest from pi slots + disk

Idempotent Mac-side repair so GT recording ids and claimed stems are
first-class in manifest.json without a destructive full re-pull.
EOF
)"
```

---

### Task 2: Lux Holm – Omega → `mix_instrumental` proxy (labeling contract)

Formalize what the annotator already did for “Lux x Spaceman”: the host bed is `mix_instrumental.flac`, marked unalignable / proxy, not a fake Lux download.

**Files:**
- Modify: `labeling/reconcile_aligning_manifest.py` (proxy registry)
- Modify: `labeling/export_als_to_gt.py` (ensure `_PLACEHOLDER_RE` + notes stay the SSOT)
- Modify: `labeling/fixtures/bb12_ground_truth.yaml` only if re-export changes notes
- Modify: `docs/inventory_coherence_contract.md` — add **Proxy / unalignable** section
- Test: `tests/labeling/test_export_als_placeholder.py` (create if missing) or extend existing export tests

**Interfaces:**
- Produces: manifest row for slot `032` / `tlp2853023` with  
  `local_path = <set_dir>/mix_instrumental.flac`,  
  `stem = "instrumental"`,  
  `gap = "proxy:mix_instrumental — original unavailable (Lux Holm - Omega)"`,  
  `satisfaction = "fallback"`,  
  and GT rows continue to use `unalignable: true` + `source_note` containing `mix_instrumental` / Lux.

- [ ] **Step 1: Failing test — reconcile wires Lux to mix_instrumental, never AFROJACK**

```python
def test_lux_omega_proxy_uses_mix_instrumental(tmp_path: Path, monkeypatch):
    set_dir = tmp_path / "1fsnxchk__x"
    (set_dir / "tracks").mkdir(parents=True)
    _touch_audio(set_dir / "mix_instrumental.flac")
    _touch_audio(set_dir / "tracks" / "032__AFROJACK - Ten Feet Tall.wav")
    (set_dir / "manifest.json").write_text(json.dumps({"set_id": "1fsnxchk", "tracks": []}))

    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.ssh_sqlite",
        lambda _sql: [{
            "slot_label": "032",
            "recording_id": "tlp2853023",
            "claimed_stem": "regular",  # scrape may say regular; proxy is instrumental channel
            "claimed_variant": "regular",
            "name": "Lux Holm - Omega",
        }],
    )
    monkeypatch.setattr(
        "labeling.reconcile_aligning_manifest.fetch_tracks", lambda _s: []
    )
    from labeling.reconcile_aligning_manifest import PROXY_SLOT_AUDIO, reconcile_manifest

    assert PROXY_SLOT_AUDIO[("1fsnxchk", "032")] == "mix_instrumental.flac"
    report = reconcile_manifest(set_dir, dry_run=False)
    doc = json.loads((set_dir / "manifest.json").read_text())
    row = next(t for t in doc["tracks"] if t["track_id"] == "tlp2853023")
    assert row["local_path"].endswith("mix_instrumental.flac")
    assert "proxy:mix_instrumental" in (row.get("gap") or "")
    assert "032" not in report.unresolved
```

- [ ] **Step 2: Run test — FAIL until `PROXY_SLOT_AUDIO` exists**

- [ ] **Step 3: Implement proxy registry**

In `labeling/reconcile_aligning_manifest.py`:

```python
# (set_id, slot_label) -> filename under set_dir (not under tracks/)
PROXY_SLOT_AUDIO: dict[tuple[str, str], str] = {
    ("1fsnxchk", "032"): "mix_instrumental.flac",  # Lux Holm - Omega — no commercial release
}
```

When wiring that slot: set `local_path` to `set_dir / PROXY_SLOT_AUDIO[...]`, `stem="instrumental"`, do not rsync from pi, do not create a fake `track_audio` on pi in this task.

- [ ] **Step 4: ALS / GT labeling note**

Document in `docs/inventory_coherence_contract.md`:

- Ableton clip for Lux bed should reference `mix_instrumental.flac` (already supported by `export_als_to_gt._PLACEHOLDER_RE`).
- Scorer / training must honor `unalignable: true` / `skip_training` (already in `anchor_check` / schema) — **do not** score Lux as a normal identity+traj span.
- If current ALS still points at a missing Lux file, human relinks once to `mix_instrumental.flac` and re-exports GT.

- [ ] **Step 5: Apply reconcile on BB12; verify resolve**

```bash
venvs/audio/bin/python -m labeling.reconcile_aligning_manifest 1fsnxchk --apply
venvs/audio/bin/python - <<'PY'
from eda.alignment.spectrogram_review.source_audio import load_manifest_tracks, resolve_source_audio
idx = load_manifest_tracks("1fsnxchk")
p = resolve_source_audio("1fsnxchk", "tlp2853023", gt_stem="instrumental", tracks=idx, slot="032", name="Lux Holm - Omega")
assert p is not None and p.name == "mix_instrumental.flac", p
print("OK", p)
PY
```

- [ ] **Step 6: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(labeling): Lux Holm Omega uses mix_instrumental proxy

Original is unavailable; aligning manifest + GT contract point at the
set mix_instrumental instead of a colliding slot folder.
EOF
)"
```

---

### Task 3: Harden source-audio resolver (fail closed) + aligner preflight

Ensure no consumer silently plays the wrong `NNN__*` file when the manifest stub is unresolved.

**Files:**
- Modify: `eda/alignment/spectrogram_review/source_audio.py` (already partially done — land + extend)
- Modify: `tests/eda/test_spectrogram_source_audio.py`
- Modify: aligner entry that loads refs (search `resolve_source_audio` / `find_aligning_dir` / `_resolve_ref_audio` in `workspaces/alignment_prototype/`) — add preflight that lists unresolved GT ids before a run

**Interfaces:**
- `resolve_source_audio`: if `tracks[recording_id]` exists and paths are dead → `None` (no slot FS fallback)
- New: `preflight_aligning_audio(set_id: str, gt_tracks: list[dict]) -> list[str]` returning recording_ids with no resolvable audio (excluding `unalignable` / `skip_training`)

- [ ] **Step 1: Confirm / extend unit tests** (unresolved stub + Heiress disambiguation already sketched)

- [ ] **Step 2: Wire preflight into `score_timeline_vs_gt.py` and the main BB12 predict entry** so a run prints:

```text
[preflight] 0 missing ref audio (excluding unalignable)
```

or aborts with `--strict-inventory` when `>0`.

- [ ] **Step 3: pytest + commit**

```bash
venvs/audio/bin/python -m pytest tests/eda/test_spectrogram_source_audio.py -v
git commit -m "fix(eda): fail closed on unresolved manifest stubs; aligner audio preflight"
```

---

### Task 4: Pi-side wrong_stem payloads — Demucs promote or acquire

Clear the **~20 blocking `wrong_stem`** acappella claims that only have `regular` on pi. Prefer baby-rule Demucs vocals already on the Mac aligning tree; promote winners into `track_audio` with `stem=acappella` via existing ingest scripts so the next pull is satisfied.

**Files:**
- Use (do not rewrite): `scripts/ingest_stem_url.py`, `scripts/ingest_candidate_winners.py`, `scripts/log_acquisition.py`
- Optional helper: `scripts/promote_aligning_demucs_stem.py` **only if** no existing path covers “local flac → pi object + track_audio row”

**Worklist source (BB12, 2026-07-19):** slots with local `stems/<slot>__*/vocals.flac` and pi `wrong_stem` needing acappella, including:  
`004w1, 004w3, 005w4, 011w2, 014w2, 014w3, 015w1, 017w2, 020w1, 020w3, 023w3, 035w2, 042w3`  
(plus “prefer acappella” payloads currently claimed `regular` if GT says acappella — reconcile claim vs GT in Task 5 before promoting).

- [ ] **Step 1: Emit CSV worklist**

```bash
venvs/audio/bin/python labeling/pull_set_for_alignment.py 1fsnxchk --check > /tmp/bb12_inv.txt
# parse WRONG_STEM + MISSING into labeling/fixtures/bb12_inventory_worklist.csv
# columns: slot_label,recording_id,claimed_stem,status,local_vocals,local_track,action
```

`action` ∈ `{promote_demucs_vocals, acquire_acappella, proxy_mix_instrumental, remap_gt_slot}`.

- [ ] **Step 2: For each `promote_demucs_vocals` row (dry-run first)**

1. Verify local `vocals.flac` NCC-agrees with any `candidates/vocals/cand*` if present (spot-check; skip if no candidate).
2. Upload/ingest as `stem=acappella` using the repo’s existing ingest entrypoint (same pattern as `.claude/skills/replace-track-audio`).
3. Set `is_reference=1` on the new row when it is the slot’s claimed stem.
4. Re-run `--check` for that slot → expect `satisfied`.

- [ ] **Step 3: For rows with no usable Demucs / wrong content**

Open acquisition cases (Phase-1 data engine) with `problem=missing_acappella` / `wrong_stem`; do not block the rest of this plan.

- [ ] **Step 4: Mac delta pull (no prune unless dry-run clean)**

```bash
venvs/audio/bin/python labeling/pull_set_for_alignment.py 1fsnxchk
venvs/audio/bin/python -m labeling.reconcile_aligning_manifest 1fsnxchk --apply
venvs/audio/bin/python labeling/pull_set_for_alignment.py 1fsnxchk --check
```

Target: `blocking: 0` for BB12, or only acquisition-queued rows left.

- [ ] **Step 5: Commit worklist + any new script; do not commit `~/aligning`**

---

### Task 5: GT `slot_label` remap to pi tracklist labels

Stop the dual numbering (`076` vs `022`, `132` vs `037`). Remap by `track_id` (+ `claimed_stem` when one id appears twice).

**Files:**
- Create: `labeling/remap_gt_slot_labels.py`
- Modify: `labeling/fixtures/bb12_ground_truth.yaml` (via script write)
- Test: `tests/labeling/test_remap_gt_slot_labels.py`

**Interfaces:**
- `remap_slots(gt_doc: dict, pi_slots: list[dict]) -> tuple[dict, list[RemapEvent]]`
- Collision rule: if one `track_id` maps to multiple pi slots, match on `claimed_stem`, then on closest `set_start_s` vs scrape order only if still ambiguous — else leave unchanged and emit `needs_human`.

- [ ] **Step 1: Failing test — Heiress `132` → `037`, Saints `076` → `022`**

```python
def test_remap_heiress_and_saints():
    gt = {"tracks": [
        {"track": "Heiress", "slot_label": "132", "track_id": "94tc2y5", "claimed_stem": "instrumental"},
        {"track": "Saints", "slot_label": "076", "track_id": "1r8f4fc5", "claimed_stem": "instrumental"},
    ]}
    pi = [
        {"slot_label": "037", "recording_id": "94tc2y5", "claimed_stem": "instrumental"},
        {"slot_label": "022", "recording_id": "1r8f4fc5", "claimed_stem": "instrumental"},
    ]
    from labeling.remap_gt_slot_labels import remap_slots
    out, events = remap_slots(gt, pi)
    assert out["tracks"][0]["slot_label"] == "037"
    assert out["tracks"][1]["slot_label"] == "022"
    assert len(events) == 2
```

- [ ] **Step 2: Implement + dry-run on fixture**

```bash
venvs/audio/bin/python -m labeling.remap_gt_slot_labels \
  --yaml labeling/fixtures/bb12_ground_truth.yaml --set-id 1fsnxchk --dry-run
```

Expected: ~150 remaps, `needs_human: 0` (or a short printed list).

- [ ] **Step 3: `--apply`**, run `make gt-gate SET=1fsnxchk` (or project’s GT gate) before write-back.

- [ ] **Step 4: Commit fixture + tool**

```bash
git commit -m "fix(labeling): remap BB12 GT slot_labels to pi tracklist labels"
```

---

### Task 6: Purge / quarantine colliding slot folders (Mac hygiene)

After remap + reconcile, `tracks/037__Blackout` next to `tracks/037__Heiress` is debt. Do **not** delete annotator-tagged dirs blindly.

**Files:**
- Extend: `labeling/reconcile_aligning_manifest.py` with `--report-collisions`
- Optional: `labeling/quarantine_aligning_orphans.py` moves non-manifest `tracks|stems/NNN__*` into `~/aligning/<set>/_orphan_slot_collisions/` (never `--prune` tagged files)

- [ ] **Step 1: Report collisions for BB12**

```bash
venvs/audio/bin/python -m labeling.reconcile_aligning_manifest 1fsnxchk --report-collisions
```

- [ ] **Step 2: Quarantine only untagged orphans whose names do not match any manifest `local_path` / stems path** (dry-run then apply).

- [ ] **Step 3: Re-run source_audio resolve over all GT rows — expect 0 wrong-folder picks; Lux → mix_instrumental.**

---

### Task 7: Rebuild evaluation surfaces

- [ ] **Step 1: Rebuild span table** from the **current** BB12 timeline you care about (agentic or classical — pick one and record SHA in the commit message), not the Jul 18 stale CSV.

```bash
# use the repo’s existing failure_analysis entrypoint, e.g.:
venvs/audio/bin/python -m eda.alignment.failure_analysis.build_span_table \
  --set-id 1fsnxchk \
  --timeline workspaces/alignment_prototype/out/1fsnxchk_agentic_timeline.json \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --out eda/alignment/failure_analysis/out/span_table.csv
```

(Adjust flags to match the actual CLI in `build_span_table.py`.)

- [ ] **Step 2: Re-render spectrogram review**

```bash
venvs/audio/bin/python -m eda.alignment.spectrogram_review.render \
  --set-id 1fsnxchk --outcome all --limit 0 \
  --out-dir eda/alignment/spectrogram_review/out/1fsnxchk_all
```

Expect: `src=yes` for all non-unalignable cards; Lux card absent or explicitly proxy-labeled.

- [ ] **Step 3: Record stem_mismatch + src-miss counts** in the PR description (not in `alignment_status.md` until Task 8 regenerates it).

---

### Task 8: Re-score BB12 and accept/reject the hypothesis

- [ ] **Step 1: Run the same classical + agentic scorers used for the status board** on BB12 only, with strict inventory preflight on.

- [ ] **Step 2: Diff vs previous status board** — identity, traj_strict, stem_mismatch rate, placement err. If traj barely moves, inventory was not the main blocker; park further inventory polish and return to algo. If stem_mismatch and decode-residual collapse, keep BB11 pass next.

- [ ] **Step 3: Regenerate `docs/alignment_status.md` via the project’s scorer stamp path** (never hand-edit headline numbers).

- [ ] **Step 4: EXPERIMENTS ledger note** in `workspaces/alignment_prototype/attic/EXPERIMENTS.md`:  
  `BB12 inventory coherence (2026-07-19) — verdict: <helps | neutral | insufficient>`.

---

### Task 9: BB11 pass (same tooling, smaller write-up)

- [ ] Run `reconcile_aligning_manifest` + `--check` on `2nvzlh2k`
- [ ] Remap GT slots if drift present
- [ ] Only open acquisition for blocking gaps that appear in GT/spans
- [ ] Re-score; update status doc once

---

## Out of scope (explicit)

- Fixing Ableton gain-curve long-ramp export bugs (Saints 88s ramp) — separate ALS export ticket
- Training a new model architecture
- Deleting pi `track_audio` rows for superseded Spotify rips without a replace skill run
- Changing scrape tokenizer claims en masse (Task 4 may fix individual claim/GT disagreements only when evidence is clear)

## Success criteria

1. `make check-inventory SET=1fsnxchk` → **0 blocking** (or only acquisition-queued with open cases).
2. Manifest: every GT `track_id` present; `stem` matches pi `claimed_stem` **or** documented proxy gap.
3. `resolve_source_audio` over all non-unalignable GT rows → **0 None**; Lux resolves to `mix_instrumental.flac`.
4. No unresolved stub returns a colliding slot file.
5. Fresh span_table + spectrogram review show **0 `src=no`** for BB12 playable spans.
6. Hypothesis accepted/rejected with a status-doc regeneration + EXPERIMENTS ledger line.

## Execution order

```text
Task 1 (reconcile CLI) → Task 2 (Lux proxy) → Task 3 (fail-closed + preflight)
    → Task 5 (GT slot remap) → Task 4 (pi promote/acquire) → Task 6 (quarantine)
    → Task 7 (rebuild EDA) → Task 8 (re-score) → Task 9 (BB11)
```

Task 5 before Task 4 reduces promoting the wrong stem for a mis-keyed human path. Task 2 can parallelize with Task 1 once the registry hook exists.

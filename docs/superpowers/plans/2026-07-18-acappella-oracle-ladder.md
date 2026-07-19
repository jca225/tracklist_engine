# Acappella Oracle Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an eval harness that attributes the acappella oracle→e2e trajectory gap to {routing, identity, placement} and isolates the decode-instance headroom, to decide whether a learned instance selector is worth building — without training anything.

**Architecture:** A layered oracle-substitution ladder. For each set (BB11 `2nvzlh2k`, BB12 `1fsnxchk`) and each rung R0→R3, the harness transforms the acappella spans of the current `_lt` timeline (or synthesizes them from GT), re-decodes via the existing `joint_ref_decode --decoder looptrace` (decoder held fixed across rungs), and scores every rung against a **fixed GT-acappella-row denominator** using the scorer's own `trajectory_acc`. Lifts between rungs attribute the gap; the strict→fiber gap at R0 and R3 is the instance-selection headroom.

**Tech Stack:** Python 3.14, `venvs/audio/bin/python`, pytest. Reuses `joint_ref_decode.main`, `path_decode.trajectory_acc` / `_pred_segs_from_span`, `ref_fibers.compute_fibers`, `score_timeline_vs_gt.norm_slot`. No new decode/model code.

## Global Constraints

- **Decoder held fixed at `--decoder looptrace`** across ALL rungs (matches the `_lt` scorecard source of truth). Never mix decoders within a ladder run — lifts must reflect oracle-input substitution only.
- **Fixed denominator = GT-acappella rows.** Every rung's acappella traj is the mean over the *same* set of GT-acap rows (a row with no decodable span scores 0). Never aggregate over predicted spans (that drops mis-identified spans and inflates R0).
- **Oracle substitution is eval-only.** No production timeline/routing/ingest/identity change. Do not edit `joint_ref_decode.py`, `harness/axes.py`, or anything under `workspaces/pws_aligner/**`.
- **Axis rule:** `claimed_stem` for scoring comes from the GT row, never the timeline span.
- **n=2 → report per-set** (BB11, BB12 separately). Never pool into a cross-set CI.
- **Set ids:** BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`.
- Run from repo root (worktree root) with `venvs/audio/bin/python`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Rung timeline builder (pure, slot-matched substitution)

**Files:**
- Create: `workspaces/alignment_prototype/evals/oracle_ladder.py`
- Test: `tests/alignment_prototype/test_oracle_ladder.py`

**Interfaces:**
- Consumes: `score_timeline_vs_gt.norm_slot` (slot normalization for matching).
- Produces:
  - `RUNGS: tuple[str, ...] = ("R0", "R1", "R2", "R3")`
  - `build_rung_timeline(rung: str, r0_spans: list[dict], gt_acap_rows: list[dict]) -> list[dict]` — returns the acappella input-span list for `rung`. Each returned span carries at least `slot_label, recording_id, set_start_s, set_end_s, claimed_stem, ref_start_s` (ref_start_s=0.0 placeholder; `joint_ref_decode` reads it before overwriting). Matching r0_spans↔gt rows is by `norm_slot(slot_label)`.
    - R0: matched r0 spans unchanged (only those whose slot matches a gt_acap row).
    - R1: matched r0 spans, `claimed_stem="acappella"` (predicted recording_id + placement retained).
    - R2: matched r0 spans, `claimed_stem="acappella"`, `recording_id = gt.track_id` (predicted placement retained; unmatched gt rows omitted — no placement to decode).
    - R3: ALL gt_acap rows → one span each: `recording_id = gt.track_id`, `claimed_stem="acappella"`, `set_start_s = gt.set_start_s`, `set_end_s = gt.set_end_s` (full oracle identity + placement).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignment_prototype/test_oracle_ladder.py
from workspaces.alignment_prototype.evals import oracle_ladder as ol


def _r0():
    return [
        # matched, mis-identified, stale stem
        {"slot_label": "1w1", "recording_id": "PRED_A", "set_start_s": 10.0,
         "set_end_s": 30.0, "claimed_stem": "regular", "ref_start_s": 5.0},
        # matched, correctly identified already
        {"slot_label": "2w2", "recording_id": "GT_B", "set_start_s": 50.0,
         "set_end_s": 70.0, "claimed_stem": "acappella", "ref_start_s": 0.0},
    ]


def _gt():
    return [
        {"slot_label": "1w1", "track_id": "GT_A", "claimed_stem": "acappella",
         "set_start_s": 12.0, "set_end_s": 31.0},
        {"slot_label": "2w2", "track_id": "GT_B", "claimed_stem": "acappella",
         "set_start_s": 49.0, "set_end_s": 69.0},
        {"slot_label": "9w9", "track_id": "GT_C", "claimed_stem": "acappella",
         "set_start_s": 90.0, "set_end_s": 110.0},  # never-matched in R0
    ]


def test_r0_keeps_matched_spans_unchanged():
    out = ol.build_rung_timeline("R0", _r0(), _gt())
    slots = sorted(s["slot_label"] for s in out)
    assert slots == ["1w1", "2w2"]  # 9w9 has no r0 span
    a = next(s for s in out if s["slot_label"] == "1w1")
    assert a["recording_id"] == "PRED_A" and a["claimed_stem"] == "regular"


def test_r1_forces_acappella_routing_only():
    out = ol.build_rung_timeline("R1", _r0(), _gt())
    a = next(s for s in out if s["slot_label"] == "1w1")
    assert a["claimed_stem"] == "acappella"       # routing fixed
    assert a["recording_id"] == "PRED_A"          # identity NOT fixed
    assert a["set_start_s"] == 10.0               # placement NOT fixed


def test_r2_fixes_identity_keeps_predicted_placement():
    out = ol.build_rung_timeline("R2", _r0(), _gt())
    a = next(s for s in out if s["slot_label"] == "1w1")
    assert a["recording_id"] == "GT_A"            # identity fixed to GT
    assert a["claimed_stem"] == "acappella"
    assert a["set_start_s"] == 10.0               # predicted placement retained
    assert all(s["slot_label"] != "9w9" for s in out)  # never-matched omitted


def test_r3_full_oracle_covers_all_gt_rows():
    out = ol.build_rung_timeline("R3", _r0(), _gt())
    slots = sorted(s["slot_label"] for s in out)
    assert slots == ["1w1", "2w2", "9w9"]         # includes never-matched
    c = next(s for s in out if s["slot_label"] == "9w9")
    assert c["recording_id"] == "GT_C" and c["set_start_s"] == 90.0
    assert c["set_end_s"] == 110.0 and c["claimed_stem"] == "acappella"
    assert "ref_start_s" in c                      # placeholder present for decoder
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_oracle_ladder.py -q`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError: build_rung_timeline`).

- [ ] **Step 3: Write minimal implementation**

```python
# workspaces/alignment_prototype/evals/oracle_ladder.py
"""Acappella oracle→e2e gap decomposition ("oracle ladder").

Attributes the acappella trajectory gap between end-to-end and oracle-placement
to {routing, identity, placement} by re-decoding the SAME acappella population
under progressively more oracle inputs, decoder held fixed at looptrace, scored
against a fixed GT-acappella-row denominator. Design + decision rule:
docs/superpowers/specs/2026-07-18-acappella-oracle-ladder-design.md.
"""
from __future__ import annotations

from workspaces.alignment_prototype.score_timeline_vs_gt import norm_slot

RUNGS: tuple[str, ...] = ("R0", "R1", "R2", "R3")


def _by_slot(spans: list[dict]) -> dict[str, dict]:
    return {norm_slot(str(s["slot_label"])): s for s in spans}


def build_rung_timeline(
    rung: str, r0_spans: list[dict], gt_acap_rows: list[dict]
) -> list[dict]:
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r} (expected one of {RUNGS})")
    r0_by_slot = _by_slot(r0_spans)
    out: list[dict] = []
    for g in gt_acap_rows:
        slot = norm_slot(str(g["slot_label"]))
        p = r0_by_slot.get(slot)
        if rung == "R3":
            out.append(
                {
                    "slot_label": str(g["slot_label"]),
                    "recording_id": str(g["track_id"]),
                    "claimed_stem": "acappella",
                    "set_start_s": float(g["set_start_s"]),
                    "set_end_s": float(g["set_end_s"]),
                    "ref_start_s": 0.0,
                    "ref_end_s": 0.0,
                }
            )
            continue
        if p is None:
            continue  # R0/R1/R2: no predicted placement -> row scores 0 downstream
        s = dict(p)
        if rung in ("R1", "R2"):
            s["claimed_stem"] = "acappella"
        if rung == "R2":
            s["recording_id"] = str(g["track_id"])
        s.setdefault("ref_start_s", 0.0)
        out.append(s)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_oracle_ladder.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/evals/oracle_ladder.py tests/alignment_prototype/test_oracle_ladder.py
git commit -m "feat(evals): oracle-ladder rung timeline builder (slot-matched substitution)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: GT-row-centric acappella aggregation (fixed denominator)

**Files:**
- Modify: `workspaces/alignment_prototype/evals/oracle_ladder.py`
- Test: `tests/alignment_prototype/test_oracle_ladder.py`

**Interfaces:**
- Consumes: an injected `score_fn(span: dict, gt_row: dict) -> tuple[float, float]` returning `(strict, fiber)` in [0,1] — dependency-injected so aggregation is testable without audio. (Task 3 supplies the real one built on `trajectory_acc`.)
- Produces:
  - `summarize_acappella(decoded_spans: list[dict], gt_acap_rows: list[dict], score_fn) -> dict` with keys: `n` (len gt_acap_rows), `n_scored` (rows with a matched decoded span), `strict` (mean over ALL gt rows, missing→0), `fiber` (same). Matching decoded_spans↔gt rows by `norm_slot`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignment_prototype/test_oracle_ladder.py
def test_summarize_fixed_denominator_missing_scores_zero():
    gt = _gt()  # 3 rows
    decoded = [
        {"slot_label": "1w1", "recording_id": "GT_A"},
        {"slot_label": "2w2", "recording_id": "GT_B"},
        # 9w9 absent -> must count as 0 in BOTH strict and fiber
    ]
    scores = {"1w1": (1.0, 1.0), "2w2": (0.0, 0.5)}

    def score_fn(span, gt_row):
        return scores[ol.norm_slot(str(span["slot_label"]))]

    out = ol.summarize_acappella(decoded, gt, score_fn)
    assert out["n"] == 3 and out["n_scored"] == 2
    assert abs(out["strict"] - (1.0 + 0.0 + 0.0) / 3) < 1e-9
    assert abs(out["fiber"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_oracle_ladder.py::test_summarize_fixed_denominator_missing_scores_zero -q`
Expected: FAIL (`AttributeError: summarize_acappella`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to workspaces/alignment_prototype/evals/oracle_ladder.py
from typing import Callable  # (add to imports at top)


def summarize_acappella(
    decoded_spans: list[dict],
    gt_acap_rows: list[dict],
    score_fn: "Callable[[dict, dict], tuple[float, float]]",
) -> dict:
    dec_by_slot = _by_slot(decoded_spans)
    n = len(gt_acap_rows)
    n_scored = 0
    strict_sum = 0.0
    fiber_sum = 0.0
    for g in gt_acap_rows:
        slot = norm_slot(str(g["slot_label"]))
        span = dec_by_slot.get(slot)
        if span is None:
            continue  # missing decode -> contributes 0 to both sums
        st, fb = score_fn(span, g)
        strict_sum += float(st)
        fiber_sum += float(fb)
        n_scored += 1
    return {
        "n": n,
        "n_scored": n_scored,
        "strict": strict_sum / n if n else 0.0,
        "fiber": fiber_sum / n if n else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_oracle_ladder.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/evals/oracle_ladder.py tests/alignment_prototype/test_oracle_ladder.py
git commit -m "feat(evals): fixed-denominator acappella aggregation for the oracle ladder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Ladder runner CLI + real score_fn (integration, anchor-gated)

**Files:**
- Modify: `workspaces/alignment_prototype/evals/oracle_ladder.py`

**Interfaces:**
- Consumes:
  - `joint_ref_decode.main(argv: list[str]) -> int` — re-decodes an input timeline; call with `["--set-id", sid, "--decoder", "looptrace", "--timeline", <in>, "--out", <out>, "--workers", "8"]`.
  - `path_decode.trajectory_acc(pred_segs, gt_row, fiber=fib) -> (strict, n_pred, facc)`, `path_decode._pred_segs_from_span` is NOT used; instead reuse `score_timeline_vs_gt._pred_segs_from_span(span, anchor_s=float(gt["set_start_s"]))` and `score_timeline_vs_gt._resolve_ref_audio`.
  - `ref_fibers.compute_fibers(hf, FPS, audio_path=...)`, `path_decode._ensure_feat`.
- Produces:
  - `real_score_fn(set_id, by_tid, fibers: bool)` factory returning a `score_fn(span, gt_row)` that: resolves ref audio (GT stem), computes fibers (cached) when `fibers`, and returns `trajectory_acc` `(strict, facc)`.
  - `run_ladder(set_id: str, gt_path, r0_timeline, out_dir, fibers: bool = True) -> dict[str, dict]` — for each rung: build input timeline JSON, run `joint_ref_decode.main`, aggregate with `summarize_acappella`. Returns `{rung: {"strict":..,"fiber":..,"n":..,"n_scored":..}}`. R0 skips re-decode (uses `r0_timeline` directly, filtered to acap slots).
  - `main(argv=None)` CLI: `--set-id`, `--gt`, `--r0-timeline`, `--out-dir` (default `out/oracle_ladder/<set_id>/`), `--no-fibers`. Prints the per-rung table and writes `out/oracle_ladder/<set_id>/ladder.json`.

Notes for the implementer:
- **GT-acap rows** = `[r for r in gt_doc["tracks"] if (r.get("claimed_stem")=="acappella") and str(r.get("slot_label"))!="mix" and r.get("track_id")]`. Apply the `labeling/fixtures/id_maps/<set_id>.json` remap to `track_id` exactly as `score_spans` does (lines 199-207 of `score_timeline_vs_gt.py`) so identity matches the manifest.
- **manifest / by_tid**: `find_aligning_dir(set_id)` → `manifest.json`; build `by_tid` including `recording_id` fallback (mirror `joint_ref_decode` lines 151-154).
- **R0 filtering**: from `r0_timeline["spans"]`, keep spans whose `norm_slot(slot_label)` is in the GT-acap slot set; pass through `summarize_acappella` with the real score_fn (no re-decode).
- **Anchor logging**: print R0 strict/fiber and R3 strict/fiber next to their expected ranges (`R0 ~0.09–0.18 strict / ~0.28–0.36 fiber`; `R3 ~0.38–0.48 strict` looptrace-oracle ballpark per looptrace/NOTES.md). These are sanity ranges, not asserts (n=2, decoder-dependent).

- [ ] **Step 1: Implement the runner**

```python
# add to workspaces/alignment_prototype/evals/oracle_ladder.py
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from workspaces.alignment_prototype import joint_ref_decode as jrd
from workspaces.alignment_prototype import score_timeline_vs_gt as sc
from workspaces.alignment_prototype.path_decode import (
    FPS,
    _ensure_feat,
    trajectory_acc,
)
from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir

_REPO = Path(__file__).resolve().parents[3]


def _load_gt_acap(set_id: str, gt_path: Path) -> list[dict]:
    doc = yaml.safe_load(gt_path.read_text())
    if doc.get("set_id") != set_id:
        raise ValueError(f"GT set_id {doc.get('set_id')} != {set_id}")
    rows = [
        r
        for r in doc["tracks"]
        if (r.get("claimed_stem") == "acappella")
        and str(r.get("slot_label")) != "mix"
        and r.get("track_id")
    ]
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    if id_map_path.exists():
        id_map = json.loads(id_map_path.read_text())
        for r in rows:
            tid = str(r.get("track_id") or "")
            if tid in id_map and id_map[tid] != tid:
                r["track_id"] = id_map[tid]
    return rows


def _by_tid(set_dir: Path) -> dict[str, dict]:
    manifest = json.loads((set_dir / "manifest.json").read_text())
    d = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            d.setdefault(t["recording_id"], t)
    return d


def real_score_fn(set_id: str, by_tid: dict, fibers: bool):
    cache: dict[str, tuple] = {}

    def _fib(ref_audio):
        if not fibers or ref_audio is None:
            return None
        if ref_audio not in cache:
            from workspaces.alignment_prototype.ref_fibers import compute_fibers

            hf = np.load(_ensure_feat(ref_audio, ref_audio, "hubert", 9))
            cache[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
        return cache[ref_audio]

    def score_fn(span: dict, gt_row: dict) -> tuple[float, float]:
        ref_audio = sc._resolve_ref_audio(
            span, by_tid.get(span["recording_id"]), stem="acappella"
        )
        fib = _fib(ref_audio)
        pred = sc._pred_segs_from_span(span, anchor_s=float(gt_row["set_start_s"]))
        strict, _n, facc = trajectory_acc(pred, gt_row, fiber=fib)
        return float(strict), float(facc if fibers else strict)

    return score_fn


def run_ladder(
    set_id: str,
    gt_path: Path,
    r0_timeline: Path,
    out_dir: Path,
    fibers: bool = True,
) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_acap = _load_gt_acap(set_id, gt_path)
    set_dir = find_aligning_dir(set_id)
    by_tid = _by_tid(set_dir)
    score_fn = real_score_fn(set_id, by_tid, fibers)
    r0 = json.loads(Path(r0_timeline).read_text())
    r0_spans = r0["spans"]

    table: dict[str, dict] = {}
    for rung in RUNGS:
        spans = build_rung_timeline(rung, r0_spans, gt_acap)
        if rung == "R0":
            decoded = spans  # already decoded in the _lt timeline
        else:
            in_path = out_dir / f"{rung}_in.json"
            out_path = out_dir / f"{rung}_out.json"
            in_path.write_text(json.dumps({"spans": spans}))
            rc = jrd.main(
                [
                    "--set-id", set_id,
                    "--decoder", "looptrace",
                    "--timeline", str(in_path),
                    "--out", str(out_path),
                    "--workers", "8",
                ]
            )
            if rc != 0:
                raise RuntimeError(f"joint_ref_decode failed for {set_id} {rung}")
            decoded = json.loads(out_path.read_text())["spans"]
        table[rung] = summarize_acappella(decoded, gt_acap, score_fn)
        print(
            f"  {set_id} {rung}: strict={table[rung]['strict']:.3f} "
            f"fiber={table[rung]['fiber']:.3f} "
            f"(n={table[rung]['n']}, scored={table[rung]['n_scored']})"
        )
    (out_dir / "ladder.json").write_text(json.dumps(table, indent=2))
    return table


def _print_attribution(set_id: str, t: dict[str, dict]) -> None:
    f = {k: t[k]["fiber"] for k in RUNGS}
    print(f"\n[{set_id}] fiber-aware attribution (pp of the R0->R3 gap):")
    print(f"  routing  (R1-R0): {100 * (f['R1'] - f['R0']):+.1f}")
    print(f"  identity (R2-R1): {100 * (f['R2'] - f['R1']):+.1f}")
    print(f"  placement(R3-R2): {100 * (f['R3'] - f['R2']):+.1f}")
    print(
        f"  instance headroom  strict->fiber @R0: "
        f"{100 * (t['R0']['fiber'] - t['R0']['strict']):+.1f}  "
        f"@R3: {100 * (t['R3']['fiber'] - t['R3']['strict']):+.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--r0-timeline", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-fibers", action="store_true")
    a = p.parse_args(argv)
    out_dir = a.out_dir or (
        Path(__file__).resolve().parent.parent / "out" / "oracle_ladder" / a.set_id
    )
    t = run_ladder(a.set_id, a.gt, a.r0_timeline, out_dir, fibers=not a.no_fibers)
    _print_attribution(a.set_id, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the pure tests still pass**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_oracle_ladder.py -q`
Expected: PASS (5 tests; the runner import must not break collection).

- [ ] **Step 3: Anchor smoke on BB12 (the correctness gate)**

Run:
```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.evals.oracle_ladder \
  --set-id 1fsnxchk \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --r0-timeline workspaces/alignment_prototype/out/1fsnxchk_predicted_timeline_lt.json
```
Expected: prints R0..R3 strict/fiber. **Gate:** R0 strict in ~0.09–0.18 and fiber ~0.28–0.36 (reproduces the scorecard acappella e2e); R3 strict in ~0.38–0.48 (looptrace-oracle ballpark). If R0 or R3 is far outside range, STOP — the harness is wrong (debug matching / ref-audio resolution / decoder args) before trusting any lift.

- [ ] **Step 4: Commit**

```bash
git add workspaces/alignment_prototype/evals/oracle_ladder.py
git commit -m "feat(evals): oracle-ladder runner (looptrace-fixed re-decode + attribution)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Run both sets, write findings, apply decision rule

**Files:**
- Create: `workspaces/alignment_prototype/evals/ORACLE_LADDER_FINDINGS.md`
- Modify: `workspaces/alignment_prototype/looptrace/NOTES.md` (append a dated pointer to the findings)

- [ ] **Step 1: Run BB11 and BB12 (backgroundable; features build on first use)**

```bash
for sid_gt in "2nvzlh2k:bb11" "1fsnxchk:bb12"; do
  sid=${sid_gt%%:*}; nm=${sid_gt##*:}
  venvs/audio/bin/python -m workspaces.alignment_prototype.evals.oracle_ladder \
    --set-id $sid --gt labeling/fixtures/${nm}_ground_truth.yaml \
    --r0-timeline workspaces/alignment_prototype/out/${sid}_predicted_timeline_lt.json \
    2>&1 | tee workspaces/alignment_prototype/out/oracle_ladder/${sid}.log
done
```
Expected: two per-set tables + attribution blocks. Confirm both R0/R3 anchors are in range for BOTH sets.

- [ ] **Step 2: Cross-check R3 against the legacy oracle (decoder-difference note)**

Run (BB12 shown; repeat BB11):
```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.path_decode --eval \
  --feature hubert --stems acappella --fibers --workers 8 \
  --gt labeling/fixtures/bb12_ground_truth.yaml 2>&1 | tail -15
```
Record its acappella strict/fiber (legacy-oracle, expected ~37/61). Note the R3-looptrace vs legacy-oracle delta in the findings — a large gap is itself a finding (decoder choice matters at oracle placement).

- [ ] **Step 3: Write the findings note**

Create `ORACLE_LADDER_FINDINGS.md` with: the two per-set rung tables (strict+fiber), the attribution (routing/identity/placement pp of the R0→R3 fiber gap), the strict→fiber instance-headroom at R0 and R3, the legacy-oracle cross-check, and the **decision** per the spec's rule (§4): *build the selector* iff the decode-instance slice is the largest single reachable slice AND positive in both sets AND the R3 strict→fiber gap is also large; else *redirect to placement/routing*. State the verdict in one sentence at the top. Cite `docs/alignment_status.md` for headline provenance; do not hand-type new status numbers into other docs.

- [ ] **Step 4: Append a pointer to looptrace/NOTES.md**

Add a dated section "Oracle→e2e gap decomposition (2026-07-18)" summarizing the verdict in 2-3 sentences and linking `evals/ORACLE_LADDER_FINDINGS.md` — this closes the "THE next analytic step" open item recorded in NOTES.md.

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/evals/ORACLE_LADDER_FINDINGS.md workspaces/alignment_prototype/looptrace/NOTES.md
git commit -m "docs(evals): oracle-ladder findings + decision on the acappella selector

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** §2 ladder (Tasks 1+3), §3 outputs/table (Tasks 3+4), §4 decision rule (Task 4 Step 3), §5 correctness gate/anchors (Task 3 Step 3 + Task 4 Step 1), §6 reuse map (Tasks 1-3 interfaces), §7 scope (Global Constraints — no model, eval-only), §8 risks/decoder-difference (Task 4 Step 2). All covered.

**Placeholder scan:** no TBD/TODO; all code blocks concrete. Anchor ranges are explicit numbers, not "appropriate".

**Type consistency:** `build_rung_timeline`, `summarize_acappella`, `run_ladder`, `real_score_fn`, `norm_slot` names consistent across tasks. `score_fn` signature `(span, gt_row) -> (strict, fiber)` identical in Task 2 (injected) and Task 3 (real). `RUNGS` defined once (Task 1), reused (Tasks 2-3).

**Open risk flagged for the implementer:** `sc._pred_segs_from_span` and `sc._resolve_ref_audio` are private helpers — confirm their exact signatures at implementation time (Task 3 Step 1) and adjust the call sites if they differ; they are the only non-public reuse points.

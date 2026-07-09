#!/usr/bin/env python3
"""Q1 — validate candidate-quality scorers against the human's revealed winners.

Plan of record: docs/stem_quality_ranking_plan.md. The ONLY clean human labels
are BB12's `quality:human_winner` ledger rows (107; BB11's 67 are the HuBERT
gate's own picks — circular). For every winner slot, score ALL sibling
candidates with each scorer and measure within-slot winner agreement vs the
cand1-prior baseline. Scores are cached to JSONL so reruns are incremental.

Scorers (Q0 battery, local subset — SingMOS deferred to the Vast leg):
  - audiobox PQ/CE/CU/PC  (facebook/audiobox-aesthetics; music-trained)
  - rolloff95_hz, hf16k_ratio_db  (separation_qa.cleanliness_features —
    the only cheap features the closed experiment found pointing the
    right way; bandwidth-as-quality)

Run inside venvs/quality (audiobox + librosa):
    venvs/quality/bin/python -m workspaces.separation_qa.q1_validate_winners \
        [--score-only] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import core as _core  # run via `python -m` from repo root (house rule)

_REPO = Path(_core.__file__).resolve().parent.parent

LEDGER = (
    _REPO / "tests" / "fixtures" / "ingest" / "correction_ledger_snapshot_20260709.tsv"
)
OUT = Path(__file__).resolve().parent / "out"
SCORES = OUT / "q1_candidate_scores.jsonl"
ALIGNING = Path.home() / "aligning"
BB12_DIR = next(ALIGNING.glob("1fsnxchk__*"))


def _winner_slots() -> dict[str, str]:
    """{slot -> winning candN prefix} from the BB12 human-winner ledger rows."""
    out: dict[str, str] = {}
    with LEDGER.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            blob = f"{r['reason']} {r['source']}"
            if "labeling_bb12" not in blob or "human_winner" not in blob:
                continue
            slot_m = re.search(r"slot:(\w+)", blob)
            file_m = re.search(r"file:(cand\d+)__", blob)
            if slot_m and file_m:
                out[slot_m.group(1)] = file_m.group(1)
    return out


def _slot_candidates(slot: str) -> list[Path]:
    """All external candidate audio files for a BB12 slot (flat candN__ layout)."""
    cands: list[Path] = []
    for d in BB12_DIR.glob(f"stems/{slot}__*"):
        cands += sorted(d.glob("cand*__*.m4a"))
        cands += sorted(d.glob("candidates/**/cand*__*.m4a"))
    # de-dup by name (tagged/untagged dirs may both exist)
    seen: set[str] = set()
    uniq = []
    for c in cands:
        if c.name not in seen:
            seen.add(c.name)
            uniq.append(c)
    return uniq


def _load_cached() -> dict[str, dict]:
    if not SCORES.exists():
        return {}
    out = {}
    for line in SCORES.read_text().splitlines():
        r = json.loads(line)
        out[r["path"]] = r
    return out


def _score_files(files: list[Path]) -> None:
    """Append audiobox + bandwidth scores for any un-scored files."""
    cached = _load_cached()
    todo = [f for f in files if str(f) not in cached]
    if not todo:
        return
    from audiobox_aesthetics.infer import initialize_predictor

    from workspaces.separation_qa.cleanliness_features import extract as clean_extract

    predictor = initialize_predictor()
    OUT.mkdir(exist_ok=True)
    with SCORES.open("a") as fh:
        for i, f in enumerate(todo, 1):
            row: dict = {"path": str(f), "name": f.name}
            try:
                ab = predictor.forward([{"path": str(f)}])[0]
                row.update({k: float(v) for k, v in ab.items()})
            except Exception as e:  # noqa: BLE001 — score what we can, log the rest
                row["audiobox_error"] = str(e)[:120]
            try:
                cl = clean_extract(f)
                row["rolloff95_hz"] = float(cl.rolloff95_hz)
                row["hf16k_ratio_db"] = float(cl.hf16k_ratio_db)
            except Exception as e:  # noqa: BLE001
                row["clean_error"] = str(e)[:120]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"  scored {i}/{len(todo)}  {f.name[:60]}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap slots (smoke)")
    args = ap.parse_args(argv)

    winners = _winner_slots()
    slots = dict(
        sorted(winners.items())[: args.limit] if args.limit else winners.items()
    )
    print(f"winner slots: {len(slots)} (of {len(winners)})")

    slot_files: dict[str, list[Path]] = {s: _slot_candidates(s) for s in slots}
    multi = {s: fs for s, fs in slot_files.items() if len(fs) >= 2}
    print(f"slots with >=2 candidates on disk: {len(multi)}")
    _score_files([f for fs in multi.values() for f in fs])
    if args.score_only:
        return 0

    cached = _load_cached()
    metrics = ["PQ", "CE", "CU", "PC", "rolloff95_hz", "hf16k_ratio_db"]
    agree: dict[str, list[int]] = defaultdict(list)
    cand1_prior: list[int] = []
    n_eval = 0
    for slot, files in sorted(multi.items()):
        want = winners[slot]
        rows = [cached.get(str(f)) for f in files]
        rows = [r for r in rows if r]
        if len(rows) < 2 or not any(r["name"].startswith(want + "__") for r in rows):
            continue  # winner file not on disk / not scored -> can't judge
        n_eval += 1
        cand1_prior.append(int(want == "cand1"))
        for m in metrics:
            vals = [(r.get(m), r["name"]) for r in rows if r.get(m) is not None]
            if len(vals) < 2:
                continue
            top = max(vals)[1]
            agree[m].append(int(top.startswith(want + "__")))
    print(f"\nevaluable slots (winner on disk, >=2 scored cands): {n_eval}")
    print(f"{'scorer':<16} {'winner-agreement':>16}")
    print(f"{'cand1 prior':<16} {sum(cand1_prior) / max(1, len(cand1_prior)):>15.0%}")
    for m in metrics:
        if agree[m]:
            print(
                f"{m:<16} {sum(agree[m]) / len(agree[m]):>15.0%}  (n={len(agree[m])})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Resolve stem-candidate winners implicitly from a labeling `.als`.

A DJ-set labeling session auditions candidate acappella/instrumental stems
(``stems/<slot>/candidates/<layer>/cand*.m4a``); the WINNER for a slot is simply
the candidate the human actually placed on the timeline. So the winner map is the
set of placed acappella/instrumental clips, deduped per (slot, stem) — no separate
human pick needed. This unblocks A2 (BB11 identity) and per-stem placement: the
winner file is the reference audio for that slot's stem.

recording_id assignment (corpus ingest) is a separate step — here we emit the
winner file + the song label parsed from the placement; ingest matches that to a
recording (scripts/ingest_stem_url.py).

Usage:
    venvs/audio/bin/python -m labeling.extract_winners \
        --als "$HOME/aligning/<set>/<proj>/<name>.als" \
        --set-dir "$HOME/aligning/<set>"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.export_als_to_gt import collect_kept_clip_rows

_STEMS = ("acappella", "instrumental")


@dataclass(frozen=True)
class Winner:
    slot_label: str
    claimed_stem: str
    display: str
    winner_path: str
    span_s: float  # total placed mix-time (the pick criterion)


def extract_winners(als_path: Path, set_dir: Path) -> list[Winner]:
    """Per (slot, stem) the placed candidate with the most mix-time = the winner."""
    _set_id, rows, _ = collect_kept_clip_rows(als_path, set_dir)
    by_key: dict[tuple[str, str], list] = {}
    for r in rows:
        if r.claimed_stem not in _STEMS:
            continue
        by_key.setdefault((r.slot_label, r.claimed_stem), []).append(r)
    winners: list[Winner] = []
    for (slot, stem), group in by_key.items():
        # winner = the placed file occupying the most total mix-time for this slot
        by_path: dict[str, float] = {}
        disp: dict[str, str] = {}
        for r in group:
            by_path[r.clip.path] = by_path.get(r.clip.path, 0.0) + (
                r.set_end_s - r.set_start_s
            )
            disp[r.clip.path] = r.display
        path = max(by_path, key=by_path.get)
        winners.append(Winner(slot, stem, disp[path], path, round(by_path[path], 1)))
    return sorted(winners, key=lambda w: (w.slot_label, w.claimed_stem))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--als", type=Path, required=True)
    p.add_argument("--set-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.als.is_file() or not args.set_dir.is_dir():
        print("als/set-dir not found", file=sys.stderr)
        return 2
    winners = extract_winners(args.als, args.set_dir)
    n_aca = sum(1 for w in winners if w.claimed_stem == "acappella")
    n_ins = sum(1 for w in winners if w.claimed_stem == "instrumental")
    print(f"{len(winners)} winners ({n_aca} acappella, {n_ins} instrumental)")
    for w in winners:
        print(
            f"  {w.slot_label:7} {w.claimed_stem:12} {w.span_s:5.0f}s  "
            f"{Path(w.winner_path).name[:46]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

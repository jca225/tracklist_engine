"""Shadow CLI: ``python -m alignment.daw_env``."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from alignment.agentic.events import EventLog
from alignment.daw_env.loop import resolve_daw
from alignment.daw_env.session import (
    DAW_REACT_SUFFIX,
    DawSession,
    load_timeline,
)

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "alignment" / "out"
_SET_DISPLAY = {
    "1fsnxchk": "BB12",
    "2nvzlh2k": "BB11",
    "w1mgcjt": "BB10",
}


def _default_timeline(set_id: str) -> Path:
    for name in (
        f"{set_id}_predicted_timeline_lt.json",
        f"{set_id}_predicted_timeline_agentic.json",
        f"{set_id}_predicted_timeline.json",
    ):
        p = OUT / name
        if p.is_file():
            return p
    return OUT / f"{set_id}_predicted_timeline.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ableton ReAct harness (ALS-first shadow)")
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--mode", choices=("a", "b"), default="a")
    ap.add_argument("--timeline", type=Path, default=None)
    ap.add_argument(
        "--als", type=Path, default=None, help="optional working .als to mutate"
    )
    ap.add_argument(
        "--set-dir", type=Path, default=None, help="aligning folder with mix.*"
    )
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument(
        "--budget-spans", type=int, default=None, help="limit spans (smoke)"
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--no-live-sensors",
        action="store_true",
        help="skip LiveContext fp/HuBERT/lyrics (onset-only)",
    )
    ap.add_argument(
        "--with-lyrics",
        action="store_true",
        help="also run Whisper JOINT lyrics (slow cold-start; off by default)",
    )
    ap.add_argument(
        "--stems",
        default=None,
        help="comma-separated claimed_stem filter (e.g. acappella,instrumental)",
    )
    args = ap.parse_args(argv)

    tl_path = args.timeline or _default_timeline(args.set_id)
    if not tl_path.is_file():
        print(f"error: timeline not found: {tl_path}")
        return 2
    out_dir = args.out_dir or (OUT / "daw_env" / args.set_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    als_path: Path | None = None
    if args.als is not None:
        if not args.als.is_file():
            print(f"error: --als not found: {args.als}")
            return 2
        if "align.als" in args.als.name.lower() and "DAW REACT" not in args.als.name:
            print("error: refusing to use * align.als as working session")
            return 2
        # Copy to DAW REACT stamp — never overwrite * align.als
        display = _SET_DISPLAY.get(args.set_id, args.set_id)
        dest = out_dir / f"{display}{DAW_REACT_SUFFIX}"
        if args.als.resolve() != dest.resolve():
            shutil.copy2(args.als, dest)
        als_path = dest

    session = DawSession.from_timeline(
        args.set_id,
        load_timeline(tl_path),
        out_dir=out_dir,
        als_path=als_path,
        set_dir=args.set_dir,
    )
    if args.set_dir is not None:
        (out_dir / "set_dir.txt").write_text(str(args.set_dir.resolve()) + "\n")

    log = EventLog(out_dir / "events.jsonl")
    result = resolve_daw(
        session,
        mode=args.mode,
        log=log,
        max_steps_per_span=args.max_steps,
        budget_spans=args.budget_spans,
        use_live_sensors=not args.no_live_sensors,
        with_lyrics=args.with_lyrics,
        stems=(
            frozenset(s.strip() for s in args.stems.split(",") if s.strip())
            if args.stems
            else None
        ),
    )
    # Re-bind set_dir into render by monkey-patching via module-level is awkward;
    # if set_dir given, re-render is not needed — loop already ran. Document path.
    tl_out = out_dir / f"{args.set_id}_daw_react_timeline.json"
    tl_out.write_text(json.dumps(session.to_timeline(), indent=2) + "\n")
    hist = session.write_history(out_dir / "action_history.json")

    print(result.summary())
    print(f"timeline: {tl_out}")
    print(f"history:  {hist}")
    print(f"events:   {log.path}")
    if als_path is not None:
        print(f"als:      {als_path}")
        if args.mode == "b":
            print(
                "Mode B: open the DAW REACT.als in Live, correct AGENT PROPOSE "
                "locators/clips, then: python -m labeling.extract.export_als_to_gt "
                f"--als {als_path} --set-dir <aligning-folder>"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

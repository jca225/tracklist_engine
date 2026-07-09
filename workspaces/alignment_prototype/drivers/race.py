"""Race the end-to-end drivers on one scorecard.

Runs each requested driver on each set, scores every produced timeline through
the SAME `score_timeline_vs_gt`, and prints a side-by-side board. The classical
driver runs first per set (it is the base the agentic/ml drivers refine); pass
`--reuse-base SET=path` to skip the expensive infer stage and start from an
existing timeline.

    venvs/audio/bin/python -m workspaces.alignment_prototype.drivers.race \
        --sets 1fsnxchk,2nvzlh2k --drivers classical,agentic,ml [--fibers]

    # iterate fast on the board using an existing classical timeline as base:
    venvs/audio/bin/python -m workspaces.alignment_prototype.drivers.race \
        --sets 2nvzlh2k --drivers classical,agentic,ml \
        --reuse-base 2nvzlh2k=out/2nvzlh2k_predicted_timeline_lt_v2.json
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from .agentic import AgenticDriver
from .base import _REPO, SetContext
from .classical import ClassicalDriver
from .ml import HybridMlDriver

_METRICS = {
    "identity": re.compile(r"identity:\s+\d+/\d+\s+\((\d+)%\)"),
    "place_med": re.compile(r"set placement \|pred-gt\|:\s+median=([\d.]+)s"),
    "ref_med": re.compile(r"ref offset \|pred-gt\|.*?median=([\d.]+)s"),
    "traj_head": re.compile(r"HEADLINE multiseg\+loop n=\s*\d+ traj-acc=(\d+)%"),
    "traj_acap": re.compile(r"stem\s+acappella\s+n=\s*\d+ traj-acc=(\d+)%"),
}


def score_timeline(set_id: str, timeline: Path, *, fibers: bool = False) -> dict:
    """Run the canonical scorer on a timeline; parse its board metrics."""
    argv = [
        sys.executable,
        "-m",
        "workspaces.alignment_prototype.score_timeline_vs_gt",
        "--set-id",
        set_id,
        "--timeline",
        str(timeline),
    ]
    if fibers:
        argv.append("--fibers")
    proc = subprocess.run(argv, cwd=str(_REPO), capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out)
        raise RuntimeError(f"scorer failed for {set_id} / {timeline.name}")
    scores = {"_raw": out}
    for key, rx in _METRICS.items():
        m = rx.search(out)
        scores[key] = m.group(1) if m else None
    return scores


def _parse_reuse(pairs: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--reuse-base expects SET=path, got {p!r}")
        sid, path = p.split("=", 1)
        out[sid] = Path(path)
    return out


def run(
    sets: list[str],
    drivers: list[str],
    *,
    reuse_base: dict[str, Path] | None = None,
    live: bool = True,
    fibers: bool = False,
    lam: float = 16.0,
    source_set: str | None = None,
    ml_gate: float | None = None,
) -> dict[tuple[str, str], dict]:
    reuse_base = reuse_base or {}
    board: dict[tuple[str, str], dict] = {}

    for set_id in sets:
        ctx = SetContext.for_set(set_id, source_set_id=source_set)
        print(f"\n=== set {set_id} (source={ctx.source_set_id}) ===")

        # classical base — reused or freshly run. Every other driver refines it,
        # so it must exist even if 'classical' isn't in the scored list.
        if set_id in reuse_base:
            base = reuse_base[set_id]
            print(f"[classical] reusing base {base}")
        else:
            print("[classical] running infer -> joint_ref_decode …")
            base = ClassicalDriver().align_set(ctx)

        produced: dict[str, Path] = {"classical": base}
        for name in drivers:
            if name == "classical":
                continue
            print(f"[{name}] refining base …")
            if name == "agentic":
                produced[name] = AgenticDriver(base, live=live).align_set(ctx)
            elif name == "ml":
                produced[name] = HybridMlDriver(
                    base, lam=lam, gate_margin=ml_gate
                ).align_set(ctx)
            else:
                raise SystemExit(f"unknown driver {name!r}")

        for name in drivers:
            path = produced[name]
            print(f"[score] {set_id} / {name}")
            board[(set_id, name)] = score_timeline(set_id, path, fibers=fibers)

    _print_board(board, sets, drivers, fibers=fibers)
    return board


def _print_board(board, sets, drivers, *, fibers: bool) -> None:
    traj_label = "traj(fib)%" if fibers else "traj%"
    cols = ["id%", "place_med", "ref_med", f"head_{traj_label}", f"acap_{traj_label}"]
    keys = ["identity", "place_med", "ref_med", "traj_head", "traj_acap"]
    print("\n" + "=" * 78)
    print("SCORECARD — end-to-end driver race")
    print("=" * 78)
    header = f"{'set':10} {'driver':10} " + " ".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for set_id in sets:
        for name in drivers:
            s = board.get((set_id, name), {})
            cells = " ".join(f"{(s.get(k) or '–'):>12}" for k in keys)
            print(f"{set_id:10} {name:10} {cells}")
    print("=" * 78)
    print(
        "id%=identity  place_med/ref_med=median |pred-gt| seconds  "
        f"head=multiseg+loop {traj_label}  acap=acappella {traj_label}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets", default="1fsnxchk,2nvzlh2k")
    p.add_argument("--drivers", default="classical,agentic,ml")
    p.add_argument(
        "--reuse-base",
        action="append",
        metavar="SET=PATH",
        help="use an existing timeline as the classical base for a set "
        "(skips infer); repeatable",
    )
    p.add_argument(
        "--replay",
        action="store_true",
        help="agentic driver uses replay runners (no audio) instead of live probes",
    )
    p.add_argument(
        "--fibers", action="store_true", help="fiber-aware trajectory scoring"
    )
    p.add_argument("--lam", type=float, default=16.0, help="ml Viterbi jump cost")
    p.add_argument(
        "--ml-gate",
        type=float,
        default=None,
        help="ml confidence gate: keep classical segments unless the learned "
        "decode margin clears this (per-checkpoint scale; sweep it)",
    )
    p.add_argument(
        "--source-set",
        default=None,
        help="cross-set supervision source (default: the other complete GT set)",
    )
    args = p.parse_args(argv)

    run(
        sets=[s for s in args.sets.split(",") if s],
        drivers=[d for d in args.drivers.split(",") if d],
        reuse_base=_parse_reuse(args.reuse_base),
        live=not args.replay,
        fibers=args.fibers,
        lam=args.lam,
        source_set=args.source_set,
        ml_gate=args.ml_gate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

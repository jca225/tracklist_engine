"""Race the end-to-end drivers on one scorecard.

Runs each requested driver on each set, scores every produced timeline through
the SAME `score_timeline_vs_gt`, and prints a side-by-side board. The classical
driver runs first per set (it is the base the agentic/ml drivers refine); pass
`--reuse-base SET=path` to skip the expensive infer stage and start from an
existing timeline.

    venvs/audio/bin/python -m alignment.drivers.race \
        --sets 1fsnxchk,2nvzlh2k --drivers classical,agentic,ml

    # iterate fast on the board using an existing classical timeline as base:
    venvs/audio/bin/python -m alignment.drivers.race \
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
    "traj_nopile": re.compile(
        r"HEADLINE \(pileups excluded\) n=\s*\d+ traj-acc=(\d+)%"
    ),
    "traj_acap": re.compile(r"stem\s+acappella\s+n=\s*\d+ traj-acc=(\d+)%"),
}


def _run_scorer(set_id: str, timeline: Path, *, fibers: bool) -> str:
    argv = [
        sys.executable,
        "-m",
        "alignment.score_timeline_vs_gt",
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
    return out


def score_timeline(set_id: str, timeline: Path) -> dict:
    """Run the canonical scorer twice (strict + fiber-aware); parse both.

    The board shows both rulers side by side: strict = exact-instance
    coordinates; fiber-aware = repeat-equivalent credit (a within-fiber pick
    plays the same audio — the honest product metric). `abstain` counts spans
    whose fiber_ambiguity flags the instance as content-undecidable."""
    strict = _run_scorer(set_id, timeline, fibers=False)
    fib = _run_scorer(set_id, timeline, fibers=True)
    scores = {"_raw": fib}
    for key, rx in _METRICS.items():
        m = rx.search(strict)
        scores[key] = m.group(1) if m else None
        m = rx.search(fib)
        scores[f"{key}_fib"] = m.group(1) if m else None
    try:
        import json

        spans = json.loads(timeline.read_text())["spans"]
        decoded = [s for s in spans if s.get("ref_segments")]
        amb = sum(
            1
            for s in decoded
            if isinstance(s.get("fiber_ambiguity"), dict)
            and s["fiber_ambiguity"].get("ambiguous")
        )
        scores["abstain"] = f"{amb}/{len(decoded)}" if decoded else None
    except Exception:
        scores["abstain"] = None
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
            board[(set_id, name)] = score_timeline(set_id, path)

    _print_board(board, sets, drivers)
    return board


def _print_board(board, sets, drivers) -> None:
    cols = [
        "id%",
        "place_med",
        "ref_med",
        "head%",
        "headF%",
        "noPileF%",
        "acapF%",
        "abstain",
    ]
    keys = [
        "identity",
        "place_med",
        "ref_med",
        "traj_head",
        "traj_head_fib",
        "traj_nopile_fib",
        "traj_acap_fib",
        "abstain",
    ]
    print("\n" + "=" * 100)
    print("SCORECARD — end-to-end driver race (strict AND fiber-aware)")
    print("=" * 100)
    header = f"{'set':10} {'driver':10} " + " ".join(f"{c:>9}" for c in cols)
    print(header)
    print("-" * len(header))
    for set_id in sets:
        for name in drivers:
            s = board.get((set_id, name), {})
            cells = " ".join(f"{(s.get(k) or '–'):>9}" for k in keys)
            print(f"{set_id:10} {name:10} {cells}")
    print("=" * 100)
    print(
        "id%=identity  place/ref_med=median |pred-gt| s  head%=multiseg+loop strict  "
        "headF%=fiber-aware  noPileF%=fiber-aware excl. medley pileups  "
        "acapF%=acappella fiber-aware  abstain=instance-ambiguous/decoded spans"
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
        lam=args.lam,
        source_set=args.source_set,
        ml_gate=args.ml_gate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Ablation CLI: run a YAML config through the runner, print a board, append the
ledger. Compose grain is wired end-to-end (the `make race` replacement); isolate
grain's trajectory backend/scorer ship with the TRM build (bake-off).

    venvs/audio/bin/python -m workspaces.alignment_prototype.pipeline.cli \
        --config workspaces/alignment_prototype/pipeline/configs/race_default.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import runner
from .stages import AblationSpec, Grain

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_LEDGER = Path(__file__).resolve().parents[1] / "out" / "ablation_runs.jsonl"


def load_spec(cfg: dict) -> AblationSpec:
    """Config dict -> AblationSpec (pure; validated by AblationSpec.__post_init__
    and Grain())."""
    return AblationSpec(
        grain=Grain(cfg["grain"]),
        sets=tuple(cfg["sets"]),
        actors=tuple(cfg["actors"]),
        base=cfg.get("base", "classical"),
        split=cfg.get("split", "set"),
        seed=int(cfg.get("seed", 0)),
        bridge=cfg.get("bridge", "none"),
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO), text=True
        ).strip()
    except Exception:
        return "unknown"


def _backends_for(grain: Grain):
    if grain is Grain.COMPOSE:
        from .adapters.backends import EndToEndBackend
        from .adapters.scorers import TimelineScorer

        return EndToEndBackend(), TimelineScorer()
    raise SystemExit(
        "isolate-grain backend/scorer ship with the TRM build "
        "(docs/trm_decoder_bakeoff.md); compose grain is wired today."
    )


def _print_board(rows: list[runner.RunRow]) -> None:
    print(f"\n{'set':<10} {'role':<9} {'actor':<20} {'strict':>7} {'fiber':>7}  flag")
    print("-" * 64)
    for r in rows:
        strict = r.metrics.get("strict")
        fiber = r.metrics.get("fiber")
        flag = "LEARNED-NOTHING" if r.learned_nothing else ""
        print(
            f"{r.set_id:<10} {r.role:<9} {r.actor:<20} "
            f"{'' if strict is None else f'{strict:.3f}':>7} "
            f"{'' if fiber is None else f'{fiber:.3f}':>7}  {flag}"
        )


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ledger", type=Path, default=_DEFAULT_LEDGER)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    spec = load_spec(cfg)
    backend, scorer = _backends_for(spec.grain)
    rows = runner.run(
        spec,
        backend=backend,
        scorer=scorer,
        ledger_path=args.ledger,
        git_sha=_git_sha(),
        now=datetime.now(timezone.utc).isoformat(),
    )
    _print_board(rows)
    print(f"\nledger: {args.ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

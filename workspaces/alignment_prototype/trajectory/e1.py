"""E1 orchestration: BB10 pseudo-labels → BB11 held-out TRM comparison.

Artifact-first pipeline (trm_flywheel_design.md §7):
  base timeline → pseudo-safe agentic timeline → pseudo-GT YAML → train runs.

No canonical DB writes. Generated artifacts stay under ``out/`` and are not
committed. Headline metrics are not hand-copied into result JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.contracts import MANIFEST_FILENAME, load_timeline
from workspaces.alignment_prototype.agentic.events import EventLog
from workspaces.alignment_prototype.agentic.loop import SpanCtx
from workspaces.alignment_prototype.agentic.policy import Ladder
from workspaces.alignment_prototype.drivers.agentic import refine_agentic_spans
from workspaces.alignment_prototype.drivers.base import (
    finalize,
    load_timeline_dict,
    preflight_set,
)
from workspaces.alignment_prototype.trajectory.data import GT_FIXTURES
from workspaces.alignment_prototype.trajectory.pseudo_materialize import (
    materialize_pseudo_gt,
)

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "out" / "e1"
_PSEUDO_LADDER = Ladder(auto=0.75, review=0.40, suggest=0.10, combine=True)


@dataclass(frozen=True)
class E1Config:
    pool_set: str
    eval_set: str
    base_timeline: Path
    synthetic_root: Path
    out_dir: Path
    smoke_only: bool = False
    reuse_agentic: bool = False

    def __post_init__(self) -> None:
        if self.pool_set == self.eval_set:
            raise ValueError(
                f"pool_set and eval_set are the same set ({self.pool_set!r}) — "
                "LOSO leakage refuse"
            )


def build_train_commands(cfg: E1Config, pseudo_yaml: Path) -> list[list[str]]:
    """Full-mode train commands: conv, synthetic-only TRM, pseudo TRM."""
    py = sys.executable
    mod = "workspaces.alignment_prototype.trajectory.train"
    eval_args = ["--eval-set", cfg.eval_set, "--split", "set"]
    common = [*eval_args, "--device", "auto"]

    conv = [
        py,
        "-m",
        mod,
        "--model",
        "conv",
        "--train-yaml",
        str(pseudo_yaml),
        *common,
    ]
    # synthetic-only still builds a real train Dataset for scaffolding, then
    # excludes it from the loader. Use a non-eval GT fixture as the placeholder.
    others = [sid for sid in sorted(GT_FIXTURES) if sid != cfg.eval_set]
    if not others:
        raise ValueError(
            "no non-eval GT fixture available for synthetic-only placeholder"
        )
    placeholder = others[0]
    synth = [
        py,
        "-m",
        mod,
        "--model",
        "trm",
        "--synthetic-only",
        "--synthetic-root",
        str(cfg.synthetic_root),
        "--train-set",
        placeholder,
        *common,
    ]
    trm = [
        py,
        "-m",
        mod,
        "--model",
        "trm",
        "--train-yaml",
        str(pseudo_yaml),
        *common,
    ]
    return [conv, synth, trm]


def build_smoke_command(cfg: E1Config, pseudo_yaml: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "workspaces.alignment_prototype.trajectory.train",
        "--model",
        "trm",
        "--train-yaml",
        str(pseudo_yaml),
        "--eval-set",
        cfg.eval_set,
        "--split",
        "set",
        "--smoke",
        "--device",
        "auto",
    ]


def _write_result(out_dir: Path, payload: dict) -> Path:
    path = out_dir / "e1_result.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(proc.stdout)
        sys.stdout.write(proc.stdout)
    return int(proc.returncode)


def _produce_agentic_timeline(cfg: E1Config, aligning: Path) -> Path:
    import os

    from workspaces.alignment_prototype.agentic.live_runners import (
        LiveContext,
        build_live_runners,
    )
    from workspaces.alignment_prototype.agentic.novelty import surprise_runner_for_set

    out_path = cfg.out_dir / f"{cfg.pool_set}_agentic_pseudo_timeline.json"
    if cfg.reuse_agentic and out_path.is_file():
        load_timeline(out_path)
        return out_path

    # Unlabeled-pool mass lever: admit landmark fp into belief (still fail-closed
    # for ordinary agentic/race on BB11/BB12). fp precision 0.53 cannot solo
    # auto-commit; E1 needs it as an independent content vote with HuBERT.
    os.environ.setdefault("AGENTIC_LIVE_ENABLE_FP_PLACEMENT", "1")

    base = load_timeline_dict(cfg.base_timeline)
    if str(base.get("set_id")) != cfg.pool_set:
        raise ValueError(
            f"base timeline set_id={base.get('set_id')!r} != pool_set={cfg.pool_set!r}"
        )
    load_timeline(cfg.base_timeline)

    spans_ctx: list[SpanCtx] = []
    for s in base["spans"]:
        stem = s.get("claimed_stem") or "regular"
        spans_ctx.append(
            SpanCtx(
                slot_label=str(s["slot_label"]),
                recording_id=str(s["recording_id"]),
                claimed_stem=stem,
                data={**s, "claimed_stem": stem},
            )
        )

    lctx = LiveContext.from_set(cfg.pool_set, base["spans"], apply_fiber_gate=True)
    if lctx is None:
        raise RuntimeError(
            f"agentic live: no LiveContext for {cfg.pool_set} (missing mix MERT). "
            "Export/cache MERT for the pool set before E1, or pass --reuse-agentic "
            "with an existing agentic timeline."
        )
    runners = build_live_runners(lctx)
    surprise = surprise_runner_for_set(cfg.pool_set)
    if surprise is not None:
        runners["surprise"] = surprise

    log_path = cfg.out_dir / "agentic" / f"{cfg.pool_set}_e1_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    out_spans, res = refine_agentic_spans(
        base,
        spans_ctx,
        runners,
        EventLog(log_path),
        ladder=_PSEUDO_LADDER,
        pseudo_safe=True,
    )
    print(f"  e1 agentic: {res.summary()}")
    payload = {
        **base,
        "spans": out_spans,
        "driver": "agentic (e1 pseudo-safe)",
        "pseudo_safety": {
            "combine": True,
            "fiber_gate": True,
            "auto": _PSEUDO_LADDER.auto,
            "review": _PSEUDO_LADDER.review,
            "suggest": _PSEUDO_LADDER.suggest,
        },
    }
    return finalize(payload, out_path)


def run_e1(cfg: E1Config) -> int:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    aligning = preflight_set(cfg.pool_set)
    manifest_path = aligning / MANIFEST_FILENAME

    try:
        agentic_path = _produce_agentic_timeline(cfg, aligning)
    except Exception as exc:  # noqa: BLE001 — surface as failed stage
        _write_result(
            cfg.out_dir,
            {
                "status": "failed",
                "stage": "agentic",
                "error": f"{type(exc).__name__}: {exc}",
                "pool_set": cfg.pool_set,
                "eval_set": cfg.eval_set,
            },
        )
        raise

    pseudo_path = cfg.out_dir / f"{cfg.pool_set}_pseudo_gt.yaml"
    report = materialize_pseudo_gt(agentic_path, manifest_path, pseudo_path)
    print(
        f"  e1 materialize: accepted {report.accepted}/{report.total} "
        f"dropped={report.dropped}"
    )
    if report.accepted == 0:
        _write_result(
            cfg.out_dir,
            {
                "status": "starved",
                "pool_set": cfg.pool_set,
                "eval_set": cfg.eval_set,
                "agentic_timeline": str(agentic_path),
                "pseudo_yaml": str(pseudo_path),
                "accepted": 0,
                "total": report.total,
                "dropped": report.dropped,
            },
        )
        return 2

    commands: list[tuple[str, list[str]]] = []
    if cfg.smoke_only:
        commands.append(("smoke_trm", build_smoke_command(cfg, pseudo_path)))
    else:
        commands.append(("smoke_trm", build_smoke_command(cfg, pseudo_path)))
        for i, cmd in enumerate(build_train_commands(cfg, pseudo_path)):
            name = ("conv", "synth_trm", "pseudo_trm")[i]
            commands.append((name, cmd))

    logs: dict[str, str] = {}
    for name, cmd in commands:
        log_path = cfg.out_dir / "logs" / f"{name}.log"
        code = _run_logged(cmd, log_path)
        logs[name] = str(log_path)
        if code != 0:
            _write_result(
                cfg.out_dir,
                {
                    "status": "failed",
                    "stage": name,
                    "exit_code": code,
                    "pool_set": cfg.pool_set,
                    "eval_set": cfg.eval_set,
                    "agentic_timeline": str(agentic_path),
                    "pseudo_yaml": str(pseudo_path),
                    "logs": logs,
                },
            )
            return code

    _write_result(
        cfg.out_dir,
        {
            "status": "completed",
            "pool_set": cfg.pool_set,
            "eval_set": cfg.eval_set,
            "agentic_timeline": str(agentic_path),
            "pseudo_yaml": str(pseudo_path),
            "accepted": report.accepted,
            "total": report.total,
            "dropped": report.dropped,
            "logs": logs,
            "smoke_only": cfg.smoke_only,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-set", required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--base-timeline", type=Path, required=True)
    ap.add_argument("--synthetic-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--reuse-agentic", action="store_true")
    args = ap.parse_args(argv)

    cfg = E1Config(
        pool_set=args.pool_set,
        eval_set=args.eval_set,
        base_timeline=args.base_timeline,
        synthetic_root=args.synthetic_root,
        out_dir=args.out_dir,
        smoke_only=args.smoke_only,
        reuse_agentic=args.reuse_agentic,
    )
    return run_e1(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

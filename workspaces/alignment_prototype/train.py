#!/usr/bin/env python3
"""CLI entry for the P5 aligner prototype (offline).

Modes:
  --dry-run     dataset summary
  (default)     CopyGT sanity check on full YAML
  --eval        held-out slot split + baseline metrics (P5 checklist)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.result import Err, Ok
from workspaces.alignment_prototype.dataset import load_set, slot_candidates_from_targets
from workspaces.alignment_prototype.eval import evaluate
from workspaces.alignment_prototype.losses import batch_loss
from workspaces.alignment_prototype.model import (
    CopyGTBaseline,
    MeanSpanBaseline,
    NullIdentityBaseline,
)
from workspaces.alignment_prototype.split import split_targets

DEFAULT_YAML = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"


def _torch_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _run_mert_eval(
    gt_set_id: str,
    train: tuple,
    eval_: tuple,
    slot_pools: dict,
    *,
    refresh_mert: bool,
) -> int:
    from workspaces.alignment_prototype.mert_model import TrainConfig, build_aligner
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    print(f"loading MERT from pi-storage for {gt_set_id}…")
    match load_bb12_mert(gt_set_id, refresh=refresh_mert):
        case Err(msg):
            print(f"MERT load failed: {msg}", file=sys.stderr)
            return 1
        case Ok((_sid, mix, refs)):
            print(f"  mix measures={mix.n_measures} ref tracks={len(refs)}")

    device = _torch_device()
    cfg = TrainConfig(epochs=40, search_margin_s=90.0)
    print(f"training MertAlignHead on {len(train)} spans (device={device}, margin={cfg.search_margin_s}s)…")
    aligner = build_aligner(
        train,
        mix,
        refs,
        slot_pools,
        cfg=cfg,
        device=device,
    )
    preds = aligner.predict(eval_)
    report = evaluate(preds, eval_)
    print("\nMertLearnedAligner (eval):")
    for line in report.lines():
        print(f"  {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    p.add_argument("--dry-run", action="store_true", help="Print dataset summary only")
    p.add_argument(
        "--eval",
        action="store_true",
        help="Held-out eval: train/eval split + baseline metrics",
    )
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Fraction of base slots held out (default 0.2)",
    )
    p.add_argument(
        "--train-mert",
        action="store_true",
        help="Train MERT span+identity head and eval on held-out split",
    )
    p.add_argument(
        "--refresh-mert",
        action="store_true",
        help="Re-fetch MERT cache from pi-storage",
    )
    args = p.parse_args(argv)

    match load_set(args.yaml):
        case Err(msg):
            print(f"load failed: {msg}", file=sys.stderr)
            return 1
        case Ok((gt, targets)):
            pass

    slots = slot_candidates_from_targets(targets)
    unresolved = sum(1 for t in targets if not t.recording_id)
    print(f"set_id={gt.set_id} spans={len(targets)} slots_with_candidates={len(slots)} unresolved={unresolved}")

    if args.dry_run:
        for t in targets[:5]:
            print(f"  {t.slot_label:6} {t.claimed_stem:12} {_fmt(t.set_start_s)}–{_fmt(t.set_end_s)}  {t.label[:40]}")
        if len(targets) > 5:
            print(f"  ... +{len(targets) - 5} more")
        return 0

    if args.eval:
        train, eval_ = split_targets(targets, eval_fraction=args.eval_fraction)
        print(f"split train={len(train)} eval={len(eval_)} (eval_fraction={args.eval_fraction})")
        cases: tuple[tuple[str, object, tuple], ...] = (
            ("CopyGT (eval)", CopyGTBaseline(), eval_),
            ("MeanSpan oracle-id (eval)", MeanSpanBaseline(train), eval_),
            ("NullIdentity oracle-span (eval)", NullIdentityBaseline(), eval_),
        )
        for name, model, tgts in cases:
            preds = model.predict(tgts)
            report = evaluate(preds, tgts)
            print(f"\n{name}:")
            for line in report.lines():
                print(f"  {line}")
        if args.train_mert:
            slots = slot_candidates_from_targets(targets)
            rc = _run_mert_eval(
                gt.set_id,
                train,
                eval_,
                slots,
                refresh_mert=args.refresh_mert,
            )
            if rc != 0:
                return rc
        return 0

    model = CopyGTBaseline()
    preds = model.predict(targets)
    loss = batch_loss(preds, targets)
    print(f"CopyGTBaseline loss={loss:.6f} (expect 0.0)")
    return 0 if loss == 0.0 else 1


def _fmt(sec: float) -> str:
    m = int(sec // 60)
    return f"{m}:{sec - m * 60:06.3f}"


if __name__ == "__main__":
    sys.exit(main())

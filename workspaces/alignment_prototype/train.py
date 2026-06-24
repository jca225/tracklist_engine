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
from workspaces.alignment_prototype.dataset import (
    load_set,
    slot_candidates_from_targets,
)
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
    targets: tuple,
    train: tuple,
    eval_: tuple,
    slot_pools: dict,
    *,
    refresh_mert: bool,
    pretrain_checkpoint: Path | None = None,
) -> int:
    from workspaces.alignment_prototype.external.checkpoint import load_head
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
    init_head = None
    if pretrain_checkpoint is not None:
        try:
            init_head, meta = load_head(
                pretrain_checkpoint, device=device, expected_dim=mix.dim
            )
            print(f"  pretrain checkpoint dim={meta.dim} ({meta.feature_kind})")
        except ValueError as exc:
            print(f"  pretrain not loaded: {exc}", file=sys.stderr)
    print(
        f"training MertAlignHead on {len(train)} spans (device={device}, margin={cfg.search_margin_s}s)…"
    )
    aligner = build_aligner(
        train,
        mix,
        refs,
        slot_pools,
        cfg=cfg,
        device=device,
        init=init_head,
    )
    preds = aligner.predict(eval_)
    report = evaluate(preds, eval_)
    print("\nMertLearnedAligner (eval):")
    for line in report.lines():
        print(f"  {line}")

    # Sequence decode runs over ALL spans (placement needs the full tiling
    # context — no GT used at decode time), scored on the held-out subset.
    all_preds = aligner.predict_sequence(targets)
    eval_ids = {id(t) for t in eval_}
    seq_pairs = [(p, t) for p, t in zip(all_preds, targets) if id(t) in eval_ids]
    seq_report = evaluate(
        tuple(p for p, _ in seq_pairs),
        tuple(t for _, t in seq_pairs),
    )
    print("\nMertLearnedAligner + monotonic decode (eval):")
    for line in seq_report.lines():
        print(f"  {line}")

    # Identity over ALL spans (train + eval): within-slot swaps live on
    # training spans (058/059) and are invisible to the held-out report.
    train_ids = {id(t) for t in train}
    resolved = [(p, t) for p, t in zip(all_preds, targets) if t.recording_id]
    misses = [
        (p, t)
        for p, t in resolved
        if (p.recording_id, p.claimed_stem) != (t.recording_id, t.claimed_stem)
    ]
    print(
        f"\nall-span identity (sequence decode): {len(resolved) - len(misses)}/{len(resolved)}"
    )
    for p, t in misses:
        split = "train" if id(t) in train_ids else "eval"
        print(
            f"  MISS slot={t.slot_label} ({split}) gt={t.recording_id}/{t.claimed_stem}"
            f" pred={p.recording_id}/{p.claimed_stem} gt_start={t.set_start_s:.0f}s"
        )

    # Decoupled per-span DTW fine refinement (audio). Snaps each coarse start to
    # the best local match within a ±band corridor; no-op if the aligning folder
    # (mix_instrumental.flac + manifest) is absent.
    from workspaces.alignment_prototype.fine_refine import (
        AudioContext,
        refine_placements,
    )

    ctx = AudioContext.from_set(gt_set_id)
    if ctx is None:
        print("\n(fine refinement skipped — no aligning audio for this set)")
        return 0

    import numpy as np

    def _dist(name: str, preds_seq, base_seq=None) -> None:
        errs, base = [], []
        for p, b, t in zip(preds_seq, base_seq or preds_seq, targets):
            if id(t) not in eval_ids or t.recording_id is None:
                continue
            errs.append(abs(p.set_start_s - t.set_start_s))
            base.append(abs(b.set_start_s - t.set_start_s))
        e = np.asarray(errs)
        tag = ""
        if base_seq is not None:
            be = np.asarray(base)
            better = int((e < be - 3).sum())
            worse = int((e > be + 3).sum())
            tag = f"  [{better} better / {worse} worse / {len(e) - better - worse} tie vs coarse]"
        print(
            f"  {name:34} n={len(e):2d} median={np.median(e):5.1f}s mean={e.mean():5.1f}s "
            f"<8s:{(e < 8).sum()} <16s:{(e < 16).sum()} <30s:{(e < 30).sum()} max={e.max():.0f}s{tag}"
        )

    print("\nfine-placement distribution (eval, set_start error):")
    _dist("coarse monotonic decode", all_preds)
    for band_s, gate_z in ((30.0, None), (45.0, None), (30.0, 1.0)):
        refined = refine_placements(
            all_preds, targets, ctx, band_s=band_s, gate_z=gate_z
        )
        gate = "no gate" if gate_z is None else f"gate z≥{gate_z}"
        _dist(f"per-span DTW ±{band_s:.0f}s ({gate})", refined, base_seq=all_preds)

    from workspaces.alignment_prototype.fp_placement_refine import (
        FpPlacementContext,
        refine_placements_fp,
    )

    mix_mid = 0.5 * (mix.start_s + mix.end_s)
    fp_ctx = FpPlacementContext.from_set(gt_set_id, measure_mid_s=mix_mid)
    if fp_ctx is None:
        print("\n(fp placement skipped — no aligning audio for this set)")
    else:
        print("\nfp placement distribution (eval, set_start error):")
        _dist("coarse monotonic decode", all_preds)
        fp_refined = refine_placements_fp(all_preds, fp_ctx, band_s=45.0, gate_z=1.0)
        _dist(
            "per-span fp ±45s (gate z≥1.0)",
            fp_refined,
            base_seq=all_preds,
        )
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
    p.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Warm-start MertAlignHead from UnmixDB pretrain (see pretrain.py)",
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
    print(
        f"set_id={gt.set_id} spans={len(targets)} slots_with_candidates={len(slots)} unresolved={unresolved}"
    )

    if args.dry_run:
        for t in targets[:5]:
            print(
                f"  {t.slot_label:6} {t.claimed_stem:12} {_fmt(t.set_start_s)}–{_fmt(t.set_end_s)}  {t.label[:40]}"
            )
        if len(targets) > 5:
            print(f"  ... +{len(targets) - 5} more")
        return 0

    if args.eval:
        train, eval_ = split_targets(targets, eval_fraction=args.eval_fraction)
        print(
            f"split train={len(train)} eval={len(eval_)} (eval_fraction={args.eval_fraction})"
        )
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
                targets,
                train,
                eval_,
                slots,
                refresh_mert=args.refresh_mert,
                pretrain_checkpoint=args.pretrain_checkpoint,
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

#!/usr/bin/env python3
"""Pretrain MertAlignHead on UnmixDB or synthetic mashup corpus, then ablate.

UnmixDB download: https://zenodo.org/records/1422385
Synthetic generator: workspaces.alignment_prototype.synthetic_mix.generate

Examples:
  # Parse-only / pipeline smoke (no audio):
  venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain --dry-run \\
    --unmixdb-root /path/to/unmixdb-v1.1

  # Fast chroma pretrain (validates the loop; weights won't transfer to MERT):
  venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain \\
    --unmixdb-root /path/to/unmixdb-v1.1 --features chroma --max-mixes 50 \\
    --out workspaces/alignment_prototype/.cache/pretrain_chroma.pt

  # MERT pretrain for weight-transfer ablation on BB12:
  venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain \\
    --unmixdb-root /path/to/unmixdb-v1.1 --features mert --max-mixes 100 \\
    --out workspaces/alignment_prototype/.cache/pretrain_mert.pt

  # Decisive ablation (frozen BB12 held-out):
  venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain --ablation \\
    --pretrain-checkpoint workspaces/alignment_prototype/.cache/pretrain_mert.pt

  # Synthetic corpus pretrain (fast chroma smoke):
  venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain \\
    --synthetic-root data/synthetic_mixes --features chroma --max-mixes 50 \\
    --out workspaces/alignment_prototype/.cache/pretrain_synthetic_chroma.pt
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
from workspaces.alignment_prototype.external.checkpoint import (
    PretrainMeta,
    load_head,
    save_head,
)
from workspaces.alignment_prototype.external.feature_series import build_mix_bundle
from workspaces.alignment_prototype.external.unmixdb import (
    _recording_id,
    iter_mixes,
    labels_to_targets,
    slot_pools_for_mix,
    summarize_mixes,
)
from workspaces.alignment_prototype.mert_features import build_examples
from workspaces.alignment_prototype.mert_model import (
    TrainConfig,
    build_aligner,
    train_ensemble,
)
from workspaces.alignment_prototype.mert_store import load_bb12_mert
from workspaces.alignment_prototype.split import split_targets

DEFAULT_YAML = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"
DEFAULT_OUT = Path(__file__).resolve().parent / ".cache" / "pretrain_unmixdb.pt"


def _torch_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collect_unmix_examples(
    mixes,
    *,
    feature_kind: str,
    device: str,
    search_margin_s: float,
    synthetic: bool = False,
):
    from workspaces.alignment_prototype.mert_features import MertSpanExample

    if synthetic:
        from workspaces.alignment_prototype.synthetic_mix.corpus import (
            slot_pools_for_mix as synthetic_slot_pools,
            targets_for_mix,
        )
        from workspaces.alignment_prototype.external.unmixdb import _recording_id

    examples: list[MertSpanExample] = []
    skipped = 0
    for mix in mixes:
        if synthetic:
            ref_paths = {
                _recording_id(sp.filename): mix.track_audio[sp.track_idx]
                for sp in mix.spans
            }
        else:
            ref_paths = {
                _recording_id(sp.filename): mix.track_audio[sp.track_idx]
                for sp in mix.spans
            }
        try:
            mix_series, refs = build_mix_bundle(
                mix.mix_audio,
                ref_paths,
                feature_kind=feature_kind,
                device=device,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            skipped += 1
            print(f"  skip {mix.mix_id}: {exc}", file=sys.stderr)
            continue
        if synthetic:
            targets = targets_for_mix(mix)
            pools = synthetic_slot_pools(mix)
        else:
            targets = labels_to_targets(mix)
            pools = slot_pools_for_mix(mix)
        rows = build_examples(
            targets,
            mix_series,
            refs,
            pools,
            search_margin_s=search_margin_s,
        )
        examples.extend(rows)
    return examples, skipped


def run_pretrain(args: argparse.Namespace) -> int:
    if args.synthetic_root is not None:
        from workspaces.alignment_prototype.synthetic_mix.corpus import (
            iter_mixes as iter_synthetic,
            summarize_mixes as summarize_synthetic,
        )

        mixes = iter_synthetic(args.synthetic_root, max_mixes=args.max_mixes)
        print(summarize_synthetic(mixes))
    elif args.unmixdb_root is not None:
        mixes = iter_mixes(
            args.unmixdb_root,
            good_only=not args.all_mixes,
            max_mixes=args.max_mixes,
        )
        print(summarize_mixes(mixes))
    else:
        print("--unmixdb-root or --synthetic-root required", file=sys.stderr)
        return 1
    if args.dry_run:
        for mix in mixes[:3]:
            print(f"  {mix.mix_id} spans={len(mix.spans)} mix={mix.mix_audio.name}")
        if len(mixes) > 3:
            print(f"  ... +{len(mixes) - 3} more")
        return 0 if mixes else 1

    device = _torch_device()
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        search_margin_s=args.search_margin_s,
        n_heads=args.n_heads,
    )
    print(
        f"building examples feature={args.features} device={device} "
        f"margin={cfg.search_margin_s}s …"
    )
    examples, skipped = collect_unmix_examples(
        mixes,
        feature_kind=args.features,
        device=device,
        search_margin_s=cfg.search_margin_s,
        synthetic=args.synthetic_root is not None,
    )
    print(f"examples={len(examples)} skipped_mixes={skipped}")
    if not examples:
        print("no training examples", file=sys.stderr)
        return 1

    head = train_ensemble(tuple(examples), cfg=cfg, device=device)
    dim = examples[0].mix_segment.shape[0]
    meta = PretrainMeta(
        feature_kind=args.features,
        dim=dim,
        n_heads=cfg.n_heads if cfg.n_heads > 1 else 1,
        n_examples=len(examples),
        n_mixes=len(mixes) - skipped,
    )
    save_head(head, args.out, meta=meta, cfg=cfg)
    print(f"wrote {args.out} dim={dim} {meta.to_json()}")
    return 0


def _bb12_eval(
    *,
    yaml_path: Path,
    eval_fraction: float,
    pretrained: Path | None,
    device: str,
    cfg: TrainConfig,
    refresh_mert: bool,
    label: str,
) -> tuple[int, object | None]:
    match load_set(yaml_path):
        case Err(msg):
            print(f"load failed: {msg}", file=sys.stderr)
            return 1, None
        case Ok((gt, targets)):
            pass

    train, eval_ = split_targets(targets, eval_fraction=eval_fraction)
    slots = slot_candidates_from_targets(targets)

    match load_bb12_mert(gt.set_id, refresh=refresh_mert):
        case Err(msg):
            print(f"MERT load failed: {msg}", file=sys.stderr)
            return 1, None
        case Ok((_sid, mix, refs)):
            pass

    init_head = None
    if pretrained is not None:
        try:
            init_head, meta = load_head(pretrained, device=device, expected_dim=mix.dim)
            print(
                f"{label}: loaded pretrain dim={meta.dim} ({meta.feature_kind}, n={meta.n_examples})"
            )
        except ValueError as exc:
            print(f"{label}: pretrain not loaded — {exc}", file=sys.stderr)

    aligner = build_aligner(
        train,
        mix,
        refs,
        slots,
        cfg=cfg,
        device=device,
        init=init_head,
    )

    all_preds = aligner.predict_sequence(targets)
    eval_ids = {id(t) for t in eval_}
    seq_pairs = [(p, t) for p, t in zip(all_preds, targets) if id(t) in eval_ids]
    report = evaluate(
        tuple(p for p, _ in seq_pairs),
        tuple(t for _, t in seq_pairs),
    )
    print(f"\n=== {label} (BB12 held-out, monotonic decode) ===")
    for line in report.lines():
        print(f"  {line}")
    return 0, report


def run_ablation(args: argparse.Namespace) -> int:
    device = _torch_device()
    cfg = TrainConfig(epochs=args.epochs, search_margin_s=90.0, n_heads=args.n_heads)
    rc0, finetune_only = _bb12_eval(
        yaml_path=args.yaml,
        eval_fraction=args.eval_fraction,
        pretrained=None,
        device=device,
        cfg=cfg,
        refresh_mert=args.refresh_mert,
        label="finetune-only",
    )
    if rc0 != 0:
        return rc0
    rc1, pretrain_finetune = _bb12_eval(
        yaml_path=args.yaml,
        eval_fraction=args.eval_fraction,
        pretrained=args.pretrain_checkpoint,
        device=device,
        cfg=cfg,
        refresh_mert=args.refresh_mert,
        label="pretrain→finetune",
    )
    if rc1 != 0:
        return rc1

    if finetune_only and pretrain_finetune:
        print("\n=== ablation delta (pretrain→finetune minus finetune-only) ===")
        print(
            f"  identity_acc: {pretrain_finetune.identity_accuracy - finetune_only.identity_accuracy:+.1%}"
        )
        print(
            f"  MAE set_start: {pretrain_finetune.mean_abs_set_start_s - finetune_only.mean_abs_set_start_s:+.3f}s"
        )
        print(
            f"  batch_loss: {pretrain_finetune.batch_loss - finetune_only.batch_loss:+.4f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--unmixdb-root",
        type=Path,
        default=None,
        help="Path to UnmixDB root or v1.1 dir",
    )
    p.add_argument(
        "--synthetic-root",
        type=Path,
        default=None,
        help="Path to synthetic mashup corpus (from synthetic_mix.generate)",
    )
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Checkpoint output path"
    )
    p.add_argument("--features", choices=("chroma", "mert"), default="chroma")
    p.add_argument("--max-mixes", type=int, default=None)
    p.add_argument(
        "--all-mixes", action="store_true", help="Include v1 mixes with long silences"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument(
        "--search-margin-s", type=float, default=45.0, help="Unmix mixes are ~100s"
    )
    p.add_argument(
        "--n-heads", type=int, default=3, help="Seed ensemble size (smaller for speed)"
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Run finetune-only vs pretrain→finetune on BB12",
    )
    p.add_argument("--pretrain-checkpoint", type=Path, default=None)
    p.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--refresh-mert", action="store_true")
    args = p.parse_args(argv)

    if args.ablation:
        if args.pretrain_checkpoint is None:
            print("--ablation requires --pretrain-checkpoint", file=sys.stderr)
            return 1
        return run_ablation(args)
    return run_pretrain(args)


if __name__ == "__main__":
    sys.exit(main())

"""Train the trajectory decoder; two split protocols.

    # set-level holdout (honest cross-set generalization):
    venvs/audio/bin/python -m workspaces.alignment_prototype.trajectory.train \
        --split set --train-set 1fsnxchk --eval-set 2nvzlh2k

    # slot-level pooled split (each mashup is an example; more train signal):
    venvs/audio/bin/python -m workspaces.alignment_prototype.trajectory.train \
        --split slot --val-frac 0.2 --seed 0

`--split slot` pools every GT set and holds out by BASE slot (002 groups with
002w1 ...): layered spans of one slot overlap in time, so splitting them
apart would leak the answer. Set-level holdout remains the honest
generalization number; the slot split trades a little leakage (same mix,
same DJ) for ~4x the training examples.

Eval decodes greedily (per-frame argmax -> collapse to segments) and scores
with the pipeline's own `path_decode.trajectory_acc` (strict, no fibers —
fiber scoring needs HuBERT labels per ref and belongs to the full scorer).
A no-model control (argmax of the raw match-similarity channel) prints
alongside: the learned decoder must beat it or it has learned nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.path_decode import trajectory_acc  # noqa: E402
from workspaces.alignment_prototype.trajectory.data import (  # noqa: E402
    GT_FIXTURES,
    TrajectorySpanDataset,
    collate_spans,
)
from workspaces.alignment_prototype.trajectory.model import TrajectoryDecoder  # noqa: E402
from workspaces.alignment_prototype.trajectory.recon_loss import (  # noqa: E402
    reconstruction_loss,
    trajectory_ce,
)
from workspaces.alignment_prototype.trajectory.decode import (  # noqa: E402
    viterbi_segments,
)
from workspaces.alignment_prototype.trajectory.targets import (  # noqa: E402
    frames_to_segments,
)
from workspaces.alignment_prototype.trajectory.offset_coords import (  # noqa: E402
    OffsetVocab,
)
from workspaces.alignment_prototype.trajectory.trm import (  # noqa: E402
    TRMDecoder,
    trm_decode_segments,
    trm_offset_ce,
    trm_offset_targets,
)

CKPT_DIR = Path(__file__).resolve().parents[1] / ".cache" / "trajectory"


class SpanSubset:
    """Index view over a TrajectorySpanDataset (keeps .specs/.bin_s contract)."""

    def __init__(self, base: TrajectorySpanDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = list(indices)
        self.bin_s = base.bin_s
        self.specs = [base.specs[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> dict:
        return self.base[self.indices[k]]


def base_slot(slot_label: str, label: str) -> str:
    """'154w1' -> '154'; empty slot labels fall back to the track label."""
    s = slot_label.partition("w")[0]
    return s if s else label


def slot_split(
    ds: TrajectorySpanDataset, val_frac: float, seed: int
) -> tuple[SpanSubset, SpanSubset]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, spec in enumerate(ds.specs):
        key = (
            spec.set_id,
            base_slot(
                str(spec.row.get("slot_label") or ""), str(spec.row.get("track"))
            ),
        )
        groups[key].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = int(round(val_frac * len(ds)))
    val_idx: list[int] = []
    train_idx: list[int] = []
    for k in keys:
        (val_idx if len(val_idx) < n_val else train_idx).extend(groups[k])
    return SpanSubset(ds, sorted(train_idx)), SpanSubset(ds, sorted(val_idx))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decode_segments(
    logits: torch.Tensor, tr: int, bin_s: float
) -> list[tuple[float, float, float]]:
    """Greedy per-frame decode of (Tm, Tr+1) logits into segment triples."""
    arg = logits.argmax(dim=-1).cpu().numpy()  # (Tm,)
    null_mask = arg >= tr
    return frames_to_segments(arg.clip(max=tr - 1), null_mask, bin_s)


_FIBER_CACHE: dict[str, tuple] = {}


def _fibers_for(ref_audio: str) -> tuple:
    """HuBERT self-repeat classes for one ref (same recipe as the full
    scorer, score_timeline_vs_gt --fibers). Cached per file; the HuBERT
    forward itself persists in .feat_cache."""
    if ref_audio not in _FIBER_CACHE:
        from workspaces.alignment_prototype.path_decode import FPS, _ensure_feat
        from workspaces.alignment_prototype.ref_fibers import compute_fibers

        hf = np.load(_ensure_feat(ref_audio, ref_audio, "hubert", 9))
        _FIBER_CACHE[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
    return _FIBER_CACHE[ref_audio]


@torch.no_grad()
def evaluate(
    model: TrajectoryDecoder | None,
    ds: TrajectorySpanDataset,
    device: torch.device,
    tag: str,
    lam: float | None = None,
    back_ratio: float = 3.0,
    fibers: bool = False,
) -> dict[str, float]:
    """Mean strict trajectory_acc per span_class and stem. model=None = the
    raw-similarity control (argmax of match channel, no NULL). lam=None =
    greedy argmax decode; otherwise offset-state Viterbi with that jump cost.
    fibers=True additionally reports repeat-equivalent accuracy (the
    hand-decoder bar is fiber-aware; strict-vs-strict understates us)."""
    if model is not None:
        model.eval()
    by_class: dict[str, list[float]] = defaultdict(list)
    by_stem: dict[str, list[float]] = defaultdict(list)
    alls: list[float] = []
    fib_alls: list[float] = []
    fib_by_class: dict[str, list[float]] = defaultdict(list)
    for i in range(len(ds)):
        x = ds[i]
        if bool(x["abstain"]):
            continue
        sim = x["sim"][None].to(device)
        tr = sim.shape[-1]
        if isinstance(model, TRMDecoder):
            # TRM emits offset-coord logits; argmax-collapse to segments (no
            # Viterbi/lam — the offset run-length IS the segmentation).
            rv = torch.ones(1, tr, dtype=torch.bool, device=device)
            out = model(sim, x["feat_kind"][None].to(device), rv)
            segs = trm_decode_segments(out, ds.bin_s, model.vocab)
        else:
            if model is None:
                grid = sim[0, 0]  # (Tm, Tr) raw match similarity
                logits = torch.cat([grid, torch.full_like(grid[:, :1], -1e9)], dim=-1)
            else:
                rv = torch.ones(1, tr, dtype=torch.bool, device=device)
                logits = model(sim, x["feat_kind"][None].to(device), rv)[0]
            if lam is None:
                segs = decode_segments(logits, tr, ds.bin_s)
            else:
                segs = viterbi_segments(
                    logits, ds.bin_s, lam=lam, back_ratio=back_ratio
                )
        fiber = _fibers_for(str(ds.specs[i].audio.ref_path)) if fibers else None
        acc, _, fib_acc = trajectory_acc(segs, ds.specs[i].row, fiber=fiber)
        alls.append(acc)
        by_class[x["meta"]["span_class"]].append(acc)
        by_stem[x["meta"]["claimed_stem"]].append(acc)
        if fibers:
            fib_alls.append(fib_acc)
            fib_by_class[x["meta"]["span_class"]].append(fib_acc)

    def _m(v: list[float]) -> float:
        return float(np.mean(v)) if v else float("nan")

    headline = _m(by_class.get("multiseg", []) + by_class.get("loop", []))
    print(
        f"  [{tag}] traj-acc ALL {_m(alls):.3f} | HEADLINE multiseg+loop {headline:.3f}"
    )
    if fibers:
        fib_headline = _m(
            fib_by_class.get("multiseg", []) + fib_by_class.get("loop", [])
        )
        print(
            f"    FIBER-AWARE: ALL {_m(fib_alls):.3f} | "
            f"HEADLINE multiseg+loop {fib_headline:.3f}"
        )
    print(
        "    class: "
        + "  ".join(
            f"{k} {_m(v):.3f} (n={len(v)})" for k, v in sorted(by_class.items())
        )
    )
    print(
        "    stem:  "
        + "  ".join(f"{k} {_m(v):.3f} (n={len(v)})" for k, v in sorted(by_stem.items()))
    )
    return {"all": _m(alls), "headline": headline}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("set", "slot"), default="set")
    ap.add_argument("--train-set", required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--recon-weight", type=float, default=0.5)
    ap.add_argument("--bin-s", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="2 epochs, first 8 spans")
    ap.add_argument(
        "--synthetic-root",
        default=None,
        metavar="DIR",
        help="Fold synthetic-mix windows (generate_v2 output) into the TRAIN "
        "loader only; eval stays on real held-out GT.",
    )
    ap.add_argument(
        "--eval-only",
        metavar="CKPT",
        default=None,
        help="skip training; load this checkpoint and run the decode A/B",
    )
    ap.add_argument(
        "--lam-sweep",
        default="8,16,32",
        help="Viterbi jump costs swept on the TRAIN split (best applied to eval)",
    )
    ap.add_argument(
        "--back-sweep",
        default="1,2,3",
        help="backward-jump cost ratios swept with lam (loops ARE backward jumps)",
    )
    ap.add_argument(
        "--slopes",
        default="",
        help="stretch-indexed accumulation channels, e.g. '0.5,0.95,1.0,1.05,2.0' "
        "(must match between training and --eval-only)",
    )
    ap.add_argument(
        "--fibers",
        action="store_true",
        help="fiber-aware scoring on the FINAL eval-set evaluations (sweep stays "
        "strict); one HuBERT pass per ref on first use",
    )
    ap.add_argument(
        "--model",
        choices=("conv", "trm"),
        default="conv",
        help="decoder: conv+Viterbi baseline, or the TRM recursive refiner over "
        "offset coordinates (docs/trm_decoder_bakeoff.md)",
    )
    ap.add_argument("--trm-dmodel", type=int, default=64)
    ap.add_argument("--trm-nsup", type=int, default=3, help="deep-supervision steps")
    ap.add_argument("--trm-ninner", type=int, default=6, help="inner recursion n")
    ap.add_argument(
        "--trm-T", type=int, default=3, help="recursion rounds (all backprop)"
    )
    ap.add_argument("--trm-ymode", choices=("once", "every"), default="once")
    ap.add_argument("--trm-off-lo", type=int, default=-256, help="offset-vocab low bin")
    ap.add_argument(
        "--trm-off-hi", type=int, default=1408, help="offset-vocab high bin"
    )
    ap.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="truncate the train split to the first N spans (0 = all); for the "
        "v0 overfit sanity (few spans, many epochs)",
    )
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    print(f"device: {device}")

    if args.split == "slot":
        pooled = TrajectorySpanDataset(
            [(sid, GT_FIXTURES[sid]) for sid in sorted(GT_FIXTURES)], bin_s=args.bin_s
        )
        print(f"pooled: {len(pooled)} spans; skipped:")
        print(pooled.report_skipped())
        train_ds, eval_ds = slot_split(pooled, args.val_frac, args.seed)
        print(
            f"slot split (seed {args.seed}): train {len(train_ds)} spans, "
            f"eval {len(eval_ds)} spans"
        )
        eval_tag = f"heldout-slots seed{args.seed}"
        ckpt_tag = f"slotsplit_seed{args.seed}"
    else:
        train_ds = TrajectorySpanDataset(
            [(args.train_set, GT_FIXTURES[args.train_set])], bin_s=args.bin_s
        )
        eval_ds = TrajectorySpanDataset(
            [(args.eval_set, GT_FIXTURES[args.eval_set])], bin_s=args.bin_s
        )
        print(f"train {args.train_set}: {len(train_ds)} spans; skipped:")
        print(train_ds.report_skipped())
        print(f"eval  {args.eval_set}: {len(eval_ds)} spans; skipped:")
        print(eval_ds.report_skipped())
        eval_tag = args.eval_set
        ckpt_tag = args.train_set

    if args.smoke:
        train_ds = SpanSubset(
            train_ds.base if isinstance(train_ds, SpanSubset) else train_ds,
            (
                train_ds.indices
                if isinstance(train_ds, SpanSubset)
                else list(range(len(train_ds)))
            )[:8],
        )
        eval_ds = SpanSubset(
            eval_ds.base if isinstance(eval_ds, SpanSubset) else eval_ds,
            (
                eval_ds.indices
                if isinstance(eval_ds, SpanSubset)
                else list(range(len(eval_ds)))
            )[:8],
        )
        args.epochs = 2

    if args.max_train and len(train_ds) > args.max_train:
        base = train_ds.base if isinstance(train_ds, SpanSubset) else train_ds
        idxs = (
            train_ds.indices
            if isinstance(train_ds, SpanSubset)
            else list(range(len(train_ds)))
        )
        train_ds = SpanSubset(base, idxs[: args.max_train])
        eval_ds = train_ds  # overfit sanity: eval == train, watch it memorize
        print(f"max-train: overfitting {len(train_ds)} spans (eval == train)")

    # Synthetic mashups augment the gradient path only — never train_ds itself,
    # so the lam-sweep and train-eval below still report on real GT.
    loader_ds = train_ds
    if args.synthetic_root:
        from torch.utils.data import ConcatDataset

        from workspaces.alignment_prototype.trajectory.synthetic_adapter import (
            build_synthetic_sets,
        )

        syn_sets, syn_dirs = build_synthetic_sets(args.synthetic_root)
        synth_ds = TrajectorySpanDataset(
            syn_sets, bin_s=args.bin_s, aligning_dirs=syn_dirs
        )
        print(
            f"synthetic: +{len(synth_ds)} train spans from {len(syn_sets)} windows; "
            f"skipped:\n{synth_ds.report_skipped()}"
        )
        loader_ds = ConcatDataset([train_ds, synth_ds])

    loader = DataLoader(
        loader_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_spans,
        num_workers=0,  # features are cached in-process; workers would re-pool
    )

    slopes = tuple(float(v) for v in args.slopes.split(",") if v.strip())
    if args.model == "trm":
        vocab = OffsetVocab(lo_bin=args.trm_off_lo, hi_bin=args.trm_off_hi)
        model = TRMDecoder(
            vocab,
            d_model=args.trm_dmodel,
            n_sup=args.trm_nsup,
            n_inner=args.trm_ninner,
            T=args.trm_T,
            y_mode=args.trm_ymode,
        ).to(device)
        print(
            f"TRM: offset-vocab {vocab.size} classes ({vocab.lo_bin}..{vocab.hi_bin})"
        )
    else:
        model = TrajectoryDecoder(stretch_slopes=slopes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")

    if args.eval_only:
        state = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(state["model"])
        print(f"loaded {args.eval_only}\n\n== decode A/B on checkpoint ==")
        evaluate(model, eval_ds, device, f"argmax eval {eval_tag}", fibers=args.fibers)
        best, best_acc = None, -1.0
        for lam in [float(v) for v in args.lam_sweep.split(",")]:
            for br in [float(v) for v in args.back_sweep.split(",")]:
                r = evaluate(
                    model,
                    train_ds,
                    device,
                    f"viterbi lam={lam:g} back={br:g} TRAIN",
                    lam=lam,
                    back_ratio=br,
                )
                if r["all"] > best_acc:
                    best, best_acc = (lam, br), r["all"]
        lam, br = best
        print(f"\nbest on train: lam={lam:g} back={br:g} (train ALL {best_acc:.3f})")
        evaluate(
            model,
            eval_ds,
            device,
            f"viterbi lam={lam:g} back={br:g} eval {eval_tag}",
            lam=lam,
            back_ratio=br,
            fibers=args.fibers,
        )
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("\n== no-model control (raw match-sim argmax) ==")
    evaluate(None, eval_ds, device, f"control {eval_tag}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ce_sum = rc_sum = 0.0
        n_batches = 0
        for batch in loader:
            sim = batch["sim"].to(device)
            if isinstance(model, TRMDecoder):
                out = model(
                    sim,
                    batch["feat_kind"].to(device),
                    batch["ref_valid"].to(device),
                )
                labels = trm_offset_targets(
                    batch["target_idx"].to(device),
                    batch["target_null"].to(device),
                    batch["mix_valid"].to(device),
                    model.vocab,
                )
                ce = trm_offset_ce(out, labels)
                rc = torch.zeros((), device=device)  # recon aux deferred (v1)
            else:
                logits = model(
                    sim,
                    batch["feat_kind"].to(device),
                    batch["ref_valid"].to(device),
                )
                ce = trajectory_ce(
                    logits,
                    batch["target_idx"].to(device),
                    batch["target_null"].to(device),
                    batch["mix_valid"].to(device),
                    batch["ref_valid"].to(device),
                )
                rc = reconstruction_loss(
                    logits,
                    batch["ref_mel"].to(device),
                    batch["mix_mel"].to(device),
                    batch["mix_valid"].to(device),
                    batch["ref_valid"].to(device),
                    batch["recon_ok"].to(device),
                )
            loss = ce + args.recon_weight * rc
            opt.zero_grad()
            loss.backward()
            if isinstance(model, TRMDecoder):
                # the recursion blows up unclipped (bake-off trap)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ce_sum += float(ce.detach())
            rc_sum += float(rc.detach())
            n_batches += 1
        print(
            f"epoch {epoch:3d}  ce {ce_sum / n_batches:.4f}"
            f"  recon {rc_sum / n_batches:.4f}"
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate(model, eval_ds, device, f"eval {eval_tag} ep{epoch}")
            evaluate(model, train_ds, device, f"train ep{epoch}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"decoder_{ckpt_tag}.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
    print(f"\ncheckpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

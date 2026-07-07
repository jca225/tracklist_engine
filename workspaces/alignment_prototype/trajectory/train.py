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
from workspaces.alignment_prototype.trajectory.targets import (  # noqa: E402
    frames_to_segments,
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


@torch.no_grad()
def evaluate(
    model: TrajectoryDecoder | None,
    ds: TrajectorySpanDataset,
    device: torch.device,
    tag: str,
) -> dict[str, float]:
    """Mean strict trajectory_acc per span_class and stem. model=None = the
    raw-similarity control (argmax of match channel, no NULL)."""
    if model is not None:
        model.eval()
    by_class: dict[str, list[float]] = defaultdict(list)
    by_stem: dict[str, list[float]] = defaultdict(list)
    alls: list[float] = []
    for i in range(len(ds)):
        x = ds[i]
        if bool(x["abstain"]):
            continue
        sim = x["sim"][None].to(device)
        tr = sim.shape[-1]
        if model is None:
            grid = sim[0, 0]  # (Tm, Tr) raw match similarity
            logits = torch.cat([grid, torch.full_like(grid[:, :1], -1e9)], dim=-1)
        else:
            rv = torch.ones(1, tr, dtype=torch.bool, device=device)
            logits = model(sim, x["feat_kind"][None].to(device), rv)[0]
        segs = decode_segments(logits, tr, ds.bin_s)
        acc, _, _ = trajectory_acc(segs, ds.specs[i].row)
        alls.append(acc)
        by_class[x["meta"]["span_class"]].append(acc)
        by_stem[x["meta"]["claimed_stem"]].append(acc)

    def _m(v: list[float]) -> float:
        return float(np.mean(v)) if v else float("nan")

    headline = _m(by_class.get("multiseg", []) + by_class.get("loop", []))
    print(
        f"  [{tag}] traj-acc ALL {_m(alls):.3f} | HEADLINE multiseg+loop {headline:.3f}"
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
    ap.add_argument("--train-set", default="1fsnxchk")
    ap.add_argument("--eval-set", default="2nvzlh2k")
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

    loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_spans,
        num_workers=0,  # features are cached in-process; workers would re-pool
    )

    model = TrajectoryDecoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("\n== no-model control (raw match-sim argmax) ==")
    evaluate(None, eval_ds, device, f"control {eval_tag}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ce_sum = rc_sum = 0.0
        n_batches = 0
        for batch in loader:
            sim = batch["sim"].to(device)
            logits = model(
                sim, batch["feat_kind"].to(device), batch["ref_valid"].to(device)
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

"""Multi-set co-training: concatenate per-set MERT examples, train one head.

The head trains on set-agnostic materialized examples (`build_examples`), so
co-train is a concat + single train_ensemble. LOSO wraps the head per held-out
set (see train.py --loso)."""

from __future__ import annotations

from dataclasses import dataclass

from workspaces.alignment_prototype.mert_model import (
    TrainConfig,
    build_examples,
    train_ensemble,
)
from workspaces.alignment_prototype.records import SpanTarget


@dataclass(frozen=True)
class SetStores:
    set_id: str
    train_spans: tuple[SpanTarget, ...]
    mix: object  # MertSeries
    refs: dict
    slot_pools: dict


def cotrain(
    train_sets, *, cfg: TrainConfig | None = None, device: str = "cpu", init=None
):
    cfg = cfg or TrainConfig()
    all_examples: list = []
    for s in train_sets:
        all_examples.extend(
            build_examples(
                s.train_spans,
                s.mix,
                s.refs,
                s.slot_pools,
                search_margin_s=cfg.search_margin_s,
            )
        )
    return train_ensemble(tuple(all_examples), cfg=cfg, device=device, init=init)

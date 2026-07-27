"""Save/load pretrained MertAlignHead checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..mert_model import MertAlignEnsemble, MertAlignHead, TrainConfig


@dataclass(frozen=True)
class PretrainMeta:
    feature_kind: str
    dim: int
    n_heads: int
    n_examples: int
    n_mixes: int
    source: str = "unmixdb"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def save_head(
    head: MertAlignHead | MertAlignEnsemble,
    path: Path | str,
    *,
    meta: PretrainMeta,
    cfg: TrainConfig | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = cfg or TrainConfig()
    payload = {
        "meta": asdict(meta),
        "cfg": asdict(cfg),
        "n_heads": meta.n_heads,
        "state_dicts": (
            [h.state_dict() for h in head.heads]
            if isinstance(head, MertAlignEnsemble)
            else [head.state_dict()]
        ),
    }
    torch.save(payload, out)
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    sidecar.write_text(meta.to_json() + "\n")


def load_head(
    path: Path | str,
    *,
    device: str = "cpu",
    expected_dim: int | None = None,
) -> tuple[MertAlignHead | MertAlignEnsemble, PretrainMeta]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    meta = PretrainMeta(**payload["meta"])
    if expected_dim is not None and meta.dim != expected_dim:
        raise ValueError(
            f"checkpoint dim={meta.dim} != expected {expected_dim} "
            f"(pretrain used {meta.feature_kind}; BB12 finetune needs mert features)"
        )
    heads: list[MertAlignHead] = []
    for sd in payload["state_dicts"]:
        dim = int(sd["mix_id.weight"].shape[0])
        model = MertAlignHead(dim).to(device)
        model.load_state_dict(sd)
        model.eval()
        heads.append(model)
    if len(heads) == 1:
        return heads[0], meta
    ens = MertAlignEnsemble(heads)
    ens.eval()
    return ens, meta

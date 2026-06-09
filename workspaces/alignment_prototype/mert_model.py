"""Learned MERT retrieval head for span + identity (P5 prototype)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mert_features import MertSpanExample, build_examples, median_duration_by_slot, slide_duration
from .mert_store import MertSeries
from .records import SpanPrediction, SpanTarget
from .slot_priors import slot_anchor


class MertAlignHead(nn.Module):
    """Bilinear scorer for identity + per-measure span logits."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mix_id = nn.Linear(dim, dim, bias=False)
        self.ref_id = nn.Linear(dim, dim, bias=False)
        self.mix_span = nn.Linear(dim, dim, bias=False)
        self.ref_span = nn.Linear(dim, dim, bias=False)

    def identity_logits(self, mix_seg: torch.Tensor, ref_segs: torch.Tensor) -> torch.Tensor:
        """mix_seg (B, D), ref_segs (B, C, D) -> (B, C)."""
        m = F.normalize(self.mix_id(mix_seg), dim=-1)
        r = F.normalize(self.ref_id(ref_segs), dim=-1)
        return (m.unsqueeze(1) * r).sum(dim=-1)

    def span_logits(self, mix_measures: torch.Tensor, ref_seg: torch.Tensor) -> torch.Tensor:
        """mix_measures (B, T, D), ref_seg (B, D) -> (B, T)."""
        m = F.normalize(self.mix_span(mix_measures), dim=-1)
        r = F.normalize(self.ref_span(ref_seg), dim=-1).unsqueeze(1)
        return (m * r).sum(dim=-1)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    lr: float = 3e-3
    identity_weight: float = 1.0
    span_weight: float = 1.0
    search_margin_s: float = 90.0


def _pad_batch(
    examples: tuple[MertSpanExample, ...],
) -> tuple[torch.Tensor, ...]:
    max_c = max(len(ex.candidate_ids) for ex in examples)
    max_t = max(ex.mix_window_vectors.shape[0] for ex in examples)
    dim = examples[0].mix_segment.shape[0]
    b = len(examples)

    mix_seg = torch.zeros(b, dim)
    ref_segs = torch.zeros(b, max_c, dim)
    id_mask = torch.zeros(b, max_c, dtype=torch.bool)
    span_x = torch.zeros(b, max_t, dim)
    span_mask = torch.zeros(b, max_t)
    span_valid = torch.zeros(b, max_t, dtype=torch.bool)
    pos_idx = torch.zeros(b, dtype=torch.long)

    for i, ex in enumerate(examples):
        mix_seg[i] = torch.from_numpy(ex.mix_segment)
        c = len(ex.candidate_ids)
        ref_segs[i, :c] = torch.from_numpy(ex.ref_segments)
        id_mask[i, :c] = True
        t = ex.mix_window_vectors.shape[0]
        span_x[i, :t] = torch.from_numpy(ex.mix_window_vectors)
        span_mask[i, :t] = torch.from_numpy(ex.span_mask)
        span_valid[i, :t] = True
        pos_idx[i] = ex.positive_idx

    return mix_seg, ref_segs, id_mask, span_x, span_mask, span_valid, pos_idx


def train_head(
    examples: tuple[MertSpanExample, ...],
    *,
    cfg: TrainConfig | None = None,
    device: str = "cpu",
) -> MertAlignHead:
    if not examples:
        raise ValueError("no training examples")
    cfg = cfg or TrainConfig()
    dim = examples[0].mix_segment.shape[0]
    model = MertAlignHead(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for _epoch in range(cfg.epochs):
        model.train()
        mix_seg, ref_segs, id_mask, span_x, span_mask, span_valid, pos_idx = _pad_batch(examples)
        mix_seg = mix_seg.to(device)
        ref_segs = ref_segs.to(device)
        id_mask = id_mask.to(device)
        span_x = span_x.to(device)
        span_mask = span_mask.to(device)
        span_valid = span_valid.to(device)
        pos_idx = pos_idx.to(device)

        id_logits = model.identity_logits(mix_seg, ref_segs)
        id_logits = id_logits.masked_fill(~id_mask, -1e9)
        id_loss = F.cross_entropy(id_logits, pos_idx)

        ref_pos = ref_segs[torch.arange(len(examples), device=device), pos_idx]
        span_logits = model.span_logits(span_x, ref_pos)
        span_logits = span_logits.masked_fill(~span_valid, -1e9)
        target = span_mask / span_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        log_p = F.log_softmax(span_logits, dim=1)
        span_loss = -(target * log_p).sum(dim=1).mean()

        loss = cfg.identity_weight * id_loss + cfg.span_weight * span_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model


@dataclass(frozen=True)
class MertLearnedAligner:
    head: MertAlignHead
    mix: MertSeries
    refs: dict[str, MertSeries]
    slot_medians: dict[str, float]
    slot_pools: dict[str, tuple]
    train_medians: dict[str, float]
    search_margin_s: float = 90.0
    device: str = "cpu"

    def predict(self, targets: tuple[SpanTarget, ...]) -> tuple[SpanPrediction, ...]:
        from .dataset import slot_candidates_from_targets

        pools = self.slot_pools or slot_candidates_from_targets(targets)
        mix_mid = 0.5 * (self.mix.start_s + self.mix.end_s)

        with torch.no_grad():
            return tuple(self._predict_one(t, pools, mix_mid) for t in targets)

    def _predict_one(
        self,
        t: SpanTarget,
        pools: dict[str, tuple],
        mix_mid: np.ndarray,
    ) -> SpanPrediction:
        from .mert_features import candidate_list

        cand_ids, cand_stems = candidate_list(t.slot_label, pools, ())
        if not cand_ids:
            all_ids = tuple(sorted(self.refs))
            cand_ids, cand_stems = candidate_list(t.slot_label, pools, all_ids)
        if not cand_ids:
            return _fallback_pred(t)

        ref_segs = []
        for cid in cand_ids:
            rs = self.refs.get(cid)
            ref_segs.append(rs.track_mean() if rs else np.zeros(self.mix.dim, dtype=np.float32))
        ref_arr = np.stack(ref_segs, axis=0).astype(np.float32)

        dur = slide_duration(t, self.slot_medians)
        anchor = slot_anchor(t.slot_label, train_medians=self.train_medians)
        margin = self.search_margin_s * (1.5 if "w" in t.slot_label else 1.0)
        lo = max(0.0, anchor - margin)
        hi = min(float(mix_mid[-1]), anchor + margin + dur)
        band = np.where((mix_mid >= lo) & (mix_mid <= hi))[0]
        if band.size == 0:
            band = np.arange(self.mix.n_measures)

        mix_t = torch.from_numpy(self.mix.vectors).to(self.device)
        ref_t = torch.from_numpy(ref_arr).unsqueeze(0).to(self.device)

        best_score = -1e9
        best_rid = cand_ids[0]
        best_stem = cand_stems[0]
        best_start = float(mix_mid[band[0]])
        best_end = best_start + dur

        anchor_i = int(band[np.argmin(np.abs(mix_mid[band] - anchor))])
        end_i = int(np.searchsorted(mix_mid, float(mix_mid[anchor_i]) + dur, side="right"))
        end_i = min(max(end_i, anchor_i + 1), self.mix.n_measures)
        anchor_win = mix_t[anchor_i:end_i]
        if anchor_win.shape[0] > 0:
            id_at_anchor = self.head.identity_logits(anchor_win.mean(dim=0, keepdim=True), ref_t)[0]
            best_ci = int(id_at_anchor.argmax())
            best_rid = cand_ids[best_ci]
            best_stem = cand_stems[best_ci]
        else:
            best_ci = 0

        ref_vec = ref_t[:, best_ci]
        for start_i in band:
            end_s = float(mix_mid[start_i]) + dur
            end_i = int(np.searchsorted(mix_mid, end_s, side="right"))
            end_i = min(max(end_i, start_i + 1), self.mix.n_measures)
            win = mix_t[start_i:end_i]
            if win.shape[0] == 0:
                continue
            span_logits = self.head.span_logits(win.unsqueeze(0), ref_vec)
            score = float(span_logits.max())
            if score > best_score:
                best_score = score
                best_start = float(mix_mid[start_i])
                best_end = float(mix_mid[end_i - 1])

        ref_start = 0.0
        ref_end = None
        if best_rid in self.refs:
            ref_start = float(self.refs[best_rid].start_s[0])
            ref_end = ref_start + dur

        return SpanPrediction(
            slot_label=t.slot_label,
            recording_id=best_rid,
            claimed_stem=best_stem,
            set_start_s=best_start,
            set_end_s=best_end,
            ref_start_s=ref_start,
            ref_end_s=ref_end,
            confidence=float(best_score),
        )


def _fallback_pred(t: SpanTarget) -> SpanPrediction:
    return SpanPrediction(
        slot_label=t.slot_label,
        recording_id=t.recording_id,
        claimed_stem=t.claimed_stem,
        set_start_s=t.set_start_s,
        set_end_s=t.set_end_s,
        ref_start_s=t.ref_start_s,
        ref_end_s=t.ref_end_s,
        confidence=0.0,
    )


def build_aligner(
    train_targets: tuple[SpanTarget, ...],
    mix: MertSeries,
    refs: dict[str, MertSeries],
    slot_pools: dict[str, tuple],
    *,
    cfg: TrainConfig | None = None,
    device: str = "cpu",
) -> MertLearnedAligner:
    cfg = cfg or TrainConfig()
    from .slot_priors import median_start_by_label

    train_starts = median_start_by_label(train_targets)
    examples = build_examples(
        train_targets,
        mix,
        refs,
        slot_pools,
        search_margin_s=cfg.search_margin_s,
    )
    head = train_head(examples, cfg=cfg, device=device)
    return MertLearnedAligner(
        head=head,
        mix=mix,
        refs=refs,
        slot_medians=median_duration_by_slot(train_targets),
        slot_pools=slot_pools,
        train_medians=train_starts,
        search_margin_s=cfg.search_margin_s,
        device=device,
    )

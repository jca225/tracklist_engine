"""Learned MERT retrieval head for span + identity (P5 prototype)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

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


class MertAlignEnsemble(nn.Module):
    """Average identity/span logits over independently-seeded heads.

    A 40-epoch head on ~130 examples is init-sensitive — single-seed runs
    flip individual identity picks. Averaging logits removes that variance.
    """

    def __init__(self, heads: list[MertAlignHead]) -> None:
        super().__init__()
        if not heads:
            raise ValueError("empty ensemble")
        self.heads = nn.ModuleList(heads)

    def identity_logits(self, mix_seg: torch.Tensor, ref_segs: torch.Tensor) -> torch.Tensor:
        return torch.stack([h.identity_logits(mix_seg, ref_segs) for h in self.heads]).mean(dim=0)

    def span_logits(self, mix_measures: torch.Tensor, ref_seg: torch.Tensor) -> torch.Tensor:
        return torch.stack([h.span_logits(mix_measures, ref_seg) for h in self.heads]).mean(dim=0)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    lr: float = 3e-3
    identity_weight: float = 1.0
    span_weight: float = 1.0
    search_margin_s: float = 90.0
    n_heads: int = 5  # seed-ensemble size; 1 = single head


def _ref_window_pools(
    rs: MertSeries, dur_s: float, max_windows: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled sliding ~dur_s windows over a ref series -> ((W, dim), (W,) start_s).

    Identity scoring against a candidate's best window (MaxSim) instead of its
    track_mean: two acappellas have near-identical global stats (slot 039),
    and training positives are span pools, not track means — windows match
    both the discriminative need and the training distribution.
    """
    mid = 0.5 * (rs.start_s + rs.end_s)
    n = rs.n_measures
    step = max(1, n // max_windows)
    pools: list[np.ndarray] = []
    starts: list[float] = []
    for i in range(0, n, step):
        j = int(np.searchsorted(mid, mid[i] + dur_s, side="right"))
        j = min(max(j, i + 1), n)
        pools.append(rs.vectors[i:j].mean(axis=0))
        starts.append(float(rs.start_s[i]))
    return np.stack(pools, axis=0).astype(np.float32), np.array(starts, dtype=np.float64)


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
    seed: int | None = None,
) -> MertAlignHead:
    if not examples:
        raise ValueError("no training examples")
    cfg = cfg or TrainConfig()
    if seed is not None:
        torch.manual_seed(seed)
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


def train_ensemble(
    examples: tuple[MertSpanExample, ...],
    *,
    cfg: TrainConfig | None = None,
    device: str = "cpu",
) -> MertAlignHead | MertAlignEnsemble:
    """Train `cfg.n_heads` independently-seeded heads; average their logits."""
    cfg = cfg or TrainConfig()
    if cfg.n_heads <= 1:
        return train_head(examples, cfg=cfg, device=device, seed=0)
    heads = [
        train_head(examples, cfg=cfg, device=device, seed=s) for s in range(cfg.n_heads)
    ]
    ens = MertAlignEnsemble(heads)
    ens.eval()
    return ens


@dataclass(frozen=True)
class MertLearnedAligner:
    head: MertAlignHead | MertAlignEnsemble
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

        # Decode per slot, not per span: a slot played twice with different
        # recordings (BB12 slot 039) is undecidable span-by-span because every
        # span of the label shares the same anchor + pool.
        groups: dict[str, list[int]] = {}
        for i, t in enumerate(targets):
            groups.setdefault(t.slot_label, []).append(i)

        preds: list[SpanPrediction | None] = [None] * len(targets)
        with torch.no_grad():
            for idxs in groups.values():
                group = tuple(targets[i] for i in idxs)
                for i, p in zip(idxs, self._predict_slot(group, pools, mix_mid)):
                    preds[i] = p
        return tuple(preds)  # type: ignore[arg-type]

    def _predict_slot(
        self,
        ts: tuple[SpanTarget, ...],
        pools: dict[str, tuple],
        mix_mid: np.ndarray,
    ) -> tuple[SpanPrediction, ...]:
        from .mert_features import candidate_list

        t0 = ts[0]
        cand_ids, cand_stems = candidate_list(t0.slot_label, pools, ())
        if not cand_ids:
            all_ids = tuple(sorted(self.refs))
            cand_ids, cand_stems = candidate_list(t0.slot_label, pools, all_ids)
        if not cand_ids:
            return tuple(_fallback_pred(t) for t in ts)

        dur = slide_duration(t0, self.slot_medians)

        ref_windows: list[np.ndarray] = []  # per candidate (W, dim)
        for cid in cand_ids:
            rs = self.refs.get(cid)
            if rs is None:
                log.warning(
                    "predict slot=%s: candidate %s has no MERT embedding — "
                    "zero-filled, cannot win identity",
                    t0.slot_label,
                    cid,
                )
                ref_windows.append(np.zeros((1, self.mix.dim), dtype=np.float32))
            else:
                wins, _starts = _ref_window_pools(rs, dur)
                ref_windows.append(wins)

        anchor = slot_anchor(t0.slot_label, train_medians=self.train_medians)
        margin = self.search_margin_s * (1.5 if "w" in t0.slot_label else 1.0)
        lo = max(0.0, anchor - margin)
        hi = min(float(mix_mid[-1]), anchor + margin + dur)
        band = np.where((mix_mid >= lo) & (mix_mid <= hi))[0]
        if band.size == 0:
            band = np.arange(self.mix.n_measures)

        mix_t = torch.from_numpy(self.mix.vectors).to(self.device)

        # Pooled sliding mix window at every band start: (Wm, dim).
        win_starts: list[int] = []
        win_ends: list[int] = []
        win_pools: list[torch.Tensor] = []
        for start_i in band:
            end_s = float(mix_mid[start_i]) + dur
            end_i = int(np.searchsorted(mix_mid, end_s, side="right"))
            end_i = min(max(end_i, start_i + 1), self.mix.n_measures)
            win = mix_t[start_i:end_i]
            if win.shape[0] == 0:
                continue
            win_pools.append(win.mean(dim=0))
            win_starts.append(int(start_i))
            win_ends.append(end_i)
        if not win_pools:
            return tuple(_fallback_pred(t) for t in ts)
        mix_wins = torch.stack(win_pools)  # (Wm, dim)

        # Identity by max-over-placements: each candidate's score is its best
        # (mix window, ref window) pair in the band — MaxSim on both sides.
        cand_score: list[float] = []
        cand_loc: list[int] = []      # index into win_starts
        cand_ref_win: list[int] = []  # best ref window per candidate
        for wins in ref_windows:
            wt = torch.from_numpy(wins).to(self.device)  # (Wr, dim)
            ref_b = wt.unsqueeze(0).expand(mix_wins.shape[0], -1, -1)
            lg = self.head.identity_logits(mix_wins, ref_b)  # (Wm, Wr)
            flat = int(lg.argmax())
            cand_loc.append(flat // lg.shape[1])
            cand_ref_win.append(flat % lg.shape[1])
            cand_score.append(float(lg.max()))

        # Assign the slot's k spans to the top-k distinct candidates, ordered
        # by where each candidate matched in mix time (tracklist span order).
        k = len(ts)
        by_score = sorted(range(len(cand_ids)), key=lambda c: -cand_score[c])
        chosen = by_score[:k]
        while len(chosen) < k:  # pool smaller than span count: repeat best
            chosen.append(by_score[0])
        chosen.sort(key=lambda c: cand_loc[c])

        return tuple(
            self._place_candidate(t, ci, cand_ids, cand_stems, ref_windows,
                                  cand_ref_win, cand_score, band, mix_t, mix_mid, dur)
            for t, ci in zip(ts, chosen)
        )

    def _place_candidate(
        self,
        t: SpanTarget,
        ci: int,
        cand_ids: tuple[str, ...],
        cand_stems: tuple[str, ...],
        ref_windows: list[np.ndarray],
        cand_ref_win: list[int],
        cand_score: list[float],
        band: np.ndarray,
        mix_t: torch.Tensor,
        mix_mid: np.ndarray,
        dur: float,
    ) -> SpanPrediction:
        best_rid = cand_ids[ci]
        best_stem = cand_stems[ci]
        ref_vec = (
            torch.from_numpy(ref_windows[ci][cand_ref_win[ci]])
            .unsqueeze(0)
            .to(self.device)
        )

        best_score = -1e9
        best_start = float(mix_mid[band[0]])
        best_end = best_start + dur
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
            # Track head, not the best-matching window: BB12 GT ref_starts sit
            # near 0 almost everywhere, and the window estimate is noisy
            # (MAE 0.8s vs 75s when tried).
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
            confidence=float(cand_score[ci]),
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
    head = train_ensemble(examples, cfg=cfg, device=device)
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

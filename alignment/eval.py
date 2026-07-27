"""Held-out evaluation metrics for span alignment (P5 verification)."""
from __future__ import annotations

from dataclasses import dataclass

from .losses import batch_loss, span_placement_loss
from .records import SpanPrediction, SpanTarget


@dataclass(frozen=True)
class EvalReport:
    n_spans: int
    n_resolved: int
    mean_abs_set_start_s: float
    mean_abs_set_end_s: float
    mean_abs_ref_start_s: float
    mean_span_loss: float
    identity_accuracy: float
    batch_loss: float

    def lines(self) -> tuple[str, ...]:
        return (
            f"spans={self.n_spans} resolved={self.n_resolved}",
            f"MAE set_start={self.mean_abs_set_start_s:.3f}s "
            f"set_end={self.mean_abs_set_end_s:.3f}s "
            f"ref_start={self.mean_abs_ref_start_s:.3f}s",
            f"mean_span_loss={self.mean_span_loss:.4f} "
            f"identity_acc={self.identity_accuracy:.1%} "
            f"batch_loss={self.batch_loss:.4f}",
        )


def evaluate(
    preds: tuple[SpanPrediction, ...],
    targets: tuple[SpanTarget, ...],
    *,
    identity_weight: float = 1.0,
) -> EvalReport:
    if len(preds) != len(targets):
        raise ValueError(f"pred/target length mismatch: {len(preds)} vs {len(targets)}")
    if not targets:
        return EvalReport(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    abs_start = abs_end = abs_ref = span_losses = 0.0
    id_ok = id_total = 0
    resolved = 0

    for p, t in zip(preds, targets):
        abs_start += abs(p.set_start_s - t.set_start_s)
        abs_end += abs(p.set_end_s - t.set_end_s)
        abs_ref += abs(p.ref_start_s - t.ref_start_s)
        span_losses += span_placement_loss(p, t)
        if t.recording_id:
            resolved += 1
            id_total += 1
            if (p.recording_id, p.claimed_stem) == (t.recording_id, t.claimed_stem):
                id_ok += 1

    n = len(targets)
    return EvalReport(
        n_spans=n,
        n_resolved=resolved,
        mean_abs_set_start_s=abs_start / n,
        mean_abs_set_end_s=abs_end / n,
        mean_abs_ref_start_s=abs_ref / n,
        mean_span_loss=span_losses / n,
        identity_accuracy=id_ok / id_total if id_total else 0.0,
        batch_loss=batch_loss(preds, targets, identity_weight=identity_weight),
    )

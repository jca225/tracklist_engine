"""Multi-set co-training: concatenate per-set MERT examples, train one head.

The head trains on set-agnostic materialized examples (`build_examples`), so
co-train is a concat + single train_ensemble. LOSO wraps the head per held-out
set (see train.py --loso)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from alignment.eval import evaluate
from alignment.mert_model import (
    MertLearnedAligner,
    TrainConfig,
    build_examples,
    median_duration_by_slot,
    train_ensemble,
)
from alignment.records import SpanTarget


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


def _load_set_stores(set_id_or_yaml) -> "SetStores":
    """Load a set's (targets, mix, refs, slot_pools) into SetStores.

    Accepts either:
    - a yaml Path / path-like (passed through to load_set directly), or
    - a bare set_id string (e.g. "bb11") — resolved to
      labeling/fixtures/<set_id>_ground_truth.yaml relative to the repo root.

    SSHes pi-storage for MERT (result is cached locally).
    """
    from core.result import Err, Ok
    from alignment.dataset import (
        load_set,
        slot_candidates_from_targets,
    )
    from alignment.mert_store import load_bb12_mert

    p = Path(set_id_or_yaml)
    if not p.suffix:
        # Bare set_id — resolve to canonical fixture path.
        # alignment/cotrain.py → repo root (avoid parents[N])
        _repo = Path(__file__).resolve().parent.parent.parent
        p = _repo / "labeling" / "fixtures" / f"{set_id_or_yaml}_ground_truth.yaml"

    match load_set(p):
        case Err(msg):
            raise RuntimeError(f"load_set failed: {msg}")
        case Ok((gt, targets)):
            pass
    match load_bb12_mert(gt.set_id):
        case Err(msg):
            raise RuntimeError(f"MERT load failed for {gt.set_id}: {msg}")
        case Ok((_sid, mix, refs)):
            pass
    pools = slot_candidates_from_targets(targets)
    return SetStores(gt.set_id, targets, mix, refs, pools)


def _cue_anchor(set_id: str) -> dict:
    """{slot_label: cue_seconds} from scraped tracklist cues (aligner INPUT, not
    GT). Returns {} if unavailable."""
    try:
        from alignment.infer import fetch_slot_rows

        rows = fetch_slot_rows(set_id)
        out = {}
        for r in rows:
            # fetch_slot_rows emits the cue under key "cue_s" (not "cue_seconds").
            cue = r.get("cue_s") if isinstance(r, dict) else None
            if cue not in (None, ""):
                # Key on the label as fetch_slot_rows provides it (already passed
                # through normalize_slot). The anchor lookup normalizes GT labels
                # the same way, so these exact-match; do NOT re-pad (that forces
                # the coarser interpolation fallback and degrades placement).
                out[r["slot_label"]] = float(cue)
        return out
    except Exception:  # noqa: BLE001
        return {}


def run_loso(
    set_yamls: dict,
    *,
    cfg: TrainConfig | None = None,
    device: str = "cpu",
    anchor_from_cues: bool = True,
) -> dict:
    """Leave-one-set-out co-train eval.

    For each held-out set: cotrain on the others, wrap the head around the
    held-out stores with the scraped-cue anchor, predict_sequence on the
    held-out spans, evaluate. Returns a per-held-out-set report dict.
    """
    stores: dict[str, SetStores] = {sid: _load_set_stores(sid) for sid in set_yamls}
    report = {}
    for held in stores:
        train_sets = [s for sid, s in stores.items() if sid != held]
        head = cotrain(train_sets, cfg=cfg, device=device)
        h = stores[held]
        # Fetch scraped cues by the REAL gt.set_id (h.set_id), not the dict key
        # (a fixture stem like "bb11"); the cue DB is keyed on set_id. Using the
        # stem returns no rows -> empty cues -> silent floor mode.
        cue_anchor = _cue_anchor(h.set_id) if anchor_from_cues else {}
        try:
            slot_medians = median_duration_by_slot(h.train_spans)
        except AttributeError:
            slot_medians = {}
        aligner = MertLearnedAligner(
            head=head,
            mix=h.mix,
            refs=h.refs,
            slot_medians=slot_medians,
            slot_pools=h.slot_pools,
            train_medians=(cue_anchor or {}),
            # 30 s ≈ ±1σ placement prior around each scraped cue; None when cues
            # absent (unseen-set decode collapses to front-of-mix = honest floor).
            anchor_sigma_s=(30.0 if cue_anchor else None),
            device=device,
        )
        preds = aligner.predict_sequence(h.train_spans)
        rep = evaluate(preds, h.train_spans)
        report[held] = {
            "anchor": "cues" if cue_anchor else "none (floor)",
            "lines": list(rep.lines()),
        }
    return report

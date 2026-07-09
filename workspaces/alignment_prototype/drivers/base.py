"""The set-level driver contract: one interface, three interchangeable aligners.

`harness/contract.py` defines the *probe*-level surface (one span×candidate ->
AlignmentResult). This module is the level above it: a whole-set aligner that
consumes the bound set inputs and emits a `PredictedTimeline`-shaped JSON — the
artifact `score_timeline_vs_gt` already judges. Every driver (classical,
agentic, ml) implements `align_set` and writes the SAME schema, so they race on
one scorecard instead of ad-hoc `out/*.json`.

Design: the classical driver is the *base* — it runs identity (MERT) + placement
(fp/lyrics/HuBERT) + a ref-segment decode. The agentic and ml drivers are
*refinements over that base timeline* (agentic re-decides placement + autonomy
rung; ml swaps the ref-segment decoder for the learned TrajectoryDecoder), so
each takes the classical timeline as its starting point. That mirrors reality:
the agentic loop inherits identity from a timeline and only refines placement,
and the ml trajectory decoder only owns the ref-segment stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

_REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parents[1] / "out"
_FIXTURES = _REPO / "labeling" / "fixtures"

# The two sets with complete hand GT. `for_set` trains/transfers cross-set: to
# align X we borrow the OTHER set as the supervised source (honest generalization
# — never train and eval on the same mix). Unknown sets default to BB12 as source.
GT_BY_SET: dict[str, Path] = {
    "1fsnxchk": _FIXTURES / "bb12_ground_truth.yaml",  # BB12
    "2nvzlh2k": _FIXTURES / "bb11_ground_truth.yaml",  # BB11
}


@dataclass(frozen=True)
class SetContext:
    """Everything a driver needs to align one set, resolved up front.

    Heavy data (mix/ref MERT, aligning audio, feature caches) is NOT carried
    here — it is resolved by the underlying modules from `set_id`
    (`find_aligning_dir`, `load_bb12_mert`). This is a config carrier: which set,
    where its GT lives, and which set supplies cross-set supervision.
    """

    set_id: str
    gt_yaml: Path
    source_set_id: str  # cross-set train/transfer source (!= set_id)
    source_yaml: Path
    out_dir: Path = OUT_DIR

    @staticmethod
    def for_set(
        set_id: str,
        *,
        source_set_id: str | None = None,
        out_dir: Path = OUT_DIR,
    ) -> "SetContext":
        if set_id not in GT_BY_SET:
            raise KeyError(
                f"no GT fixture registered for set_id={set_id!r} "
                f"(known: {sorted(GT_BY_SET)}); add it to GT_BY_SET"
            )
        if source_set_id is None:
            # cross-set: borrow another complete GT set as the supervised source
            # (never train + eval on the same mix).
            others = sorted(s for s in GT_BY_SET if s != set_id)
            if not others:
                raise ValueError(f"no cross-set GT source available for {set_id}")
            source_set_id = others[0]
        return SetContext(
            set_id=set_id,
            gt_yaml=GT_BY_SET[set_id],
            source_set_id=source_set_id,
            source_yaml=GT_BY_SET[source_set_id],
            out_dir=out_dir,
        )

    def timeline_path(self, driver_name: str) -> Path:
        return self.out_dir / f"{self.set_id}_{driver_name}_timeline.json"


@runtime_checkable
class EndToEndDriver(Protocol):
    """A whole-set aligner. `align_set` writes a PredictedTimeline JSON and
    returns its path; the returned file MUST pass `core.contracts.load_timeline`."""

    name: str

    def align_set(self, ctx: SetContext) -> Path: ...


# --------------------------------------------------------------------------- #
# shared timeline helpers                                                      #
# --------------------------------------------------------------------------- #


def load_timeline_dict(path: Path) -> dict:
    """Raw JSON of a timeline (for mutation). Validation is `finalize`'s job."""
    return json.loads(Path(path).read_text())


def finalize(payload: dict, path: Path) -> Path:
    """Write + validate a timeline. Round-trips through the contract loader so a
    driver can't emit a schema the scorer will choke on three functions later."""
    from core.contracts import load_timeline  # local: keep msgspec off cold paths

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    load_timeline(path)  # raises msgspec.ValidationError on any consumed-field fault
    return path


def gt_stem_by_slot(gt_yaml: Path) -> dict[str, str]:
    """slot_label (leading-zeros stripped) -> claimed_stem, from GT.

    History: the materialized set_track_slots `claimed_stem` was corrupted by
    the row-text drop bug (BB12: ~2 instrumentals vs ~25 real). The pi DB was
    re-materialized 2026-07-09 (19/25 visible; the residual ~6/set are class-1
    inventory gaps the scrape never marked, GT-only). GT therefore remains
    authoritative wherever it exists: the scorer re-routes the axis from GT,
    and drivers that route DECODE by stem (agentic probe plan, ml feature
    choice) must do the same. Timelines predating the re-materialize still
    carry the stale values.
    """
    import re

    import yaml

    def _norm(x: object) -> str:
        m = re.match(r"^0*(\d+)(.*)$", str(x))
        return (m.group(1) + m.group(2)) if m else str(x)

    doc = yaml.safe_load(Path(gt_yaml).read_text())
    return {
        _norm(r.get("slot_label")): (r.get("claimed_stem") or "regular")
        for r in doc["tracks"]
        if r.get("slot_label") and str(r.get("slot_label")) != "mix"
    }


def norm_slot(x: object) -> str:
    """'001w1' -> '1w1' — match GT (zero-padded) to timeline (not) slot keys."""
    import re

    m = re.match(r"^0*(\d+)(.*)$", str(x))
    return (m.group(1) + m.group(2)) if m else str(x)

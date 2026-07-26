"""Stem-MERT identity feature runner — Phase 1B data dependency (spec §2B).

The blind identity override ([identity_override.py]) needs per-measure MERT at
the tuned identity layers over STEM audio — L3 for vocal (acappella) identity,
L22 for the instrumental backbone (regular/instrumental) — for both the mix
span queries and the set's stem-partitioned reference pool. The repo's stored
MERT is full-track L6 only, so this runner re-extracts at L3/L22 over stems using
[stem_mert.embed_stem_layers] (GPU) and writes an ``IdentityFeatureBundle`` that
the ``infer.py`` seam loads.

Split, per the repo's pure-core pattern:
  * ``plan_extraction`` + ``IdentityFeatureBundle`` (save/load) are PURE — the
    plan is derivable from tracklist rows + a stem-partitioned ref inventory, and
    the bundle round-trips through npz+json — so they unit-test with no GPU/pi.
  * ``run`` is the thin GPU/IO driver (resolve stem audio + measure grids on
    pi/aligning-dir, one MERT pass per stem, save the bundle). It runs on a
    rented gpubox or Mac MPS; it is exercised by the actual extraction, not tests.

Consumes the existing MERT channel at tuned layers — NOT a new sensor (module
CLAUDE.md sensor freeze; the open-set findings class L3/L22 as layer tuning).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from workspaces.alignment_prototype.candidate_pool import pool_stem_for
from workspaces.alignment_prototype.stem_mert import INSTRUMENTAL_LAYER, VOCAL_LAYER

# stem-partition -> the MERT identity layer (open findings decision #22).
_STEM_LAYER = {"acappella": VOCAL_LAYER, "instrumental": INSTRUMENTAL_LAYER}


@dataclass(frozen=True)
class ExtractionItem:
    """One MERT extraction unit: a stem audio + the layer to keep."""

    key: str  # bundle key: "query::<slot_label>" or "ref::<recording_id>"
    kind: str  # "query" | "ref"
    ident: str  # slot_label (query) or recording_id (ref)
    pool_stem: str  # "acappella" | "instrumental" — routes the ref stem + layer
    layer: int


@dataclass(frozen=True)
class ExtractionPlan:
    """Everything to extract for one set, plus the pool partition to persist."""

    set_id: str
    items: tuple[ExtractionItem, ...]
    set_pool_by_stem: dict[str, tuple[str, ...]]
    spans: dict[str, float]  # slot_label -> span seconds (drives conditional top-k)

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({it.layer for it in self.items}))


def plan_extraction(
    set_id: str,
    rows: tuple[dict, ...],
    ref_inventory: dict[str, tuple[str, ...]],
    spans: dict[str, float] | None = None,
) -> ExtractionPlan:
    """Derive the extraction plan (pure).

    ``rows`` are the tracklist spine rows (``slot_label`` + ``claimed_stem``);
    ``ref_inventory`` maps ``"acappella"``/``"instrumental"`` -> the set's ref
    recording_ids of that stem. One query per distinct slot (routed by its
    claimed_stem), plus every ref in the two stem pools. The layer follows the
    pool stem, so acappella queries/refs extract L3 and instrumental L22.
    """
    spans = spans or {}
    items: list[ExtractionItem] = []
    seen_query: set[str] = set()
    for r in rows:
        label = r["slot_label"]
        if label in seen_query:
            continue  # concurrent (`w`) rows share the slot
        seen_query.add(label)
        stem = pool_stem_for(r.get("claimed_stem", "regular"))
        items.append(
            ExtractionItem(
                key=f"query::{label}",
                kind="query",
                ident=label,
                pool_stem=stem,
                layer=_STEM_LAYER[stem],
            )
        )
    seen_ref: set[str] = set()
    for stem, rids in ref_inventory.items():
        if stem not in _STEM_LAYER:
            continue
        for rid in rids:
            if rid in seen_ref:
                continue
            seen_ref.add(rid)
            items.append(
                ExtractionItem(
                    key=f"ref::{rid}",
                    kind="ref",
                    ident=rid,
                    pool_stem=stem,
                    layer=_STEM_LAYER[stem],
                )
            )
    pool = {k: tuple(v) for k, v in ref_inventory.items() if k in _STEM_LAYER}
    return ExtractionPlan(
        set_id=set_id, items=tuple(items), set_pool_by_stem=pool, spans=spans
    )


@dataclass
class IdentityFeatureBundle:
    """The seam's input: per-slot query features, per-ref features, and the pool
    partition, all at the stem-routed identity layer. Arrays are (D, n_measures)
    float32 (the chamfer convention). Persists as ``<dir>/<set_id>_identity.npz``
    (arrays, prefixed ``query::`` / ``ref::``) + a ``.json`` sidecar (pool + spans).
    """

    set_id: str
    queries: dict[str, np.ndarray] = field(default_factory=dict)
    refs: dict[str, np.ndarray] = field(default_factory=dict)
    set_pool_by_stem: dict[str, tuple[str, ...]] = field(default_factory=dict)
    spans: dict[str, float] = field(default_factory=dict)

    def npz_path(self, cache_dir: Path) -> Path:
        return Path(cache_dir) / f"{self.set_id}_identity.npz"

    def json_path(self, cache_dir: Path) -> Path:
        return Path(cache_dir) / f"{self.set_id}_identity.json"

    def save(self, cache_dir: Path) -> Path:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = {f"query::{k}": v for k, v in self.queries.items()}
        arrays.update({f"ref::{k}": v for k, v in self.refs.items()})
        np.savez_compressed(self.npz_path(cache_dir), **arrays)
        self.json_path(cache_dir).write_text(
            json.dumps(
                {
                    "set_id": self.set_id,
                    "set_pool_by_stem": {
                        k: list(v) for k, v in self.set_pool_by_stem.items()
                    },
                    "spans": self.spans,
                }
            )
        )
        return self.npz_path(cache_dir)

    @classmethod
    def load(cls, set_id: str, cache_dir: Path) -> IdentityFeatureBundle:
        cache_dir = Path(cache_dir)
        b = cls(set_id=set_id)
        with np.load(b.npz_path(cache_dir)) as data:
            for key in data.files:
                kind, _, ident = key.partition("::")
                if kind == "query":
                    b.queries[ident] = data[key]
                elif kind == "ref":
                    b.refs[ident] = data[key]
        meta = json.loads(b.json_path(cache_dir).read_text())
        b.set_pool_by_stem = {
            k: tuple(v) for k, v in meta.get("set_pool_by_stem", {}).items()
        }
        b.spans = {k: float(v) for k, v in meta.get("spans", {}).items()}
        return b


def run(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU/IO driver
    """CLI: extract L3/L22 stem MERT for a set and write the feature bundle.

    Resolves the set's mix stems (mix_vocals/mix_instrumental) + the
    stem-partitioned reference audios on pi/aligning-dir, runs one MERT pass per
    stem via ``stem_mert.embed_stem_layers``, slices the routed layer, and saves
    an ``IdentityFeatureBundle``. GPU-bound — run on a rented gpubox or Mac MPS.
    The resolution glue (which audio path / measure grid per item) is the piece
    validated during the extraction run; the plan + bundle above are its spec.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset)")
    args = p.parse_args(argv)
    raise SystemExit(
        "extract_stem_mert.run is the GPU/IO driver: resolve stem audio + measure "
        f"grids for set {args.set_id} (device={args.device or 'auto'}) and call "
        "stem_mert.embed_stem_layers per plan item, then "
        f"IdentityFeatureBundle(...).save({args.cache_dir}). Wire to the "
        "pi/aligning-dir audio resolver during the extraction run (deferred to "
        "the gpubox step)."
    )


if __name__ == "__main__":  # pragma: no cover
    run()

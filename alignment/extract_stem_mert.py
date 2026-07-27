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

from alignment.candidate_pool import pool_stem_for
from alignment.stem_mert import INSTRUMENTAL_LAYER, VOCAL_LAYER

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


# ---- Pure resolution (unit-tested; no FS/DB/GPU) ------------------------------


def measure_times_from_series(start_s, end_s) -> tuple[float, ...]:
    """N+1 measure boundaries from a MertSeries' per-measure start/end arrays
    (the grid the stored MERT was computed on — Demucs stems share the source's
    timeline, so it applies to the stem). Empty -> ()."""
    starts = [float(x) for x in start_s]
    if not starts:
        return ()
    return tuple(starts) + (float(end_s[-1]),)


def span_measure_mask(start_s, end_s, span_start: float, span_end: float):
    """Boolean mask over measures whose midpoint falls in [span_start, span_end]
    — used to slice the mix query to a slot's play span."""
    mids = [0.5 * (float(a) + float(b)) for a, b in zip(start_s, end_s)]
    return [span_start <= m <= span_end for m in mids]


def resolve_parent_cues(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    """Fill acappella/``w``-row cues from their parent slot.

    Acappella rows are concurrent overlays (e.g. ``001w1`` layered over ``001``)
    and carry cue 0 / NULL in the DB — their timing is the parent slot's. Without
    this every acappella query windows the same front-of-mix segment (identical
    queries → chance identity). Idempotent: a non-``w`` slot is its own parent.
    """
    import re

    cue_by_label = {r["slot_label"]: r.get("cue_s") for r in rows}
    out: list[dict] = []
    for r in rows:
        r = dict(r)
        if not r.get("cue_s"):
            parent = re.sub(r"w\d+$", "", r["slot_label"])
            if parent != r["slot_label"] and cue_by_label.get(parent):
                r["cue_s"] = cue_by_label[parent]
        out.append(r)
    return tuple(out)


def acappella_targets(rows: tuple[dict, ...]) -> list[dict]:
    """The set's acappella slots (claimed_stem == 'acappella') with a claim —
    the only slots the acappella-only extraction touches."""
    return [
        r
        for r in rows
        if pool_stem_for(r.get("claimed_stem")) == "acappella" and r.get("recording_id")
    ]


def vocals_stem_path(aligning_dir, slot_label: str, name: str):
    """Path to a ref's Demucs vocals stem in the aligning dir. Tries the canonical
    ``stems/<slot>__<name>/vocals.flac`` then globs ``stems/<slot>__*/vocals.flac``
    (aligning dirs carry tag-suffixed variant folders) and takes the shortest
    (least-annotated) match. Returns None if no vocals stem exists."""
    from pathlib import Path as _P

    stems = _P(aligning_dir) / "stems"
    canonical = stems / f"{slot_label}__{name}" / "vocals.flac"
    if canonical.exists():
        return canonical
    cands = sorted(
        stems.glob(f"{slot_label}__*/vocals.flac"), key=lambda p: len(str(p))
    )
    return cands[0] if cands else None


# ---- I/O driver (GPU/audio; exercised by the extraction run) -------------------


def run(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU/IO driver
    """Extract L3 acappella stem-MERT for a set and write the feature bundle.

    ACAPPELLA-ONLY first cut (the contested axis; instrumental identity is already
    strong). For every acappella slot: compute L3 MERT over the ref's Demucs
    vocals stem (grid = that recording's stored MERT measure grid) and over the
    mix-vocal window for that slot's play span, and save an
    ``IdentityFeatureBundle`` the ``infer.py`` seam loads. Device auto-selects
    cuda->mps->cpu; run on a rented gpubox (fast) or Mac MPS.
    """
    import argparse

    import librosa

    from analysis.adapters import mert_adapter
    from core.result import Err, Ok
    from alignment import infer
    from alignment.mert_store import load_bb12_mert
    from alignment.stem_mert import VOCAL_LAYER, embed_stem_layers

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument(
        "--aligning-dir", type=Path, required=True, help="~/aligning/<set>__*"
    )
    p.add_argument("--device", default="auto", help="cuda|mps|cpu|auto")
    p.add_argument("--limit", type=int, default=0, help="cap refs (0=all) for smoke")
    p.add_argument(
        "--rows-json",
        type=Path,
        default=None,
        help="pre-fetched slot rows (JSON list of {slot_label, recording_id, "
        "claimed_stem, cue_s, name}); avoids SSHing pi on a box with no pi access",
    )
    args = p.parse_args(argv)
    sr = mert_adapter.MERT_SR

    match load_bb12_mert(args.set_id):
        case Err(msg):
            print(f"MERT bundle load failed: {msg}", file=sys.stderr)
            return 1
        case Ok((_sid, mix, refs)):
            pass

    if args.rows_json is not None:
        rows = tuple(json.loads(Path(args.rows_json).read_text()))
    else:
        rows = infer.fetch_slot_rows(args.set_id)
    # Acappella w-rows inherit the parent slot's cue; without this the query
    # windows collapse to the front of the mix (identical → chance identity).
    rows = resolve_parent_cues(rows)
    mix_end = float(mix.end_s[-1]) if len(mix.end_s) else 0.0
    targets, _anchors, _medians = infer.build_stub_targets(rows, mix_end)
    span_by_slot = {t.slot_label: (t.set_start_s, t.set_end_s) for t in targets}
    aca = acappella_targets(rows)
    if args.limit:
        aca = aca[: args.limit]
    print(f"acappella slots: {len(aca)}  (device resolving via {args.device})")

    match mert_adapter.load(device=args.device):
        case Err(e):
            print(f"MERT model load failed: {e}", file=sys.stderr)
            return 1
        case Ok(handle):
            pass

    # Mix-vocal L3 once over the whole mix (sliced per slot afterwards).
    mix_vocals = Path(args.aligning_dir) / "mix_vocals.flac"
    if not mix_vocals.exists():
        print(f"missing {mix_vocals}", file=sys.stderr)
        return 1
    mv, _ = librosa.load(str(mix_vocals), sr=sr, mono=True)
    mix_grid = measure_times_from_series(mix.start_s, mix.end_s)
    mix_l3 = embed_stem_layers(handle, mv, mix_grid, layers=(VOCAL_LAYER,))
    if mix_l3 is None or VOCAL_LAYER not in mix_l3:
        print("mix-vocal MERT extraction failed", file=sys.stderr)
        return 1
    mix_l3 = mix_l3[VOCAL_LAYER]  # (dim, n_mix_measures)
    print(f"mix-vocal L3: {mix_l3.shape}")

    bundle = IdentityFeatureBundle(set_id=args.set_id)
    ref_ids: list[str] = []
    for r in aca:
        rid = r["recording_id"]
        slot = r["slot_label"]
        # Query: slice the mix-vocal L3 to this slot's play span.
        span = span_by_slot.get(slot)
        if span:
            mask = span_measure_mask(mix.start_s, mix.end_s, span[0], span[1])
            cols = [i for i, m in enumerate(mask) if m]
            if cols:
                bundle.queries[slot] = mix_l3[:, cols]
                bundle.spans[slot] = float(span[1] - span[0])
        # Ref: L3 over the recording's Demucs vocals stem.
        series = refs.get(rid)
        vpath = vocals_stem_path(args.aligning_dir, slot, r.get("name", ""))
        if series is None or vpath is None:
            print(f"  skip ref {slot}/{rid}: series={series is not None} stem={vpath}")
            continue
        rv, _ = librosa.load(str(vpath), sr=sr, mono=True)
        grid = measure_times_from_series(series.start_s, series.end_s)
        emb = embed_stem_layers(handle, rv, grid, layers=(VOCAL_LAYER,))
        if emb is None or VOCAL_LAYER not in emb:
            print(f"  skip ref {slot}/{rid}: extraction returned None")
            continue
        bundle.refs[rid] = emb[VOCAL_LAYER]
        ref_ids.append(rid)
        print(f"  ref {slot}/{rid}: {emb[VOCAL_LAYER].shape}")

    bundle.set_pool_by_stem = {"acappella": tuple(ref_ids), "instrumental": ()}
    out = bundle.save(args.cache_dir)
    print(
        f"wrote {out}: {len(bundle.queries)} queries, {len(bundle.refs)} refs "
        f"(acappella pool)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())

"""Labeling functions (Snorkel LFs) for the Rust PWS matrix.

Each LF is an ``EvidenceSource`` in spirit: it votes on ``InferenceUnit``s and
must set ``abstained`` explicitly. Emissions cross the boundary as JSON
validated by ``schemas/evidence_emission.json``, then ``dj_migrate pws-fit``
runs majority vote + canary gate in Rust.

Promote canary gold is independent (``canary_stem_gold.json``); the canary
source is ``title_stem_heuristic_lf``, not ``claimed_stem_lf``.
"""

from sensors.lfs.canary_gold import load_canary_stem_gold
from sensors.lfs.claimed_stem import run_claimed_stem_lf, write_vertical_slice
from sensors.lfs.feature_ref import emit_with_feature_ref, write_feature_ref_demo
from sensors.lfs.title_stem import guess_stem_from_title, run_title_stem_heuristic_lf

__all__ = [
    "emit_with_feature_ref",
    "guess_stem_from_title",
    "load_canary_stem_gold",
    "run_claimed_stem_lf",
    "run_title_stem_heuristic_lf",
    "write_feature_ref_demo",
    "write_vertical_slice",
]

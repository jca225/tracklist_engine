"""Success / binding-cause rules — mirrors failure_analysis.analyze."""

from __future__ import annotations

from typing import Any, Mapping

# Keep in lockstep with eda.alignment.failure_analysis.analyze
TRAJ_OK = 0.5
PLACEMENT_MISS_S = 15.0
DISTINCT_HI = 0.4


def _f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key, default)
    if v is None or v == "":
        return default
    return float(v)


def is_success(row: Mapping[str, Any]) -> bool:
    """True when >= TRAJ_OK of GT seconds decode within 2 s (and identity ok).

    Cross-recording stem-compatible matches (the audible truth is a different
    recording of the same song — e.g. an "acappella" pred over a remix's vocal
    layer) get trajectory=0 by construction: the pred's ref segments are in the
    predicted recording's coordinate space, the truth row's are in a different
    recording's space, so they cannot align. For those, identity + placement
    within PLACEMENT_MISS_S is the win — trajectory is not comparable across
    recordings."""
    if row.get("span_class") in (None, ""):
        return False
    identity = row.get("identity_hit", 1)
    if identity == "" or identity is None:
        identity = 1
    if int(identity) == 0:
        return False
    if _f(row, "traj_strict") >= TRAJ_OK:
        return True
    # cross-recording stem-compatible match: trajectory not comparable; fall back
    # to identity + placement.
    gt_rid = row.get("gt_recording_id", "")
    pred_rid = row.get("recording_id", "")
    if gt_rid and pred_rid and str(gt_rid) != str(pred_rid):
        return _f(row, "set_start_err_s") <= PLACEMENT_MISS_S
    return False


def binding_cause(row: Mapping[str, Any]) -> str:
    """Single binding cause per span (dependency order — same as analyze._cause)."""
    identity = row.get("identity_hit", 1)
    if identity == "" or identity is None:
        identity = 1
    if int(identity) == 0:
        return "identity"
    if _f(row, "set_start_err_s") > PLACEMENT_MISS_S:
        return "placement"
    gt_stem = row.get("gt_stem") or "regular"
    pred_stem = row.get("pred_stem") or "regular"
    if (gt_stem == "acappella") != (pred_stem == "acappella"):
        return "mis-route"
    if row.get("span_class") == "oddratio":
        return "tempo/octave"
    if row.get("span_class") == "loop":
        return "loop-instance"
    frac_distinct = row.get("frac_distinct")
    if frac_distinct not in (None, "") and float(frac_distinct) >= DISTINCT_HI:
        return "instance-ambiguity"
    return "decode-residual"


# placement still "good enough" for a success but visibly off
SUCCESS_PLACE_LOOSE_S = 8.0


def success_type(row: Mapping[str, Any]) -> str:
    """Bucket for organizing successes (stem × structure × placement tightness)."""
    stem = row.get("gt_stem") or "regular"
    cls = row.get("span_class") or "linear"
    place = _f(row, "set_start_err_s")
    tight = "tight" if place <= SUCCESS_PLACE_LOOSE_S else "loose-place"
    return f"{stem} · {cls} · {tight}"


def failure_type(row: Mapping[str, Any]) -> str:
    """Alias of binding cause — failures organize by binding cause."""
    return binding_cause(row)

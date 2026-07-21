"""Confident-Learning verifier + GT-calibration report for the PWS aligner.

Two public APIs:

1. ``prune_confident_errors(soft_labels, predicted_proba)``
   Implements the confident-joint estimate from Northcutt et al. (1911.00068):
   per-class thresholds t_j = mean predicted proba on spans labeled j,
   counts the C_{ỹ,y*} confident-joint matrix, calibrates to marginals,
   and returns off-diagonal span_ids (i.e. spans whose soft label is likely a
   label error) plus a JointEstimate summary.

2. ``calibration_report(set_id, *, votes_path, accuracy_path, bin_s)``
   Validation-only: compare DS-learned probe accuracies against GT-measured
   accuracy for each probe.  NEVER feeds back into fit.

CLI:
    venvs/audio/bin/python -m workspaces.pws_aligner.verifier \\
        --calibrate --sets 2nvzlh2k,1fsnxchk
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.pws_aligner.continuous_model import ProbeNoise
from workspaces.pws_aligner.hypotheses import Hypothesis, vote_to_hypothesis
from workspaces.pws_aligner.votes import AbstainReason, Vote

# Default out/ directory (mirrors run_phase1.py)
_DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[2] / "workspaces" / "alignment_prototype" / "out"
)

# Divergence threshold: flag probe when |learned − gt_measured| > this value
_DIVERGE_THRESH: float = 0.15


# ---------------------------------------------------------------------------
# JointEstimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointEstimate:
    """Confident-joint estimate Q_{ỹ,y*}.

    Attributes
    ----------
    classes:
        Ordered list of class labels (rows/cols of the joint matrix).
    raw_counts:
        C matrix (len(classes) × len(classes)) of confident-joint raw counts.
        C[i, j] = count of spans labeled class i (noisy) that the model is
        confident belong to class j (true).
    calibrated_joint:
        Calibrated joint probability matrix (sums to ~1 within each row to the
        marginal p_j).  Calibrated per Northcutt et al. §3.2.
    """

    classes: tuple[str, ...]
    raw_counts: tuple[tuple[int, ...], ...]
    calibrated_joint: tuple[tuple[float, ...], ...]


# ---------------------------------------------------------------------------
# ProbeCalibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeCalibration:
    """Per-probe continuous calibration snapshot.

    Attributes
    ----------
    probe:
        Probe name.
    learned_sigma_s:
        Learned offset std-dev from continuous label model.
    measured_mad_s:
        Median absolute deviation of offset residuals vs GT on identity-correct votes.
        None if no identity-correct votes were scored.
    learned_accuracy:
        Learned identity accuracy from continuous label model.
    measured_accuracy:
        Identity hit-rate over **all non-abstaining** votes for this probe
        (denominator = every vote that did not abstain, regardless of whether
        a residual could be measured).  Distinct from ``n_scored``, which
        counts only the identity-correct subset.  None if no votes were cast.
    n_scored:
        Number of identity-correct votes with measurable residuals (the
        denominator for ``measured_mad_s``).  Always ≤ the non-abstaining
        vote count used by ``measured_accuracy``.
    """

    probe: str
    learned_sigma_s: float
    measured_mad_s: float | None
    learned_accuracy: float
    measured_accuracy: float | None
    n_scored: int


# ---------------------------------------------------------------------------
# Confident-Learning: prune_confident_errors
# ---------------------------------------------------------------------------


def prune_confident_errors(
    soft_labels: dict[str, str],
    predicted_proba: dict[str, dict[str, float]],
) -> tuple[set[str], JointEstimate]:
    """Return spans with likely label errors + the confident-joint estimate.

    Implements Northcutt et al. 1911.00068 §3:

    1. Per-class threshold t_j = mean P(y=j | x) over spans with soft_label j.
    2. Confident-joint C[ỹ=i, y*=j]: count spans labeled i where P(y=j) ≥ t_j
       (taking the argmax j that crosses its threshold, falling back to the
       argmax if no class crosses).
    3. Calibrate C to the empirical marginals p_j.
    4. Prune off-diagonal entries (C[i,j] where i≠j) by the Probability-
       Based Calibration (PBC) rule: each off-diagonal cell contributes its
       spans as errors.

    Parameters
    ----------
    soft_labels:
        {span_id: class_label} — the (possibly noisy) label for each span.
    predicted_proba:
        {span_id: {class_label: probability}} — per-span soft predictions
        from the label model.

    Returns
    -------
    to_prune:
        Set of span_ids identified as likely label errors.
    estimate:
        JointEstimate with raw counts and calibrated joint matrix.
    """
    if not soft_labels:
        empty: tuple[tuple[float, ...], ...] = ()
        return set(), JointEstimate(classes=(), raw_counts=(), calibrated_joint=empty)

    classes = sorted({lbl for lbl in soft_labels.values()})
    cls_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    n = len(soft_labels)

    # Step 1: per-class thresholds t_j = mean P(y=j) over spans labeled j
    thresholds: dict[str, float] = {}
    for j, cls in enumerate(classes):
        spans_j = [sid for sid, lbl in soft_labels.items() if lbl == cls]
        if not spans_j:
            thresholds[cls] = 0.5
            continue
        proba_j = []
        for sid in spans_j:
            p = predicted_proba.get(sid, {})
            proba_j.append(p.get(cls, 0.0))
        thresholds[cls] = float(np.mean(proba_j)) if proba_j else 0.5

    # Step 2: build confident-joint C[noisy_label_row, true_label_col]
    # We keep track of which span_ids fall into each off-diagonal cell.
    C = np.zeros((K, K), dtype=int)
    cell_spans: dict[tuple[int, int], list[str]] = {
        (i, j): [] for i in range(K) for j in range(K)
    }

    for sid, noisy_lbl in soft_labels.items():
        i = cls_idx.get(noisy_lbl)
        if i is None:
            continue
        p = predicted_proba.get(sid, {})
        if not p:
            # No predictions: assume self (no error evidence)
            C[i, i] += 1
            cell_spans[(i, i)].append(sid)
            continue

        # Find classes crossing their threshold
        above = [
            j for j, cls in enumerate(classes) if p.get(cls, 0.0) >= thresholds[cls]
        ]

        if above:
            # Take argmax among those that cross threshold
            j = max(above, key=lambda j_: p.get(classes[j_], 0.0))
        else:
            # Fall back to global argmax if no class crosses threshold
            j = max(range(K), key=lambda j_: p.get(classes[j_], 0.0))

        C[i, j] += 1
        cell_spans[(i, j)].append(sid)

    # Step 3: calibrate C to empirical marginals
    # Marginal count p_j: total spans (across all noisy labels) where the model
    # is confident of true class j (column sums of C)
    col_sums = C.sum(axis=0).astype(float)  # per-true-class counts
    row_sums = C.sum(axis=1).astype(float)  # per-noisy-label counts

    # Calibrated joint Q[i,j] = C[i,j] / (col_sums[j] * row_sums[i]) * n
    # then normalize so each row sums to the empirical label frequency.
    # Per Northcutt §3.2: Q_{ỹ,y*} = C_{ỹ,y*} / (sum_k C_{ỹ,k} * sum_k C_{k,y*}) * n
    calibrated = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            denom = row_sums[i] * col_sums[j]
            if denom > 0:
                calibrated[i, j] = C[i, j] * n / denom

    # Normalize so the matrix sums to 1
    total = calibrated.sum()
    if total > 0:
        calibrated /= total

    # Step 4: collect off-diagonal span_ids as errors
    to_prune: set[str] = set()
    for i in range(K):
        for j in range(K):
            if i != j:
                to_prune.update(cell_spans[(i, j)])

    # Build immutable JointEstimate
    raw_counts_tuple: tuple[tuple[int, ...], ...] = tuple(
        tuple(int(x) for x in row) for row in C
    )
    calibrated_tuple: tuple[tuple[float, ...], ...] = tuple(
        tuple(float(x) for x in row) for row in calibrated
    )
    estimate = JointEstimate(
        classes=tuple(classes),
        raw_counts=raw_counts_tuple,
        calibrated_joint=calibrated_tuple,
    )
    return to_prune, estimate


# ---------------------------------------------------------------------------
# GT loader (reads labeling/fixtures YAML by set_id)
# ---------------------------------------------------------------------------


def _load_gt_rows(set_id: str, gt_path: Path | None = None) -> list[dict]:
    """Load GT rows for *set_id* from the labeling/fixtures YAML.

    Returns list of track dicts with at least: track_id, set_start_s, ref_start_s.
    Rows with slot_label == "mix" are excluded.
    """
    import yaml

    if gt_path is None:
        fixtures = sorted((_REPO / "labeling" / "fixtures").glob("*_ground_truth.yaml"))
        matches = [
            f for f in fixtures if yaml.safe_load(f.read_text()).get("set_id") == set_id
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 GT fixture for set_id={set_id}, found {len(matches)} "
                f"in labeling/fixtures/"
            )
        gt_path = matches[0]

    doc = yaml.safe_load(gt_path.read_text())
    rows = [r for r in doc.get("tracks", []) if str(r.get("slot_label")) != "mix"]

    # Apply the id_maps bridge exactly as score_timeline_vs_gt does: stale GT
    # track_ids (e.g. BB11's tlp* namespace, 144 remaps) -> current DB ids.
    # Without this, votes (current ids) never match GT rows and gt_measured is
    # silently deflated for every probe.
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    if id_map_path.exists():
        id_map: dict[str, str] = json.loads(id_map_path.read_text())
        for r in rows:
            tid = str(r.get("track_id") or "")
            if tid in id_map and id_map[tid] != tid:
                r["track_id"] = id_map[tid]
    return rows


# ---------------------------------------------------------------------------
# calibration_report
# ---------------------------------------------------------------------------


def calibration_report(
    set_id: str,
    *,
    votes_path: Path | None = None,
    accuracy_path: Path | None = None,
    bin_s: float = 2.0,
    gt_path: Path | None = None,
    _gt_rows_override: list[dict] | None = None,
) -> dict[str, dict]:
    """Compare DS-learned probe accuracies against GT-measured accuracy.

    For each probe, computes:
    - learned: accuracy from ``<set_id>_pws_probe_accuracy.json``
    - gt_measured: fraction of non-abstaining votes whose hypothesis
      (via vote_to_hypothesis) matches the GT hypothesis for that span
      within the same 2s offset bin.

    GT hypothesis for a span = GT recording_id + bin(ref_start_s − set_start_s).

    NOTE — end-to-end semantics, not oracle-identity: a vote matches only if
    BOTH its recording_id and its offset bin agree with GT. An offset-correct
    vote on the wrong recording counts as incorrect, so probes score zero on
    identity-missed spans. This is deliberate (it measures the probe as
    deployed) but understates a probe's isolated offset skill.

    Parameters
    ----------
    set_id:
        DJ-set identifier.
    votes_path:
        Path to ``<set_id>_probe_votes.json``.  Defaults to
        ``out/<set_id>_probe_votes.json``.
    accuracy_path:
        Path to ``<set_id>_pws_probe_accuracy.json``.  Defaults to
        ``out/<set_id>_pws_probe_accuracy.json``.
    bin_s:
        Seconds per offset bin (must match run_phase1).
    gt_path:
        Optional explicit GT YAML path (for testing; bypasses fixture lookup).
    _gt_rows_override:
        Internal: override GT rows directly (for tests).

    Returns
    -------
    dict mapping probe_name -> {
        "learned": float,
        "gt_measured": float,
        "n_votes": int,
        "diverges": bool  (|learned - gt_measured| > 0.15)
    }
    """
    out_dir = _DEFAULT_OUT_DIR

    if votes_path is None:
        votes_path = out_dir / f"{set_id}_probe_votes.json"
    if accuracy_path is None:
        accuracy_path = out_dir / f"{set_id}_pws_probe_accuracy.json"

    if not votes_path.exists():
        sys.exit(f"ERROR: votes file not found: {votes_path}")
    if not accuracy_path.exists():
        sys.exit(f"ERROR: accuracy file not found: {accuracy_path}")

    votes_raw: list[dict] = json.loads(votes_path.read_text())
    learned_acc: dict[str, float] = json.loads(accuracy_path.read_text())

    # Load GT rows
    if _gt_rows_override is not None:
        gt_rows = _gt_rows_override
    else:
        gt_rows = _load_gt_rows(set_id, gt_path=gt_path)

    # Build GT hypothesis lookup: recording_id -> list of (set_start_s, ref_start_s)
    # We match spans by recording_id from the votes file against GT rows by track_id.
    # For each span, we find the closest GT row by set_start_s to get the GT hypothesis.
    gt_by_rid: dict[str, list[dict]] = {}
    for row in gt_rows:
        rid = str(row.get("track_id") or "")
        if rid:
            gt_by_rid.setdefault(rid, []).append(row)

    # Per-probe accumulators: correct votes, total non-abstaining votes
    probe_correct: dict[str, int] = {}
    probe_total: dict[str, int] = {}

    for span_doc in votes_raw:
        # The votes file records recording_id per span (the MERT identity decision).
        # GT match: find GT row for this span's recording_id closest to set_start_s.
        span_rid = str(span_doc.get("recording_id") or "")
        span_set_start = float(span_doc.get("set_start_s", 0.0))

        gt_hyp: Hypothesis | None = None
        if span_rid and span_rid in gt_by_rid:
            gt_candidates = gt_by_rid[span_rid]
            closest = min(
                gt_candidates,
                key=lambda r: abs(float(r["set_start_s"]) - span_set_start),
            )
            gt_offset_s = float(closest["ref_start_s"]) - float(closest["set_start_s"])
            gt_hyp = Hypothesis(span_rid, round(gt_offset_s / bin_s))

        for probe_doc in span_doc.get("probes", []):
            if probe_doc.get("abstain", False):
                continue
            probe_name = str(probe_doc["probe"])
            probe_correct.setdefault(probe_name, 0)
            probe_total.setdefault(probe_name, 0)
            probe_total[probe_name] += 1

            if gt_hyp is None:
                # No GT match for this span's recording_id → can't measure correctness
                continue

            # Build Vote-like object to get hypothesis

            vote = Vote(
                probe=probe_name,
                span_id=str(span_doc.get("span_id", "")),
                recording_id=probe_doc.get("recording_id"),
                offset_s=float(probe_doc.get("offset_s", 0.0)),
                confidence=float(probe_doc.get("confidence", 0.0)),
                abstained=False,
                reason=AbstainReason.NONE,
                features=(),
            )
            vote_hyp = vote_to_hypothesis(vote, bin_s=bin_s)
            if vote_hyp == gt_hyp:
                probe_correct[probe_name] += 1

    # Build report
    all_probes = sorted(set(probe_total.keys()) | set(learned_acc.keys()))
    report: dict[str, dict] = {}
    for probe in all_probes:
        n_votes = probe_total.get(probe, 0)
        n_correct = probe_correct.get(probe, 0)
        gt_measured = n_correct / n_votes if n_votes > 0 else 0.0
        learned = learned_acc.get(probe, 0.0)
        report[probe] = {
            "learned": learned,
            "gt_measured": gt_measured,
            "n_votes": n_votes,
            "diverges": abs(learned - gt_measured) > _DIVERGE_THRESH,
        }
    return report


# ---------------------------------------------------------------------------
# continuous_calibration_report
# ---------------------------------------------------------------------------


def continuous_calibration_report(
    noise: dict[str, ProbeNoise],
    spans: Sequence[Sequence[Vote]],
    gt: dict[str, tuple[str, float]],
) -> list[ProbeCalibration]:
    """Continuous calibration: learned σ vs GT-measured offset residuals.

    For each probe, measures:
    - learned_sigma_s: from the continuous label model
    - measured_mad_s: median absolute deviation of |vote offset - GT offset|
      over identity-correct votes only
    - learned_accuracy: from the continuous label model
    - measured_accuracy: fraction of non-abstaining votes matching GT recording_id
    - n_scored: number of identity-correct votes with residuals

    Parameters
    ----------
    noise:
        {probe_name: ProbeNoise} from the continuous label model.
    spans:
        Sequence of spans, each a sequence of votes per probe.
    gt:
        {span_id: (recording_id, offset_s)} ground truth placements.

    Returns
    -------
    List of ProbeCalibration records, one per probe (in sorted order).
    """
    residuals: dict[str, list[float]] = defaultdict(list)
    id_hits: dict[str, list[bool]] = defaultdict(list)
    for span in spans:
        for v in span:
            if v.abstained or v.recording_id is None or v.span_id not in gt:
                continue
            gt_rec, gt_off = gt[v.span_id]
            hit = v.recording_id == gt_rec
            id_hits[v.probe].append(hit)
            if hit:
                residuals[v.probe].append(abs(v.offset_s - gt_off))
    out = []
    for probe, n in sorted(noise.items()):
        res = sorted(residuals.get(probe, []))
        hits = id_hits.get(probe, [])
        out.append(
            ProbeCalibration(
                probe=probe,
                learned_sigma_s=n.sigma_s,
                measured_mad_s=statistics.median(res) if res else None,
                learned_accuracy=n.accuracy,
                measured_accuracy=(sum(hits) / len(hits)) if hits else None,
                n_scored=len(res),
            )
        )
    return out


def sigma_rank_inversions(
    report: list[ProbeCalibration], min_n: int = 10
) -> list[tuple[str, str]]:
    """Flag rank inversions: learned σ order contradicts measured MAD order.

    When the model trusts the wrong probes (as in the DS gate failure),
    learned σ ordering will invert relative to measured noise. This function
    flags pairs where the ranking contradicts.

    Parameters
    ----------
    report:
        List of ProbeCalibration records from continuous_calibration_report.
    min_n:
        Minimum n_scored required to include a probe in the check.

    Returns
    -------
    List of (probe_a, probe_b) tuples where learned_sigma_s rank contradicts
    measured_mad_s rank. Empty list = model is well-calibrated.
    """
    scored = [r for r in report if r.measured_mad_s is not None and r.n_scored >= min_n]
    bad = []
    for i, a in enumerate(scored):
        for b in scored[i + 1 :]:
            learned = a.learned_sigma_s - b.learned_sigma_s
            measured = a.measured_mad_s - b.measured_mad_s  # type: ignore[operator]
            if learned * measured < 0:
                bad.append((a.probe, b.probe))
    return bad


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_calibration_table(set_id: str, report: dict[str, dict]) -> None:
    print(f"\n=== GT calibration report: {set_id} ===")
    header = f"{'probe':20s}  {'learned':>8}  {'gt_meas':>8}  {'|diff|':>7}  {'n_votes':>8}  {'flag':>6}"
    print(header)
    print("-" * len(header))
    for probe in sorted(report):
        r = report[probe]
        flag = "DIVERGES" if r["diverges"] else "ok"
        diff = abs(r["learned"] - r["gt_measured"])
        print(
            f"{probe:20s}  {r['learned']:8.3f}  {r['gt_measured']:8.3f}  "
            f"{diff:7.3f}  {r['n_votes']:8d}  {flag:>8}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PWS verifier: confident-learning pruner + GT calibration report."
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="run GT calibration report for the given sets",
    )
    p.add_argument(
        "--sets",
        default="",
        help="comma-separated set_ids (e.g. 2nvzlh2k,1fsnxchk)",
    )
    p.add_argument(
        "--votes",
        type=Path,
        default=None,
        help="path to <set_id>_probe_votes.json (single-set override)",
    )
    p.add_argument(
        "--accuracy",
        type=Path,
        default=None,
        help="path to <set_id>_pws_probe_accuracy.json (single-set override)",
    )
    p.add_argument(
        "--bin-s",
        type=float,
        default=2.0,
        help="seconds per offset bin (default 2.0)",
    )
    args = p.parse_args(argv)

    if not args.calibrate:
        p.print_help()
        return 1

    set_ids = [s.strip() for s in args.sets.split(",") if s.strip()]
    if not set_ids:
        sys.exit(
            "ERROR: --sets is required when --calibrate is given (e.g. --sets 2nvzlh2k,1fsnxchk)"
        )

    for set_id in set_ids:
        report = calibration_report(
            set_id,
            votes_path=args.votes,
            accuracy_path=args.accuracy,
            bin_s=args.bin_s,
        )
        _print_calibration_table(set_id, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

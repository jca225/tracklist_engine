from __future__ import annotations

"""Fit-on-unlabeled harness — the PWS mechanism (as opposed to single-set fusion).

This is what turns a pile of labeling functions into Snorkel-style programmatic
weak supervision: fit ONE ``ContinuousLabelModel`` on the POOLED votes of a whole
cohort of UNLABELED sets (the fit uses no GT labels), learning GLOBAL per-LF
accuracy/σ, then grade on the held-out GT sets (BB11/BB12).

Because a fit is driven by a cohort set-list (``cohorts/*.json``), an ablation is
just two fits with different cohorts — e.g. Big-Bootie-only vs Big-Bootie+others
(see cohorts/README.md). ``ablation_compare`` diffs the two learned noise models,
flagging whether the per-LF σ ordering shifts (evidence that pooling across
archetypes needs FABLE instance-conditioning).

Vote-file loading reuses run_phase1's loaders so pooled votes match exactly what
the single-set path feeds the model. Pooling is concatenation: the label-model fit
processes each span independently and learns per-PROBE parameters, so span-id
collisions across sets are harmless.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .continuous_model import ContinuousLabelModel
from .lf_runner import run_all_spans
from .run_phase1 import _load_votes_file, _span_votes
from .votes import Vote


@dataclass(frozen=True)
class Cohort:
    """A named set-list that parameterizes a fit-on-unlabeled run."""

    name: str
    set_ids: tuple[str, ...]
    grade_sets: tuple[str, ...]  # GT sets held out for grading (BB11/BB12)


def load_cohort(path: str | Path) -> Cohort:
    """Load a cohort JSON (see cohorts/big_bootie.json)."""
    doc = json.loads(Path(path).read_text())
    set_ids = tuple(s["set_id"] for s in doc["sets"])
    return Cohort(
        name=str(doc["cohort"]),
        set_ids=set_ids,
        grade_sets=tuple(doc.get("grade_sets", ())),
    )


def pool_set_votes(
    set_ids: Sequence[str], votes_dir: str | Path
) -> tuple[list[tuple[Vote, ...]], list[str]]:
    """Pool per-span votes across every set in the cohort that has a votes file.

    Returns ``(pooled_spans, missing_set_ids)``. Sets with no
    ``<set_id>_probe_votes.json`` are reported in ``missing`` (not captured yet) —
    they are skipped, never silently treated as empty.
    """
    votes_dir = Path(votes_dir)
    pooled: list[tuple[Vote, ...]] = []
    missing: list[str] = []
    for sid in set_ids:
        path = votes_dir / f"{sid}_probe_votes.json"
        if not path.exists():
            missing.append(sid)
            continue
        for span_doc in _load_votes_file(path):
            pooled.append(_span_votes(span_doc))
    return pooled, missing


def pool_set_lf_votes(
    set_ids: Sequence[str], votes_dir: str | Path
) -> tuple[list[tuple[Vote, ...]], list[object], list[str]]:
    """Pool the FULL LF set's votes across the cohort (B1 breadth path).

    Like ``pool_set_votes`` but runs the per-span LF runner over every span,
    so the pooled continuous spans include the cue-time LF alongside the audio
    probes, and the categorical LF votes (claimed-identity, ordering, any
    operation votes captured upstream) are collected for coverage reporting.

    Axis discipline: only the continuous spans feed ``fit_cohort`` — the
    categorical votes are returned separately and must NEVER enter the offset
    EM (the Dawid–Skene granularity lesson).

    Returns ``(pooled_continuous_spans, categorical_votes, missing_set_ids)``.
    """
    votes_dir = Path(votes_dir)
    pooled: list[tuple[Vote, ...]] = []
    categorical: list[object] = []
    missing: list[str] = []
    for sid in set_ids:
        path = votes_dir / f"{sid}_probe_votes.json"
        if not path.exists():
            missing.append(sid)
            continue
        span_docs = _load_votes_file(path)
        for sv in run_all_spans(span_docs, set_id=sid):
            pooled.append(sv.continuous)
            categorical.extend(sv.categorical)
    return pooled, categorical, missing


def fit_cohort(pooled_spans: Sequence[tuple[Vote, ...]]) -> ContinuousLabelModel:
    """Fit one continuous label model on the pooled, unlabeled cohort spans."""
    model = ContinuousLabelModel()
    model.fit(pooled_spans)
    return model


def probe_noise_dict(model: ContinuousLabelModel) -> dict[str, dict[str, float]]:
    """Serialize the learned per-probe noise to a plain dict."""
    return {
        probe: {"accuracy": n.accuracy, "sigma_s": n.sigma_s, "inlier": n.inlier}
        for probe, n in model.probe_noise().items()
    }


def ablation_compare(
    noise_a: dict[str, dict[str, float]],
    noise_b: dict[str, dict[str, float]],
) -> dict:
    """Diff two arms' learned noise models (e.g. BB-only vs BB+others).

    Reports per-probe σ from each arm and whether the σ *ordering* across probes
    changed between arms. A changed ordering means the two cohorts disagree on
    which probes are precise — direct evidence that a flat (non-instance-conditioned)
    label model is being pulled in different directions by different archetypes, i.e.
    the FABLE instance-conditioning lane is warranted before scaling.
    """
    shared = sorted(set(noise_a) & set(noise_b))
    per_probe = {
        p: {"sigma_a": noise_a[p]["sigma_s"], "sigma_b": noise_b[p]["sigma_s"]}
        for p in shared
    }
    order_a = sorted(shared, key=lambda p: noise_a[p]["sigma_s"])
    order_b = sorted(shared, key=lambda p: noise_b[p]["sigma_s"])
    return {
        "shared_probes": shared,
        "per_probe": per_probe,
        "sigma_order_a": order_a,
        "sigma_order_b": order_b,
        "sigma_order_changed": order_a != order_b,
    }

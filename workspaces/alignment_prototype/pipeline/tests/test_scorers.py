"""Scorer stdout parsers — the pure core that turns the frozen referees'
console output into a metrics dict. Integration (real subprocess over pulled
sets) is out of scope for unit tests; the parsing is not."""

from __future__ import annotations

from workspaces.alignment_prototype.pipeline.adapters import scorers

# a representative slice of score_timeline_vs_gt stdout (compose grain referee)
_TIMELINE_STDOUT = """
identity: 124/150 (83%)
set placement |pred-gt|: median=7.9s p90=61.0s
ref offset |pred-gt| (straight): median=8.7s p90=143.6s
HEADLINE multiseg+loop n= 96 traj-acc=16%
  stem acappella n= 40 traj-acc=12%
"""

# trajectory/train.py evaluate() console lines (isolate grain referee)
_TRAJ_STDOUT = """
== no-model control (raw match-sim argmax) ==
  [control eval bb11] traj-acc ALL 0.301 | HEADLINE multiseg+loop 0.284
== trained model ==
  [trm eval bb11] traj-acc ALL 0.552 | HEADLINE multiseg+loop 0.611
"""


def test_parse_timeline_metrics_extracts_headline_and_identity():
    m = scorers.parse_timeline_metrics(_TIMELINE_STDOUT)
    assert m["identity"] == 83.0
    assert m["place_median_s"] == 7.9
    assert m["traj_headline"] == 16.0
    assert m["traj_acappella"] == 12.0


def test_parse_timeline_metrics_missing_field_is_absent_not_zero():
    m = scorers.parse_timeline_metrics("identity: 10/20 (50%)\n")
    assert m["identity"] == 50.0
    assert "traj_headline" not in m  # absent, never silently 0


def test_parse_trajectory_metrics_keys_by_tag():
    m = scorers.parse_trajectory_metrics(_TRAJ_STDOUT)
    assert m["control eval bb11"]["all"] == 0.301
    assert m["control eval bb11"]["headline"] == 0.284
    assert m["trm eval bb11"]["headline"] == 0.611


def test_timeline_scorer_strict_key_is_the_headline_fraction():
    # the runner's learned-nothing check reads "strict"; for the compose grain
    # that is the multiseg+loop headline as a fraction in [0,1].
    m = scorers.timeline_metrics_to_referee(
        scorers.parse_timeline_metrics(_TIMELINE_STDOUT)
    )
    assert m["strict"] == 0.16

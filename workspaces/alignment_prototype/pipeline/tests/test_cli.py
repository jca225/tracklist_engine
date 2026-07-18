"""CLI config → AblationSpec loading (pure). Running a matrix is integration."""

from __future__ import annotations

import pytest

from workspaces.alignment_prototype.pipeline import cli
from workspaces.alignment_prototype.pipeline.stages import Grain


def test_load_spec_from_config_dict():
    spec = cli.load_spec(
        {
            "grain": "compose",
            "sets": ["1fsnxchk", "2nvzlh2k"],
            "actors": ["viterbi", "agentic"],
            "base": "classical",
            "seed": 3,
        }
    )
    assert spec.grain is Grain.COMPOSE
    assert spec.sets == ("1fsnxchk", "2nvzlh2k")
    assert spec.actors == ("viterbi", "agentic")
    assert spec.seed == 3


def test_load_spec_rejects_unknown_grain():
    with pytest.raises(ValueError):
        cli.load_spec({"grain": "wobble", "sets": ["x"], "actors": ["viterbi"]})


def test_load_spec_defaults_seed_and_base():
    spec = cli.load_spec({"grain": "isolate", "sets": ["2nvzlh2k"], "actors": ["trm"]})
    assert spec.seed == 0
    assert spec.base == "classical"

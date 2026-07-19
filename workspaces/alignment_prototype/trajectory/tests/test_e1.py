"""E1 orchestration command construction and failure/resume behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from workspaces.alignment_prototype.trajectory.e1 import (
    E1Config,
    build_train_commands,
    run_e1,
)


def test_e1_commands_keep_pool_and_eval_disjoint(tmp_path):
    cfg = E1Config(
        pool_set="w1mgcjt",
        eval_set="2nvzlh2k",
        base_timeline=tmp_path / "base.json",
        synthetic_root=tmp_path / "synthetic",
        out_dir=tmp_path / "out",
    )
    commands = build_train_commands(cfg, tmp_path / "pseudo.yaml")
    assert all(
        cmd[cmd.index("--eval-set") : cmd.index("--eval-set") + 2]
        == ["--eval-set", "2nvzlh2k"]
        for cmd in commands
    )
    assert sum("--train-yaml" in cmd for cmd in commands) == 2
    assert any("--synthetic-only" in cmd for cmd in commands)
    # order: conv, synthetic-only TRM, pseudo TRM
    assert "--model" in commands[0] and commands[0][commands[0].index("--model") + 1] == "conv"
    assert "--synthetic-only" in commands[1]
    assert commands[2][commands[2].index("--model") + 1] == "trm"
    assert "--train-yaml" in commands[2]


def test_e1_config_rejects_same_pool_and_eval(tmp_path):
    with pytest.raises(ValueError, match="same set"):
        E1Config(
            pool_set="2nvzlh2k",
            eval_set="2nvzlh2k",
            base_timeline=tmp_path / "base.json",
            synthetic_root=tmp_path / "synthetic",
            out_dir=tmp_path / "out",
        )


def test_e1_starvation_skips_training(tmp_path, monkeypatch):
    from workspaces.alignment_prototype.trajectory import e1 as e1_mod
    from workspaces.alignment_prototype.trajectory.pseudo_materialize import (
        MaterializeReport,
    )

    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "set_id": "w1mgcjt",
                "spans": [
                    {
                        "slot_label": "1",
                        "recording_id": "r1",
                        "set_start_s": 0.0,
                        "set_end_s": 10.0,
                        "claimed_stem": "regular",
                    }
                ],
            }
        )
    )
    aligning = tmp_path / "aligning"
    aligning.mkdir()
    (aligning / "manifest.json").write_text(
        json.dumps({"set_id": "w1mgcjt", "tracks": []})
    )
    (aligning / "mix.m4a").write_bytes(b"x")
    (aligning / "tracks").mkdir()

    cfg = E1Config(
        pool_set="w1mgcjt",
        eval_set="2nvzlh2k",
        base_timeline=base,
        synthetic_root=tmp_path / "synthetic",
        out_dir=tmp_path / "out",
        reuse_agentic=True,
    )
    agentic = cfg.out_dir / f"{cfg.pool_set}_agentic_pseudo_timeline.json"
    agentic.parent.mkdir(parents=True, exist_ok=True)
    agentic.write_text(
        json.dumps(
            {
                "set_id": "w1mgcjt",
                "spans": [
                    {
                        "slot_label": "1",
                        "recording_id": "r1",
                        "set_start_s": 0.0,
                        "set_end_s": 10.0,
                        "claimed_stem": "regular",
                        "driver_mode": "review",
                    }
                ],
                "pseudo_safety": {"combine": True, "fiber_gate": True, "auto": 0.75},
            }
        )
    )

    monkeypatch.setattr(e1_mod, "preflight_set", lambda sid: aligning)
    calls: list[list[str]] = []

    def _no_sub(*a, **k):
        calls.append(list(a[0]) if a else [])
        raise AssertionError("subprocess must not run when starved")

    monkeypatch.setattr(e1_mod.subprocess, "run", _no_sub)

    def _mat(tl, mf, out):
        out.write_text(yaml.safe_dump({"set_id": "w1mgcjt", "tracks": [], "provenance": {}}))
        return MaterializeReport(total=1, accepted=0, dropped={"g0_mode": 1}, output=out)

    monkeypatch.setattr(e1_mod, "materialize_pseudo_gt", _mat)

    code = run_e1(cfg)
    assert code == 2
    assert calls == []
    result = json.loads((cfg.out_dir / "e1_result.json").read_text())
    assert result["status"] == "starved"
    assert "metrics" not in result

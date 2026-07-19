"""Unit tests for daw_env (ALS-first Ableton ReAct) — no Live required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")

from workspaces.alignment_prototype.daw_env.actions import (  # noqa: E402
    EscalateHuman,
    NudgeSetStart,
    PlaceSpan,
    RenderListen,
    SoloLayer,
)
from workspaces.alignment_prototype.daw_env.als_mutate import (  # noqa: E402
    make_minimal_als,
    sync_span_to_als,
)
from workspaces.alignment_prototype.daw_env.loop import resolve_daw  # noqa: E402
from workspaces.alignment_prototype.daw_env.session import (  # noqa: E402
    DAW_REACT_SUFFIX,
    DawSession,
    SpanGeom,
)
from labeling.als import load_als_xml  # noqa: E402


def test_place_and_nudge_update_geometry(tmp_path: Path) -> None:
    session = DawSession(
        set_id="test",
        spans={
            "001": SpanGeom("001", "rid", "regular", 10.0, 18.0, 0.0, cue_anchor_s=12.0)
        },
        out_dir=tmp_path,
    )
    session.apply(PlaceSpan("001", 20.0, 28.0, 5.0))
    assert session.spans["001"].set_start_s == 20.0
    assert session.spans["001"].ref_start_s == 5.0
    session.apply(NudgeSetStart("001", -2.0))
    assert session.spans["001"].set_start_s == 18.0
    assert session.spans["001"].set_end_s == 26.0
    assert any(h["kind"] == "PlaceSpan" for h in session.history)


def test_render_listen_writes_wav(tmp_path: Path) -> None:
    # Synthetic mix in set_dir
    import soundfile as sf

    from workspaces.alignment_prototype.refine_ref_offsets import SR

    set_dir = tmp_path / "set"
    set_dir.mkdir()
    y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(int(2 * SR)) / SR)).astype(np.float32)
    sf.write(str(set_dir / "mix.wav"), y, SR)

    session = DawSession(
        set_id="test",
        spans={"001": SpanGeom("001", "r", "regular", 0.2, 1.0, 0.0)},
        out_dir=tmp_path / "out",
        set_dir=set_dir,
    )
    session.apply(SoloLayer("mix"))
    session.apply(RenderListen(0.0, 1.0))
    assert session.last_listen_wav is not None
    assert session.last_listen_wav.is_file()
    assert session.last_listen_wav.stat().st_size > 100


def test_als_sync_moves_clip(tmp_path: Path) -> None:
    als = make_minimal_als(tmp_path / "t.als", slot="001")
    geom = SpanGeom("001", "r", "regular", 30.0, 38.0, 2.0)
    sync_span_to_als(als, geom)
    root = load_als_xml(als)
    clip = root.find(".//AudioClip")
    assert clip is not None
    assert abs(float(clip.get("Time")) - 30.0) < 1e-3
    assert abs(float(clip.find("CurrentStart").get("Value")) - 30.0) < 1e-3


def test_mode_b_escalate_writes_locator(tmp_path: Path) -> None:
    als = make_minimal_als(tmp_path / "t.als", slot="001")
    session = DawSession(
        set_id="test",
        spans={"001": SpanGeom("001", "r", "regular", 10.0, 18.0, 0.0)},
        out_dir=tmp_path,
        als_path=als,
    )
    session.apply(EscalateHuman("001"))
    assert session.spans["001"].escalated
    root = load_als_xml(als)
    names = [n.get("Value") for n in root.findall(".//Locator/Name")]
    assert any(n and n.startswith("AGENT PROPOSE 001") for n in names)


def test_resolve_daw_mode_a_commits(tmp_path: Path) -> None:
    session = DawSession(
        set_id="test",
        spans={
            "001": SpanGeom(
                "001", "r", "instrumental", 10.0, 18.0, 0.0, cue_anchor_s=14.0
            )
        },
        out_dir=tmp_path,
    )
    result = resolve_daw(session, mode="a", max_steps_per_span=2, budget_spans=1)
    assert "001" in result.committed or session.spans["001"].committed
    assert result.steps >= 1
    tl = session.to_timeline()
    assert tl["spans"][0]["start_source"] == "daw_react"


def test_daw_react_suffix_never_align() -> None:
    assert "align.als" not in DAW_REACT_SUFFIX
    assert "DAW REACT" in DAW_REACT_SUFFIX


def test_content_target_prefers_fp_over_cue() -> None:
    from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
    from workspaces.alignment_prototype.daw_env.sense import content_target_set_start

    b = SpanBelief("1", "r", "instrumental")
    b = b.observe(Observation("cue_prior", 10.0, 0.7, 0.50, detail="cue"))
    b = b.observe(Observation("fp", 40.0, 0.8, 0.53, detail="fp"))
    assert content_target_set_start(b) == pytest.approx(40.0)


def test_propose_nudge_toward_content() -> None:
    from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
    from workspaces.alignment_prototype.daw_env.loop import _propose_nudge

    b = SpanBelief("1", "r", "instrumental")
    b = b.observe(Observation("fp", 28.0, 0.8, 0.53))
    nudge = _propose_nudge(b, geom_start=10.0)
    assert nudge is not None
    assert nudge == pytest.approx(12.0)  # capped at +12


def test_propose_nudge_ignores_onset_only() -> None:
    from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
    from workspaces.alignment_prototype.daw_env.loop import _propose_nudge

    b = SpanBelief("1", "r", "regular")
    b = b.observe(Observation("daw_onset", 40.0, 0.85, 0.55))
    assert _propose_nudge(b, geom_start=10.0) is None


def test_propose_nudge_rejects_wild_content() -> None:
    from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
    from workspaces.alignment_prototype.daw_env.loop import _propose_nudge

    b = SpanBelief("1", "r", "acappella")
    b = b.observe(Observation("stem_hubert", 91.0, 0.3, 0.75))
    assert _propose_nudge(b, geom_start=1316.0) is None
    # within old 90s band but beyond 24s refine window — still reject
    b2 = SpanBelief("2", "r", "acappella")
    b2 = b2.observe(Observation("stem_hubert", 1510.0 + 60.0, 0.8, 0.75))
    assert _propose_nudge(b2, geom_start=1510.0) is None

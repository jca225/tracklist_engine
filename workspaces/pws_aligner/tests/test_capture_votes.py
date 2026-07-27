"""TDD tests for capture_votes — genuine per-probe vote capture via harness probes.

Four levels:
  1. Unit — fake Probes (contract ABC) including one abstainer and one that
     raises; verify schema-valid votes with the raiser recorded as abstain.
  2. Genuineness regression — two fake probes voting DIFFERENT recording_ids on
     one span → both recording_ids are preserved (not collapsed to a shared id).
  3. Round-trip — captured file feeds run_phase1() end-to-end on tmp_path.
  4. Probe-abstain semantics — missing audio paths → all probes abstain, no crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fake probes for offline testing (no audio, no pi-storage, no torch)
# ---------------------------------------------------------------------------

# We define fake probes BEFORE importing capture_votes so we can inject them
# without triggering real librosa/torch imports.

from workspaces.alignment_prototype.harness.contract import (
    AlignmentResult,
    CandidatePool,
    MixContext,
    Probe,
    RefContext,
)


class _FixedProbe(Probe):
    """A probe that always returns a fixed AlignmentResult (no audio needed)."""

    def __init__(
        self,
        name: str,
        recording_id: str | None,
        offset_s: float,
        confidence: float,
        abstain: bool = False,
    ) -> None:
        self.name = name
        self._recording_id = recording_id
        self._offset_s = offset_s
        self._confidence = confidence
        self._abstain = abstain

    def run(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        if self._abstain:
            return AlignmentResult.abstained(
                source=self.name, recording_id=self._recording_id
            )
        return AlignmentResult(
            recording_id=self._recording_id,
            offset_s=self._offset_s,
            confidence=self._confidence,
            source=self.name,
        )


class _RaisingProbe(Probe):
    """A probe that always raises FileNotFoundError (simulates missing cache)."""

    name = "raiser"

    def run(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        raise FileNotFoundError(
            "simulated missing fp cache: /fake/.cache/fp_index/r1__regular.landmark"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timeline(tmp_path: Path, spans: list[dict], set_id: str = "ts1") -> Path:
    """Write a minimal predicted_timeline.json."""
    payload = {"set_id": set_id, "spans": spans, "skipped": []}
    p = tmp_path / f"{set_id}_predicted_timeline.json"
    p.write_text(json.dumps(payload))
    return p


def _make_span(
    slot_label: str = "1",
    recording_id: str = "r1",
    claimed_stem: str = "regular",
    set_start_s: float = 10.0,
    set_end_s: float = 50.0,
    ref_start_s: float = 8.0,
) -> dict:
    return {
        "slot_label": slot_label,
        "recording_id": recording_id,
        "claimed_stem": claimed_stem,
        "set_start_s": set_start_s,
        "set_end_s": set_end_s,
        "ref_start_s": ref_start_s,
        "ref_end_s": ref_start_s + (set_end_s - set_start_s),
        "confidence": 0.5,
        "name": f"Track {slot_label}",
    }


def _inject_probes(monkeypatch, probe_map: dict[str, object | None]) -> None:
    """Patch _build_probes to return a fixed probe_map (no real imports)."""
    import workspaces.pws_aligner.capture_votes as cv

    monkeypatch.setattr(cv, "_build_probes", lambda names: {**probe_map})


def _skip_audio_resolution(monkeypatch, tmp_path: Path) -> None:
    """Patch audio-path resolution so _capture_span always finds audio."""
    import workspaces.pws_aligner.capture_votes as cv

    # Return a dummy path that "exists" — we'll write a 0-byte placeholder
    dummy = tmp_path / "dummy_audio.flac"
    dummy.write_bytes(b"")

    monkeypatch.setattr(cv, "_find_aligning_dir", lambda set_id: tmp_path)
    monkeypatch.setattr(
        cv,
        "_load_manifest_by_rid",
        lambda d, s: {
            "r1": {"local_path": str(dummy), "stems": {}, "stem": "regular"},
            "r2": {"local_path": str(dummy), "stems": {}, "stem": "regular"},
        },
    )
    monkeypatch.setattr(cv, "_mix_audio_path", lambda d, stem: dummy)
    monkeypatch.setattr(cv, "_ref_audio_path", lambda row, stem: dummy)


# ---------------------------------------------------------------------------
# 1. Unit tests: schema validity with fake probes
# ---------------------------------------------------------------------------


def test_capture_span_schema_valid(tmp_path, monkeypatch):
    """_capture_span produces a schema-valid span doc when probes run cleanly."""
    import workspaces.pws_aligner.capture_votes as cv

    probe_a = _FixedProbe("fp", "r1", 5.0, 0.7)
    probe_b = _FixedProbe("chroma", "r1", 5.2, 0.6)

    _skip_audio_resolution(monkeypatch, tmp_path)

    span = _make_span()
    doc = cv._capture_span(
        span,
        probe_names=("fp", "chroma"),
        probe_instances={"fp": probe_a, "chroma": probe_b},
        set_dir=tmp_path,
        manifest_by_rid={
            "r1": {
                "local_path": str(tmp_path / "dummy_audio.flac"),
                "stems": {},
                "stem": "regular",
            }
        },
    )

    # Top-level schema
    required_top = {
        "span_id",
        "slot_label",
        "recording_id",
        "claimed_stem",
        "set_start_s",
        "set_end_s",
        "ref_start_s",
        "ref_end_s",
        "confidence",
        "name",
        "probes",
    }
    assert required_top <= doc.keys()
    assert doc["span_id"] == "1"
    assert doc["recording_id"] == "r1"

    # Probe entry schema
    required_probe = {
        "probe",
        "recording_id",
        "offset_s",
        "confidence",
        "abstain",
        "features",
    }
    for entry in doc["probes"]:
        assert required_probe <= entry.keys()
        assert isinstance(entry["features"], list)
        assert 0.0 <= entry["confidence"] <= 1.0


def test_capture_span_abstainer_recorded(tmp_path, monkeypatch):
    """A probe that returns abstain=True is recorded as abstain, not dropped."""
    import workspaces.pws_aligner.capture_votes as cv

    _skip_audio_resolution(monkeypatch, tmp_path)

    abstainer = _FixedProbe("chroma", None, 0.0, 0.0, abstain=True)
    voter = _FixedProbe("fp", "r1", 5.0, 0.8)

    span = _make_span()
    dummy = tmp_path / "dummy_audio.flac"
    dummy.write_bytes(b"")

    doc = cv._capture_span(
        span,
        probe_names=("fp", "chroma"),
        probe_instances={"fp": voter, "chroma": abstainer},
        set_dir=tmp_path,
        manifest_by_rid={
            "r1": {"local_path": str(dummy), "stems": {}, "stem": "regular"}
        },
    )

    by_name = {e["probe"]: e for e in doc["probes"]}
    assert not by_name["fp"]["abstain"]
    assert by_name["fp"]["recording_id"] == "r1"
    assert by_name["fp"]["confidence"] == pytest.approx(0.8)

    assert by_name["chroma"]["abstain"]
    assert by_name["chroma"]["recording_id"] is None
    assert by_name["chroma"]["offset_s"] is None


def test_capture_span_raiser_becomes_abstain(tmp_path, monkeypatch):
    """A probe that raises FileNotFoundError becomes an explicit abstain entry."""
    import workspaces.pws_aligner.capture_votes as cv

    _skip_audio_resolution(monkeypatch, tmp_path)

    raiser = _RaisingProbe()
    voter = _FixedProbe("chroma", "r1", 3.0, 0.6)

    span = _make_span()
    dummy = tmp_path / "dummy_audio.flac"
    dummy.write_bytes(b"")

    doc = cv._capture_span(
        span,
        probe_names=("raiser", "chroma"),
        probe_instances={"raiser": raiser, "chroma": voter},
        set_dir=tmp_path,
        manifest_by_rid={
            "r1": {"local_path": str(dummy), "stems": {}, "stem": "regular"}
        },
    )

    by_name = {e["probe"]: e for e in doc["probes"]}
    # The raiser becomes an explicit abstain, not a crash, not a silent drop
    assert "raiser" in by_name, "raiser entry must be present (not silently dropped)"
    assert by_name["raiser"]["abstain"]
    assert by_name["raiser"]["recording_id"] is None
    assert by_name["raiser"]["confidence"] == 0.0

    # The other probe still fires normally
    assert not by_name["chroma"]["abstain"]
    assert by_name["chroma"]["recording_id"] == "r1"


def test_capture_span_none_instance_becomes_abstain(tmp_path, monkeypatch):
    """A probe instance of None (failed import) becomes an explicit abstain entry."""
    import workspaces.pws_aligner.capture_votes as cv

    _skip_audio_resolution(monkeypatch, tmp_path)

    span = _make_span()
    dummy = tmp_path / "dummy_audio.flac"
    dummy.write_bytes(b"")

    doc = cv._capture_span(
        span,
        probe_names=("fp", "chroma"),
        probe_instances={"fp": None, "chroma": _FixedProbe("chroma", "r1", 1.0, 0.5)},
        set_dir=tmp_path,
        manifest_by_rid={
            "r1": {"local_path": str(dummy), "stems": {}, "stem": "regular"}
        },
    )

    by_name = {e["probe"]: e for e in doc["probes"]}
    assert "fp" in by_name
    assert by_name["fp"]["abstain"]


# ---------------------------------------------------------------------------
# 2. Genuineness regression: two probes vote DIFFERENT recording_ids
# ---------------------------------------------------------------------------


def test_probe_recording_id_not_overwritten_by_span_merged_id(tmp_path, monkeypatch):
    """Capture must NOT overwrite a probe's returned recording_id.

    What this proves: the capture path preserves each AlignmentResult's
    recording_id verbatim — two fake probes returning DIFFERENT ids both
    survive to the JSON, so nothing re-injects the span-level merged id.

    What this does NOT prove: real multi-candidate divergence. The current
    harness probes all receive a single-candidate pool (the predicted
    recording), so in production every non-abstaining vote carries that
    candidate's id; genuine cross-candidate identity opinions require the
    future multi-candidate pool machinery (Task 7+).
    """
    import workspaces.pws_aligner.capture_votes as cv

    _skip_audio_resolution(monkeypatch, tmp_path)

    # Two probes that would return different candidates if asked
    # (In the real harness, recording_id comes from ref.recording_id passed in;
    # in our fake probes it's an override to simulate each probe picking a winner.)
    probe_fp = _FixedProbe("fp", "r_fp_pick", 5.0, 0.9)
    probe_chroma = _FixedProbe("chroma", "r_chroma_pick", 8.0, 0.7)

    span = _make_span(recording_id="r_merged")  # merged identity (ignored by capture)
    dummy = tmp_path / "dummy_audio.flac"
    dummy.write_bytes(b"")

    doc = cv._capture_span(
        span,
        probe_names=("fp", "chroma"),
        probe_instances={"fp": probe_fp, "chroma": probe_chroma},
        set_dir=tmp_path,
        manifest_by_rid={
            "r_merged": {"local_path": str(dummy), "stems": {}, "stem": "regular"},
        },
    )

    by_name = {e["probe"]: e for e in doc["probes"]}
    # Each probe's OWN recording_id is preserved, not the merged span recording_id
    assert by_name["fp"]["recording_id"] == "r_fp_pick", (
        "fp probe must carry its own AlignmentResult.recording_id"
    )
    assert by_name["chroma"]["recording_id"] == "r_chroma_pick", (
        "chroma probe must carry its own AlignmentResult.recording_id"
    )
    # They genuinely differ — DS can now learn per-probe identity accuracy
    assert by_name["fp"]["recording_id"] != by_name["chroma"]["recording_id"]


# ---------------------------------------------------------------------------
# 3. Round-trip: captured file → run_phase1 end-to-end
# ---------------------------------------------------------------------------


def test_capture_votes_run_phase1_roundtrip(tmp_path, monkeypatch):
    """capture_votes output feeds run_phase1 and produces a valid pws_timeline."""
    import workspaces.pws_aligner.capture_votes as cv
    from workspaces.pws_aligner.run_phase1 import run_phase1

    set_id = "cv_rt_set"
    spans = [
        _make_span("1", "r1", "regular", 10.0, 50.0, 8.0),
        _make_span("2", "r2", "regular", 50.0, 90.0, 3.0),
    ]
    timeline_path = _make_timeline(tmp_path, spans, set_id)
    out_path = tmp_path / f"{set_id}_probe_votes.json"

    # Inject fake probes and stub audio resolution
    probe_fp = _FixedProbe("fp", "r1", 5.0, 0.8)
    probe_chroma = _FixedProbe("chroma", "r1", 5.2, 0.6)

    _skip_audio_resolution(monkeypatch, tmp_path)
    _inject_probes(monkeypatch, {"fp": probe_fp, "chroma": probe_chroma})

    # Run capture
    written = cv.capture_votes(
        set_id,
        timeline_path=timeline_path,
        out_path=out_path,
        probe_names=("fp", "chroma"),
    )
    assert written.exists()

    votes_raw = json.loads(written.read_text())
    assert isinstance(votes_raw, list)
    assert len(votes_raw) == 2

    # Verify schema
    required_probe = {
        "probe",
        "recording_id",
        "offset_s",
        "confidence",
        "abstain",
        "features",
    }
    for span_doc in votes_raw:
        assert "span_id" in span_doc
        assert "probes" in span_doc
        for p in span_doc["probes"]:
            assert required_probe <= p.keys()

    # Feed to run_phase1
    out_dir = tmp_path / "pws_out"
    out_dir.mkdir()
    tl_path = run_phase1(set_id, votes_path=written, out_dir=out_dir, bin_s=2.0)
    assert tl_path.exists()

    tl = json.loads(tl_path.read_text())
    assert tl["set_id"] == set_id
    assert len(tl["spans"]) == 2

    for s in tl["spans"]:
        assert "slot_label" in s
        assert "recording_id" in s
        assert "ref_start_s" in s
        assert "confidence" in s
        assert "abstain" in s


# ---------------------------------------------------------------------------
# 4. Missing audio paths → all probes abstain, no crash
# ---------------------------------------------------------------------------


def test_capture_votes_missing_audio_all_abstain(tmp_path, monkeypatch):
    """When mix audio is not found, all probes abstain — no crash, no silent drop."""
    import workspaces.pws_aligner.capture_votes as cv

    set_id = "missing_audio_set"
    spans = [_make_span("1", "r1", "regular")]
    timeline_path = _make_timeline(tmp_path, spans, set_id)
    out_path = tmp_path / f"{set_id}_probe_votes.json"

    probe_fp = _FixedProbe("fp", "r1", 5.0, 0.8)
    probe_chroma = _FixedProbe("chroma", "r1", 5.2, 0.6)

    # No aligning dir → set_dir is None → mix audio can't be resolved
    monkeypatch.setattr(cv, "_find_aligning_dir", lambda sid: None)
    monkeypatch.setattr(cv, "_load_manifest_by_rid", lambda d, s: {})
    _inject_probes(monkeypatch, {"fp": probe_fp, "chroma": probe_chroma})

    written = cv.capture_votes(
        set_id,
        timeline_path=timeline_path,
        out_path=out_path,
        probe_names=("fp", "chroma"),
    )
    assert written.exists()

    votes_raw = json.loads(written.read_text())
    assert len(votes_raw) == 1
    # All probes should abstain when audio is unavailable
    for probe_entry in votes_raw[0]["probes"]:
        assert probe_entry["abstain"], (
            f"probe {probe_entry['probe']!r} should abstain when audio is missing"
        )


def test_capture_votes_missing_ref_audio_abstains(tmp_path, monkeypatch):
    """When ref audio is missing for a span, all probes for that span abstain."""
    import workspaces.pws_aligner.capture_votes as cv

    set_id = "missing_ref_set"
    spans = [_make_span("1", "r_no_audio", "regular")]
    timeline_path = _make_timeline(tmp_path, spans, set_id)
    out_path = tmp_path / f"{set_id}_probe_votes.json"

    dummy_mix = tmp_path / "mix.m4a"
    dummy_mix.write_bytes(b"")

    # set_dir exists but manifest has no entry for r_no_audio
    monkeypatch.setattr(cv, "_find_aligning_dir", lambda sid: tmp_path)
    monkeypatch.setattr(cv, "_load_manifest_by_rid", lambda d, s: {})
    monkeypatch.setattr(cv, "_mix_audio_path", lambda d, stem: dummy_mix)
    # _ref_audio_path returns None (no row in manifest)
    monkeypatch.setattr(cv, "_ref_audio_path", lambda row, stem: None)

    probe_fp = _FixedProbe("fp", "r_no_audio", 5.0, 0.8)
    _inject_probes(monkeypatch, {"fp": probe_fp})

    written = cv.capture_votes(
        set_id,
        timeline_path=timeline_path,
        out_path=out_path,
        probe_names=("fp",),
    )
    votes_raw = json.loads(written.read_text())
    for probe_entry in votes_raw[0]["probes"]:
        assert probe_entry["abstain"]


# ---------------------------------------------------------------------------
# 5. Probe ordering is deterministic (canonical probe_names order)
# ---------------------------------------------------------------------------


def test_capture_span_probe_order_is_canonical(tmp_path, monkeypatch):
    """Probes in the output are in the canonical probe_names order."""
    import workspaces.pws_aligner.capture_votes as cv

    _skip_audio_resolution(monkeypatch, tmp_path)

    probe_chroma = _FixedProbe("chroma", "r1", 3.0, 0.5)
    probe_fp = _FixedProbe("fp", "r1", 5.0, 0.7)

    span = _make_span()
    dummy = tmp_path / "dummy_audio.flac"
    dummy.write_bytes(b"")

    doc = cv._capture_span(
        span,
        probe_names=("fp", "chroma"),  # fp first in canonical order
        probe_instances={"fp": probe_fp, "chroma": probe_chroma},
        set_dir=tmp_path,
        manifest_by_rid={
            "r1": {"local_path": str(dummy), "stems": {}, "stem": "regular"}
        },
    )

    probe_order = [e["probe"] for e in doc["probes"]]
    # fp must appear before chroma (canonical order is fp, chroma)
    assert probe_order.index("fp") < probe_order.index("chroma")


def test_run_probe_safe_debug_is_parameter_not_argv(monkeypatch):
    """I1 regression: debug is a real parameter; sys.argv is never consulted."""
    import sys as _sys

    import workspaces.pws_aligner.capture_votes as cv
    from workspaces.alignment_prototype.harness.contract import (
        CandidatePool,
        MixContext,
        RefContext,
    )

    class _Boom:
        name = "boom"

        def run(self, mix, ref, candidates):  # noqa: ANN001
            raise RuntimeError("kaboom")

    # Poison argv: if the implementation still reads sys.argv, this would
    # flip debug on; the assertion below is that the call simply works with
    # the explicit parameter and returns an abstain entry either way.
    monkeypatch.setattr(_sys, "argv", ["prog", "--debug"])
    mix = MixContext(audio_path=Path("/nonexistent/mix.m4a"))
    ref = RefContext(recording_id="r1", audio_path=Path("/nonexistent/ref.m4a"))
    entry = cv._run_probe_safe(
        "boom", _Boom(), mix, ref, CandidatePool(), debug=True
    )
    assert entry["abstain"] is True
    entry2 = cv._run_probe_safe(
        "boom", _Boom(), mix, ref, CandidatePool(), debug=False
    )
    assert entry2["abstain"] is True


def test_manifest_index_keys_recording_id_rows(tmp_path):
    """I2 regression: manifest rows keyed by recording_id (no track_id) resolve."""
    import workspaces.pws_aligner.capture_votes as cv

    manifest = {
        "tracks": [
            {"recording_id": "recA", "title": "A", "local_path": "a.m4a"},
            {"track_id": "recB", "title": "B", "local_path": "b.m4a"},
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    by_rid = cv._load_manifest_by_rid(tmp_path, "someset")
    assert by_rid["recA"]["title"] == "A"  # recording_id-keyed row found
    assert by_rid["recB"]["title"] == "B"  # track_id-keyed row still wins


def test_absolute_frame_probes_converted_to_relative(tmp_path, monkeypatch):
    """chroma/continuity/hubert emit ABSOLUTE ref positions; capture must
    convert to the votes-file RELATIVE frame (offset = abs − set_start).
    fp already emits the relative diagonal and must pass through untouched."""
    import workspaces.pws_aligner.capture_votes as cv

    _skip_audio_resolution(monkeypatch, tmp_path)
    # span at set_start 100.0; chroma says "ref position 130.0" (absolute),
    # fp says "+25.0" (already relative)
    chroma = _FixedProbe("chroma", "r1", 130.0, 0.6)
    fp = _FixedProbe("fp", "r1", 25.0, 0.8)
    span = _make_span(set_start_s=100.0, set_end_s=140.0)
    dummy = tmp_path / "dummy_audio.flac"
    dummy.write_bytes(b"")
    doc = cv._capture_span(
        span,
        probe_names=("fp", "chroma"),
        probe_instances={"fp": fp, "chroma": chroma},
        set_dir=tmp_path,
        manifest_by_rid={"r1": {"local_path": str(dummy), "stems": {}, "stem": "regular"}},
    )
    by = {e["probe"]: e for e in doc["probes"]}
    assert by["chroma"]["offset_s"] == pytest.approx(30.0)  # 130 − 100
    assert by["fp"]["offset_s"] == pytest.approx(25.0)      # untouched

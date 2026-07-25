"""The analysis forward-routing dual-write — additive, flagged, defensive.

Tests the three load-bearing properties directly on the helper (constructing a
full TrackAnalysisResult + DB is unnecessary — the helper is where the flag /
defensive logic lives):
  * FLAGGED  : off by default ⇒ no store touched.
  * ADDITIVE : on ⇒ a canonical mert_features artifact lands, provenanced.
  * DEFENSIVE: a provenance failure is swallowed, never raised.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from analysis.models import MeasureEmbedding
from analysis.persistence import (
    _dual_write_enabled,
    _maybe_dual_write_provenance,
    _serialize_measures,
)
from core.provenance import ArtifactStore, connect
from core.provenance.laws import LawVerdict, check_laws


def _measures(n=3, dim=4):
    return [
        MeasureEmbedding(
            track_audio_id=42,
            measure_idx=i,
            start_s=float(i),
            end_s=float(i + 1),
            dim=dim,
            dtype="float16",
            embedding_bytes=np.full(dim, i, dtype=np.float16).tobytes(),
        )
        for i in range(n)
    ]


def _result(measures):
    return SimpleNamespace(
        track_audio_id=42,
        measures=measures,
        analyzer_versions={"mert": "v1"},
    )


def test_serialize_is_deterministic_and_order_independent():
    ms = _measures()
    a = _serialize_measures(ms)
    b = _serialize_measures(list(reversed(ms)))  # sorted by idx inside
    assert a == b and a.startswith(b'{"')


def test_flag_off_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("PROVENANCE_DUAL_WRITE", raising=False)
    monkeypatch.setenv("PROVENANCE_STORE_ROOT", str(tmp_path / "store"))
    assert _dual_write_enabled() is False
    _maybe_dual_write_provenance(_result(_measures()))
    # nothing created — the guard returns before importing / opening a store
    assert not (tmp_path / "store").exists()


def test_flag_on_dual_writes_a_canonical_feature(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("PROVENANCE_DUAL_WRITE", "1")
    monkeypatch.setenv("PROVENANCE_STORE_ROOT", str(root))
    _maybe_dual_write_provenance(_result(_measures()))

    conn = connect(root)
    store = ArtifactStore(root, conn)
    n = conn.execute(
        "SELECT COUNT(*) n FROM artifact WHERE kind='mert_features'"
    ).fetchone()["n"]
    assert n == 1
    verdicts = {r.number: r.verdict for r in check_laws(conn)}
    assert verdicts[1] != LawVerdict.VIOLATED
    assert verdicts[14] != LawVerdict.VIOLATED


def test_empty_measures_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVENANCE_DUAL_WRITE", "1")
    monkeypatch.setenv("PROVENANCE_STORE_ROOT", str(tmp_path / "store"))
    _maybe_dual_write_provenance(_result([]))
    assert not (tmp_path / "store").exists()


def test_provenance_failure_is_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVENANCE_DUAL_WRITE", "1")
    monkeypatch.setenv("PROVENANCE_STORE_ROOT", str(tmp_path / "store"))

    def _boom(*a, **k):
        raise RuntimeError("provenance store on fire")

    # the helper lazy-imports `from core.provenance import register_computed_feature`
    monkeypatch.setattr("core.provenance.register_computed_feature", _boom)
    # must NOT raise — analysis is unaffected by a provenance failure
    _maybe_dual_write_provenance(_result(_measures()))

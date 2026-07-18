from __future__ import annotations

from workspaces.alignment_prototype.landmark_fp import LandmarkFingerprint
from workspaces.pws_aligner.mix_fp_store import load_or_build

_FP = LandmarkFingerprint(fps=43.0, duration_s=12.0, hashes={(1, 2, 3): (10, 20)})


def test_builds_then_persists_and_reads_without_rebuilding(tmp_path):
    calls = {"n": 0}

    def fake_build(mix_path, **kw):
        calls["n"] += 1
        return _FP

    fp1 = load_or_build(tmp_path, "77", "/mix.m4a", build=fake_build)
    assert fp1.hashes == _FP.hashes
    assert (tmp_path / "77.fp").is_file()
    # second call reads the cache — build NOT invoked again
    fp2 = load_or_build(tmp_path, "77", "/mix.m4a", build=fake_build)
    assert fp2.hashes == _FP.hashes
    assert calls["n"] == 1


def test_corrupt_blob_triggers_rebuild(tmp_path):
    (tmp_path / "9.fp").write_bytes(b"not a valid blob")
    calls = {"n": 0}

    def fake_build(mix_path, **kw):
        calls["n"] += 1
        return _FP

    fp = load_or_build(tmp_path, "9", "/mix.m4a", build=fake_build)
    assert fp.hashes == _FP.hashes
    assert calls["n"] == 1  # rebuilt over the corrupt file


def test_atomic_no_tmp_left_behind(tmp_path):
    load_or_build(tmp_path, "5", "/mix.m4a", build=lambda p, **kw: _FP)
    assert not list(tmp_path.glob("*.tmp"))

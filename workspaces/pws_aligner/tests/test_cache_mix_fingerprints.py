from __future__ import annotations

from workspaces.alignment_prototype.landmark_fp import LandmarkFingerprint
from scripts.cache_mix_fingerprints import distinct_mixes, warm_cache


class _Slot:
    def __init__(self, set_audio_id, mix_full_path):
        self.set_audio_id = set_audio_id
        self.mix_full_path = mix_full_path


def test_distinct_mixes_dedups_by_set_audio_id():
    slots = [
        _Slot(10, "/a.m4a"),
        _Slot(10, "/a.m4a"),
        _Slot(11, "/b.m4a"),
    ]
    assert sorted(distinct_mixes(slots)) == [(10, "/a.m4a"), (11, "/b.m4a")]


def test_warm_cache_builds_then_is_resumable(tmp_path):
    mixes = [(10, "/a.m4a"), (11, "/b.m4a")]
    calls = {"n": 0}

    def fake_build(mix_path, **kw):
        calls["n"] += 1
        return LandmarkFingerprint(fps=43.0, duration_s=1.0, hashes={(1, 1, 1): (0,)})

    built, skipped, failed = warm_cache(mixes, tmp_path, build=fake_build)
    assert (built, skipped, failed) == (2, 0, 0)
    assert calls["n"] == 2
    # second run: both cached → 0 builds
    built2, skipped2, failed2 = warm_cache(mixes, tmp_path, build=fake_build)
    assert (built2, skipped2, failed2) == (0, 2, 0)
    assert calls["n"] == 2


def test_warm_cache_counts_build_failures(tmp_path):
    def boom(mix_path, **kw):
        raise RuntimeError("undecodable")

    built, skipped, failed = warm_cache([(1, "/x.m4a")], tmp_path, build=boom)
    assert (built, skipped, failed) == (0, 0, 1)

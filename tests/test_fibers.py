"""Unit fixtures for the fibers package (Phase A2 of the fiber program).

First direct coverage of `fibers.detect` — until now the only checks were
extrinsic A/B evals on frozen predictions. Fixtures encode the audited failure
modes (see fibers/__init__.py design rules and looptrace AUDIT.md):
copy-paste repeats must group, silence must never fiber, average linkage must
block transitive merges, and two runs must agree byte-for-byte (determinism
is a kernel W1 target — fibers feed timelines).

Synthetic features only (no HuBERT, no audio I/O) so the suite stays in the
fast `make check` tier.
"""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.fibers.detect import (
    _avg_linkage,
    compute_fibers,
    compute_fibers_soft,
    fiber_at,
    fiber_intervals,
    same_fiber,
)

FPS = 43.066  # canonical SR/HOP grid the production callers use


def _rand_feat(rng: np.random.Generator, seconds: float, dim: int = 32) -> np.ndarray:
    """Smooth random (D, T) feature — content-like, non-repeating."""
    t = int(seconds * FPS)
    x = rng.standard_normal((dim, t)).astype(np.float32)
    # temporal smoothing so cosine self-similarity behaves like real features
    k = 15
    kernel = np.ones(k, dtype=np.float32) / k
    return np.stack([np.convolve(row, kernel, mode="same") for row in x])


def _with_repeat(
    rng: np.random.Generator, *, section_s: float, gap_s: float, n_inst: int = 2
) -> np.ndarray:
    """Feature with one section copy-pasted n_inst times, separated by unique
    filler — the bounced-audio-duplication case fibers exist to find."""
    section = _rand_feat(rng, section_s)
    parts = [section]
    for _ in range(n_inst - 1):
        parts += [_rand_feat(rng, gap_s), section.copy()]
    parts += [_rand_feat(rng, gap_s)]
    return np.concatenate(parts, axis=1)


def test_copy_paste_repeat_forms_one_fiber() -> None:
    rng = np.random.default_rng(0)
    feat = _with_repeat(rng, section_s=12.0, gap_s=20.0)
    labels, hz = compute_fibers(feat, FPS)
    assert labels.max() >= 0, "an exact 12s copy-paste repeat must fiber"
    # both instances carry the same label at their midpoints
    assert same_fiber(labels, hz, 6.0, 12.0 + 20.0 + 6.0)


def test_three_instances_group_together() -> None:
    rng = np.random.default_rng(1)
    feat = _with_repeat(rng, section_s=10.0, gap_s=15.0, n_inst=3)
    labels, hz = compute_fibers(feat, FPS)
    mids = [5.0, 10.0 + 15.0 + 5.0, 2 * (10.0 + 15.0) + 5.0]
    ids = {fiber_at(labels, hz, m) for m in mids}
    assert len(ids) == 1 and -1 not in ids


def test_non_repeating_content_yields_no_fibers() -> None:
    rng = np.random.default_rng(2)
    labels, _hz = compute_fibers(_rand_feat(rng, 90.0), FPS)
    assert labels.max() < 0, "unique content must not fiber"


def test_short_repeat_below_min_repeat_s_ignored() -> None:
    rng = np.random.default_rng(3)
    feat = _with_repeat(rng, section_s=3.0, gap_s=25.0)  # < min_repeat_s=6
    labels, _hz = compute_fibers(feat, FPS)
    assert labels.max() < 0


def test_silence_never_fibers_even_though_self_similar(tmp_path) -> None:
    """Silence is perfectly self-similar; the RMS gate must kill it.

    Constant (zero-ish) features + a real silent audio file exercising the
    audio_path gate — the audit-v1 'silence fiber' regression."""
    import soundfile as sf

    dur = 60.0
    feat = np.full((16, int(dur * FPS)), 1e-6, dtype=np.float32)
    wav = tmp_path / "silence.wav"
    sf.write(wav, np.zeros(int(dur * 22050), dtype=np.float32), 22050)
    labels, _hz = compute_fibers(feat, FPS, audio_path=str(wav))
    assert labels.max() < 0, "silence must never form a fiber"


def test_same_fiber_silence_semantics() -> None:
    labels = np.array([-1, -1, 0, 0, -1], dtype=np.int32)
    assert not same_fiber(labels, 1.0, 0.0, 1.0), "-1 pairs are never equivalent"
    assert same_fiber(labels, 1.0, 2.0, 3.0)


def test_avg_linkage_blocks_transitive_merge() -> None:
    """A matches B, B matches C, but A~C is low -> average linkage must not
    chain all three into one class (the audit's 18s~174s=0.62 / 18s~96s=0.00
    case)."""
    sim = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.9],
            [0.0, 0.9, 1.0],
        ]
    )
    groups = _avg_linkage(sim, thresh=0.5)
    assert len(set(groups)) >= 2, "transitive chain must not fully merge"


def test_determinism_two_runs_identical() -> None:
    rng = np.random.default_rng(4)
    feat = _with_repeat(rng, section_s=10.0, gap_s=18.0, n_inst=3)
    a_labels, a_hz = compute_fibers(feat.copy(), FPS)
    b_labels, b_hz = compute_fibers(feat.copy(), FPS)
    assert a_hz == b_hz and np.array_equal(a_labels, b_labels)


def test_soft_outputs_consistent_with_hard_labels() -> None:
    rng = np.random.default_rng(5)
    feat = _with_repeat(rng, section_s=12.0, gap_s=20.0)
    labels, hz, mu, conf = compute_fibers_soft(feat, FPS)
    hard, hard_hz = compute_fibers(feat, FPS)
    assert hz == hard_hz and np.array_equal(labels, hard)
    assert mu.shape == labels.shape
    fibered = labels >= 0
    if fibered.any():
        assert float(mu[fibered].max()) <= 1.0 + 1e-6
        assert float(mu[fibered].mean()) > 0.0
        assert all(0.0 <= v <= 1.0 + 1e-6 for v in conf.values())


def test_fiber_intervals_roundtrip() -> None:
    labels = np.array(
        [-1] * 10 + [0] * 50 + [-1] * 10 + [0] * 50 + [1] * 40, dtype=np.int32
    )
    hz = 8.0
    ivs = fiber_intervals(labels, hz, min_len_s=4.0)
    assert [(round(s, 2), round(e, 2), l) for s, e, l in ivs] == [
        (1.25, 7.5, 0),
        (8.75, 15.0, 0),
        (15.0, 20.0, 1),
    ]


def test_breathy_repeat_survives_silence_gate(tmp_path) -> None:
    """A repeated vocal phrase with breath gaps every ~1.6s must still fiber.

    The old per-frame gate ANDed silence into the diagonal run, fragmenting it
    below min_repeat_s at every breath — the structural acappella recall hole
    (looptrace audit-v2). The per-run voiced-fraction gate must keep it."""
    import soundfile as sf

    rng = np.random.default_rng(6)
    sr = 22050
    # one 12s "phrase": 1.6s voiced bursts separated by 0.4s breaths
    burst = rng.standard_normal(int(1.6 * sr)).astype(np.float32) * 0.3
    breath = np.zeros(int(0.4 * sr), dtype=np.float32)
    phrase = np.concatenate([np.concatenate([burst, breath]) for _ in range(6)])
    filler = rng.standard_normal(int(20 * sr)).astype(np.float32) * 0.3
    y = np.concatenate([phrase, filler, phrase, np.zeros(int(5 * sr), np.float32)])
    wav = tmp_path / "breathy.wav"
    sf.write(wav, y, sr)

    # matching features: repeat the same smooth content at both positions,
    # near-zero columns during breaths (silence is self-similar)
    def _sec(rng2, s):
        return _rand_feat(rng2, s)

    rng2 = np.random.default_rng(7)
    b_feat = _sec(rng2, 1.6)
    br_feat = np.full((32, int(0.4 * FPS)), 1e-6, dtype=np.float32)
    phrase_feat = np.concatenate(
        [np.concatenate([b_feat, br_feat], axis=1) for _ in range(6)], axis=1
    )
    fill_feat = _sec(rng2, 20.0)
    tail = np.full((32, int(5 * FPS)), 1e-6, dtype=np.float32)
    feat = np.concatenate([phrase_feat, fill_feat, phrase_feat.copy(), tail], axis=1)

    labels, hz = compute_fibers(feat, FPS, audio_path=str(wav))
    assert labels.max() >= 0, "breathy repeat must fiber under the per-run gate"
    assert same_fiber(labels, hz, 3.0, 12.0 + 20.0 + 3.0)

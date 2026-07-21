from __future__ import annotations

import numpy as np
from PIL import Image

from eda.alignment.spectrogram_review.classify import (
    binding_cause,
    is_success,
    success_type,
)
from eda.alignment.spectrogram_review.labels import (
    fmt_clock,
    fmt_delta,
    group_title,
    one_liner,
)
from eda.alignment.spectrogram_review.spans import SpanCard, enrich_row, select_cards
from eda.alignment.spectrogram_review.stats import weak_axes_blurb
from eda.alignment.spectrogram_review.spectrogram import (
    TimeBox,
    draw_od_boxes,
    mel_rgb,
    to_square,
    window_bounds,
)


def test_is_success_requires_traj_and_identity() -> None:
    ok = {
        "span_class": "linear",
        "identity_hit": 1,
        "traj_strict": 0.6,
        "set_start_err_s": 1.0,
        "gt_stem": "regular",
        "pred_stem": "regular",
    }
    assert is_success(ok)
    assert not is_success({**ok, "traj_strict": 0.4})
    assert not is_success({**ok, "identity_hit": 0})
    assert not is_success({**ok, "span_class": ""})


def test_binding_cause_dependency_order() -> None:
    base = {
        "identity_hit": 1,
        "set_start_err_s": 1.0,
        "gt_stem": "regular",
        "pred_stem": "regular",
        "span_class": "linear",
        "frac_distinct": 0.0,
        "traj_strict": 0.1,
    }
    assert binding_cause({**base, "identity_hit": 0}) == "identity"
    assert binding_cause({**base, "set_start_err_s": 20.0}) == "placement"
    assert (
        binding_cause({**base, "gt_stem": "acappella", "pred_stem": "regular"})
        == "mis-route"
    )
    assert binding_cause({**base, "span_class": "oddratio"}) == "tempo/octave"
    assert binding_cause({**base, "span_class": "loop"}) == "loop-instance"
    assert binding_cause({**base, "frac_distinct": 0.5}) == "instance-ambiguity"
    assert binding_cause(base) == "decode-residual"


def test_success_type_buckets_stem_class_place() -> None:
    row = {
        "gt_stem": "acappella",
        "span_class": "loop",
        "set_start_err_s": 2.0,
        "identity_hit": 1,
        "traj_strict": 0.9,
        "pred_stem": "acappella",
    }
    assert success_type(row) == "acappella · loop · tight"
    assert (
        success_type({**row, "set_start_err_s": 12.0})
        == "acappella · loop · loose-place"
    )


def test_window_bounds_union_and_clamp() -> None:
    lo, hi = window_bounds(100.0, 120.0, 105.0, 125.0, pad_s=5.0, max_window_s=90.0)
    assert lo == 95.0
    assert hi == 130.0
    lo, hi = window_bounds(
        0.0, 10.0, 200.0, 210.0, pad_s=5.0, max_window_s=40.0, mix_duration_s=300.0
    )
    assert hi - lo == 40.0
    assert lo >= 0.0


def test_to_square_is_square() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=22_050 * 3).astype(np.float32)
    native = mel_rgb(y, sr=22_050, hop_length=512)
    sq = to_square(native, size=256)
    assert sq.size == (256, 256)


def test_draw_od_boxes_marks_actual_and_pred() -> None:
    img = Image.new("RGB", (200, 200), color=(0, 0, 0))
    out = draw_od_boxes(
        img,
        [
            TimeBox(1.0, 3.0, class_name="ACTUAL", color=(80, 220, 120), band="top"),
            TimeBox(2.0, 4.0, class_name="PRED", color=(255, 60, 220), band="bottom"),
        ],
        win_start_s=0.0,
        win_end_s=10.0,
        edge_color=(0, 255, 0),
    )
    assert out.size == (200, 200)
    assert out.getpixel((0, 0)) == (0, 255, 0)
    assert out.getpixel((20, 4)) == (80, 220, 120)
    assert out.getpixel((40, int(200 * 0.52))) == (255, 60, 220)


def _card(
    *,
    success: bool,
    sec_lost: float,
    traj: float,
    slot: str,
    group: str,
    stem: str = "regular",
) -> SpanCard:
    return SpanCard(
        set="BB12",
        set_id="1fsnxchk",
        slot=slot,
        recording_id="x",
        name="n",
        gt_stem=stem,
        pred_stem="regular",
        span_class="linear",
        gt_set_start_s=0.0,
        gt_set_end_s=10.0,
        pred_set_start_s=0.0,
        pred_set_end_s=10.0,
        gt_ref_start_s=5.0,
        gt_ref_end_s=15.0,
        pred_ref_start_s=8.0,
        pred_ref_end_s=18.0,
        ref_offset_err_s=3.0,
        tempo_ratio=1.0,
        identity_hit=1,
        set_start_err_s=0.0,
        traj_strict=traj,
        success=success,
        cause="decode-residual" if success else group,
        group_key=group if not success else f"{stem} · linear · tight",
        win_start_s=0.0,
        win_end_s=15.0,
        ref_win_start_s=0.0,
        ref_win_end_s=20.0,
        sec_lost=sec_lost,
    )


def test_select_cards_diversifies_failure_causes() -> None:
    cards = [
        _card(success=False, sec_lost=50, traj=0.1, slot="1", group="placement"),
        _card(success=False, sec_lost=49, traj=0.1, slot="2", group="placement"),
        _card(success=False, sec_lost=40, traj=0.1, slot="3", group="identity"),
        _card(success=False, sec_lost=30, traj=0.1, slot="4", group="decode-residual"),
        _card(success=True, sec_lost=1, traj=0.9, slot="5", group="ok", stem="regular"),
        _card(
            success=True, sec_lost=2, traj=0.8, slot="6", group="ok", stem="acappella"
        ),
    ]
    picked = select_cards(cards, limit=4, outcome="all")
    assert len(picked) == 4
    fail_groups = {c.group_key for c in picked if not c.success}
    assert len(fail_groups) >= 2


def test_enrich_row_includes_ref_windows() -> None:
    assert enrich_row({"span_class": "", "slot": "1"}) is None
    card = enrich_row(
        {
            "set": "BB12",
            "set_id": "1fsnxchk",
            "slot": "12",
            "recording_id": "abc",
            "name": "Track",
            "gt_stem": "acappella",
            "pred_stem": "regular",
            "span_class": "linear",
            "gt_set_start_s": "100",
            "gt_duration_s": "20",
            "gt_audible_s": "18",
            "gt_ref_start_s": "40",
            "pred_set_start_s": "130",
            "pred_ref_start_s": "55",
            "tempo_ratio": "1.0",
            "identity_hit": "1",
            "set_start_err_s": "30",
            "traj_strict": "0.1",
            "ref_offset_err_s": "15",
            "frac_distinct": "",
        }
    )
    assert card is not None
    assert card.success is False
    assert card.cause == "placement"
    assert card.group_key == "placement"
    assert card.gt_ref_start_s == 40.0
    assert card.gt_ref_end_s == 60.0
    assert card.pred_ref_start_s == 55.0
    assert card.ref_win_start_s < card.gt_ref_start_s
    assert card.ref_offset_err_s == 15.0


def test_ableton_track_num_from_slot() -> None:
    from eda.alignment.spectrogram_review.ableton_label import parse_ableton_track_num

    assert parse_ableton_track_num("053") == 53
    assert parse_ableton_track_num("42w3") == 42
    assert parse_ableton_track_num("6w2") == 6


def test_human_labels_are_plain_english() -> None:
    assert fmt_clock(125.0) == "2:05.0"
    assert "too late" in fmt_delta(12.0)
    assert "too early" in fmt_delta(-3.0)
    assert group_title("placement", success=False) == "Wrong time in the mix"
    card = _card(
        success=False,
        sec_lost=40,
        traj=0.0,
        slot="9",
        group="placement",
        stem="acappella",
    )
    # place_delta uses pred-gt starts; both 0 in _card → "about on time" path avoided
    # by using enrich-style values via replace
    from dataclasses import replace

    card = replace(card, pred_set_start_s=30.0, cause="placement", success=False)
    line = one_liner(card)
    assert "Missed" in line
    assert "mix" in line.lower()


def test_weak_axes_blurb_ranks_stems() -> None:
    cards = [
        _card(
            success=False,
            sec_lost=100,
            traj=0.1,
            slot="1",
            group="placement",
            stem="acappella",
        ),
        _card(
            success=True,
            sec_lost=1,
            traj=0.9,
            slot="2",
            group="ok",
            stem="regular",
        ),
    ]
    # fix stem on cards — _card sets gt_stem from stem arg
    blurb = weak_axes_blurb(cards)
    assert blurb["n"] == 2
    assert blurb["worst_stems"][0]["key"] == "acappella"


def test_mix_truth_uses_audible_not_silent_clip_head() -> None:
    """Honest instrumental: Truth must not start at silent clip set_start."""
    from eda.alignment.spectrogram_review.audible import mix_truth_span, ref_truth_span
    from eda.alignment.spectrogram_review.spans import enrich_row, load_span_rows
    from pathlib import Path

    _span_csv = Path("eda/alignment/failure_analysis/out/span_table.csv")
    if not _span_csv.exists():
        import pytest

        pytest.skip("requires generated span_table.csv (failure_analysis output)")
    rows = load_span_rows(_span_csv)
    row = next(
        r
        for r in rows
        if r.get("set_id") == "1fsnxchk"
        and r.get("slot") == "42w3"
        and r.get("span_class")
    )
    card = enrich_row(row)
    assert card is not None
    lo, hi = mix_truth_span(card)
    # audible lead-in: Truth starts after clip set_start when frac/end say so
    assert lo > card.gt_set_start_s + 5.0
    assert hi <= card.gt_set_end_s + 1.0
    r0, r1 = ref_truth_span(card, source_path=None)
    assert r0 > 5.0  # not sitting on silent stem head at ref 0


def test_resolve_ableton_picks_audible_truth_not_prediction_recording() -> None:
    """42w3: pred said Honest acappella (dgrjwux); audible truth is SAVI remix.

    The old resolver filtered GT rows by the prediction's recording_id and so
    landed on lane 149 (instrumental, ducked under SAVI). The fix picks by
    audible overlap + stem agreement → lane 152 (SAVI remix, the vocal layer).
    """
    from pathlib import Path

    from eda.alignment.spectrogram_review.ableton_label import resolve_ableton
    from eda.alignment.spectrogram_review.spans import enrich_row, load_span_rows

    _span_csv = Path("eda/alignment/failure_analysis/out/span_table.csv")
    if not _span_csv.exists():
        import pytest

        pytest.skip("requires generated span_table.csv (failure_analysis output)")
    rows = load_span_rows(_span_csv)
    row = next(
        r
        for r in rows
        if r.get("set_id") == "1fsnxchk"
        and r.get("slot") == "42w3"
        and r.get("span_class")
    )
    card = enrich_row(row)
    assert card is not None
    ab = resolve_ableton(
        card.set_id,
        recording_id=card.recording_id,
        gt_set_start_s=card.gt_set_start_s,
        fallback_slot=card.slot,
        fallback_name=card.name,
        gt_stem=card.gt_stem,
        pred_set_start_s=card.pred_set_start_s,
        pred_set_end_s=card.pred_set_end_s,
        pred_stem=card.pred_stem,
    )
    assert ab["ableton_slot"] == "152"
    assert "SAVI" in ab["ableton_name"]
    assert ab["ableton_recording_id"] == "h80tgnf"
    assert ab["ableton_stem"] == "regular"


def test_resolve_ableton_clean_hit_still_resolves_own_row() -> None:
    """A correct pred (≈ gt, same stem, same recording) must still resolve to its
    own GT row — the audible-overlap heuristic must not regress clean hits."""
    from pathlib import Path

    from eda.alignment.spectrogram_review.ableton_label import resolve_ableton
    from eda.alignment.spectrogram_review.spans import enrich_row, load_span_rows

    _span_csv = Path("eda/alignment/failure_analysis/out/span_table.csv")
    if not _span_csv.exists():
        import pytest

        pytest.skip("requires generated span_table.csv (failure_analysis output)")
    rows = load_span_rows(_span_csv)
    # 42w5: SAVI remix, identity_hit=1, traj_strict=0.95 — a clean hit.
    row = next(
        r
        for r in rows
        if r.get("set_id") == "1fsnxchk"
        and r.get("slot") == "42w5"
        and r.get("span_class")
    )
    card = enrich_row(row)
    assert card is not None
    ab = resolve_ableton(
        card.set_id,
        recording_id=card.recording_id,
        gt_set_start_s=card.gt_set_start_s,
        fallback_slot=card.slot,
        fallback_name=card.name,
        gt_stem=card.gt_stem,
        pred_set_start_s=card.pred_set_start_s,
        pred_set_end_s=card.pred_set_end_s,
        pred_stem=card.pred_stem,
    )
    assert ab["ableton_slot"] == "152"
    assert ab["ableton_recording_id"] == card.recording_id


def test_stem_energy_onset_skips_leading_silence(tmp_path) -> None:
    import subprocess
    from eda.alignment.spectrogram_review.audible import stem_energy_onset_s

    silence = tmp_path / "s.wav"
    tone = tmp_path / "t.wav"
    wav = tmp_path / "pad.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            "1",
            str(silence),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:r=44100",
            "-t",
            "1",
            str(tone),
        ],
        check=True,
    )
    lst = tmp_path / "list.txt"
    lst.write_text(f"file '{silence}'\nfile '{tone}'\n")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c",
            "copy",
            str(wav),
        ],
        check=True,
    )
    onset = stem_energy_onset_s(wav, thresh_rms=0.01, hop_s=0.1, max_scan_s=3.0)
    assert onset is not None
    assert 0.7 <= onset <= 1.3

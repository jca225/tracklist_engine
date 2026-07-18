from workspaces.alignment_prototype.evals import oracle_ladder as ol


def _r0():
    return [
        # matched, mis-identified, stale stem
        {
            "slot_label": "1w1",
            "recording_id": "PRED_A",
            "set_start_s": 10.0,
            "set_end_s": 30.0,
            "claimed_stem": "regular",
            "ref_start_s": 5.0,
        },
        # matched, correctly identified already
        {
            "slot_label": "2w2",
            "recording_id": "GT_B",
            "set_start_s": 50.0,
            "set_end_s": 70.0,
            "claimed_stem": "acappella",
            "ref_start_s": 0.0,
        },
    ]


def _gt():
    return [
        {
            "slot_label": "1w1",
            "track_id": "GT_A",
            "claimed_stem": "acappella",
            "set_start_s": 12.0,
            "set_end_s": 31.0,
        },
        {
            "slot_label": "2w2",
            "track_id": "GT_B",
            "claimed_stem": "acappella",
            "set_start_s": 49.0,
            "set_end_s": 69.0,
        },
        {
            "slot_label": "9w9",
            "track_id": "GT_C",
            "claimed_stem": "acappella",
            "set_start_s": 90.0,
            "set_end_s": 110.0,
        },  # never-matched in R0
    ]


def test_r0_keeps_matched_spans_unchanged():
    out = ol.build_rung_timeline("R0", _r0(), _gt())
    slots = sorted(s["slot_label"] for s in out)
    assert slots == ["1w1", "2w2"]  # 9w9 has no r0 span
    a = next(s for s in out if s["slot_label"] == "1w1")
    assert a["recording_id"] == "PRED_A" and a["claimed_stem"] == "regular"


def test_r1_forces_acappella_routing_only():
    out = ol.build_rung_timeline("R1", _r0(), _gt())
    a = next(s for s in out if s["slot_label"] == "1w1")
    assert a["claimed_stem"] == "acappella"  # routing fixed
    assert a["recording_id"] == "PRED_A"  # identity NOT fixed
    assert a["set_start_s"] == 10.0  # placement NOT fixed


def test_r2_fixes_identity_keeps_predicted_placement():
    out = ol.build_rung_timeline("R2", _r0(), _gt())
    a = next(s for s in out if s["slot_label"] == "1w1")
    assert a["recording_id"] == "GT_A"  # identity fixed to GT
    assert a["claimed_stem"] == "acappella"
    assert a["set_start_s"] == 10.0  # predicted placement retained
    assert all(s["slot_label"] != "9w9" for s in out)  # never-matched omitted


def test_r3_full_oracle_covers_all_gt_rows():
    out = ol.build_rung_timeline("R3", _r0(), _gt())
    slots = sorted(s["slot_label"] for s in out)
    assert slots == ["1w1", "2w2", "9w9"]  # includes never-matched
    c = next(s for s in out if s["slot_label"] == "9w9")
    assert c["recording_id"] == "GT_C" and c["set_start_s"] == 90.0
    assert c["set_end_s"] == 110.0 and c["claimed_stem"] == "acappella"
    assert "ref_start_s" in c  # placeholder present for decoder


def test_summarize_fixed_denominator_missing_scores_zero():
    gt = _gt()  # 3 rows
    decoded = [
        {"slot_label": "1w1", "recording_id": "GT_A"},
        {"slot_label": "2w2", "recording_id": "GT_B"},
        # 9w9 absent -> must count as 0 in BOTH strict and fiber
    ]
    scores = {"1w1": (1.0, 1.0), "2w2": (0.0, 0.5)}

    def score_fn(span, gt_row):
        return scores[ol.norm_slot(str(span["slot_label"]))]

    out = ol.summarize_acappella(decoded, gt, score_fn)
    assert out["n"] == 3 and out["n_scored"] == 2
    assert abs(out["strict"] - (1.0 + 0.0 + 0.0) / 3) < 1e-9
    assert abs(out["fiber"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9

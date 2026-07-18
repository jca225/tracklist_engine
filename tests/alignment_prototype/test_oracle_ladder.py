from workspaces.alignment_prototype.evals import oracle_ladder as ol


def _r0():
    # timeline slots use the w-layer convention; GT uses flat numeric slots.
    # The join is by recording_id (+ time), NEVER by slot_label.
    return [
        # covers GT_A's moment but predicted the WRONG recording (mis-identified)
        {
            "slot_label": "1w1",
            "recording_id": "PRED_A",
            "set_start_s": 10.0,
            "set_end_s": 30.0,
            "claimed_stem": "regular",
            "ref_start_s": 5.0,
        },
        # correctly identified GT_B, but stale routing (regular)
        {
            "slot_label": "2w2",
            "recording_id": "GT_B",
            "set_start_s": 50.0,
            "set_end_s": 70.0,
            "claimed_stem": "regular",
            "ref_start_s": 0.0,
        },
    ]


def _gt():
    return [
        {
            "slot_label": "002",
            "track_id": "GT_A",
            "claimed_stem": "acappella",
            "set_start_s": 12.0,
            "set_end_s": 31.0,
        },  # mis-identified in R0
        {
            "slot_label": "004",
            "track_id": "GT_B",
            "claimed_stem": "acappella",
            "set_start_s": 49.0,
            "set_end_s": 69.0,
        },  # correctly identified
        {
            "slot_label": "009",
            "track_id": "GT_C",
            "claimed_stem": "acappella",
            "set_start_s": 300.0,
            "set_end_s": 320.0,
        },  # never covered at all
    ]


def _stamp(span):
    return span[ol.GT_IDX_KEY]


def test_r0_only_correctly_identified_rows_get_a_span():
    out = ol.build_rung_timeline("R0", _r0(), _gt())
    # GT_A mis-identified (no same-recording span) -> no R0 span
    # GT_B correctly identified -> its span, unchanged (stale routing kept)
    # GT_C never covered -> no span
    stamps = sorted(_stamp(s) for s in out)
    assert stamps == [1]  # only GT_B (index 1)
    b = out[0]
    assert b["recording_id"] == "GT_B" and b["claimed_stem"] == "regular"


def test_r1_routing_oracle_on_identified_span_only():
    out = ol.build_rung_timeline("R1", _r0(), _gt())
    stamps = sorted(_stamp(s) for s in out)
    assert stamps == [1]  # GT_A still unrecoverable (wrong identity)
    assert out[0]["claimed_stem"] == "acappella"  # routing fixed
    assert out[0]["recording_id"] == "GT_B"  # identity untouched


def test_r2_identity_oracle_recovers_misidentified_via_time_span():
    out = ol.build_rung_timeline("R2", _r0(), _gt())
    by_idx = {_stamp(s): s for s in out}
    assert set(by_idx) == {0, 1}  # GT_A recovered, GT_B kept; GT_C not
    a = by_idx[0]
    assert a["recording_id"] == "GT_A"  # identity fixed to GT
    assert a["claimed_stem"] == "acappella"
    assert a["set_start_s"] == 10.0  # predicted placement (time-span) retained
    assert 2 not in by_idx  # GT_C never covered -> no time span


def test_r3_full_oracle_covers_all_rows():
    out = ol.build_rung_timeline("R3", _r0(), _gt())
    by_idx = {_stamp(s): s for s in out}
    assert set(by_idx) == {0, 1, 2}  # all GT rows, including never-covered
    c = by_idx[2]
    assert c["recording_id"] == "GT_C" and c["set_start_s"] == 300.0
    assert c["set_end_s"] == 320.0 and c["claimed_stem"] == "acappella"
    assert "ref_start_s" in c


def test_summarize_fixed_denominator_by_gt_index_missing_scores_zero():
    gt = _gt()  # 3 rows
    decoded = [
        {ol.GT_IDX_KEY: 0, "recording_id": "GT_A"},
        {ol.GT_IDX_KEY: 1, "recording_id": "GT_B"},
        # index 2 (GT_C) absent -> counts as 0 in BOTH strict and fiber
    ]
    scores = {0: (1.0, 1.0), 1: (0.0, 0.5)}

    def score_fn(span, gt_row):
        return scores[span[ol.GT_IDX_KEY]]

    out = ol.summarize_acappella(decoded, gt, score_fn)
    assert out["n"] == 3 and out["n_scored"] == 2
    assert abs(out["strict"] - (1.0 + 0.0 + 0.0) / 3) < 1e-9
    assert abs(out["fiber"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9

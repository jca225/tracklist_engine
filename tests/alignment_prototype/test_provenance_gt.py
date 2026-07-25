"""Brick 6 adapter — the shipped GT YAML expressed as §7 human-label provenance.

The load-bearing behaviors: every asserted field carries its OWN uncertainty
model (Law 9); identity abstains/placeholders yield persisted abstentions, never
a smuggled recording (Laws 4/7); slot-less annotator rows are persisted
diagnostics, not silent drops (Law 6); and grounding on the real BB12 fixture
flips laws 8 AND 9 to PASS with nothing else violated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    LawVerdict,
    ProvenanceRepository,
    check_laws,
    connect,
)
from workspaces.alignment_prototype.provenance_gt import (
    import_gt_bundle,
    run_gt_import_for_set,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BB12_FIXTURE = REPO_ROOT / "labeling" / "fixtures" / "bb12_ground_truth.yaml"


@pytest.fixture
def env(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    source = store.put_bytes(
        b"set_id: 1fsnxchk\n",
        kind=ArtifactKind.LABEL_MANIFEST,
        media_type="application/x-yaml",
        source_system="labeling.ground_truth",
    )
    return conn, store, repo, source


GT = {
    "set_id": "1fsnxchk",
    "annotated_by": "user",
    "tracks": [
        {  # fully identified row -> assertions across field families
            "track": "A - B Acapella",
            "slot_label": "002",
            "track_id": "281u6p4x",
            "claimed_stem": "acappella",
            "set_start_s": 64.4327,
            "set_end_s": 94.3862,
            "ref_start_s": 170.004,
            "ref_end_s": 200.728,
            "tempo_ratio": 1.02573,
            "pitch_shift_semi": 1,
            "gain_curve": [[64.433, 1.0471], [94.386, 0.0003]],
            "id_source": "content",
        },
        {  # identity abstain -> NO recording_id assertion, persisted abstention
            "track": "C - D",
            "slot_label": "003",
            "claimed_stem": "instrumental",
            "set_start_s": 64.4457,
            "set_end_s": 129.767,
            "ref_start_s": 30.7515,
            "ref_end_s": 206.514,
            "tempo_ratio": 1,
            "id_source": "abstain",
        },
        {  # tlp placeholder -> treated as unidentified, never an identity
            "track": "E - F",
            "slot_label": "004",
            "track_id": "tlp2594105",
            "set_start_s": 79.0,
            "set_end_s": 136.7,
            "ref_start_s": 39.6,
            "ref_end_s": 157.1,
            "tempo_ratio": 1.008,
            "id_source": "content",
        },
        {  # annotator helper row with no slot -> persisted, not dropped
            "track": "mix",
            "set_start_s": 1453.32,
            "set_end_s": 3273.63,
            "ref_start_s": 0.0,
            "ref_end_s": 1820.31,
            "tempo_ratio": 1,
            "id_source": "abstain",
            "unalignable": True,
        },
    ],
}


def test_each_field_gets_its_own_uncertainty_model(env):
    conn, store, repo, source = env
    imported = import_gt_bundle("1fsnxchk", GT, store=store, repo=repo, source=source)
    by_field = {}
    for a in imported.assertions:
        by_field.setdefault(a.field, a.uncertainty_model)
    assert by_field["recording_id"]["kind"] == "categorical_prior"
    assert by_field["mix_start_s"]["kind"] == "student_t"
    assert by_field["mix_start_s"]["scale_s"] == 0.25
    assert by_field["ref_end_s"]["kind"] == "student_t"
    assert by_field["tempo_ratio"]["kind"] == "curve_error"
    assert by_field["gain_curve"]["kind"] == "curve_error"
    # DIFFERENT models across fields — never one global scalar (Law 9).
    assert by_field["recording_id"] != by_field["claimed_stem"]
    assert by_field["tempo_ratio"] != by_field["gain_curve"]
    assert all(a.uncertainty_model for a in imported.assertions)


def test_abstain_and_placeholder_rows_never_assert_identity(env):
    conn, store, repo, source = env
    imported = import_gt_bundle("1fsnxchk", GT, store=store, repo=repo, source=source)
    rec_ids = [
        a.observed_value for a in imported.assertions if a.field == "recording_id"
    ]
    assert rec_ids == ["281u6p4x"]  # abstain + tlp rows assert nothing
    assert imported.n_identity_abstain == 2
    diags = {
        r["diagnostic_code"]
        for r in conn.execute(
            "SELECT diagnostic_code FROM observation "
            "WHERE predicate='gt_recording_id' AND status='abstained'"
        )
    }
    assert diags == {"gt_identity_abstain", "placeholder_track_id"}


def test_slotless_row_is_persisted_not_dropped(env):
    conn, store, repo, source = env
    imported = import_gt_bundle("1fsnxchk", GT, store=store, repo=repo, source=source)
    assert imported.n_unlabeled_rows == 1
    row = conn.execute(
        "SELECT subject_id, status, diagnostic_code FROM observation "
        "WHERE predicate='gt_row_without_slot'"
    ).fetchone()
    assert row["subject_id"] == "1fsnxchk::mix"
    assert row["status"] == "abstained"
    assert row["diagnostic_code"] == "no_slot_label"


def test_subject_matches_bricks_2b_and_3(env):
    conn, store, repo, source = env
    imported = import_gt_bundle("1fsnxchk", GT, store=store, repo=repo, source=source)
    subjects = {a.subject.subject_id for a in imported.assertions}
    assert "1fsnxchk::002" in subjects
    assert all(a.subject.subject_type == "set-slot" for a in imported.assertions)


def test_grounded_bb12_fixture_flips_laws_8_and_9_to_pass(tmp_path):
    if not BB12_FIXTURE.exists():
        pytest.skip("shipped BB12 GT fixture not present")
    imported = run_gt_import_for_set("1fsnxchk", db_root=tmp_path, gt_path=BB12_FIXTURE)
    assert len(imported.assertions) > 500  # real GT, not a toy
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    by = {r.number: r for r in check_laws(conn, store)}
    assert by[8].verdict is LawVerdict.PASS and by[8].checked > 0
    assert by[9].verdict is LawVerdict.PASS and by[9].checked > 0
    assert not any(r.verdict is LawVerdict.VIOLATED for r in by.values())


def test_wrong_set_id_fails_fast(tmp_path):
    if not BB12_FIXTURE.exists():
        pytest.skip("shipped BB12 GT fixture not present")
    with pytest.raises(SystemExit, match="is for set"):
        run_gt_import_for_set("2nvzlh2k", db_root=tmp_path, gt_path=BB12_FIXTURE)

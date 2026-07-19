from workspaces.alignment_prototype.fp_segments.fuse import corroborate_segments
from workspaces.alignment_prototype.fp_segments.schema import ConstituentSegment


def _seg(channel, m0, m1, r0, *, confidence=0.4):
    return ConstituentSegment(
        "r", None, channel, channel, m0, m1, r0, r0 + (m1 - m0), 1.0, 10.0, confidence
    )


def test_agreement_boosts_without_shifting_primary():
    primary = _seg("instrumental", 10, 20, 30)
    result = corroborate_segments((primary,), {"full": (_seg("regular", 11, 19, 31),)})
    assert result[0].segment is primary
    assert result[0].agreeing_channels == ("full",)
    assert result[0].corroborated_confidence > primary.confidence


def test_corrupted_channel_penalizes_but_cannot_shift():
    primary = _seg("instrumental", 10, 20, 30)
    wrong = _seg("regular", 10, 20, 80)
    result = corroborate_segments((primary,), {"full": (wrong,)})
    assert result[0].segment is primary
    assert result[0].contradicting_channels == ("full",)
    assert result[0].corroborated_confidence < primary.confidence


def test_missing_is_distinct_from_available_negative():
    primary = _seg("instrumental", 10, 20, 30)
    missing = corroborate_segments((primary,), {"full": None})[0]
    negative = corroborate_segments((primary,), {"full": ()})[0]
    assert missing.missing_channels == ("full",)
    assert missing.corroborated_confidence == primary.confidence
    assert negative.contradicting_channels == ("full",)
    assert negative.corroborated_confidence < primary.confidence

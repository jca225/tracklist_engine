"""Tests for the .als semantic well-formedness pass."""

from __future__ import annotations

from lxml import etree

from labeling.als.validate import has_errors, validate_session
from tests.labeling.synth_session import session_root, session_xml


def _codes(diags):
    return {d.code for d in diags}


def test_clean_synthetic_session_validates():
    diags = validate_session(session_root())
    assert not has_errors(diags), [d.render() for d in diags]
    assert "version-unknown" not in _codes(diags)


def test_duplicate_pointee_id_is_error():
    xml = session_xml().replace(
        b'AutomationTarget Id="9000"', b'AutomationTarget Id="8770"'
    )
    diags = validate_session(etree.fromstring(xml))
    assert "pointee-dup" in _codes(diags)
    assert has_errors(diags)


def test_nonpositive_tempo_is_error():
    diags = validate_session(session_root(tempo_events=((0.0, 120.0), (64.0, -5.0))))
    assert "tempo-nonpositive" in _codes(diags)
    assert has_errors(diags)


def test_malformed_tempo_value_is_error():
    xml = session_xml(tempo_events=((0.0, 120.0),)).replace(
        b'Value="120.0"', b'Value="nan"', 1
    )
    diags = validate_session(etree.fromstring(xml))
    assert "tempo-malformed" in _codes(diags)


def test_duplicate_beat_warp_cluster_is_warning():
    # the Aftershock class: markers clustered on a single beat
    diags = validate_session(session_root(mix_warp_points=((0.0, 0.366), (0.0, 0.388))))
    assert "warp-duplicate-beats" in _codes(diags)
    assert not has_errors(diags)


def test_nonmonotonic_warp_sec_is_warning():
    diags = validate_session(session_root(mix_warp_points=((0.0, 10.0), (64.0, 5.0))))
    assert "warp-sec-nonmonotonic" in _codes(diags)


def test_incomplete_clip_is_warning():
    xml = session_xml().replace(b'<CurrentStart Value="16.0"/>', b"", 1)
    diags = validate_session(etree.fromstring(xml))
    assert "clip-incomplete" in _codes(diags)


def test_gain_out_of_range_is_warning():
    xml = session_xml().replace(b'Time="16.0" Value="1.0"', b'Time="16.0" Value="7.5"')
    diags = validate_session(etree.fromstring(xml))
    assert "gain-out-of-range" in _codes(diags)


def test_unknown_version_is_warning():
    xml = session_xml().replace(b'MajorVersion="5"', b'MajorVersion="6"')
    diags = validate_session(etree.fromstring(xml))
    assert "version-unknown" in _codes(diags)
    assert not has_errors(diags)


def test_validator_total_on_hostile_root():
    # not a session at all — must return diagnostics, never raise
    diags = validate_session(etree.fromstring(b"<NotAbleton><Junk/></NotAbleton>"))
    assert has_errors(diags)  # version-unknown error at minimum

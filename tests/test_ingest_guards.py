"""Decision tests for the shared acquisition guards (ingest/guards.py) and the
set-mix link preference, replayed from failure classes the corpus already paid
for. Companion dataset: tests/fixtures/ingest/correction_ledger_snapshot_*.tsv
— a frozen snapshot of pi's `track_audio_correction` (every hand-ledgered
acquisition failure through 2026-07-09). The snapshot's invariants are asserted
here so the history stays loadable offline as a fixture for future gate tuning.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ingest.guards import duration_sane, parse_play_time, sniff_audio_payload
from ingest.main import _pick_mix_link
from core.models import SetMediaLink

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ingest"
LEDGER = FIXTURES / "correction_ledger_snapshot_20260709.tsv"


# ── parse_play_time ───────────────────────────────────────────────────────────


def test_play_time_formats():
    # the four dominant scrape formats (corpus: '1h' x5944, '59m' x5160, ...)
    assert parse_play_time("1h") == 3600
    assert parse_play_time("59m") == 3540
    assert parse_play_time("1h 2m") == 3720
    assert parse_play_time("2h") == 7200


def test_play_time_garbage_is_none():
    assert parse_play_time("") is None
    assert parse_play_time(None) is None
    assert parse_play_time("about an hour") is None


# ── duration_sane ─────────────────────────────────────────────────────────────


def test_duration_accepts_real_pairs():
    # real canonical pairs: BB12 '1h 2m'/3729s, BB11 '1h 1m'/3581s
    assert duration_sane(3729.0, parse_play_time("1h 2m"))
    assert duration_sane(3581.0, parse_play_time("1h 1m"))


def test_duration_rejects_preview_clip_class():
    # the 46s-preview class (project_wrong_version_preview_clip): a teaser
    # must never become canonical audio for a listed 1h set
    assert not duration_sane(46.0, parse_play_time("1h"))


def test_duration_rejects_stream_rip_class():
    # a 7h livestream VOD grabbed for a listed 1h set is the wrong asset
    assert not duration_sane(7 * 3600.0, parse_play_time("1h"))


def test_duration_passes_when_unverifiable():
    # conservative: nothing listed / nothing measured -> nothing to contradict
    assert duration_sane(None, 3600)
    assert duration_sane(46.0, None)
    assert duration_sane(None, None)


# ── sniff_audio_payload ───────────────────────────────────────────────────────


def test_sniff_accepts_known_audio_magics():
    for head in (
        b"ID3\x04\x00" + b"\x00" * 8,  # mp3+ID3
        b"fLaC\x00\x00\x00\x22" + b"\x00" * 8,  # flac
        b"OggS\x00\x02" + b"\x00" * 8,  # ogg/opus
        b"RIFF\x24\x00\x00\x00WAVE",  # wav
        b"\x1aE\xdf\xa3" + b"\x00" * 8,  # webm/matroska
        b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 4,  # m4a (magic at offset 4)
    ):
        assert sniff_audio_payload(head), head[:8]


def test_sniff_rejects_html_interstitial_class():
    # the D2 class: Dropbox served login pages with HTTP 200; 52 were cached
    # as "downloaded" audio before the guard existed
    assert not sniff_audio_payload(b"<!DOCTYPE html><html><head>")
    assert not sniff_audio_payload(b"\n  <html lang='en'>")
    assert not sniff_audio_payload(b"ID3\x04\x00", content_type="text/html")


def test_sniff_unknown_container_passes():
    # only a positive HTML contradiction rejects; unknown magic is not proof
    assert sniff_audio_payload(b"\x00\x01\x02\x03unknowncontainer")


# ── set-mix link preference (characterization) ───────────────────────────────


def _link(platform: str) -> SetMediaLink:
    return SetMediaLink(set_id="s", platform=platform, url=f"https://x/{platform}")


def test_pick_mix_link_platform_preference():
    """CHARACTERIZATION of current behavior: YouTube outranks SoundCloud.

    NB the 2026-07-09 ingest-queue work argues SC-first (usually the DJ's own
    full-length upload; the queue scores SC over YT) — if the preference
    flips, update this test WITH the flip so the change is deliberate.
    """
    picked = _pick_mix_link((_link("soundcloud"), _link("youtube")))
    assert picked is not None and picked.platform == "youtube"
    only_sc = _pick_mix_link((_link("soundcloud"),))
    assert only_sc is not None and only_sc.platform == "soundcloud"
    assert _pick_mix_link(()) is None


# ── correction-ledger snapshot invariants ────────────────────────────────────


def _ledger_rows() -> list[dict]:
    with LEDGER.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_ledger_snapshot_loads_and_schema_holds():
    rows = _ledger_rows()
    assert len(rows) == 1267  # frozen snapshot 2026-07-09
    assert all(r["axis"] in ("version", "variant", "stem") for r in rows)
    assert all(r["action"] in ("replace", "add") for r in rows)


def test_ledger_snapshot_contains_the_paid_for_classes():
    """The named failure classes this suite exists to remember. If a class
    disappears from the snapshot, the fixture was regenerated incorrectly."""
    reasons = " || ".join(r["reason"] for r in _ledger_rows())
    for marker in (
        "wrong-version: mac_redownload rescue fetched original/edit",  # YTM hits[0]
        "wrong-identity: 2026-06-09",  # stem-candidate batch misfire
        "pure orphan (tid has no track_audio)",  # orphan class
        "hubert_vocal_gate",  # automated acappella gate
        "human_winner",  # labeling audition winners
        "audio-verified",  # discord corpus verification
    ):
        assert marker in reasons, f"ledger class missing: {marker}"


def test_ledger_axis_distribution_frozen():
    rows = _ledger_rows()
    by_axis = {
        a: sum(1 for r in rows if r["axis"] == a)
        for a in ("stem", "version", "variant")
    }
    assert by_axis == {"stem": 780, "version": 483, "variant": 4}

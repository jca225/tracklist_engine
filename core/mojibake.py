"""Detect and repair double-encoded (mojibake) UTF-8 strings.

The single, tested home for the `.encode('latin-1').decode('utf-8')` repair that
was previously scattered as an ad-hoc band-aid across the loop drivers and pull
tooling (issue #74).

**Root cause (issue #74):** pi-storage's non-interactive locale is not UTF-8
(`LANG=en_US` -> CPython `sys.getfilesystemencoding()='iso8859-1'`). When a pi
process moves a non-ASCII filename across the disk<->DB boundary under that
locale, it decodes the file's correct UTF-8 bytes (e.g. an en-dash `e2 80 93`)
as latin-1 -> the string `U+00E2 U+0080 U+0093` (``â€"``), and SQLite persists
*that* as UTF-8. The file on disk is correct; the stored `track_audio.path` is
mojibake. It resolves on pi by accident (iso8859-1 re-encodes the bad string
back to the original bytes) but a UTF-8 client (Mac rsync) asks for a path that
does not exist -> GT pulls fail.

`is_double_encoded_utf8` recognises exactly that class; `repair_double_encoded_utf8`
inverts it. Both are pure-stdlib and locale-independent (they operate on the
Python `str`, not the filesystem), so they give the same answer on pi and on a
UTF-8 client.
"""

from __future__ import annotations


def is_double_encoded_utf8(s: str) -> bool:
    """True iff ``s`` looks like UTF-8 bytes that were mis-decoded as latin-1
    (classic mojibake), i.e. re-encoding to latin-1 yields valid UTF-8 that
    differs from ``s``.

    A genuinely-accented name is *not* flagged: e.g. ``"café"`` -> latin-1
    ``b'caf\\xe9'`` is not valid UTF-8 (a lone 0xE9), so the round-trip raises
    and we return False. Only sequences that were actually UTF-8-then-latin1
    (0xC2/0xC3/0xE2… lead bytes) round-trip cleanly to a *different* string.

    A path that already holds a correctly-decoded non-Latin-1 character (e.g. a
    real ``U+2013`` en-dash) is likewise not flagged: ``str.encode('latin-1')``
    raises on any code point > U+00FF. This makes the check idempotent — a
    repaired path is never re-flagged.
    """
    try:
        repaired = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    return repaired != s


def repair_double_encoded_utf8(s: str) -> str:
    """Return the correct UTF-8 form of a mojibake string, or ``s`` unchanged
    if it is not double-encoded.

    Single-pass: inverts one latin-1 mis-decode, which covers the observed
    corpus class (issue #74 forensics: ``U+00E2 U+0080 U+0093`` -> ``U+2013``).
    Idempotent — repairing an already-correct string is a no-op.
    """
    if is_double_encoded_utf8(s):
        return s.encode("latin-1").decode("utf-8")
    return s

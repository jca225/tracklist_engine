"""Annotator bracket-tag handling (`[NNNbpm KK]`, `[no-features]`).

The annotator renames files/subdirs in `~/aligning/` to expose tempo + key
inline (see labeling/CLAUDE.md "Annotator rename convention"). These tags are
user territory — one-sided, Mac-only, never written back to pi-storage — so
codec-side path handling must be able to strip them.
"""

from __future__ import annotations

import re

_USER_TAG_PATTERN = re.compile(
    r"\[\s*\d+\s*bpm\b|\[no-features\]",
    re.IGNORECASE,
)
_BRACKET_TAG = re.compile(r"\s*\[[^\]]*\]\s*")


def strip_user_tags(name: str) -> str:
    return _BRACKET_TAG.sub("", name).strip()

"""Fuzzy artist+title matching between a tracklist and the SC like vocabulary.

The SoundCloud upload title is free-form ("Scars - Juice WRLD", "love nwantiti
(ah ah ah)", "Mike Posner & NIIKO X SWAE - Cooler Than Me"), and the uploader
handle is unreliable as the artist. So we tokenize the *whole* title into a bag
and require the tracklist title's core tokens to be contained in it, gated by an
artist-token guard to kill false positives on short common titles ("Middle",
"Closer").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Dropped from token bags: filler + version/stem qualifiers (removed on BOTH
# sides so "Emily (Remix)" matches "emily", "Congratulations Acapella" → core).
_STOP = {
    "the", "a", "an", "of", "and", "to", "in", "on", "for", "with", "feat",
    "ft", "featuring", "vs", "x", "dj", "official", "audio", "video", "lyrics",
    "lyric", "hd", "hq", "prod", "ep", "is", "it", "my", "me", "you", "your",
    "or", "but", "remix", "edit", "extended", "vip", "bootleg", "flip",
    "version", "mix", "mashup", "rework", "remaster", "remastered",
    "acapella", "acappella", "acappela", "acap", "instrumental", "intro", "outro",
}
_PARENS = re.compile(r"[\(\[\{].*?[\)\]\}]")
_FEAT = re.compile(r"\b(feat|ft|featuring|with|prod)\b.*$", re.I)


def tokens(s: str) -> list[str]:
    s = (s or "").lower().replace("&", " and ")
    s = _PARENS.sub(" ", s)
    s = _FEAT.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [t for t in s.split() if t and t not in _STOP and len(t) > 1]


def split_artist_title(label: str) -> tuple[list[str], list[str]]:
    """'Post Malone - Congratulations Acapella' → (artist_tok, title_tok)."""
    raw = label.split(" - ", 1)
    if len(raw) == 2:
        return tokens(raw[0]), tokens(raw[1])
    return [], tokens(label)


@dataclass
class Vocab:
    track_ids: list[int]
    bags: list[frozenset]          # title-token bag per row
    unames: list[frozenset]        # uploader-handle tokens per row
    likers: list[int]              # distinct-liker count per row (for ranking)
    index: dict[str, list[int]]    # token → row indices

    @classmethod
    def build(cls, rows: list[tuple]) -> "Vocab":
        # rows: (track_id, title, username, likers)
        track_ids, bags, unames, likers = [], [], [], []
        index: dict[str, list[int]] = {}
        for i, (tid, title, uname, lk) in enumerate(rows):
            bag = frozenset(tokens(title))
            track_ids.append(int(tid)); bags.append(bag)
            unames.append(frozenset(tokens(uname or ""))); likers.append(int(lk))
            for tok in bag:
                index.setdefault(tok, []).append(i)
        return cls(track_ids, bags, unames, likers, index)

    def match(self, artist_tok: list[str], title_tok: list[str]) -> list[int]:
        """Return matched row indices for one tracklist entry."""
        if not title_tok:
            return []
        postings = [self.index.get(t) for t in title_tok]
        if any(p is None for p in postings):
            return []
        cand = set(min(postings, key=len))
        for p in postings:
            cand &= set(p)
        a = set(artist_tok)
        out = []
        for idx in cand:
            artist_ok = (not a) or bool(a & (self.bags[idx] | self.unames[idx]))
            # short/common titles must clear the artist guard; long titles may pass alone
            if len(title_tok) < 2 and not artist_ok:
                continue
            if len(title_tok) < 3 and not artist_ok:
                continue
            out.append(idx)
        return out

"""Heuristic bot / low-quality listener scoring from warehouse signals."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from personalization.persistence import upsert_bot_scores
from personalization.records import ListenerBotScoreRow

DEFAULT_BOT_THRESHOLD = 0.55

_GENERIC_HANDLE = re.compile(r"^user[-_]?\d+$", re.I)
_GENERIC_USERNAME = re.compile(r"^user\s*\d+$", re.I)


@dataclass(frozen=True)
class BotAssessment:
    user_id: str
    sc_user_id: int | None
    bot_score: float
    is_bot: bool
    reasons: tuple[str, ...]


def _score_listener(
    *,
    handle: str,
    username: str | None,
    followers: int | None,
    followings: int | None,
    verified: bool,
    like_count: int,
    max_likes_one_day: int,
    playlist_count: int,
) -> BotAssessment:
    score = 0.0
    reasons: list[str] = []

    if _GENERIC_HANDLE.match(handle):
        score += 0.30
        reasons.append("generic_permalink")

    if username and _GENERIC_USERNAME.match(username.strip()):
        score += 0.20
        reasons.append("generic_username")

    fc = followers if followers is not None else 0
    fg = followings if followings is not None else 0
    if fc == 0 and fg == 0:
        score += 0.15
        reasons.append("zero_social_graph")

    if like_count >= 200 and max_likes_one_day / max(like_count, 1) >= 0.45:
        score += 0.25
        reasons.append("like_burst_single_day")

    if like_count >= 500 and playlist_count == 0 and fc <= 1:
        score += 0.15
        reasons.append("high_likes_no_playlists")

    if verified:
        score -= 0.35
        reasons.append("verified_discount")

    if fc >= 20:
        score -= 0.15
        reasons.append("followers_discount")

    if playlist_count >= 3 and like_count >= 20:
        score -= 0.10
        reasons.append("curator_discount")

    score = max(0.0, min(1.0, score))
    is_bot = score >= DEFAULT_BOT_THRESHOLD
    return BotAssessment(
        user_id="",
        sc_user_id=None,
        bot_score=score,
        is_bot=is_bot,
        reasons=tuple(reasons),
    )


def _max_likes_one_day(conn: sqlite3.Connection, user_id: str) -> int:
    rows = conn.execute(
        "SELECT substr(liked_at, 1, 10) AS d, COUNT(*) AS n FROM sc_likes WHERE user_id = ? GROUP BY d",
        (user_id,),
    ).fetchall()
    if not rows:
        return 0
    return max(int(r["n"]) for r in rows)


def score_mix_listeners(conn: sqlite3.Connection, mix_id: str) -> dict[str, int]:
    """Recompute bot scores for all listeners in a mix. Returns summary counts."""
    now = datetime.now(timezone.utc).isoformat()
    listeners = conn.execute(
        """
        SELECT user_id, sc_user_id, handle, username, followers_count,
               followings_count, verified
        FROM listeners WHERE mix_id = ?
        """,
        (mix_id,),
    ).fetchall()

    rows: list[ListenerBotScoreRow] = []
    for lst in listeners:
        uid = str(lst["user_id"])
        like_count = int(
            conn.execute("SELECT COUNT(*) FROM sc_likes WHERE user_id = ?", (uid,)).fetchone()[0]
        )
        playlist_count = int(
            conn.execute("SELECT COUNT(*) FROM sc_playlists WHERE user_id = ?", (uid,)).fetchone()[0]
        )
        assessment = _score_listener(
            handle=str(lst["handle"]),
            username=lst["username"],
            followers=lst["followers_count"],
            followings=lst["followings_count"],
            verified=bool(lst["verified"]),
            like_count=like_count,
            max_likes_one_day=_max_likes_one_day(conn, uid),
            playlist_count=playlist_count,
        )
        rows.append(
            ListenerBotScoreRow(
                user_id=uid,
                mix_id=mix_id,
                sc_user_id=lst["sc_user_id"],
                bot_score=assessment.bot_score,
                is_bot=assessment.is_bot,
                reasons_json=json.dumps(list(assessment.reasons)),
                computed_at=now,
            )
        )

    upsert_bot_scores(conn, tuple(rows))
    flagged = sum(1 for r in rows if r.is_bot)
    reason_counts = Counter()
    for r in rows:
        for reason in json.loads(r.reasons_json):
            reason_counts[reason] += 1
    return {
        "listeners_scored": len(rows),
        "bots_flagged": flagged,
        "bot_rate": round(flagged / max(len(rows), 1), 4),
        "top_reasons": dict(reason_counts.most_common(8)),
    }

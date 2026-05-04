from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from data_models import DJSet, DJSetCrawl, DJSetMediaLink, DJSetTrackMediaLink, ScrapeFailure, DJSetRow




class MusicDatabase:
    def __init__(self, db_path: str, schema_path: str) -> None:
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # WAL: readers don't block writers; busy_timeout: brief contention
        # waits instead of raising 'database is locked' immediately. Without
        # these the jobqueue's UPDATE scrape_failures path would 500 whenever
        # a sibling reader held the lock.
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()
        self._init_tables()

    def _init_tables(self) -> None:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at: {self.schema_path}")
        schema_sql = self.schema_path.read_text(encoding="utf-8")
        self.cursor.executescript(schema_sql)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def close_thread(self) -> None:
        # Compatibility with older threading patterns.
        pass

    def set_exists(self, set_id: str) -> bool:
        self.cursor.execute("SELECT 1 FROM dj_sets WHERE set_id = ?", (set_id,))
        return self.cursor.fetchone() is not None

    def insert_set(self, djset: DJSet) -> None:
        sql = """
        INSERT INTO dj_sets (
            set_id, set_url, title, date_played, artists,
            creator_name, creator_url,
            views, ided_tracks, total_tracks, likes,
            play_time, styles
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(set_id) DO UPDATE SET
            set_url=excluded.set_url,
            title=excluded.title,
            date_played=excluded.date_played,
            artists=excluded.artists,
            creator_name=excluded.creator_name,
            creator_url=excluded.creator_url,
            views=excluded.views,
            ided_tracks=excluded.ided_tracks,
            total_tracks=excluded.total_tracks,
            likes=excluded.likes,
            play_time=excluded.play_time,
            styles=excluded.styles
        """
        self.conn.execute(sql, (
            djset.set_id,
            djset.set_url,
            djset.title or "",
            djset.date_played or "",
            djset.artists,
            djset.creator_name,
            djset.creator_url,
            djset.views,
            djset.ided_tracks,
            djset.total_tracks,
            djset.likes,
            djset.play_time,
            djset.styles,
        ))
        self.conn.commit()

    def insert_crawl(self, crawl: DJSetCrawl) -> None:
        sql = """
        INSERT INTO dj_set_crawls (
            set_id, set_url,
            http_status, etag, last_modified,
            html_sha256, html_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.execute(sql, (
            crawl.set_id,
            crawl.set_url,
            crawl.http_status,
            crawl.etag,
            crawl.last_modified,
            crawl.html_sha256,
            crawl.html_path,
        ))
        self.conn.commit()

    def insert_set_media_links(self, links: Iterable[DJSetMediaLink]) -> None:
        rows = []
        for link in links:
            rows.append((
                link.set_id,
                link.platform,
                link.url,
                link.id_item,
                link.id_source,
            ))

        if not rows:
            return

        sql = """
        INSERT INTO dj_set_media_links (
            set_id, platform, url, id_item, id_source
        )
        VALUES (?, ?, ?, ?, ?)
        """
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def insert_track_media_links(self, links: Iterable[DJSetTrackMediaLink]) -> None:
        rows = []
        for link in links:
            rows.append((
                link.set_id,
                link.tlp_id,
                link.track_id,
                link.platform,
                link.player_id,
                link.id_object,
                link.id_item,
                link.id_source,
                link.view_source,
                link.view_item,
            ))

        if not rows:
            return

        sql = """
        INSERT INTO dj_set_track_media_links (
            set_id, tlp_id, track_id,
            platform, player_id,
            id_object, id_item, id_source, view_source, view_item
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def insert_failures(self, failures: Iterable[ScrapeFailure]) -> None:
        rows = []
        for failure in failures:
            rows.append((
                failure.set_id,
                failure.set_url,
                failure.stage,
                failure.track_title,
                failure.track_id,
                failure.tlp_id,
                failure.params_json,
                failure.error,
            ))

        if not rows:
            return

        sql = """
        INSERT INTO scrape_failures (
            set_id, set_url, stage,
            track_title, track_id, tlp_id, params_json,
            error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def get_ajax_failures(self, max_retries: int) -> list[dict]:
        sql = """
        SELECT failure_id, set_id, set_url, track_title, track_id, tlp_id,
               params_json, error, retries
        FROM scrape_failures
        WHERE stage = 'ajax' AND retries < ?
        ORDER BY set_id, failure_id
        """
        self.cursor.execute(sql, (max_retries,))
        cols = [d[0] for d in self.cursor.description]
        return [dict(zip(cols, row)) for row in self.cursor.fetchall()]

    def fetch_ajax_failures(self, max_retries: int, title_like: str | None) -> list[dict]:
        """AJAX failures with `retries < max_retries`, optionally scoped to a
        `dj_sets.title LIKE` pattern. `set_url` comes from the canonical
        `dj_sets` row (more reliable than `scrape_failures.set_url`).
        Mirrored by jobqueue.client.JobQueueClient.fetch_ajax_failures."""
        if title_like is None:
            sql = """
            SELECT f.failure_id, f.set_id, s.set_url, f.track_title, f.track_id,
                   f.tlp_id, f.params_json, f.error, f.retries
            FROM scrape_failures f
            JOIN dj_sets s USING(set_id)
            WHERE f.stage = 'ajax'
              AND f.retries < ?
            ORDER BY f.set_id, f.failure_id
            """
            params: tuple = (max_retries,)
        else:
            sql = """
            SELECT f.failure_id, f.set_id, s.set_url, f.track_title, f.track_id,
                   f.tlp_id, f.params_json, f.error, f.retries
            FROM scrape_failures f
            JOIN dj_sets s USING(set_id)
            WHERE s.title LIKE ?
              AND f.stage = 'ajax'
              AND f.retries < ?
            ORDER BY f.set_id, f.failure_id
            """
            params = (title_like, max_retries)
        self.cursor.execute(sql, params)
        cols = [d[0] for d in self.cursor.description]
        return [dict(zip(cols, row)) for row in self.cursor.fetchall()]

    def increment_failure_retries(self, failure_id: int) -> None:
        self.conn.execute(
            "UPDATE scrape_failures SET retries = retries + 1 WHERE failure_id = ?",
            (failure_id,),
        )
        self.conn.commit()

    def delete_failure(self, failure_id: int) -> None:
        self.conn.execute(
            "DELETE FROM scrape_failures WHERE failure_id = ?",
            (failure_id,),
        )
        self.conn.commit()

    def insert_rows(self, rows_iter: Iterable[DJSetRow]) -> None:
        rows = []
        for row in rows_iter:
            rows.append((
                row.set_id,
                row.row_index,
                row.element_id,
                row.classes,
                row.data_attrs_json,
                row.text_excerpt,
                row.raw_html,
            ))

        if not rows:
            return

        sql = """
        INSERT INTO dj_set_rows (
            set_id, row_index, element_id, classes,
            data_attrs_json, text_excerpt, raw_html
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.executemany(sql, rows)
        self.conn.commit()

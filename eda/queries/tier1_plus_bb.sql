-- Materialize the "tier 1 ∪ Big Bootie 10-15" set list as JSON job rows.
--
-- Tier 1 = high-quality, easily-alignable sets:
--   * 100% IDed tracks (ided_tracks = total_tracks)
--   * <=3 user suggestions (loose enough to admit BB-style mashup mixes)
--   * >=5,000 views (community engagement → confirmation pressure)
--   * >=88% of tracks have a media link (we can fetch audio)
--   * >=30 tracks (no DJ-promo shorts)
--   * play_time recorded
--
-- Then UNION the explicit Big Bootie 10-15 set_ids so they're guaranteed in
-- regardless of which side of any threshold they fall on.
--
-- Run from the Mac:
--   ssh pi-storage 'sqlite3 -json /mnt/storage/data/db/music_database.db' \
--     < data_analysis/queries/tier1_plus_bb.sql \
--     > data/djs/tier1_plus_bb.json
--
-- Output schema matches existing data/djs/*.json files (array of objects
-- with tracklist_id / title / url / ... fields), so it drops into the
-- scraper / downloader as a job file with no glue code.

WITH
  sug_per_set AS (
    SELECT set_id, COUNT(*) AS sug_count
    FROM track_suggestions
    GROUP BY set_id
  ),
  media_per_set AS (
    SELECT set_id, COUNT(DISTINCT track_id) AS tracks_with_media
    FROM dj_set_track_media_links
    WHERE track_id IS NOT NULL AND track_id != ''
    GROUP BY set_id
  ),
  tier1 AS (
    SELECT s.set_id
    FROM dj_sets s
    LEFT JOIN sug_per_set   g ON g.set_id = s.set_id
    LEFT JOIN media_per_set m ON m.set_id = s.set_id
    WHERE s.ided_tracks = s.total_tracks
      AND COALESCE(g.sug_count, 0) <= 3
      AND s.views >= 5000
      AND COALESCE(m.tracks_with_media, 0) * 1.0 / NULLIF(s.total_tracks, 0) >= 0.88
      AND s.total_tracks >= 30
      AND s.play_time IS NOT NULL AND s.play_time != ''
  ),
  big_bootie_10_15(set_id) AS (
    VALUES ('w1mgcjt'),    -- Vol 10
           ('2nvzlh2k'),   -- Episode 11
           ('1fsnxchk'),   -- Vol 12
           ('qj4v0wt'),    -- Vol 13
           ('1yl70ql1'),   -- Vol 14
           ('237tdqmk')    -- Vol 15
  ),
  selection AS (
    SELECT set_id FROM tier1
    UNION
    SELECT set_id FROM big_bootie_10_15
  )
SELECT
  s.set_id        AS tracklist_id,
  s.title         AS title,
  s.set_url       AS url,
  s.date_played   AS date,
  s.creator_name  AS creator_name,
  s.creator_url   AS creator_url,
  s.views         AS views,
  s.ided_tracks   AS ided_tracks,
  s.total_tracks  AS total_tracks,
  s.play_time     AS play_time,
  s.likes         AS likes,
  s.styles        AS styles
FROM dj_sets s
JOIN selection sel ON sel.set_id = s.set_id
ORDER BY
  CASE WHEN s.set_id IN (SELECT set_id FROM big_bootie_10_15) THEN 0 ELSE 1 END,
  s.date_played DESC;

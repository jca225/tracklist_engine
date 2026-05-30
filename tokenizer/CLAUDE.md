# tokenizer/ — scrape-row → track_metadata materialization

`track_tokenizer.py` turns scraped `dj_set_rows` (raw 1001tracklists HTML
attributes) into structured `track_metadata` rows. It sits between the scraper
([web_crawler/CLAUDE.md](../web_crawler/CLAUDE.md)) and `ingest/`: the
`track_id` it mints is the key the whole downstream chain joins on.

This is the **authoring home** for the two scrape-stage design rules below.
They are *consumed* in `ingest/` and `labeling/` but documented here once, next
to the code that implements them — don't duplicate them downstream, just point
back here.

## Remix and version-qualifier handling (design rule, not a bug)

A 1001tracklists track row like
`Martin Garrix & Troye Sivan - There For You (Madison Mars Remix) (Instrumental) EPIC AMSTERDAM/STMPD`
stores in `track_metadata` as:

```
title:        "There For You"
full_name:    "Martin Garrix & Troye Sivan - There For You (Madison Mars Remix)"
version_tag:  "Remix"          # scrape-time Title Case → DB version "remix"
claimed_stem: "instrumental"    # from (Instrumental) in row text / full_name
```

Three axes are parsed in [identity_axes.py](identity_axes.py) and materialized
to lowercase DB columns (see root CLAUDE.md "Track identity"):

- **`full_name`** — from `<meta itemprop="name">`; **keeps remixer qualifier**
  `(Madison Mars Remix)`; strips `(Acappella)` / `(Instrumental)` parentheticals
  for search (vocal qualifiers are sparse/noisy on YT Music).
- **`version_tag`** on `TrackRow` — version axis only: `Rework | Remix |
  AltVersion | None`. **Never** `Acappella` (that was a pre-2026-05 conflation).
- **`claimed_stem`** — stem axis: `regular | acappella | instrumental` from
  row text + `full_name` via `derive_claimed_stem()`.
- **`claimed_variant`** — `regular | extended` from "Extended Mix" patterns.

`materialize.py` writes `track_metadata.version`, `set_track_slots.claimed_*`,
and syncs `recording` / `recording_id`.

**Search / download:** [ingest/search_query.py](../ingest/search_query.py) and
[redownload_via_ytmusic.py](../scripts/redownload_via_ytmusic.py) use `full_name`
so remixer qualifiers resolve the right release; ID Remix/Bootleg placeholders
strip back to bare `"Artist - Title"`.

**Aligner hint:** instrumental and acappella slots are visible on
`set_track_slots.claimed_stem` (and in pull `manifest.json` as `stem` / `axes_key`)
even though `full_name` drops the parenthetical — the aligner should prefer
Demucs stems over expecting a second download (baby rule in root CLAUDE.md).

## Known scraper gap: sided rows with no `data-trackid` (the "Rvmor gap")

Some 1001tracklists `w/` rows (`data-isided="true"`) have a unique per-set
SoundCloud annotation but no global `data-trackid` HTML attribute — typically
obscure fan remixes that lack a 1001tracklists global track entry. The scraper
still extracts the link into `dj_set_track_media_links` (with `track_id=NULL`),
but the rest of the chain drops it:

1. **Tokenizer** ([_extract_track_key, track_tokenizer.py:153-155](track_tokenizer.py#L153-L155)) reads `data-trackid` to mint `track_key` (→ `track_id`). Missing attribute → no `track_metadata` row.
2. **Ingest** keys on `track_id`. Missing → never downloaded into `track_audio`.
3. **Alignment pull** ([labeling/pull_set_for_alignment.py:211-214](../labeling/pull_set_for_alignment.py#L211-L214)) filters `dj_set_rows` by `data_attrs_json LIKE '%trackid%'`. Missing → row skipped, slot invisible in manifest.

Field example: BB12 row 150 (`Porter Robinson & Madeon - Shelter (Rvmor Remix)`,
tlp_id 2853054), SoundCloud-only (player_id 833168986). Currently handled by
manually dropping the audio into `~/aligning/.../tracks/{slot}w{K}__...m4a`
outside the canonical pipeline.

**Implemented:** synthetic `track_id` = `tlp{tlp_id}` when sided rows have media
but no `data-trackid`; `set_track_slots.source='synthetic'`. Pull and ingest
accept the synthetic id in SQL (no longer filter on `data-trackid` only).

See the `project_tlp_gap` memory and the field-evidence list in
`project_official_stems_search` for tracking instances.

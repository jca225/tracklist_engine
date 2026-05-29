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
version_tag:  "Remix"
```

Two fields, two different behaviors:

- **`full_name`** comes from 1001tracklists' `<meta itemprop="name">`
  ([track_tokenizer.py:263](track_tokenizer.py#L263)). It **keeps the remixer
  qualifier** `(Madison Mars Remix)`, but [_VOCAL_QUALIFIER_RE](track_tokenizer.py#L92-L95)
  strips any `(Acappella)` / `(Instrumental)` / `(Inst.)` / `(Instr.)`
  parenthetical ([applied at line 265](track_tokenizer.py#L265)). The label tag
  (`EPIC AMSTERDAM/STMPD`) is never in the meta name to begin with.
- **`version_tag`** is a coarse enum `Acappella | Rework | Remix | AltVersion | None`,
  derived by a row-text scan in [_derive_version_flags](track_tokenizer.py#L182-L210).

**Acapella vs instrumental are NOT symmetric** (this surprised us, so it's
spelled out):

- **Acapella IS captured** — the row-text scan sets `version_tag = "Acappella"`
  ([line 202](track_tokenizer.py#L202)). The literal `(Acappella)` is stripped
  from `full_name`, but the signal survives in `version_tag`.
- **Instrumental is NOT captured anywhere** — there is intentionally no
  `Instrumental` value in the enum and no `has_instrumental` branch, so
  `version_tag` stays `None`; and `(Instrumental)` is *also* stripped from
  `full_name`. An instrumental slot is therefore **indistinguishable from a
  plain full-track slot** in the structured record.

Why the asymmetry is (currently) intended, on two axes:

1. **Remixer qualifier IS preserved in search**: [redownload_via_ytmusic.py:113](../scripts/redownload_via_ytmusic.py#L113)
   sends `full_name` verbatim to YT Music, so the search hits the *Madison Mars
   Remix* release rather than the original Martin Garrix track. A bare
   `"Artist - Title"` search would silently resolve to the original — root cause
   of the corpus's variant-bleed bug, now fixed.
2. **Vocal/instrumental qualifier is deliberately NOT preserved in `full_name`**:
   YT Music's `filter='songs'` index doesn't reliably carry `(Instrumental)`
   variants as separate releases, and isolated-vocal/instrumental uploads are
   sparse and noisy. The system resolves to the canonical (vocal) master and
   lets Demucs extract stems downstream — `version_tag` (e.g. `Acappella`) tells
   the alignment-side code which stem to use without needing a separately
   downloaded instrumental.

Carve-out for unknowns: when `full_name` contains 1001tracklists' "(ID Remix)" /
"(ID Bootleg)" placeholders (remixer unknown), the script strips back to
`"Artist - Title"` since a literal "ID" in the YT Music query corrupts results
([redownload_via_ytmusic.py:70-76](../scripts/redownload_via_ytmusic.py#L70-L76)).

> **OPEN QUESTION (deferred 2026-05-29):** is dropping the instrumental
> qualifier entirely the right call, or a gap? Cost: when a slot was played as
> the *instrumental* cut, the aligner loses the hint "expect this track's
> vocals to be **absent** in the mix here" and must infer it from audio. Acapella
> gets a first-class `version_tag`; instrumental gets nothing. Decide intended
> vs. gap-to-fix before building the aligner; if a fix, it likely means adding
> an `Instrumental` enum value (or a separate boolean) fed from the same
> `_VOCAL_QUALIFIER_RE` match that currently only strips.

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

Proper fix would mint a synthetic `track_id` (e.g. `tlp{tlp_id}`) and backfill
`track_metadata`, `dj_set_track_media_links.track_id`, AND
`dj_set_rows.data_attrs_json` (because of the pull-script filter above). Logical
home for the synthesis is here in the tokenizer when it sees an isolated
`tlp_id` with media links but no `data-trackid` — emit a stable synthetic key
tied to the tlp_id, mark the source as `synthetic` in `track_metadata`.

See the `project_tlp_gap` memory and the field-evidence list in
`project_official_stems_search` for tracking instances.

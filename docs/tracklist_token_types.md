# 1001tracklists Token Types — Exhaustive Catalog

**Purpose.** The tokenizer's job is **schema archaeology**: recover 1001tracklists'
internal DJ-set data model from the markup they leak. This is the complete
enumeration of every row type, `data-*` attribute, CSS class, `itemprop`,
`onclick` handler, and icon that appears in a tracklist DOM — reconciled against
what we currently parse, persist, and feed downstream, so we can see exactly
what we drop. Companion to the [[project_tokenizer_schema_archaeology]] framing.

**Empirical basis (2026-07-23).** Mined all 41,492 saved snapshots in
`data/html/` (1.24M track rows) for row-level `data-*`, and an exhaustive
all-element sweep of a converged 2,000-file sample (every attribute/class/
onclick/itemprop/icon on every descendant of `#tlTab`). Name-level vocabularies
converged: **23 `data-*` keys, 10 itemprops, 18 suggestion `data-type`s, 37
semantic icons, 32 onclick handlers, 17 row skeletons.** Miners:
`scratchpad/mine_tokens.py`, `scratchpad/mine_all_signals.py`.

**Coverage legend:**
- ✅ **kept** — parsed AND persisted to a DB column a consumer reads
- ⚠️ **lossy** — row parsed, but this field's payload is dropped at materialize
- 🅿️ **parsable-dropped** — a parser exists but `materialize.py` never calls it / never stores it
- ❌ **unparsed** — no code extracts it
- 🔒 **unwired** — `id_tokenizer.py` parses it but is not called by `materialize.py`

Parsers: `tokenizer/{track_tokenizer,suggestion_tokenizer,text_tokenizer,id_tokenizer,tokenizer}.py`.
Persistence: `tokenizer/materialize.py` → `track_metadata` / `set_track_slots` /
`track_suggestions`. Dispatch is by the outer `#tlTab` child's class.

---

## 1. Row types (top-level `#tlTab` child skeleton)

| Skeleton (normalized) | Row type | Dispatch | Coverage |
|---|---|---|---|
| `bItm tlpItem tlpTog trRowN` | track (solo) | `materialize.py:292` | ✅ |
| `bItm con tlpItem tlpTog` | track, concurrent (`w/` overlay) | ✅ | is_concurrent ✅ |
| `bItm subPosTog tlpItem tlpTog` | track, has sub-positions (mashup parent) | — | ❌ `data-subpos` (WS-A) |
| `bItm con subPosN tgHid tlpItem tlpSubTog` | mashup **constituent** (collapsed) | processed as a slot | ❌ `data-mashpos` (WS-A) |
| `bItm deleted tlpItem tlpTog` | **deleted track** | none | ❌ |
| `bItm con ntB sugTog tlp_N` | **suggestion / correction** | `materialize.py:361` | ⚠️ identity only |
| `bItmH` / `bItmH flex` / `bItmH interaction noUser` | **notice / section header** | `materialize.py:385` counts only | 🅿️ **dropped** |
| `mt10` / `<none>` | layout spacers | — | n/a |

---

## 2. Track-row signals

### 2a. `data-*` on the track `<div>`
| Attribute | Meaning | Field | Coverage |
|---|---|---|---|
| `data-trackid` | global track id | `track_key` → `track_id` | ✅ |
| `data-id` | per-set tlp id | `tlp_id` | ✅ |
| `data-trno` | track ordinal | `data_trno` | ✅ |
| `data-isided` | is an ID / unknown track | `is_ided` | ✅ |
| **`data-isid`** | ID-row flag (distinct from `isided`) | — | 🔒 (id_tokenizer) |
| **`data-protected`** | broadcast-protected track | — | 🔒 |
| **`data-rbcst`** | re-broadcast flag | — | 🔒 |
| **`data-mashup`** = N | mashup member count (parent) | — | ❌ (WS-A) |
| **`data-mashpos`** | this row is a mashup constituent | — | ❌ (WS-A) |
| **`data-subpos`** | row has collapsible sub-positions | — | ❌ (WS-A) |
| `data-remix` = 1 | remix flag (on `.mediaRow`) | `version='remix'` | ✅ |
| `data-mode` = hours | cue display format (cosmetic) | — | ❌ (skip) |

### 2b. `itemprop` (schema.org MusicRecording, per track)
| itemprop | Field | Coverage |
|---|---|---|
| `name` | `full_name` (keeps remixer qualifier) | ✅ |
| `genre` | `genre` | ✅ |
| `duration` (ISO-8601) | `duration_seconds` | ✅ |
| `url` | `track_page_href` | ✅ |
| **`byArtist`** | structured artist entity | ❌ (artists derived from trackValue text-split instead) |
| `publisher` | label | ✅ via `.trackLabel` |

### 2c. cue / timing
| Signal | Field | Coverage |
|---|---|---|
| `input#..._cue_seconds` (hidden) | `cue_seconds` | ✅ |
| `div#cue_...` `mm:ss`/`h:mm:ss` | `cue_time_seconds` | ✅ |

### 2d. classes carrying track semantics
| Class | Meaning | Coverage |
|---|---|---|
| `con` | concurrent / played-with | `is_concurrent` ✅ |
| `mashupTrack` | styled as a mashup track | ❌ (redundant with data-mashup) |
| `bootleg` (on content div, `title="mashup track"`) | mashup/bootleg styling | ❌ |
| **`remixValue`** | **remixer-name span (structured remixer)** | ❌ → `version_artist` never written |
| `trackLabel` | label link(s) | ✅ `publisher_labels` |
| `badgeSpotify` | Spotify save/pre-save CTA | ⚠️ (`spotify_cta_*` parsed, not persisted) |
| `wRow` | IDers + plays row | ⚠️ (parsed, not persisted) |
| `trackStatus` / `trackNote` | status / note text | ❌ |

---

## 3. Mashup / overlay structure → **WS-A plan**

`data-mashup="N"` (parent) + N following `data-mashpos` rows (`con tgHid`, often
no `data-trackid`) + `data-subpos` (parent has children). Explicit, deterministic
grouping we currently reconstruct from string-regex in
`core/slot_inventory.py:derive_layer_role`. Plan:
`docs/superpowers/plans/2026-07-23-mashup-structure-tokens.md`.

---

## 4. ID / unknown-track workflow → **id_tokenizer is 🔒 unwired**

`data-isid`, `data-protected`, `data-rbcst`, plus tooltips `IDer of this track`,
`additional IDer(s)`, `Number of users watching this ID`, `linked IDs`,
`play audio link for id`. `id_tokenizer.py` parses linked-ID hints, watchers,
pre-save counts — but `materialize.py:288-290` skips it ("track_id_links
populated by a focused later pass — to be written"). **Nothing persisted.**

---

## 5. Suggestion / correction rows (`sugTog`) → **WS-C plan (proposed)**

Community corrections, keyed by `data-type`. **Parsed** by `suggestion_tokenizer.py`,
but `materialize.py:363-382` stores only identity into `track_suggestions` — **no
`suggestion_type`, no corrected cue, no `[poll: correct/incorrect/unsure]` votes.**
⚠️ across the board.

| `data-type` | Meaning | Axis |
|---|---|---|
| **5** | **"track wasn't played"** | existence (drop slot) |
| **14** | **"correct cue time is H:MM:SS"** | placement |
| 1 | `[wrong track]` | identity |
| 17 | `track is rework of X` (source named) | identity |
| 6 | `track misspelled` | identity |
| 12 | `correct label is [...]` | metadata |
| 8 | `tracknumber should be N` | order |
| 2 / 3 | `[track before]` / `[track after]` (+ cue) | order/placement |
| 4 / 9 / 10 / 15 | mashup/overlay: `played together w/`, `part of mashup`, `[mashup part]` | structure |
| 0 / 7 | plain suggested track | identity |
| 13 | `correct headline is X` (fixes a section header) | metadata |
| 255 | `[other correction]` | — |
| None | poll meta ("N users marked this incorrect") | confidence |

Suggestion-row `data-*`: `data-type`, `data-tlp` (target), `data-pos`,
`data-user`/`data-guest` (suggester), `data-nospam`, `data-noguestsug`,
`data-videos`, `data-track`, `data-value` — all ⚠️ dropped.

**Calibration:** suggestions are *unresolved proposals*, not truth — target
candidate-corrections + confidence (poll), not facts to apply blindly
(Fellegi-Sunter accept/review/abstain).

---

## 6. Notice / section rows (`bItmH`) → **materialize drops all (🅿️)**

`text_tokenizer.py` classifies these by leading `fa-` icon, but
`materialize.py:385-386` only counts them. **Nothing persisted.**

| Icon | Row type |
|---|---|
| `<no-icon>` (`.breakAll`) | section / editorial-segment header ("Tune Of The Week:", "Aoki Pick:", …) |
| `fa-video-camera` | "add a (live) video…" CTA |
| `fa-trash` | `<artist> played:` / removal |
| `fa-info-circle` | availability notice ("no full recording available…") |
| `fa-recycle` | "contains identical tracklist(s)" |
| `fa-star` | "identical tracklist start …" |
| `fa-exclamation-triangle` | "title contains wrong information" warning |

---

## 7. Set-level itemprops (the tracklist itself)

`author` (the DJ), `datePublished`, `numTracks` — schema.org on the set
container. ❌ not captured by the tokenizer (set metadata lives in `dj_sets`
from the scraper instead).

---

## 8. Behavioral layer (`onclick` handlers) — reference only

Encode UI/AJAX, not persistable identity: `MediaViewer`, `playPosition`,
`toggleCue`, `rowToggle`, `MediaSubmitter`, `UserSuggest`, `EditMenu`,
`Spotify.save`, `copyPaste`, `SecureCheck`. Useful as a map of what actions the
model exposes; nothing to persist. Not a coverage gap.

---

## Coverage summary — what we drop

| Category | Status | Fix |
|---|---|---|
| Track identity core (id/cue/version/stem/variant/genre/duration/label) | ✅ | — |
| Mashup structure (`data-mashup/mashpos/subpos`) | ❌ | WS-A (plan written) |
| Suggestion/correction semantics (type 5/14/1/17 + poll) | ⚠️ dropped | WS-C (proposed) |
| Notice/section rows (`bItmH`) | 🅿️ dropped | WS-C (call text_tokenizer + `set_notices` table) |
| ID workflow (`data-isid/protected/rbcst`, linked IDs, watchers) | 🔒 unwired | wire `id_tokenizer` + `track_id_links` |
| Remixer entity (`remixValue` → `version_artist`) | ❌ | WS-B (identity edges) |
| `byArtist` structured artist | ❌ | WS-B |
| `rework of track X` source edge | ❌ (boolean only) | WS-B |
| `feat.`/`featuring` | ❌ (only `" ft. "`) | separate |
| `radio` variant, `deleted` rows | ❌ | separate |

**Downstream truth:** the aligner's live contract reads only 5 `set_track_slots`
fields (`alignment/infer.py:fetch_slot_rows`), so every gap
above needs threading through `materialize → set_track_slots/schema → manifest →
infer` before it reaches the model. See WS-B in the mashup plan's Deferred
section.

## Regenerate

```bash
venvs/audio/bin/python scratchpad/mine_tokens.py 100000        # row-level data-* (full corpus)
venvs/audio/bin/python scratchpad/mine_all_signals.py 100000   # exhaustive all-element sweep
```

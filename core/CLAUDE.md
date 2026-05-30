# core/ — shared substrate

**The substrate rule: `core/` imports nothing upward.** It depends on no chain
module (scrape / ingest / analysis / labeling / alignment) and no analysis- or
scrape-domain types. Every other stage may import `core`; `core` imports only
the stdlib and itself. If you find yourself wanting to import a stage type into
`core`, the code belongs in that stage instead (see the
`persistence.py` vs `core/db.py` boundary in
[analysis/CLAUDE.md](../analysis/CLAUDE.md)).

Modules:

- **`identity.py`** — three-axis vocabulary (`version`, `stem`, `variant`),
  `RecordingAxes`, `normalize_*`, `parse_axes_key`, concatenated key
  `version__stem__variant`. Single source of truth for DB lowercase values;
  tokenizer maps scrape Title Case → these via `scrape_version_to_db` /
  `identity_axes.py`. Legacy aliases: `full` / `original` on the stem axis →
  `regular`.
- **`result.py`** — the `Result[T, E]` / `Ok` / `Err` monad. Errors-as-values
  for library/core code (CLI scripts fail-fast with `sys.exit` instead).
- **`models.py`** — frozen dataclasses shared across stages (`Track`,
  `AudioAsset` with `stem` + `variant`, `MediaSource`, `SetAudioAsset`, …) plus
  URL normalizers. `AudioAsset.track_id` is the `recording_id` alias.
- **`db.py`** — generic SQLite adapter: converts sqlite exceptions into
  `DbError` Results, knows only `core` types. `insert_audio_or_reap()` unlinks
  the file on insert failure; `_ensure_recording()` upserts `work`+`recording`
  before `track_audio` insert. Domain code never imports `sqlite3` directly.
  Analysis-domain writers live in `analysis/persistence.py`, not here.
- **`errors.py`** — shared error types (`DbError`, …).

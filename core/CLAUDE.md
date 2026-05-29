# core/ — shared substrate

**The substrate rule: `core/` imports nothing upward.** It depends on no chain
module (scrape / ingest / analysis / labeling / alignment) and no analysis- or
scrape-domain types. Every other stage may import `core`; `core` imports only
the stdlib and itself. If you find yourself wanting to import a stage type into
`core`, the code belongs in that stage instead (see the
`persistence.py` vs `core/db.py` boundary in
[analysis/CLAUDE.md](../analysis/CLAUDE.md)).

Modules:

- **`result.py`** — the `Result[T, E]` / `Ok` / `Err` monad. Errors-as-values
  for library/core code (CLI scripts fail-fast with `sys.exit` instead).
- **`models.py`** — frozen dataclasses shared across stages (`Track`,
  `AudioAsset`, `MediaSource`, `SetAudioAsset`, …) plus URL normalizers.
- **`db.py`** — generic SQLite adapter: converts sqlite exceptions into
  `DbError` Results, knows only `core` types. Domain code never imports
  `sqlite3` directly. Analysis-domain writers live in `analysis/persistence.py`,
  not here.
- **`errors.py`** — shared error types (`DbError`, …).

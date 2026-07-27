# Gold fixtures (BB11 / BB12)

Small checked-in extracts for dry-run CI. **No audio bytes.**

| File | Role |
|------|------|
| `inventory.json` | Combined inventory for both gold sets |
| `2nvzlh2k_*` | BB11 (Two Friends – Big Bootie Mix Episode 11) |
| `1fsnxchk_*` | BB12 (Volume 12) |

## Live inventory

```bash
export LEGACY_DB_PATH=/path/to/music_database.db
cargo run -p dj_migrate -- inventory --out staging/inventory.json
```

Default set filter is BB11+BB12. Paths under `/mnt/storage/objects/...` are
expected to be unreachable on a laptop until transfer — `verify-manifest`
warns; use `--require-reachable` only when the object store is mounted.

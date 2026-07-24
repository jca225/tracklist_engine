"""Run a SQL query against pi-storage's canonical sqlite DB over SSH.

Canonical primitive — was duplicated as ``pull_set_for_alignment.ssh_sqlite``
(JSON-mode, hardcoded host/db) and ``enrich_gt_track_ids._ssh_sql``
(pipe-separated text mode, host/db kwargs). Merged here on the JSON-mode body
(native types, robust to delimiter characters in TEXT columns) with the
kwarg-parameterized host/db from the enrich variant, plus its ConnectTimeout
safety net.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"


def ssh_sqlite(
    query: str, *, host: str = PI_HOST, db: str = PI_DB
) -> list[dict[str, Any]]:
    """Run a sqlite3 query on `host` and return parsed JSON rows.
    `.mode json` emits a JSON array; '' for no results."""
    script = f".mode json\n{query.strip()}\n"
    cmd = ["ssh", "-o", "ConnectTimeout=15", host, f"sqlite3 {db}"]
    out = subprocess.run(
        cmd,
        input=script,
        capture_output=True,
        text=True,
        check=True,
    )
    body = out.stdout.strip()
    if not body:
        return []
    rows: list[dict[str, Any]] = json.loads(body)
    return rows

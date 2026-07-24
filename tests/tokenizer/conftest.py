import gzip
import json
import sqlite3
from pathlib import Path
import pytest

_FIXTURE = Path("tests/tokenizer/fixtures/dj_set_rows_sample.json.gz")


def build_fixture_db(tmp_path: Path) -> Path:
    """Temp DB seeded from the checked-in dj_set_rows sample. Hermetic."""
    if not _FIXTURE.exists():
        pytest.skip(f"{_FIXTURE} missing — run Task 1 Step 0 to generate it")
    rows = json.loads(gzip.decompress(_FIXTURE.read_bytes()))
    if not rows:
        pytest.skip("empty fixture sample")
    dst = tmp_path / "fixture.db"
    conn = sqlite3.connect(dst)
    conn.executescript(
        "CREATE TABLE dj_set_rows (row_id INTEGER PRIMARY KEY, set_id TEXT, "
        "row_index INTEGER, raw_html TEXT);"
    )
    conn.executemany(
        "INSERT INTO dj_set_rows (row_id,set_id,row_index,raw_html) VALUES (?,?,?,?)",
        [(r["row_id"], r["set_id"], r["row_index"], r["raw_html"]) for r in rows],
    )
    conn.commit()
    conn.close()
    return dst


def snapshot(db_path: Path) -> dict:
    """Order-stable dump of the two output tables materialize populates."""
    conn = sqlite3.connect(db_path)
    try:
        out = {}
        for tbl, order in (
            ("set_track_slots", "set_id,row_index"),
            ("track_suggestions", "sug_id"),
        ):
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info({tbl})")]
            data_cols = [
                c for c in cols if c != "parsed_at"
            ]  # drop nondeterministic timestamp
            sel = ",".join(data_cols)
            out[tbl] = conn.execute(
                f"SELECT {sel} FROM {tbl} ORDER BY {order}"
            ).fetchall()
        return out
    finally:
        conn.close()


@pytest.fixture
def fixture_db(tmp_path):
    return build_fixture_db(tmp_path)

#!/usr/bin/env python3
"""Append redownload failures from pi/mac logs into logs/redownload_failures_bb10_15.tsv.

Usage:
  venvs/audio/bin/python scripts/collect_redownload_failures.py
  venvs/audio/bin/python scripts/collect_redownload_failures.py --pi-log /tmp/redownload_bb10_15.log
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "logs/redownload_failures_bb10_15.tsv"
MAC_LOG = REPO / "logs/mac_redownload_bb_remix.log"
PI_HOST = "pi-storage"
PI_LOG_DEFAULT = "/tmp/redownload_bb10_15.log"

FAIL_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} )?(?:WARNING|ERROR) .*?"
    r"(?:FAIL\s+(\S+)\s+\[([^\]]+)\]:\s*(.+)|DB\s+(\S+)\s+insert:\s*(.+))$"
)
MAC_TRACK_RE = re.compile(
    r"\[(\d+)/\d+\] (\S+) taid=(\d+) v=(\S+) q=(.+)$"
)


def _existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            keys.add((parts[1], parts[2], parts[6] if len(parts) > 6 else ""))
    return keys


def _append_rows(rows: list[str]) -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.write_text(
            "run\tsource\ttrack_id\ttrack_audio_id\tversion\tquery\tfailure_kind\tdetail\n"
        )
    existing = _existing_keys(OUT)
    added = 0
    with OUT.open("a") as fh:
        for row in rows:
            parts = row.split("\t")
            key = (parts[1], parts[2], parts[6])
            if key in existing:
                continue
            fh.write(row + "\n")
            existing.add(key)
            added += 1
    return added


def _parse_pi_log(text: str, run: str, source: str) -> list[str]:
    rows: list[str] = []
    for line in text.splitlines():
        m = FAIL_RE.search(line.strip())
        if not m:
            continue
        if m.group(1):
            track_id, kind, detail = m.group(1), m.group(2), m.group(3)
        else:
            track_id, kind, detail = m.group(4), "insert", m.group(5)
        rows.append(
            f"{run}\t{source}\t{track_id}\t\t\t\t{kind}\t{detail.strip()}"
        )
    return rows


def _parse_mac_log(text: str, run: str, source: str) -> list[str]:
    rows: list[str] = []
    last: dict[str, str] | None = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = MAC_TRACK_RE.search(line)
        if m:
            last = {
                "track_id": m.group(2),
                "taid": m.group(3),
                "version": m.group(4),
                "query": m.group(5).strip("'").strip('"'),
            }
        if "ERROR replace failed" not in line or last is None:
            continue
        detail = "replace failed"
        for extra in lines[i : i + 8]:
            if "OperationalError" in extra:
                detail = extra.split(":", 1)[-1].strip()
                break
        rows.append(
            f"{run}\t{source}\t{last['track_id']}\t{last['taid']}\t"
            f"{last['version']}\t{last['query']}\tdb_locked\t{detail}"
        )
    return rows


def _fetch_pi_log(remote: str) -> str:
    proc = subprocess.run(
        ["ssh", PI_HOST, f"cat {remote} 2>/dev/null || true"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-log", default=PI_LOG_DEFAULT)
    ap.add_argument("--skip-pi", action="store_true")
    ap.add_argument("--skip-mac", action="store_true")
    args = ap.parse_args(argv)

    run = date.today().isoformat()
    rows: list[str] = []
    if not args.skip_mac and MAC_LOG.exists():
        rows.extend(_parse_mac_log(MAC_LOG.read_text(), run, "mac_redownload_bb_remix"))
    if not args.skip_pi:
        pi_text = _fetch_pi_log(args.pi_log)
        if pi_text.strip():
            rows.extend(_parse_pi_log(pi_text, run, "pi_redownload_via_ytmusic"))

    added = _append_rows(rows)
    print(f"{OUT}: appended {added} new row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

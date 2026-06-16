#!/usr/bin/env python3
"""Merge validated roster candidates into personalization/config/mixes.yaml.

Appends new mix entries as a commented block (preserving the existing hand-written
scene annotations — a yaml.safe_dump round-trip would destroy them). Dedups against
existing entries by BOTH the set_id key AND the resolved SoundCloud permalink, so a
set already configured under a hand-slug (e.g. BB11/BB12) is never double-scraped.

  venvs/audio/bin/python scripts/merge_taste_roster.py            # dry-run (default)
  venvs/audio/bin/python scripts/merge_taste_roster.py --apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MIXES = ROOT / "personalization/config/mixes.yaml"
CANDS = ROOT / "data/taste/roster_candidates.yaml"


def norm(url: str) -> str:
    return (url or "").rstrip("/").split("?")[0].lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    existing = yaml.safe_load(MIXES.read_text())["mixes"]
    existing_keys = set(existing)
    existing_urls = {norm(v.get("soundcloud_url", "")) for v in existing.values()}

    cands = yaml.safe_load(CANDS.read_text())["mixes"]

    new, skipped = [], []
    for set_id, v in cands.items():
        if set_id in existing_keys or norm(v["soundcloud_url"]) in existing_urls:
            skipped.append((set_id, v["_artist"]))
            continue
        new.append((set_id, v))

    # group new entries by artist for a readable appended block
    new.sort(key=lambda kv: (kv[1]["_artist"], -kv[1]["_sc_likes"]))
    lines = ["", f"  # --- roster scrape-up ({len(new)} sets, build_taste_roster.py) ---"]
    cur = None
    for set_id, v in new:
        if v["_artist"] != cur:
            cur = v["_artist"]
            lines.append(f"  # {cur}")
        title = v["title"].replace('"', "'")
        lines.append(f'  {set_id}:')
        lines.append(f'    set_id: "{set_id}"')
        lines.append(f'    title: "{title}"')
        lines.append(f'    soundcloud_url: "{v["soundcloud_url"]}"   # {v["_sc_likes"]} SC likes')
    block = "\n".join(lines) + "\n"

    print(f"new: {len(new)}   skipped (already configured): {len(skipped)}")
    for sid, a in skipped:
        print(f"  skip {sid} ({a})")
    if args.apply:
        with MIXES.open("a") as f:
            f.write(block)
        print(f"\nappended {len(new)} entries to {MIXES.relative_to(ROOT)}")
    else:
        print("\n--- block to append (dry-run; pass --apply to write) ---")
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

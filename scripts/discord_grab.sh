#!/usr/bin/env bash
# Incremental Discord grab loop (macOS).
#
#   1. Scroll to the top of a channel in Discord.
#   2. Select a screenful of messages, Cmd+C.
#   3. Run:  scripts/discord_grab.sh <label>
#      -> appends the clipboard to data/discord_paste/<label>.txt, then downloads
#         every NEW link (manifest dedups, so re-runs skip what's already pulled).
#   4. Scroll down, select the next chunk, repeat until you hit the bottom.
#
# <label> is one of: instrumentals | acappellas | stem_packs (or any name).
set -euo pipefail
LABEL="${1:?usage: discord_grab.sh <label>   (instrumentals|acappellas|stem_packs)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="$ROOT/data/discord_paste/${LABEL}.txt"
mkdir -p "$(dirname "$FILE")"

CLIP="$(pbpaste)"
if [ -z "$CLIP" ]; then
  echo "clipboard is empty — copy some channel text first" >&2
  exit 1
fi
printf '\n%s\n' "$CLIP" >> "$FILE"
echo "appended $(printf '%s' "$CLIP" | grep -c '') lines to $FILE"

"$ROOT/venvs/audio/bin/python" "$ROOT/scripts/discord_scrape.py" \
  --paste "$FILE" --label "$LABEL"

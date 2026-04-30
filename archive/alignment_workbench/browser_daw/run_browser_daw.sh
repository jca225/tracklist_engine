#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/johnnycabrahams/Desktop/tracklist_engine/workspaces/alignment_workbench/browser_daw"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
APP_URL="http://127.0.0.1:5173"

if [[ ! -d "$BACKEND_DIR" || ! -d "$FRONTEND_DIR" ]]; then
  echo "Expected directories not found under: $ROOT"
  exit 1
fi

BACKEND_CMD='cd "/Users/johnnycabrahams/Desktop/tracklist_engine/workspaces/alignment_workbench/browser_daw/backend" && if [ ! -d ".venv" ]; then python3 -m venv .venv; fi && source .venv/bin/activate && pip install -r requirements.txt && uvicorn main:app --reload --port 8000'

FRONTEND_CMD='cd "/Users/johnnycabrahams/Desktop/tracklist_engine/workspaces/alignment_workbench/browser_daw/frontend" && python3 -m http.server 5173'

launch_terminal() {
  local cmd="$1"
  osascript - "$cmd" <<'APPLESCRIPT'
on run argv
  tell application "Terminal"
    activate
    do script (item 1 of argv)
  end tell
end run
APPLESCRIPT
}

# Start backend in a new Terminal window.
launch_terminal "$BACKEND_CMD"

sleep 1

# Start frontend in a second Terminal window.
launch_terminal "$FRONTEND_CMD"

sleep 1

# Open the app in the default browser.
open "$APP_URL"

echo "Started Browser DAW."
echo "Backend: $BACKEND_DIR"
echo "Frontend: $FRONTEND_DIR"
echo "URL: $APP_URL"

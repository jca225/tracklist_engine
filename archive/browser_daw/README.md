# Browser DAW (Prototype)

This is an initial scaffold for a browser-based arranger that reads track metadata
from `tracklist_engine`'s existing SQLite database.

## Current features

- Backend API:
  - `GET /health`
  - `GET /tracks?limit=...`
  - `GET /tracks/{track_id}/analysis`
  - `GET /tracks/{track_id}/audio`
  - `POST /tracks/metadata-cache/clear`
  - `GET /projects`
  - `POST /projects`
  - `PATCH /projects/{project_id}`
  - `GET /projects/{project_id}/clips`
  - `POST /projects/{project_id}/clips`
  - `PATCH /clips/{clip_id}`
  - `POST /clips/{clip_id}/split?split_at_src_s=...`
  - `DELETE /clips/{clip_id}`
- Frontend:
  - Load tracks from backend
  - Dynamic metadata resolution (no schema changes) with lightweight file cache
  - Track browser shows richer metadata (song, artist, label, ISRC, key, integer BPM)
  - Track browser has independent scroll and live search by song/artist/id
  - Per-track variant picker when adding clips (`original`, `acappella`, `instrumental`, etc. as available)
  - Variant picker includes filesystem-discovered variants from audio drive folders
  - `instrumental`/`acappella` can be derived from stems (no pre-rendered file required)
  - Dynamic lanes (add lanes at runtime, drag clips across lanes)
  - Clips fill full lane height (clip/lane heights are matched)
  - Theme switcher with multiple presets (`dark`, `light`, `blue`, `sunset`)
  - Persisted project settings (master BPM/key)
  - Project harmonic target uses Camelot wheel (`1A`..`12B`) instead of key/mode fields
  - Add clips with auto-sync to project BPM/key
  - Drag clips horizontally across time (persisted)
  - Select clip, split, trim, and delete
  - Move selected clip between lanes
  - Split/trim snap to measure + cue points when analysis exists
  - Transport play/stop with moving playhead
  - Pause/resume transport (without resetting playhead)
  - Loop region with start/end controls for faster auditioning
  - Real clip audio playback via Web Audio (from backend audio endpoint)
  - Click arranger to seek playhead
  - Per-clip volume control
  - Waveform rendered inside each clip

## Run backend

```bash
cd /Users/johnnycabrahams/Desktop/tracklist_engine/browser_daw/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Run frontend (new terminal window)

```bash
cd /Users/johnnycabrahams/Desktop/tracklist_engine/browser_daw/frontend
python3 -m http.server 5173
```

Open: <http://127.0.0.1:5173>

## Next implementation steps

1. Add waveform rendering and per-clip zoom
2. Add real audio playback engine (AudioWorklet) instead of visual transport only
3. Add non-destructive clip slices (`clip_slices`) and reorder tool
4. Add WASM time-stretch/pitch-shift (Rubber Band) for high-quality sync

# Spectrogram review player

Human inspection of BB11/BB12 alignment spans: dual MIX | SOURCE spectrograms with OD-style **Truth** / **Our guess** boxes, Ableton labels, and playable windows.

**Session state / next work:** see  
[docs/agent_handoff_spectrogram_review_gt_capture_20260719.md](../../../docs/agent_handoff_spectrogram_review_gt_capture_20260719.md).

**Important:** if Truth disagrees with what you hear, suspect **Ableton GT capture** (`labeling/export_als_to_gt.py`), not the player math. Scorer success needs `traj_strict ≥ 0.5` (±2 s).

## Run

```bash
venvs/audio/bin/python -m eda.alignment.spectrogram_review.render \
  --set-id 1fsnxchk --limit 12 --outcome all

venvs/audio/bin/python -m eda.alignment.spectrogram_review.serve \
  --dir eda/alignment/spectrogram_review/out/1fsnxchk_all --port 8765
```

Open `http://127.0.0.1:8765/index.html` (not `file://`). Expect header **`audio-v5`**. If Web Audio is silent, use the **Mix backup** slider.

## Tests

```bash
venvs/audio/bin/python -m pytest tests/eda/alignment/test_spectrogram_review.py -q
```

## Modules

| File | Role |
|------|------|
| `render.py` | CLI gallery build |
| `player_html.py` | Interactive player template |
| `serve.py` | Range-capable HTTP |
| `audible.py` | Audible Truth extents (viewer) |
| `audio_clips.py` | ffmpeg window cuts (afade `st=` required) |
| `ableton_label.py` | GT fixture → Ableton track line |
| `spans.py` / `classify.py` | span_table enrichment / success rules |

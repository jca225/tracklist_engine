# Session handoff — spectrogram review + Ableton GT capture (2026-07-19)

**Branch:** `trm-ablation-framework` (dirty tree — do not assume clean).  
**Next session goal:** debug **Ableton → GT capture fidelity**, not the aligner / more sensors / more sets.

---

## 1. Binding conclusion (read this first)

Operator correction (do not re-litigate):

> The problem is **not** more DJ sets, more sensors, or tweaking the alignment algorithm against current scores. **We are not capturing Ableton correctly.** When the model’s guess matches what you hear and the exported GT disagrees, the label is wrong.

Smoking gun — BB12 spine slot `42w3` “Honest (Acappella)”:

| Layer | Says |
|--------|------|
| Tracklist / spine name | Acappella |
| Model `pred_stem` | `acappella` |
| Exported GT | Ableton **149** · **instrumental** · `ref_source=online_candidate` · `track_id=dgrjwux` |
| Scorer | Miss (`traj_strict=0`, `set_start_err≈61s`, group often `placement`) |

GT YAML (`labeling/fixtures/bb12_ground_truth.yaml`): instrumental row at mix `3456.18–3607.62`, stem from **file path** via `labeling/als/identity.py::classify_path` (e.g. `…/instrumental.flac` or candidate instrumental). Real acappella Honest GT is Ableton **053** at mix `1211–1262`, not this window.

**Next debugging entry points**

1. `labeling/export_als_to_gt.py` + `labeling/als/identity.py` (`classify_path`, `resolve_identity`)
2. BB12 `.als` (see [eda/alignment/failure_analysis/FOLLOWUPS.md](../eda/alignment/failure_analysis/FOLLOWUPS.md) for paths — verify before edit)
3. Already-related: **WS0** in that FOLLOWUPS (phantom / silent / deactivated clips) — same family as “audibility ≠ clip extent”
4. Regenerated YAML + write-back only after capture fix; then re-score — do not tune aligner on poisoned spans

---

## 2. What was built this session

Interactive OD-style review player under **`eda/alignment/spectrogram_review/`** (untracked package).

| Piece | Role |
|--------|------|
| `render.py` | Build gallery: mix/src spectrograms, cut audio, HTML player |
| `player_html.py` | Fullscreen dual MIX\|SRC + Web Audio (`audio-v5`) + native backup sliders |
| `serve.py` | HTTP + Range/206 + `Cache-Control: no-store` |
| `spans.py` / `classify.py` | span_table → cards; success = `traj_strict≥0.5` + identity |
| `ableton_label.py` | Resolve Ableton track # / name from GT fixtures |
| `audible.py` | Truth boxes from audible mix extents + stem energy onset (not silent padding) |
| `audio_clips.py` | ffmpeg cut — **must** use `afade=t=out:st={dur-0.05}` (see §3) |
| `source_audio.py` | Resolve stem from `~/aligning` manifest / filesystem |
| `tests/eda/alignment/test_spectrogram_review.py` | Unit tests (incl. audible / afade-related) |

**How to run**

```bash
# rebuild gallery (BB12 sample)
venvs/audio/bin/python -m eda.alignment.spectrogram_review.render \
  --set-id 1fsnxchk --limit 12 --outcome all

# serve (do not use file://)
venvs/audio/bin/python -m eda.alignment.spectrogram_review.serve \
  --dir eda/alignment/spectrogram_review/out/1fsnxchk_all --port 8765
# → http://127.0.0.1:8765/index.html
```

Header stamp should say **`audio-v5`**. If silent: use **Mix backup** `<audio>` slider; hard-refresh with `?v=…`.

Also: `eda/alignment/spectrogram_review/README.md`.

---

## 3. Bugs we hit (so you don’t rediscover them)

1. **Stale `index.html`** — editing `player_html.py` without `write_player` / `render` leaves the browser on old JS. Always regenerate HTML; server uses `no-store`.
2. **ffmpeg `afade=t=out:d=0.05` without `st=`** — fades out in the first 50 ms and leaves the rest of the clip **silent**. Fixed in `audio_clips.py`. Re-cut if clips are ~40 KB for 90 s or mid-clip RMS ≈ 0.
3. **Browser autoplay / Safari** — Web Audio: create `AudioContext` on gesture; prefetch bytes; native backup sliders in v5.
4. **Truth over silence** — clip `ref_start=0` / full set span includes stem padding & fader lead-in. Review now uses `audible.py` (GT `audible_*` + energy onset). That improves the *viewer*; it does **not** fix bad Ableton export.
5. **Spine name ≠ GT stem** — card titled “Honest (Acappella)” while Truth followed instrumental GT 149. WIP: prefer Ableton GT name on cards + `Truth · {stem}` on boxes (`render.py` `_gt_display_name`). Re-render before trusting UI labels.
6. **“Miss” while placement looks close** (e.g. Ableton 76 / slot `22`) — success needs `traj_strict≥0.5` (±2 s). A steady ~4 s offset fails almost all seconds even when a human hears “close.”

---

## 4. Git / tree state

- Package `eda/alignment/spectrogram_review/` is **untracked** (`??`).
- `tests/eda/alignment/test_spectrogram_review.py` may be untracked or modified — check `git status`.
- Gallery artifacts under `eda/alignment/spectrogram_review/out/` (gitignored or untracked noise) — regenerate as needed.
- Other dirty paths on the branch (`labeling/*`, plans, etc.) may be **other agents’ WIP** — don’t sweep into a commit without checking.
- Player server on **8765** may be down; restart with §2 if needed.
- **Do not commit** unless operator asks; this handoff is the persist.

---

## 5. Suggested next-session checklist

1. Open BB12 `.als` at ~57:36 — what clip is actually audible for Honest / acappella vs instrumental lane 149?
2. Trace that clip through `export_als_to_gt` → YAML row 149 (`claimed_stem`, `ref_source`, warps, gain).
3. Decide capture rules: path-stem vs arranged stem vs tracklist claim; deactivated/muted/zero-fader (WS0).
4. Re-export fixture → write-back (coord on pi-storage) → rebuild `span_table` → re-score.
5. Only then re-open spectrogram review to validate labels against hearing.

---

## 6. Related docs

- [eda/alignment/failure_analysis/FOLLOWUPS.md](../eda/alignment/failure_analysis/FOLLOWUPS.md) — WS0 GT export blocker  
- [docs/alignment_status.md](alignment_status.md) — headline metrics only (do not hand-edit)  
- [docs/alignment_recharacterization.md](alignment_recharacterization.md) — three axes  
- `labeling/export_als_to_gt.py`, `labeling/als/identity.py`

#!/usr/bin/env python3
"""Render A/B verification snippets + review page for a predicted timeline.

For each predicted span, extract two loudness-normalized snippets from the
canonical audio in ~/aligning/<set>/:

  A = mix at the predicted set position
  B = reference track at the predicted ref offset (same delta into the span)

If the prediction is right, A and B are the same section of the same song.
A single self-contained review.html plays A then B per span, takes a verdict
(pass / nudge / wrong) via keyboard, and exports verdicts as JSON. Spans are
ordered most-suspicious-first so human attention lands where the model is
most likely wrong.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.render_review_snippets \\
        --set-id 2nvzlh2k [--snippet-s 6]
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT_DIR = Path(__file__).resolve().parent / "out"
ALIGNING_ROOT = Path.home() / "aligning"

# How far into the predicted span to sample (skip the transition blur at the
# very cue point), capped so short spans still land inside themselves.
_OFFSET_FRAC = 0.25
_OFFSET_CAP_S = 10.0
_UNANCHORED_SCORE = 9999.0  # zero/missing cue -> decode-only, review early


@dataclass(frozen=True)
class ReviewItem:
    rank: int
    slot_label: str
    name: str
    recording_id: str
    set_start_s: float
    ref_start_s: float
    cue_anchor_s: float | None
    confidence: float
    suspicion: float
    mix_clip: str
    ref_clip: str


def find_aligning_dir(set_id: str) -> Path:
    hits = sorted(ALIGNING_ROOT.glob(f"{set_id}__*"))
    if not hits:
        sys.exit(f"no ~/aligning folder for {set_id} — run pull_set_for_alignment.py first")
    return hits[0]


_STEM_FILE = {"acappella": "vocals", "instrumental": "instrumental"}


def pick_audio(span: dict, track: dict) -> Path | None:
    """Audio to A/B against the mix: the Demucs stem for acappella /
    instrumental claims (the mix has a different instrumental under those
    vocals — the full track is barely verifiable by ear), else the full
    track. Stems share the full track's timeline, so ref offsets hold."""
    stem_key = _STEM_FILE.get(span.get("claimed_stem") or "regular")
    if stem_key:
        stem_path = (track.get("stems") or {}).get(stem_key)
        if stem_path and Path(stem_path).is_file():
            return Path(stem_path)
    p = Path(track["local_path"])
    return p if p.is_file() else None


_dur_cache: dict[Path, float] = {}


def audio_duration_s(path: Path) -> float:
    if path not in _dur_cache:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        try:
            _dur_cache[path] = float(r.stdout.strip())
        except ValueError:
            _dur_cache[path] = 0.0
    return _dur_cache[path]


def suspicion_score(span: dict) -> float:
    cue = span.get("cue_anchor_s")
    if cue is None or cue <= 0.0:
        return _UNANCHORED_SCORE
    return abs(span["set_start_s"] - cue)


def ffmpeg_snippet(src: Path, start_s: float, dur_s: float, dst: Path) -> bool:
    """Extract one loudness-normalized snippet; True on success."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, start_s):.3f}", "-t", f"{dur_s:.3f}",
        "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.5,afade=t=in:d=0.05",
        "-ac", "2", "-ar", "44100", "-b:a", "128k",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed for {dst.name}: {r.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def render_html(set_id: str, title: str, items: list[ReviewItem]) -> str:
    data = json.dumps([i.__dict__ for i in items])
    n = len(items)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)} — span review</title>
<style>
  body {{ font: 15px/1.5 -apple-system, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; background:#111; color:#eee; }}
  h1 {{ font-size: 1.2rem; }}
  #card {{ border: 1px solid #444; border-radius: 10px; padding: 1.2rem; background:#1a1a1a; }}
  .slot {{ font-size: 1.5rem; font-weight: 700; }}
  .name {{ font-size: 1.15rem; margin: .3rem 0 .8rem; }}
  .meta {{ color: #999; font-size: .85rem; }}
  .sus-high {{ color: #ff6b6b; }} .sus-mid {{ color: #ffd166; }} .sus-low {{ color: #6bcb77; }}
  button {{ font-size: 1rem; padding: .5rem 1rem; margin: .3rem .3rem 0 0; border-radius: 8px; border: 1px solid #555; background:#222; color:#eee; cursor: pointer; }}
  button:hover {{ background:#333; }}
  .verdict-pass {{ border-color:#6bcb77; }} .verdict-nudge {{ border-color:#ffd166; }} .verdict-wrong {{ border-color:#ff6b6b; }}
  #prog {{ margin: 1rem 0; color:#999; }}
  .playing {{ outline: 2px solid #4da3ff; }}
  kbd {{ background:#333; border-radius:4px; padding:0 .4em; font-size:.85em; }}
  #done {{ display:none; padding:1rem; border:1px solid #6bcb77; border-radius:10px; margin-top:1rem; }}
</style></head><body>
<h1>{html.escape(title)} — predicted-span review ({n} spans, worst first)</h1>
<p class="meta">Per span: <b>A</b> = mix at predicted position, <b>B</b> = reference at predicted offset.
Same song, same section ⇒ <kbd>1</kbd> pass. Right song, timing feels off ⇒ <kbd>2</kbd> nudge.
Different song/section ⇒ <kbd>3</kbd> wrong. <kbd>a</kbd>/<kbd>b</kbd> replay, <kbd>space</kbd> play A→B,
<kbd>←</kbd>/<kbd>→</kbd> navigate. Verdicts autosave to localStorage.</p>
<div id="prog"></div>
<div id="card">
  <div class="slot"></div><div class="name"></div><div class="meta" id="detail"></div>
  <div>
    <button id="bA">▶ A mix</button><button id="bB">▶ B ref</button><button id="bAB">▶ A→B</button>
  </div>
  <div>
    <button id="vP" class="verdict-pass">1 pass</button>
    <button id="vN" class="verdict-nudge">2 nudge</button>
    <button id="vW" class="verdict-wrong">3 wrong</button>
  </div>
</div>
<div id="done"><b>All spans reviewed.</b> <button id="bExport">Download verdicts JSON</button></div>
<p><button id="bExport2">Export verdicts so far</button></p>
<script>
const ITEMS = {data};
const KEY = "review_verdicts_{set_id}";
let verdicts = JSON.parse(localStorage.getItem(KEY) || "{{}}");
let idx = ITEMS.findIndex(it => !(it.slot_label in verdicts)); if (idx < 0) idx = 0;
const audio = new Audio(); let chain = null;
function fmt(s) {{ const m = Math.floor(s/60); return m + ":" + String(Math.floor(s%60)).padStart(2,"0"); }}
function susClass(s) {{ return s >= 9999 ? "sus-high" : s > 45 ? "sus-high" : s > 25 ? "sus-mid" : "sus-low"; }}
function show() {{
  const it = ITEMS[idx];
  document.querySelector(".slot").textContent = "slot " + it.slot_label;
  document.querySelector(".name").textContent = it.name;
  const sus = it.suspicion >= 9999 ? "no cue anchor (decode-only)" : "|pred−cue| = " + it.suspicion.toFixed(0) + "s";
  document.getElementById("detail").innerHTML =
    `pred ${{fmt(it.set_start_s)}} · ref offset ${{fmt(it.ref_start_s)}} · conf ${{it.confidence.toFixed(2)}} · <span class="${{susClass(it.suspicion)}}">${{sus}}</span>` +
    (it.slot_label in verdicts ? ` · <b>verdict: ${{verdicts[it.slot_label]}}</b>` : "");
  const ndone = Object.keys(verdicts).length;
  document.getElementById("prog").textContent = `span ${{idx+1}}/${{ITEMS.length}} — ${{ndone}} verdicts saved`;
  document.getElementById("done").style.display = ndone >= ITEMS.length ? "block" : "none";
}}
function play(src, then) {{ chain = then || null; audio.src = src; audio.play(); }}
audio.onended = () => {{ if (chain) {{ const c = chain; chain = null; play(c); }} }};
function playA(then) {{ play(ITEMS[idx].mix_clip, then); }}
function playB() {{ play(ITEMS[idx].ref_clip); }}
function verdict(v) {{
  verdicts[ITEMS[idx].slot_label] = v;
  localStorage.setItem(KEY, JSON.stringify(verdicts));
  if (idx < ITEMS.length - 1) {{ idx++; show(); playA(ITEMS[idx].ref_clip); }} else show();
}}
function exportJson() {{
  const blob = new Blob([JSON.stringify({{set_id: "{set_id}", verdicts: verdicts}}, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "{set_id}_review_verdicts.json"; a.click();
}}
document.getElementById("bA").onclick = () => playA();
document.getElementById("bB").onclick = playB;
document.getElementById("bAB").onclick = () => playA(ITEMS[idx].ref_clip);
document.getElementById("vP").onclick = () => verdict("pass");
document.getElementById("vN").onclick = () => verdict("nudge");
document.getElementById("vW").onclick = () => verdict("wrong");
document.getElementById("bExport").onclick = exportJson;
document.getElementById("bExport2").onclick = exportJson;
document.addEventListener("keydown", e => {{
  if (e.key === "1") verdict("pass"); else if (e.key === "2") verdict("nudge");
  else if (e.key === "3") verdict("wrong"); else if (e.key === "a") playA();
  else if (e.key === "b") playB(); else if (e.key === " ") {{ e.preventDefault(); playA(ITEMS[idx].ref_clip); }}
  else if (e.key === "ArrowRight" && idx < ITEMS.length-1) {{ idx++; show(); }}
  else if (e.key === "ArrowLeft" && idx > 0) {{ idx--; show(); }}
}});
show();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--snippet-s", type=float, default=6.0)
    p.add_argument("--limit", type=int, default=0, help="render only first N (debug)")
    args = p.parse_args(argv)

    timeline_path = OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    timeline = json.loads(timeline_path.read_text())
    spans = timeline["spans"]

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    mix_path = Path(manifest["mix_local_path"])
    if not mix_path.is_file():
        sys.exit(f"mix audio missing: {mix_path}")

    review_dir = OUT_DIR / "review" / args.set_id
    clips_dir = review_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(spans, key=suspicion_score, reverse=True)
    if args.limit:
        ordered = ordered[: args.limit]

    items: list[ReviewItem] = []
    missing_audio: list[str] = []
    for rank, s in enumerate(ordered, start=1):
        t = by_tid.get(s["recording_id"])
        ref_audio = pick_audio(s, t) if t is not None else None
        if ref_audio is None:
            missing_audio.append(f"{s['slot_label']} {s['name'][:50]}")
            continue
        span_len = max(0.0, s["set_end_s"] - s["set_start_s"])
        delta = min(_OFFSET_CAP_S, span_len * _OFFSET_FRAC)
        # keep the ref snippet inside the file
        ref_dur = audio_duration_s(ref_audio)
        ref_at = min(s["ref_start_s"] + delta, max(0.0, ref_dur - args.snippet_s))
        mix_clip = clips_dir / f"{rank:03d}__{s['slot_label']}__mix.mp3"
        ref_clip = clips_dir / f"{rank:03d}__{s['slot_label']}__ref.mp3"
        ok = ffmpeg_snippet(mix_path, s["set_start_s"] + delta, args.snippet_s, mix_clip) \
            and ffmpeg_snippet(ref_audio, ref_at, args.snippet_s, ref_clip)
        if not ok:
            missing_audio.append(f"{s['slot_label']} (ffmpeg)")
            continue
        items.append(ReviewItem(
            rank=rank, slot_label=s["slot_label"], name=s["name"],
            recording_id=s["recording_id"], set_start_s=s["set_start_s"],
            ref_start_s=s["ref_start_s"], cue_anchor_s=s.get("cue_anchor_s"),
            confidence=s["confidence"], suspicion=suspicion_score(s),
            mix_clip=f"clips/{mix_clip.name}", ref_clip=f"clips/{ref_clip.name}",
        ))
        if rank % 25 == 0:
            print(f"  rendered {rank}/{len(ordered)}")

    html_path = review_dir / "review.html"
    html_path.write_text(render_html(args.set_id, manifest.get("title", args.set_id), items))
    print(f"\nrendered {len(items)} spans -> {review_dir}")
    if missing_audio:
        print(f"SKIPPED {len(missing_audio)} spans (no local audio):")
        for m in missing_audio:
            print(f"  {m}")
    print(f"open: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

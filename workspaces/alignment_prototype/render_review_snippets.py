#!/usr/bin/env python3
"""Render A/B verification snippets + review page for a predicted timeline.

For each predicted span, extract two loudness-normalized windows from the
canonical audio in ~/aligning/<set>/:

  A = mix from just before the predicted span start (3 s pre-roll)
  B = reference from the corresponding predicted ref offset

The windows are time-corresponded: position t in A maps to position
t * tempo_ratio in B, so the review player can hot-swap A <-> B (or overlay
both) at the same musical moment. If the prediction is right, "both" sounds
like a doubled track; if it's off, you hear flam/echo immediately.

review.html is a self-contained player: scrub bar, play/pause, A/B/Both
modes, seek keys, verdicts (pass / nudge / wrong) with localStorage
persistence + JSON export. Spans are ordered most-suspicious-first.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.render_review_snippets \\
        --set-id 2nvzlh2k [--max-window-s 45]
"""
from __future__ import annotations

import argparse
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

_PRE_ROLL_S = 3.0        # context before the predicted span start
_WIN_MIN_S = 20.0
_PAD_S = 8.0             # window = span length + pad (clamped)
_UNANCHORED_SCORE = 9999.0  # zero/missing cue -> decode-only, review early


@dataclass(frozen=True)
class ReviewItem:
    rank: int
    slot_label: str
    name: str
    recording_id: str
    claimed_stem: str
    set_start_s: float
    ref_start_s: float
    cue_anchor_s: float | None
    confidence: float
    suspicion: float
    win_s: float          # total A window length (mix seconds, incl. pre-roll)
    pre_s: float          # pre-roll before the predicted start
    ratio: float          # ref seconds per mix second (predicted stretch)
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


def ffmpeg_snippet(src: Path, start_s: float, dur_s: float, dst: Path,
                   *, lead_silence_s: float = 0.0) -> bool:
    af = "loudnorm=I=-16:TP=-1.5,afade=t=in:d=0.05"
    if lead_silence_s > 0.0:
        # keep A<->B clock correspondence when the ref window starts before 0
        af += f",adelay={int(lead_silence_s * 1000)}:all=1"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, start_s):.3f}", "-t", f"{dur_s:.3f}",
        "-i", str(src),
        "-af", af,
        "-ac", "2", "-ar", "44100", "-b:a", "128k",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed for {dst.name}: {r.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__ — span review</title>
<style>
  :root { --bg:#111; --card:#1a1a1a; --line:#3a3a3a; --txt:#eee; --dim:#999;
          --green:#6bcb77; --yellow:#ffd166; --red:#ff6b6b; --blue:#4da3ff; }
  body { font: 15px/1.5 -apple-system, sans-serif; max-width: 860px;
         margin: 1.5rem auto; padding: 0 1rem; background:var(--bg); color:var(--txt); }
  h1 { font-size: 1.15rem; margin: 0 0 .4rem; }
  .meta { color: var(--dim); font-size: .85rem; }
  #card { border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.3rem;
          background:var(--card); margin-top: .8rem; }
  .slot { font-size: 1.4rem; font-weight: 700; display:inline-block; }
  .name { font-size: 1.15rem; margin: .15rem 0 .5rem; }
  .sus-high { color: var(--red); } .sus-mid { color: var(--yellow); } .sus-low { color: var(--green); }
  button { font-size: .95rem; padding: .45rem .9rem; margin: .25rem .25rem 0 0;
           border-radius: 8px; border: 1px solid #555; background:#222; color:var(--txt);
           cursor: pointer; }
  button:hover { background:#333; }
  button.active { background:#2a4d7a; border-color: var(--blue); }
  .verdict-pass.chosen { background:#234d2a; border-color:var(--green); }
  .verdict-nudge.chosen { background:#4d4423; border-color:var(--yellow); }
  .verdict-wrong.chosen { background:#4d2323; border-color:var(--red); }
  #bar { position: relative; height: 38px; background:#222; border-radius: 8px;
         margin: .9rem 0 .3rem; cursor: pointer; border: 1px solid var(--line); }
  #fill { position:absolute; top:0; left:0; bottom:0; background:#2a4d7a;
          border-radius: 8px 0 0 8px; pointer-events:none; }
  #startmark { position:absolute; top:0; bottom:0; width:2px; background:var(--yellow);
               pointer-events:none; }
  #lstart, #lend { position:absolute; top:0; bottom:0; width:2px; background:var(--green);
                   pointer-events:none; display:none; }
  #loopshade { position:absolute; top:0; bottom:0; background:rgba(107,203,119,.12);
               pointer-events:none; display:none; }
  #clock { font-variant-numeric: tabular-nums; color: var(--dim); font-size:.85rem; }
  #prog { margin: .7rem 0 0; color: var(--dim); font-size:.9rem; }
  kbd { background:#333; border-radius:4px; padding:0 .4em; font-size:.85em; }
  #done { display:none; padding:1rem; border:1px solid var(--green); border-radius:10px;
          margin-top:1rem; }
  .row { display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }
  #vol { width: 110px; }
  table.help { color:var(--dim); font-size:.8rem; border-collapse:collapse; margin-top:.8rem; }
  table.help td { padding: 0 .9rem 0 0; }
  #jump { width: 56px; background:#222; color:var(--txt); border:1px solid #555;
          border-radius:6px; padding:.3rem .4rem; }
</style></head><body>
<h1>__TITLE__ — predicted-span review (__N__ spans, worst first)</h1>
<div class="meta"><b>A</b> = mix at predicted position · <b>B</b> = reference at predicted
offset (tempo-corrected) · <b>Both</b> overlays them — aligned sounds doubled, misaligned
sounds flam/echo. Yellow line = predicted span start (after 3 s pre-roll).</div>

<div id="card">
  <div class="row" style="justify-content:space-between">
    <div><span class="slot"></span> <span id="verdict-badge"></span></div>
    <div id="navpos" class="meta"></div>
  </div>
  <div class="name"></div>
  <div class="meta" id="detail"></div>

  <div id="bar">
    <div id="loopshade"></div><div id="fill"></div>
    <div id="startmark"></div><div id="lstart"></div><div id="lend"></div>
  </div>
  <div class="row" style="justify-content:space-between">
    <span id="clock">0:00 / 0:00</span>
    <span class="row"><label class="meta">vol</label>
      <input id="vol" type="range" min="0" max="1" step="0.01" value="1"></span>
  </div>

  <div class="row" style="margin-top:.5rem">
    <button id="bPlay">▶ play</button>
    <button id="mA">A mix</button>
    <button id="mB">B ref</button>
    <button id="mS">Both</button>
    <button id="bRestart">↩ span start</button>
    <button id="bLoop">loop: off</button>
  </div>
  <div class="row" style="margin-top:.4rem">
    <button id="vP" class="verdict-pass">1 pass</button>
    <button id="vN" class="verdict-nudge">2 nudge</button>
    <button id="vW" class="verdict-wrong">3 wrong</button>
    <span style="flex:1"></span>
    <button id="bPrev">‹ prev</button>
    <button id="bNext">next ›</button>
    <button id="bUnjudged">next unjudged »</button>
    <span class="meta">go to <input id="jump" placeholder="slot"></span>
  </div>
</div>

<div id="prog"></div>
<div id="done"><b>All spans reviewed.</b> <button id="bExport">Download verdicts JSON</button></div>
<p><button id="bExport2">Export verdicts so far</button></p>

<table class="help"><tr>
  <td><kbd>space</kbd> play/pause</td><td><kbd>a</kbd>/<kbd>b</kbd>/<kbd>s</kbd> A / B / Both</td>
  <td><kbd>←</kbd>/<kbd>→</kbd> seek 5 s</td><td><kbd>0</kbd> span start</td>
  <td><kbd>l</kbd> loop A↔B point</td>
</tr><tr>
  <td><kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> pass/nudge/wrong</td>
  <td><kbd>j</kbd>/<kbd>k</kbd> prev/next</td><td><kbd>u</kbd> next unjudged</td>
  <td colspan="2">verdicts autosave (localStorage)</td>
</tr></table>

<script>
const ITEMS = __DATA__;
const KEY = "review_verdicts___SETID__";
let verdicts = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = ITEMS.findIndex(it => !(it.slot_label in verdicts)); if (idx < 0) idx = 0;
let mode = "A";                 // A | B | S
let loopA = null, loopB = null; // loop region in mix-window seconds
const mix = new Audio(), ref = new Audio();
mix.preload = ref.preload = "auto";
const $ = id => document.getElementById(id);

function it() { return ITEMS[idx]; }
function fmt(s) { s = Math.max(0, s); const m = Math.floor(s/60);
  return m + ":" + String(Math.floor(s%60)).padStart(2,"0"); }
function susClass(s) { return s >= 9999 || s > 45 ? "sus-high" : s > 25 ? "sus-mid" : "sus-low"; }

// ---- transport: master clock is mix-window seconds -------------------------
function t() { return mode === "B" ? ref.currentTime / it().ratio : mix.currentTime; }
function seek(sec) {
  sec = Math.min(Math.max(0, sec), it().win_s - 0.05);
  mix.currentTime = sec; ref.currentTime = sec * it().ratio;
  draw();
}
function applyMode() {
  mix.muted = (mode === "B");
  ref.muted = (mode === "A");
  mix.volume = ref.volume = (mode === "S" ? 0.6 : 1) * parseFloat($("vol").value);
  ["mA","mB","mS"].forEach(id => $(id).classList.remove("active"));
  $({A:"mA",B:"mB",S:"mS"}[mode]).classList.add("active");
}
function playing() { return !mix.paused; }
function play() {
  ref.playbackRate = it().ratio;            // tempo-correct B onto A's clock
  ref.preservesPitch = true;
  ref.currentTime = mix.currentTime * it().ratio;
  Promise.allSettled([mix.play(), ref.play()]).then(()=>{});
  $("bPlay").textContent = "❚❚ pause";
}
function pause() { mix.pause(); ref.pause(); $("bPlay").textContent = "▶ play"; }
function toggle() { playing() ? pause() : play(); }

mix.ontimeupdate = () => {
  // keep B locked to A's clock; resync if drifted
  const want = mix.currentTime * it().ratio;
  if (Math.abs(ref.currentTime - want) > 0.06) ref.currentTime = want;
  if (loopA !== null && loopB !== null && mix.currentTime >= loopB) seek(loopA);
  draw();
};
mix.onended = () => { if (loopA !== null && loopB !== null) { seek(loopA); play(); } else pause(); };

// ---- scrub bar --------------------------------------------------------------
const bar = $("bar");
function barSeek(e) {
  const r = bar.getBoundingClientRect();
  seek((e.clientX - r.left) / r.width * it().win_s);
}
let dragging = false;
bar.addEventListener("mousedown", e => { dragging = true; barSeek(e); });
window.addEventListener("mousemove", e => { if (dragging) barSeek(e); });
window.addEventListener("mouseup", () => dragging = false);

function draw() {
  const w = it().win_s;
  $("fill").style.width = (100 * t() / w) + "%";
  $("startmark").style.left = (100 * it().pre_s / w) + "%";
  $("clock").textContent = fmt(t()) + " / " + fmt(w) +
    "   (mix " + fmt(it().set_start_s - it().pre_s + t()) + ")";
  const showLoop = loopA !== null && loopB !== null;
  $("lstart").style.display = loopA !== null ? "block" : "none";
  $("lend").style.display = $("loopshade").style.display = showLoop ? "block" : "none";
  if (loopA !== null) $("lstart").style.left = (100 * loopA / w) + "%";
  if (showLoop) {
    $("lend").style.left = (100 * loopB / w) + "%";
    $("loopshade").style.left = (100 * loopA / w) + "%";
    $("loopshade").style.width = (100 * (loopB - loopA) / w) + "%";
  }
}

// ---- span switching ----------------------------------------------------------
function show(autoplay) {
  const x = it();
  pause();
  loopA = loopB = null; $("bLoop").textContent = "loop: off";
  mix.src = x.mix_clip; ref.src = x.ref_clip;
  document.querySelector(".slot").textContent = "slot " + x.slot_label;
  document.querySelector(".name").textContent = x.name +
    (x.claimed_stem !== "regular" ? "  [" + x.claimed_stem + " → stem]" : "");
  const sus = x.suspicion >= 9999 ? "no cue anchor (decode-only)"
                                  : "|pred−cue| = " + x.suspicion.toFixed(0) + "s";
  $("detail").innerHTML =
    "pred " + fmt(x.set_start_s) + " in mix · ref offset " + fmt(x.ref_start_s) +
    " · stretch ×" + x.ratio.toFixed(3) + " · conf " + x.confidence.toFixed(2) +
    " · <span class='" + susClass(x.suspicion) + "'>" + sus + "</span>";
  const v = verdicts[x.slot_label];
  $("verdict-badge").innerHTML = v ? "· <b class='" +
    (v==="pass"?"sus-low":v==="nudge"?"sus-mid":"sus-high") + "'>" + v + "</b>" : "";
  ["vP","vN","vW"].forEach(id => $(id).classList.remove("chosen"));
  if (v) $({pass:"vP",nudge:"vN",wrong:"vW"}[v]).classList.add("chosen");
  $("navpos").textContent = (idx+1) + " / " + ITEMS.length;
  const ndone = Object.keys(verdicts).length;
  $("prog").textContent = ndone + " / " + ITEMS.length + " verdicts saved";
  $("done").style.display = ndone >= ITEMS.length ? "block" : "none";
  applyMode();
  seek(0);
  if (autoplay) { mix.oncanplay = () => { mix.oncanplay = null; play(); }; }
}
function go(d) { idx = Math.min(Math.max(0, idx + d), ITEMS.length - 1); show(false); }
function nextUnjudged() {
  const j = ITEMS.findIndex((x, i) => i > idx && !(x.slot_label in verdicts));
  const k = j >= 0 ? j : ITEMS.findIndex(x => !(x.slot_label in verdicts));
  if (k >= 0) { idx = k; show(true); }
}
function verdict(v) {
  verdicts[it().slot_label] = v;
  localStorage.setItem(KEY, JSON.stringify(verdicts));
  nextUnjudged(); show(true);
}
function exportJson() {
  const blob = new Blob([JSON.stringify({set_id: "__SETID__", verdicts}, null, 2)],
                        {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "__SETID___review_verdicts.json"; a.click();
}
function setLoopPoint() {
  if (loopA === null) { loopA = t(); $("bLoop").textContent = "loop: set end…"; }
  else if (loopB === null) {
    loopB = Math.max(loopA + 0.5, t()); $("bLoop").textContent = "loop: on";
    seek(loopA); if (!playing()) play();
  } else { loopA = loopB = null; $("bLoop").textContent = "loop: off"; }
  draw();
}

$("bPlay").onclick = toggle;
$("mA").onclick = () => { mode = "A"; applyMode(); };
$("mB").onclick = () => { mode = "B"; applyMode(); };
$("mS").onclick = () => { mode = "S"; applyMode(); };
$("bRestart").onclick = () => seek(it().pre_s);
$("bLoop").onclick = setLoopPoint;
$("vP").onclick = () => verdict("pass");
$("vN").onclick = () => verdict("nudge");
$("vW").onclick = () => verdict("wrong");
$("bPrev").onclick = () => go(-1);
$("bNext").onclick = () => go(1);
$("bUnjudged").onclick = nextUnjudged;
$("bExport").onclick = $("bExport2").onclick = exportJson;
$("vol").oninput = applyMode;
$("jump").onchange = () => {
  const j = ITEMS.findIndex(x => x.slot_label === $("jump").value.trim());
  if (j >= 0) { idx = j; show(true); } $("jump").value = "";
};

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key;
  if (k === " ") { e.preventDefault(); toggle(); }
  else if (k === "a") { mode = "A"; applyMode(); }
  else if (k === "b") { mode = "B"; applyMode(); }
  else if (k === "s") { mode = "S"; applyMode(); }
  else if (k === "ArrowLeft") { e.preventDefault(); seek(t() - 5); }
  else if (k === "ArrowRight") { e.preventDefault(); seek(t() + 5); }
  else if (k === "0") seek(it().pre_s);
  else if (k === "l") setLoopPoint();
  else if (k === "1") verdict("pass");
  else if (k === "2") verdict("nudge");
  else if (k === "3") verdict("wrong");
  else if (k === "j") go(-1);
  else if (k === "k") go(1);
  else if (k === "u") nextUnjudged();
});
show(false);
</script></body></html>
"""


def render_html(set_id: str, title: str, items: list[ReviewItem]) -> str:
    import html as _html
    return (_HTML
            .replace("__DATA__", json.dumps([i.__dict__ for i in items]))
            .replace("__TITLE__", _html.escape(title))
            .replace("__SETID__", set_id)
            .replace("__N__", str(len(items))))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--max-window-s", type=float, default=45.0)
    p.add_argument("--limit", type=int, default=0, help="render only first N (debug)")
    args = p.parse_args(argv)

    timeline_path = OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    timeline = json.loads(timeline_path.read_text())
    spans = timeline["spans"]

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
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
        span_len = max(1.0, s["set_end_s"] - s["set_start_s"])
        ref_len = max(1.0, (s["ref_end_s"] or s["ref_start_s"] + span_len) - s["ref_start_s"])
        ratio = min(1.25, max(0.8, ref_len / span_len))
        win = min(args.max_window_s, max(_WIN_MIN_S, span_len + _PAD_S))
        pre = min(_PRE_ROLL_S, s["set_start_s"], s["ref_start_s"] / ratio)
        ref_dur = audio_duration_s(ref_audio)
        mix_dur = audio_duration_s(mix_path)
        total = pre + win  # fixed window length in mix seconds
        mix_at = s["set_start_s"] - pre
        # a tail span can start near the mix end — slide the fixed-length
        # window left so it stays inside the file; the start-marker (pre)
        # moves with it so the yellow line stays honest
        if mix_at + total > mix_dur:
            mix_at = max(0.0, mix_dur - total)
            pre = s["set_start_s"] - mix_at
        ref_at = s["ref_start_s"] - pre * ratio
        lead = max(0.0, -ref_at)
        ref_at = max(0.0, ref_at)
        ref_take = min(total * ratio + 0.5 - lead, max(1.0, ref_dur - ref_at))
        mix_clip = clips_dir / f"{rank:03d}__{s['slot_label']}__mix.mp3"
        ref_clip = clips_dir / f"{rank:03d}__{s['slot_label']}__ref.mp3"
        ok = ffmpeg_snippet(mix_path, mix_at, total, mix_clip) \
            and ffmpeg_snippet(ref_audio, ref_at, ref_take, ref_clip,
                               lead_silence_s=lead)
        if not ok:
            missing_audio.append(f"{s['slot_label']} (ffmpeg)")
            continue
        items.append(ReviewItem(
            rank=rank, slot_label=s["slot_label"], name=s["name"],
            recording_id=s["recording_id"],
            claimed_stem=s.get("claimed_stem") or "regular",
            set_start_s=s["set_start_s"], ref_start_s=s["ref_start_s"],
            cue_anchor_s=s.get("cue_anchor_s"), confidence=s["confidence"],
            suspicion=suspicion_score(s), win_s=total, pre_s=pre,
            ratio=ratio,
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

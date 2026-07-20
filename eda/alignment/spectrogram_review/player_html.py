"""Fullscreen spectrogram player — pick a clip, Enter, play with moving playhead."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Spectrogram player — aligner review</title>
<style>
  :root {
    --bg: #0c0d10;
    --panel: #16181e;
    --text: #f2f2f2;
    --muted: #9aa0a6;
    --line: #2a2f3a;
    --ok: #3dd68c;
    --bad: #ff6b5a;
    --truth: #3dd68c;
    --guess: #e879f9;
    --accent: #7cb3ff;
    --focus: #ffd166;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
  body.playing-mode { overflow: hidden; }

  /* ---- picker ---- */
  #picker { min-height: 100%; padding: 20px 22px 48px; max-width: 920px; margin: 0 auto; }
  #picker h1 { margin: 0 0 6px; font-size: 22px; font-weight: 700; }
  #picker .sub { color: var(--muted); font-size: 14px; margin: 0 0 16px; max-width: 40rem; line-height: 1.45; }
  .keys {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; font-size: 12px; color: var(--muted);
  }
  .keys kbd {
    background: var(--panel); border: 1px solid var(--line); border-radius: 4px;
    padding: 2px 7px; color: var(--text); font-family: ui-monospace, monospace;
  }
  .filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  .filters select, .filters input {
    background: var(--panel); color: var(--text); border: 1px solid var(--line);
    border-radius: 6px; padding: 7px 10px; font: inherit; font-size: 13px;
  }
  .group { margin: 22px 0 10px; }
  .group h2 { margin: 0 0 4px; font-size: 15px; font-weight: 700; }
  .group.fail h2 { color: var(--bad); }
  .group.ok h2 { color: var(--ok); }
  .group .blurb { color: var(--muted); font-size: 12px; margin: 0 0 8px; }
  .list { display: flex; flex-direction: column; gap: 6px; }
  .clip-btn {
    text-align: left; width: 100%;
    background: var(--panel); color: var(--text);
    border: 1px solid var(--line); border-radius: 8px;
    padding: 10px 12px; cursor: pointer; font: inherit;
    display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: center;
  }
  .clip-btn:hover { border-color: #4a5568; }
  .clip-btn.focused, .clip-btn:focus {
    outline: none; border-color: var(--focus);
    box-shadow: 0 0 0 2px rgba(255,209,102,0.25);
  }
  .clip-btn .badge {
    font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
  }
  .clip-btn .badge.miss { background: rgba(255,107,90,0.2); color: var(--bad); }
  .clip-btn .badge.hit { background: rgba(61,214,140,0.18); color: var(--ok); }
  .clip-btn .name { font-weight: 600; font-size: 14px; }
  .clip-btn .meta { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .clip-btn .hint { color: var(--muted); font-size: 11px; white-space: nowrap; }
  .clip-btn.hidden { display: none; }
  .group.hidden { display: none; }
  .weak {
    margin: 0 0 18px; padding: 12px 14px; background: #1c1810;
    border: 1px solid #5c4a1f; border-radius: 8px; font-size: 13px;
  }
  .weak strong { color: #ffd166; }

  /* ---- stage (fullscreen player) ---- */
  #stage {
    display: none; position: fixed; inset: 0; background: #000; z-index: 50;
    flex-direction: column;
  }
  body.playing-mode #picker { display: none; }
  body.playing-mode #stage { display: flex; }

  #stage-top {
    flex: 0 0 auto; padding: 10px 14px 8px;
    display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center;
    background: linear-gradient(#14161c, #0c0d10); border-bottom: 1px solid var(--line);
  }
  #stage-top h2 { margin: 0; font-size: 16px; font-weight: 700; flex: 1 1 200px; }
  #stage-top .blurb { width: 100%; color: var(--muted); font-size: 13px; margin: 0; }
  #stage-top .pill {
    font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
  }
  #stage-top .pill.miss { background: rgba(255,107,90,0.2); color: var(--bad); }
  #stage-top .pill.hit { background: rgba(61,214,140,0.18); color: var(--ok); }

  #viz-wrap {
    flex: 1 1 auto; min-height: 0;
    display: flex; align-items: center; justify-content: center;
    background: #000; cursor: crosshair; padding: 8px;
  }
  #viz-frame {
    position: relative; max-width: 100%; max-height: 100%;
    display: inline-block; line-height: 0;
  }
  #spec {
    max-width: min(100vw - 24px, 100%);
    max-height: calc(100vh - 160px);
    width: auto; height: auto;
    object-fit: contain; display: block; user-select: none;
  }
  #overlay {
    position: absolute; left: 0; top: 0; right: 0; bottom: 0;
    pointer-events: none;
  }
  #playhead {
    position: absolute; top: 0; bottom: 0; width: 3px;
    background: #fff;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.85), 0 0 14px rgba(255,255,255,0.55);
    left: 0%; transform: translateX(-1px);
  }
  #viz-label {
    position: absolute; left: 10px; top: 10px; z-index: 2;
    background: rgba(0,0,0,0.65); color: #fff; font-size: 12px; font-weight: 600;
    padding: 4px 8px; border-radius: 4px; pointer-events: none;
  }

  #transport {
    flex: 0 0 auto; padding: 10px 14px 14px;
    background: #101218; border-top: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  }
  #transport button, #stage-top button {
    background: var(--panel); color: var(--text); border: 1px solid var(--line);
    border-radius: 6px; padding: 8px 12px; font: inherit; font-size: 13px; cursor: pointer;
  }
  #transport button:hover, #stage-top button:hover { border-color: #5a6578; }
  #transport button.active { border-color: var(--accent); color: var(--accent); }
  #clock { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 13px; min-width: 9rem; }
  #help {
    width: 100%; color: var(--muted); font-size: 11px; margin-top: 4px;
  }
  #help kbd {
    background: #1c2030; border: 1px solid var(--line); border-radius: 3px;
    padding: 1px 5px; font-family: ui-monospace, monospace; color: var(--text);
  }
</style>
</head>
<body>
<div id="picker">
  <h1>Spectrogram player</h1>
  <p class="sub">Pick a clip with the mouse or ↑/↓, then press <kbd>Enter</kbd>.
  The spectrogram fills the screen — press <kbd>Space</kbd> to play, and the white line
  follows the sound. Green = truth, magenta = our guess.</p>
  <div class="keys">
    <span><kbd>↑</kbd><kbd>↓</kbd> move</span>
    <span><kbd>Enter</kbd> open</span>
    <span><kbd>Space</kbd> play/pause</span>
    <span><kbd>M</kbd>/<kbd>S</kbd> mix / song</span>
    <span><kbd>Esc</kbd> back to list</span>
  </div>
  __WEAK__
  <div class="filters">
    <select id="f-outcome">
      <option value="all">Everything</option>
      <option value="failure">Only misses</option>
      <option value="success">Only hits</option>
    </select>
    <select id="f-stem">
      <option value="all">All track types</option>
      <option value="acappella">Vocals</option>
      <option value="instrumental">Instrumental</option>
      <option value="regular">Full track</option>
    </select>
    <input id="f-q" type="search" placeholder="Search song name…"/>
  </div>
  __LIST__
</div>

<div id="stage">
  <div id="stage-top">
    <span id="badge" class="pill miss">Miss</span>
    <h2 id="title">—</h2>
    <button type="button" id="bBack">← List <kbd>Esc</kbd></button>
    <p class="blurb" id="blurb"></p>
  </div>
  <div id="viz-wrap">
    <div id="viz-frame">
      <div id="viz-label">DJ mix</div>
      <img id="spec" alt="spectrogram"/>
      <div id="overlay"><div id="playhead"></div></div>
    </div>
  </div>
  <div id="transport">
    <button type="button" id="bPlay">▶ Play</button>
    <button type="button" id="bMix" class="active">Mix</button>
    <button type="button" id="bSrc">Song</button>
    <button type="button" id="bPrev">‹ Prev</button>
    <button type="button" id="bNext">Next ›</button>
    <span id="clock">0:00 / 0:00</span>
    <div id="help">
      <kbd>Space</kbd> play/pause ·
      <kbd>←</kbd><kbd>→</kbd> seek 2s ·
      <kbd>M</kbd> mix · <kbd>S</kbd> original song ·
      click spectrogram to jump ·
      <kbd>Esc</kbd> back
    </div>
  </div>
</div>

<script>
const ITEMS = __ITEMS__;
let visible = [];
let focusIdx = 0;   // index into visible[]
let stageIdx = 0;   // index into ITEMS when on stage
let mode = "mix";   // mix | src
const audio = new Audio();
audio.preload = "auto";

const $ = id => document.getElementById(id);
const buttons = () => [...document.querySelectorAll(".clip-btn:not(.hidden)")];

function fmt(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60);
  return m + ":" + String(Math.floor(s % 60)).padStart(2, "0");
}

function applyFilters() {
  const outcome = $("f-outcome").value;
  const stem = $("f-stem").value;
  const q = $("f-q").value.trim().toLowerCase();
  document.querySelectorAll(".clip-btn").forEach(btn => {
    const okO = outcome === "all" || btn.dataset.outcome === outcome;
    const okS = stem === "all" || btn.dataset.stem === stem;
    const okQ = !q || (btn.dataset.search || "").includes(q);
    btn.classList.toggle("hidden", !(okO && okS && okQ));
  });
  document.querySelectorAll(".group").forEach(g => {
    const any = [...g.querySelectorAll(".clip-btn")].some(b => !b.classList.contains("hidden"));
    g.classList.toggle("hidden", !any);
  });
  visible = buttons();
  if (!visible.length) { focusIdx = 0; return; }
  focusIdx = Math.min(focusIdx, visible.length - 1);
  setFocus(focusIdx);
}

function setFocus(i) {
  const bs = buttons();
  if (!bs.length) return;
  focusIdx = Math.max(0, Math.min(i, bs.length - 1));
  bs.forEach(b => b.classList.remove("focused"));
  const el = bs[focusIdx];
  el.classList.add("focused");
  el.focus({ preventScroll: false });
  el.scrollIntoView({ block: "nearest" });
}

function openStage(itemIndex, autoplay) {
  stageIdx = itemIndex;
  document.body.classList.add("playing-mode");
  showItem(autoplay);
}

function closeStage() {
  pause();
  document.body.classList.remove("playing-mode");
  // restore focus on matching button
  const bs = buttons();
  const j = bs.findIndex(b => Number(b.dataset.idx) === stageIdx);
  if (j >= 0) setFocus(j);
}

function item() { return ITEMS[stageIdx]; }

function showItem(autoplay) {
  const it = item();
  pause();
  $("badge").textContent = it.success ? "Hit" : "Miss";
  $("badge").className = "pill " + (it.success ? "hit" : "miss");
  $("title").textContent = it.name;
  $("blurb").textContent = it.blurb;
  setMode(mode, /*keepTime*/ false);
  if (autoplay) {
    audio.oncanplay = () => { audio.oncanplay = null; play(); };
  }
}

function setMode(m, keepTime) {
  mode = m;
  const it = item();
  const t = keepTime ? audio.currentTime : 0;
  const hasSrc = !!(it.src_audio && it.src_img);
  $("bSrc").disabled = !hasSrc;
  $("bMix").classList.toggle("active", mode === "mix");
  $("bSrc").classList.toggle("active", mode === "src");
  if (mode === "src" && !hasSrc) mode = "mix";
  const img = mode === "src" ? it.src_img : it.mix_img;
  const clip = mode === "src" ? it.src_audio : it.mix_audio;
  $("spec").src = img;
  $("viz-label").textContent = mode === "src"
    ? "Original song — where inside the track"
    : "DJ mix — when it plays in the set";
  const wasPlaying = !audio.paused;
  audio.src = clip || "";
  audio.currentTime = 0;
  // after metadata, restore relative position if keepTime
  audio.onloadedmetadata = () => {
    if (keepTime && audio.duration) {
      audio.currentTime = Math.min(t, audio.duration - 0.05);
    }
    drawHead();
    if (wasPlaying && keepTime) play();
  };
  drawHead();
}

function play() {
  if (!audio.src) return;
  audio.play().catch(() => {});
  $("bPlay").textContent = "❚❚ Pause";
}
function pause() {
  audio.pause();
  $("bPlay").textContent = "▶ Play";
}
function toggle() { audio.paused ? play() : pause(); }

function drawHead() {
  const dur = audio.duration || item()[mode === "src" ? "ref_dur_s" : "mix_dur_s"] || 1;
  const t = audio.currentTime || 0;
  const pct = Math.max(0, Math.min(100, (t / dur) * 100));
  $("playhead").style.left = pct + "%";
  $("clock").textContent = fmt(t) + " / " + fmt(dur);
}

audio.ontimeupdate = drawHead;
audio.onended = () => pause();
window.addEventListener("resize", drawHead);
$("spec").addEventListener("load", drawHead);

function seekFrac(frac) {
  const dur = audio.duration || item()[mode === "src" ? "ref_dur_s" : "mix_dur_s"] || 0;
  if (!dur) return;
  audio.currentTime = Math.max(0, Math.min(dur - 0.05, frac * dur));
  drawHead();
}

$("viz-wrap").addEventListener("click", e => {
  const img = $("spec");
  const r = img.getBoundingClientRect();
  if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
  seekFrac((e.clientX - r.left) / r.width);
});

function go(d) {
  let i = stageIdx + d;
  i = Math.max(0, Math.min(ITEMS.length - 1, i));
  stageIdx = i;
  showItem(false);
}

$("bPlay").onclick = toggle;
$("bMix").onclick = () => setMode("mix", true);
$("bSrc").onclick = () => setMode("src", true);
$("bPrev").onclick = () => go(-1);
$("bNext").onclick = () => go(1);
$("bBack").onclick = closeStage;

document.querySelectorAll(".clip-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    openStage(Number(btn.dataset.idx), true);
  });
  btn.addEventListener("focus", () => {
    const bs = buttons();
    const j = bs.indexOf(btn);
    if (j >= 0) focusIdx = j;
    bs.forEach(b => b.classList.remove("focused"));
    btn.classList.add("focused");
  });
});

["f-outcome","f-stem","f-q"].forEach(id => $(id).addEventListener("input", applyFilters));

window.addEventListener("keydown", e => {
  const tag = (e.target && e.target.tagName) || "";
  const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  const onStage = document.body.classList.contains("playing-mode");

  if (onStage) {
    if (e.key === "Escape") { e.preventDefault(); closeStage(); return; }
    if (e.key === " " || e.code === "Space") { e.preventDefault(); toggle(); return; }
    if (e.key === "m" || e.key === "M") { e.preventDefault(); setMode("mix", true); return; }
    if (e.key === "s" || e.key === "S") { e.preventDefault(); setMode("src", true); return; }
    if (e.key === "ArrowLeft") { e.preventDefault(); audio.currentTime = Math.max(0, audio.currentTime - 2); drawHead(); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); audio.currentTime = Math.min((audio.duration||0)-0.05, audio.currentTime + 2); drawHead(); return; }
    if (e.key === "ArrowUp" || e.key === "j") { e.preventDefault(); go(-1); return; }
    if (e.key === "ArrowDown" || e.key === "k") { e.preventDefault(); go(1); return; }
    return;
  }

  if (typing) return;
  if (e.key === "ArrowDown") { e.preventDefault(); setFocus(focusIdx + 1); return; }
  if (e.key === "ArrowUp") { e.preventDefault(); setFocus(focusIdx - 1); return; }
  if (e.key === "Enter") {
    e.preventDefault();
    const bs = buttons();
    if (!bs.length) return;
    openStage(Number(bs[focusIdx].dataset.idx), true);
  }
});

applyFilters();
</script>
</body>
</html>
"""


def write_player(
    out_html: Path,
    items: Sequence[dict[str, Any]],
    *,
    list_html: str,
    weak_html: str = "",
) -> None:
    text = (
        _TEMPLATE.replace("__ITEMS__", json.dumps(list(items)))
        .replace("__LIST__", list_html)
        .replace("__WEAK__", weak_html)
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(text)
    out_html.with_suffix(".json").write_text(
        json.dumps({"items": list(items)}, indent=2)
    )

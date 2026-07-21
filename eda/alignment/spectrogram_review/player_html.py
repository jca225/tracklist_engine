"""Fullscreen dual spectrogram player — mix | song, synced playheads + audio."""

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
    --bg: #0c0d10; --panel: #16181e; --text: #f2f2f2; --muted: #9aa0a6;
    --line: #2a2f3a; --ok: #3dd68c; --bad: #ff6b5a; --accent: #7cb3ff; --focus: #ffd166;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
  body.playing-mode { overflow: hidden; }

  #picker { min-height: 100%; padding: 20px 22px 48px; max-width: 960px; margin: 0 auto; }
  #picker h1 { margin: 0 0 6px; font-size: 22px; font-weight: 700; }
  #picker .sub { color: var(--muted); font-size: 14px; margin: 0 0 16px; max-width: 44rem; line-height: 1.45; }
  .keys { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; font-size: 12px; color: var(--muted); }
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
  .group.fail h2 { color: var(--bad); } .group.ok h2 { color: var(--ok); }
  .group .blurb { color: var(--muted); font-size: 12px; margin: 0 0 8px; }
  .list { display: flex; flex-direction: column; gap: 6px; }
  .clip-btn {
    text-align: left; width: 100%; background: var(--panel); color: var(--text);
    border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
    cursor: pointer; font: inherit;
    display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: center;
  }
  .clip-btn:hover { border-color: #4a5568; }
  .clip-btn.focused, .clip-btn:focus {
    outline: none; border-color: var(--focus);
    box-shadow: 0 0 0 2px rgba(255,209,102,0.25);
  }
  .clip-btn .badge { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
  .clip-btn .badge.miss { background: rgba(255,107,90,0.2); color: var(--bad); }
  .clip-btn .badge.hit { background: rgba(61,214,140,0.18); color: var(--ok); }
  .clip-btn .name { font-weight: 600; font-size: 14px; }
  .clip-btn .meta { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .clip-btn .ableton { color: #c7d2fe; font-size: 12px; margin-top: 3px; }
  .clip-btn .hint { color: var(--muted); font-size: 11px; white-space: nowrap; }
  .clip-btn.hidden, .group.hidden { display: none; }
  .weak {
    margin: 0 0 18px; padding: 12px 14px; background: #1c1810;
    border: 1px solid #5c4a1f; border-radius: 8px; font-size: 13px;
  }
  .weak strong { color: #ffd166; }
  #filewarn {
    display: none; margin: 0 0 14px; padding: 10px 12px;
    background: #3a1515; border: 1px solid #7a3030; border-radius: 8px; font-size: 13px;
  }
  #filewarn.show { display: block; }

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
  #stage-top .blurb, #stage-top .ableton-line {
    width: 100%; color: var(--muted); font-size: 13px; margin: 0;
  }
  #stage-top .ableton-line { color: #c7d2fe; font-weight: 600; }
  #stage-top .pill { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
  #stage-top .pill.miss { background: rgba(255,107,90,0.2); color: var(--bad); }
  #stage-top .pill.hit { background: rgba(61,214,140,0.18); color: var(--ok); }
  #stage-top button, #transport button {
    background: var(--panel); color: var(--text); border: 1px solid var(--line);
    border-radius: 6px; padding: 8px 12px; font: inherit; font-size: 13px; cursor: pointer;
  }
  #transport button.active { border-color: var(--accent); color: var(--accent); }

  #viz-wrap {
    flex: 1 1 auto; min-height: 0;
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    padding: 10px; background: #000;
  }
  @media (max-width: 800px) { #viz-wrap { grid-template-columns: 1fr; } }
  .panel {
    position: relative; min-height: 0; min-width: 0;
    display: flex; flex-direction: column; background: #0a0a0c;
    border: 1px solid var(--line); border-radius: 8px; overflow: hidden;
  }
  .panel .lab {
    flex: 0 0 auto; padding: 8px 10px; font-size: 12px; font-weight: 700;
    color: var(--muted); border-bottom: 1px solid var(--line); background: #12141a;
  }
  .panel .lab b { color: var(--text); }
  .panel .canvas {
    flex: 1 1 auto; position: relative; min-height: 0;
    display: flex; align-items: center; justify-content: center; cursor: crosshair;
  }
  .panel img {
    max-width: 100%; max-height: 100%; width: auto; height: auto;
    object-fit: contain; display: block; user-select: none;
  }
  .panel .frame { position: relative; line-height: 0; max-width: 100%; max-height: 100%; }
  .panel .overlay { position: absolute; inset: 0; pointer-events: none; }
  .panel .playhead {
    position: absolute; top: 0; bottom: 0; width: 3px; left: 0%;
    background: #fff;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.85), 0 0 14px rgba(255,255,255,0.55);
    transform: translateX(-1px);
  }

  #transport {
    flex: 0 0 auto; padding: 10px 14px 14px;
    background: #101218; border-top: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  }
  #clock { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 13px; min-width: 9rem; }
  #status { color: var(--bad); font-size: 12px; min-height: 1.2em; width: 100%; }
  #status.ok { color: var(--ok); }
  #native-audio {
    width: 100%; display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  }
  #native-audio label {
    font-size: 11px; color: var(--muted); display: flex; gap: 6px; align-items: center; flex: 1;
    min-width: 220px;
  }
  #native-audio audio { height: 32px; width: 100%; }
  #help { width: 100%; color: var(--muted); font-size: 11px; }
  #help kbd {
    background: #1c2030; border: 1px solid var(--line); border-radius: 3px;
    padding: 1px 5px; font-family: ui-monospace, monospace; color: var(--text);
  }
</style>
</head>
<body>
<div id="picker">
  <h1>Spectrogram player <span style="font-size:12px;color:#7cb3ff;font-weight:600">audio-v5</span></h1>
  <p class="sub">Pick a clip (↑/↓ then <kbd>Enter</kbd> or click). Mix and original song
  sit side by side; play moves both white lines. Each row shows the <b>Ableton track</b>
  and mix timestamp from the canonical labeling session.</p>
  <div id="filewarn">
    You’re on <code>file://</code> — audio won’t work. Open via the local server instead
    (port 8765).
  </div>
  <div class="keys">
    <span><kbd>↑</kbd><kbd>↓</kbd> move</span>
    <span><kbd>Enter</kbd> open</span>
    <span><kbd>Space</kbd> play/pause</span>
    <span><kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> hear mix / song / both</span>
    <span><kbd>Esc</kbd> list</span>
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
    <input id="f-q" type="search" placeholder="Search song / Ableton track…"/>
  </div>
  __LIST__
</div>

<div id="stage">
  <div id="stage-top">
    <span id="badge" class="pill miss">Miss</span>
    <h2 id="title">—</h2>
    <button type="button" id="bBack">← List</button>
    <p class="ableton-line" id="ableton"></p>
    <p class="blurb" id="blurb"></p>
  </div>
  <div id="viz-wrap">
    <div class="panel" id="panel-mix" data-which="mix">
      <div class="lab"><b>DJ mix</b> — when it plays in the set</div>
      <div class="canvas">
        <div class="frame">
          <img id="img-mix" alt="mix spectrogram"/>
          <div class="overlay"><div class="playhead" id="head-mix"></div></div>
        </div>
      </div>
    </div>
    <div class="panel" id="panel-src" data-which="src">
      <div class="lab"><b>Original song</b> — where inside the track</div>
      <div class="canvas">
        <div class="frame">
          <img id="img-src" alt="song spectrogram"/>
          <div class="overlay"><div class="playhead" id="head-src"></div></div>
        </div>
      </div>
    </div>
  </div>
  <div id="transport">
    <button type="button" id="bPlay">▶ Play</button>
    <button type="button" id="bHearMix" class="active">Hear mix</button>
    <button type="button" id="bHearSrc">Hear song</button>
    <button type="button" id="bHearBoth">Hear both</button>
    <button type="button" id="bPrev">‹ Prev</button>
    <button type="button" id="bNext">Next ›</button>
    <span id="clock">0:00 / 0:00</span>
    <div id="status"></div>
    <div id="native-audio">
      <label>Mix backup <audio id="fb-mix" controls preload="auto"></audio></label>
      <label>Song backup <audio id="fb-src" controls preload="auto"></audio></label>
    </div>
    <div id="help">
      Prefetched Web Audio · if silent, use the Mix backup slider ·
      <kbd>Space</kbd> · <kbd>d</kbd> diag · <kbd>Esc</kbd> back
    </div>
  </div>
</div>

<script>
const ITEMS = __ITEMS__;
let focusIdx = 0;
let stageIdx = 0;
let hear = "mix";
let loadToken = 0;

const $ = id => document.getElementById(id);
const buttons = () => [...document.querySelectorAll(".clip-btn:not(.hidden)")];
const absUrl = p => p ? new URL(p, location.href).href : "";
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

if (location.protocol === "file:") $("filewarn").classList.add("show");

function fmt(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60);
  return m + ":" + String(Math.floor(s % 60)).padStart(2, "0");
}

/**
 * Web Audio engine — ported from mashup_demo/studio/studio.html.
 * Do NOT create AudioContext outside a user gesture: on Safari a context
 * born early is permanently muted even after resume().
 * Fetch compressed bytes anytime; create ctx + decode on first play/open click.
 */
const eng = {
  ctx: null, master: null, mixGain: null, srcGain: null,
  mixBuf: null, srcBuf: null, dur: 1,
  bytesCache: new Map(),
  playing: false, startCtx: 0, startPos: 0, nodes: [],

  // Call this FIRST in any click/keydown handler — sync, before any await.
  ensureCtxSync() {
    if (this.ctx) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.95;
    this.master.connect(this.ctx.destination);
    this.mixGain = this.ctx.createGain();
    this.srcGain = this.ctx.createGain();
    this.mixGain.connect(this.master);
    this.srcGain.connect(this.master);
  },

  async resume() {
    if (!this.ctx) return;
    if (this.ctx.state !== "running") {
      try { await this.ctx.resume(); } catch (e) {}
    }
  },

  async fetchBytes(url) {
    if (this.bytesCache.has(url)) return this.bytesCache.get(url);
    const r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status + " for " + url.split("/").pop());
    const buf = await r.arrayBuffer();
    this.bytesCache.set(url, buf);
    return buf;
  },

  async decodeUrl(url) {
    const bytes = await this.fetchBytes(url);
    // decodeAudioData detaches the buffer — decode from a copy, keep cache.
    return await this.ctx.decodeAudioData(bytes.slice(0));
  },

  async loadClip(it) {
    const mixUrl = absUrl(it.mix_audio);
    if (!mixUrl) throw new Error("no mix_audio on item");
    const srcUrl = absUrl(it.src_audio);
    this.mixBuf = await this.decodeUrl(mixUrl);
    this.srcBuf = srcUrl ? await this.decodeUrl(srcUrl) : null;
    this.dur = this.mixBuf.duration || Number(it.mix_dur_s) || 1;
    this.startPos = 0;
  },

  applyHear() {
    if (!this.mixGain) return;
    if (hear === "mix") {
      this.mixGain.gain.value = 1; this.srcGain.gain.value = 0;
    } else if (hear === "src") {
      this.mixGain.gain.value = 0; this.srcGain.gain.value = 1;
    } else {
      this.mixGain.gain.value = 0.75; this.srcGain.gain.value = 0.75;
    }
  },

  _stop() {
    this.nodes.forEach(n => { try { n.stop(); } catch (e) {} });
    this.nodes = [];
  },

  pos() {
    if (!this.playing || !this.ctx) return this.startPos;
    return clamp(this.startPos + (this.ctx.currentTime - this.startCtx), 0, this.dur);
  },

  schedule(pos) {
    if (!this.ctx || !this.mixBuf) return;
    this._stop();
    this.applyHear();
    pos = clamp(pos, 0, Math.max(0, this.dur - 0.02));
    const t0 = this.ctx.currentTime + 0.03;

    const m = this.ctx.createBufferSource();
    m.buffer = this.mixBuf;
    m.connect(this.mixGain);
    const mRemain = this.mixBuf.duration - pos;
    if (mRemain > 0.01) m.start(t0, pos, mRemain);
    this.nodes.push(m);

    if (this.srcBuf) {
      const frac = this.dur > 0 ? pos / this.dur : 0;
      const srcPos = frac * this.srcBuf.duration;
      const s = this.ctx.createBufferSource();
      s.buffer = this.srcBuf;
      s.connect(this.srcGain);
      const sRemain = this.srcBuf.duration - srcPos;
      if (sRemain > 0.01) s.start(t0, srcPos, sRemain);
      this.nodes.push(s);
    }

    this.startCtx = t0;
    this.startPos = pos;
    this.playing = true;
    $("bPlay").textContent = "❚❚ Pause";
  },

  pause() {
    if (!this.playing) return;
    const p = this.pos();
    this._stop();
    this.playing = false;
    this.startPos = p;
    $("bPlay").textContent = "▶ Play";
  },

  seekFrac(frac) {
    const p = clamp(frac, 0, 1) * this.dur;
    if (this.playing) this.schedule(p);
    else { this.startPos = p; drawHeads(); }
  },

  diag() {
    const c = this.ctx;
    return "ctx=" + (c ? c.state : "none") +
      " sr=" + (c ? c.sampleRate : "-") +
      " mix=" + (this.mixBuf ? this.mixBuf.duration.toFixed(1) + "s" : "nil") +
      " src=" + (this.srcBuf ? this.srcBuf.duration.toFixed(1) + "s" : "nil") +
      " gain=" + (this.master ? this.master.gain.value : "-") +
      " hear=" + hear +
      " playing=" + this.playing +
      " pos=" + this.pos().toFixed(2);
  }
};

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
  const bs = buttons();
  if (!bs.length) { focusIdx = 0; return; }
  focusIdx = Math.min(focusIdx, bs.length - 1);
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

function item() { return ITEMS[stageIdx]; }

function paintMeta(it) {
  $("badge").textContent = it.success ? "Hit" : "Miss";
  $("badge").className = "pill " + (it.success ? "hit" : "miss");
  $("title").textContent = it.name;
  $("blurb").textContent = it.blurb;
  $("ableton").textContent = it.ableton_line || "";
  $("img-mix").src = it.mix_img || "";
  $("img-src").src = it.src_img || it.mix_img || "";
  const hasSrc = !!(it.src_audio && it.src_img);
  $("bHearSrc").disabled = !hasSrc;
  $("bHearBoth").disabled = !hasSrc;
  if (!hasSrc && hear !== "mix") hear = "mix";
  paintHearButtons();
}

function paintHearButtons() {
  ["bHearMix","bHearSrc","bHearBoth"].forEach(id => $(id).classList.remove("active"));
  $({mix:"bHearMix", src:"bHearSrc", both:"bHearBoth"}[hear]).classList.add("active");
}

function setStatus(msg, ok) {
  $("status").textContent = msg || "";
  $("status").className = ok ? "ok" : "";
}

function drawHeads() {
  const dur = eng.dur || (item() && item().mix_dur_s) || 1;
  const p = clamp(eng.pos() / dur, 0, 1) * 100;
  $("head-mix").style.left = p + "%";
  $("head-src").style.left = p + "%";
  $("clock").textContent = fmt(eng.pos()) + " / " + fmt(dur);
}

function wireFallback(it) {
  const mixUrl = absUrl(it.mix_audio);
  const srcUrl = absUrl(it.src_audio);
  const fbMix = $("fb-mix");
  const fbSrc = $("fb-src");
  if (mixUrl) fbMix.src = mixUrl;
  if (srcUrl) { fbSrc.src = srcUrl; fbSrc.parentElement.style.display = ""; }
  else { fbSrc.removeAttribute("src"); fbSrc.parentElement.style.display = "none"; }
}

async function openStage(itemIndex, autoplay) {
  // CRITICAL: birth AudioContext inside this gesture before any await.
  eng.ensureCtxSync();
  eng.pause();
  stageIdx = itemIndex;
  document.body.classList.add("playing-mode");
  const it = item();
  paintMeta(it);
  wireFallback(it);
  drawHeads();
  const my = ++loadToken;
  setStatus("Loading audio…", true);
  try {
    await eng.resume();
    await eng.loadClip(it); // bytes usually already prefetched
    if (my !== loadToken) return;
    setStatus(eng.playing || autoplay ? "Playing (Web Audio)" : "Ready", true);
    drawHeads();
    if (autoplay) {
      eng.schedule(0);
      setStatus("Playing — if silent, hit ▶ on Mix backup below", true);
    }
  } catch (e) {
    if (my !== loadToken) return;
    setStatus("Web Audio failed: " + (e && e.message ? e.message : e) + " — use Mix backup ▶");
    console.error(e);
    // Native fallback still works from the same gesture's click chain if user hits ▶
  }
}

function closeStage() {
  eng.pause();
  document.body.classList.remove("playing-mode");
  setStatus("");
  const bs = buttons();
  const j = bs.findIndex(b => Number(b.dataset.idx) === stageIdx);
  if (j >= 0) setFocus(j);
}

async function showItem(autoplay) {
  eng.ensureCtxSync();
  eng.pause();
  const it = item();
  paintMeta(it);
  wireFallback(it);
  const my = ++loadToken;
  setStatus("Loading audio…", true);
  try {
    await eng.resume();
    await eng.loadClip(it);
    if (my !== loadToken) return;
    setStatus("");
    drawHeads();
    if (autoplay) {
      eng.schedule(0);
      setStatus("Playing — if silent, hit ▶ on Mix backup below", true);
    }
  } catch (e) {
    if (my !== loadToken) return;
    setStatus("Web Audio failed: " + (e && e.message ? e.message : e) + " — use Mix backup ▶");
  }
}

async function toggle() {
  eng.ensureCtxSync();
  await eng.resume();
  if (eng.playing) {
    eng.pause();
  } else {
    if (!eng.mixBuf) {
      try {
        setStatus("Loading audio…", true);
        await eng.loadClip(item());
        setStatus("");
      } catch (e) {
        setStatus("Audio failed: " + e.message);
        return;
      }
    }
    eng.schedule(eng.startPos >= eng.dur - 0.05 ? 0 : eng.startPos);
  }
}

function go(d) {
  stageIdx = Math.max(0, Math.min(ITEMS.length - 1, stageIdx + d));
  showItem(true);
}

function setHear(h) {
  hear = h;
  paintHearButtons();
  eng.applyHear();
}

$("bPlay").onclick = () => { toggle(); };
$("bHearMix").onclick = () => setHear("mix");
$("bHearSrc").onclick = () => setHear("src");
$("bHearBoth").onclick = () => setHear("both");
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

function bindSeek(panelId) {
  $(panelId).querySelector(".canvas").addEventListener("click", e => {
    const img = $(panelId).querySelector("img");
    const r = img.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right) return;
    eng.seekFrac((e.clientX - r.left) / r.width);
  });
}
bindSeek("panel-mix");
bindSeek("panel-src");

window.addEventListener("keydown", e => {
  const tag = (e.target && e.target.tagName) || "";
  const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  const onStage = document.body.classList.contains("playing-mode");

  if (e.key === "d" && !typing) {
    e.preventDefault();
    alert(eng.diag());
    return;
  }

  if (onStage) {
    if (e.key === "Escape") { e.preventDefault(); closeStage(); return; }
    if (e.key === " " || e.code === "Space") {
      e.preventDefault(); toggle(); return;
    }
    if (e.key === "1") { e.preventDefault(); setHear("mix"); return; }
    if (e.key === "2") { e.preventDefault(); setHear("src"); return; }
    if (e.key === "3") { e.preventDefault(); setHear("both"); return; }
    if (e.key === "ArrowLeft") { e.preventDefault(); eng.seekFrac(eng.pos() / eng.dur - 2 / eng.dur); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); eng.seekFrac(eng.pos() / eng.dur + 2 / eng.dur); return; }
    if (e.key === "ArrowUp" || e.key === "j") { e.preventDefault(); go(-1); return; }
    if (e.key === "ArrowDown" || e.key === "k") { e.preventDefault(); go(1); return; }
    return;
  }
  if (typing) return;
  if (e.key === "ArrowDown") { e.preventDefault(); setFocus(focusIdx + 1); return; }
  if (e.key === "ArrowUp") { e.preventDefault(); setFocus(focusIdx - 1); return; }
  if (e.key === "Enter") {
    e.preventDefault();
    // Birth ctx on the keydown gesture before async work.
    eng.ensureCtxSync();
    const bs = buttons();
    if (bs.length) openStage(Number(bs[focusIdx].dataset.idx), true);
  }
});

// Safari: context suspends on tab-switch/idle — re-resume on every gesture.
(() => {
  const ensure = () => {
    if (eng.ctx && eng.ctx.state !== "running") eng.ctx.resume().catch(() => {});
  };
  document.addEventListener("pointerdown", ensure);
  document.addEventListener("keydown", ensure);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) ensure(); });
})();

function tick() {
  if (eng.playing && eng.pos() >= eng.dur - 0.03) {
    eng.pause();
    eng.startPos = 0;
  }
  drawHeads();
  requestAnimationFrame(tick);
}
tick();
applyFilters();

// Prefetch compressed bytes up front (mashup studio pattern). Click then only
// births AudioContext + decode — no network after the gesture.
(async () => {
  const urls = new Set();
  for (const it of ITEMS) {
    if (it.mix_audio) urls.add(absUrl(it.mix_audio));
    if (it.src_audio) urls.add(absUrl(it.src_audio));
  }
  setStatus("Prefetching " + urls.size + " audio files…", true);
  let ok = 0, bad = 0;
  await Promise.all([...urls].map(async u => {
    try { await eng.fetchBytes(u); ok++; }
    catch (e) { bad++; console.warn("prefetch", u, e); }
  }));
  setStatus(
    bad ? ("Prefetch done (" + ok + " ok, " + bad + " failed) — click a clip")
        : ("Audio ready (" + ok + " clips) — click a clip"),
    true
  );
})();
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
    # Side-car next to the HTML only — never clobber a fuller index.json from render.
    out_html.with_name(out_html.stem + "_items.json").write_text(
        json.dumps({"items": list(items)}, indent=2)
    )

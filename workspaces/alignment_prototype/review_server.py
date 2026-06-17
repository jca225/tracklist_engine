#!/usr/bin/env python3
"""Unified review UI — two tabs: (1) stem-source winner, (2) fiber confirmation.

One local site, two jobs the project needs a human in the loop for:

  • **Stem winner** — the fiber audit proved we can't build reliable structure on
    noisy separated stems ([[project_fibers]]); the fix is to outsource clean
    acappellas/instrumentals and PICK the best source per layer. Lists the Demucs
    baseline + each downloaded candidate (`stems/<song>/candidates/<layer>/cand*`)
    side by side, plays a 30s chunk, flags preview clips by length, records the
    winner -> out/discern/picks.jsonl (the labels for the eventual quality ranker,
    [[project_official_stems_search]]).

  • **Fibers** — verify a track's self-repeat equivalence classes by ear
    ([[project_fibers]]). Plays each fiber's member segments grouped together so you
    confirm the repeats are the same content (robust to a singer delivering a
    section slightly differently) and FLAG anything wrongly merged. Flags ->
    out/fiber_review/flags.jsonl (the equivalence-class GT the fiber-aware
    objective needs). Diagonal sim shown (pooled cosine is fooled at 0.9+).

Supersedes the two single-purpose servers (discern_server.py / fiber_server.py).
Audio is cut on demand and cached. Single-user, localhost only.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.review_server \
        [--port 8800] [--open]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.path_decode import _ensure_feat  # noqa: E402
from workspaces.alignment_prototype.ref_fibers import (  # noqa: E402
    _diag_sim,
    compute_fibers,
    fiber_intervals,
)
from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP,
    SR,
    _STEM_FILE,
)

FPS = SR / HOP
ALIGNING = Path.home() / "aligning"
OUT = _REPO / "workspaces/alignment_prototype/out"
SNIP = OUT / "review/_chunks"
PICKS = OUT / "discern/picks.jsonl"
FLAGS = OUT / "fiber_review/flags.jsonl"
CHUNK_S = 30.0  # seconds served per stem source (from 25% in) for A/B listening
_DEFAULT_FEATURE = {
    "acappella": "hubert",
    "instrumental": "chroma",
    "regular": "chroma",
}

# one audio map for both tabs: id -> (src, start_s, end_s)
_audio_map: dict[str, tuple[str, float, float]] = {}
_lock = threading.Lock()


# ----------------------------------------------------------------------------- shared
def _register(src: str, start: float, end: float) -> str:
    sid = hashlib.md5(f"{src}{start:.3f}{end:.3f}".encode()).hexdigest()[:16]
    with _lock:
        _audio_map[sid] = (src, start, end)
    return sid


def _ffprobe_dur(path: str) -> float:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            capture_output=True,
            text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _cut(src: str, s: float, e: float) -> bytes | None:
    SNIP.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{src}{s:.3f}{e:.3f}".encode()).hexdigest()[:16]
    out = SNIP / f"{key}.mp3"
    if not out.is_file():
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{s:.3f}",
            "-t",
            f"{max(0.3, e - s):.3f}",
            "-i",
            src,
            # full-bandwidth stereo: quality judgement (and fiber timbre) needs the
            # top octave — a 22.05 kHz/mono/56k cut sounds lowpassed/"filtered".
            "-ac",
            "2",
            "-ar",
            "44100",
            "-b:a",
            "192k",
            str(out),
        ]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None
    return out.read_bytes()


def _set_dir(set_id: str) -> Path | None:
    hits = sorted(ALIGNING.glob(f"{set_id}__*"))
    return hits[0] if hits else None


def list_sets() -> list[dict]:
    out = []
    for d in sorted(ALIGNING.glob("*__*")):
        if (d / "manifest.json").is_file() or (d / "stems").is_dir():
            out.append({"id": d.name.split("__", 1)[0], "name": d.name})
    return out


# ----------------------------------------------------------------------------- tab 1: stem winner
def _candidate_files(stem_dir: Path, layer: str) -> list[Path]:
    base = stem_dir / "candidates"
    out: list[Path] = []
    for d in (base / layer, base):
        if d.is_dir():
            out += sorted(d.glob("cand*.m4a"))
    seen, uniq = set(), []
    for f in out:
        if f.name not in seen:
            seen.add(f.name)
            uniq.append(f)
    return uniq


def build_layers(set_id: str, only: str | None) -> dict:
    sd = _set_dir(set_id)
    if sd is None:
        return {"error": f"no set {set_id}"}
    stems = sd / "stems"
    if not stems.is_dir():
        return {"error": "no stems/ dir"}
    layers = []
    for folder in sorted(p for p in stems.iterdir() if p.is_dir()):
        for layer in ("vocals", "instrumental"):
            if only and layer != only:
                continue
            cands = _candidate_files(folder, layer)
            if not cands:
                continue  # nothing to discern (only the Demucs baseline)
            baseline = folder / f"{layer}.flac"
            base_dur = _ffprobe_dur(str(baseline)) if baseline.is_file() else 0.0
            sources = []

            def _src(name, kind, dur, path, match):
                start = max(0.0, dur * 0.25)
                return {
                    "name": name,
                    "kind": kind,
                    "dur": round(dur, 1),
                    "match": match,
                    "audio": _register(str(path), start, start + CHUNK_S),
                }

            if baseline.is_file():
                sources.append(
                    _src("Demucs (baseline)", "demucs", base_dur, baseline, True)
                )
            for c in cands:
                d = _ffprobe_dur(str(c))
                match = base_dur > 0 and abs(d - base_dur) <= 5.0
                sources.append(_src(c.name[:54], "candidate", d, c, bool(match)))
            layers.append(
                {
                    "folder": folder.name,
                    "layer": layer,
                    "n_cand": len(cands),
                    "sources": sources,
                }
            )
    return {"set": set_id, "layers": layers}


# ----------------------------------------------------------------------------- tab 2: fibers
def build_fibers(set_id, stem, feature, k, min_section, max_refs) -> dict:
    sd = _set_dir(set_id)
    if sd is None:
        return {"error": f"no set {set_id}"}
    mpath = sd / "manifest.json"
    if not mpath.is_file():
        return {"error": "no manifest.json (pull the set first)"}
    manifest = json.loads(mpath.read_text())
    feature = feature or _DEFAULT_FEATURE.get(stem, "chroma")
    refs, seen, n = [], set(), 0
    for tr in manifest["tracks"]:
        if n >= max_refs:
            break
        rid = str(tr.get("recording_id") or tr.get("track_id"))
        if rid in seen:
            continue
        sk = _STEM_FILE.get(stem)
        sp = (tr.get("stems") or {}).get(sk) if sk else tr.get("local_path")
        if not sp or not Path(sp).is_file():
            continue
        feat = np.load(_ensure_feat(sp, sp, feature, 9))
        labels, hz = compute_fibers(
            feat, FPS, k=k, min_section_s=min_section, audio_path=sp
        )
        ivs = fiber_intervals(labels, hz, min_len_s=min_section)
        by_lab: dict[int, list] = {}
        for s, e, lab in ivs:
            by_lab.setdefault(lab, []).append((s, e))
        fibers = []
        for lab, members in sorted(by_lab.items(), key=lambda kv: -len(kv[1])):
            if len(members) < 2:
                continue
            rs, re = max(members, key=lambda m: m[1] - m[0])
            ref_feat = np.ascontiguousarray(feat[:, int(rs * FPS) : int(re * FPS)])
            ms = []
            for s, e in members:
                seg = np.ascontiguousarray(feat[:, int(s * FPS) : int(e * FPS)])
                sim = _diag_sim(seg, ref_feat)
                ms.append(
                    {
                        "start": round(s, 1),
                        "end": round(e, 1),
                        "sim": round(float(sim), 3),
                        "audio": _register(sp, s, e),
                    }
                )
            fibers.append({"label": int(lab), "members": ms})
        if fibers:
            seen.add(rid)
            n += 1
            refs.append(
                {
                    "rid": rid,
                    "title": tr.get("title") or tr.get("name") or rid,
                    "fibers": fibers,
                }
            )
    return {"set": set_id, "stem": stem, "feature": feature, "refs": refs}


# ----------------------------------------------------------------------------- page
PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Alignment review</title>
<style>
:root{
  --bg:#0f1117;--panel:#181b24;--card:#1f2330;--line:#2c3140;--ink:#e8eaf0;
  --dim:#9aa3b2;--accent:#5b8cff;--good:#3ecf8e;--warn:#ff7a90;--chip:#262b3a;
}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
  background:var(--bg);color:var(--ink);font-size:15px;line-height:1.45}
header{position:sticky;top:0;z-index:5;background:rgba(15,17,23,.92);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:.7rem 1.2rem}
.bar{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;max-width:74rem;margin:0 auto}
h1{font-size:1.05rem;margin:0;font-weight:650;letter-spacing:.2px}
.tabs{display:flex;gap:.3rem;background:var(--chip);padding:.25rem;border-radius:10px}
.tab{padding:.4rem .9rem;border-radius:8px;cursor:pointer;color:var(--dim);
  font-weight:550;border:0;background:transparent;font-size:.92rem}
.tab.on{background:var(--accent);color:#fff}
select,input[type=range]{accent-color:var(--accent)}
select{background:var(--card);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:.35rem .5rem;font-size:.9rem;max-width:22rem}
button.go{background:var(--accent);color:#fff;border:0;border-radius:8px;
  padding:.42rem 1rem;font-weight:600;cursor:pointer;font-size:.9rem}
button.go:hover{filter:brightness(1.08)}
main{max-width:74rem;margin:0 auto;padding:1.1rem 1.2rem 4rem}
.panel{display:none}.panel.on{display:block}
.toolbar{display:flex;gap:1rem;align-items:end;flex-wrap:wrap;margin-bottom:1rem;
  background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem}
.fld{display:flex;flex-direction:column;gap:.25rem}
.fld label{font-size:.72rem;color:var(--dim);text-transform:uppercase;letter-spacing:.4px}
#status,#fstatus{color:var(--dim);font-size:.85rem}
.note{color:var(--dim);font-size:.88rem;margin:.2rem 0 1rem;max-width:58rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:.9rem 1rem;margin:.8rem 0}
.card h3{margin:0 0 .2rem;font-size:1rem}.card .sub{color:var(--dim);font-size:.82rem;margin-bottom:.5rem}
.src{display:flex;align-items:center;gap:.7rem;padding:.4rem .5rem;border-radius:9px;
  border:1px solid transparent}
.src+.src{margin-top:.3rem}
.src:hover{background:var(--card)}
.src.demucs .nm{color:var(--dim)}
.src.clean{background:rgba(62,207,142,.14);border-color:var(--good)}
.src.keep{background:rgba(91,140,255,.1);border-color:var(--accent)}
.src.diff{background:rgba(255,122,144,.08);border-color:var(--warn);opacity:.7}
.nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.9rem}
.badge{font-size:.7rem;padding:.1rem .45rem;border-radius:999px;font-weight:600}
.badge.demucs{background:#33384a;color:var(--dim)}
.badge.cand{background:#2a3a5c;color:#9cc0ff}
.dur{font-variant:tabular-nums;color:var(--dim);font-size:.82rem;width:4.5rem;text-align:right}
.dur.bad{color:var(--warn)}
audio{height:2rem}
.marks{display:flex;gap:.25rem}
button.mk{background:var(--card);color:var(--dim);border:1px solid var(--line);
  border-radius:7px;padding:.28rem .55rem;cursor:pointer;font-size:.8rem;white-space:nowrap}
button.mk:hover{color:var(--ink)}
button.mclean.on{background:var(--good);color:#06231a;border-color:var(--good);font-weight:600}
button.mkeep.on{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
button.mdiff.on{background:var(--warn);color:#2a0710;border-color:var(--warn);font-weight:600}
.card .hd{display:flex;align-items:center;gap:.6rem;justify-content:space-between}
.card .hd .meta{min-width:0}
.tally{font-size:.78rem;color:var(--dim);font-variant:tabular-nums}
.tally.has{color:var(--good)}
button.unsure{background:transparent;color:var(--dim);border:1px solid var(--line);
  border-radius:8px;padding:.3rem .7rem;cursor:pointer;font-size:.82rem;white-space:nowrap}
button.unsure:hover{border-color:var(--warn);color:var(--warn)}
.card.notsure{border-color:var(--warn)}
.card.notsure button.unsure{background:var(--warn);color:#2a0710;border-color:var(--warn);font-weight:600}
button.notfound{background:transparent;color:var(--dim);border:1px solid var(--line);
  border-radius:8px;padding:.3rem .7rem;cursor:pointer;font-size:.82rem;white-space:nowrap}
button.notfound:hover{border-color:#d9a066;color:#d9a066}
.card.missing{border-color:#d9a066}
.card.missing button.notfound{background:#d9a066;color:#241405;border-color:#d9a066;font-weight:600}
.fiber{border:1px solid var(--line);border-radius:10px;padding:.6rem .7rem;margin:.5rem 0;background:var(--card)}
.fiber .ttl{font-weight:600;font-size:.9rem;margin-bottom:.35rem}
.m{display:flex;align-items:center;gap:.7rem;padding:.28rem .3rem;border-radius:8px}
.m+.m{border-top:1px solid var(--line)}
.m.bad{background:rgba(255,122,144,.1)}
.m.flagged{opacity:.45;text-decoration:line-through}
.span{font-variant:tabular-nums;width:9rem;font-size:.85rem;color:var(--dim)}
.sim{font-variant:tabular-nums;width:5rem;font-size:.85rem}
.m.bad .sim{color:var(--warn);font-weight:600}
button.flag{background:transparent;color:var(--dim);border:1px solid var(--line);
  border-radius:8px;padding:.2rem .6rem;cursor:pointer;font-size:.8rem}
button.flag:hover{border-color:var(--warn);color:var(--warn)}
.m.flagged button.flag{color:var(--warn);border-color:var(--warn)}
.empty{color:var(--dim);padding:2rem 0;text-align:center}
</style></head><body>
<header><div class=bar>
  <h1>🎚 Alignment review</h1>
  <div class=tabs>
    <button class="tab on" data-t=stem>Stem winner</button>
    <button class=tab data-t=fiber>Fibers</button>
  </div>
  <div class=fld><label>set</label><select id=set></select></div>
</div></header>
<main>
  <!-- ===== STEM WINNER ===== -->
  <section class="panel on" id=panel-stem>
    <div class=toolbar>
      <div class=fld><label>view</label>
        <div class=tabs id=layerToggle>
          <button class="tab on" data-only="vocals">Acappella</button>
          <button class=tab data-only="instrumental">Instrumental</button>
          <button class=tab data-only="">Both</button>
        </div></div>
      <button class=go id=loadStem>Load</button>
      <span id=status></span>
    </div>
    <p class=note>Per layer, play each source and mark it:
      <b style="color:var(--good)">★ clean</b> = virtually no artifacts ·
      <b style="color:var(--accent)">✓ keep</b> = usable but some artifacts ·
      <b style="color:var(--warn)">✗ diff</b> = completely different / wrong track.
      Mark as many as apply (two equally-clean sources → ★ both). Layer-level:
      <b>not sure</b> = can't tell which is best · <b>wrong version</b> = none is the
      actual track (only remixes/noise found). A <b style="color:var(--warn)">pink
      duration</b> doesn't match the baseline length — likely a preview clip. Every
      change logs the layer's full verdict to out/discern/picks.jsonl.</p>
    <div id=stemOut></div>
  </section>
  <!-- ===== FIBERS ===== -->
  <section class=panel id=panel-fiber>
    <div class=toolbar>
      <div class=fld><label>stem</label><select id=fstem>
        <option>acappella</option><option>instrumental</option><option>regular</option></select></div>
      <div class=fld><label>feature</label><select id=ffeat>
        <option value="">auto</option><option>hubert</option><option>chroma</option></select></div>
      <div class=fld><label>sections K · <span id=kv>6</span></label>
        <input id=k type=range min=3 max=12 value=6></div>
      <div class=fld><label>min sec · <span id=msv>4</span>s</label>
        <input id=ms type=range min=2 max=12 value=4></div>
      <div class=fld><label>max refs · <span id=mrv>10</span></label>
        <input id=mr type=range min=1 max=40 value=10></div>
      <div class=fld><label>borderline · <span id=bv>0.65</span></label>
        <input id=b type=range min=0 max=100 value=65></div>
      <button class=go id=compFiber>Compute</button>
      <span id=fstatus></span>
    </div>
    <p class=note>Each fiber groups sections the algorithm calls the same content — play
      them in sequence; they should sound like the same part. <b style="color:var(--warn)">
      Pink</b> = diagonal sim below the threshold (a member that may differ — the singer's
      emphasis — or a wrong merge). Hit <b>flag</b> on anything that doesn't belong.
      Flags log to out/fiber_review/flags.jsonl.</p>
    <div id=fiberOut></div>
  </section>
</main>
<script>
const $=id=>document.getElementById(id);
// tabs
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
  $('panel-stem').classList.toggle('on',t.dataset.t==='stem');
  $('panel-fiber').classList.toggle('on',t.dataset.t==='fiber');
});
// sets
async function loadSets(){const s=await(await fetch('/api/sets')).json();
  $('set').innerHTML=s.map(x=>`<option value="${x.id}">${x.name}</option>`).join('');}

// ---- stem winner ---- per-source mark (clean|keep|diff) + layer verdict
let stemState={};  // card index -> {set,folder,layer,mark:{name->m},not_sure,not_found}
const MARKCLS={clean:'clean',keep:'keep',diff:'diff'};
function submitLayer(i){const v=stemState[i],names=Object.keys(v.mark);
  fetch('/api/pick',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({set:v.set,folder:v.folder,layer:v.layer,
      clean:names.filter(n=>v.mark[n]==='clean'),
      keep:names.filter(n=>v.mark[n]==='keep'),
      different:names.filter(n=>v.mark[n]==='diff'),
      not_sure:v.not_sure,not_found:v.not_found})});}
function syncCard(i){const v=stemState[i],card=$('card'+i);
  card.querySelectorAll('.src').forEach(s=>{
    const m=v.mark[s.dataset.name]||'';
    s.classList.remove('clean','keep','diff');if(m)s.classList.add(MARKCLS[m]);
    s.querySelectorAll('.mk').forEach(b=>b.classList.toggle('on',b.dataset.m===m));});
  card.classList.toggle('notsure',v.not_sure);
  card.classList.toggle('missing',v.not_found);
  card.querySelector('.unsure').textContent=v.not_sure?'🤷 not sure':'not sure';
  card.querySelector('.notfound').textContent=v.not_found?'⚠ wrong version':'wrong version';
  const cnt=m=>Object.values(v.mark).filter(x=>x===m).length;
  const t=card.querySelector('.tally');
  t.textContent=v.not_found?'no right version':v.not_sure?'not sure'
    :[[cnt('clean'),'clean'],[cnt('keep'),'keep'],[cnt('diff'),'different']]
       .filter(([n])=>n).map(([n,l])=>n+' '+l).join(' · ')||'—';
  t.classList.toggle('has',(cnt('clean')+cnt('keep'))>0&&!v.not_sure&&!v.not_found);}
function setMark(i,name,m){const v=stemState[i];
  if(v.mark[name]===m)delete v.mark[name];else v.mark[name]=m;
  v.not_sure=false;v.not_found=false;syncCard(i);submitLayer(i);}
function toggleUnsure(i){const v=stemState[i];
  v.not_sure=!v.not_sure;if(v.not_sure){v.mark={};v.not_found=false;}
  syncCard(i);submitLayer(i);}
function toggleNotFound(i){const v=stemState[i];
  v.not_found=!v.not_found;if(v.not_found){v.mark={};v.not_sure=false;}
  syncCard(i);submitLayer(i);}
let stemOnly='vocals';  // acappella view by default
$('layerToggle').querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  $('layerToggle').querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
  stemOnly=t.dataset.only;loadStem();});
async function loadStem(){
  $('status').textContent='loading…';$('stemOut').innerHTML='';stemState={};
  const q=new URLSearchParams({set:$('set').value,only:stemOnly});
  const d=await(await fetch('/api/layers?'+q)).json();
  if(d.error){$('status').textContent=d.error;return;}
  if(!d.layers.length){$('stemOut').innerHTML='<p class=empty>No candidates downloaded yet — run fetch_candidate_stems.py.</p>';$('status').textContent='';return;}
  $('status').textContent=d.layers.length+' layers with candidates';
  $('stemOut').innerHTML=d.layers.map((L,i)=>{
    stemState[i]={set:d.set,folder:L.folder,layer:L.layer,mark:{},not_sure:false,not_found:false};
    return `<div class=card id=card${i}>
    <div class=hd><div class=meta><h3>${esc(L.folder)}</h3>
      <div class=sub>${L.layer} · ${L.n_cand} candidates</div></div>
      <div style="display:flex;gap:.6rem;align-items:center">
        <span class=tally>—</span>
        <button class=unsure onclick='toggleUnsure(${i})'>not sure</button>
        <button class=notfound onclick='toggleNotFound(${i})'>wrong version</button></div></div>`+
    L.sources.map(s=>`<div class="src ${s.kind}" data-name="${esc(s.name)}">
      <span class="badge ${s.kind==='demucs'?'demucs':'cand'}">${s.kind==='demucs'?'baseline':'cand'}</span>
      <span class=nm>${esc(s.name)}</span>
      <span class="dur ${s.match?'':'bad'}">${s.dur}s</span>
      <audio controls preload=none src="/audio?id=${s.audio}"></audio>
      <span class=marks>
        <button class="mk mclean" data-m=clean title="clean — virtually no artifacts" onclick='setMark(${i},${attr(s.name)},"clean")'>★ clean</button>
        <button class="mk mkeep" data-m=keep title="usable but some artifacts" onclick='setMark(${i},${attr(s.name)},"keep")'>✓ keep</button>
        <button class="mk mdiff" data-m=diff title="completely different / wrong track" onclick='setMark(${i},${attr(s.name)},"diff")'>✗ diff</button>
      </span>
    </div>`).join('')+`</div>`;}).join('');
}
$('loadStem').onclick=loadStem;
$('set').onchange=()=>{if($('panel-stem').classList.contains('on'))loadStem();};

// ---- fibers ----
for(const [s,l] of [['k','kv'],['ms','msv'],['mr','mrv']])
  $(s).oninput=()=>$(l).textContent=$(s).value;
$('b').oninput=()=>{$('bv').textContent=(+$('b').value/100).toFixed(2);recolor();};
function recolor(){const t=+$('b').value/100;
  document.querySelectorAll('.m').forEach(m=>m.classList.toggle('bad',+m.dataset.sim<t));}
async function flag(btn,d){await fetch('/api/flag',{method:'POST',
  headers:{'content-type':'application/json'},body:JSON.stringify(d)});
  const m=btn.closest('.m');m.classList.toggle('flagged');
  btn.textContent=m.classList.contains('flagged')?'unflag':'flag';}
$('compFiber').onclick=async()=>{
  $('fstatus').textContent='computing…';$('fiberOut').innerHTML='';
  const q=new URLSearchParams({set:$('set').value,stem:$('fstem').value,
    feature:$('ffeat').value,k:$('k').value,min_section:$('ms').value,max_refs:$('mr').value});
  const d=await(await fetch('/api/fibers?'+q)).json();
  if(d.error){$('fstatus').textContent=d.error;return;}
  if(!d.refs.length){$('fiberOut').innerHTML='<p class=empty>No multi-member fibers found.</p>';$('fstatus').textContent='';return;}
  $('fstatus').textContent=d.refs.length+' refs · '+d.feature;
  $('fiberOut').innerHTML=d.refs.map(ref=>`<div class=card><h3>${esc(ref.title)}</h3>
    <div class=sub>${d.stem} · ${d.feature}</div>`+
    ref.fibers.map(f=>`<div class=fiber><div class=ttl>fiber ${f.label} · ${f.members.length} members</div>`+
      f.members.map(m=>`<div class=m data-sim="${m.sim}">
        <span class=span>${m.start}–${m.end}s</span>
        <span class=sim>sim ${m.sim.toFixed(2)}</span>
        <audio controls preload=none src="/audio?id=${m.audio}"></audio>
        <button class=flag onclick='flag(this,${attr({set:d.set,stem:d.stem,rid:ref.rid,label:f.label,start:m.start,end:m.end,sim:m.sim})})'>flag</button>
      </div>`).join('')+`</div>`).join('')+`</div>`).join('');
  recolor();
};
function esc(s){return (s+'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function attr(o){return esc(JSON.stringify(o));}
loadSets();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/sets":
            return self._send(200, list_sets())
        if u.path == "/api/layers":
            try:
                data = build_layers(q.get("set", ""), q.get("only") or None)
            except Exception as ex:
                data = {"error": f"{type(ex).__name__}: {ex}"}
            return self._send(200, data)
        if u.path == "/api/fibers":
            try:
                data = build_fibers(
                    q.get("set", ""),
                    q.get("stem", "acappella"),
                    q.get("feature") or None,
                    int(q.get("k", 6)),
                    float(q.get("min_section", 4)),
                    int(q.get("max_refs", 10)),
                )
            except Exception as ex:
                data = {"error": f"{type(ex).__name__}: {ex}"}
            return self._send(200, data)
        if u.path == "/audio":
            with _lock:
                rec = _audio_map.get(q.get("id", ""))
            if not rec:
                return self._send(404, {"error": "unknown"})
            audio = _cut(*rec)
            if audio is None:
                return self._send(500, {"error": "cut failed"})
            return self._send(200, audio, "audio/mpeg")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        target = {"/api/pick": PICKS, "/api/flag": FLAGS}.get(path)
        if target is None:
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode() if n else "{}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock, target.open("a") as f:
            f.write(body.strip() + "\n")
        return self._send(200, {"ok": True})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--open", action="store_true")
    args = p.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"review UI -> {url}  (picks -> {PICKS}; flags -> {FLAGS})")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

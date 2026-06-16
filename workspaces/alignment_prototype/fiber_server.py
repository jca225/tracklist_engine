#!/usr/bin/env python3
"""Interactive fiber inspector — a local website to verify self-repeat classes.

Static export (fiber_ui.py) renders once; this serves a live site so you can
pick the set/stem/feature, slide K and the borderline threshold, recompute, play
each fiber's member segments back-to-back, and FLAG a member that doesn't belong
(the nuanced case: a singer emphasising a word differently, or two sections
wrongly merged). Flags are appended to out/fiber_review/flags.jsonl — that
feedback is exactly the equivalence-class GT the fiber-aware objective needs.

Audio is cut on demand (cached) from the SAME stem the fibers were computed on,
so you hear precisely what the algorithm compared. Single-user, localhost only.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.fiber_server \
        [--port 8765] [--open]
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
OUT = _REPO / "workspaces/alignment_prototype/out/fiber_review"
SNIP = OUT / "_snips"
FLAGS = OUT / "flags.jsonl"
_DEFAULT_FEATURE = {
    "acappella": "hubert",
    "instrumental": "chroma",
    "regular": "chroma",
}

_snip_map: dict[str, tuple[str, float, float]] = {}  # id -> (src, start, end)
_lock = threading.Lock()


def _pooled(feat: np.ndarray, s: float, e: float) -> np.ndarray:
    a, b = int(s * FPS), int(e * FPS)
    seg = feat[:, a : max(a + 1, b)]
    v = seg.mean(axis=1)
    return v / (np.linalg.norm(v) + 1e-9)


def list_sets() -> list[dict]:
    out = []
    for d in sorted(ALIGNING.glob("*__*")):
        if (d / "manifest.json").is_file():
            sid = d.name.split("__", 1)[0]
            out.append({"id": sid, "name": d.name})
    return out


def _set_dir(set_id: str) -> Path | None:
    hits = sorted(ALIGNING.glob(f"{set_id}__*"))
    return hits[0] if hits else None


def build_fibers(set_id, stem, feature, k, min_section, max_refs) -> dict:
    sd = _set_dir(set_id)
    if sd is None:
        return {"error": f"no set {set_id}"}
    manifest = json.loads((sd / "manifest.json").read_text())
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
            cen = np.mean([_pooled(feat, s, e) for s, e in members], axis=0)
            cen /= np.linalg.norm(cen) + 1e-9
            ms = []
            for s, e in members:
                sim = float(_pooled(feat, s, e) @ cen)
                sid = hashlib.md5(f"{sp}{s}{e}".encode()).hexdigest()[:16]
                with _lock:
                    _snip_map[sid] = (sp, s, e)
                ms.append(
                    {
                        "start": round(s, 1),
                        "end": round(e, 1),
                        "sim": round(sim, 3),
                        "audio": sid,
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


def _cut(src: str, s: float, e: float) -> bytes | None:
    SNIP.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{src}{s}{e}".encode()).hexdigest()[:16]
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
            "-ac",
            "1",
            "-ar",
            "22050",
            str(out),
        ]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None
    return out.read_bytes()


PAGE = """<!doctype html><meta charset=utf-8><title>Fiber inspector</title>
<style>
body{font-family:system-ui;margin:1.5rem;max-width:64rem}
.row{display:flex;gap:1rem;align-items:end;flex-wrap:wrap;margin-bottom:1rem}
label{font-size:.8rem;color:#555;display:block}
select,input{font-size:1rem}
button{font-size:1rem;padding:.4rem .9rem;cursor:pointer}
.fiber{border:1px solid #ccc;border-radius:8px;padding:.6rem;margin:.6rem 0}
.m{display:flex;align-items:center;gap:.6rem;padding:.25rem 0}
.m.bad{background:#ffecec}.m.flagged{opacity:.5;text-decoration:line-through}
.sim{font-variant:tabular-nums;color:#666;width:5rem}
.m.bad .sim{color:#c00;font-weight:600}
small{color:#888}audio{height:1.9rem}h2{margin:.4rem 0}
.flag{font-size:.8rem;padding:.15rem .5rem}
#status{color:#888;margin-left:.5rem}
</style>
<h1>Fiber inspector</h1>
<p>Each fiber groups sections the algorithm calls the same content — play them in
sequence; they should sound like the same part. <b>Pink</b> = similarity below the
threshold (a member that may differ, e.g. the singer's emphasis, or a wrong merge).
Hit <b>flag</b> on anything that doesn't belong.</p>
<div class=row>
  <div><label>set</label><select id=set></select></div>
  <div><label>stem</label><select id=stem>
    <option>acappella</option><option>instrumental</option><option>regular</option>
  </select></div>
  <div><label>feature</label><select id=feature>
    <option value="">auto</option><option>hubert</option><option>chroma</option>
  </select></div>
  <div><label>sections K <span id=kv>6</span></label>
    <input id=k type=range min=3 max=12 value=6></div>
  <div><label>min section s <span id=msv>4</span></label>
    <input id=ms type=range min=2 max=12 value=4></div>
  <div><label>max refs <span id=mrv>10</span></label>
    <input id=mr type=range min=1 max=30 value=10></div>
  <div><label>borderline <span id=bv>0.60</span></label>
    <input id=b type=range min=0 max=100 value=60></div>
  <button id=go>Compute</button><span id=status></span>
</div>
<div id=out></div>
<script>
const $=id=>document.getElementById(id);
for(const [s,l] of [['k','kv'],['ms','msv'],['mr','mrv']])
  $(s).oninput=()=>$(l).textContent=$(s).value;
$('b').oninput=()=>{$('bv').textContent=(+$('b').value/100).toFixed(2);recolor();};
function recolor(){const t=+$('b').value/100;
  document.querySelectorAll('.m').forEach(m=>m.classList.toggle('bad',+m.dataset.sim<t));}
async function load(){const r=await fetch('/api/sets');const sets=await r.json();
  $('set').innerHTML=sets.map(s=>`<option value="${s.id}">${s.name}</option>`).join('');}
async function flag(btn,d){await fetch('/api/flag',{method:'POST',
  headers:{'content-type':'application/json'},body:JSON.stringify(d)});
  btn.closest('.m').classList.toggle('flagged');btn.textContent=
  btn.closest('.m').classList.contains('flagged')?'unflag':'flag';}
$('go').onclick=async()=>{
  $('status').textContent='computing…';$('out').innerHTML='';
  const q=new URLSearchParams({set:$('set').value,stem:$('stem').value,
    feature:$('feature').value,k:$('k').value,min_section:$('ms').value,
    max_refs:$('mr').value});
  const r=await fetch('/api/fibers?'+q);const d=await r.json();
  if(d.error){$('status').textContent=d.error;return;}
  $('status').textContent=d.refs.length+' refs ('+d.feature+')';
  $('out').innerHTML=d.refs.map(ref=>`<h2>${ref.title}</h2>`+
    ref.fibers.map(f=>`<div class=fiber><b>fiber ${f.label}</b> — ${f.members.length} members`+
      f.members.map((m,i)=>`<div class=m data-sim="${m.sim}">
        <span>${m.start}–${m.end}s</span>
        <span class=sim>sim ${m.sim.toFixed(2)}</span>
        <audio controls preload=none src="/audio?id=${m.audio}"></audio>
        <button class=flag onclick='flag(this,${JSON.stringify(
          {set:d.set,stem:d.stem,rid:ref.rid,label:f.label,start:m.start,end:m.end,sim:m.sim})})'>flag</button>
      </div>`).join('')+`</div>`).join('')).join('');
  recolor();
};
load();
</script>
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
            except Exception as ex:  # surface errors to the page, don't 500 blindly
                data = {"error": f"{type(ex).__name__}: {ex}"}
            return self._send(200, data)
        if u.path == "/audio":
            sid = q.get("id", "")
            with _lock:
                rec = _snip_map.get(sid)
            if not rec:
                return self._send(404, {"error": "unknown snippet"})
            audio = _cut(*rec)
            if audio is None:
                return self._send(500, {"error": "cut failed"})
            return self._send(200, audio, "audio/mpeg")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path == "/api/flag":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode() if n else "{}"
            FLAGS.parent.mkdir(parents=True, exist_ok=True)
            with _lock, FLAGS.open("a") as f:
                f.write(body.strip() + "\n")
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open the browser")
    args = p.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"fiber inspector → {url}  (Ctrl-C to stop; flags → {FLAGS})")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

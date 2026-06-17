#!/usr/bin/env python3
"""Stem source-discernment UI — A/B pick the best acappella/instrumental source.

The fiber audit proved we can't build reliable structure on noisy separated
stems ([[project_fibers]]); the fix is to outsource clean acappellas/instrumentals
and PICK the best source per layer. `scripts/fetch_candidate_stems.py` already
downloads YouTube candidates into `stems/<song>/candidates/`; this is the missing
"listen and pick a winner" step the project planned ([[project_official_stems_search]]
"collect A/B preferences -> learned quality-ranking gate").

Per used layer (vocals / instrumental) it lists every source side by side — the
Demucs baseline + each downloaded candidate — plays a chunk from the middle of
each, shows the auto quality signal (duration, match-to-baseline-length which
filters preview clips), and records your pick. Picks append to
out/discern/picks.jsonl — the training labels for the eventual ranker. Set-
agnostic, so it also serves BB11/BB10 once their candidates land (another agent
is acquiring those). Localhost only.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.discern_server \
        [--port 8799] [--open]
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

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

ALIGNING = Path.home() / "aligning"
OUT = _REPO / "workspaces/alignment_prototype/out/discern"
SNIP = OUT / "_chunks"
PICKS = OUT / "picks.jsonl"
CHUNK_S = 30.0  # seconds served per source (from 25% in) for A/B listening

_src_map: dict[str, str] = {}  # id -> absolute source path
_lock = threading.Lock()


def list_sets() -> list[dict]:
    out = []
    for d in sorted(ALIGNING.glob("*__*")):
        if (d / "stems").is_dir():
            out.append({"id": d.name.split("__", 1)[0], "name": d.name})
    return out


def _set_dir(set_id):
    hits = sorted(ALIGNING.glob(f"{set_id}__*"))
    return hits[0] if hits else None


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


def _register(path: Path) -> str:
    sid = hashlib.md5(str(path).encode()).hexdigest()[:16]
    with _lock:
        _src_map[sid] = str(path)
    return sid


def _candidate_files(stem_dir: Path, layer: str) -> list[Path]:
    base = stem_dir / "candidates"
    out: list[Path] = []
    for d in (base / layer, base):
        if d.is_dir():
            out += sorted(d.glob("cand*.m4a"))
    # de-dup by name (candidates/ and candidates/<layer>/ can overlap)
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
            baseline = folder / f"{layer}.flac"
            if not cands and not baseline.is_file():
                continue
            if not cands:
                continue  # nothing to discern yet (only the Demucs baseline)
            sources = []
            base_dur = _ffprobe_dur(str(baseline)) if baseline.is_file() else 0.0
            if baseline.is_file():
                sources.append(
                    {
                        "name": "Demucs (baseline)",
                        "kind": "demucs",
                        "dur": round(base_dur, 1),
                        "match": True,
                        "audio": _register(baseline),
                    }
                )
            for c in cands:
                d = _ffprobe_dur(str(c))
                match = base_dur > 0 and abs(d - base_dur) <= 5.0
                sources.append(
                    {
                        "name": c.name[:48],
                        "kind": "candidate",
                        "dur": round(d, 1),
                        # length match to the baseline filters preview clips (the key
                        # auto signal from the official-stems plan)
                        "match": bool(match),
                        "audio": _register(c),
                    }
                )
            layers.append(
                {
                    "folder": folder.name,
                    "layer": layer,
                    "n_cand": len(cands),
                    "sources": sources,
                }
            )
    return {"set": set_id, "layers": layers}


def _chunk(src: str) -> bytes | None:
    SNIP.mkdir(parents=True, exist_ok=True)
    dur = _ffprobe_dur(src)
    start = max(0.0, dur * 0.25)
    key = hashlib.md5(f"{src}{start}{CHUNK_S}".encode()).hexdigest()[:16]
    out = SNIP / f"{key}.mp3"
    if not out.is_file():
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{CHUNK_S:.0f}",
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


PAGE = """<!doctype html><meta charset=utf-8><title>Stem discernment</title>
<style>
body{font-family:system-ui;margin:1.5rem;max-width:64rem}
.layer{border:1px solid #ccc;border-radius:8px;padding:.7rem;margin:.7rem 0}
.src{display:flex;align-items:center;gap:.6rem;padding:.3rem 0;border-top:1px solid #eee}
.src.demucs{color:#555}.src.picked{background:#e7f7e7;border-radius:6px}
.bad{color:#c00}.dur{font-variant:tabular-nums;color:#666;width:6rem}
button{font-size:.95rem;padding:.25rem .7rem;cursor:pointer}
audio{height:1.9rem}.row{display:flex;gap:1rem;align-items:end;margin-bottom:1rem}
small{color:#888}h3{margin:.3rem 0}#status{color:#888;margin-left:.5rem}
.none{color:#999;font-size:.85rem}
</style>
<h1>Stem source discernment</h1>
<p>Per layer, play a 30s chunk of each source and <b>pick the best</b> (cleanest
isolation, fewest artifacts, full length). <b>Pink length</b> = doesn't match the
baseline duration (likely a preview clip). Picks log to out/discern/picks.jsonl.</p>
<div class=row>
  <div><label>set</label><br><select id=set></select></div>
  <div><label>layer</label><br><select id=layer>
    <option value="">both</option><option>vocals</option><option>instrumental</option>
  </select></div>
  <button id=go>Load</button><span id=status></span>
</div>
<div id=out></div>
<script>
const $=id=>document.getElementById(id);
async function load(){const r=await fetch('/api/sets');const s=await r.json();
  $('set').innerHTML=s.map(x=>`<option value="${x.id}">${x.name}</option>`).join('');}
async function pick(btn,d){await fetch('/api/pick',{method:'POST',
  headers:{'content-type':'application/json'},body:JSON.stringify(d)});
  const box=btn.closest('.layer');
  box.querySelectorAll('.src').forEach(s=>s.classList.remove('picked'));
  btn.closest('.src').classList.add('picked');}
$('go').onclick=async()=>{
  $('status').textContent='loading…';$('out').innerHTML='';
  const q=new URLSearchParams({set:$('set').value,only:$('layer').value});
  const d=await(await fetch('/api/layers?'+q)).json();
  if(d.error){$('status').textContent=d.error;return;}
  $('status').textContent=d.layers.length+' layers with candidates';
  if(!d.layers.length){$('out').innerHTML='<p class=none>No candidates downloaded yet — run fetch_candidate_stems.py.</p>';return;}
  $('out').innerHTML=d.layers.map(L=>`<div class=layer><h3>${L.folder} <small>(${L.layer}, ${L.n_cand} candidates)</small></h3>`+
    L.sources.map(s=>`<div class="src ${s.kind}">
      <button onclick='pick(this,${JSON.stringify({set:d.set,folder:L.folder,layer:L.layer,pick:s.name})})'>pick</button>
      <span>${s.name}</span>
      <span class="dur ${s.match?'':'bad'}">${s.dur}s</span>
      <audio controls preload=none src="/audio?id=${s.audio}"></audio>
    </div>`).join('')+`</div>`).join('');
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
        if u.path == "/api/layers":
            try:
                data = build_layers(q.get("set", ""), q.get("only") or None)
            except Exception as ex:
                data = {"error": f"{type(ex).__name__}: {ex}"}
            return self._send(200, data)
        if u.path == "/audio":
            with _lock:
                src = _src_map.get(q.get("id", ""))
            if not src:
                return self._send(404, {"error": "unknown"})
            audio = _chunk(src)
            if audio is None:
                return self._send(500, {"error": "cut failed"})
            return self._send(200, audio, "audio/mpeg")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path == "/api/pick":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode() if n else "{}"
            PICKS.parent.mkdir(parents=True, exist_ok=True)
            with _lock, PICKS.open("a") as f:
                f.write(body.strip() + "\n")
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--open", action="store_true")
    args = p.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"stem discernment -> {url}  (picks -> {PICKS})")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch YouTube acappella/instrumental *candidates* for the stem layers a
labeling Ableton set actually uses, for A/B audition during ground-truth work.

This is a **labeling-support** tool, not a corpus ingest path. It does NOT touch
the canonical DB or `track_audio`; it only reads a pulled `~/aligning/<set>/`
folder + its Ableton `.als`, and writes candidate audio into
`stems/<song>/candidates/` for the annotator to listen to and pick a winner.

Why it exists: Demucs `vocals.flac` / `instrumental.flac` is a fine baseline, but
a *real* acappella or instrumental sourced online is often better (cleaner, no
separation artifacts). We download several candidates per used layer so the
annotator can swap the better one into Ableton. The chosen winners become the
training signal for the future aligner ("for song X, replace the Demucs stem").

What counts as "used": only the `vocals.flac` / `instrumental.flac` stems that
the `.als` actually references as clips (a set uses ~113 of 155 layers), plus the
handful of real acappellas dragged in from `~/Downloads`. Full tracks (played
whole, no isolated layer) are skipped.

Query strategy (the version-axis-aware part):
  - VOCAL layer  -> the acappella is version-independent (a remix reuses the
    original vocal), so we search the ORIGINAL song with vocal qualifiers:
    "acapella" / "acappella" / "vocals only" / "a cappella". Remix qualifier
    deliberately dropped.
  - INSTRUMENTAL layer -> version-SPECIFIC. A remix's instrumental != the
    original's. So for remix/rework layers we keep the remixer name in the query
    ("<artist> <title> <remixer> Remix instrumental"). Generic "(Remix)" with no
    named remixer falls back to "<artist> <title> remix instrumental" and is
    flagged low-confidence.
  Each query is run via regular YouTube search (yt-dlp `ytsearch`), merged +
  de-duped by videoId, then ranked by stem-keyword match. YT Music's API was
  the first pass but missed uploads that rank on youtube.com (Acapella World,
  official voice tracks, etc.).

Usage (from repo root):
  # Sync every layer in your .als that lacks candidates (default — safe to re-run):
  venvs/audio/bin/python scripts/fetch_candidate_stems.py \
    --als "$HOME/Desktop/big bootie 12 labeling Project/big bootie 12 labeling_fast.als"
  # Preview searches only:
  venvs/audio/bin/python scripts/fetch_candidate_stems.py --als ... --dry-run
  # Re-search everything (not just gaps):
  venvs/audio/bin/python scripts/fetch_candidate_stems.py --als ... --all
  # Legacy batching (avoid unless debugging):
  venvs/audio/bin/python scripts/fetch_candidate_stems.py --limit 15 --offset 0
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger("fetch_candidate_stems")

DEFAULT_SET = Path.home() / "aligning" / "1fsnxchk__Two Friends - Big Bootie Mix Volume 12"

# The 7 real acappellas dragged in from ~/Downloads -> their song stem folders.
# (Mapped by substring so the messy download filenames resolve to a song.)
MANUAL_DOWNLOAD_VOCALS = {
    "Everytime We Touch - Cascada": "013__Cascada - Everytime We Touch (Rework) [143bpm 4B]",
    "Michael Jackson - Beat it": "009__Michael Jackson - Beat It (Acappella) [137bpm 2A]",
    "What's My Age Again": "024__Blink-182 - What's My Age Again- (Acappella) [156bpm 2B]",
    "Dance Floor Anthem": "024w1__Good Charlotte - Dance Floor Anthem (Acappella) [124bpm 2B]",
    "Bored to Death": "029__Blink-182 - Bored To Death (Acappella) [162bpm 8B]",
    "Wouldn't it be nice": "035__The Beach Boys - Wouldn't It Be Nice (Acappella) [123bpm 8B]",
    "Mo Money Mo Problems": "087__The Notorious B.I.G - Mo Money Mo Problems (Acappella) [104bpm 12B]",
}

_REMIX_WORDS = ("remix", "rework", "edit", "bootleg", "mashup", "altversion", "vip", "flip")
_VOCAL_SUFFIXES = ("acapella", "acappella", "vocals only", "a cappella")
_INSTR_SUFFIXES = ("instrumental", "instrumental only")
_BRACKET_RE = re.compile(r"\s*\[[^\]]*\]\s*$")           # trailing " [128bpm 1A]" / "[no-features]"
_LEAD_NUM_RE = re.compile(r"^\d+(?:w\d+)?__")            # "048__" / "024w1__"
_PAREN_RE = re.compile(r"\s*\(([^)]*)\)\s*$")            # trailing "(Madison Mars Remix)"


@dataclass(frozen=True)
class Layer:
    folder: str          # stem folder name, e.g. "048__Martin Garrix... (Madison Mars Remix) [..]"
    layer: str           # "vocals" | "instrumental"
    artist: str
    title: str           # base title, version qualifier stripped
    version_tag: str     # lowercased: "" | "remix" | "rework" | ...
    remixer: str         # named remixer for instrumental queries, or ""
    track_id: str = ""

    @property
    def num(self) -> tuple[int, str]:
        mo = re.match(r"(\d+)(w\d+)?", self.folder)
        return (int(mo.group(1)), mo.group(2) or "") if mo else (9999, "")

    @property
    def is_remix(self) -> bool:
        return self.version_tag in _REMIX_WORDS

    @property
    def low_confidence(self) -> bool:
        # remix instrumental whose specific remixer we couldn't name
        return self.layer == "instrumental" and self.is_remix and not self.remixer

    def queries(self) -> list[str]:
        a, t = self.artist, self.title
        if self.layer == "vocals":
            base = f"{a} {t}".strip()
            return [f"{base} {s}" for s in _VOCAL_SUFFIXES]
        # instrumental
        if self.remixer:
            base = f"{a} {t} {self.remixer} Remix".strip()
        elif self.is_remix:
            base = f"{a} {t} {self.version_tag}".strip()
        else:
            base = f"{a} {t}".strip()
        return [f"{base} {s}" for s in _INSTR_SUFFIXES]


@dataclass
class Hit:
    video_id: str
    title: str
    uploader: str
    duration: str
    src_filter: str
    src_query: str
    score: int = 0


def _parse_folder(folder: str) -> tuple[str, str, str, str]:
    """folder name -> (artist, base_title, version_tag, remixer)."""
    s = _LEAD_NUM_RE.sub("", folder)
    s = _BRACKET_RE.sub("", s).strip()
    version_tag, remixer = "", ""
    mo = _PAREN_RE.search(s)
    if mo:
        qual = mo.group(1).strip()
        s = _PAREN_RE.sub("", s).strip()
        words = qual.split()
        last = words[-1].lower() if words else ""
        if last in _REMIX_WORDS:
            version_tag = last
            name = " ".join(words[:-1]).strip()
            # "Madison Mars Remix" -> remixer "Madison Mars"; bare "Remix" -> none
            if name and name.lower() not in ("instrumental", "extended", "club", "radio"):
                remixer = name
        elif qual.lower() in ("acappella", "acapella", "instrumental", "instrumental mix"):
            pass  # stem-ish label, not a version
        else:
            version_tag = qual.lower()
    artist, _, title = s.partition(" - ")
    if not title:                       # no " - " split; treat whole as title
        artist, title = artist, ""
    return artist.strip(), title.strip(), version_tag, remixer


def extract_layers(set_dir: Path, als_xml: str) -> list[Layer]:
    """Used (folder, layer) pairs from the .als, enriched from manifest+folder."""
    manifest = json.loads((set_dir / "manifest.json").read_text())
    folder2meta: dict[str, dict] = {}
    for t in manifest.get("tracks", []):
        for layer in ("vocals", "instrumental"):
            p = (t.get("stems") or {}).get(layer)
            if p:
                folder2meta[Path(p).parent.name] = t

    seen: dict[tuple[str, str], Layer] = {}

    def add(folder: str, layer: str) -> None:
        key = (folder, layer)
        if key in seen:
            return
        meta = folder2meta.get(folder)
        f_artist, f_title, f_vtag, f_remixer = _parse_folder(folder)
        if meta:
            artist = meta.get("artist") or f_artist
            title = meta.get("title") or f_title
            vtag = (meta.get("version_tag") or f_vtag or "").lower()
            tid = meta.get("track_id", "")
        else:
            artist, title, vtag, tid = f_artist, f_title, f_vtag, ""
        seen[key] = Layer(folder, layer, artist, title, vtag, f_remixer, tid)

    import html
    for m in re.finditer(r'<Path Value="([^"]*/stems/([^/]+)/(vocals|instrumental)\.flac)"', als_xml):
        add(html.unescape(m.group(2)), m.group(3))
    for f in MANUAL_DOWNLOAD_VOCALS.values():
        add(f, "vocals")

    return sorted(seen.values(), key=lambda l: (l.num, l.layer))


def _kw_for(layer: str) -> tuple[str, ...]:
    return ("acapella", "acappella", "a cappella", "vocals only", "vocal", "voice track") if layer == "vocals" \
        else ("instrumental", "instr", "inst.")


def _fmt_duration(secs: float | int | None) -> str:
    if not secs:
        return ""
    s = int(secs)
    return f"{s // 60}:{s % 60:02d}"


def rank_hits(layer: Layer, hits: list[Hit]) -> list[Hit]:
    kws = _kw_for(layer.layer)
    a_tok = {w for w in re.split(r"\W+", layer.artist.lower()) if len(w) > 2}
    t_tok = {w for w in re.split(r"\W+", layer.title.lower()) if len(w) > 2}
    rmx = layer.remixer.lower()
    for h in hits:
        tl = h.title.lower()
        s = 0
        if any(k in tl for k in kws):
            s += 5
        s += 2 * len(a_tok & set(re.split(r"\W+", (h.title + " " + h.uploader).lower())))
        s += 2 * len(t_tok & set(re.split(r"\W+", tl)))
        if rmx and rmx in tl:
            s += 4
        if "voice track" in tl or "official" in tl:
            s += 2
        h.score = s
    return sorted(hits, key=lambda h: h.score, reverse=True)


def search_layer(layer: Layer, per_query: int = 5, cookies_browser: str | None = None) -> list[Hit]:
    import yt_dlp
    by_id: dict[str, Hit] = {}
    opts: dict = {"quiet": True, "no_warnings": True, "extract_flat": True}
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        for q in layer.queries():
            try:
                info = ydl.extract_info(f"ytsearch{per_query}:{q}", download=False)
            except Exception as e:                       # noqa: BLE001
                _log.debug("search err %r: %s", q, e)
                continue
            for r in info.get("entries") or []:
                vid = r.get("id")
                if not vid or vid in by_id:
                    continue
                by_id[vid] = Hit(
                    video_id=vid, title=str(r.get("title") or ""),
                    uploader=str(r.get("uploader") or r.get("channel") or ""),
                    duration=_fmt_duration(r.get("duration")),
                    src_filter="youtube", src_query=q,
                )
    return rank_hits(layer, list(by_id.values()))


def _safe(name: str, limit: int = 80) -> str:
    name = re.sub(r"[/\\:]", "-", name)
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "", name).strip()
    return name[:limit].strip() or "untitled"


def download_hit(hit: Hit, dest: Path, rank: int, cookies_browser: str | None) -> Path | None:
    import yt_dlp
    stem = f"cand{rank}__{_safe(hit.title)}__{_safe(hit.uploader, 40)}"
    outtmpl = str(dest / (stem + ".%(ext)s"))
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "0"}],
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    url = f"https://www.youtube.com/watch?v={hit.video_id}"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:                               # noqa: BLE001
        _log.warning("    dl FAIL %s (%s): %s", hit.video_id, hit.title[:40], str(e)[:120])
        return None
    out = dest / (stem + ".m4a")
    return out if out.exists() else None


def _layer_sort_key(entry: dict) -> tuple[int, str, str]:
    mo = re.match(r"(\d+)(w\d+)?", entry.get("folder", ""))
    num = int(mo.group(1)) if mo else 9999
    suffix = (mo.group(2) or "") if mo else ""
    return (num, suffix, entry.get("layer", ""))


def _candidate_dirs(set_dir: Path, folder: str, layer: str) -> list[Path]:
    base = set_dir / "stems" / folder / "candidates"
    return [p for p in (base / layer, base) if p.is_dir()]


def _count_candidates_on_disk(set_dir: Path, folder: str, layer: str) -> int:
    seen: set[str] = set()
    for d in _candidate_dirs(set_dir, folder, layer):
        for f in d.glob("cand*.m4a"):
            seen.add(f.name)
    return len(seen)


def _attach_files_from_disk(entry: dict, set_dir: Path) -> None:
    folder, layer = entry["folder"], entry["layer"]
    by_rank: dict[int, str] = {}
    for d in _candidate_dirs(set_dir, folder, layer):
        for f in sorted(d.glob("cand*.m4a")):
            mo = re.match(r"cand(\d+)__", f.name)
            if mo:
                by_rank[int(mo.group(1))] = f.name
    for c in entry.get("candidates", []):
        rank = c.get("rank")
        if rank in by_rank:
            c["file"] = by_rank[rank]
    for rank, fname in sorted(by_rank.items()):
        if not any(c.get("rank") == rank for c in entry.get("candidates", [])):
            entry.setdefault("candidates", []).append({
                "rank": rank, "video_id": "", "title": fname, "uploader": "",
                "duration": "", "score": 0, "src_filter": "disk", "file": fname,
            })


def _stub_entry(layer: Layer) -> dict:
    return {
        "folder": layer.folder, "layer": layer.layer, "artist": layer.artist,
        "title": layer.title, "version_tag": layer.version_tag, "remixer": layer.remixer,
        "track_id": layer.track_id, "low_confidence": layer.low_confidence,
        "queries": layer.queries(), "candidates": [],
    }


def _finalize_manifest(
    path: Path,
    set_dir: Path,
    all_als_layers: list[Layer],
    run_ledger: list[dict],
    *,
    prune_stale: bool,
) -> list[dict]:
    """Manifest = one row per layer in the current .als, with on-disk files attached."""
    als_keys = {(l.folder, l.layer) for l in all_als_layers}
    run_by_key = {(e["folder"], e["layer"]): e for e in run_ledger}
    old_by_key: dict[tuple[str, str], dict] = {}
    if path.exists():
        for e in json.loads(path.read_text()):
            old_by_key[(e["folder"], e["layer"])] = e

    out: list[dict] = []
    for layer in all_als_layers:
        key = (layer.folder, layer.layer)
        entry = run_by_key.get(key) or old_by_key.get(key) or _stub_entry(layer)
        _attach_files_from_disk(entry, set_dir)
        out.append(entry)

    if not prune_stale:
        for key, entry in old_by_key.items():
            if key not in als_keys:
                _attach_files_from_disk(entry, set_dir)
                out.append(entry)

    return sorted(out, key=_layer_sort_key)


def _audit(
    set_dir: Path,
    all_als_layers: list[Layer],
    merged: list[dict],
    *,
    need: int,
) -> int:
    """Log gaps; return count of layers still missing `need` candidates on disk."""
    by_key = {(e["folder"], e["layer"]): e for e in merged}
    missing: list[str] = []
    partial: list[str] = []
    for layer in all_als_layers:
        n = _count_candidates_on_disk(set_dir, layer.folder, layer.layer)
        if n >= need:
            continue
        label = f"{layer.folder} ({layer.layer}) — {n}/{need} on disk"
        if n == 0:
            missing.append(label)
        else:
            partial.append(label)
    stale = [
        f"{e['folder']} ({e['layer']})"
        for e in merged
        if (e["folder"], e["layer"]) not in {(l.folder, l.layer) for l in all_als_layers}
    ]
    if missing:
        _log.warning("%d layer(s) in .als with NO candidates:", len(missing))
        for line in missing:
            _log.warning("  %s", line)
    if partial:
        _log.warning("%d layer(s) with partial candidates:", len(partial))
        for line in partial:
            _log.warning("  %s", line)
    if stale:
        _log.info("%d manifest row(s) not in this .als (kept on disk only)", len(stale))
    complete = len(all_als_layers) - len(missing) - len(partial)
    _log.info("audit: %d/%d .als layers have >=%d candidates", complete, len(all_als_layers), need)
    return len(missing) + len(partial)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-dir", type=Path, default=DEFAULT_SET)
    p.add_argument("--als", type=Path, default=None, help="Defaults to newest *_fast.als / *.als in the project.")
    p.add_argument("--candidates", type=int, default=3, help="Top-N hits to download per layer.")
    p.add_argument("--only", choices=("vocals", "instrumental"), default=None)
    p.add_argument("--filter", default=None, help="Substring on folder name (e.g. '080' or 'Coldplay').")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0, help="Skip first N layers after filters (legacy batching).")
    p.add_argument(
        "--all", action="store_true",
        help="Process every .als layer, not only those missing candidates on disk (default: gaps only).",
    )
    p.add_argument(
        "--keep-stale", action="store_true",
        help="Keep manifest rows for layers no longer in this .als (default: prune).",
    )
    p.add_argument("--dry-run", action="store_true", help="Search + print resolved hits; download nothing.")
    p.add_argument("--cookies-from-browser", default=None, help="e.g. 'safari' / 'chrome' if bot-checked.")
    p.add_argument("--als-xml", type=Path, default=None, help="Pre-decompressed .als XML (internal/testing).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _load_als_xml(args: argparse.Namespace) -> str:
    if args.als_xml and args.als_xml.exists():
        return args.als_xml.read_text(errors="ignore")
    import gzip
    proj = args.set_dir  # the .als lives in the Ableton project, not the aligning dir
    als = args.als
    if als is None:
        # Best-effort: caller usually passes --als or --als-xml. Fall back to set-dir search.
        cands = sorted(args.set_dir.glob("*.als")) or sorted(Path.cwd().glob("*.als"))
        als = cands[0] if cands else None
    if als is None or not Path(als).exists():
        _log.error("No .als found; pass --als <project.als> or --als-xml <decompressed.xml>")
        sys.exit(2)
    return gzip.decompress(Path(als).read_bytes()).decode("utf-8", "ignore")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s: %(message)s")

    all_als_layers = extract_layers(args.set_dir, _load_als_xml(args))
    layers = list(all_als_layers)
    if args.only:
        layers = [l for l in layers if l.layer == args.only]
    if args.filter:
        layers = [l for l in layers if args.filter.lower() in l.folder.lower()]

    gaps_only = not args.all
    if gaps_only and not args.limit and not args.offset:
        before = len(layers)
        layers = [
            l for l in layers
            if _count_candidates_on_disk(args.set_dir, l.folder, l.layer) < args.candidates
        ]
        _log.info("gap sync: %d/%d .als layers need candidates", len(layers), before)

    if args.offset:
        layers = layers[args.offset :]
    if args.limit:
        layers = layers[: args.limit]

    n_voc = sum(1 for l in layers if l.layer == "vocals")
    n_ins = len(layers) - n_voc
    _log.info(
        "%d to process (%d vocal / %d instrumental); need=%d; dry_run=%s; .als layers=%d",
        len(layers), n_voc, n_ins, args.candidates, args.dry_run, len(all_als_layers),
    )

    ledger: list[dict] = []
    for i, layer in enumerate(layers, 1):
        tag = f"[{layer.version_tag}]" if layer.version_tag else ""
        flag = " (LOW-CONF remixer unknown)" if layer.low_confidence else ""
        _log.info("(%d/%d) %s  %s — %s %s%s", i, len(layers), layer.layer.upper(),
                  layer.artist, layer.title, tag, flag)
        hits = search_layer(layer, cookies_browser=args.cookies_from_browser)[: max(args.candidates, 3)]
        chosen = hits[: args.candidates]
        for h in chosen:
            _log.info("      %-7s %s  %-52s | %-26s | %s",
                      f"s={h.score}", h.video_id, h.title[:52], h.uploader[:26], h.duration)
        entry = {
            "folder": layer.folder, "layer": layer.layer, "artist": layer.artist,
            "title": layer.title, "version_tag": layer.version_tag, "remixer": layer.remixer,
            "track_id": layer.track_id, "low_confidence": layer.low_confidence,
            "queries": layer.queries(),
            "candidates": [{"rank": r + 1, "video_id": h.video_id, "title": h.title,
                            "uploader": h.uploader, "duration": h.duration, "score": h.score,
                            "src_filter": h.src_filter, "file": None}
                           for r, h in enumerate(chosen)],
        }
        if not args.dry_run and chosen:
            # vocals + instrumental for the same stem folder share one parent
            # but must not collide on cand1/cand2/cand3 filenames.
            dest = args.set_dir / "stems" / layer.folder / "candidates" / layer.layer
            dest.mkdir(parents=True, exist_ok=True)
            for r, h in enumerate(chosen):
                # resumable: skip if a file for this rank+video already present
                existing = list(dest.glob(f"cand{r + 1}__*"))
                if existing:
                    entry["candidates"][r]["file"] = existing[0].name
                    continue
                out = download_hit(h, dest, r + 1, args.cookies_from_browser)
                if out:
                    entry["candidates"][r]["file"] = out.name
        ledger.append(entry)

    out_manifest = args.set_dir / "candidate_stems_manifest.json"
    merged = _finalize_manifest(
        out_manifest, args.set_dir, all_als_layers, ledger,
        prune_stale=not args.keep_stale,
    )
    out_manifest.write_text(json.dumps(merged, indent=1, ensure_ascii=False))
    _log.info(
        "wrote ledger -> %s (%d processed this run, %d rows for this .als)",
        out_manifest, len(ledger), len(merged),
    )
    gaps = _audit(args.set_dir, all_als_layers, merged, need=args.candidates)
    if gaps and not args.dry_run and gaps_only and not args.filter:
        _log.warning("re-run the same command to retry failed downloads")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Match a library of downloaded acapella/instrumental files to existing recordings.

PROPOSE, DON'T WRITE. This emits a `proposed_matches.csv` (file -> recording_id,
score, audio verdict, accept/abstain) for human review. It never touches the DB.

Pipeline (mirrors the repo's identity stack):

  Stage 1  filename  -> (artist, title, version, remixer, stem, variant, provenance)
           reuses tokenizer.identity_axes.derive_claimed_stem + core.identity
  Stage 2  metadata  -> top-K candidate recordings (normalized artist+title rank,
           version/remixer bonus) against the canonical recording/work tables
  Stage 3  audio     -> CONFIRM the metadata guess (labels lie — verify the waveform):
             instrumental -> chromaprint similarity vs candidate reference audio
             acappella    -> HuBERT embed vs candidate's separated mix-vocals
           (only runs with --verify and when the candidate audio is reachable)
  Stage 4  decision  -> accept if top margin clears --margin, else abstain

Usage
-----
  # fast metadata-only proposal (no DB writes, no audio):
  python scripts/match_stem_library.py --src ~/discord_stems --db <canonical.db> \
      --out ~/discord_stems/proposed_matches.csv

  # add the audio-confirmation gate (run where the candidate audio is reachable):
  python scripts/match_stem_library.py --src ~/discord_stems --db <canonical.db> \
      --verify --audio-root /mnt/storage --out proposed_matches.csv

The canonical DB lives on pi-storage; the local data/db copy lacks recording/work.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

# repo imports (run from repo root with venvs/audio/bin/python)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.db import connect  # noqa: E402
from core.identity import normalize_version  # noqa: E402
from tokenizer.identity_axes import derive_claimed_stem  # noqa: E402

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".opus"}

# leading DJ annotations the corpus uses: "5A 124BPM ", "126 BPM - ", "8B 130 "
_LEAD_TAG = re.compile(r"^\s*(?:\d{1,2}[ab]?\s*)?(?:\d{2,3}\s*bpm)\s*[-\s]*", re.I)
_LEAD_CAMELOT = re.compile(r"^\s*\d{1,2}[ab]\s*[-\s]+", re.I)
# qualifier parens to strip off the title once axes are extracted
_QUALIFIER = re.compile(
    r"\s*[\(\[]\s*[^)\]]*?"
    r"(instrumental|acapp?ella|acapela|extended|studio|official|diy|uvr|stems?|"
    r"multitrack|vocals?|inst\.?|vox)"
    r"[^)\]]*?[\)\]]",
    re.I,
)
_REMIX_PAREN = re.compile(
    r"[\(\[]\s*([^)\]]*?\b(?:remix|rework|bootleg|edit|flip|vip|mashup)\b[^)\]]*?)[\)\]]",
    re.I,
)
_PROVENANCE = re.compile(
    r"\b(official|studio|diy|uvr|lalal\w*|remixsearch|filtered)\b", re.I
)
# leading playlist track-number index: "01 - ", "1 ", "03 Adrien..."
_LEAD_TRACKNO = re.compile(r"^\s*\d{1,3}\s*[-._)]*\s+(?=\D)")
_REMIX_BARE = re.compile(r"\b(remix|rework|bootleg|flip|vip|mashup)\b", re.I)
# everything from the first qualifier/version keyword onward is NOT part of the title
_CUT_AT_QUAL = re.compile(
    r"\b(instrumental|instr|inst|acapp?ella|acapela|vocals?|vox|karaoke|"
    r"extended|official|studio|diy|uvr|lalal\w*|remixsearch|filtered|stems?|"
    r"multitrack|remix|rework|bootleg|flip|vip|mashup|radio\s+edit)\b",
    re.I,
)


def detect_stem(text: str) -> str:
    """Robust stem detection on whitespace-normalized text (underscore-safe)."""
    t = text.lower()
    if re.search(r"\b(instrumental|instr|inst)\b", t):
        return "instrumental"
    if re.search(r"\b(acapp?ella|acapela|karaoke|vocals?|vox)\b", t):
        return "acappella"
    return "regular"


# --------------------------------------------------------------------------- #
# Stage 1: filename -> parsed identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Parsed:
    path: str
    raw: str
    artist: str
    title: str
    version: str  # original|remix|rework|...
    remixer: str  # version_artist, "" if none
    stem: str  # regular|acappella|instrumental
    variant: str  # regular|extended
    provenance: str  # official|studio|diy|uvr|... or ""


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(feat|ft|featuring|with)\.?\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def parse_filename(path: Path) -> Parsed:
    raw = path.stem
    s = raw.replace("_", " ").replace("–", "-").replace("—", "-")
    s = _LEAD_TAG.sub("", s)
    s = _LEAD_CAMELOT.sub("", s)
    s = _LEAD_TRACKNO.sub("", s)
    s = " ".join(s.split())

    stem = detect_stem(s)  # underscore-safe; derive_claimed_stem misses "X_Vocals"
    variant = "extended" if re.search(r"\bextended\b", s, re.I) else "regular"
    prov = ""
    mp = _PROVENANCE.search(raw)
    if mp:
        prov = mp.group(1).lower()

    # version + remixer: reliable from parens, version-only from a bare keyword
    remixer = ""
    version = "original"
    mr = _REMIX_PAREN.search(s)
    if mr:
        inner = mr.group(1).strip()
        vm = re.search(r"\b(remix|rework|bootleg|edit|flip|vip|mashup)\b", inner, re.I)
        if vm:
            remixer = inner[: vm.start()].strip(" -")
            version = normalize_version(vm.group(1)) or "remix"
    elif _REMIX_BARE.search(s):
        version = normalize_version(_REMIX_BARE.search(s).group(1)) or "remix"

    # title = everything before the first qualifier/version keyword (paren or not)
    core = re.sub(r"[\(\[][^)\]]*[\)\]]", " ", s)  # drop all parenthetical groups
    cut = _CUT_AT_QUAL.search(core)
    if cut:
        core = core[: cut.start()]
    core = " ".join(core.split()).strip(" -")

    if " - " in core:
        artist, title = core.split(" - ", 1)
    elif " -" in core or "- " in core:
        artist, title = re.split(r"\s*-\s*", core, 1)
    else:
        artist, title = "", core
    return Parsed(
        path=str(path),
        raw=raw,
        artist=artist.strip(),
        title=title.strip(),
        version=version,
        remixer=remixer.strip(),
        stem=stem,
        variant=variant,
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# Stage 2: metadata candidate retrieval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Recording:
    recording_id: str
    version: str
    version_artist: str
    stem: str
    variant: str
    artist: str  # from work.artists_json / full_name
    title: str  # work.title


def load_recordings(db_path: Path) -> list[Recording]:
    out: list[Recording] = []
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.recording_id, r.version, r.version_artist, r.stem, r.variant,
                   w.title AS work_title, w.artists_json, w.full_name AS work_full
            FROM recording r JOIN work w ON r.work_id = w.work_id
            """
        ).fetchall()
    for r in rows:
        artists = ""
        try:
            arr = json.loads(r["artists_json"] or "[]")
            artists = ", ".join(
                a if isinstance(a, str) else a.get("name", "") for a in arr
            )
        except Exception:
            pass
        if not artists and r["work_full"] and " - " in r["work_full"]:
            artists = r["work_full"].split(" - ", 1)[0]
        out.append(
            Recording(
                recording_id=r["recording_id"],
                version=r["version"] or "original",
                version_artist=r["version_artist"] or "",
                stem=r["stem"] or "regular",
                variant=r["variant"] or "regular",
                artist=artists,
                title=r["work_title"] or "",
            )
        )
    return out


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def score(p: Parsed, r: Recording) -> float:
    na, nt = _norm(p.artist), _norm(p.title)
    ra, rt = _norm(r.artist), _norm(r.title)
    title_s = 0.6 * _ratio(nt, rt) + 0.4 * _jaccard(nt, rt)
    artist_s = 0.6 * _ratio(na, ra) + 0.4 * _jaccard(na, ra)
    base = 0.62 * title_s + 0.38 * artist_s
    # bonus: version + remixer agreement
    if p.version == r.version and p.version != "original":
        base += 0.06
    if (
        p.remixer
        and r.version_artist
        and _ratio(_norm(p.remixer), _norm(r.version_artist)) > 0.7
    ):
        base += 0.06
    return min(base, 1.0)


def candidates(
    p: Parsed, recs: list[Recording], k: int
) -> list[tuple[Recording, float]]:
    scored = [(r, score(p, r)) for r in recs]
    scored.sort(key=lambda rs: rs[1], reverse=True)
    return scored[:k]


# --------------------------------------------------------------------------- #
# Stage 3: audio confirmation (only with --verify; needs reachable audio)
# --------------------------------------------------------------------------- #


def verify_instrumental(file: Path, ref_audio: Path) -> tuple[Optional[float], str]:
    """Chromaprint similarity of the file vs the candidate's reference audio."""
    try:
        from ingest.adapters.fingerprint import fingerprint_file, similarity
        from core.result import Ok
    except Exception as e:
        return None, f"fingerprint import failed: {e}"
    fa, fb = fingerprint_file(str(file)), fingerprint_file(str(ref_audio))
    if not (isinstance(fa, Ok) and isinstance(fb, Ok)):
        return None, "fingerprint compute failed"
    return float(similarity(fa.value.raw, fb.value.raw)), "chromaprint"


def verify_acappella(file: Path, vocals_stem: Path) -> tuple[Optional[float], str]:
    """HuBERT cosine of the file vs the candidate's separated mix-vocals."""
    try:
        import numpy as np
        import librosa
        from workspaces.section_hsmm.similarity_probe import _hubert
    except Exception as e:
        return None, f"hubert import failed: {e}"
    try:
        ya, _ = librosa.load(str(file), sr=22050, mono=True)
        yb, _ = librosa.load(str(vocals_stem), sr=22050, mono=True)
        ea, eb = _hubert(ya, 9).mean(axis=1), _hubert(yb, 9).mean(axis=1)
        cos = float(ea @ eb / ((np.linalg.norm(ea) * np.linalg.norm(eb)) or 1.0))
        return cos, "hubert-L9"
    except Exception as e:
        return None, f"hubert compute failed: {e}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def iter_audio(src: Path) -> Iterable[Path]:
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--src", type=Path, required=True, help="dir of downloaded audio (recursed)"
    )
    ap.add_argument(
        "--db",
        type=Path,
        help="canonical DB (recording/work) — required unless --parse-only",
    )
    ap.add_argument("--out", type=Path, default=Path("proposed_matches.csv"))
    ap.add_argument("--topk", type=int, default=3, help="candidate recordings per file")
    ap.add_argument(
        "--accept", type=float, default=0.80, help="metadata score to auto-accept"
    )
    ap.add_argument(
        "--margin", type=float, default=0.08, help="min gap top1-top2 to accept"
    )
    ap.add_argument(
        "--parse-only",
        action="store_true",
        help="Stage 1 only (no DB) — for testing the parser",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="run Stage 3 audio gate (needs --audio-root)",
    )
    ap.add_argument("--audio-root", type=Path, default=Path("/mnt/storage"))
    args = ap.parse_args()

    files = list(iter_audio(args.src))
    print(f"found {len(files)} audio files under {args.src}", file=sys.stderr)
    parsed = [parse_filename(f) for f in files]

    if args.parse_only:
        w = csv.writer(sys.stdout)
        w.writerow(
            [
                "artist",
                "title",
                "version",
                "remixer",
                "stem",
                "variant",
                "provenance",
                "file",
            ]
        )
        for p in parsed:
            w.writerow(
                [
                    p.artist,
                    p.title,
                    p.version,
                    p.remixer,
                    p.stem,
                    p.variant,
                    p.provenance,
                    Path(p.path).name,
                ]
            )
        return

    if not args.db:
        sys.exit("need --db (canonical recording/work DB) unless --parse-only")
    recs = load_recordings(args.db)
    print(f"loaded {len(recs)} recordings from {args.db}", file=sys.stderr)

    rows = []
    for p in parsed:
        cands = candidates(p, recs, args.topk)
        top, top_s = cands[0] if cands else (None, 0.0)
        second_s = cands[1][1] if len(cands) > 1 else 0.0
        margin = top_s - second_s
        audio_score, audio_note = None, ""
        if args.verify and top is not None:
            # candidate reference audio / vocals stem paths (best-effort)
            obj = args.audio_root / "objects" / top.recording_id
            ref = (
                next(iter(obj.glob(f"{top.recording_id}__*")), None)
                if obj.exists()
                else None
            )
            if p.stem == "instrumental" and ref:
                audio_score, audio_note = verify_instrumental(Path(p.path), ref)
            elif p.stem == "acappella" and ref:
                # vocals stem lives at stems/{track_audio_id}/vocals.* — resolved upstream;
                # here we fall back to the reference if a separated vocal isn't located
                audio_score, audio_note = verify_acappella(Path(p.path), ref)
            else:
                audio_note = "no candidate audio on disk"
        decision = "abstain"
        if top_s >= args.accept and margin >= args.margin:
            decision = "accept"
        elif top_s >= args.accept * 0.75:
            decision = "review"
        rows.append(
            {
                **asdict(p),
                "file": Path(p.path).name,
                "cand_recording_id": top.recording_id if top else "",
                "cand_artist": top.artist if top else "",
                "cand_title": top.title if top else "",
                "cand_version": top.version if top else "",
                "score": round(top_s, 3),
                "margin": round(margin, 3),
                "audio_score": round(audio_score, 3) if audio_score is not None else "",
                "audio_note": audio_note,
                "decision": decision,
            }
        )

    fields = [
        "file",
        "artist",
        "title",
        "version",
        "remixer",
        "stem",
        "variant",
        "provenance",
        "cand_recording_id",
        "cand_artist",
        "cand_title",
        "cand_version",
        "score",
        "margin",
        "audio_score",
        "audio_note",
        "decision",
        "path",
        "raw",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    from collections import Counter

    print("decisions:", dict(Counter(r["decision"] for r in rows)), file=sys.stderr)
    print(f"wrote {len(rows)} proposals -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

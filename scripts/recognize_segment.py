#!/usr/bin/env python3
"""Recognize a DJ-mix segment via a music-recognition API (ACRCloud / AudD).

Prototype for the open-set alignment end-state (docs/open_set_alignment_endstate.md):
the aligner flags segments it can't match to our tracklist/corpus as "unknown";
this hands such a segment to an external recognizer to get a candidate ID for the
annotator to confirm. Pluggable provider so we can A/B ACRCloud vs AudD (and not
get locked to one). Reads credentials from .env.

Immediate use: ID the real-but-missing tracks in live sets (e.g. the Murph gaps)
instead of Shazaming each by hand.

.env keys:
  ACRCLOUD_IDENTIFY_HOST   e.g. identify-eu-west-1.acrcloud.com
  ACRCLOUD_ACCESS_KEY
  ACRCLOUD_ACCESS_SECRET
  AUDD_API_TOKEN

Usage:
  venvs/audio/bin/python scripts/recognize_segment.py --set-id pwgrrb1 --start 600 --dur 12
  venvs/audio/bin/python scripts/recognize_segment.py --audio /path/mix.m4a --start 600 --provider audd
  # validate the cleaner stem instead of the full mush:
  venvs/audio/bin/python scripts/recognize_segment.py --set-id 2nvzlh2k --start 600 --stem mix_instrumental
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency in venvs/audio)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _aligning_dir(set_id: str) -> Path:
    hits = sorted((Path.home() / "aligning").glob(f"{set_id}__*"))
    if not hits:
        sys.exit(f"no ~/aligning folder for {set_id}")
    return hits[0]


def extract_segment(audio: Path, start: float, dur: float, out: Path) -> None:
    """Mono 8 kHz mp3 clip — plenty for fingerprinting, small upload."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(dur), "-i", str(audio),
         "-ac", "1", "-ar", "8000", "-f", "mp3", str(out)],
        check=True, capture_output=True,
    )


def acrcloud(sample: bytes, host: str, key: str, secret: str) -> dict:
    import requests
    ts = str(int(time.time()))
    string_to_sign = "\n".join(["POST", "/v1/identify", key, "audio", "1", ts])
    sig = base64.b64encode(
        hmac.new(secret.encode(), string_to_sign.encode(), hashlib.sha1).digest()
    ).decode()
    r = requests.post(
        f"https://{host}/v1/identify",
        files={"sample": sample},
        data={"access_key": key, "data_type": "audio", "signature_version": "1",
              "signature": sig, "sample_bytes": len(sample), "timestamp": ts},
        timeout=25,
    )
    return r.json()


def audd(sample: bytes, token: str) -> dict:
    import requests
    r = requests.post(
        "https://api.audd.io/",
        data={"api_token": token, "return": "apple_music,spotify"},
        files={"file": sample}, timeout=25,
    )
    return r.json()


def _summarize(provider: str, resp: dict) -> str:
    if provider == "acrcloud":
        status = resp.get("status", {})
        if status.get("code") != 0:
            return f"  no match / error: {status.get('msg')} (code {status.get('code')})"
        out = []
        for m in (resp.get("metadata", {}).get("music") or [])[:3]:
            arts = ", ".join(a.get("name", "") for a in m.get("artists", []))
            out.append(f"  {m.get('score','?'):>3} score  {arts} - {m.get('title','')}  "
                       f"[{(m.get('album') or {}).get('name','')}]")
        return "\n".join(out) or "  (no music in metadata)"
    # audd
    if resp.get("status") != "success" or not resp.get("result"):
        return f"  no match ({resp.get('status')})"
    r = resp["result"]
    return f"  {r.get('artist','')} - {r.get('title','')}  [{r.get('album','')}]"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--set-id", help="resolve mix from ~/aligning/<set_id>__*/")
    src.add_argument("--audio", type=Path, help="explicit audio file")
    p.add_argument("--stem", default="mix", help="mix | mix_instrumental | mix_vocals")
    p.add_argument("--start", type=float, required=True, help="segment start (s)")
    p.add_argument("--dur", type=float, default=12.0)
    p.add_argument("--provider", choices=("acrcloud", "audd", "both"), default="acrcloud")
    args = p.parse_args(argv)

    _load_env(_REPO / ".env")

    if args.audio:
        audio = args.audio
    else:
        d = _aligning_dir(args.set_id)
        cands = sorted(d.glob(f"{args.stem}.*"))
        if not cands:
            sys.exit(f"no {args.stem}.* in {d}")
        audio = cands[0]

    fd, tmp = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    tmp = Path(tmp)
    keep = False
    try:
        extract_segment(audio, args.start, args.dur, tmp)
        sample = tmp.read_bytes()
        print(f"=== recognize {audio.name} @ {args.start:.0f}s +{args.dur:.0f}s "
              f"({len(sample)//1024} KB) ===")

        providers = ("acrcloud", "audd") if args.provider == "both" else (args.provider,)
        ran = False
        for prov in providers:
            if prov == "acrcloud":
                host = os.getenv("ACRCLOUD_IDENTIFY_HOST") or os.getenv("ACRCLOUD_HOST")
                key = os.getenv("ACRCLOUD_ACCESS_KEY")
                sec = (os.getenv("ACRCLOUD_ACCESS_SECRET") or os.getenv("ACRCLOUD_SECRET_KEY")
                       or os.getenv("ACRCLOUD_SECRET"))
                if not all((host, key, sec)):
                    miss = [n for n, v in (("host", host), ("access_key", key),
                                           ("secret", sec)) if not v]
                    print(f"[acrcloud] missing in .env: {', '.join(miss)} — skipping "
                          "(host = ACRCLOUD_IDENTIFY_HOST, e.g. identify-eu-west-1.acrcloud.com)")
                    continue
                ran = True
                print("[acrcloud]")
                print(_summarize("acrcloud", acrcloud(sample, host, key, sec)))
            else:
                tok = os.getenv("AUDD_API_TOKEN")
                if not tok:
                    print("[audd] missing AUDD_API_TOKEN in .env — skipping")
                    continue
                ran = True
                print("[audd]")
                print(_summarize("audd", audd(sample, tok)))
        if not ran:
            keep = True
            print(f"\nNo credentials set. Segment extracted OK -> {tmp} (wiring works).\n"
                  "Add ACRCLOUD_* / AUDD_API_TOKEN to .env, then re-run.")
    finally:
        if not keep:
            tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

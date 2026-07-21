#!/usr/bin/env python3
"""Serve a spectrogram_review out/ folder over HTTP so browser audio works.

Uses a Range-capable handler — Safari/Chrome HTML5 <audio> sends
``Range: bytes=…`` and SimpleHTTPRequestHandler's 200-full-body responses
break seeking / can stall playback.

Usage:
    venvs/audio/bin/python -m eda.alignment.spectrogram_review.serve \\
        --dir eda/alignment/spectrogram_review/out/1fsnxchk_all
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTP + Accept-Ranges / 206 Partial Content for media files."""

    protocol_version = "HTTP/1.1"

    extensions_map = {
        **getattr(SimpleHTTPRequestHandler, "extensions_map", {}),
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".png": "image/png",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".js": "text/javascript",
    }

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        # no-store: agents regenerate index.html often; browsers were serving stale JS
        self.send_header("Cache-Control", "no-store, max-age=0")
        # Helpful when opening from another origin during debugging
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def send_head(self):  # type: ignore[override]
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        size = fs.st_size
        range_header = self.headers.get("Range")

        if not range_header:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f

        # bytes=start-end
        try:
            units, _, rng = range_header.partition("=")
            if units.strip() != "bytes":
                raise ValueError("only bytes ranges")
            start_s, _, end_s = rng.strip().partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            if start < 0 or end >= size or start > end:
                raise ValueError("bad range")
        except ValueError:
            f.close()
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        length = end - start + 1
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        # Wrap so copyfile reads only `length` bytes
        return _SizedFile(f, length)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class _SizedFile:
    def __init__(self, f, length: int) -> None:
        self._f = f
        self._left = length

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        if n < 0 or n > self._left:
            n = self._left
        data = self._f.read(n)
        self._left -= len(data)
        return data

    def close(self) -> None:
        self._f.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)

    root = args.dir.resolve()
    if not (root / "index.html").is_file():
        print(f"no index.html in {root}", file=sys.stderr)
        return 1

    handler = partial(RangeRequestHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/index.html"
    print(f"serving {root}")
    print(f"open {url}")
    print("(Range/206 enabled for <audio> seeking)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

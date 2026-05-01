"""Single source of truth for the BeautifulSoup parser used across tokenizer/.

`lxml` is roughly 3x faster than the stdlib `html.parser` on the Pi 4 for
1001tracklists row HTML. Falls back gracefully if lxml isn't installed
(e.g. in a fresh dev env) so existing imports don't break.
"""
from __future__ import annotations

try:
    import lxml  # noqa: F401  -- presence check only
    BS_PARSER = "lxml"
except ImportError:
    BS_PARSER = "html.parser"

__all__ = ["BS_PARSER"]

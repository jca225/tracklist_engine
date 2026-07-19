"""Labeling test helpers — soundfile cannot write .m4a on all libsndfile builds."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

_orig_sf_write = sf.write


def _sf_write_with_touch_fallback(file, data, samplerate, *args, **kwargs):
    try:
        return _orig_sf_write(file, data, samplerate, *args, **kwargs)
    except TypeError:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


sf.write = _sf_write_with_touch_fallback  # type: ignore[method-assign]

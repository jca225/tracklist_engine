"""Collection guards for the pws_aligner suite.

requirements-ci.txt deliberately excludes the heavy audio stack and asks that
tests needing it "lazy-import or pytest.importorskip". That instruction cannot
be followed from inside these particular test modules: the audio-analysis LFs
(``pws_aligner/lf/fx.py``, ``pws_aligner/lf/operations.py``) do ``import
librosa`` at MODULE scope, so importing the test module raises before any
``pytest.importorskip`` line in it could run. The failure is at collection, not
at test time — so the guard has to be at collection too.

Locally, ``venvs/audio`` has librosa and every module below collects and runs,
so ``make check`` still covers them in full. Only the slim CI job skips them.

If you make fx/operations lazy-import librosa, delete the matching entries here.
"""

from __future__ import annotations

import importlib.util

# Modules whose import chain reaches `import librosa` at module scope.
_NEEDS_LIBROSA = [
    "lf/test_filter_sweep_lf.py",
    "lf/test_fx_lf.py",
    "lf/test_lf_runner.py",
    "lf/test_loop_periodicity_lf.py",
    "lf/test_noise_sweep_lf.py",
    "lf/test_operations.py",
    "lf/test_sidechain_pump_lf.py",
    "quality/test_stem_routing.py",
]

collect_ignore_glob: list[str] = []

if importlib.util.find_spec("librosa") is None:
    collect_ignore_glob += _NEEDS_LIBROSA

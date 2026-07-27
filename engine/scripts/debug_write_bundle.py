"""Debug entry: Propose step → staging/vertical_lf_bundle.json.

Run from core/ (cwd). Set breakpoints in sensors.lfs.*.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow debug without editable install (venv may be missing dj_sensors).
_SENSORS_ROOT = Path(__file__).resolve().parents[1] / "python" / "dj_sensors"
if str(_SENSORS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SENSORS_ROOT))

from sensors.lfs import write_vertical_slice


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "fixtures" / "gold"
    staging = root / "staging"
    out = write_vertical_slice(fixtures, staging)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()

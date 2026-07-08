"""Bridge the synthetic-mix generator output into the trajectory Dataset.

`synthetic_mix.generate_v2` writes, per window `synthv2_NNNN/`:

    mix.flac  mix_vocals.flac  mix_instrumental.flac
    refs/<rid>_{vocals,instrumental,regular}.flac
    ground_truth.yaml            # set_id == synthv2_NNNN

`TrajectorySpanDataset` (via `features.resolve_span_audio`) instead expects a
pulled `~/aligning/<set>/` layout: a `manifest.json` mapping track_id ->
`local_path`, mix stems at the top level (`find_mix_stems`), and ref stems at
`stems/<ref-basename>/{vocals,instrumental}.flac` (`ref_stem_path`).

This module materialises that layout **in place** inside each synth dir — a
`manifest.json` plus `stems/` symlinks into `refs/` — so the exact same
resolver serves synthetic and real spans with no synthetic-specific code on the
hot path. Materialisation is idempotent.

Synthetic is **train-only**: `build_synthetic_sets` returns the
`(sets, aligning_dirs)` pair to hand a `TrajectorySpanDataset`, never the eval
set. Scoring stays on real hand GT.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

# claimed_stem -> (ref filename suffix in refs/, stem-dir key for ref_stem_path).
# regular carries no stem key: its span resolves against the full ref directly.
_ROLE = {
    "acappella": ("vocals", "vocals"),
    "instrumental": ("instrumental", "instrumental"),
    "regular": ("regular", None),
}


def _ref_file(win_dir: Path, tid: str, claimed_stem: str) -> Path | None:
    suffix, _ = _ROLE.get(claimed_stem, _ROLE["regular"])
    p = win_dir / "refs" / f"{tid}_{suffix}.flac"
    return p if p.is_file() else None


def materialize_window(win_dir: Path) -> str | None:
    """Write manifest.json + stems/ symlinks for one synth window.

    Returns None on success, else a skip reason. Idempotent.
    """
    gt_path = win_dir / "ground_truth.yaml"
    if not gt_path.is_file():
        return "no ground_truth.yaml"
    gt = yaml.safe_load(gt_path.read_text())
    if not (win_dir / "mix.flac").is_file():
        return "no mix.flac"

    tracks: dict[str, dict] = {}
    stems_root = win_dir / "stems"
    for row in gt.get("tracks", []):
        tid = str(row.get("track_id") or "")
        cs = str(row.get("claimed_stem") or "regular")
        if not tid:
            continue
        ref = _ref_file(win_dir, tid, cs)
        if ref is None:
            continue  # a missing ref surfaces later as a LOUD dataset skip
        tracks.setdefault(tid, {"track_id": tid, "local_path": str(ref.resolve())})
        # stem-routed roles need stems/<ref-basename>/<key>.flac (ref_stem_path)
        _, stem_key = _ROLE.get(cs, _ROLE["regular"])
        if stem_key is not None:
            link_dir = stems_root / ref.stem
            link_dir.mkdir(parents=True, exist_ok=True)
            link = link_dir / f"{stem_key}.flac"
            if not link.exists():
                link.symlink_to(ref.resolve())

    if not tracks:
        return "no resolvable ref tracks"

    manifest = {
        "mix_local_path": str((win_dir / "mix.flac").resolve()),
        "tracks": list(tracks.values()),
    }
    (win_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return None


def build_synthetic_sets(
    root: Path | str,
) -> tuple[list[tuple[str, Path]], dict[str, Path]]:
    """Materialise every `synthv2_*` window under `root`.

    Returns `(sets, aligning_dirs)` for `TrajectorySpanDataset(sets, ...,
    aligning_dirs=...)`. Windows that can't be served are skipped with a printed
    reason (never silently dropped).
    """
    root = Path(root)
    win_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("synthv2_")
    )
    sets: list[tuple[str, Path]] = []
    aligning_dirs: dict[str, Path] = {}
    for win in win_dirs:
        reason = materialize_window(win)
        if reason is not None:
            print(f"  SKIP synth {win.name}: {reason}")
            continue
        set_id = yaml.safe_load((win / "ground_truth.yaml").read_text())["set_id"]
        sets.append((set_id, win / "ground_truth.yaml"))
        aligning_dirs[set_id] = win
    return sets, aligning_dirs

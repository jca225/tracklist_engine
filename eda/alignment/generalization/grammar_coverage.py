#!/usr/bin/env python3
"""Grammar-coverage fingerprint — pick co-training sets by DJ-move coverage, not popularity.

The aligner's training substrate is popularity-seeded (EDM/mashup-skewed). This
tool fingerprints every scraped set by metadata proxies for the DJ-move grammar,
compares the *downloaded* corpus against the *full* corpus to find the starved
grammar corners, and ranks ingestable-but-undownloaded sets by how much each one
fills those corners.

Sibling of ``rank_ingest_queue.py`` (reuses its ``_fold`` + fetchability idea) but
scores on grammar coverage instead of popularity. Read-only; emits artifacts only
(no downloads, no canonical mutation). Design:
``docs/superpowers/specs/2026-07-16-grammar-coverage-fingerprint-design.md``.

Usage:
    venvs/audio/bin/python -m eda.alignment.generalization.grammar_coverage [--top 2000]
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"
PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"

# Stratification axes (strongest grammar proxies); the rest ride as descriptive.
STRAT_AXES = ("w_frac", "version_frac", "stem_frac", "density")
JOINT_AXES = ("w_frac", "version_frac", "stem_frac")
MIN_CELL = 20  # joint cells below this corpus mass are scrape noise, not grammar gaps

_PT_H = re.compile(r"(\d+)\s*h")
_PT_M = re.compile(r"(\d+)\s*m")


def parse_play_time(s: str) -> float | None:
    """'1h 28m' / '59m' / '1h' -> minutes; '' / unparseable -> None."""
    if not s:
        return None
    h = _PT_H.search(s)
    m = _PT_M.search(s)
    if not h and not m:
        return None
    return 60.0 * (int(h.group(1)) if h else 0) + (int(m.group(1)) if m else 0)


@dataclass(frozen=True)
class SetFingerprint:
    set_id: str
    title: str
    n_slots: int
    total_tracks: int
    play_time_min: float | None
    w_frac: float
    version_frac: float
    stem_frac: float
    id_frac: float
    mean_cue_gap: float | None
    styles: str
    has_audio: bool
    fetchable: bool

    @property
    def density(self) -> float | None:
        if not self.play_time_min:
            return None
        return self.total_tracks / self.play_time_min


# ---------------------------------------------------------------------------
# Binning — semantic bins for zero-inflated fractions; quantile bins for density.
# ---------------------------------------------------------------------------


def frac_bin(x: float) -> str:
    """4-way semantic bin for a [0,1] grammar fraction."""
    if x <= 0.0:
        return "none"
    if x <= 0.1:
        return "low"
    if x <= 0.3:
        return "mid"
    return "high"


def frac_bin3(x: float) -> str:
    """3-way collapse for the joint grid (keeps 3^3=27 cells)."""
    if x <= 0.0:
        return "none"
    if x <= 0.15:
        return "low"
    return "high"


def quantile_edges(values: list[float], n: int = 4) -> list[float]:
    """n-quantile inner edges of `values` (sorted); empty -> []."""
    xs = sorted(values)
    if not xs:
        return []
    return [xs[min(len(xs) - 1, int(len(xs) * k / n))] for k in range(1, n)]


def density_bin(x: float | None, edges: list[float]) -> str:
    if x is None:
        return "na"
    b = 0
    for e in edges:
        if x > e:
            b += 1
        else:
            break
    return f"q{b}"


def axis_bin(fp: SetFingerprint, axis: str, dens_edges: list[float]) -> str:
    if axis == "density":
        return density_bin(fp.density, dens_edges)
    return frac_bin(getattr(fp, axis))


def joint_cell(fp: SetFingerprint) -> tuple[str, str, str]:
    return tuple(frac_bin3(getattr(fp, a)) for a in JOINT_AXES)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Coverage — downloaded vs full-corpus share, per bin and per joint cell.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coverage:
    """Per (axis,bin) and per joint-cell fraction of that group already downloaded.

    `fill_weight` is a bounded *starvation* score (0..~5), not an inverse — an
    unbounded `1/(cov+eps)` let empty cells dominate at ~1e6 and made the ranking
    a binary "is this cell empty" with weak tie-breaks. Starvation = `1 - coverage`
    is interpretable and comparable across axes. The joint term is **mass-gated**:
    a zero-coverage cell holding 2 sets is scrape noise, not a grammar gap, so it
    contributes only when the cell has real corpus mass (`>= MIN_CELL`).
    """

    dens_edges: list[float]
    per_axis: dict[str, dict[str, float]]  # axis -> bin -> downloaded/corpus fraction
    joint: dict[tuple[str, str, str], float]
    joint_n: dict[tuple[str, str, str], int]  # corpus count per joint cell

    def _axis_covs(self, fp: SetFingerprint) -> list[float]:
        """Downloaded-coverage per applicable strat bin. `na` density (missing
        play_time) is a data gap, not a grammar corner — skip it so a set isn't
        rewarded for lacking metadata."""
        covs = []
        for a in STRAT_AXES:
            b = axis_bin(fp, a, self.dens_edges)
            if a == "density" and b == "na":
                continue
            covs.append(self.per_axis[a][b])
        return covs

    def bin_coverage(self, fp: SetFingerprint) -> float:
        """Mean downloaded-coverage of the strat bins this set falls in (low = starved)."""
        covs = self._axis_covs(fp)
        return sum(covs) / len(covs)

    def fill_weight(self, fp: SetFingerprint) -> float:
        """Bounded starvation: mean per-axis (1-coverage) + mass-gated joint (1-coverage)."""
        covs = self._axis_covs(fp)
        w = sum(1.0 - c for c in covs) / len(covs)  # 0..1
        cell = joint_cell(fp)
        if self.joint_n.get(cell, 0) >= MIN_CELL:
            w += 1.0 - self.joint.get(cell, 0.0)  # +0..1 for a real starved corner
        return round(w, 4)


def build_coverage(rows: list[SetFingerprint]) -> Coverage:
    dens_edges = quantile_edges([r.density for r in rows if r.density is not None])
    dl = [r for r in rows if r.has_audio]

    per_axis: dict[str, dict[str, float]] = {}
    for axis in STRAT_AXES:
        corpus = Counter(axis_bin(r, axis, dens_edges) for r in rows)
        got = Counter(axis_bin(r, axis, dens_edges) for r in dl)
        per_axis[axis] = {b: got.get(b, 0) / c for b, c in corpus.items()}

    corpus_j = Counter(joint_cell(r) for r in rows)
    got_j = Counter(joint_cell(r) for r in dl)
    joint = {c: got_j.get(c, 0) / n for c, n in corpus_j.items()}
    return Coverage(dens_edges, per_axis, joint, dict(corpus_j))


# ---------------------------------------------------------------------------
# I/O edge — one read-only aggregate query against the canonical DB.
# ---------------------------------------------------------------------------

_SQL = """
WITH slot_agg AS (
  SELECT set_id,
         COUNT(*) AS n_slots,
         AVG(CASE WHEN is_concurrent=1 THEN 1.0 ELSE 0.0 END) AS w_frac,
         AVG(CASE WHEN claimed_version IS NOT NULL
                   AND claimed_version NOT IN ('original','') THEN 1.0 ELSE 0.0 END) AS version_frac,
         AVG(CASE WHEN claimed_stem IN ('acappella','instrumental') THEN 1.0 ELSE 0.0 END) AS stem_frac,
         AVG(CASE WHEN title LIKE 'ID%' THEN 1.0 ELSE 0.0 END) AS id_frac,
         COUNT(cue_time_seconds) AS n_cued,
         MAX(cue_time_seconds) AS cue_max,
         MIN(cue_time_seconds) AS cue_min
  FROM set_track_slots GROUP BY set_id HAVING n_slots >= 5
),
audio AS (SELECT DISTINCT set_id FROM set_audio),
links AS (
  SELECT set_id, MAX(1) AS any_link
  FROM dj_set_media_links WHERE platform IN ('soundcloud','youtube') GROUP BY set_id
)
SELECT d.set_id,
       REPLACE(COALESCE(d.title,''), char(9), ' '),
       sa.n_slots, COALESCE(d.total_tracks,0), COALESCE(d.play_time,''),
       sa.w_frac, sa.version_frac, sa.stem_frac, sa.id_frac,
       sa.n_cued, COALESCE(sa.cue_max,0), COALESCE(sa.cue_min,0),
       REPLACE(COALESCE(d.styles,''), char(9), ' '),
       CASE WHEN au.set_id IS NOT NULL THEN 1 ELSE 0 END,
       COALESCE(l.any_link,0)
FROM dj_sets d
JOIN slot_agg sa ON sa.set_id=d.set_id
LEFT JOIN audio au ON au.set_id=d.set_id
LEFT JOIN links l ON l.set_id=d.set_id;
"""


def _row_to_fp(c: list[str]) -> SetFingerprint:
    n_slots = int(c[2])
    n_cued, cue_max, cue_min = int(c[9]), float(c[10]), float(c[11])
    mean_cue_gap = (cue_max - cue_min) / (n_cued - 1) if n_cued >= 2 else None
    has_audio = c[13] == "1"
    return SetFingerprint(
        set_id=c[0],
        title=c[1][:90],
        n_slots=n_slots,
        total_tracks=int(c[3]),
        play_time_min=parse_play_time(c[4]),
        w_frac=float(c[5]),
        version_frac=float(c[6]),
        stem_frac=float(c[7]),
        id_frac=float(c[8]),
        mean_cue_gap=mean_cue_gap,
        styles=c[12],
        has_audio=has_audio,
        fetchable=(c[14] == "1") and not has_audio,
    )


def fetch_rows() -> list[SetFingerprint]:
    # encoding="utf-8" is load-bearing: the pi locale is Latin-1 and unicode set
    # titles/styles would mojibake through the default (the track_audio.path bug).
    proc = subprocess.run(
        [
            "ssh",
            PI_HOST,
            f'sqlite3 -readonly -separator "\t" {PI_DB} ".timeout 60000" "{_SQL}"',
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=True,
    )
    out = []
    for line in proc.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 15:
            out.append(_row_to_fp(parts))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def thin_corners(rows: list[SetFingerprint], cov: Coverage) -> list[str]:
    lines: list[str] = []
    lines.append("--- per-axis coverage (downloaded / corpus share of each bin) ---")
    for axis in STRAT_AXES:
        binned = sorted(cov.per_axis[axis].items(), key=lambda kv: kv[1])
        cell = "  ".join(f"{b}={c:.3f}" for b, c in binned)
        lines.append(f"  {axis:>13}: {cell}")
    lines.append(
        f"--- thinnest joint cells (w,version,stem), n_corpus >= {MIN_CELL} ---"
    )
    real = {c: v for c, v in cov.joint.items() if cov.joint_n.get(c, 0) >= MIN_CELL}
    for cell, c in sorted(real.items(), key=lambda kv: kv[1])[:8]:
        lines.append(f"  {cell} coverage={c:.3f}  (n_corpus={cov.joint_n[cell]})")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=2000, help="candidate rows to write")
    args = ap.parse_args(argv)

    rows = fetch_rows()
    dl = [r for r in rows if r.has_audio]
    fetchable = [r for r in rows if r.fetchable]
    print(
        f"sets: {len(rows)}  downloaded: {len(dl)}  fetchable(undownloaded): {len(fetchable)}"
    )

    cov = build_coverage(rows)

    ranked = sorted(fetchable, key=lambda r: -cov.fill_weight(r))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "grammar_coverage.tsv"
    fields = [
        "rank",
        "fill_weight",
        "bin_coverage",
        "set_id",
        "title",
        "n_slots",
        "density",
        "w_frac",
        "version_frac",
        "stem_frac",
        "id_frac",
        "mean_cue_gap",
        "styles",
    ]
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(fields)
        for i, r in enumerate(ranked[: args.top], 1):
            w.writerow(
                [
                    i,
                    cov.fill_weight(r),
                    round(cov.bin_coverage(r), 4),
                    r.set_id,
                    r.title,
                    r.n_slots,
                    round(r.density, 3) if r.density is not None else "",
                    round(r.w_frac, 3),
                    round(r.version_frac, 3),
                    round(r.stem_frac, 3),
                    round(r.id_frac, 3),
                    round(r.mean_cue_gap, 1) if r.mean_cue_gap is not None else "",
                    r.styles[:40],
                ]
            )

    print(f"wrote top {min(args.top, len(ranked))} candidates -> {dest}\n")
    print("\n".join(thin_corners(rows, cov)))
    print("\n--- top 12 grammar-fill candidates ---")
    for i, r in enumerate(ranked[:12], 1):
        d = f"{r.density:.2f}" if r.density is not None else "  na"
        print(
            f"  {i:>2}. fill={cov.fill_weight(r):6.1f}  w={r.w_frac:.2f} ver={r.version_frac:.2f} "
            f"stem={r.stem_frac:.2f} dens={d}  {r.title[:50]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

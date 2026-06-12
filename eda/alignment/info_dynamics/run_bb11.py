"""Cross-set replication of the info-dynamics significance test on BB11.

BB11 (`2nvzlh2k`) has **no hand-labelled Ableton GT** — so the boundary source is
the scraped 1001tracklists **cue times** (per-track start seconds). That is an
*independent, human* boundary signal. Crowd-sourced cues argue for a lenient ±10 s
(measurement error), but the GT is dense (~1 boundary / 24 s), so at ±10 s random
phase already hits a boundary most of the time — the chance floor saturates and the
test loses power. **±3 s is therefore the more discriminating tolerance**; both are
reported and a cell is only convincing if it clears FDR at ±3 s.

This is the replication v5 of findings.md flagged as the real next step: does the
BB12 effect (surprise peaks localize section starts) reproduce on a *second* set,
against a *different, independent* boundary source? A positive result is stronger
evidence than BB12-vs-its-own-hand-labels.

Scope today: only the **full-mix** MERT artifact exists for BB11
(`2nvzlh2k_mix_mert.npz`). The vocals/instrumental stem cells need Demucs+MERT on
the BB11 mix (GPU) and are skipped until those artifacts exist — the runner picks
up whichever stem artifacts are present.

    venvs/audio/bin/python -m eda.alignment.info_dynamics.run_bb11

Writes data/analysis/info_dynamics_grid/bb11_robustness.md.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from .baselines import run_m1
from .continuous import pca_whiten, run_m2_continuous
from .data import study_data_from_boundaries
from .evaluate import PeakConfig, benjamini_hochberg
from .run_robustness import WARMUP, BLOCK, N_PERM, _cell

SET_ID = "2nvzlh2k"
DB = "data/db/music_database.db"
SOURCES = {  # only those present on disk are run
    "full": f"data/analysis/{SET_ID}_mix_mert.npz",
    "acappella": f"data/analysis/{SET_ID}_mix_vocals_mert.npz",
    "instrumental": f"data/analysis/{SET_ID}_mix_instrumental_mert.npz",
}
TOLERANCES = (10.0, 3.0)  # primary first — tracklist cues are coarse
OUT = Path("data/analysis/info_dynamics_grid")


def tracklist_boundaries(db_path: str, set_id: str) -> np.ndarray:
    """Per-track cue times (seconds) from the scraped tracklist rows."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT raw_html FROM dj_set_rows WHERE set_id=? AND classes LIKE '%tlpTog%' "
        "ORDER BY row_index",
        (set_id,),
    ).fetchall()
    con.close()
    cues: list[int] = []
    for (html,) in rows:
        m = re.search(r"cue:\s*'(\d+)'", html or "")
        if m:
            cues.append(int(m.group(1)))
    return np.array(sorted(set(cues)), dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PeakConfig()
    bounds = tracklist_boundaries(DB, SET_ID)
    if bounds.size == 0:
        raise SystemExit(f"no tracklist cues found for {SET_ID} in {DB}")
    print(f"[bb11] {bounds.size} tracklist cues, "
          f"{bounds.min():.0f}..{bounds.max():.0f}s, median gap "
          f"{np.median(np.diff(bounds)):.0f}s")

    cells: list[dict] = []
    for src, art in SOURCES.items():
        if not Path(art).is_file():
            print(f"[skip] {src}: {art} not present (needs Demucs+MERT on BB11 mix)")
            continue
        data = study_data_from_boundaries(art, bounds, n_tokens=24, seed=0)
        lo = float(data.bar_start_s[WARMUP])
        z = pca_whiten(data, d=64)
        models = {
            "codebook": run_m1(data),
            "continuous": run_m2_continuous(
                data, z, arch="gru", warmup=WARMUP, block=BLOCK, seed=0
            ),
        }
        for rep, ss in models.items():
            for tol in TOLERANCES:
                c = _cell(ss, data, lo, cfg, src, rep, tol=tol)
                cells.append(c)
                print(f"{src:13s} {rep:10s} ±{int(tol):2d}s  F1={c['real_f1']:.3f} "
                      f"lift={c['lift']:+.3f} p={c['p_value']:.4f} z={c['z']:.1f} "
                      f"CI[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]")

    qvals = benjamini_hochberg([c["p_value"] for c in cells])
    for c, q in zip(cells, qvals):
        c["q_value"] = float(q)

    _write(cells, bounds)


def _verdict(c: dict) -> str:
    if c["q_value"] < 0.05:
        return "✅ q<.05"
    if c["p_value"] < 0.05:
        return f"~ p<.05 q={c['q_value']:.2f}"
    return f"❌ p={c['p_value']:.2f}"


def _write(cells: list[dict], bounds: np.ndarray) -> None:
    present = sorted({c["source"] for c in cells})
    L = [
        f"# BB11 ({SET_ID}) replication — info-dynamics vs **tracklist** cue times\n",
        f"Boundary source: {bounds.size} scraped 1001tracklists cue times "
        f"({bounds.min():.0f}..{bounds.max():.0f}s, median gap "
        f"{np.median(np.diff(bounds)):.0f}s) — **no hand-label GT for BB11**. "
        "GT is dense (~1/24 s), so the wide ±10 s window saturates toward chance "
        "(random phase hits a boundary most of the time) and loses power — **±3 s is "
        "the more discriminating tolerance**. Same machinery as BB12 §v5: circular-shift "
        f"permutation ({N_PERM} perms), bootstrap 95% CI, Benjamini-Hochberg FDR "
        f"across all {len(cells)} cells.\n",
        f"Sources present: {', '.join(present)}. "
        + ("" if "instrumental" in present else
           "**Stem cells (acappella/instrumental) skipped — BB11 mix has no "
           "Demucs+MERT artifacts yet.**\n"),
        "| Source | repr. | tol | best signal | F1 | lift [95% CI] | p | q (FDR) | z | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        L.append(
            f"| {c['source']} | {c['rep']} | ±{int(c['tol_s'])}s | {c['signal']} | "
            f"{c['real_f1']:.3f} | {c['lift']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | "
            f"{c['p_value']:.4f} | {c['q_value']:.3f} | {c['z']:.1f} | {_verdict(c)} |"
        )
    L.append(
        "\n**Read-out:** a cell is a genuine cross-set replication only if it "
        "clears FDR (q < .05) at the discriminating ±3 s tolerance against the "
        "*independent* tracklist boundaries. Still n = 1 *additional* set — two "
        "sets is not yet a population claim, but a consistent BB11 + BB12 result "
        "is the first real evidence the effect generalizes beyond a single mix."
    )
    (OUT / "bb11_robustness.md").write_text("\n".join(L) + "\n")
    print(f"[done] {OUT}/bb11_robustness.md")


if __name__ == "__main__":
    main()

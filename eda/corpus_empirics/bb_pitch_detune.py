"""Why do reference tracks need pitch adjustment to sit in tune with a DJ mix?

When a set is hand-aligned in Ableton, some acappella/instrumental clips must be
transposed (whole semitones) or detuned (< a semitone, in cents) to match the
recorded mix. A calibrated cents estimator
(``workspaces/alignment_prototype/pitch_detune.py``) measures each clip's true
offset from the mix audio; ``measure_detune.py`` emits a per-clip CSV. This
script discriminates four generating mechanisms:

  H1 varispeed (keylock off)  — cents == 1200·log2(BPM_local / BPM_native)
  H2 harmonic mixing          — offsets in layered sections, move keys consonant
  H3 wrong/altered rip        — near-integer-semitone, constant across the clip
  H4 artifact / null          — vanish on clean material / fail the estimator

The per-track *unit* is one (set, slot, stem) appearance-class, represented by
its highest-confidence clip. Quantitative claims are gated on a RELIABLE subset
(cross-correlation peak ≥ ``MIN_CORR`` and not window-saturated); the estimator's
zero-point is re-checked on solo instrumentals every run.

Headline metrics persist to ``aux.analysis_results`` under
``analysis_name='bb_pitch_detune_v1'``; the narrative lives in
``eda/corpus_empirics/findings.md``.
"""

from __future__ import annotations

import csv
import math
import re
import sqlite3
import sys
from pathlib import Path

# paths are repo-root-relative (run from the project root, like the sibling bb_*.py)
CSV_PATH = Path("workspaces/alignment_prototype/out/pitch_detune_clips.csv")
AUX_DB = Path("data/analysis/aux.db")
ANALYSIS_NAME = "bb_pitch_detune_v1"

MIN_CORR = 0.30  # trust the offset only when the ref clearly localises in the mix


# ----------------------------- stat helpers --------------------------------


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def ols(xs, ys):
    """(slope, intercept, r2) of y ~ x."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sxy / sxx
    r = pearson(xs, ys)
    return slope, my - slope * mx, r * r


def median(vs):
    vs = sorted(vs)
    n = len(vs)
    if n == 0:
        return float("nan")
    return vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2


# ----------------------------- data model ----------------------------------


def fold_ratio(bpm_local: float, bpm_native: float) -> tuple[float, bool]:
    """Beatmatch-consistent tempo ratio: fold native by ×/÷2 into the octave
    nearest local (DJs beatmatch within a few %; Essentia half/double errors
    would otherwise fake ±1200¢). Returns (ratio, folded?)."""
    if not bpm_native or not bpm_local or bpm_native <= 0:
        return float("nan"), False
    r = bpm_local / bpm_native
    folded = False
    while r > 1.5:
        r /= 2.0
        folded = True
    while r < 1 / 1.5:
        r *= 2.0
        folded = True
    return r, folded


def _cam(k):
    m = re.fullmatch(r"(\d{1,2})([AB])", k or "")
    return (int(m.group(1)), m.group(2)) if m else None


def _cam_dist(a, b):
    """Camelot-wheel distance: ring steps (mod 12) + 1 if the A/B letter differs.
    0 = identical; ≤1 = harmonically compatible (relative maj/min or ±1 fifth)."""
    pa, pb = _cam(a), _cam(b)
    if not pa or not pb:
        return None
    ring = min((pa[0] - pb[0]) % 12, (pb[0] - pa[0]) % 12)
    return ring + (0 if pa[1] == pb[1] else 1)


def _transpose(k, semis):
    """Camelot key after transposing by ``semis`` semitones (+1 st = +7 hours)."""
    p = _cam(k)
    return f"{((p[0] - 1 + 7 * semis) % 12) + 1}{p[1]}" if p else None


def _overlap(a, b):
    return max(
        0.0,
        min(a["mix_end_s"], b["mix_end_s"]) - max(a["mix_start_s"], b["mix_start_s"]),
    )


def find_bed(clip, rows):
    """The instrumental/track a clip is layered over: the longest-overlapping,
    reliable, in-tune (~0¢), keyed co-track from a different song slot."""
    cands = [
        d
        for d in rows
        if d["set"] == clip["set"]
        and d["slot"] != clip["slot"]
        and d["reliable"]
        and abs(d["cents_agg"]) < 15
        and _overlap(clip, d) > 3
        and _cam(d["camelot"])
    ]
    return max(cands, key=lambda d: _overlap(clip, d)) if cands else None


def load_rows():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            for k in ("coarse", "n_cotracks", "saturated"):
                r[k] = int(r[k])
            for k in (
                "fine_dialed",
                "cents_agg",
                "residual",
                "peak_corr",
                "dur_s",
                "mix_start_s",
                "mix_end_s",
            ):
                r[k] = float(r[k])
            for k in ("bpm_native", "bpm_local"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else None
            if r["bpm_native"] and r["bpm_local"]:
                ratio, folded = fold_ratio(r["bpm_local"], r["bpm_native"])
                r["cents_pred"] = 1200.0 * math.log2(ratio) if ratio == ratio else None
                r["bpm_folded"] = folded
            else:
                r["cents_pred"] = None
                r["bpm_folded"] = False
            r["reliable"] = r["saturated"] == 0 and r["peak_corr"] >= MIN_CORR
            rows.append(r)
    return rows


def dedupe_units(rows):
    """One row per (set, slot, stem) appearance-class: the highest-corr clip."""
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["set"], r["slot"], r["stem"])
        if key not in best or r["peak_corr"] > best[key]["peak_corr"]:
            best[key] = r
    return list(best.values())


# ----------------------------- report --------------------------------------


def main() -> int:
    if not CSV_PATH.exists():
        print(f"missing {CSV_PATH}; run measure_detune.py first", file=sys.stderr)
        return 1
    all_rows = load_rows()
    units = dedupe_units(all_rows)
    findings: list[tuple[str, str, float, str, str]] = []

    def add(metric, group, val, unit="", notes=""):
        findings.append((metric, group, float(val), unit, notes))

    print(
        f"{len(all_rows)} clips → {len(units)} (set,slot,stem) units; "
        f"{sum(u['reliable'] for u in units)} reliable "
        f"(corr≥{MIN_CORR}, not saturated)\n"
    )

    # --- H4 / zero-point: solo instrumentals must read ~0 ------------------
    solo_instr = [
        u
        for u in units
        if u["reliable"]
        and u["section"] == "solo"
        and u["stem"] == "instrumental"
        and u["coarse"] == 0
    ]
    if solo_instr:
        med = median([abs(u["residual"]) for u in solo_instr])
        print(
            f"[H4/zero-point] solo instrumental |residual| median = {med:.1f}¢ "
            f"(n={len(solo_instr)}) — near 0 validates the estimator on clean audio"
        )
        add(
            "zeropoint_solo_instr_median_abs_resid",
            "all",
            med,
            "cents",
            f"n={len(solo_instr)}",
        )

    # --- counts: reproduce prior flagged tallies (reliable units) ----------
    print("\n=== detune prevalence (reliable units) ===")
    for setname in ("BB11", "BB12"):
        su = [u for u in units if u["set"] == setname and u["reliable"]]
        add("n_reliable_units", setname, len(su), "units")
        for thr in (10, 20):
            n = sum(1 for u in su if abs(u["residual"]) > thr)
            print(
                f"  {setname}: {n:3d} units |residual|>{thr}¢  (of {len(su)} reliable)"
            )
            add(f"n_residual_gt{thr}c", setname, n, "units", f"of {len(su)} reliable")

    # --- H1 varispeed ------------------------------------------------------
    print("\n=== H1 varispeed: cents_agg vs 1200·log2(BPM_local/BPM_native) ===")
    h1 = [u for u in units if u["reliable"] and u["cents_pred"] is not None]
    for label, subset in (
        ("all", h1),
        ("|pred|>15c", [u for u in h1 if abs(u["cents_pred"]) > 15]),
    ):
        if len(subset) >= 5:
            xs = [u["cents_pred"] for u in subset]
            ys = [u["cents_agg"] for u in subset]
            slope, intc, r2 = ols(xs, ys)
            # fraction whose measured offset matches the varispeed prediction (±20¢)
            match = sum(
                1 for u in subset if abs(u["cents_agg"] - u["cents_pred"]) <= 20
            )
            print(
                f"  {label:11} n={len(subset):3d}  slope={slope:+.2f}  R²={r2:.3f}"
                f"  intercept={intc:+.0f}¢  match±20¢={match}/{len(subset)}"
            )
            add("h1_slope", label, slope, "", f"n={len(subset)}")
            add("h1_r2", label, r2, "", f"n={len(subset)}")
            add("h1_frac_match_20c", label, match / len(subset), "", f"n={len(subset)}")

    # --- H2 harmonic mixing: solo vs layered detune ------------------------
    print("\n=== H2 harmonic mixing: |residual| solo vs layered (reliable) ===")
    for stem in ("acappella", "instrumental", "all"):
        for section in ("solo", "layered"):
            sub = [
                u
                for u in units
                if u["reliable"]
                and u["section"] == section
                and (stem == "all" or u["stem"] == stem)
            ]
            if sub:
                med = median([abs(u["residual"]) for u in sub])
                frac = sum(1 for u in sub if abs(u["residual"]) > 10) / len(sub)
                print(
                    f"  {stem:12} {section:8} n={len(sub):3d}  "
                    f"median|residual|={med:4.1f}¢  frac>10¢={frac:.0%}"
                )
                add(f"median_abs_resid_{section}", stem, med, "cents", f"n={len(sub)}")

    # --- H2 pairwise: is the transpose a harmonic key-match to the bed? -----
    # The mechanism behind the integer-semitone offsets: a layered acappella is
    # pitched to the KEY of the instrumental it sits over. For each coarse-shifted
    # layered acappella, compare its Camelot distance to the bed before vs after
    # the transpose. A flip from incompatible→compatible = deliberate harmonic mix.
    print("\n=== H2 pairwise: acappella transpose vs the instrumental bed's key ===")
    lay = dedupe_units(
        [
            r
            for r in all_rows
            if r["reliable"]
            and r["stem"] == "acappella"
            and r["section"] == "layered"
            and _cam(r["camelot"])
        ]
    )
    nc = toward = compat_after = compat_before = 0
    nudge_matched = 0
    for r in lay:
        b = find_bed(r, all_rows)
        if not b:
            continue
        if r["coarse"] != 0:
            nc += 1
            d0 = _cam_dist(r["camelot"], b["camelot"])
            d1 = _cam_dist(_transpose(r["camelot"], r["coarse"]), b["camelot"])
            toward += d1 < d0
            compat_after += d1 <= 1
            compat_before += d0 <= 1
        elif _cam(r["camelot"]) == _cam(b["camelot"]) and abs(r["residual"]) > 8:
            nudge_matched += 1  # fine nudge applied even when keys already agree
    if nc:
        print(f"  n={nc} coarse-transposed layered acappellas with a reliable bed")
        print(
            f"  compatible (Camelot dist≤1) with bed BEFORE transpose: {compat_before}/{nc}"
        )
        print(
            f"  compatible with bed AFTER  transpose:                  {compat_after}/{nc}"
        )
        print(f"  transpose moves the acappella TOWARD the bed's key:    {toward}/{nc}")
        print(
            f"  + {nudge_matched} sub-semitone nudges applied where key already matched bed"
        )
        add("pairwise_n_coarse_acap", "all", nc, "units")
        add(
            "pairwise_compat_before_transpose",
            "all",
            compat_before,
            "units",
            f"of {nc}",
        )
        add("pairwise_compat_after_transpose", "all", compat_after, "units", f"of {nc}")
        add("pairwise_transpose_toward_bed", "all", toward, "units", f"of {nc}")
        add("pairwise_subsemitone_nudge_matched_key", "all", nudge_matched, "units")

    # --- H3 wrong-rip: near-integer-semitone offsets -----------------------
    print("\n=== H3 wrong/altered rip: near-whole-semitone offsets (reliable) ===")

    # measured offset within 20¢ of a nonzero integer number of semitones
    def near_semitone(c):
        k = round(c / 100.0)
        return k != 0 and abs(c - 100 * k) <= 20

    near_int = [u for u in units if u["reliable"] and near_semitone(u["cents_agg"])]
    print(
        f"  {len(near_int)} units ≈ whole-semitone offset (of "
        f"{sum(u['reliable'] for u in units)} reliable)"
    )
    add("n_near_integer_semitone", "all", len(near_int), "units")
    for u in all_rows:
        low = u["ref_path"].lower()
        if "savage" in low or "hit me" in low or "benatar" in low:
            print(
                f"  SUSPECT {u['set']:5} coarse={u['coarse']:+d} "
                f"cents_agg={u['cents_agg']:+.0f} residual={u['residual']:+.0f} "
                f"corr={u['peak_corr']:.2f} {Path(u['ref_path']).name[:46]}"
            )

    # --- ear calibration: TOTAL dialed (100·coarse + fine) vs machine total --
    print("\n=== ear-vs-machine: total dialed pitch vs measured (dial_error) ===")
    print(
        "  (dial_error = machine_total − dialed_total; |error|>25¢ ⇒ suspect mislabel)"
    )
    dialed = [u for u in all_rows if abs(u["fine_dialed"]) > 1e-6]
    seen = set()
    errs = []
    for u in sorted(dialed, key=lambda u: -abs(u["fine_dialed"])):
        key = (u["set"], u["slot"], u["fine_dialed"])
        if key in seen:
            continue
        seen.add(key)
        dialed_total = 100 * u["coarse"] + u["fine_dialed"]
        err = u["cents_agg"] - dialed_total
        errs.append(abs(err))
        flag = "  <-- SUSPECT" if abs(err) > 25 and u["peak_corr"] >= 0.5 else ""
        print(
            f"  {u['set']:5} dialed={dialed_total:+6.1f}¢ (c{u['coarse']:+d}+{u['fine_dialed']:+.1f})  "
            f"machine={u['cents_agg']:+6.1f}¢ (corr={u['peak_corr']:.2f})  "
            f"err={err:+6.1f}¢  {Path(u['ref_path']).name[:30]}{flag}"
        )
    add("n_dialed_subsemitone_clips", "all", len(dialed), "clips")
    if errs:
        add("ear_median_abs_dial_error", "all", median(errs), "cents", f"n={len(seen)}")

    missed = [
        u
        for u in units
        if u["reliable"]
        and abs(u["fine_dialed"]) < 1e-6
        and abs(u["residual"]) > 20
        and u["coarse"] == 0
    ]
    print(
        f"\n  {len(missed)} reliable units with >20¢ machine detune the labeler "
        "never dialed (coarse=fine=0):"
    )
    for u in sorted(missed, key=lambda u: -abs(u["residual"]))[:12]:
        print(
            f"    {u['set']:5} residual={u['residual']:+5.0f}¢ corr={u['peak_corr']:.2f} "
            f"{u['stem']:11} {u['section']:8} {Path(u['ref_path']).name[:36]}"
        )
    add("n_undialed_detune_gt20c", "all", len(missed), "units")

    # --- persist -----------------------------------------------------------
    conn = sqlite3.connect(AUX_DB)
    cur = conn.cursor()
    for metric, group, val, unit, notes in findings:
        cur.execute(
            """INSERT INTO analysis_results
                 (analysis_name, metric, group_key, value, unit, notes)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
                 value=excluded.value, unit=excluded.unit, notes=excluded.notes,
                 computed_at=CURRENT_TIMESTAMP""",
            (ANALYSIS_NAME, metric, group, val, unit, notes),
        )
    conn.commit()
    conn.close()
    print(
        f"\npersisted {len(findings)} metrics to aux.analysis_results ({ANALYSIS_NAME})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

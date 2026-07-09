#!/usr/bin/env python3
"""Lyric-anchor ref-diagonal decode for acappella spans.

The acoustic decoder (`path_decode.decode_path`, HuBERT matched-filter) scores
only ~37% strict trajectory-accuracy on BB12 acappella / ~35% on BB11 even with
ORACLE placement (span cropped at GT set_start). It matches TIMBRE/phonetics,
which repeated choruses, key changes and tempo-stretch all confound.

Lyrics give a stronger signal on vocals: a mix word at mix_t whose TEXT matches a
ref word at ref_t puts the point (mix_t -> ref_t) on the true diagonal, exactly
what a human uses to place a vocal by ear. This module turns that into a per-span
SEGMENT LIST `[(mix_start_s, ref_start_s, ref_end_s)]` (span-relative mix start,
absolute ref time — the `path_decode.trajectory_acc` convention), the piecewise-
linear map from mix-time to ref-time (linear span = 1 segment; loops/cuts = many).

Method (per acappella span):
  1. Mix words = mix_vocals words with set_start <= t <= set_end, shifted to
     span-relative mix time. Ref words = candidate ref vocal transcription (abs).
  2. ANCHOR pairs (mix_t_rel, ref_t) by shared normalised word (IDF-weighted by
     ref rarity; stopword/2-char fillers dropped). A ref word that recurs yields
     several candidate anchors — all kept.
  3. Anchor-DP (Viterbi over anchors sorted by mix_t): reward each anchor,
     transition free if the local slope (ref advance / mix advance) is inside a
     tempo band [0.5, 2.0] and continues the current segment; a NEW segment
     (slope break / backward ref jump) costs `lam`. Collapse constant-slope runs
     into segments.

This is the lyric-anchor analogue of `path_decode._viterbi`; it reuses
`lyrics_align` transcription/normalisation and scores with
`path_decode.trajectory_acc` + `path_decode._span_class` so the table is directly
comparable to the acoustic decoder's eval. Read-only imports; nothing here mutates
the modules it borrows from.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.evals.lyrics_ref_decode \
        --gt labeling/fixtures/bb12_ground_truth.yaml [--hubert] [--lam 0.6]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.infer import _manifest_by_tid  # noqa: E402
from workspaces.alignment_prototype.lyrics_align import (  # noqa: E402
    _norm,
    load_cached,
    transcribe_words,
)
from workspaces.alignment_prototype.path_decode import (  # noqa: E402
    _span_class,
    trajectory_acc,
)
from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir  # noqa: E402

# --- decode params (NOT tuned on GT; picked to mirror lyrics_align guards) ----
STOPWORDS = {
    "the",
    "and",
    "you",
    "for",
    "are",
    "was",
    "not",
    "but",
    "all",
    "can",
    "her",
    "him",
    "his",
    "she",
    "they",
    "them",
    "this",
    "that",
    "with",
    "your",
    "have",
    "has",
    "had",
    "get",
    "got",
    "out",
    "now",
    "one",
    "its",
    "our",
}
MIN_WORD_LEN = 3  # drop 2-char fillers (matches lyrics_align _word_window)
SLOPE_LO, SLOPE_HI = 0.5, 2.0  # admissible local tempo band (ref-s per mix-s)
SEG_SLOPE_TOL = 0.12  # |slope - seg_slope| within this = same segment (free)
MIN_ANCHORS = 3  # below this, abstain (too few lyric anchors)
DEFAULT_LAM = 0.6  # cost of opening a new segment (a slope break / ref jump)
IDF_FLOOR = 1  # a word must appear at least once in ref to anchor


@dataclass(frozen=True)
class Anchor:
    mix_t: float  # span-relative mix seconds
    ref_t: float  # absolute ref seconds
    w: float  # IDF weight (1 / ref document frequency of the token)


# --- anchor construction ------------------------------------------------------
def _norm_words_in(words: list[dict], lo: float, hi: float) -> list[tuple[str, float]]:
    """Normalised (token, time) pairs for words whose centre time is in [lo, hi]."""
    out = []
    for x in words:
        t = 0.5 * (x["s"] + x["e"])
        if lo <= t <= hi:
            w = "".join(c for c in x["w"].lower() if c.isalpha() or c == "'")
            if len(w) >= MIN_WORD_LEN and w not in STOPWORDS:
                out.append((w, t))
    return out


def build_anchors(
    mix_words: list[dict],
    ref_words: list[dict],
    set_start: float,
    set_end: float,
) -> list[Anchor]:
    """Anchor cloud for one span: (mix_t_rel, ref_t) per shared distinctive token.

    IDF weight = 1/df over the REF token frequencies, so a distinctive line
    dominates and a filler word that recurs 20x in the ref contributes little
    (and spawns many low-weight anchors the DP can ignore). A recurring ref word
    keeps ALL its (mix, ref) candidate pairs — the DP picks the monotone subset.
    """
    mix_seq = _norm_words_in(mix_words, set_start, set_end)
    # ref: normalise the whole candidate transcription (absolute ref time)
    ref_seq = [
        (w, t)
        for (w, t) in (
            (
                "".join(c for c in x["w"].lower() if c.isalpha() or c == "'"),
                0.5 * (x["s"] + x["e"]),
            )
            for x in ref_words
        )
        if len(w) >= MIN_WORD_LEN and w not in STOPWORDS
    ]
    ref_by_tok: dict[str, list[float]] = {}
    for w, t in ref_seq:
        ref_by_tok.setdefault(w, []).append(t)
    anchors: list[Anchor] = []
    for w, mt in mix_seq:
        rts = ref_by_tok.get(w)
        if not rts or len(rts) < IDF_FLOOR:
            continue
        wt = 1.0 / len(rts)  # IDF: rarer ref tokens weigh more
        for rt in rts:
            anchors.append(Anchor(mix_t=mt - set_start, ref_t=rt, w=wt))
    anchors.sort(key=lambda a: (a.mix_t, a.ref_t))
    return anchors


# --- anchor Viterbi -----------------------------------------------------------
def decode_anchors(
    anchors: list[Anchor],
    lam: float = DEFAULT_LAM,
) -> tuple[list[int], float]:
    """(chain, score) — best monotone-in-mix chain through the anchor cloud.

    DP over anchors sorted by mix_t. State = last-chosen anchor. A transition
    i->j (mix_j >= mix_i) is admissible when the local slope
    s = (ref_j - ref_i)/(mix_j - mix_i) lies in [SLOPE_LO, SLOPE_HI]; it is FREE
    (continues the current segment) when s is within SEG_SLOPE_TOL of the running
    segment slope, otherwise it opens a NEW segment and pays `lam`. Reward for
    landing on anchor j is its IDF weight w_j (distinctive lyric lines dominate).
    A backward ref jump (ref_j < ref_i, a loop/replay) is always a new segment.

    This is the lyric analogue of `path_decode._viterbi`: staying on a diagonal
    is free, a jump costs lam. We maximise total anchor reward minus jump costs.
    """
    n = len(anchors)
    if n == 0:
        return [], 0.0
    NEG = -1e18
    # dp[j] over (prev index p, running segment slope bucket) is expensive; instead
    # keep dp keyed on (j, slope_of_incoming_edge) — but slope is continuous. Use
    # the standard trick: dp[j][p] = best score of a chain ending with edge p->j.
    # n is small (tens of anchors per span), so O(n^2) states, O(n) transitions
    # via a start-state is fine. We collapse to dp over j with backpointer + the
    # slope of the edge that produced dp[j], approximating the segment slope as
    # the last edge's slope (segments are near-constant-slope by construction).
    dp = [NEG] * n
    edge_slope = [0.0] * n  # slope of the edge that reached j (nan => chain start)
    back = [-1] * n
    for j in range(n):
        dp[j] = anchors[j].w  # start a chain at j (implicit first segment)
        edge_slope[j] = float("nan")
    order = list(range(n))  # already sorted by mix_t
    for jj in range(n):
        j = order[jj]
        aj = anchors[j]
        for ii in range(jj):
            i = order[ii]
            if dp[i] <= NEG:
                continue
            ai = anchors[i]
            dmix = aj.mix_t - ai.mix_t
            if dmix <= 1e-6:
                continue  # need strictly increasing mix time
            dref = aj.ref_t - ai.ref_t
            slope = dref / dmix
            if slope < SLOPE_LO or slope > SLOPE_HI:
                continue  # implausible local tempo -> not a valid same-diagonal step
            prev_s = edge_slope[i]
            same_seg = (not np.isnan(prev_s)) and abs(slope - prev_s) <= SEG_SLOPE_TOL
            cost = 0.0 if same_seg else lam
            v = dp[i] + aj.w - cost
            if v > dp[j]:
                dp[j] = v
                edge_slope[j] = slope
                back[j] = i
    end = int(np.argmax(dp))
    best_score = float(dp[end])
    chain: list[int] = []
    cur = end
    while cur != -1:
        chain.append(cur)
        cur = back[cur]
    chain.reverse()
    return chain, best_score


def chain_to_segments(
    anchors: list[Anchor],
    chain: list[int],
    span_dur: float,
) -> list[tuple[float, float, float]]:
    """Collapse a monotone anchor chain into `[(mix_start, ref_start, ref_end)]`.

    Fit each near-constant-slope RUN of consecutive chain edges into one segment.
    A run breaks when the edge slope jumps (same criterion as the DP's segment
    open) or the ref jumps backward. mix_start is span-relative; ref_start/end are
    absolute ref seconds, extrapolated to the run's mix extent. The final segment
    extends to the span end so `trajectory_acc` can sample the whole span.
    """
    if not chain:
        return []
    if len(chain) == 1:
        # single anchor: assume a straight diagonal at slope 1.0 through it,
        # covering the whole span (best we can do with one point).
        a = anchors[chain[0]]
        r0 = a.ref_t - a.mix_t  # ref at mix_t = 0, slope 1
        return [(0.0, max(0.0, r0), max(0.0, r0) + span_dur)]

    # per-edge slopes
    pts = [(anchors[c].mix_t, anchors[c].ref_t) for c in chain]
    slopes = []
    for k in range(1, len(pts)):
        dm = pts[k][0] - pts[k - 1][0]
        slopes.append((pts[k][1] - pts[k - 1][1]) / dm if dm > 1e-6 else 1.0)

    # group edges into runs of similar slope
    runs: list[list[int]] = [[0]]  # each run = list of point indices
    seg_slope = slopes[0]
    for k in range(1, len(slopes)):
        if abs(slopes[k] - seg_slope) <= SEG_SLOPE_TOL:
            runs[-1].append(k)  # extend current run (edge k connects pts[k]->pts[k+1])
        else:
            runs.append([k])
            seg_slope = slopes[k]

    segs: list[tuple[float, float, float]] = []
    for ri, run in enumerate(runs):
        i0 = run[0]  # first edge's start point
        i1 = run[-1] + 1  # last edge's end point
        m0, r0 = pts[i0]
        m1, r1 = pts[i1]
        # robust slope = total ref advance / total mix advance across the run
        dm = m1 - m0
        s = (r1 - r0) / dm if dm > 1e-6 else 1.0
        s = float(np.clip(s, SLOPE_LO, SLOPE_HI))
        # mix extent of this segment: from this run's first mix time to the next
        # run's first mix time (or span end for the last run)
        mix_start = m0 if ri > 0 else 0.0
        if ri + 1 < len(runs):
            next_i0 = runs[ri + 1][0]
            mix_end = pts[next_i0][0]
        else:
            mix_end = span_dur
        # ref at mix_start / mix_end via the run's diagonal anchored at (m0, r0)
        ref_start = r0 + (mix_start - m0) * s
        ref_end = r0 + (mix_end - m0) * s
        segs.append((mix_start, max(0.0, ref_start), max(0.0, ref_end)))
    return segs


@dataclass(frozen=True)
class LyricDecode:
    """A lyric decode plus the OBSERVABLE features a per-span selector may use.

    Everything here is computable at inference (no GT label). ``decoded`` False
    means the span abstained (too few anchors / degenerate chain) and the caller
    should fall back to the acoustic decoder.
    """

    segs: list[tuple[float, float, float]]
    decoded: bool
    n_anchors: int  # anchor cloud size (lyric density)
    inlier_anchors: int  # anchors the decoded chain actually used
    score: float  # DP path score (total IDF reward - jump costs)
    dom_slope: float  # duration-weighted dominant tempo ratio of the path
    n_segments: int  # decoded segment count
    backward_frac: (
        float  # fraction of chain edges that step BACKWARD in ref (loop-ness)
    )

    @property
    def anchor_density(self) -> float:
        """Anchors per span second — a scale-free lyric-density signal."""
        if not self.segs:
            return 0.0
        return self.n_anchors / max(1e-3, self._span_dur)

    _span_dur: float = 1.0


def decode_span_full(
    mix_words: list[dict],
    ref_words: list[dict],
    set_start: float,
    set_end: float,
    lam: float = DEFAULT_LAM,
) -> LyricDecode:
    """Rich lyric decode carrying the observable selector features (see LyricDecode)."""
    span_dur = max(1e-3, set_end - set_start)
    anchors = build_anchors(mix_words, ref_words, set_start, set_end)
    n_anch = len(anchors)
    if n_anch < MIN_ANCHORS:
        return LyricDecode([], False, n_anch, 0, 0.0, 1.0, 0, 0.0, _span_dur=span_dur)
    chain, score = decode_anchors(anchors, lam=lam)
    if len(chain) < 2:
        return LyricDecode(
            [], False, n_anch, len(chain), score, 1.0, 0, 0.0, _span_dur=span_dur
        )
    segs = chain_to_segments(anchors, chain, span_dur)

    # observable path geometry from the decoded anchor chain
    pts = [(anchors[c].mix_t, anchors[c].ref_t) for c in chain]
    back = 0
    tot_mix = ref_span = 0.0
    for k in range(1, len(pts)):
        dm = pts[k][0] - pts[k - 1][0]
        dr = pts[k][1] - pts[k - 1][1]
        if dr < -1e-6:
            back += 1
        if dm > 1e-6:
            tot_mix += dm
    # duration-weighted dominant slope = total ref advance over the LONGEST-slope
    # run; use the endpoints of the decoded segments (robust to jitter).
    if segs:
        # slope of the widest (by mix extent) segment = the span's characteristic tempo
        widest = max(
            range(len(segs)),
            key=lambda i: (
                (segs[i + 1][0] if i + 1 < len(segs) else span_dur) - segs[i][0]
            ),
        )
        ms = segs[widest][0]
        me = segs[widest + 1][0] if widest + 1 < len(segs) else span_dur
        rs, re = segs[widest][1], segs[widest][2]
        dom = (re - rs) / max(1e-3, me - ms)
    else:
        dom = 1.0
    dom = float(np.clip(dom, SLOPE_LO, SLOPE_HI))
    back_frac = back / max(1, len(pts) - 1)
    return LyricDecode(
        segs=segs,
        decoded=True,
        n_anchors=n_anch,
        inlier_anchors=len(chain),
        score=score,
        dom_slope=dom,
        n_segments=len(segs),
        backward_frac=back_frac,
        _span_dur=span_dur,
    )


def decode_span(
    mix_words: list[dict],
    ref_words: list[dict],
    set_start: float,
    set_end: float,
    lam: float = DEFAULT_LAM,
) -> tuple[list[tuple[float, float, float]], int]:
    """(segments, n_anchors). Thin back-compat wrapper over decode_span_full."""
    d = decode_span_full(mix_words, ref_words, set_start, set_end, lam=lam)
    return d.segs, d.n_anchors


# --- ref-vocal candidate resolution (mirrors lyrics_align._build_spans) -------
def _cand_ref_path(track: dict | None) -> str | None:
    if not track:
        return None
    vp = (track.get("stems") or {}).get("vocals")
    if vp and Path(vp).is_file():
        return vp
    src = track.get("local_path")
    return src if (src and Path(src).is_file()) else None


# --- HuBERT baseline (same rows, oracle crop) ---------------------------------
def _hubert_baseline_seg(
    ref_audio: str,
    mix_vocals: str,
    set_start: float,
    set_end: float,
    hubert_layer: int,
    lam: float,
) -> tuple[list[tuple[float, float, float]], float]:
    """(segments, score) — `path_decode.decode_path` on ONE span (HuBERT, oracle crop).

    Mirrors path_decode.main's per-span decode: HuBERT features on mix_vocals and
    the ref vocal, span cropped at GT set_start/set_end (oracle placement), a wide
    octave stretch band, and the default lam. `score` is the Viterbi path total
    (matched-filter confidence), used by the fusion selector. Read-only reuse."""
    from workspaces.alignment_prototype.path_decode import (
        FPS,
        _ensure_feat,
        decode_path,
    )

    mnpy = _ensure_feat(mix_vocals, f"mixvoc:{mix_vocals}", "hubert", hubert_layer)
    rnpy = _ensure_feat(ref_audio, ref_audio, "hubert", hubert_layer)
    M = np.ascontiguousarray(np.load(mnpy, mmap_mode="r"), dtype=np.float32)
    R = np.ascontiguousarray(np.load(rnpy, mmap_mode="r"), dtype=np.float32)
    a = int(set_start * FPS)
    n = int(max(0.0, set_end - set_start) * FPS)
    if n < 4:
        return [], -1.0
    Mc = np.ascontiguousarray(M[:, a : a + n])
    # octave-multiplied fine band, same spirit as path_decode._stretch_band but
    # without the beat-grid centre (we have no per-span grid here → center 1.0).
    band = sorted(
        {
            round(oct_mult * f, 4)
            for oct_mult in (0.5, 1.0, 2.0)
            for f in (0.96, 0.98, 1.0, 1.02, 1.04)
        }
    )
    wlen = int(12.0 * FPS)
    hop = int(2.0 * FPS)
    segs, score = decode_path(Mc, R, tuple(band), lam, wlen, hop, lam)
    # normalise by window count so the score is comparable across span lengths
    # (decode_path sums per-window emissions; longer spans have more windows).
    n_win = max(1, len(range(0, max(1, Mc.shape[1] - wlen + 1), hop)) or 1)
    return segs, (score / n_win if score > 0 else score)


# --- fiber cache --------------------------------------------------------------
def _fibers_for(ref_audio: str, hubert_layer: int):
    from workspaces.alignment_prototype.path_decode import FPS, _ensure_feat
    from workspaces.alignment_prototype.ref_fibers import compute_fibers

    hf = np.load(_ensure_feat(ref_audio, ref_audio, "hubert", hubert_layer))
    return compute_fibers(hf, FPS, audio_path=ref_audio)


# --- fusion selector ----------------------------------------------------------
# Thresholds, NOT learned (n is too small for a fitted model). Each is motivated
# by the head-to-head class result, using ONLY inference-observable features:
#   * lyric wins big on ODDRATIO (extreme tempo) + LINEAR (dense lyric) — route
#     to lyric when it fired densely OR detected an extreme slope.
#   * lyric LOSES on loops (repeat-instance ambiguity) — a high backward-jump
#     fraction in the anchor chain is the observable loop tell → keep HuBERT.
SEL_MIN_ANCHORS = 8  # lyric must have fired with at least this many anchors
SEL_EXTREME_LOG2 = 0.30  # |log2(slope)| above this = oddratio-ish (≈>1.23x or <0.81x)
SEL_MIN_DENSITY = 0.8  # anchors per span-second: dense enough to trust the lyric path
SEL_MAX_BACKWARD = 0.34  # above this backward-edge fraction the span looks like a loop


def select_route(ld: "LyricDecode") -> tuple[str, str]:
    """(route, reason) — 'lyric' or 'hubert', using ONLY observable features.

    Rule (interpretable, two effective thresholds):
      abstain / too-few-anchors / loop-like  -> HuBERT (its always-on, loop-strong)
      fired densely AND (extreme slope OR dense anchors) -> LYRIC
      else -> HuBERT.
    """
    if not ld.decoded or ld.n_anchors < SEL_MIN_ANCHORS:
        return "hubert", "lyric-abstain/sparse"
    if ld.backward_frac > SEL_MAX_BACKWARD:
        return "hubert", "loop-like(backward)"
    log2s = abs(np.log2(max(1e-3, ld.dom_slope)))
    extreme = log2s > SEL_EXTREME_LOG2
    dense = ld.anchor_density >= SEL_MIN_DENSITY
    if extreme:
        return "lyric", "extreme-slope"
    if dense:
        return "lyric", "dense-anchors"
    return "hubert", "lyric-weak-default"


# --- eval ---------------------------------------------------------------------
def _raw_rows(gt_path: Path) -> dict:
    import yaml

    return {
        (str(r.get("slot_label")), round(float(r.get("set_start_s", -1)), 2)): r
        for r in yaml.safe_load(gt_path.read_text()).get("tracks", [])
    }


def _report(rows: list[dict], label: str, with_fiber: bool) -> None:
    """rows = [{cls, strict, fiber, n_pred, decoded}]. Mirror path_decode's rep."""

    def rep(name: str, sel: list[dict]) -> None:
        dec = [r for r in sel if r["decoded"]]
        if not sel:
            return
        # traj-acc averaged over DECODED spans (abstentions excluded from the
        # mean, but coverage is reported alongside).
        n = len(sel)
        nd = len(dec)
        strict = np.array([r["strict"] for r in dec]) if dec else np.array([0.0])
        line = (
            f"  {name:18} n={n:3} decoded={nd:3} ({100 * nd / max(1, n):3.0f}%)  "
            f"strict {100 * strict.mean():3.0f}%  >=80% {100 * (strict >= 0.8).mean():3.0f}%"
        )
        if with_fiber and dec:
            fib = np.array([r["fiber"] for r in dec])
            line += f"  || fiber {100 * fib.mean():3.0f}%"
        print(line)

    print(f"\n=== {label} — trajectory accuracy (acappella, oracle crop) ===")
    rep("ALL", rows)
    for cls in ("linear", "multiseg", "loop", "oddratio"):
        rep(cls, [r for r in rows if r["cls"] == cls])


def main(argv: list[str] | None = None) -> int:
    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--lam", type=float, default=DEFAULT_LAM)
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument(
        "--hubert",
        action="store_true",
        help="also run the HuBERT matched-filter baseline on the SAME rows "
        "(needs features; slower — computes HuBERT on each ref + mix_vocals)",
    )
    p.add_argument(
        "--transcribe",
        action="store_true",
        help="transcribe missing candidates (needs GPU/MPS); default cache-only",
    )
    p.add_argument("--fibers", action="store_true", default=True)
    p.add_argument("--no-fibers", dest="fibers", action="store_false")
    p.add_argument(
        "--dump-features",
        type=Path,
        default=None,
        help="write per-span observable features + both decoders' scores as JSON "
        "(for offline selector calibration / analysis)",
    )
    args = p.parse_args(argv)

    gt = yaml.safe_load(args.gt.read_text())
    set_id = gt["set_id"]
    set_dir = find_aligning_dir(set_id)
    by_tid = _manifest_by_tid(set_dir, set_id)
    raw = _raw_rows(args.gt)

    # mix vocal words (cached) — the single anchor stream for every span
    mix_vocals = set_dir / "mix_vocals.flac"
    mix_words = load_cached(mix_vocals)
    if mix_words is None:
        if args.transcribe:
            mix_words = transcribe_words(mix_vocals, enhance=False)
        else:
            sys.exit(f"mix_vocals not transcribed for {set_id}; run --transcribe")

    aca = [t for t in gt["tracks"] if t.get("claimed_stem") == "acappella"]
    print(
        f"{set_id}: {len(aca)} acappella spans | mix_vocals words={len(mix_words)}",
        file=sys.stderr,
    )

    # one record per span carrying BOTH decoders + the observable selector
    # features, so the report can show HuBERT-ALL / Lyric-route-ALL / FUSED-ALL.
    recs: list[dict] = []
    abstain_reasons: dict[str, int] = {}
    fiber_cache: dict[str, tuple] = {}

    def _blank() -> dict:
        return {"strict": 0.0, "fiber": 0.0, "n_pred": 0, "decoded": False}

    for t in aca:
        row = raw.get(
            (str(t.get("slot_label")), round(float(t.get("set_start_s", -1)), 2))
        )
        if row is None:
            continue
        ss, se = float(t["set_start_s"]), float(t["set_end_s"])
        if se - ss < 4:
            continue
        cls = _span_class(row)
        rec = {
            "cls": cls,
            "lyric": _blank(),
            "hubert": _blank(),
            "ld": None,  # LyricDecode (observable features)
            "route": "hubert",
            "reason": "no-lyric",
        }
        recs.append(rec)

        track = by_tid.get(t.get("track_id"))
        ref_path = _cand_ref_path(track)
        if ref_path is None:
            abstain_reasons["no_ref_audio"] = abstain_reasons.get("no_ref_audio", 0) + 1
        ref_words = None
        if ref_path is not None:
            ref_words = (
                load_cached(ref_path)
                if not args.transcribe
                else transcribe_words(ref_path)
            )
            if not ref_words:
                abstain_reasons["no_ref_transcript"] = (
                    abstain_reasons.get("no_ref_transcript", 0) + 1
                )

        fib = None
        if args.fibers and ref_path is not None:
            if ref_path not in fiber_cache:
                fiber_cache[ref_path] = _fibers_for(ref_path, args.hubert_layer)
            fib = fiber_cache[ref_path]

        # --- lyric decode (rich, carries observable features) ---
        if ref_words:
            ld = decode_span_full(mix_words, ref_words, ss, se, lam=args.lam)
            rec["ld"] = ld
            if not ld.decoded:
                abstain_reasons["too_few_anchors"] = (
                    abstain_reasons.get("too_few_anchors", 0) + 1
                )
            else:
                strict, n_pred, facc = trajectory_acc(ld.segs, row, fiber=fib)
                rec["lyric"] = {
                    "strict": strict,
                    "fiber": facc,
                    "n_pred": n_pred,
                    "decoded": True,
                }

        # --- HuBERT baseline on the same row ---
        if args.hubert and ref_path is not None:
            hsegs, hscore = _hubert_baseline_seg(
                ref_path, str(mix_vocals), ss, se, args.hubert_layer, 0.15
            )
            rec["hubert_score"] = hscore
            if hsegs:
                hstrict, hn, hfacc = trajectory_acc(hsegs, row, fiber=fib)
                rec["hubert"] = {
                    "strict": hstrict,
                    "fiber": hfacc,
                    "n_pred": hn,
                    "decoded": True,
                }

        # --- selector (observable-only) ---
        route, reason = (
            select_route(rec["ld"]) if rec["ld"] is not None else ("hubert", "no-lyric")
        )
        rec["route"], rec["reason"] = route, reason

    def _rows(key: str, subset: list[dict]) -> list[dict]:
        return [{"cls": r["cls"], **r[key]} for r in subset]

    def _fused_rows(subset: list[dict], use_oracle: bool = False) -> list[dict]:
        """Route each span per the selector (or an ORACLE picker) → its chosen
        decoder's result. Falls back to HuBERT whenever lyric didn't decode."""
        out = []
        for r in subset:
            ly, hu = r["lyric"], r["hubert"]
            if use_oracle:
                # oracle upper bound: pick whichever decoder scored better (strict)
                pick = ly if (ly["decoded"] and ly["strict"] >= hu["strict"]) else hu
            else:
                pick = ly if (r["route"] == "lyric" and ly["decoded"]) else hu
            out.append({"cls": r["cls"], **pick})
        return out

    if not args.hubert:
        _report(_rows("lyric", recs), f"LYRIC decode (lam={args.lam})", args.fibers)
        print("\ncoverage / abstain reasons:", dict(abstain_reasons), file=sys.stderr)
        return 0

    _report(
        _rows("hubert", recs),
        "HuBERT-ALL  (baseline, matched-filter, lam=0.15)",
        args.fibers,
    )
    # Lyric-route-ALL: lyric where it fired, HuBERT elsewhere (its coverage policy)
    lyric_route = [
        {"cls": r["cls"], **(r["lyric"] if r["lyric"]["decoded"] else r["hubert"])}
        for r in recs
    ]
    _report(
        lyric_route, "LYRIC-route-ALL  (lyric where it fires, HuBERT else)", args.fibers
    )
    _report(_fused_rows(recs), "FUSED  (per-span selector)", args.fibers)
    _report(
        _fused_rows(recs, use_oracle=True),
        "ORACLE-select  (upper bound: pick better per span)",
        args.fibers,
    )

    _route_diagnostics(recs, args.fibers)
    if args.dump_features:
        feats = []
        for r in recs:
            ld = r["ld"]
            feats.append(
                {
                    "cls": r["cls"],  # GT label, dump-only (analysis, NOT a feature)
                    "lyric_decoded": r["lyric"]["decoded"],
                    "lyric_strict": r["lyric"]["strict"],
                    "hubert_strict": r["hubert"]["strict"],
                    "hubert_score": r.get("hubert_score", -1.0),
                    "n_anchors": (ld.n_anchors if ld else 0),
                    "anchor_density": (ld.anchor_density if ld else 0.0),
                    "dom_slope": (ld.dom_slope if ld else 1.0),
                    "log2_slope": (
                        float(abs(np.log2(max(1e-3, ld.dom_slope)))) if ld else 0.0
                    ),
                    "n_segments": (ld.n_segments if ld else 0),
                    "backward_frac": (ld.backward_frac if ld else 0.0),
                    "lyric_score": (ld.score if ld else 0.0),
                    "route": r["route"],
                }
            )
        args.dump_features.parent.mkdir(parents=True, exist_ok=True)
        args.dump_features.write_text(json.dumps(feats, indent=1))
        print(
            f"dumped {len(feats)} span features -> {args.dump_features}",
            file=sys.stderr,
        )
    print("\ncoverage / abstain reasons:", dict(abstain_reasons), file=sys.stderr)
    return 0


def _route_diagnostics(recs: list[dict], with_fiber: bool) -> None:
    """Which spans did the selector send where, and was it right?

    For every span where BOTH decoders ran, count how often lyric actually beats
    HuBERT, and how well each observable signal (extreme slope, density, backward-
    frac) predicts that — this tells a future learned selector what matters."""
    both = [
        r for r in recs if r["lyric"]["decoded"] and r["hubert"]["decoded"] and r["ld"]
    ]
    if not both:
        return
    print("\n=== selector diagnostics (spans where both fired, n=%d) ===" % len(both))
    lyric_better = [
        r for r in both if r["lyric"]["strict"] > r["hubert"]["strict"] + 1e-6
    ]
    print(
        f"  lyric strictly beats HuBERT on {len(lyric_better)}/{len(both)} "
        f"({100 * len(lyric_better) / len(both):.0f}%) of both-fired spans"
    )

    def _signal(name: str, pred) -> None:
        pos = [r for r in both if pred(r)]
        neg = [r for r in both if not pred(r)]
        if not pos or not neg:
            print(f"  {name:22} n_pos={len(pos):2} n_neg={len(neg):2}  (degenerate)")
            return
        # precision = of spans flagged, how many is lyric actually better on
        prec = np.mean(
            [r["lyric"]["strict"] > r["hubert"]["strict"] + 1e-6 for r in pos]
        )
        base = np.mean(
            [r["lyric"]["strict"] > r["hubert"]["strict"] + 1e-6 for r in both]
        )
        print(
            f"  {name:22} flags {len(pos):2}: lyric-better {100 * prec:3.0f}% "
            f"(base {100 * base:3.0f}%, lift {100 * (prec - base):+3.0f}pp)"
        )

    _signal(
        "extreme-slope",
        lambda r: abs(np.log2(max(1e-3, r["ld"].dom_slope))) > SEL_EXTREME_LOG2,
    )
    _signal("dense-anchors", lambda r: r["ld"].anchor_density >= SEL_MIN_DENSITY)
    _signal("many-anchors(>=8)", lambda r: r["ld"].n_anchors >= SEL_MIN_ANCHORS)
    _signal("low-backward(<=.34)", lambda r: r["ld"].backward_frac <= SEL_MAX_BACKWARD)
    _signal("lyric_score>0", lambda r: r["ld"].score > 0)
    _signal(
        "lyric_score>hub_score",
        lambda r: r["ld"].score > r.get("hubert_score", -1.0),
    )


if __name__ == "__main__":
    sys.exit(main())

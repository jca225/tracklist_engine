"""Instrumental stem-to-stem fingerprint probe (identity + set_start placement).

The two weak instrumental cells are identity (which track) and placement (where
it comes in): instrumental has no vocals and sits under mix layering, so MERT
(regular audio, identity) and chroma (placement) both fail. The regular-stem
landmark fingerprint is our sharpest identity+placement signal (0.2s / 76% on
regular). This probe tests whether fingerprinting the INSTRUMENTAL stem on both
sides does the same for instrumental spans:

  mix_instrumental.flac  <-fp->  ref Demucs stems.instrumental

For each instrumental GT span in a set:
  * identity = does the CORRECT ref win the landmark-vote count (restricted to
    the span's mix-time window on each candidate's best diagonal) among the
    set's candidate instrumental refs?
  * placement = set_start from the correct ref's dominant vote-diagonal cluster
    (mix_fp_hits.offset_candidates), the same primitive as regular placement.

Baseline (side by side):
  * identity via MERT (mert_store.load_bb12_mert; regular-audio, layer 6) — pool
    the mix window MERT, cosine-match to each candidate ref track-mean.
  * placement via chroma matched-filter (refine_ref_offsets) — the known-weak
    instrumental placement baseline (offset, converted to set_start via GT
    ref_start for a fair placement comparison is NOT possible; we report the
    chroma set_start by scanning the mix band — see _chroma_set_start).

Ref instrumental fingerprints do not exist in the cache (only __regular /
__acappella). This probe BUILDS them from manifest stems.instrumental into a
side cache (alignment/.cache/fp_instr/), mirroring
scripts/backfill_track_fingerprints.py.

Usage:
    venvs/audio/bin/python -m alignment.instrumental_probe \
        --set bb11 --set bb12
    # or a single set:
    venvs/audio/bin/python -m alignment.instrumental_probe --set bb12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alignment.fp_index import FpKey, load_cached, save_cached
from alignment.landmark_fp import (
    FHOP,
    SR,
    LandmarkFingerprint,
    constellation,
    fingerprint_from_audio,
    hashes as fp_hashes,
)
from alignment.mix_fp_hits import _vote_pairs, offset_candidates

INSTR_CACHE = Path(__file__).resolve().parent / ".cache" / "fp_instr"

SETS = {
    "bb11": {
        "set_id": "2nvzlh2k",
        "dir": "2nvzlh2k__Two Friends - Big Bootie Mix Episode 11",
        "yaml": "labeling/fixtures/bb11_ground_truth.yaml",
    },
    "bb12": {
        "set_id": "1fsnxchk",
        "dir": "1fsnxchk__Two Friends - Big Bootie Mix Volume 12",
        "yaml": "labeling/fixtures/bb12_ground_truth.yaml",
    },
}


# ------------------------- data loading -----------------------------------


@dataclass(frozen=True)
class Span:
    slot_label: str
    track_id: str
    display: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    instr_path: str | None  # ref Demucs instrumental stem


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def _manifest_by_tid(set_dir: Path, set_id: str) -> dict[str, dict]:
    rows = json.loads((set_dir / "manifest.json").read_text())["tracks"]
    by_tid = {row["track_id"]: row for row in rows}
    map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    if map_path.exists():
        for tlp, rec in json.loads(map_path.read_text()).items():
            if tlp in by_tid:
                by_tid.setdefault(rec, by_tid[tlp])
    return by_tid


import re


def _resolve_stem_by_slot(set_dir: Path, slot_label: str) -> str | None:
    """Fallback: glob ``stems/<slot>__*/instrumental.flac``.

    The manifest ``stems.instrumental`` field is stale/incomplete for ~8 of the
    51 GT instrumental spans, but the files exist on disk keyed by SLOT (not
    track_id). A slot may have multiple dirs (one plain, one annotator-tagged
    ``... [126bpm 9B]``); prefer the ``[NNNbpm KK]``-tagged dir — that's the
    annotator's canonical instrumental. Verified with the coordinator: with this
    fallback all 26/26 (BB11) and 24/24 real (BB12) spans resolve.
    """
    if not slot_label or slot_label == "?":
        return None
    stems_dir = set_dir / "stems"
    if not stems_dir.is_dir():
        return None
    matches = sorted(stems_dir.glob(f"{slot_label}__*"))
    matches = [m for m in matches if (m / "instrumental.flac").is_file()]
    if not matches:
        return None
    tagged = [m for m in matches if re.search(r"\[\d+bpm", m.name)]
    chosen = tagged[0] if tagged else matches[0]
    return str(chosen / "instrumental.flac")


def _instr_stem(row: dict | None) -> str | None:
    if not row:
        return None
    ip = (row.get("stems") or {}).get("instrumental")
    if ip and os.path.isfile(ip):
        return ip
    return None


def _resolve_span_stem(
    set_dir: Path, row: dict | None, slot_label: str
) -> tuple[str | None, list[str]]:
    """(path, tried) — manifest field first, slot-glob fallback second."""
    tried: list[str] = []
    ip = _instr_stem(row)
    if ip:
        return ip, tried
    if row:
        mf = (row.get("stems") or {}).get("instrumental")
        if mf:
            tried.append(f"manifest:{mf}")
    slot = _resolve_stem_by_slot(set_dir, slot_label)
    if slot:
        return slot, tried
    tried.append(f"glob:{set_dir}/stems/{slot_label}__*/instrumental.flac")
    return None, tried


def load_spans(key: str) -> tuple[str, Path, list[Span]]:
    cfg = SETS[key]
    set_id = cfg["set_id"]
    set_dir = Path(os.path.expanduser(f"~/aligning/{cfg['dir']}"))
    gt = _load_yaml(_REPO / cfg["yaml"])
    by_tid = _manifest_by_tid(set_dir, gt["set_id"])
    spans: list[Span] = []
    for r in gt["tracks"]:
        if r.get("claimed_stem") != "instrumental":
            continue
        tid = r.get("track_id")
        # the "mix_instrumental" GT row (track_id None, ref_start==set_start) is
        # the mix bus itself, not a per-track span -> no identity possible, skip.
        if not tid and (r.get("track") or "").strip() == "mix_instrumental":
            print(f"  [skip mix-bus row: {r.get('track')!r}]")
            continue
        row = by_tid.get(tid) if tid else None
        slot = r.get("slot_label") or ""
        instr, tried = _resolve_span_stem(set_dir, row, slot)
        if instr is None:
            print(
                f"  [UNRESOLVED stem] slot={slot!r} tid={tid!r} "
                f"track={r.get('track')!r} tried={tried}",
                file=sys.stderr,
            )
        spans.append(
            Span(
                slot_label=slot or "?",
                track_id=tid or "",
                display=r.get("track", ""),
                set_start_s=float(r["set_start_s"]),
                set_end_s=float(r["set_end_s"]),
                ref_start_s=float(r["ref_start_s"]),
                instr_path=instr,
            )
        )
    return set_id, set_dir, spans


def candidate_pool(set_dir: Path, set_id: str, spans: list[Span]) -> dict[str, str]:
    """{track_id: instrumental_stem_path} for the identity candidate pool.

    Union of (a) every manifest ref whose Demucs instrumental stem is on disk
    and (b) every GT-span ref resolved via the slot-glob fallback — so a span
    whose correct ref only resolves by slot still has its correct candidate in
    the pool (else identity is unwinnable for it).
    """
    rows = json.loads((set_dir / "manifest.json").read_text())["tracks"]
    pool: dict[str, str] = {}
    for row in rows:
        ip = _instr_stem(row)
        if ip:
            pool[row["track_id"]] = ip
    for sp in spans:
        if sp.track_id and sp.instr_path and sp.track_id not in pool:
            pool[sp.track_id] = sp.instr_path
    return pool


# ------------------------- fingerprint building ---------------------------


def _load_audio(path: str) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def build_ref_fp(track_id: str, instr_path: str) -> LandmarkFingerprint | None:
    """Landmark fp for a ref instrumental stem, cached under __instrumental."""
    key = FpKey(recording_id=track_id, stem="instrumental")
    hit = load_cached(key, INSTR_CACHE)
    if hit is not None:
        return hit
    try:
        y = _load_audio(instr_path)
    except Exception as exc:  # noqa: BLE001
        print(f"    [fp build fail] {track_id}: {exc}", file=sys.stderr)
        return None
    if y.size < SR:  # <1s -> unusable stem
        return None
    fp = fingerprint_from_audio(y)
    save_cached(fp, key, INSTR_CACHE)
    return fp


def build_pool_fps(pool: dict[str, str]) -> dict[str, LandmarkFingerprint]:
    fps: dict[str, LandmarkFingerprint] = {}
    n_built = 0
    for tid, path in pool.items():
        if load_cached(FpKey(tid, "instrumental"), INSTR_CACHE) is None:
            n_built += 1
        fp = build_ref_fp(tid, path)
        if fp is not None:
            fps[tid] = fp
    print(f"  ref instrumental fps: {len(fps)} usable ({n_built} newly built)")
    return fps


def mix_instr_hashes(set_dir: Path) -> dict:
    """Landmark hashes for mix_instrumental.flac (computed once per set)."""
    mix_path = set_dir / "mix_instrumental.flac"
    y = _load_audio(str(mix_path))
    tf, fb = constellation(y)
    return fp_hashes(tf, fb)


# ------------------------- fp identity + placement ------------------------


def _window_votes(
    mix_hashes: dict,
    ref_fp: LandmarkFingerprint,
    *,
    win_lo_s: float,
    win_hi_s: float,
    tol: int = 1,
) -> tuple[int, float]:
    """Votes on the ref's best diagonal whose MIX time falls in [lo,hi].

    Identity score for one candidate ref restricted to the span's mix window:
    the correct ref should place a dense on-diagonal vote cluster exactly where
    it plays. Returns (votes_in_window, offset_s_of_best_diagonal).
    """
    votes, pairs = _vote_pairs(mix_hashes, ref_fp)
    if not votes:
        return 0, 0.0
    lo_f = win_lo_s * SR / FHOP
    hi_f = win_hi_s * SR / FHOP
    # score each dominant diagonal by votes landing inside the mix window
    best_v = 0
    best_off = 0
    # consider the top diagonals (by global count) to keep it cheap
    for off, _cnt in sorted(votes.items(), key=lambda kv: -kv[1])[:12]:
        v = sum(1 for o, mt in pairs if abs(o - off) <= tol and lo_f <= mt <= hi_f)
        if v > best_v:
            best_v, best_off = v, off
    return best_v, float(best_off * FHOP / SR)


def fp_identity_and_placement(
    span: Span,
    mix_hashes: dict,
    pool_fps: dict[str, LandmarkFingerprint],
    *,
    win_pad_s: float = 4.0,
) -> tuple[bool | None, int, float | None]:
    """(correct_is_argmax, n_candidates_scored, fp_set_start_s).

    Identity: among all candidate instrumental refs, the correct track_id must
    have the most in-window votes. None if the correct ref has no fp.
    Placement: set_start from the correct ref's dominant vote-diagonal cluster.
    """
    lo = max(0.0, span.set_start_s - win_pad_s)
    hi = span.set_end_s + win_pad_s
    scores: dict[str, int] = {}
    for tid, fp in pool_fps.items():
        v, _off = _window_votes(mix_hashes, fp, win_lo_s=lo, win_hi_s=hi)
        scores[tid] = v
    if not scores:
        return None, 0, None
    winner = max(scores.items(), key=lambda kv: kv[1])[0]
    correct_fp = pool_fps.get(span.track_id)
    id_correct: bool | None
    if correct_fp is None:
        id_correct = None  # correct ref has no fp -> cannot be identified
    else:
        id_correct = winner == span.track_id and scores[span.track_id] > 0

    # placement (RAW, no GT): correct ref's single strongest vote-diagonal
    # cluster over the whole mix. This is the unleashed fp — sharp when the ref
    # is distinctive, but a wrong-diagonal repeat can land it hundreds of s off.
    # The pipeline-faithful, leak-free placement is the set-level monotonic
    # decode (decode_instr_placements below); this raw number shows the ceiling
    # and the fat tail the decode must tame.
    set_start = None
    if correct_fp is not None:
        cands = offset_candidates(mix_hashes, correct_fp, topk=6)
        if cands:
            pick = max(cands, key=lambda c: c[2])
            set_start = float(pick[0])
    return id_correct, len(scores), set_start


def decode_instr_placements(
    spans: list[Span],
    mix_hashes: dict,
    pool_fps: dict[str, LandmarkFingerprint],
    *,
    mix_dur_s: float,
) -> list[float | None]:
    """Pipeline-faithful set_start per span: top-K diagonals -> monotonic decode
    over tracklist order (mix_fp_hits.decode_placements), the SAME primitive that
    gives regular-stem 4-7s. Uses only slot order, no GT leak. Spans whose
    correct ref has no fp get None.
    """
    from alignment.mix_fp_hits import decode_placements

    order = sorted(range(len(spans)), key=lambda i: spans[i].set_start_s)
    # NB decode enforces non-decreasing set_start; feed spans in GT-set order.
    idx = [i for i in order if pool_fps.get(spans[i].track_id) is not None]
    ref_fps = [pool_fps[spans[i].track_id] for i in idx]
    if not ref_fps:
        return [None] * len(spans)
    placed = decode_placements(mix_hashes, ref_fps, mix_dur_s=mix_dur_s, topk=6)
    out: list[float | None] = [None] * len(spans)
    for i, p in zip(idx, placed):
        out[i] = float(p[0]) if p is not None else None
    return out


# ------------------------- MERT identity baseline -------------------------


def mert_identity(
    span: Span,
    mix: "object",
    refs: dict,
    pool_ids: set[str],
) -> bool | None:
    """Cosine-match the span's mix-window MERT to candidate ref track-means.

    mix/refs are mert_store.MertSeries (regular audio, layer 6). Baseline for
    instrumental identity: there is no instrumental-specific MERT, so this is
    exactly what the current pipeline uses. Restrict candidates to the pool
    (refs present in this set's MERT bundle AND with an instrumental stem)."""
    mv = mix.pool(span.set_start_s, span.set_end_s)
    if mv is None:
        return None
    mv = mv / (np.linalg.norm(mv) + 1e-9)
    best_tid, best_cos = None, -2.0
    for tid, series in refs.items():
        if tid not in pool_ids:
            continue
        try:
            rv = series.track_mean()
        except ValueError:
            continue
        rv = rv / (np.linalg.norm(rv) + 1e-9)
        cos = float(mv @ rv)
        if cos > best_cos:
            best_cos, best_tid = cos, tid
    if best_tid is None:
        return None
    if span.track_id not in refs:
        return None  # correct ref absent from MERT bundle -> cannot ID
    return best_tid == span.track_id


# ------------------------- chroma placement baseline ----------------------


def chroma_set_start(
    span: Span,
    mix_instr_y: np.ndarray,
    *,
    band_s: float = 90.0,
    win_s: float = 20.0,
) -> float | None:
    """Chroma matched-filter set_start: slide the ref-instrumental window over
    a band of mix_instrumental, take the argmax-correlation mix time.

    This mirrors the known-0% instrumental chroma placement path: chroma locks
    onto whatever content correlates in the layered mix. Returns the mix time
    (set_start estimate) of the best chroma match within +-band_s of GT set_start
    -> an OPTIMISTIC chroma baseline (band anchored on truth); if fp beats even
    this, the verdict is clear.
    """
    if span.instr_path is None:
        return None
    from alignment.refine_ref_offsets import chroma

    try:
        ry = _load_audio(span.instr_path)
    except Exception:  # noqa: BLE001
        return None
    ref_win = ry[: int(win_s * SR)]
    if ref_win.size < int(4 * SR):
        return None
    rf = chroma(ref_win)
    lo = max(0.0, span.set_start_s - band_s)
    hi = span.set_start_s + band_s
    mf_full = chroma(mix_instr_y[int(lo * SR) : int(hi * SR)])
    if mf_full.shape[1] <= rf.shape[1]:
        return None
    # normalized cross-correlation of chroma column means (cheap matched filter)
    r = rf / (np.linalg.norm(rf, axis=0, keepdims=True) + 1e-9)
    m = mf_full / (np.linalg.norm(mf_full, axis=0, keepdims=True) + 1e-9)
    w = r.shape[1]
    best_j, best_s = 0, -2.0
    for j in range(m.shape[1] - w):
        s = float((m[:, j : j + w] * r).sum())
        if s > best_s:
            best_s, best_j = s, j
    # chroma frame rate: refine_ref_offsets.chroma hop -> derive from shapes
    frames_per_s = mf_full.shape[1] / (hi - lo)
    return lo + best_j / frames_per_s


# ------------------------- scoring ----------------------------------------


def _pct(errs: list[float], thr: float) -> float:
    return 100.0 * sum(1 for e in errs if e <= thr) / len(errs) if errs else 0.0


def run_set(key: str, *, do_chroma: bool, do_mert: bool) -> dict:
    set_id, set_dir, spans = load_spans(key)
    print(f"\n=== {key} ({set_id}): {len(spans)} instrumental GT spans ===")

    n_resolved = sum(1 for s in spans if s.instr_path is not None)
    print(f"  spans with a resolved ref instrumental stem: {n_resolved}/{len(spans)}")
    pool = candidate_pool(set_dir, set_id, spans)
    print(f"  candidate instrumental refs on disk: {len(pool)}")
    pool_fps = build_pool_fps(pool)
    print("  fingerprinting mix_instrumental.flac ...")
    mix_hashes = mix_instr_hashes(set_dir)

    mix_series = None
    refs = None
    pool_ids = set(pool_fps.keys())
    if do_mert:
        from core.result import Ok
        from alignment.mert_store import load_bb12_mert

        res = load_bb12_mert(set_id)
        if isinstance(res, Ok):
            _saudio, mix_series, refs = res.value
        else:
            print(f"  [MERT unavailable: {res}]")
            do_mert = False

    mix_instr_y = None
    if do_chroma:
        mix_instr_y = _load_audio(str(set_dir / "mix_instrumental.flac"))

    mix_dur = max(sp.set_end_s for sp in spans) + 60.0
    decoded = decode_instr_placements(spans, mix_hashes, pool_fps, mix_dur_s=mix_dur)

    rows = []
    for sp, dec_ss in zip(spans, decoded):
        fp_id, n_cand, fp_ss = fp_identity_and_placement(sp, mix_hashes, pool_fps)
        mert_id = (
            mert_identity(sp, mix_series, refs, pool_ids)
            if do_mert and mix_series is not None
            else None
        )
        ch_ss = chroma_set_start(sp, mix_instr_y) if do_chroma else None
        rows.append((sp, fp_id, fp_ss, mert_id, ch_ss, n_cand, dec_ss))
        fp_e = abs(fp_ss - sp.set_start_s) if fp_ss is not None else None
        dec_e = abs(dec_ss - sp.set_start_s) if dec_ss is not None else None
        ch_e = abs(ch_ss - sp.set_start_s) if ch_ss is not None else None
        print(
            f"  {sp.slot_label:5} gt_ss={sp.set_start_s:7.1f}  "
            f"fp_id={_b(fp_id)} fp_raw={_f(fp_ss)} e={_f(fp_e)}  "
            f"dec_e={_f(dec_e)}  mert_id={_b(mert_id)}  ch_e={_f(ch_e)}  "
            f"| {sp.display[:30]}"
        )
    return {"key": key, "rows": rows}


def _b(v) -> str:
    return "-" if v is None else ("Y" if v else "n")


def _f(v) -> str:
    return "  -  " if v is None else f"{v:5.1f}"


def summarize(results: list[dict]) -> None:
    def collect(rows, kind):
        out = []
        for sp, fp_id, fp_ss, mert_id, ch_ss, _n, dec_ss in rows:
            if kind == "fp_id":
                out.append(fp_id)
            elif kind == "mert_id":
                out.append(mert_id)
            elif kind == "fp_err":
                out.append(abs(fp_ss - sp.set_start_s) if fp_ss is not None else None)
            elif kind == "dec_err":
                out.append(abs(dec_ss - sp.set_start_s) if dec_ss is not None else None)
            elif kind == "ch_err":
                out.append(abs(ch_ss - sp.set_start_s) if ch_ss is not None else None)
        return out

    print("\n" + "=" * 74)
    print("SUMMARY — instrumental identity + set_start placement")
    print("=" * 74)

    def id_acc(vals):
        scored = [v for v in vals if v is not None]
        return (
            100.0 * sum(1 for v in scored if v) / len(scored)
            if scored
            else float("nan"),
            len(scored),
            sum(1 for v in vals if v is None),
        )

    def placement_stats(errs):
        e = [x for x in errs if x is not None]
        if not e:
            return None
        return (
            float(np.median(e)),
            float(np.mean(e)),
            _pct(e, 5),
            _pct(e, 15),
            len(e),
        )

    per = {}
    pooled = {k: [] for k in ("fp_id", "mert_id", "fp_err", "dec_err", "ch_err")}
    for res in results:
        rows = res["rows"]
        per[res["key"]] = {k: collect(rows, k) for k in pooled}
        for k in pooled:
            pooled[k] += per[res["key"]][k]

    print("\n-- IDENTITY (accuracy %, n scored, n unidentifiable) --")
    print(
        f"{'set':6} {'fp acc':>9} {'fp n':>5} {'fp None':>7}   {'mert acc':>9} {'mert n':>6} {'mert None':>9}"
    )
    for key in [r["key"] for r in results] + (["POOLED"] if len(results) > 1 else []):
        v = per[key] if key != "POOLED" else pooled
        fa, fn, fnone = id_acc(v["fp_id"])
        ma, mn, mnone = id_acc(v["mert_id"])
        print(f"{key:6} {fa:8.1f}% {fn:5d} {fnone:7d}   {ma:8.1f}% {mn:6d} {mnone:9d}")

    print("\n-- SET_START PLACEMENT (error seconds) --")
    print(
        f"{'set':6} {'src':6} {'median':>8} {'mean':>8} {'<5s%':>7} {'<15s%':>7} {'n':>5}"
    )
    for key in [r["key"] for r in results] + (["POOLED"] if len(results) > 1 else []):
        v = per[key] if key != "POOLED" else pooled
        for src, errs in (
            ("fp_raw", v["fp_err"]),
            ("fp_dec", v["dec_err"]),
            ("chroma", v["ch_err"]),
        ):
            st = placement_stats(errs)
            if st is None:
                print(f"{key:6} {src:6}    (no data)")
            else:
                md, mn, p5, p15, n = st
                print(
                    f"{key:6} {src:6} {md:8.1f} {mn:8.1f} {p5:7.0f} {p15:7.0f} {n:5d}"
                )

    print("\nVERDICT reference: regular-stem fp ~= 76% id / ~4-7s placement.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", action="append", choices=list(SETS), help="repeatable")
    p.add_argument("--no-mert", action="store_true", help="skip MERT identity baseline")
    p.add_argument(
        "--no-chroma", action="store_true", help="skip chroma placement baseline (slow)"
    )
    args = p.parse_args(argv)
    keys = args.set or ["bb11", "bb12"]
    results = [
        run_set(k, do_chroma=not args.no_chroma, do_mert=not args.no_mert) for k in keys
    ]
    summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Lyrics-ASR alignment channel for acappella placement.

The acappella axis is the weak spot of the acoustic (MERT/chroma/HuBERT)
matched-filter aligner: key changes, tempo-stretch, and repeated choruses all
defeat it (see docs/acappella_warp_decode_plan.md, the project memories). Lyrics
sidestep all three — words are key/tempo/pitch invariant and (mostly) unique per
song, which is what a human actually uses to place a vocal by ear.

Pipeline (per set):
  1. Whisper word-timestamp transcription of mix_vocals.flac + each candidate
     acappella vocals stem (cached; GPU-transcribable, see scripts notes).
  2. For each acappella span, find candidate alignment DIAGONALS by Hough voting
     over (slope, intercept) on shared word-bigram anchors, scored by DISTINCT
     RARE bigrams (weight 1/df) so common filler doesn't dominate.
  3. Joint monotonic decode over tracklist order with a Gaussian position prior
     (each span anchored near slot_fraction * mix_dur), abstaining when no
     candidate is good enough.

BB12 acappella set_start vs GT (n=49): median 2.3s, 62% <5s, 74% <15s, 42/49
placed / 7 abstained — vs the prior pipeline's 42.5s median (~18x). At the oracle
ceiling (2.1s). Memory: project_lyrics_alignment_channel.

Transcription needs torch+transformers (Whisper large-v3-turbo); on Mac MPS it's
~2-3 min/stem, on a Vast GPU ~18s/stem (transcribe-on-box -> import JSON by cache
key). The eval below runs on the cache alone (no GPU).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.lyrics_align \
        --set-dir "$HOME/aligning/1fsnxchk__*" \
        --gt labeling/fixtures/bb12_ground_truth.yaml --eval
    # warm the cache (transcribe mix + candidates) first if needed:
    venvs/audio/bin/python -m workspaces.alignment_prototype.lyrics_align \
        --set-dir "$HOME/aligning/1fsnxchk__*" --gt ... --transcribe
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CACHE = Path(__file__).resolve().parent / ".cache" / "lyrics"
MODEL = "openai/whisper-large-v3-turbo"
BLOCK_S = 300.0

# diagonal search / decode params
SLOPES = np.arange(0.5, 2.01, 0.02)
INTERCEPT_BIN = 4.0
INLIER_TOL = 6.0
MIN_DISTINCT = 3
MONO_SLACK = 5.0
ABSTAIN_PEN = 0.6
POS_SIGMA = 150.0

_PIPE = None


# --- transcription (cached) ----------------------------------------------
def _pipe():
    global _PIPE
    if _PIPE is None:
        import torch
        from transformers import pipeline

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"loading {MODEL} on {dev} …", file=sys.stderr)
        _PIPE = pipeline(
            "automatic-speech-recognition",
            model=MODEL,
            device=dev,
            dtype=torch.float16 if dev == "mps" else torch.float32,
            chunk_length_s=28,
            stride_length_s=4,
        )
    return _PIPE


def _cache_file(path: str | Path) -> Path:
    st = Path(path).stat()
    key = hashlib.md5(f"{path}:{st.st_mtime}:{st.st_size}".encode()).hexdigest()[:16]
    return CACHE / f"{key}.json"


def load_cached(path: str | Path) -> list[dict] | None:
    cf = _cache_file(path)
    return json.loads(cf.read_text()) if cf.is_file() else None


def transcribe_words(path: str | Path) -> list[dict]:
    """[{w,s,e}] word stream, global timestamps; blocked for bounded GPU memory."""
    cf = _cache_file(path)
    if cf.is_file():
        return json.loads(cf.read_text())
    import librosa

    dur = librosa.get_duration(path=str(path))
    words: list[dict] = []
    t0 = 0.0
    while t0 < dur:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(
                str(path), sr=16000, mono=True, offset=t0, duration=BLOCK_S
            )
        if y.size:
            out = _pipe()(
                y,
                return_timestamps="word",
                generate_kwargs={"language": "en", "task": "transcribe"},
            )
            for ch in out.get("chunks", []):
                ts = ch.get("timestamp") or (None, None)
                w = (ch.get("text") or "").strip()
                if w and ts[0] is not None:
                    words.append(
                        {
                            "w": w,
                            "s": float(ts[0]) + t0,
                            "e": float(ts[1] or ts[0]) + t0,
                        }
                    )
        t0 += BLOCK_S
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(words))
    return words


# --- word/bigram features ------------------------------------------------
def _norm(words: list[dict]) -> list[tuple[str, float]]:
    out = []
    for x in words:
        w = re.sub(r"[^a-z']", "", x["w"].lower())
        if w:
            out.append((w, 0.5 * (x["s"] + x["e"])))
    return out


def _word_window(seq: list[tuple[str, float]], t: float, half: float = 8.0) -> set[str]:
    """Distinctive words within +/-half seconds of time t (drops 2-char fillers)."""
    return {w for (w, wt) in seq if abs(wt - t) <= half and len(w) > 2}


def same_fiber(seq, t1: float, t2: float, thresh: float = 0.5) -> bool:
    """Are the lyrics at t1 and t2 the SAME (a repeated section = one fiber)? Jaccard
    of the surrounding word windows. Lets ref_start scoring credit picking a
    different-but-equivalent chorus instance, the ref-side repeat ambiguity."""
    a, b = _word_window(seq, t1), _word_window(seq, t2)
    if len(a) < 3 or len(b) < 3:
        return False
    return len(a & b) / len(a | b) >= thresh


def _bigram_times(seq: list[tuple[str, float]]) -> dict[str, list[float]]:
    g: dict[str, list[float]] = {}
    for i in range(len(seq) - 1):
        g.setdefault(seq[i][0] + " " + seq[i + 1][0], []).append(
            0.5 * (seq[i][1] + seq[i + 1][1])
        )
    return g


# --- candidate diagonals (IDF / distinct-bigram Hough) -------------------
def candidate_diagonals(cand_seq, mix_bt, k=5):
    """[(set_start, ref_start, score)] — score = sum of weights (1/df) of DISTINCT
    shared bigrams lying on the diagonal, so a sequence of distinctive lyric words
    dominates and scattered common filler doesn't."""
    cbt = _bigram_times(cand_seq)
    anchors = []
    for bg, cts in cbt.items():
        if bg in mix_bt:
            w = 1.0 / len(mix_bt[bg])
            for ct in cts:
                for mt in mix_bt[bg]:
                    anchors.append((ct, mt, bg, w))
    if len(anchors) < MIN_DISTINCT:
        return []
    ct = np.array([a[0] for a in anchors])
    mt = np.array([a[1] for a in anchors])
    bg = [a[2] for a in anchors]
    w = np.array([a[3] for a in anchors])
    found, used = [], np.zeros(len(anchors), bool)
    for _ in range(k):
        best = None
        for sl in SLOPES:
            bins = np.round((mt - sl * ct) / INTERCEPT_BIN).astype(int)
            for b in np.unique(bins):
                a0 = b * INTERCEPT_BIN
                mask = (np.abs(mt - (a0 + sl * ct)) < INLIER_TOL) & (~used)
                if mask.sum() < MIN_DISTINCT:
                    continue
                seen = {}
                for i in np.where(mask)[0]:
                    seen.setdefault(bg[i], w[i])
                if len(seen) < MIN_DISTINCT:
                    continue
                score = float(sum(seen.values()))
                if best is None or score > best[0]:
                    best = (score, mask)
        if best is None:
            break
        score, mask = best
        inl_ct = ct[mask]
        i0 = int(np.argmin(inl_ct))
        found.append((float(mt[mask][i0]), float(inl_ct[i0]), score))
        used |= mask
    return found


# --- monotonic decode with position prior --------------------------------
def _slot_order(slot: str) -> tuple[int, int]:
    s = str(slot)
    base = "".join(c for c in s.split("w")[0] if c.isdigit()) or "0"
    sub = (
        "".join(c for c in (s.split("w")[1] if "w" in s else "0") if c.isdigit()) or "0"
    )
    return int(base), int(sub)


def monotonic_decode(spans):
    """spans = [(cands, expected_pos)] in slot order. Returns chosen (set_start,
    ref_start) per span ((None, None)=abstain). DP maximizing total position-
    weighted score s.t. set_start non-decreasing across tracklist order."""
    opts = []
    for cands, epos in spans:
        o = []
        for ss, rs, score in cands:
            w = (
                1.0
                if epos is None
                else float(np.exp(-0.5 * ((ss - epos) / POS_SIGMA) ** 2))
            )
            o.append((ss, rs, score * w))
        o.append((None, None, ABSTAIN_PEN))
        opts.append(o)
    n = len(opts)
    NEG = -1e18
    dp = [[NEG] * len(opts[i]) for i in range(n)]
    back = [[-1] * len(opts[i]) for i in range(n)]
    for j, (_ss, _rs, sc) in enumerate(opts[0]):
        dp[0][j] = sc
    for i in range(1, n):
        for j, (ss, _rs, sc) in enumerate(opts[i]):
            for k, (pss, _prs, _psc) in enumerate(opts[i - 1]):
                if dp[i - 1][k] <= NEG:
                    continue
                if not ((ss is None) or (pss is None) or (ss >= pss - MONO_SLACK)):
                    continue
                v = dp[i - 1][k] + sc
                if v > dp[i][j]:
                    dp[i][j] = v
                    back[i][j] = k
    j = int(np.argmax(dp[-1]))
    chosen = [(None, None)] * n
    for i in range(n - 1, -1, -1):
        chosen[i] = (opts[i][j][0], opts[i][j][1])
        if i > 0:
            j = back[i][j]
    return chosen


# --- set assembly + eval -------------------------------------------------
def _resolve_glob(pat: str) -> Path:
    hits = glob.glob(str(Path(pat).expanduser()))
    if not hits:
        sys.exit(f"no match for {pat}")
    return Path(sorted(hits)[0])


def _build_spans(set_dir: Path, gt: dict, cached_only: bool):
    import yaml  # noqa: F401  (gt already parsed; kept for parity)

    manifest = json.loads((set_dir / "manifest.json").read_text())
    byid = {t["track_id"]: t for t in manifest["tracks"]}
    mix_bt = _bigram_times(_norm(transcribe_words(set_dir / "mix_vocals.flac")))
    mix_dur = float(manifest.get("mix_duration_s") or 0) or max(
        float(t.get("set_end_s") or 0) for t in gt["tracks"]
    )
    max_slot = (
        max(
            (
                _slot_order(t["slot_label"])[0]
                for t in gt["tracks"]
                if t.get("slot_label")
            ),
            default=1,
        )
        or 1
    )
    aca = [s for s in gt["tracks"] if s.get("claimed_stem") == "acappella"]
    aca.sort(key=lambda s: _slot_order(s["slot_label"]))
    spans, meta = [], []
    for s in aca:
        vpath = (byid.get(s["track_id"], {}).get("stems") or {}).get("vocals")
        if not vpath or not Path(vpath).is_file():
            continue
        cw = load_cached(vpath) if cached_only else transcribe_words(vpath)
        if not cw:
            continue
        cands = candidate_diagonals(_norm(cw), mix_bt)
        if not cands:
            continue
        epos = _slot_order(s["slot_label"])[0] / max_slot * mix_dur
        spans.append((cands, epos))
        meta.append(
            (
                s["slot_label"],
                s["track"][:30],
                float(s["set_start_s"]),
                float(s.get("ref_start_s") or 0.0),
                cands,
                _norm(cw),  # candidate word stream (for lyric-fiber scoring)
            )
        )
    return spans, meta


def main(argv: list[str] | None = None) -> int:
    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-dir", required=True)
    p.add_argument(
        "--gt", default=str(_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    )
    p.add_argument("--eval", action="store_true", help="score vs GT (cache only)")
    p.add_argument(
        "--transcribe", action="store_true", help="warm cache (needs GPU/MPS)"
    )
    args = p.parse_args(argv)

    set_dir = _resolve_glob(args.set_dir)
    gt = yaml.safe_load(Path(args.gt).read_text())
    spans, meta = _build_spans(set_dir, gt, cached_only=not args.transcribe)
    print(f"acappella spans with candidates: {len(spans)}")
    if not args.eval:
        return 0
    if len(spans) < 2:
        print("not enough cached spans — run --transcribe first")
        return 0

    chosen = monotonic_decode(spans)
    mono, strong, oracle = [], [], []
    ref_mono, ref_oracle, ref_fiber = [], [], []
    n_fiber_credit = 0
    print(
        "\n  slot  track                          GT_ss   mono_ss  | GT_rs  mono_rs  fib"
    )
    for (slot, name, gt_ss, gt_rs, cands, cseq), (ch_ss, ch_rs) in zip(meta, chosen):
        strong.append(abs(max(cands, key=lambda c: c[2])[0] - gt_ss))
        oracle.append(abs(min(cands, key=lambda c: abs(c[0] - gt_ss))[0] - gt_ss))
        ref_oracle.append(abs(min(cands, key=lambda c: abs(c[1] - gt_rs))[1] - gt_rs))
        fib = ""
        if ch_ss is not None:
            mono.append(abs(ch_ss - gt_ss))
            re_err = abs(ch_rs - gt_rs)
            ref_mono.append(re_err)
            # fiber-aware: a different chorus instance with the SAME lyrics counts
            if re_err < 5 or same_fiber(cseq, ch_rs, gt_rs):
                ref_fiber.append(0.0)
                if re_err >= 5:
                    n_fiber_credit += 1
                    fib = "=fiber"
            else:
                ref_fiber.append(re_err)
            ms, mr = f"{abs(ch_ss - gt_ss):.1f}", f"{re_err:.1f}"
        else:
            ms = mr = "abstain"
        print(
            f"  {slot:5s} {name:30s} {gt_ss:6.0f}  {ms:>8s} | {gt_rs:6.0f} {mr:>8s}  {fib}"
        )

    def stat(x, lab):
        a = np.array(x)
        print(
            f"  {lab:32s} median {np.median(a):6.1f}s  <5s {100 * np.mean(a < 5):3.0f}%"
            f"  <15s {100 * np.mean(a < 15):3.0f}%   (n={len(a)})"
        )

    print("\n=== acappella SET_START error (prior pipeline baseline: 42.5s) ===")
    stat(strong, "strongest-line (IDF)")
    stat(mono, "MONOTONIC + position prior")
    stat(oracle, "oracle ceiling")
    print("\n=== acappella REF_START error (which part of the song; ~50s wall) ===")
    stat(ref_mono, "MONOTONIC (strict)")
    stat(ref_fiber, f"MONOTONIC (fiber-aware, +{n_fiber_credit} equiv-repeat)")
    stat(ref_oracle, "oracle ceiling (strict)")
    print(f"\nabstained: {sum(1 for c in chosen if c[0] is None)}/{len(chosen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

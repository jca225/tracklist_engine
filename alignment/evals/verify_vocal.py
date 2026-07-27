#!/usr/bin/env python3
"""verify_vocal.py — vocal-identity VERIFIER (offline proof of mechanism).

The open-set idea (John, 2026-07-06): a DJ rework preserves the ORIGINAL's
vocal, so identity that the full-mix fingerprint misses can be recovered by
matching the mix vocal against a candidate reference vocal on the
key/tempo-invariant axis (HuBERT). This is a VERIFY loop, not a 1-of-N decode:
propose a candidate → separate + match → ACCEPT / REJECT-and-try-next / ABSTAIN.

This module proves the verifier DECISION before any fetch/retrieve plumbing:
- separate the mix vocal in a region (Demucs, local/MPS),
- score it against each candidate ref vocal via `stem_placement.place_joint`
  (HuBERT slide → on-diagonal match peak),
- decide from the peak + the margin over the runner-up, with a no-vocal ABSTAIN.

Run (offline, no pi-storage):
    venvs/audio/bin/python -m alignment.evals.verify_vocal --set-id 1rfb0yl9

Proof harness: known-truth windows (confident vocal tracks + one instrumental)
× candidate ref vocals → a confusion matrix; the diagonal should win, the
instrumental row should ABSTAIN.
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alignment.stem_placement import (  # noqa: E402
    FPS,
    hubert_of,
    place_joint,
)

# decision thresholds (HuBERT correlate_window peaks are ~cosine, 0..1)
ACCEPT_PEAK = 0.45
ACCEPT_MARGIN = 0.06
VOCAL_RMS_FLOOR = 0.010  # separated-vocal RMS below this => no vocal => ABSTAIN

# known-truth Mammoth windows: (slot, set_start_s, expected). Confident fp tracks;
# 030 Bamako is the likely-instrumental ABSTAIN control.
WINDOWS = [
    ("012", 2124.3, "vocal"),  # Cutting Loose (Anabel Englund)
    ("016", 2779.2, "vocal"),  # Last Night (acappella slot)
    ("019", 3308.5, "vocal"),  # Save Me
    ("022", 3884.6, "vocal"),  # Disco Boy
    ("030", 5537.7, "instr?"),  # Bamako — abstain control
]
CAND_SLOTS = ["012", "016", "019", "022", "030"]


def _slot_dir(set_dir: Path, slot: str) -> Path | None:
    hits = sorted(glob.glob(str(set_dir / "stems" / f"{slot}__*")))
    return Path(hits[0]) if hits else None


def _ref_vocal(set_dir: Path, slot: str) -> Path | None:
    d = _slot_dir(set_dir, slot)
    if not d:
        return None
    p = d / "vocals.flac"
    return p if p.is_file() else None


def _sep_mix_vocal(
    mix_path: Path, t0: float, t1: float, model, device: str, tmp: Path
) -> tuple[Path, float]:
    """Slice mix[t0:t1], Demucs-separate the vocal, return (wav_path, vocal_rms)."""
    import librosa
    import torch
    from demucs.apply import apply_model

    voc_path = tmp / f"voc_{int(t0)}.wav"
    if voc_path.is_file():  # cached separation — reuse (keeps Whisper cache warm too)
        v, _ = sf.read(str(voc_path))
        return voc_path, float(np.sqrt(np.mean(v**2)))

    sr = model.samplerate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(
            str(mix_path), sr=sr, mono=False, offset=t0, duration=t1 - t0
        )
    if y.ndim == 1:
        y = np.stack([y, y])
    wav = torch.tensor(y, dtype=torch.float32)  # (2, N)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, wav[None], device=device, split=True, overlap=0.25)[0]
    out = out * ref.std() + ref.mean()
    voc = out[model.sources.index("vocals")].mean(0).cpu().numpy()  # mono
    voc_path = tmp / f"voc_{int(t0)}.wav"
    sf.write(str(voc_path), voc, sr)
    return voc_path, float(np.sqrt(np.mean(voc**2)))


def _decide(peaks: dict[str, float], vocal_rms: float) -> tuple[str, str, float]:
    """(decision, best_slot, margin) from candidate peaks + mix-vocal energy."""
    if vocal_rms < VOCAL_RMS_FLOOR:
        return "ABSTAIN(no-vocal)", "-", 0.0
    ranked = sorted(peaks.items(), key=lambda kv: -kv[1])
    best_slot, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best - second
    if best >= ACCEPT_PEAK and margin >= ACCEPT_MARGIN:
        return "ACCEPT", best_slot, margin
    return "REJECT(try-next)", best_slot, margin


def _wordset(path: Path | str | None) -> set[str]:
    """Content words (>=3 chars) transcribed from an audio file (Whisper, cached)."""
    if path is None:
        return set()
    from alignment.lyrics_align import _norm, transcribe_words

    return {w for w, _ in _norm(transcribe_words(path)) if len(w) >= 3}


def _idf_recall(
    mix_words: set[str], cand_words: set[str], idf: dict[str, float], default: float
) -> float:
    """IDF-weighted fraction of the window's words that appear in a candidate.

    Weighting by inverse candidate-frequency downweights filler words shared across
    many candidates ("you", "night") and lets a few distinctive words decide."""
    den = sum(idf.get(w, default) for w in mix_words) or 1.0
    num = sum(idf.get(w, default) for w in (mix_words & cand_words))
    return num / den


def _decide_lyrics(
    lyr: dict[str, float], vocal_rms: float, n_words: int
) -> tuple[str, str, float]:
    """Verdict from lyric-overlap recall (fraction of window words in a candidate)."""
    if vocal_rms < VOCAL_RMS_FLOOR or n_words < 3:
        return "ABSTAIN(no-vocal/words)", "-", 0.0
    ranked = sorted(lyr.items(), key=lambda kv: -kv[1])
    best_slot, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best - second
    if best >= 0.5 and margin >= 0.2:
        return "ACCEPT", best_slot, margin
    return "REJECT(try-next)", best_slot, margin


def run_localize(set_dir: Path, win_s: float) -> None:
    """Whole-track search: slide each candidate over the FULL mix vocal to recover
    its true region, then lyrics-verify THERE — no fixed-window guessing.

    Proves the fix for the 016/019 confusion: if a candidate localizes near its
    known set_start and the recovered window's lyrics self-match, the verifier just
    needed a clean window, not a better score."""
    import math
    from collections import Counter

    import soundfile as sf

    mixv = set_dir / "mix_vocals.flac"
    if not mixv.is_file():
        sys.exit(f"no {mixv} — run the mix-vocal separation first")

    print("HuBERT over full mix_vocals…", flush=True)
    mix_feat = hubert_of(mixv)
    dur = mix_feat.shape[1] / FPS
    known = {s: ss for s, ss, _ in WINDOWS}

    cand_feat: dict[str, np.ndarray] = {}
    cand_words: dict[str, set[str]] = {}
    for s in CAND_SLOTS:
        p = _ref_vocal(set_dir, s)
        f = hubert_of(p) if p else None
        if f is not None:
            cand_feat[s] = f
        cand_words[s] = _wordset(p)
    df = Counter(w for s in CAND_SLOTS for w in cand_words[s])
    idf = {w: math.log((len(CAND_SLOTS) + 1) / df[w]) for w in df}
    default_idf = math.log(len(CAND_SLOTS) + 1)

    mixy, msr = sf.read(str(mixv))
    cache = (
        _REPO
        / "alignment/.cache/verify_windows"
        / f"{set_dir.name.split('__')[0]}_loc"
    )
    cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in CAND_SLOTS:
        loc = (
            place_joint(
                mix_feat,
                cand_feat[s],
                prior_ss=dur / 2,
                span_dur=0.0,
                band_s=dur / 2 + 10,
            )
            if s in cand_feat
            else None
        )
        if loc is None:
            rows.append((s, None, 0.0, 0, {}))
            continue
        rss, _, peak = loc
        seg = mixy[int(rss * msr) : int((rss + win_s) * msr)]
        segp = cache / f"loc_{s}.wav"
        sf.write(str(segp), seg, msr)
        mw = _wordset(segp)
        lyr = {
            c: round(_idf_recall(mw, cand_words[c], idf, default_idf), 3)
            for c in CAND_SLOTS
        }
        rows.append((s, rss, peak, len(mw), lyr))
        print(
            f"  cand {s}: recovered {rss:.0f}s (known {known[s]:.0f}s) peak {peak:.2f}",
            flush=True,
        )

    hdr = (
        f"{'cand':4} {'recov':>6} {'known':>6} {'Δs':>5} {'peak':>5}  "
        + "  ".join(f"L{c}" for c in CAND_SLOTS)
        + "  verify"
    )
    print(f"\n[Localize + verify — whole-track search]\n{hdr}\n{'-' * len(hdr)}")
    for s, rss, peak, nw, lyr in rows:
        if rss is None:
            print(f"{s:4}   (no localization)")
            continue
        best = max(lyr, key=lyr.get) if lyr else "-"
        ok = "✓" if best == s else " "
        cells = "  ".join(f"{lyr.get(c, 0):.2f}" for c in CAND_SLOTS)
        print(
            f"{s:4} {rss:6.0f} {known[s]:6.0f} {abs(rss - known[s]):5.0f} {peak:5.2f}  {cells}  L{best} {ok}"
        )
    print("\n(recovered≈known = localization worked; L-self-match ✓ = verify worked)")


def run_banded(set_dir: Path, set_id: str, win_s: float) -> None:
    """Production verifier: banded search primed by the fingerprint timeline.

    Confident slots (fp votes>=100) search ±90s around their fp set_start; abstained
    slots search only their tracklist-order BRACKET between the neighbouring confident
    anchors. ~100x faster than blind, and it targets the actual open-set question:
    are the abstained REWORKS' preserved vocals detectable in the mix (identity the
    full-mix fingerprint missed)?"""
    import json
    import math
    import re
    from collections import Counter

    import soundfile as sf

    mixv = set_dir / "mix_vocals.flac"
    if not mixv.is_file():
        sys.exit(f"no {mixv} — run the mix-vocal separation first")
    tl = _REPO / "alignment/out" / f"{set_id}_fused_timeline.json"
    spans = json.loads(tl.read_text())["spans"]

    def base(sl: str) -> int:
        m = re.match(r"(\d+)", sl)
        return int(m.group(1)) if m else 999

    conf = sorted(
        [(s["slot_label"], s["set_start_s"]) for s in spans if s["votes"] >= 100],
        key=lambda x: x[1],
    )
    conf_ss = {base(sl): ss for sl, ss in conf}

    print("HuBERT over full mix_vocals…", flush=True)
    mix_feat = hubert_of(mixv)
    dur = mix_feat.shape[1] / FPS

    def bracket(slot: str) -> tuple[float, float]:
        n = base(slot)
        prev = [ss for sl, ss in conf if base(sl) < n]
        nxt = [ss for sl, ss in conf if base(sl) > n]
        return (max(prev) if prev else 0.0, min(nxt) if nxt else dur)

    # confident vocal controls (should reproduce the blind result, fast) + the
    # abstained reworks (the real target — is the preserved vocal there?)
    targets = ["012", "016", "019", "004", "006", "008", "010"]
    ref_feat: dict[str, np.ndarray | None] = {}
    ref_words: dict[str, set[str]] = {}
    for s in targets:
        p = _ref_vocal(set_dir, s)
        ref_feat[s] = hubert_of(p) if p else None
        ref_words[s] = _wordset(p)
    df = Counter(w for s in targets for w in ref_words[s])
    idf = {w: math.log((len(targets) + 1) / df[w]) for w in df}
    dflt = math.log(len(targets) + 1)

    mixy, msr = sf.read(str(mixv))
    cache = (
        _REPO
        / "alignment/.cache/verify_windows"
        / f"{set_id}_banded"
    )
    cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in targets:
        n = base(s)
        if n in conf_ss:
            prior, band, kind = conf_ss[n], 90.0, "fp±90"
        else:
            lo, hi = bracket(s)
            prior, band, kind = (
                (lo + hi) / 2,
                (hi - lo) / 2 + 10,
                f"brkt[{lo:.0f},{hi:.0f}]",
            )
        if ref_feat[s] is None:
            rows.append((s, kind, None))
            continue
        loc = place_joint(
            mix_feat, ref_feat[s], prior_ss=prior, span_dur=0.0, band_s=band
        )
        if loc is None:
            rows.append((s, kind, None))
            continue
        rss, _, peak = loc
        seg = mixy[int(rss * msr) : int((rss + win_s) * msr)]
        segp = cache / f"b_{s}.wav"
        sf.write(str(segp), seg, msr)
        mw = _wordset(segp)
        lyr = {c: round(_idf_recall(mw, ref_words[c], idf, dflt), 3) for c in targets}
        rows.append((s, kind, (rss, peak, len(mw), lyr)))
        print(
            f"  {s} ({kind}): recovered {rss:.0f}s peak {peak:.2f} words {len(mw)}",
            flush=True,
        )

    reworks = {"004", "006", "008", "010"}
    hdr = f"{'slot':4} {'role':8} {'prior/band':16} {'recov':>6} {'peak':>5} {'self':>5}  verdict"
    print(f"\n[Banded verify — fp/bracket-primed]\n{hdr}\n{'-' * len(hdr)}")
    for s, kind, res in rows:
        role = "rework" if s in reworks else "control"
        if res is None:
            print(f"{s:4} {role:8} {kind:16} {'—':>6}     —      —   no ref/loc")
            continue
        rss, peak, nw, lyr = res
        best = max(lyr, key=lyr.get)
        selfsc = lyr.get(s, 0.0)
        ok = best == s and selfsc >= 0.5 and selfsc - sorted(lyr.values())[-2] >= 0.2
        verdict = (
            f"ACCEPT L{best}"
            if ok
            else ("ABSTAIN(no words)" if nw < 3 else f"REJECT (best L{best})")
        )
        print(
            f"{s:4} {role:8} {kind:16} {rss:6.0f} {peak:5.2f} {selfsc:5.2f}  {verdict}"
        )
    print(
        "\ncontrols should ACCEPT-self; reworks ACCEPT = preserved vocal found where fp abstained"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--win-s", type=float, default=40.0)
    ap.add_argument(
        "--offset-s", type=float, default=25.0, help="skip past the track intro"
    )
    ap.add_argument(
        "--localize",
        action="store_true",
        help="blind whole-track search over mix_vocals.flac (vs fixed-window matrix)",
    )
    ap.add_argument(
        "--banded",
        action="store_true",
        help="production: fp/bracket-primed banded search, targets the abstained reworks",
    )
    args = ap.parse_args()

    hits = glob.glob(str(Path.home() / "aligning" / f"{args.set_id}*"))
    if not hits:
        sys.exit(f"no staged folder for {args.set_id}")
    set_dir = Path(hits[0])
    mix_path = set_dir / "mix.m4a"

    if args.banded:
        run_banded(set_dir, args.set_id, args.win_s)
        return
    if args.localize:
        run_localize(set_dir, args.win_s)
        return

    windows = WINDOWS
    cand_slots = CAND_SLOTS

    print("loading candidate ref vocals (HuBERT + lyrics)…", flush=True)
    cand_feat: dict[str, np.ndarray] = {}
    cand_words: dict[str, set[str]] = {}
    for s in cand_slots:
        p = _ref_vocal(set_dir, s)
        f = hubert_of(p) if p else None
        if f is not None:
            cand_feat[s] = f
        cand_words[s] = _wordset(p)
        print(
            f"  cand {s}: hubert={'ok' if f is not None else 'MISS'} words={len(cand_words[s])}",
            flush=True,
        )

    import math
    from collections import Counter

    df = Counter(w for s in cand_slots for w in cand_words[s])
    idf = {w: math.log((len(cand_slots) + 1) / df[w]) for w in df}
    default_idf = math.log(len(cand_slots) + 1)  # unseen word = maximally distinctive

    import torch
    from demucs.pretrained import get_model

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading Demucs (htdemucs) on {device}…", flush=True)
    model = get_model("htdemucs")
    model.eval()

    tmp = _REPO / "alignment/.cache/verify_windows" / args.set_id
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for slot, ss, expect in windows:
        t0 = ss + args.offset_s
        voc_path, rms = _sep_mix_vocal(
            mix_path, t0, t0 + args.win_s, model, device, tmp
        )
        mix_feat = hubert_of(voc_path)
        peaks = {}
        for cs, cf in cand_feat.items():
            r = place_joint(mix_feat, cf, prior_ss=0.0, span_dur=0.0, band_s=1e6)
            peaks[cs] = round(r[2], 3) if r else 0.0
        mix_words = _wordset(voc_path)
        lyr = {
            cs: round(_idf_recall(mix_words, cand_words[cs], idf, default_idf), 3)
            for cs in cand_slots
        }
        hub = _decide(peaks, rms)
        lyd = _decide_lyrics(lyr, rms, len(mix_words))
        rows.append((slot, expect, rms, len(mix_words), peaks, lyr, hub, lyd))
        print(
            f"  window {slot} ({expect}) done  rms={rms:.3f} words={len(mix_words)}",
            flush=True,
        )

    def _matrix(title: str, vals_idx: int, dec_idx: int) -> None:
        hdr = "true expect  " + "  ".join(f"c{c}" for c in cand_slots) + "   decision"
        print(f"\n[{title}]\n{hdr}\n{'-' * len(hdr)}")
        for r in rows:
            slot, expect, vals = r[0], r[1], r[vals_idx]
            dec, best, mgn = r[dec_idx]
            cells = "  ".join(f"{vals.get(c, 0):.2f}" for c in cand_slots)
            ok = "✓" if best == slot and dec.startswith("ACCEPT") else ""
            print(
                f"{slot:4} {expect:7} {cells}   {dec} (best c{best}, mgn {mgn:.2f}) {ok}"
            )

    _matrix("HuBERT acoustic", 4, 6)
    _matrix("Lyrics IDF-overlap", 5, 7)
    print("\n(correct candidate = the row's own slot; 030 should ABSTAIN)")


if __name__ == "__main__":
    main()

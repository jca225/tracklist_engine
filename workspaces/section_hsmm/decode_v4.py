#!/usr/bin/env python3
"""v4 — NULL-state (abstain) Viterbi: route hard stretches through an explicit
abstain state instead of gating the decode after the fact.

abstain_eval.py proved precision@coverage works, but POST-HOC: it runs the full
v2/v3 decode forcing the path through real song-states at every frame, then
throws away low-margin frames afterward. The cost of that is Viterbi continuity —
a wrong state chosen in a hard stretch becomes the "previous state" its neighbors
must transition from, so one unsure frame can drag the frames around it onto the
wrong song. The honest fix is to let the path itself abstain.

v4 adds one state, NULL, to the state space:

    real  -> NULL     enter abstain   (null_enter_pen)
    NULL  -> NULL     stay abstain    (free)
    NULL  -> real     resume anywhere (null_exit_pen — like a switch, from null)

The principled part is NULL's *emission*. We proved absolute chroma/MERT cosine
is useless for confidence and MARGIN (decoded track vs best other track) is the
signal. So NULL does not emit a constant floor (that would compete on absolute
cosine). Instead, per frame:

    null_emit[t] = second_best_track_max[t] + null_margin

A real track then beats NULL at frame t only if its emission clears the runner-up
track by more than null_margin — i.e. exactly the margin criterion, but decided
INSIDE the path so the continuity benefit (no wrong-state drag) applies. Sweeping
null_margin traces the same precision@coverage curve abstain_eval traces post-hoc,
so the two are directly comparable. That comparison is the experiment.

Same closed vocab, same data-derived transitions (fwd/back/switch/hold/skip) as
v2 (chroma bed) and v3 (MERT overlay); this only changes the decoder.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_v4 \
        --set-id 1fsnxchk [--frame-s 2.0] [--mert-layer 6] \
        [--null-enter-pen 0.10] [--null-exit-pen 0.10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir  # noqa: E402
from workspaces.section_hsmm.decode_hsmm import NEG, Vocab, _gt_track_ids  # noqa: E402
from workspaces.section_hsmm.decode_v2 import (  # noqa: E402
    CHANNELS, _pooled, build_channel_vocab, viterbi_v2,
)
from workspaces.section_hsmm.decode_v3 import build_overlay_vocab_mert, _assemble_vocab  # noqa: E402
from workspaces.section_hsmm.mert_emit import ensure_mert_cache, load_pooled_mert  # noqa: E402

PENS = dict(fwd_pen=0.15, back_pen=0.30, switch_pen=0.50, hold_pen=0.10, skip_pen=0.10)
# null_margin (β) sweep. NULL emits top2+β, so with β>=0 it always beats a
# non-negative margin and the path abstains everywhere — the live operating range
# is a thin negative band, where a real track is kept alive unless it falls BELOW
# the runner-up by |β|. That band is where NULL-state continuity beats post-hoc.
TAUS = (-1.0, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0)


def _track_max(emis: np.ndarray, vocab: Vocab) -> np.ndarray:
    """(T, K) per-frame max emission within each track's ref states."""
    T = emis.shape[0]
    K = len(vocab.tids)
    tm = np.full((T, K), NEG)
    for k, (lo, hi) in enumerate(vocab.slices):
        tm[:, k] = emis[:, lo:hi].max(axis=1)
    return tm


def _null_emit(track_max: np.ndarray, null_margin: float) -> np.ndarray:
    """Per-frame NULL emission = 2nd-best track max + margin. With <2 tracks the
    runner-up is undefined, so fall back to an absolute floor of null_margin."""
    K = track_max.shape[1]
    if K < 2:
        return np.full(track_max.shape[0], null_margin, dtype=np.float64)
    top2 = np.partition(track_max, -2, axis=1)[:, -2]  # 2nd largest per frame
    return top2.astype(np.float64) + null_margin


def viterbi_null(emis: np.ndarray, vocab: Vocab, null_emit: np.ndarray, *,
                 fwd_pen: float, back_pen: float, switch_pen: float,
                 hold_pen: float, skip_pen: float,
                 null_enter_pen: float, null_exit_pen: float) -> np.ndarray:
    """v2's Viterbi extended with state S = NULL. Returns a path over 0..S where
    S means 'abstain' (no prediction at that frame)."""
    T, S = emis.shape
    NULLS = S  # null state index lives one past the real states
    V = np.empty(S + 1, dtype=np.float64)
    V[:S] = emis[0]
    V[S] = null_emit[0]
    bp = np.empty((T, S + 1), dtype=np.int32)
    bp[0] = -1

    is_start = np.zeros(S, dtype=bool)
    start2 = np.zeros(S, dtype=bool)
    for lo, _hi in vocab.slices:
        is_start[lo] = True
        if lo + 1 < S:
            start2[lo + 1] = True
    idx = np.arange(S)

    for t in range(1, T):
        e = emis[t]
        Vr = V[:S]
        # --- transitions among real states (identical to viterbi_v2) ---
        adv = np.full(S, NEG); adv[1:] = Vr[:-1]; adv[is_start] = NEG
        hold = Vr - hold_pen
        skip = np.full(S, NEG); skip[2:] = Vr[:-2] - skip_pen
        skip[is_start] = NEG; skip[start2] = NEG
        local = np.stack([adv, hold, skip])
        local_src = np.stack([idx - 1, idx, idx - 2])
        li = local.argmax(0)
        local_best = np.take_along_axis(local, li[None], 0)[0]
        local_bp = np.take_along_axis(local_src, li[None], 0)[0]
        jump_best = np.full(S, NEG); jump_bp = np.full(S, -1, np.int32)
        for lo, hi in vocab.slices:
            seg = Vr[lo:hi]
            a = int(seg.argmax())
            li_loc = np.arange(hi - lo)
            pen = np.where(li_loc >= a, fwd_pen, back_pen)
            jump_best[lo:hi] = seg[a] - pen
            jump_bp[lo:hi] = lo + a
        g = int(Vr.argmax())
        switch = np.full(S, Vr[g] - switch_pen)
        # --- resume from NULL into any real state (the abstain exit) ---
        from_null = np.full(S, V[S] - null_exit_pen)
        cand = np.stack([local_best, jump_best, switch, from_null])
        cand_bp = np.stack([local_bp, jump_bp, np.full(S, g, np.int32),
                            np.full(S, NULLS, np.int32)])
        ci = cand.argmax(0)
        new_real = np.take_along_axis(cand, ci[None], 0)[0] + e
        new_real_bp = np.take_along_axis(cand_bp, ci[None], 0)[0]
        # --- transitions into NULL: stay abstaining, or enter from best real ---
        enter_null = Vr[g] - null_enter_pen
        stay_null = V[S]
        if enter_null >= stay_null:
            new_null, null_bp = enter_null + null_emit[t], g
        else:
            new_null, null_bp = stay_null + null_emit[t], NULLS
        V = np.empty(S + 1, dtype=np.float64)
        V[:S] = new_real; V[S] = new_null
        bp[t, :S] = new_real_bp; bp[t, S] = null_bp

    path = np.empty(T, dtype=np.int32)
    path[-1] = int(V.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def _channel_emis_vocab(channel, set_dir, set_id, by_tid, gt_tids, frame_s, layer):
    """Return (emis (T,S), vocab) for a channel, reusing the v2/v3 builders."""
    if channel == "bed":
        vocab = build_channel_vocab("bed", by_tid, gt_tids, frame_s)
        cfg = CHANNELS["bed"]
        mix = _pooled(set_dir / cfg["mix_file"], f"{set_id}_{cfg['mix_key']}",
                      f"{set_id}_{cfg['mix_key']}_pool{frame_s}", frame_s)
    else:
        key_of, items = build_overlay_vocab_mert(by_tid, gt_tids, frame_s, layer)
        ensure_mert_cache(items + [(set_dir / CHANNELS["overlay"]["mix_file"],
                                    f"{set_id}_mix_vocals")], frame_s, layer)
        vocab = _assemble_vocab(key_of, frame_s, layer)
        mix = load_pooled_mert(f"{set_id}_mix_vocals", frame_s, layer)
    if not vocab.tids:
        return None, vocab
    emis = (mix @ vocab.emit_ref.T).astype(np.float64)
    return emis, vocab


def _gt_active_mask(channel, frame_s, T, gt_rows):
    """Per-frame: (is_GT_active, set_of_active_tids) for this channel's stems."""
    stems = CHANNELS[channel]["stems"]
    rows = [r for r in gt_rows if (r.get("claimed_stem") or "regular") in stems]
    acts = []
    for i in range(T):
        ts = (i + 0.5) * frame_s
        acts.append({str(r["track_id"]) for r in rows
                     if float(r["set_start_s"]) - 2 <= ts <= float(r["set_end_s"]) + 2})
    return acts


def _posthoc_margin(emis, vocab, path):
    """Reproduce abstain_eval's margin signal for a plain v2 decode: own track
    max minus best OTHER track max, per frame."""
    T = emis.shape[0]
    tm = _track_max(emis, vocab)
    dk = vocab.track_of[path]
    own = tm[np.arange(T), dk]
    other = tm.copy(); other[np.arange(T), dk] = NEG
    return own - other.max(axis=1), dk


def _curve_null(channel, emis, vocab, frame_s, acts, *, null_enter_pen,
                null_exit_pen):
    """NULL-state Viterbi precision@coverage: sweep null_margin."""
    tm = _track_max(emis, vocab)
    rows = []
    for tau in TAUS:
        path = viterbi_null(emis, vocab, _null_emit(tm, tau),
                            null_enter_pen=null_enter_pen, null_exit_pen=null_exit_pen,
                            **PENS)
        S = emis.shape[1]
        npred = ntot = ncorr = 0
        for i, st in enumerate(path):
            if not acts[i]:
                continue
            ntot += 1
            if st == S:  # abstained
                continue
            npred += 1
            if vocab.tids[vocab.track_of[st]] in acts[i]:
                ncorr += 1
        cov = npred / max(ntot, 1)
        prec = ncorr / max(npred, 1)
        rows.append((tau, cov, prec, npred))
    return rows


def _curve_posthoc(channel, emis, vocab, frame_s, acts):
    """Post-hoc margin gating on the plain v2 decode (the abstain_eval method)."""
    path = viterbi_v2(emis, vocab, **PENS)
    margin, dk = _posthoc_margin(emis, vocab, path)
    frames = [(vocab.tids[dk[i]] in acts[i], float(margin[i]))
              for i in range(len(path)) if acts[i]]
    n = len(frames)
    rows = []
    for tau in TAUS:
        pred = [(c, m) for c, m in frames if m >= tau]
        cov = len(pred) / max(n, 1)
        prec = (np.mean([c for c, _ in pred]) if pred else 0.0)
        rows.append((tau, cov, prec, len(pred)))
    return rows


def _print_compare(channel, posthoc, null):
    print(f"\n[{channel}]  margin-gate (post-hoc)   vs   NULL-state Viterbi (v4)")
    print(f"  {'tau':>6} | {'cov':>5} {'prec':>5} | {'cov':>5} {'prec':>5} | {'Δprec@cov':>9}")
    for (tau, c0, p0, _), (_, c1, p1, _) in zip(posthoc, null):
        dp = p1 - p0
        print(f"  {tau:6.2f} | {100*c0:4.0f}% {100*p0:4.0f}% | "
              f"{100*c1:4.0f}% {100*p1:4.0f}% | {100*dp:+8.0f}%")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--mert-layer", type=int, default=6)
    p.add_argument("--null-enter-pen", type=float, default=0.10)
    # exit must be >= switch_pen so routing through NULL is never a cheaper switch
    p.add_argument("--null-exit-pen", type=float, default=0.50)
    p.add_argument("--channels", default="bed,overlay")
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    import yaml
    gt_rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")]
    stem_by_tid: dict[str, str] = {}
    for r in gt_rows:
        stem_by_tid.setdefault(str(r["track_id"]), r.get("claimed_stem") or "regular")
    for tid, t in by_tid.items():
        t["_stem"] = stem_by_tid.get(tid, "regular")
    gt_tids = _gt_track_ids(_REPO / "labeling/fixtures/bb12_ground_truth.yaml")

    print(f"=== v4 NULL-state Viterbi ({args.set_id}) ===")
    print(f"frame={args.frame_s}s  null_enter={args.null_enter_pen} "
          f"null_exit={args.null_exit_pen}  (cov/prec over GT-active frames)")
    for channel in args.channels.split(","):
        emis, vocab = _channel_emis_vocab(channel, set_dir, args.set_id, by_tid,
                                          gt_tids, args.frame_s, args.mert_layer)
        if emis is None:
            print(f"  [{channel}] no vocab", file=sys.stderr)
            continue
        acts = _gt_active_mask(channel, args.frame_s, emis.shape[0], gt_rows)
        posthoc = _curve_posthoc(channel, emis, vocab, args.frame_s, acts)
        null = _curve_null(channel, emis, vocab, args.frame_s, acts,
                           null_enter_pen=args.null_enter_pen,
                           null_exit_pen=args.null_exit_pen)
        _print_compare(channel, posthoc, null)
    return 0


if __name__ == "__main__":
    sys.exit(main())

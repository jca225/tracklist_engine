#!/usr/bin/env python3
"""Reconcile the slow vs fast labeling sessions and emit one audio-verified GT.

The two .als sessions (labeling_slow / labeling_fast) are the same human labeling
at different arrangement tempos, so once each is mapped to mix-seconds they should
agree on (song, mix position, ref offset, pitch). Where they disagree, that is a
labeling inconsistency to surface. This tool:

  * matches clips across the two audits by (song, claimed_stem) + nearest mix start
  * reports coverage gaps (a clip in one session, absent in the other) and
    per-field disagreements (position / pitch / status)
  * emits a merged ground-truth YAML where each span carries the audio-verified
    position when the matched filter found strong evidence of a misplacement, the
    resolved identity, and an `audit:` provenance block (status, scores, source)

    venvs/audio/bin/python -m workspaces.source_detection.als_reconcile \\
        --fast out/1fsnxchk_audit_fast.json --slow out/1fsnxchk_audit_slow.json \\
        --set-id 1fsnxchk --out out/1fsnxchk_ground_truth_verified.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MATCH_POS_TOL_S = 6.0       # cross-session clips this close (same song/stem) are "the same"
DISAGREE_POS_S = 4.0        # matched pair positions differing by more than this == disagreement
STRONG_CORRECTION = 0.62    # only override a placement when audio peak is at least this

# Clips that reference the mix's OWN audio (the mix used as a source layer) are
# self-references, not external sources — exclude them from detection/eval.
_MIX_SELF = {"mix", "mix instrumental", "mix_instrumental", "mix vocals", "mix_vocals"}

# Outsourced host instrumentals: not available online, so a stand-in (the mix's
# own instrumental stem) was used. Keyed by an Ableton group-name substring.
_OUTSOURCED = {
    "lux x spaceman": "Lux Omega — host instrumental not available online; "
                      "outsourced / substituted with the mix_instrumental stem",
}


def _is_mix_self(r: dict) -> bool:
    song = r["song"].strip().lower()
    src = os.path.basename(r.get("src_path") or "").lower()
    return song in _MIX_SELF or src.startswith(("mix.", "mix_instrumental", "mix_vocals"))


def _outsourced_note(r: dict) -> str | None:
    grp = (r.get("group") or "").lower()
    for key, note in _OUTSOURCED.items():
        if key in grp:
            return note
    return None


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _rows(audit: dict) -> list[dict]:
    """Join facts+verdicts by idx into flat, comparable span rows."""
    verd = {v["idx"]: v for v in audit["verdicts"]}
    out = []
    for f in audit["facts"]:
        v = verd.get(f["idx"], {})
        out.append({**f, **{k: v.get(k) for k in (
            "status", "flags", "placed_score", "best_score", "best_pos_s",
            "pos_error_s", "detected_pitch")}})
    return out


def _key(r: dict) -> tuple[str, str]:
    return (r["song"].lower(), r["claimed_stem"])


def match_across(fast: list[dict], slow: list[dict]):
    """Greedy nearest-position match within each (song, stem) bucket."""
    from collections import defaultdict
    slow_by = defaultdict(list)
    for r in slow:
        slow_by[_key(r)].append(r)
    pairs, fast_only = [], []
    used = set()
    for fr in fast:
        cands = [s for s in slow_by[_key(fr)] if id(s) not in used]
        if not cands:
            fast_only.append(fr); continue
        best = min(cands, key=lambda s: abs(s["mix_start_s"] - fr["mix_start_s"]))
        if abs(best["mix_start_s"] - fr["mix_start_s"]) <= MATCH_POS_TOL_S:
            pairs.append((fr, best)); used.add(id(best))
        else:
            fast_only.append(fr)
    slow_only = [s for s in slow if id(s) not in used]
    return pairs, fast_only, slow_only


def best_span(fr: dict | None, sr: dict | None) -> dict:
    """Choose the better-verified of a matched pair (or the only one present)."""
    cands = [r for r in (fr, sr) if r]
    return max(cands, key=lambda r: (r.get("placed_score") or -9))


def gt_row(r: dict) -> dict:
    """One GT row. The human's placement is KEPT (the audio peak is only a
    suggestion — BB reprises the same acapella, so the global peak is often a
    different legitimate usage, not a correction). Mismatches are flagged for
    review with a suggested alternative position, never auto-applied."""
    set_start = r["mix_start_s"]; set_end = r["mix_end_s"]
    status = r.get("status")
    ref_span = max(r["ref_end_s"] - r["ref_start_s"], 1e-3)
    set_span = max(set_end - set_start, 1e-3)
    audible = r.get("audible_frac", 1.0)
    audit = {
        "status": status,
        "id_method": r["id_method"],
        "als_track": r.get("track_name"),     # Ableton lane, e.g. "61-mix_instrumental"
        "als_group": r.get("group") or None,
        "audible_frac": audible,
        "placed_score": r.get("placed_score"),
        "best_score": r.get("best_score"),
        "needs_review": status not in ("OK", "MUTED"),
        "flags": r.get("flags") or [],
    }
    if status == "MUTED" or audible < 0.1:
        audit["ignore"] = True
        audit["ignore_reason"] = (f"volume-muted (audible {audible:.0%}): present in the "
                                  "arrangement but silenced by the track volume slider — "
                                  "not in the mix; exclude from detection/eval")
    if _is_mix_self(r):
        audit["ignore"] = True
        audit["ignore_reason"] = ("mix self-reference (the mix's own audio used as "
                                  "a clip) — not an external source; exclude from detection/eval")
        # the outsourced note belongs ONLY on the substituted host instrumental
        # (the mix-self clip), not on the real vocals layered in the same group.
        note = _outsourced_note(r)
        if note:
            audit["source_note"] = note
    if status == "POSITION_MISMATCH" and (r.get("best_score") or 0) >= STRONG_CORRECTION:
        audit["suggested_position_s"] = r.get("best_pos_s")
    return {
        "track": r["song"],
        "slot_label": r["slot"],
        "claimed_stem": r["claimed_stem"],
        "set_start_s": round(set_start, 3),
        "set_end_s": round(set_end, 3),
        "ref_start_s": round(r["ref_start_s"], 3),
        "ref_end_s": round(r["ref_end_s"], 3),
        "tempo_ratio": round(ref_span / set_span, 5),
        "pitch_shift_semi": r["pitch_coarse"],
        "audit": audit,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", required=True)
    ap.add_argument("--slow", required=True)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--canonical", choices=("fast", "slow"), default="fast",
                    help="which session is the GT spine (default: fast — the session edited by hand)")
    args = ap.parse_args(argv)

    fast = _rows(_load(args.fast))
    slow = _rows(_load(args.slow))
    pairs, fast_only, slow_only = match_across(fast, slow)

    print(f"fast clips: {len(fast)}   slow clips: {len(slow)}")
    print(f"matched across sessions: {len(pairs)}")
    print(f"fast-only (no slow counterpart): {len(fast_only)}")
    print(f"slow-only (no fast counterpart): {len(slow_only)}")

    disagree = []
    for fr, sr in pairs:
        dpos = abs(fr["mix_start_s"] - sr["mix_start_s"])
        dpitch = fr["pitch_coarse"] != sr["pitch_coarse"]
        dstat = fr.get("status") != sr.get("status")
        if dpos > DISAGREE_POS_S or dpitch or dstat:
            disagree.append((dpos, fr, sr, dpitch, dstat))
    disagree.sort(reverse=True)
    print(f"\ncross-session disagreements: {len(disagree)}")
    for dpos, fr, sr, dpitch, dstat in disagree[:25]:
        bits = [f"Δpos={dpos:.1f}s"] if dpos > DISAGREE_POS_S else []
        if dpitch: bits.append(f"pitch {fr['pitch_coarse']}/{sr['pitch_coarse']}")
        if dstat: bits.append(f"status {fr.get('status')}/{sr.get('status')}")
        print(f"  {fr['song'][:40]:40} fast@{fr['mix_start_s']:.0f}s slow@{sr['mix_start_s']:.0f}s  {', '.join(bits)}")

    # Emit GT from ONE canonical session (the hand-edited one) to avoid
    # double-counting loops the two sessions split differently. The other
    # session is only a cross-check (disagreements above).
    canonical = fast if args.canonical == "fast" else slow
    other_name = "slow" if args.canonical == "fast" else "fast"
    rows = [gt_row(r) for r in canonical]
    rows.sort(key=lambda r: r["set_start_s"])

    from collections import Counter
    st = Counter(r["audit"]["status"] for r in rows)
    nrev = sum(1 for r in rows if r["audit"]["needs_review"])
    nign = sum(1 for r in rows if r["audit"].get("ignore"))
    nout = sum(1 for r in rows if r["audit"].get("source_note"))
    print(f"\ncanonical session: {args.canonical} ({len(canonical)} spans); cross-checked vs {other_name}")
    print(f"GT spans: {len(rows)}   status={dict(st)}   needs_review={nrev}   "
          f"ignore(mix-self)={nign}   outsourced-noted={nout}")

    try:
        import yaml
        payload = {"set_id": args.set_id, "source": "als_audit_reconciled",
                   "annotated_by": "user+audio_audit", "tracks": rows}
        Path(args.out).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    except Exception:
        Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))
        print("(pyyaml unavailable — wrote JSON)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Render A/B review clips for the spans the audit flagged, so a human can
adjudicate by ear. Drives off an als_audit JSON.

Per flagged span:
  * SOURCE        — the source segment [ref_start, ref_end] (the acapella/instr)
  * MIX @placed   — the mix stem at the human-labeled position
  * MIX @suggested— the mix stem at the matched-filter peak (POSITION_MISMATCH only)
so you can hear whether the source belongs at the placed spot, the suggested
spot, or neither (a reprise / false peak).

    venvs/audio/bin/python -m workspaces.source_detection.render_flagged \\
        --audit out/1fsnxchk_audit_slow.json --set-id 1fsnxchk
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.source_detection import config  # noqa: E402

SR = 22050
MAX_CLIP_S = 30.0


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:48]


def _excerpt(path: Path, start_s: float, dur_s: float, out: Path) -> bool:
    import librosa
    import soundfile as sf
    dur_s = max(0.5, min(dur_s, MAX_CLIP_S))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=SR, mono=True,
                                offset=max(0.0, start_s), duration=dur_s)
        if y.size < int(0.25 * SR):
            return False
        sf.write(str(out), y, SR)
        return True
    except Exception as e:
        print(f"    ! excerpt failed {path.name}@{start_s:.0f}s: {e}")
        return False


def _mix_stem(set_dir: Path, claimed_stem: str) -> Path | None:
    name = {"acappella": "mix_vocals.flac",
            "instrumental": "mix_instrumental.flac"}.get(claimed_stem, "mix_instrumental.flac")
    f = set_dir / name
    if not f.is_file():
        f = set_dir / "mix_instrumental.flac"
    return f if f.is_file() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--include-pitch", action="store_true", default=True,
                    help="also render spans with a pitch flag (default on)")
    args = ap.parse_args(argv)

    audit = json.loads(Path(args.audit).read_text())
    facts = {f["idx"]: f for f in audit["facts"]}
    verdicts = audit["verdicts"]

    set_dir = next(iter(sorted((Path.home() / "aligning").glob(f"{args.set_id}__*"))), None)
    if set_dir is None:
        sys.exit(f"no ~/aligning folder for {args.set_id}")

    flagged = []
    for v in verdicts:
        pitch_flag = any(str(fl).startswith("pitch") for fl in (v.get("flags") or []))
        if v["status"] != "OK" or (args.include_pitch and pitch_flag):
            flagged.append(v)
    print(f"rendering {len(flagged)} flagged spans from {set_dir.name}\n")

    out_dir = config.OUT_ROOT / "review" / args.set_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for n, v in enumerate(sorted(flagged, key=lambda x: x["mix_start_s"])):
        f = facts[v["idx"]]
        stem = f["claimed_stem"]
        dur = f["mix_end_s"] - f["mix_start_s"]
        base = f"{n:02d}__{_slug(f['song'])}"
        clips = []
        mix_stem = _mix_stem(set_dir, stem)

        # SOURCE segment
        if f.get("src_path") and Path(f["src_path"]).is_file():
            o = out_dir / f"{base}__SOURCE.wav"
            if _excerpt(Path(f["src_path"]), f["ref_start_s"],
                        f["ref_end_s"] - f["ref_start_s"], o):
                clips.append(("source segment", o.name))
        # MIX @ placed
        if mix_stem:
            o = out_dir / f"{base}__MIX_placed_{int(f['mix_start_s'])}s.wav"
            if _excerpt(mix_stem, f["mix_start_s"], dur, o):
                clips.append((f"mix @ PLACED {f['mix_start_s']:.0f}s", o.name))
        # MIX @ suggested (position mismatches)
        sugg = v.get("best_pos_s")
        if v["status"] == "POSITION_MISMATCH" and mix_stem and sugg is not None:
            o = out_dir / f"{base}__MIX_suggested_{int(sugg)}s.wav"
            if _excerpt(mix_stem, sugg, dur, o):
                clips.append((f"mix @ SUGGESTED {sugg:.0f}s", o.name))

        cards.append((n, f, v, clips))
        flagtxt = "; ".join(v.get("flags") or []) or "—"
        print(f"  [{v['status']:17}] {f['mix_start_s']:7.1f}s {f['song'][:40]:40} "
              f"placed={v.get('placed_score')} best={v.get('best_score')}@{v.get('best_pos_s')}s  {flagtxt}")

    # HTML index
    rows = []
    for n, f, v, clips in cards:
        players = "".join(
            f"<div class=p><span>{html.escape(lbl)}</span>"
            f"<audio controls preload=none src='{fn}'></audio></div>" for lbl, fn in clips)
        flags = "; ".join(html.escape(x) for x in (v.get("flags") or [])) or "—"
        rows.append(f"""<div class=card>
  <h3>#{n} · {html.escape(f['song'])} <small>[{f['claimed_stem']}] slot {f['slot']}</small></h3>
  <div class=meta><b>{v['status']}</b> · placed={v.get('placed_score')} best={v.get('best_score')}@{v.get('best_pos_s')}s
     · ref {f['ref_start_s']:.1f}–{f['ref_end_s']:.1f}s · flags: {flags}</div>
  {players}
</div>""")
    page = f"""<!doctype html><meta charset=utf-8><title>flagged review · {args.set_id}</title>
<style>body{{font:14px/1.5 system-ui;margin:2rem;max-width:880px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}}
.meta{{color:#555;font-size:13px;margin-bottom:.5rem}} h3 small{{color:#888;font-weight:400}}
.p{{display:flex;align-items:center;gap:1rem;margin:.25rem 0}} .p span{{width:220px;color:#333}}
audio{{height:32px}}</style>
<h1>Flagged spans — {args.set_id} ({len(cards)})</h1>
<p>Position-mismatches show mix @PLACED vs @SUGGESTED — does the source belong at
either? (A higher peak elsewhere is often a legit reprise, not an error.)</p>
{''.join(rows)}"""
    idx = out_dir / "review.html"
    idx.write_text(page)
    print(f"\nwrote {idx}\nopen: open '{idx}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

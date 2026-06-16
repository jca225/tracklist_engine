#!/usr/bin/env python3
"""Audible fiber inspector — verify the self-repeat equivalence classes by ear.

Fibers ([[project_fibers]]) decide what counts as a correct placement, so we must
trust the equivalence classes. This renders, per reference track, each fiber's
member segments as playable audio grouped together, so you can confirm the
repeats really are the same content — and catch the nuanced failure you flagged:
a singer delivering a section *slightly* differently (extra emphasis, an ad-lib)
that should still be one fiber, or two genuinely-different sections wrongly
merged. Each member shows its cosine similarity to the fiber centroid; members
below a threshold are highlighted as "borderline — listen closely".

Output: out/fiber_review/<set_id>/index.html + per-segment mp3 snippets,
cut from the SAME stem the fibers were computed on (e.g. the vocal stem), so you
hear exactly what the algorithm compared.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.fiber_ui \
        --set-id 1fsnxchk [--stems acappella] [--feature hubert] [--k 6] \
        [--max-refs 12] [--borderline 0.6]
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.path_decode import _ensure_feat  # noqa: E402
from workspaces.alignment_prototype.ref_fibers import (  # noqa: E402
    compute_fibers,
    fiber_intervals,
)
from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP,
    SR,
    _STEM_FILE,
    find_aligning_dir,
)

FPS = SR / HOP
# vocal stems carry phonetic content -> HuBERT; harmonic beds -> chroma is fine
_DEFAULT_FEATURE = {
    "acappella": "hubert",
    "instrumental": "chroma",
    "regular": "chroma",
}


def _pooled(feat: np.ndarray, s: float, e: float) -> np.ndarray:
    a, b = int(s * FPS), int(e * FPS)
    seg = feat[:, a : max(a + 1, b)]
    v = seg.mean(axis=1)
    return v / (np.linalg.norm(v) + 1e-9)


def _cut(src: Path, s: float, e: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{s:.3f}",
        "-t",
        f"{max(0.3, e - s):.3f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "22050",
        str(out),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--stems", default="acappella")
    p.add_argument("--feature", default=None, help="chroma|hubert (default per stem)")
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--min-section-s", type=float, default=4.0)
    p.add_argument("--max-refs", type=int, default=12)
    p.add_argument(
        "--borderline",
        type=float,
        default=0.85,
        help="flag members whose similarity to the fiber centroid is below this "
        "(within-fiber sims cluster ~0.93; the 0.74-0.84 tail is where a singer's "
        "varied emphasis / a wrong merge shows — those are what to listen to)",
    )
    args = p.parse_args(argv)
    want = {s.strip() for s in args.stems.split(",") if s.strip()}

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    out_dir = _REPO / f"workspaces/alignment_prototype/out/fiber_review/{args.set_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # dedup references that have a usable stem for one of the wanted axes
    seen: set[str] = set()
    cards: list[str] = []
    n_refs = 0
    for tr in manifest["tracks"]:
        if n_refs >= args.max_refs:
            break
        rid = str(tr.get("recording_id") or tr.get("track_id"))
        if rid in seen:
            continue
        for stem in want:
            sk = _STEM_FILE.get(stem)
            sp = (tr.get("stems") or {}).get(sk) if sk else tr.get("local_path")
            if not sp or not Path(sp).is_file():
                continue
            feature = args.feature or _DEFAULT_FEATURE.get(stem, "chroma")
            feat = np.load(_ensure_feat(sp, sp, feature, args.hubert_layer))
            labels, hz = compute_fibers(
                feat, FPS, k=args.k, min_section_s=args.min_section_s
            )
            ivs = fiber_intervals(labels, hz, min_len_s=args.min_section_s)
            by_lab: dict[int, list] = {}
            for s, e, lab in ivs:
                by_lab.setdefault(lab, []).append((s, e))
            multi = {k: v for k, v in by_lab.items() if len(v) >= 2}
            if not multi:
                continue
            seen.add(rid)
            n_refs += 1
            name = html.escape(tr.get("title") or tr.get("name") or rid)
            blocks = [f"<h2>{name} <small>({stem}, {feature})</small></h2>"]
            for lab, members in sorted(multi.items(), key=lambda kv: -len(kv[1])):
                centroid = np.mean([_pooled(feat, s, e) for s, e in members], axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-9
                blocks.append(
                    f'<div class="fiber"><b>fiber {lab}</b> — {len(members)} members'
                )
                for j, (s, e) in enumerate(members):
                    sim = float(_pooled(feat, s, e) @ centroid)
                    snip = out_dir / f"{rid}_L{lab}_{j}.mp3"
                    ok = _cut(Path(sp), s, e, snip)
                    cls = "bad" if sim < args.borderline else "ok"
                    audio = (
                        f'<audio controls preload="none" src="{snip.name}"></audio>'
                        if ok
                        else "<i>cut failed</i>"
                    )
                    blocks.append(
                        f'<div class="m {cls}">{s:6.1f}–{e:5.1f}s '
                        f'<span class="sim">sim {sim:.2f}</span> {audio}</div>'
                    )
                blocks.append("</div>")
            cards.append("\n".join(blocks))
            break

    style = (
        "body{font-family:system-ui;margin:2rem;max-width:60rem}"
        ".fiber{border:1px solid #ccc;border-radius:8px;padding:.6rem;margin:.6rem 0}"
        ".m{display:flex;align-items:center;gap:.6rem;padding:.2rem 0}"
        ".m.bad{background:#ffecec}.sim{font-variant:tabular-nums;color:#666}"
        ".m.bad .sim{color:#c00;font-weight:600}small{color:#888}"
        "audio{height:1.8rem}"
    )
    doc = (
        f"<!doctype html><meta charset=utf-8><title>fibers {args.set_id}</title>"
        f"<style>{style}</style>"
        f"<h1>Fiber review — {args.set_id} ({args.stems})</h1>"
        f"<p>Each fiber groups sections the algorithm calls the same content. "
        f"Play them in sequence: they should sound like the same part. "
        f"<b>Pink rows</b> (sim &lt; {args.borderline}) are borderline — a member "
        f"that may differ (e.g. the singer's emphasis) or be wrongly merged.</p>"
        + ("\n".join(cards) or "<p>no multi-member fibers found</p>")
    )
    index = out_dir / "index.html"
    index.write_text(doc)
    print(f"wrote {index}  ({n_refs} refs with repeated fibers)")
    print(f"open: file://{index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

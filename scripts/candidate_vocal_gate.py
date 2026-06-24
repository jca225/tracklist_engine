#!/usr/bin/env python3
"""Auto-pick the correct acappella candidate per payload slot.

For every blocking payload slot (the gate flagged ``wrong_stem`` — a vocal
overlay backed only by a full regular track), we already fetched ~3 YouTube
"acappella" candidates into ``stems/<slot>/candidates/vocals/``. Many are wrong
(cover bands, college a-cappella groups, reverbed rips). This verifies each
candidate against the *studio recording's own separated vocals* — the true
isolated vocal of that exact master matches near-exactly; a cover does not —
using the HuBERT-L9 matched-filter from the vocal-verification work
(``similarity_probe``). The winner (high score + clear margin) gets a
``WINNER.txt`` that ``ingest_candidate_winners.py`` then ingests; low-margin /
all-bad slots (e.g. Mumford "The Cave", whose candidates are all choir covers)
ABSTAIN for human review.

Usage (report only, default):
    venvs/audio/bin/python scripts/candidate_vocal_gate.py \
        --set-dir "~/aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11"
Add ``--write`` to emit WINNER.txt for confident picks, ``--labels 030w1,034w1``
to restrict to specific slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP,
    SR,
    STRETCHES,
    detect_offset,
)
from workspaces.section_hsmm.similarity_probe import _feat  # noqa: E402

FPS = SR / HOP
_AUDIO_EXT = {".m4a", ".mp3", ".flac", ".wav", ".opus", ".ogg"}


@dataclass
class CandScore:
    file: Path
    candidates_dir: Path  # the stems/<slot>/candidates dir holding it
    peak: float = 0.0


@dataclass
class SlotDecision:
    label: str
    track_id: str
    ref_vocal: Path
    scores: list[CandScore] = field(default_factory=list)
    status: str = "abstain"  # winner | abstain | skip
    reason: str = ""

    @property
    def best(self) -> CandScore | None:
        return self.scores[0] if self.scores else None

    @property
    def margin(self) -> float:
        return (
            self.scores[0].peak - self.scores[1].peak
            if len(self.scores) >= 2
            else self.scores[0].peak
            if self.scores
            else 0.0
        )


def _cache_key(path: Path) -> str:
    h = hashlib.sha1(str(path).encode()).hexdigest()[:12]
    return f"cand_{h}_voc"


def _windows(feat: np.ndarray, win_s: float, k: int) -> list[np.ndarray]:
    """k evenly-spaced query windows of win_s seconds from a feature matrix."""
    n = feat.shape[1]
    w = int(win_s * FPS)
    if n <= w:
        return [feat]
    starts = np.linspace(0, n - w, k).astype(int)
    return [feat[:, s : s + w] for s in starts]


def _score_candidate(
    cand_feat: np.ndarray, ref_feat: np.ndarray, win_s: float, k: int
) -> float:
    """Best matched-filter peak of any candidate window against the ref vocal.

    The shorter feature is always slid over the longer one (detect_offset needs
    ref longer than the query), so this is symmetric to which clip is longer."""
    short, long_ = (
        (cand_feat, ref_feat)
        if cand_feat.shape[1] <= ref_feat.shape[1]
        else (ref_feat, cand_feat)
    )
    best = 0.0
    for win in _windows(short, win_s, k):
        if win.shape[1] < 8 or long_.shape[1] <= win.shape[1]:
            continue
        _, peak, _ = detect_offset(win, long_, STRETCHES)
        best = max(best, peak)
    return best


def _slot_dirs(stems: Path, label: str) -> list[Path]:
    """All stem dirs for a slot label (tagged + untagged dup folders)."""
    pat = re.compile(rf"^{re.escape(label)}__")
    return [d for d in stems.iterdir() if d.is_dir() and pat.match(d.name)]


def _candidate_files(stems: Path, label: str) -> list[Path]:
    """External acappella candidates only.

    Excludes ``separated__*`` — that is the studio track's OWN Demucs vocals
    (added by add_separated_to_candidates), which trivially self-matches the
    reference at peak 1.0. It is the Demucs-fallback, not an external candidate.
    """
    out: list[Path] = []
    for d in _slot_dirs(stems, label):
        vd = d / "candidates" / "vocals"
        if vd.is_dir():
            out += [
                f
                for f in sorted(vd.glob("*"))
                if f.suffix.lower() in _AUDIO_EXT and not f.name.startswith("separated")
            ]
    return out


def evaluate(
    set_dir: Path,
    *,
    labels: set[str] | None,
    win_s: float,
    k: int,
    layer: int,
) -> list[SlotDecision]:
    man = json.loads((set_dir / "manifest.json").read_text())
    stems = set_dir / "stems"
    block = [
        t
        for t in man["tracks"]
        if t.get("layer_role") == "payload" and t.get("satisfaction") == "wrong_stem"
    ]
    decisions: list[SlotDecision] = []
    for t in block:
        label = t["label"]
        if labels and label not in labels:
            continue
        ref_vocal = (t.get("stems") or {}).get("vocals")
        tid = t.get("track_id") or ""
        if not ref_vocal or not Path(ref_vocal).is_file():
            decisions.append(
                SlotDecision(
                    label,
                    tid,
                    Path(ref_vocal or ""),
                    status="skip",
                    reason="no studio ref vocal stem",
                )
            )
            continue
        cfiles = _candidate_files(stems, label)
        if not cfiles:
            decisions.append(
                SlotDecision(
                    label,
                    tid,
                    Path(ref_vocal),
                    status="skip",
                    reason="no candidate vocals on disk",
                )
            )
            continue
        ref_feat = _feat(Path(ref_vocal), f"ref_{tid}_voc", "hubert", layer)
        d = SlotDecision(label, tid, Path(ref_vocal))
        for cf in cfiles:
            cfeat = _feat(cf, _cache_key(cf), "hubert", layer)
            peak = _score_candidate(cfeat, ref_feat, win_s, k)
            d.scores.append(CandScore(cf, cf.parent.parent, peak=round(float(peak), 4)))
        d.scores.sort(key=lambda c: c.peak, reverse=True)
        decisions.append(d)
        print(
            f"  scored {label}: "
            + ", ".join(f"{c.file.name.split('__')[0]}={c.peak:.3f}" for c in d.scores),
            file=sys.stderr,
        )
    return decisions


def decide(d: SlotDecision, floor: float, margin: float) -> None:
    if d.status == "skip" or not d.scores:
        return
    if d.best.peak < floor:
        d.status, d.reason = "abstain", f"best {d.best.peak:.3f} < floor {floor}"
    elif d.margin < margin:
        d.status, d.reason = "abstain", f"margin {d.margin:.3f} < {margin}"
    else:
        d.status, d.reason = "winner", f"peak {d.best.peak:.3f} margin {d.margin:.3f}"


def write_winner(d: SlotDecision) -> Path:
    """WINNER.txt = file (relative to candidates dir), track_id, role."""
    cand_dir = d.best.candidates_dir
    rel = d.best.file.relative_to(cand_dir)
    wf = cand_dir / "WINNER.txt"
    wf.write_text(f"{rel}\n{d.track_id}\nacappella\n")
    return wf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-dir", type=Path, required=True)
    ap.add_argument("--labels", help="comma-separated slot labels to restrict to")
    ap.add_argument("--win-s", type=float, default=20.0)
    ap.add_argument(
        "--windows", type=int, default=3, help="query windows per candidate"
    )
    ap.add_argument("--hubert-layer", type=int, default=9)
    ap.add_argument(
        "--floor",
        type=float,
        default=0.6,
        help="min winning peak (real acappellas ~0.8+, covers ~0.4)",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="min best-2nd margin (0 = absolute floor decides; two real "
        "acappellas of one song both score high and either is fine)",
    )
    ap.add_argument("--write", action="store_true", help="emit WINNER.txt for winners")
    ap.add_argument("--json", type=Path, help="write full decision report as JSON")
    args = ap.parse_args(argv)

    set_dir = args.set_dir.expanduser().resolve()
    labels = {s.strip() for s in args.labels.split(",")} if args.labels else None
    decisions = evaluate(
        set_dir,
        labels=labels,
        win_s=args.win_s,
        k=args.windows,
        layer=args.hubert_layer,
    )
    for d in decisions:
        decide(d, args.floor, args.margin)

    winners = [d for d in decisions if d.status == "winner"]
    abstain = [d for d in decisions if d.status == "abstain"]
    skip = [d for d in decisions if d.status == "skip"]

    print("\n=== candidate vocal gate ===")
    print(f"floor={args.floor} margin={args.margin} layer={args.hubert_layer}")
    for d in sorted(decisions, key=lambda x: x.label):
        tag = {"winner": "WIN ", "abstain": "ABST", "skip": "SKIP"}[d.status]
        b = (
            f"{d.best.file.name}  peak={d.best.peak:.3f} margin={d.margin:+.3f}"
            if d.best
            else ""
        )
        print(f"  [{tag}] {d.label:7s} {b}")
        if d.status != "winner" and d.reason:
            print(f"           ↳ {d.reason}")
    print(f"\nwinners: {len(winners)}  abstain: {len(abstain)}  skip: {len(skip)}")

    if args.write:
        for d in winners:
            wf = write_winner(d)
            print(f"  wrote {wf}")
        print(f"wrote {len(winners)} WINNER.txt — run ingest_candidate_winners.py next")
    else:
        print("(dry-run: no WINNER.txt written; pass --write to emit)")

    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "label": d.label,
                        "track_id": d.track_id,
                        "status": d.status,
                        "reason": d.reason,
                        "scores": [
                            {"file": c.file.name, "peak": c.peak} for c in d.scores
                        ],
                    }
                    for d in decisions
                ],
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

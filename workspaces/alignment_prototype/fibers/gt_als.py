#!/usr/bin/env python3
"""Fiber ground truth via Ableton — render machine fibers as an .als overlay,
parse the hand-corrected session back into pairwise fiber GT.

The web review UI (`review/fiber_server.py`) can only audit fibers that FIRED
— it is structurally blind to misses, which is the known failure direction
(SALAMI: P 0.88 / R 0.065, under-merge). This tool closes that hole by putting
the *whole ref track* plus every detector arm's proposals in front of John in
his native tool:

  render:  one .als per ref audio file —
    * "1-mix (ref ruler)" — the full audio, unwarped, at a flat 60 BPM
      (1 beat == 1 second: arrangement position IS ref time)
    * "F<k> hubert …"     — one track per production fiber
      (`compute_fibers_soft`), clips at instance intervals, μ in the name
    * "CAND fp L<k>"      — `compute_fibers_fp` proposals (melodic repeats
      HuBERT is blind to)
    * "CAND harm L<k>"    — `harmony_fibers` proposals (CENS lag stripes)

  correction protocol (in Live): a TRACK is an equivalence class.
    * wrong instance on a fiber track     -> delete the clip
    * candidate confirms an F-class       -> drag it onto that track
    * candidates that repeat each other   -> new track, any name not
                                             starting with "CAND"
    * missed repeat no arm proposed       -> duplicate a region of the ruler
                                             onto a class track (position = the
                                             ref seconds; identity warp)
    * candidate is NOT a repeat           -> move it to a track named "REJ"
                                             (explicit negative)
    * leftovers on CAND tracks            -> unreviewed (NOT rejections)

  import:  parse the corrected .als -> `<name>.fiber_gt.yaml` (classes =
    interval lists) + pairwise precision/recall of each machine arm against
    the determined GT (0.5 s frame-pair algebra, same definition as
    external/fiber_validation.py).

Provenance: output is stamped "<name> FIBERS SEEDED.als" and locator #0 says
machine-predicted — same rules as review/seed_als_from_timeline (never
"<SET> align.als").

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.fibers.gt_als \
        --audio "<ref audio>" [--label "Emily vocals"] [--arms hubert,fp,harmony]
    venvs/audio/bin/python -m workspaces.alignment_prototype.fibers.gt_als \
        --import-als "~/aligning/_fibers/Emily vocals FIBERS SEEDED.als"
"""

from __future__ import annotations

import argparse
import copy
import gzip
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from lxml import etree

from labeling.als import load_als_xml, parse_layer_clips
from workspaces.alignment_prototype.fibers.detect import (
    compute_fibers_fp,
    compute_fibers_soft,
    fiber_intervals,
)
from workspaces.alignment_prototype.fibers.harmony import harmony_fibers
from workspaces.alignment_prototype.refine_ref_offsets import HOP, SR

FPS = SR / HOP
OUT_ROOT = Path.home() / "aligning" / "_fibers"
_HUBERT_LAYER = 9
# Live 11 palette (same indices the timeline seeder uses)
_COLOR_GREEN, _COLOR_YELLOW, _COLOR_RED, _COLOR_BLUE = 5, 2, 14, 9
_PAIR_GRID_HZ = 2.0  # 0.5 s cells for pairwise P/R (matches fiber_validation)


@dataclass(frozen=True)
class ArmClass:
    """One equivalence class proposed by one detector arm."""

    arm: str  # "hubert" | "fp" | "harmony"
    label: int
    instances: tuple[tuple[float, float], ...]
    mu: tuple[float, ...] = ()  # per-instance membership (hubert arm only)


def _runs(
    labels: np.ndarray, hz: float, min_len_s: float = 3.0
) -> dict[int, list[tuple[float, float]]]:
    by: dict[int, list[tuple[float, float]]] = {}
    for s, e, lab in fiber_intervals(labels, hz, min_len_s=min_len_s):
        by.setdefault(lab, []).append((s, e))
    # a class needs >= 2 instances to say anything about repeats
    return {k: v for k, v in by.items() if len(v) >= 2}


def _hubert_feat(audio: Path) -> np.ndarray:
    from workspaces.alignment_prototype.path_decode import _ensure_feat

    return np.load(
        _ensure_feat(audio, audio.stem, "hubert", _HUBERT_LAYER), mmap_mode="r"
    )


def detect_arms(audio: Path, arms: list[str]) -> list[ArmClass]:
    out: list[ArmClass] = []
    if "hubert" in arms:
        labels, hz, mu, _conf = compute_fibers_soft(
            np.asarray(_hubert_feat(audio)), FPS, audio_path=str(audio)
        )
        for lab, ivs in sorted(_runs(labels, hz).items()):
            mus = tuple(
                float(np.mean(mu[int(s * hz) : max(int(s * hz) + 1, int(e * hz))]))
                for s, e in ivs
            )
            out.append(ArmClass("hubert", lab, tuple(ivs), mus))
    if "fp" in arms:
        labels, hz = compute_fibers_fp(str(audio))
        for lab, ivs in sorted(_runs(labels, hz).items()):
            out.append(ArmClass("fp", lab, tuple(ivs)))
    if "harmony" in arms:
        labels, hz = harmony_fibers(str(audio))
        for lab, ivs in sorted(_runs(labels, hz).items()):
            out.append(ArmClass("harmony", lab, tuple(ivs)))
    return out


# --- render ----------------------------------------------------------------
def render(
    audio: Path, label: str, arms: list[str], template: Path, out: Path | None
) -> Path:
    from workspaces.alignment_prototype.review.seed_als_from_timeline import (
        _TEMPO_BPM,
        DEFAULT_TEMPLATE,
        build_track,
        doc_max_id,
        ffprobe_audio,
        find_template_track,
        renumber_pointee_ids,
    )

    assert template or DEFAULT_TEMPLATE
    dur, sr = ffprobe_audio(audio)
    classes = detect_arms(audio, arms)
    n_clips = sum(len(c.instances) for c in classes) + 1

    root = load_als_xml(template)
    template_track = copy.deepcopy(find_template_track(root))
    tracks_node = root.find(".//LiveSet/Tracks")
    for t in list(tracks_node):
        if t.tag in ("AudioTrack", "GroupTrack", "MidiTrack"):
            tracks_node.remove(t)

    try:
        from labeling.als import write_tempo_envelope

        write_tempo_envelope(root, [(0.0, _TEMPO_BPM)])
    except Exception:
        for tempo_manual in root.xpath(".//MasterTrack//Tempo/Manual"):
            tempo_manual.set("Value", f"{_TEMPO_BPM:.1f}")

    alloc = itertools.count(doc_max_id(root) + n_clips * 2 + 2000)
    next_id = 1000

    ruler = build_track(
        template_track,
        track_id=next_id,
        track_name="1-mix (ref ruler)",  # 1-mix prefix => parse_layer_clips skips it
        name=label,
        color=_COLOR_BLUE,
        arr_start=0.0,
        arr_end=dur,
        file_path=audio,
        file_dur_s=dur,
        sample_rate=sr,
        ref_start_s=0.0,
        ref_end_s=dur,
        is_warped=False,
    )
    renumber_pointee_ids(ruler, alloc)
    tracks_node.insert(0, ruler)

    color = {"hubert": _COLOR_GREEN, "fp": _COLOR_YELLOW, "harmony": _COLOR_RED}
    placed: list[tuple[str, float, float]] = []  # (track_name, ref_start, ref_end)
    insert_at = 1
    for c in classes:
        tname = (
            f"F{c.label} hubert (n={len(c.instances)})"
            if c.arm == "hubert"
            else f"CAND {c.arm} L{c.label} (n={len(c.instances)})"
        )
        for j, (s, e) in enumerate(c.instances):
            mu_tag = f" mu={c.mu[j]:.2f}" if c.mu else ""
            next_id += 1
            # one track per clip keeps the proven deep-copy + renumber path
            # (clip clones within a track collide ids -> Live crash); Live
            # shows same-named tracks as one lane group visually.
            tr = build_track(
                template_track,
                track_id=next_id,
                track_name=tname,
                name=f"{tname.split(' (')[0]}.{j + 1}{mu_tag} [{s:.1f}-{e:.1f}s]",
                color=color[c.arm],
                arr_start=s,
                arr_end=e,
                file_path=audio,
                file_dur_s=dur,
                sample_rate=sr,
                ref_start_s=s,
                ref_end_s=e,
            )
            renumber_pointee_ids(tr, alloc)
            tracks_node.insert(insert_at, tr)
            insert_at += 1
            placed.append((tname, s, e))

    try:
        from labeling.als import write_locators

        write_locators(
            root,
            [(0.0, f"FIBER SEED {date.today().isoformat()} — machine fibers, NOT GT")],
        )
    except Exception as exc:
        print(f"  locator skipped: {exc}", file=sys.stderr)

    for npi in root.findall(".//NextPointeeId"):
        npi.set("Value", str(next(alloc)))

    out_path = out or (OUT_ROOT / f"{label} FIBERS SEEDED.als")
    if out_path.name.endswith(" align.als"):
        sys.exit("refusing '<name> align.als' — that name is hand-GT territory")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        bak = out_path.with_suffix(".als.seedbak")
        out_path.rename(bak)
        print(f"backed up existing -> {bak.name}")
    out_path.write_bytes(
        gzip.compress(
            etree.tostring(
                root, xml_declaration=True, encoding="UTF-8", standalone=True
            )
        )
    )

    # sidecar: the machine proposals, for import-time P/R against determined GT
    sidecar = {
        "audio": str(audio),
        "duration_s": dur,
        "arms": arms,
        "classes": [
            {
                "arm": c.arm,
                "label": c.label,
                "instances": [list(i) for i in c.instances],
            }
            for c in classes
        ],
    }
    out_path.with_suffix(".fibers.json").write_text(json.dumps(sidecar, indent=1))

    # round-trip self-validation through the real GT parser
    clips = parse_layer_clips(load_als_xml(out_path))
    remaining = list(placed)
    for clip in clips:
        rs, re_ = clip.ref_start_s(), clip.ref_end_s()
        hit = next(
            (
                k
                for k, (tn, s, e) in enumerate(remaining)
                if tn == clip.track_name
                and abs(rs - s) <= 0.05
                and abs(re_ - e) <= 0.05
            ),
            None,
        )
        if hit is None:
            sys.exit(f"VALIDATION FAIL: clip {clip.track_name} ref={rs:.2f} unmatched")
        remaining.pop(hit)
    if remaining:
        sys.exit(f"VALIDATION FAIL: {len(remaining)} placed clips not parsed back")
    n_f = sum(1 for c in classes if c.arm == "hubert")
    print(
        f"wrote {out_path}\n  {n_f} hubert fibers, "
        f"{sum(1 for c in classes if c.arm == 'fp')} fp cands, "
        f"{sum(1 for c in classes if c.arm == 'harmony')} harmony cands, "
        f"{len(placed)} clips — round-trip OK"
    )
    return out_path


# --- import ------------------------------------------------------------------
def _pair_mask(classes: list[list[tuple[float, float]]], dur: float) -> np.ndarray:
    """Boolean same-class matrix over the 0.5 s grid (upper triangle used)."""
    n = int(dur * _PAIR_GRID_HZ)
    lab = np.full(n, -1, dtype=np.int32)
    for k, ivs in enumerate(classes):
        for s, e in ivs:
            lab[int(s * _PAIR_GRID_HZ) : min(n, int(e * _PAIR_GRID_HZ))] = k
    m = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
    m[np.tril_indices(n)] = False
    return m


def import_als(als_path: Path, gt_out: Path | None) -> int:
    sidecar_p = als_path.with_suffix(".fibers.json")
    sidecar = json.loads(sidecar_p.read_text()) if sidecar_p.is_file() else None
    clips = parse_layer_clips(load_als_xml(als_path))
    if not clips:
        sys.exit("no layer clips parsed — is this the corrected FIBERS session?")

    classes: dict[str, list[tuple[float, float]]] = {}
    rejected: list[tuple[float, float]] = []
    unreviewed: list[tuple[float, float]] = []
    for c in clips:
        iv = (round(c.ref_start_s(), 3), round(c.ref_end_s(), 3))
        # Rejection must be an ACT, not a default: only clips the human moved
        # onto a track named REJ* count as negatives. CAND clips left in place
        # are UNREVIEWED — treating them as rejections poisoned the first GT
        # import (fp/harmony proposals of the SAME repeat family the human had
        # confirmed on the green track read back as false-merge negatives).
        if c.track_name.startswith("REJ"):
            rejected.append(iv)
        elif c.track_name.startswith("CAND"):
            unreviewed.append(iv)
        else:
            classes.setdefault(c.track_name, []).append(iv)
    classes = {k: sorted(v) for k, v in classes.items() if len(v) >= 2}

    audio = sidecar["audio"] if sidecar else (clips[0].path if clips else "")
    dur = (
        sidecar["duration_s"]
        if sidecar
        else max(e for c in classes.values() for _, e in c)
    )
    gt = {
        "audio": audio,
        "duration_s": dur,
        "determined": date.today().isoformat(),
        "source_als": str(als_path),
        "classes": [
            {"name": k, "instances": [list(i) for i in v]}
            for k, v in sorted(classes.items())
        ],
        "rejected": [list(i) for i in sorted(rejected)],
        "unreviewed": [list(i) for i in sorted(unreviewed)],
    }
    out = gt_out or als_path.with_suffix(".fiber_gt.yaml")
    import yaml

    out.write_text(yaml.safe_dump(gt, sort_keys=False, allow_unicode=True))
    n_inst = sum(len(v) for v in classes.values())
    print(
        f"wrote {out}: {len(classes)} classes, {n_inst} instances, "
        f"{len(rejected)} rejected, {len(unreviewed)} unreviewed"
    )

    if sidecar:
        gt_mask = _pair_mask(list(classes.values()), dur)
        n_gt = int(gt_mask.sum())
        print(f"\nmachine arms vs determined GT ({n_gt} true pairs @0.5s grid):")
        for arm in sidecar["arms"]:
            ivs = [
                [tuple(i) for i in c["instances"]]
                for c in sidecar["classes"]
                if c["arm"] == arm
            ]
            m = _pair_mask(ivs, dur)
            tp = int((m & gt_mask).sum())
            p = tp / max(1, int(m.sum()))
            r = tp / max(1, n_gt)
            print(f"  {arm:8s} P={p:.3f} R={r:.3f} (proposed pairs={int(m.sum())})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, help="ref audio file (stem or full)")
    p.add_argument("--label", default=None, help="display name (default: audio stem)")
    p.add_argument("--arms", default="hubert,fp,harmony")
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--import-als", type=Path, default=None, dest="import_als")
    p.add_argument("--gt-out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.import_als:
        return import_als(args.import_als.expanduser(), args.gt_out)
    if not args.audio:
        p.error("--audio required for render mode")
    from workspaces.alignment_prototype.review.seed_als_from_timeline import (
        DEFAULT_TEMPLATE,
    )

    audio = args.audio.expanduser()
    if not audio.is_file():
        sys.exit(f"no such audio: {audio}")
    render(
        audio,
        args.label or audio.stem,
        [a.strip() for a in args.arms.split(",") if a.strip()],
        args.template or DEFAULT_TEMPLATE,
        args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

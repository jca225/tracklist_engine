#!/usr/bin/env python3
"""Cross-set inference: train the head on a labeled set, predict an unlabeled one.

The aligner's only learned component is the MertAlignHead ensemble; mix/refs/
pools/priors are bound data. Inference rebinds a BB12-trained head to the
target set's data:

  * slot stubs + candidate pools  <- pi set_track_slots (tracklist claims)
  * placement anchors             <- scraped cue_seconds (149/152 on BB11)
  * span-duration priors          <- consecutive cue diffs (clamped)
  * mix / ref MERT                <- same export path as training sets
  * fine placement                <- per-span DTW vs the set's roformer
                                     mix_instrumental (aligning folder)

No ground truth is read for the target set — this is the transfer test.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.infer \\
        --set-id 2nvzlh2k [--refresh-mert] [--band-s 45]

Output: out/<set_id>_predicted_timeline.json + a printed table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from core.result import Err, Ok, Result
from workspaces.alignment_prototype.dataset import (
    load_set,
    slot_candidates_from_targets,
)
from workspaces.alignment_prototype.records import SlotCandidate, SpanTarget
from workspaces.alignment_prototype.slot_priors import normalize_slot

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"
DEFAULT_TRAIN_YAML = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"
OUT_DIR = Path(__file__).resolve().parent / "out"

_DUR_MIN_S = 15.0
_DUR_MAX_S = 180.0
_DUR_FALLBACK_S = 45.0


def _ssh_sql(sql: str) -> str:
    r = subprocess.run(
        ["ssh", PI_HOST, f'sqlite3 -separator "|" {PI_DB} "{sql}"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def fetch_slot_rows(set_id: str) -> tuple[dict, ...]:
    """Tracklist spine for the target set, in play order."""
    sql = (
        "SELECT slot_label, COALESCE(recording_id, track_id), "
        "COALESCE(claimed_stem,'regular'), COALESCE(cue_seconds, cue_time_seconds, ''), "
        "COALESCE(full_name, title, '') "
        f"FROM set_track_slots WHERE set_id='{set_id}' ORDER BY row_index"
    )
    rows: list[dict] = []
    for ln in _ssh_sql(sql).splitlines():
        parts = ln.split("|")
        if len(parts) < 5:
            continue
        label, rid, stem, cue, name = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            "|".join(parts[4:]),
        )
        rows.append(
            {
                "slot_label": normalize_slot(label),
                "recording_id": rid or None,
                "claimed_stem": stem,
                "cue_s": float(cue) if cue else None,
                "name": name,
            }
        )
    return tuple(rows)


def build_stub_targets(
    rows: tuple[dict, ...],
    mix_end_s: float,
) -> tuple[tuple[SpanTarget, ...], dict[str, float], dict[str, float]]:
    """SpanTarget stubs + cue anchors + cue-diff duration priors.

    Durations: distance to the next *distinct* cue (concurrent `w` rows share
    the parent cue), clamped to [15, 180] s; fallback 45 s where cues are
    missing or non-increasing.
    """
    cues = [r["cue_s"] for r in rows]
    n = len(rows)
    durs: list[float] = []
    for i, c in enumerate(cues):
        if c is None:
            durs.append(_DUR_FALLBACK_S)
            continue
        nxt = next(
            (cues[j] for j in range(i + 1, n) if cues[j] is not None and cues[j] > c),
            None,
        )
        end = nxt if nxt is not None else mix_end_s
        durs.append(float(np.clip(end - c, _DUR_MIN_S, _DUR_MAX_S)))

    targets: list[SpanTarget] = []
    anchors: dict[str, float] = {}
    slot_durs: dict[str, list[float]] = {}
    for r, dur in zip(rows, durs):
        start = r["cue_s"] if r["cue_s"] is not None else 0.0
        targets.append(
            SpanTarget(
                slot_label=r["slot_label"],
                recording_id=r["recording_id"],
                claimed_stem=r["claimed_stem"],
                set_start_s=start,
                set_end_s=start + dur,
                ref_start_s=0.0,
                ref_end_s=None,
                tempo_ratio=None,
                pitch_shift_semi=0,
                label=r["name"],
            )
        )
        if r["cue_s"] is not None:
            anchors[r["slot_label"]] = float(r["cue_s"])
        slot_durs.setdefault(r["slot_label"].split("w", 1)[0], []).append(dur)

    medians = {k: float(np.median(v)) for k, v in slot_durs.items()}
    return tuple(targets), anchors, medians


def slot_pools_from_rows(
    rows: tuple[dict, ...],
) -> dict[str, tuple[SlotCandidate, ...]]:
    pools: dict[str, list[SlotCandidate]] = {}
    for r in rows:
        if not r["recording_id"]:
            continue
        c = SlotCandidate(
            recording_id=r["recording_id"], claimed_stem=r["claimed_stem"]
        )
        pools.setdefault(r["slot_label"], [])
        if c not in pools[r["slot_label"]]:
            pools[r["slot_label"]].append(c)
    return {k: tuple(v) for k, v in pools.items()}


def _torch_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True, help="target (unlabeled) set_id")
    p.add_argument("--train-yaml", type=Path, default=DEFAULT_TRAIN_YAML)
    p.add_argument("--refresh-mert", action="store_true")
    p.add_argument(
        "--band-s",
        type=float,
        default=45.0,
        help="fine-placement DTW corridor half-width (0 disables)",
    )
    p.add_argument(
        "--fp-refine",
        action="store_true",
        help="per-span fingerprint argmax after coarse decode (needs aligning audio + fp cache)",
    )
    p.add_argument("--fp-band-s", type=float, default=45.0)
    p.add_argument(
        "--fp-gate-z",
        type=float,
        default=1.0,
        help="min sharpness z-score to override coarse start",
    )
    p.add_argument(
        "--fp-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="source placement from the landmark-fp vote-extent (decode_placements) "
        "instead of the MERT monotonic DP; identity still comes from predict_sequence. "
        "--no-fp-placement falls back to the old MERT placement + DTW/fp-refine.",
    )
    p.add_argument("--fp-placement-topk", type=int, default=6)
    p.add_argument("--fp-placement-gap-s", type=float, default=6.0)
    p.add_argument(
        "--fp-placement-gate-s",
        type=float,
        default=90.0,
        help="keep MERT placement when |fp-mert| exceeds this (re-leash the fp "
        "outlier tail to the anchored prior; <=0 disables, BB12-tuned 90s)",
    )
    p.add_argument(
        "--fp-placement-compare",
        action="store_true",
        help="also record the MERT set_start per span (mert_set_start_s) for A/B",
    )
    p.add_argument(
        "--stem-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="refine acappella set_start with a banded HuBERT matched filter over "
        "mix_vocals (the fp is weak on vocals). Needs mix_vocals.flac + ref vocals "
        "stems in the manifest. set_start only — ref_start stays with refine_ref_offsets.",
    )
    p.add_argument("--stem-placement-band-s", type=float, default=90.0)
    p.add_argument("--stem-placement-guard-s", type=float, default=8.0)
    args = p.parse_args(argv)

    from workspaces.alignment_prototype.mert_model import (
        MertLearnedAligner,
        TrainConfig,
        train_ensemble,
    )
    from workspaces.alignment_prototype.mert_features import build_examples
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    # ---- 1. train head on the labeled set (all spans — no held-out) --------
    match load_set(args.train_yaml):
        case Err(msg):
            print(f"train GT load failed: {msg}", file=sys.stderr)
            return 1
        case Ok((train_gt, train_targets)):
            pass
    print(f"train set={train_gt.set_id} spans={len(train_targets)}")

    match load_bb12_mert(train_gt.set_id):
        case Err(msg):
            print(f"train MERT load failed: {msg}", file=sys.stderr)
            return 1
        case Ok((_sid, train_mix, train_refs)):
            print(f"  train mix measures={train_mix.n_measures} refs={len(train_refs)}")

    device = _torch_device()
    cfg = TrainConfig(epochs=40, search_margin_s=90.0)
    train_pools = slot_candidates_from_targets(train_targets)
    examples = build_examples(
        train_targets,
        train_mix,
        train_refs,
        train_pools,
        search_margin_s=cfg.search_margin_s,
    )
    print(f"training head ensemble on {len(examples)} examples (device={device})…")
    head = train_ensemble(examples, cfg=cfg, device=device)

    # ---- 2. bind target set data -------------------------------------------
    match load_bb12_mert(args.set_id, refresh=args.refresh_mert):
        case Err(msg):
            print(f"target MERT load failed: {msg}", file=sys.stderr)
            return 1
        case Ok((_sid2, mix, refs)):
            print(
                f"target set={args.set_id} mix measures={mix.n_measures} refs={len(refs)}"
            )

    rows = fetch_slot_rows(args.set_id)
    mix_end = float(mix.end_s[-1])
    targets, anchors, slot_medians = build_stub_targets(rows, mix_end)
    pools = slot_pools_from_rows(rows)

    have_ref = [t for t in targets if t.recording_id and t.recording_id in refs]
    skipped = [t for t in targets if t not in have_ref]
    print(
        f"slots={len(targets)} decodable={len(have_ref)} "
        f"skipped={len(skipped)} (no recording/MERT) cue_anchors={len(anchors)}"
    )
    for t in skipped:
        print(f"  SKIP {t.slot_label:6} {t.label[:50]}")
    decodable = tuple(have_ref)

    aligner = MertLearnedAligner(
        head=head,
        mix=mix,
        refs=refs,
        slot_medians=slot_medians,
        slot_pools=pools,
        train_medians=anchors,  # scraped cue times = placement anchors
        search_margin_s=cfg.search_margin_s,
        device=device,
        # Cross-set decode needs the anchor prior: without it the DP has no
        # placement signal on an unseen mix and collapses to the front
        # (observed on BB11 — every span < 70 s). Cues are scrape input.
        anchor_sigma_s=60.0,
    )
    print("decoding sequence…")
    preds = aligner.predict_sequence(decodable)

    # ---- 2b. fingerprint placement (regular spans) -------------------------
    # Identity stays with predict_sequence (recording_id per span); the ~30s MERT
    # placement is replaced by the landmark-fp vote-extent, which localizes the
    # mix<->ref diagonal to ~0.2s and gives set_start ~4s median (BB12 regular).
    # ref_start_s = set_start_s + offset_s (off = ref_frame - mix_frame). Spans
    # with no cached fp (or no candidate diagonal) keep their MERT placement.
    fp_active = args.fp_placement
    mert_starts: dict[int, float] = {}
    if fp_active:
        import dataclasses

        from workspaces.alignment_prototype.fp_index import FpKey
        from workspaces.alignment_prototype.fp_index import load as fp_load
        from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir
        from workspaces.alignment_prototype.landmark_fp import constellation, hashes
        from workspaces.alignment_prototype.mix_fp_hits import (
            decode_placements,
            load_mix_mono,
        )

        set_dir = find_aligning_dir(args.set_id)
        mix_file = set_dir / "mix.m4a" if set_dir is not None else None
        # ref fps in tracklist (decodable) order — DO NOT sort
        fps = [
            fp_load(FpKey(p.recording_id, "regular")) if p.recording_id else None
            for p in preds
        ]
        keep = [i for i, fp in enumerate(fps) if fp is not None]
        if mix_file is None or not mix_file.is_file():
            print("(fp placement skipped — aligning mix.m4a missing)")
            fp_active = False
        elif not keep:
            print("(fp placement skipped — no ref fingerprints cached for chosen ids)")
            fp_active = False
        else:
            print(f"fp placement: hashing mix once ({set_dir.name})…")
            hm = hashes(*constellation(load_mix_mono(mix_file)))
            placements = decode_placements(
                hm,
                [fps[i] for i in keep],
                mix_dur_s=mix_end,
                topk=args.fp_placement_topk,
                gap_s=args.fp_placement_gap_s,
                with_offset=True,
            )
            new_preds = list(preds)
            n_placed = n_gated = 0
            gate = args.fp_placement_gate_s
            for r, i in enumerate(keep):
                pl = placements[r]
                if pl is None:
                    continue
                ss, se, off = pl
                mert_start = preds[i].set_start_s
                mert_starts[i] = mert_start
                # Consistency gate: fp is precise but UNLEASHED — a wrong-diagonal
                # or wrong-identity pick can place a span hundreds of seconds off.
                # MERT is coarse but anchored (cue prior + monotonic decode, p90
                # ~78s). Trust fp only as a local refinement of MERT; when it
                # wildly disagrees, the anchored prior is safer. Validated on BB12
                # (band 90s: p90 340->61s, median 9.2->6.6s).
                if gate > 0 and abs(ss - mert_start) > gate:
                    n_gated += 1
                    continue  # keep MERT placement untouched
                new_preds[i] = dataclasses.replace(
                    preds[i],
                    set_start_s=ss,
                    set_end_s=se,
                    ref_start_s=ss + off,
                    ref_end_s=off + se,  # ref_start + span_duration
                )
                n_placed += 1
            preds = tuple(new_preds)
            print(
                f"fp placement: {n_placed}/{len(preds)} spans placed "
                f"({n_gated} gated to MERT |fp-mert|>{gate:.0f}s, "
                f"{len(preds) - n_placed - n_gated} kept MERT — no fp / no diagonal)"
            )

    # ---- 2c. per-stem placement (acappella set_start via banded HuBERT) -----
    # The full-mix fp is weak on vocals, so acappella spans fall back to the ~30s
    # MERT placement above. HuBERT (phonetic, key-invariant) localizes the vocal
    # in mix_vocals where chroma/fp can't. Refine set_start ONLY (the joint
    # ref_start is repeat-ambiguous — left to refine_ref_offsets). Banded ±gate to
    # the coarse prior + a fusion guard (keep prior when HuBERT agrees closely)
    # makes it strictly dominate the prior on BB12 acappella (<8s 42->75%).
    if args.stem_placement:
        import dataclasses

        from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir
        from workspaces.alignment_prototype.stem_placement import hubert_of, place_joint

        ac_idx = [
            i
            for i, p in enumerate(preds)
            if (p.claimed_stem or "regular") == "acappella"
        ]
        set_dir = find_aligning_dir(args.set_id)
        mixv = set_dir / "mix_vocals.flac" if set_dir is not None else None
        if not ac_idx:
            pass
        elif mixv is None or not mixv.is_file():
            print("(stem placement skipped — mix_vocals.flac missing)")
        else:
            by_tid = {
                t["track_id"]: t
                for t in json.loads((set_dir / "manifest.json").read_text())["tracks"]
            }
            print(
                f"stem placement: HuBERT on mix_vocals + {len(ac_idx)} acappella refs…"
            )
            mix_hub = hubert_of(mixv)
            new_preds = list(preds)
            n_stem = n_keep = 0
            for i in ac_idx:
                p = preds[i]
                t = by_tid.get(p.recording_id)
                vpath = (t.get("stems") or {}).get("vocals") if t else None
                ref_hub = hubert_of(vpath) if vpath else None
                if ref_hub is None:
                    continue
                span_dur = p.set_end_s - p.set_start_s
                res = place_joint(
                    mix_hub,
                    ref_hub,
                    p.set_start_s,
                    span_dur,
                    band_s=args.stem_placement_band_s,
                )
                if res is None:
                    continue
                ss, _rs, _pk = res
                # fusion guard: keep the prior when HuBERT agrees closely (protect
                # near-hits); override only when it disagrees (fix the tail).
                if abs(ss - p.set_start_s) <= args.stem_placement_guard_s:
                    n_keep += 1
                    continue
                new_preds[i] = dataclasses.replace(
                    p, set_start_s=ss, set_end_s=ss + span_dur
                )
                n_stem += 1
            preds = tuple(new_preds)
            print(
                f"stem placement: {n_stem} acappella set_starts refined by HuBERT "
                f"({n_keep} kept prior — agreed within {args.stem_placement_guard_s:.0f}s)"
            )

    # ---- 3. fine placement (per-span DTW vs roformer mix instrumental) -----
    # Skipped when fp placement is active: DTW/fp-refine were refinements of the
    # coarse MERT placement and risk pulling a good fp start onto a spurious
    # chroma match. Use --no-fp-placement to get the old refinement path.
    refined = preds
    if not fp_active and args.band_s > 0:
        from workspaces.alignment_prototype.fine_refine import (
            AudioContext,
            refine_placements,
        )

        ctx = AudioContext.from_set(args.set_id)
        if ctx is None:
            print("(fine refinement skipped — aligning audio missing)")
        else:
            print(f"fine-placement DTW ±{args.band_s:.0f}s…")
            refined = refine_placements(preds, decodable, ctx, band_s=args.band_s)

    if not fp_active and args.fp_refine:
        from workspaces.alignment_prototype.fp_placement_refine import (
            FpPlacementContext,
            refine_placements_fp,
        )

        mix_mid = 0.5 * (mix.start_s + mix.end_s)
        fp_ctx = FpPlacementContext.from_set(args.set_id, measure_mid_s=mix_mid)
        if fp_ctx is None:
            print("(fp placement skipped — aligning audio or manifest missing)")
        else:
            print(
                f"fp placement refine ±{args.fp_band_s:.0f}s gate_z={args.fp_gate_z}…"
            )
            refined = refine_placements_fp(
                refined,
                fp_ctx,
                band_s=args.fp_band_s,
                gate_z=args.fp_gate_z,
            )

    # ---- 4. report + serialize ---------------------------------------------
    deltas = [
        abs(p.set_start_s - t.set_start_s)
        for p, t in zip(refined, decodable)
        if t.slot_label in anchors
    ]
    d = np.asarray(deltas)
    print(
        f"\npred vs scraped cue anchors: n={len(d)} median={np.median(d):.1f}s "
        f"mean={d.mean():.1f}s <16s:{(d < 16).sum()} <30s:{(d < 30).sum()} max={d.max():.0f}s"
    )
    print("(cues are coarse fan-scraped times — agreement is a sanity band, not GT)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    payload = {
        "set_id": args.set_id,
        "train_set_id": train_gt.set_id,
        "band_s": args.band_s,
        "fp_refine": args.fp_refine,
        "fp_band_s": args.fp_band_s if args.fp_refine else None,
        "fp_placement": fp_active,
        "spans": [
            {
                **asdict(p),
                "cue_anchor_s": anchors.get(p.slot_label),
                "name": t.label,
                **(
                    {"mert_set_start_s": mert_starts.get(i)}
                    if args.fp_placement_compare
                    else {}
                ),
            }
            for i, (p, t) in enumerate(zip(refined, decodable))
        ],
        "skipped": [{"slot_label": t.slot_label, "name": t.label} for t in skipped],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")

    print(f"\n{'slot':6} {'pred_start':>10} {'cue':>8} {'Δ':>6}  name")
    for p, t in zip(refined, decodable):
        cue = anchors.get(p.slot_label)
        delta = f"{p.set_start_s - cue:+6.0f}" if cue is not None else "     –"
        cue_s = f"{cue:8.0f}" if cue is not None else "       –"
        print(f"{p.slot_label:6} {p.set_start_s:10.1f} {cue_s} {delta}  {t.label[:48]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

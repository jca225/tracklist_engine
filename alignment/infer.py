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
    venvs/audio/bin/python -m alignment.infer \\
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

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from core.result import Err, Ok, Result


def _vocal_ref_path(row: dict | None) -> str | None:
    """Vocal-reference audio for a manifest row.

    Demucs never runs on non-regular recordings, so acappella-claimed rows
    have empty ``stems`` (BB11: 0/64, BB12: 0/33) — but their own audio IS
    vocals. Priority: separated vocals stem, else the acappella master
    itself. (Stem/lyrics channels only ever engaged on BB12 by accident,
    via spans whose predicted recording was the regular sibling.)
    """
    if not row:
        return None
    vpath = (row.get("stems") or {}).get("vocals")
    if vpath:
        return vpath
    if row.get("stem") == "acappella":
        return row.get("local_path") or None
    return None


def _manifest_by_tid(set_dir: Path, set_id: str) -> dict[str, dict]:
    """Manifest rows keyed by track_id AND canonical recording_id.

    Manifest ``track_id`` is the scrape (tlp*) namespace; predictions carry
    canonical ``recording_id``. Bridge via labeling/fixtures/id_maps/<set>.json
    (tlp -> recording_id, from set_track_slots) — without it, ref stem
    resolution silently misses cross-namespace and the stem/lyrics channels
    no-op (BB11 2026-07-02: 0/67 refs resolved on both channels).
    """
    rows = json.loads((set_dir / "manifest.json").read_text())["tracks"]
    by_tid = {row["track_id"]: row for row in rows}
    map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    if map_path.exists():
        for tlp, rec in json.loads(map_path.read_text()).items():
            if tlp in by_tid:
                by_tid.setdefault(rec, by_tid[tlp])
    return by_tid


from alignment.dataset import (
    load_set,
    slot_candidates_from_targets,
)
from alignment.records import SlotCandidate, SpanTarget
from alignment.slot_priors import normalize_slot

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"
DEFAULT_TRAIN_YAML = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"
OUT_DIR = Path(__file__).resolve().parent / "out"

_DUR_MIN_S = 15.0
_DUR_MAX_S = 180.0
_DUR_FALLBACK_S = 45.0

# Opinion-audit #1 (docs/opinion_audit.md): the 90s fp gate is GT-validated
# GLOBALLY but fails in the medley regime — MERT's prior collapses in 4-5-deep
# pileups and the gate then discards dead-correct diagonals. Overwhelming
# evidence breaks the leash. Calibration (BB12 gated-span table, 2026-07-10;
# sharpness here = chosen candidate's votes / strongest OTHER candidate from
# decode_placements, NOT a window histogram ratio): prisoners with GT-correct
# fp separate cleanly — Honest Virtu 2632 votes/1.92 (fp 2.1s from GT, prior
# 123s off), Outside 490/1.41 (fp 2.2s from GT) — vs junk at <=36 votes and
# decode-overridden ambiguity at sharp<1.0 (e.g. 120 votes/0.19, correctly
# kept gated). Floors sit in the gap: votes 3x above junk, sharp above the
# ambiguity band.
_FP_GATE_OVERRIDE_VOTES = 100
_FP_GATE_OVERRIDE_SHARP = 1.2


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


def _apply_open_set_identity(preds, rows, args):
    """Fail-closed blind-LF identity override (Phase 1B, spec §2C).

    Widens each slot to a real candidate pool and applies the stem-MERT chamfer
    margin gate, replacing (override) or nulling (abstain) the span's
    recording_id; accept-claim leaves it. Returns
    ``(new_preds, id_source, id_prov, counts)``. Fail-closed: with no
    cache-dir / tau / floor, or no feature bundle for the set, the claim is kept
    verbatim and no span is touched (so ``--open-set-identity`` never silently
    regresses when its inputs are absent).
    """
    import dataclasses

    from alignment.candidate_pool import build_pools
    from alignment.extract_stem_mert import IdentityFeatureBundle
    from alignment.identity_override import (
        resolve_identities,
        summarize,
    )
    from alignment.open_set_identity import Decision

    id_source = {i: "claim" for i in range(len(preds))}
    id_prov: dict[int, dict] = {}
    if (
        args.identity_cache_dir is None
        or args.identity_tau is None
        or args.identity_floor is None
    ):
        print(
            "open-set-identity: requires --identity-cache-dir + --identity-tau + "
            "--identity-floor; keeping claim (no override)",
            file=sys.stderr,
        )
        return preds, id_source, id_prov, {}
    try:
        bundle = IdentityFeatureBundle.load(args.set_id, args.identity_cache_dir)
    except FileNotFoundError:
        print(
            f"open-set-identity: no L3/L22 feature bundle for {args.set_id} in "
            f"{args.identity_cache_dir} (run extract_stem_mert); keeping claim",
            file=sys.stderr,
        )
        return preds, id_source, id_prov, {}

    pools = build_pools(rows, bundle.set_pool_by_stem)
    results = resolve_identities(
        pools,
        bundle.queries,
        bundle.refs,
        tau=args.identity_tau,
        floor=args.identity_floor,
        spans=bundle.spans,
    )
    new_preds = list(preds)
    for i, p in enumerate(preds):
        res = results.get(p.slot_label)
        if res is None:
            continue
        id_prov[i] = res.provenance
        d = res.decision
        if d.decision == Decision.OVERRIDE:
            new_preds[i] = dataclasses.replace(p, recording_id=d.recording_id)
            id_source[i] = "open_set_mert"
        elif d.decision == Decision.ABSTAIN:
            new_preds[i] = dataclasses.replace(p, recording_id=None)
            id_source[i] = "abstain"
        # accept_claim: recording_id unchanged
    counts = summarize(results)
    print(
        f"open-set-identity: accept={counts.get(Decision.ACCEPT_CLAIM, 0)} "
        f"override={counts.get(Decision.OVERRIDE, 0)} "
        f"abstain={counts.get(Decision.ABSTAIN, 0)} "
        f"(tau={args.identity_tau} floor={args.identity_floor})"
    )
    return tuple(new_preds), id_source, id_prov, counts


def override_identity_from_gt(
    rows: tuple[dict, ...], gt_yaml: Path, set_id: str
) -> tuple[dict, ...]:
    """Oracle identity: replace each slot's claimed recording_id with the GT
    content recording for that slot (matched by normalized slot_label, bridged
    to canonical via the set's id_map). Slots the GT abstains on keep the claim.

    Isolates placement/structure from identity poison — the aligner is handed
    correct identity (from human labeling) and must only place + decode. Measures
    the aligner's real placement/structure capability, since the shipped
    recording_id is otherwise the (frequently-wrong) tokenizer claim verbatim
    (state-of-record decision #19).
    """
    import yaml as _yaml

    from alignment.score_timeline_vs_gt import norm_slot

    idmap_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    idmap = json.loads(idmap_path.read_text()) if idmap_path.exists() else {}
    doc = _yaml.safe_load(gt_yaml.read_text())
    gt_by_slot: dict[str, str] = {}
    for r in doc["tracks"]:
        sl = r.get("slot_label")
        tid = r.get("track_id")
        if sl is None or str(sl) == "mix" or not tid:
            continue
        gt_by_slot.setdefault(norm_slot(str(sl)), idmap.get(str(tid), str(tid)))

    out: list[dict] = []
    n_over = n_keep = 0
    for r in rows:
        gt_rec = gt_by_slot.get(norm_slot(str(r["slot_label"])))
        if gt_rec:
            claim_canon = idmap.get(str(r["recording_id"]), str(r["recording_id"]))
            if gt_rec != claim_canon:
                n_over += 1
            r = {**r, "recording_id": gt_rec}
        else:
            n_keep += 1
        out.append(r)
    print(
        f"oracle identity: overrode {n_over} slot recording_ids from GT "
        f"({n_keep} kept — GT-abstain or no content)"
    )
    return tuple(out)


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
        "--identity-gt-yaml",
        type=Path,
        default=None,
        help="ORACLE IDENTITY: override each slot's recording_id with the GT "
        "content recording (from human labeling) before decode — measures "
        "placement/structure given correct identity, bypassing the tokenizer "
        "claim the aligner otherwise emits verbatim (state-of-record #19).",
    )
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
    p.add_argument(
        "--instr-stem-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="place INSTRUMENTAL spans by fingerprinting the instrumental stem on "
        "both sides (mix_instrumental.flac <-fp-> ref Demucs instrumental stem). The "
        "full-mix fp/chroma fail on instrumental (vocal-carrying mix vs vocal-less "
        "ref); stem-vs-stem fp recovers regular-level placement (probe: BB11 5.0s "
        "set_start). Sets set_start AND ref_start (=set_start+offset), gated to the "
        "prior like --fp-placement. Default ON since the 2026-07-09 pi "
        "re-materialize: DB claimed_stem now carries ~19/25 real instrumentals per "
        "set (residual ~6 are class-1 inventory gaps, GT-only). "
        "--instr-stem-gt-yaml still tops up routing where GT exists (scoring runs).",
    )
    p.add_argument(
        "--instr-stem-gt-yaml",
        type=Path,
        default=None,
        help="GT yaml supplying true per-span claimed_stem + original slot_label for "
        "--instr-stem-placement (the axis label only, as the scorer uses — NOT "
        "placement). Without it the channel sees only DB-visible instrumentals.",
    )
    p.add_argument("--instr-stem-gate-s", type=float, default=90.0)
    p.add_argument(
        "--lyrics-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="place acappella spans via the LYRICS channel (Whisper word-timestamps "
        "+ IDF diagonal + tracklist-order monotonic decode + position prior). Sets "
        "BOTH set_start AND ref_start; abstains on weak spans. Strictly dominates "
        "HuBERT stem-placement on BB12 (set_start 2.3s vs ~tail; ref_start 6.3s vs "
        "~50s wall). Needs mix_vocals.flac + ref vocals stems; transcribes on demand "
        "(cached). Lyrics-placed spans are skipped by HuBERT stem-placement.",
    )
    p.add_argument(
        "--strict-inventory",
        action="store_true",
        help="abort when GT rows lack resolvable ref audio (excluding unalignable)",
    )
    p.add_argument(
        "--open-set-identity",
        action="store_true",
        help="PHASE 1B: replace the size-1 tokenizer-claim pool with a real "
        "multi-candidate pool and apply the blind stem-MERT (L3/L22) chamfer "
        "override, fail-closed via the margin gate (accept-claim | override | "
        "abstain). Needs the L3/L22 feature bundle from extract_stem_mert "
        "(--identity-cache-dir); if absent, no override is applied. Default OFF "
        "until the acceptance gate (identity >= RT1 both sets) passes.",
    )
    p.add_argument("--identity-cache-dir", type=Path, default=None)
    p.add_argument(
        "--identity-tau",
        type=float,
        default=None,
        help="override margin (cosine units) — tuned LOSO, never fit on both sets",
    )
    p.add_argument("--identity-floor", type=float, default=None)
    args = p.parse_args(argv)

    from eda.alignment.spectrogram_review.source_audio import run_audio_preflight
    from alignment.drivers.base import GT_BY_SET

    if args.set_id in GT_BY_SET:
        import yaml

        gt_doc = yaml.safe_load(GT_BY_SET[args.set_id].read_text())
        gt_rows = [r for r in gt_doc["tracks"] if str(r.get("slot_label")) != "mix"]
        if run_audio_preflight(args.set_id, gt_rows, strict=args.strict_inventory):
            return 1

    from alignment.mert_model import (
        MertLearnedAligner,
        TrainConfig,
        train_ensemble,
    )
    from alignment.mert_features import build_examples
    from alignment.mert_store import load_bb12_mert

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
    if args.identity_gt_yaml is not None:
        rows = override_identity_from_gt(rows, args.identity_gt_yaml, args.set_id)
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

    # ---- 2a. Phase 1B blind identity override (before placement, so downstream
    # fp/chroma/HuBERT placement loads the CORRECTED ref) -------------------
    identity_source: dict[int, str] = {i: "claim" for i in range(len(preds))}
    identity_prov: dict[int, dict] = {}
    if args.open_set_identity:
        preds, identity_source, identity_prov, _ = _apply_open_set_identity(
            preds, rows, args
        )

    # Per-span placement provenance (serialized as start_source) — which
    # channel produced the final set_start. Starts as the MERT decode and is
    # overwritten at each override site, so engagement is auditable from the
    # timeline artifact itself, not the run log.
    start_source: dict[int, str] = {i: "mert" for i in range(len(preds))}
    # opinion-audit loudness: every gate firing (or override) lands on the span
    gate_events: dict[int, dict] = {}
    # Every probe's raw proposal, kept even when another probe wins — the
    # agentic replay needs the losing candidates to measure cross-probe
    # agreement (without them, single-source beliefs underestimate quality).
    probe_proposals: dict[int, dict[str, float]] = {
        i: {"mert_decode": p.set_start_s} for i, p in enumerate(preds)
    }
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

        from alignment.fp_index import FpKey
        from alignment.fp_index import load as fp_load
        from alignment.fp_placement_refine import find_aligning_dir
        from alignment.landmark_fp import constellation, hashes
        from alignment.mix_fp_hits import (
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
                with_strength=True,
            )
            new_preds = list(preds)
            n_placed = n_gated = n_override = 0
            gate = args.fp_placement_gate_s
            for r, i in enumerate(keep):
                pl = placements[r]
                if pl is None:
                    continue
                ss, se, off, votes, sharp = pl
                mert_start = preds[i].set_start_s
                mert_starts[i] = mert_start
                probe_proposals[i]["fp"] = ss
                # Consistency gate: fp is precise but UNLEASHED — a wrong-diagonal
                # or wrong-identity pick can place a span hundreds of seconds off.
                # MERT is coarse but anchored (cue prior + monotonic decode, p90
                # ~78s). Trust fp only as a local refinement of MERT; when it
                # wildly disagrees, the anchored prior is safer. Validated on BB12
                # (band 90s: p90 340->61s, median 9.2->6.6s). EXCEPT: overwhelming
                # evidence breaks the leash (opinion-audit #1) — and every gate
                # firing is recorded on the span, never silent.
                if gate > 0 and abs(ss - mert_start) > gate:
                    strength = {
                        "fp_votes": int(votes),
                        "fp_sharpness": round(float(sharp), 2),
                        "fp_delta_s": round(abs(ss - mert_start), 1),
                    }
                    if not (
                        votes >= _FP_GATE_OVERRIDE_VOTES
                        and sharp >= _FP_GATE_OVERRIDE_SHARP
                    ):
                        n_gated += 1
                        gate_events[i] = {"rule": f"fp-gate-{gate:.0f}s", **strength}
                        continue  # keep MERT placement untouched
                    n_override += 1
                    gate_events[i] = {"rule": "fp-gate-OVERRIDE", **strength}
                new_preds[i] = dataclasses.replace(
                    preds[i],
                    set_start_s=ss,
                    set_end_s=se,
                    ref_start_s=ss + off,
                    ref_end_s=off + se,  # ref_start + span_duration
                )
                start_source[i] = "fp"
                n_placed += 1
            preds = tuple(new_preds)
            print(
                f"fp placement: {n_placed}/{len(preds)} spans placed "
                f"({n_gated} gated to MERT |fp-mert|>{gate:.0f}s, "
                f"{n_override} gate OVERRIDES on strong evidence, "
                f"{len(preds) - n_placed - n_gated} kept MERT — no fp / no diagonal)"
            )

    # ---- 2b'. lyrics placement (acappella set_start + ref_start via Whisper) -
    # Words are key/tempo/pitch invariant — the axis the acoustic matched-filter
    # lacks. Whisper-transcribe mix_vocals + each candidate vocals stem, find
    # alignment diagonals by IDF/distinct-rare-bigram Hough, joint monotonic decode
    # over tracklist order + a Gaussian position prior, abstain on weak spans.
    # BB12 acappella: set_start 2.3s median (was 42.5s), ref_start 6.3s (was ~50s
    # repeat wall). Sets BOTH axes; HuBERT stem-placement below fills abstentions.
    lyrics_placed: set[int] = set()
    if args.lyrics_placement:
        import dataclasses

        from alignment.fp_placement_refine import find_aligning_dir
        from alignment.lyrics_align import (
            _bigram_times,
            _norm,
            _slot_order,
            candidate_diagonals,
            monotonic_decode,
            resolve_tracklist_slot,
            tracklist_max_slot,
            transcribe_words,
        )

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
            print("(lyrics placement skipped — mix_vocals.flac missing)")
        else:
            manifest = json.loads((set_dir / "manifest.json").read_text())
            by_tid = _manifest_by_tid(set_dir, args.set_id)
            mix_dur = float(manifest.get("mix_duration_s") or 0) or max(
                p.set_end_s for p in preds
            )
            man_slots = [
                str(r["slot_label"])
                for r in (manifest.get("tracks") or [])
                if isinstance(r, dict) and r.get("slot_label")
            ]
            max_slot = (
                tracklist_max_slot(man_slots)
                if man_slots
                else tracklist_max_slot(
                    [str(p.slot_label) for p in preds if p.slot_label]
                )
            )
            print(
                f"lyrics placement: transcribing mix_vocals + {len(ac_idx)} acappella "
                "refs (cached)…"
            )
            mix_bt = _bigram_times(_norm(transcribe_words(mixv)))
            items: list[tuple[tuple[int, int], int, list, float]] = []
            for i in ac_idx:
                p = preds[i]
                t = by_tid.get(p.recording_id)
                row = t if isinstance(t, dict) else None
                vpath = _vocal_ref_path(t)
                if not vpath or not Path(vpath).is_file():
                    continue
                cw = transcribe_words(vpath)
                if not cw:
                    continue
                cands = candidate_diagonals(_norm(cw), mix_bt)
                if not cands:
                    continue
                man_slot = resolve_tracklist_slot(p.slot_label, row)
                order = _slot_order(man_slot)
                epos = order[0] / max_slot * mix_dur
                items.append((order, i, cands, epos))
            if items:
                items.sort(key=lambda it: it[0])
                spans = [(cands, epos) for _order, _i, cands, epos in items]
                idxs = [i for _order, i, _cands, _epos in items]
                chosen = monotonic_decode(spans)
                new_preds = list(preds)
                n_lyr = 0
                for i, (ss, rs) in zip(idxs, chosen):
                    if ss is None:
                        continue
                    p = preds[i]
                    dur = p.set_end_s - p.set_start_s
                    new_preds[i] = dataclasses.replace(
                        p, set_start_s=ss, set_end_s=ss + dur, ref_start_s=rs
                    )
                    probe_proposals[i]["lyrics"] = ss
                    start_source[i] = "lyrics"
                    lyrics_placed.add(i)
                    n_lyr += 1
                preds = tuple(new_preds)
                print(
                    f"lyrics placement: {n_lyr}/{len(idxs)} acappella spans placed "
                    f"({len(idxs) - n_lyr} abstained)"
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

        from alignment.fp_placement_refine import find_aligning_dir
        from alignment.stem_placement import hubert_of, place_joint

        ac_idx = [
            i
            for i, p in enumerate(preds)
            if (p.claimed_stem or "regular") == "acappella" and i not in lyrics_placed
        ]
        set_dir = find_aligning_dir(args.set_id)
        mixv = set_dir / "mix_vocals.flac" if set_dir is not None else None
        if not ac_idx:
            pass
        elif mixv is None or not mixv.is_file():
            print("(stem placement skipped — mix_vocals.flac missing)")
        else:
            by_tid = _manifest_by_tid(set_dir, args.set_id)
            print(
                f"stem placement: HuBERT on mix_vocals + {len(ac_idx)} acappella refs…"
            )
            mix_hub = hubert_of(mixv)
            new_preds = list(preds)
            n_stem = n_keep = 0
            for i in ac_idx:
                p = preds[i]
                t = by_tid.get(p.recording_id)
                vpath = _vocal_ref_path(t)
                if vpath and not Path(vpath).is_file():
                    vpath = None
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
                probe_proposals[i]["stem_hubert"] = ss
                # fusion guard: keep the prior when HuBERT agrees closely (protect
                # near-hits); override only when it disagrees (fix the tail).
                if abs(ss - p.set_start_s) <= args.stem_placement_guard_s:
                    n_keep += 1
                    continue
                new_preds[i] = dataclasses.replace(
                    p, set_start_s=ss, set_end_s=ss + span_dur
                )
                start_source[i] = "stem_hubert"
                n_stem += 1
            preds = tuple(new_preds)
            print(
                f"stem placement: {n_stem} acappella set_starts refined by HuBERT "
                f"({n_keep} kept prior — agreed within {args.stem_placement_guard_s:.0f}s)"
            )

    # ---- 2d. instrumental stem-fp placement (set_start + ref_start) ---------
    # The full-mix fp / chroma fail on instrumental (vocal-carrying mix bus vs
    # vocal-less ref stem — timbral mismatch). Fingerprint the instrumental stem
    # on BOTH sides (mix_instrumental.flac <-fp-> ref Demucs instrumental) and it
    # recovers regular-level placement (probe: BB11 88% id / 5.0s median). Same
    # decode primitive + gate as --fp-placement; ref_start = set_start + offset.
    # Default ON (W1 kernel default; A/B: BB12 +0.5 / BB11 +8.7, no linear
    # penalty). DB claimed_stem re-materialized 2026-07-09 (19/25 real
    # instrumentals visible per set); --instr-stem-gt-yaml tops up the ~6
    # GT-only spans where GT exists.
    if args.instr_stem_placement:
        import dataclasses

        from alignment.fp_placement_refine import find_aligning_dir
        from alignment.instr_stem_placement import (
            load_stem_overrides,
            place_instr_spans,
        )

        set_dir = find_aligning_dir(args.set_id)
        overrides = (
            load_stem_overrides(
                args.instr_stem_gt_yaml,
                id_map_dir=_REPO / "labeling" / "fixtures" / "id_maps",
            )
            if args.instr_stem_gt_yaml
            else None
        )
        if set_dir is None or not (set_dir / "mix_instrumental.flac").is_file():
            print("(instr stem placement skipped — mix_instrumental.flac missing)")
        else:
            placements, ist = place_instr_spans(
                list(preds),
                set_dir=set_dir,
                set_id=args.set_id,
                mix_dur_s=mix_end,
                stem_overrides=overrides,
            )
            if overrides is None and ist["db_instr"] <= 2:
                print(
                    f"  WARNING: --instr-stem-placement sees only {ist['db_instr']} "
                    "instrumental spans from the STALE DB claimed_stem (row-text "
                    "drop bug marks ~2/set vs ~25 real). Pass --instr-stem-gt-yaml "
                    "to route the real instrumental spans."
                )
            gate = args.instr_stem_gate_s
            new_preds = list(preds)
            n_placed = n_gated = 0
            for i, (ss, se, off) in placements.items():
                probe_proposals[i]["instr_fp"] = ss
                prior = preds[i].set_start_s
                # Consistency gate, same as --fp-placement: the stem fp is precise
                # but unleashed (wrong-diagonal/repeat picks land far off); trust it
                # only as a local refinement of the anchored prior.
                if gate > 0 and abs(ss - prior) > gate:
                    n_gated += 1
                    continue
                new_preds[i] = dataclasses.replace(
                    preds[i],
                    set_start_s=ss,
                    set_end_s=se,
                    ref_start_s=ss + off,
                    ref_end_s=off + se,
                )
                start_source[i] = "instr_fp"
                n_placed += 1
            preds = tuple(new_preds)
            print(
                f"instr stem placement: {n_placed} instrumental spans placed "
                f"({n_gated} gated |instr_fp-prior|>{gate:.0f}s; "
                f"gt_instr={ist['gt_instr']} db_instr={ist['db_instr']} "
                f"ref_resolved={ist['resolved']} fp_decoded={ist['placed']})"
            )

    # ---- 3. fine placement (per-span DTW vs roformer mix instrumental) -----
    # Skipped when fp placement is active: DTW/fp-refine were refinements of the
    # coarse MERT placement and risk pulling a good fp start onto a spurious
    # chroma match. Use --no-fp-placement to get the old refinement path.
    refined = preds
    if not fp_active and args.band_s > 0:
        from alignment.fine_refine import (
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
        from alignment.fp_placement_refine import (
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
                "start_source": start_source.get(i, "mert"),
                "identity_source": identity_source.get(i, "claim"),
                "probe_proposals": probe_proposals.get(i, {}),
                **({"identity": identity_prov[i]} if i in identity_prov else {}),
                **({"placement_gated": gate_events[i]} if i in gate_events else {}),
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
    from datetime import datetime

    from alignment import provenance

    gt_paths = [args.train_yaml]
    if args.identity_gt_yaml is not None:
        gt_paths.append(args.identity_gt_yaml)
    payload["provenance"] = provenance.stamp(
        args.set_id,
        rows,
        gt_paths=gt_paths,
        flags={
            "band_s": args.band_s,
            "fp_placement": args.fp_placement,
            "lyrics_placement": args.lyrics_placement,
            "stem_placement": args.stem_placement,
            "instr_stem_placement": args.instr_stem_placement,
            "identity_gt": args.identity_gt_yaml.name
            if args.identity_gt_yaml
            else None,
            "open_set_identity": args.open_set_identity,
            "identity_tau": args.identity_tau if args.open_set_identity else None,
            "identity_floor": args.identity_floor if args.open_set_identity else None,
            "fp_placement_gate_s": args.fp_placement_gate_s,
        },
        written_at=datetime.now().isoformat(timespec="seconds"),
    )
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

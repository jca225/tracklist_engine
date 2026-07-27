"""FP top-K mode audit on ridge-diagnostic decoder_wall cases.

For each case where the ridge study found an fp_hit diagonal but placement
still failed: dump ``offset_candidates`` (infer defaults) vs GT audible onset
and the agentic prediction. Verdicts decide the next lever — selection vs
gate/refine vs missing candidate.

Usage (repo root of the worktree):
    venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.mode_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eda.alignment.ridge_diagnostic.cases import CaseRecord
from eda.alignment.ridge_diagnostic.features import _mix_full_path, _resolve_ref_path
from core.contracts import MANIFEST_FILENAME, load_manifest
from alignment.fp_index import FpKey
from alignment.fp_index import load as fp_load
from alignment.fp_index import save_cached
from alignment.landmark_fp import (
    constellation,
    fingerprint_from_audio,
    hashes,
)
from alignment.mix_fp_hits import load_mix_mono, offset_candidates
from alignment.refine_ref_offsets import find_aligning_dir

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "out"
_NEAR_S = 6.0  # same order as infer --fp-placement-gap-s


@dataclass(frozen=True)
class ModeAuditRow:
    case_id: str
    set_id: str
    slot: str
    claimed_stem: str
    span_class: str
    place_err_s: float
    gt_audible_onset_s: float
    pred_set_start_s: float
    n_candidates: int
    argmax_set_start_s: float | None
    argmax_votes: int | None
    argmax_err_s: float | None
    nearest_rank: int | None  # 1-based among top-K; None if none near
    nearest_set_start_s: float | None
    nearest_votes: int | None
    nearest_err_s: float | None
    pred_err_s: float
    verdict: str
    candidates_json: str


def _load_decoder_wall_cases(cases_path: Path, contrast_path: Path) -> list[CaseRecord]:
    raw = json.loads(cases_path.read_text())
    by_id = {r["case_id"]: r for r in raw}
    with contrast_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    present: set[str] = set()
    for r in rows:
        if r["verdict_channel"] == "ridge_present":
            present.add(r["case_id"])
    out: list[CaseRecord] = []
    for cid in present:
        d = by_id[cid]
        out.append(
            CaseRecord(
                case_id=d["case_id"],
                set_id=d["set_id"],
                set_label=d["set_label"],
                slot=d["slot"],
                recording_id=d["recording_id"],
                claimed_stem=d["claimed_stem"],
                span_class=d["span_class"],
                place_err_s=float(d["place_err_s"]),
                pred_set_start_s=float(d["pred_set_start_s"]),
                gt_set_start_s=float(d["gt_set_start_s"]),
                gt_audible_onset_s=float(d["gt_audible_onset_s"]),
                gt_ref_start_s=float(d["gt_ref_start_s"]),
                gt_set_end_s=float(d["gt_set_end_s"]),
                gt_ref_end_s=d.get("gt_ref_end_s"),
                tempo_ratio=float(d["tempo_ratio"]),
                pitch_shift_semi=int(d["pitch_shift_semi"]),
                ref_segments=tuple(tuple(x) for x in d["ref_segments"]),
                source_timeline=d["source_timeline"],
                taxonomy_tags=tuple(d.get("taxonomy_tags") or ()),
            )
        )
    return sorted(out, key=lambda c: -c.place_err_s)


def _ref_fp_for_case(case: CaseRecord, set_dir: Path):
    """Prefer cached regular fp (infer path); else fingerprint resolved ref audio."""
    cached = fp_load(FpKey(case.recording_id, "regular"))
    if cached is not None:
        return cached, "cache:regular"
    # instrumental claimed_stem: try instrumental cache then build from audio
    if case.claimed_stem == "instrumental":
        hit = fp_load(FpKey(case.recording_id, "instrumental"))
        if hit is not None:
            return hit, "cache:instrumental"
    manifest = load_manifest(set_dir / MANIFEST_FILENAME)
    ref_path = _resolve_ref_path(set_dir, manifest.by_track_id(), case)
    if ref_path is None or not ref_path.is_file():
        return None, "missing"
    import librosa

    y, _ = librosa.load(str(ref_path), sr=22050, mono=True)
    fp = fingerprint_from_audio(np.asarray(y, dtype=np.float32), sr=22050)
    stem_key = "instrumental" if case.claimed_stem == "instrumental" else "regular"
    save_cached(fp, FpKey(case.recording_id, stem_key))
    return fp, f"built:{stem_key}"


def _verdict(
    *,
    gt: float,
    pred: float,
    cands: list[tuple[float, float, int, float]],
    near_s: float,
) -> tuple[str, int | None, tuple[float, float, int, float] | None]:
    """Return (verdict, nearest_rank_1based, nearest_cand)."""
    if not cands:
        return "gt_absent", None, None
    ranked = sorted(
        enumerate(cands, start=1),
        key=lambda iv: abs(iv[1][0] - gt),
    )
    nearest_rank, nearest = ranked[0]
    nearest_err = abs(nearest[0] - gt)
    argmax = cands[0]
    argmax_err = abs(argmax[0] - gt)
    pred_err = abs(pred - gt)

    if nearest_err > near_s:
        return "gt_absent", None, None
    # GT is among top-K (nearest within near_s)
    if argmax_err <= near_s:
        # dominant candidate already near GT
        if pred_err <= near_s:
            return "gt_is_argmax", 1, argmax
        # argmax near GT but agentic still far → gate/refine dropped a good pick
        return "gt_is_argmax", 1, argmax
    # some non-argmax candidate near GT; agentic missed it
    if pred_err > near_s:
        return "gt_in_topk_missed", nearest_rank, nearest
    # pred already near GT somehow — still note GT was non-argmax in top-K
    return "gt_in_topk_missed", nearest_rank, nearest


def audit_case(
    case: CaseRecord,
    mix_hashes: dict,
    *,
    topk: int,
    gap_s: float,
    near_s: float,
) -> ModeAuditRow:
    set_dir = find_aligning_dir(case.set_id)
    if set_dir is None:
        raise FileNotFoundError(f"no aligning dir for {case.set_id}")
    ref_fp, _src = _ref_fp_for_case(case, set_dir)
    if ref_fp is None:
        return ModeAuditRow(
            case_id=case.case_id,
            set_id=case.set_id,
            slot=case.slot,
            claimed_stem=case.claimed_stem,
            span_class=case.span_class,
            place_err_s=case.place_err_s,
            gt_audible_onset_s=case.gt_audible_onset_s,
            pred_set_start_s=case.pred_set_start_s,
            n_candidates=0,
            argmax_set_start_s=None,
            argmax_votes=None,
            argmax_err_s=None,
            nearest_rank=None,
            nearest_set_start_s=None,
            nearest_votes=None,
            nearest_err_s=None,
            pred_err_s=abs(case.pred_set_start_s - case.gt_audible_onset_s),
            verdict="gt_absent",
            candidates_json="[]",
        )

    cands = offset_candidates(mix_hashes, ref_fp, topk=topk, gap_s=gap_s)
    gt = case.gt_audible_onset_s
    pred = case.pred_set_start_s
    verdict, nearest_rank, nearest = _verdict(
        gt=gt, pred=pred, cands=cands, near_s=near_s
    )
    argmax = cands[0] if cands else None
    cand_payload = [
        {
            "rank": i,
            "set_start_s": round(ss, 3),
            "set_end_s": round(se, 3),
            "votes": int(v),
            "offset_s": round(off, 3),
            "err_to_gt_s": round(abs(ss - gt), 3),
        }
        for i, (ss, se, v, off) in enumerate(cands, start=1)
    ]
    return ModeAuditRow(
        case_id=case.case_id,
        set_id=case.set_id,
        slot=case.slot,
        claimed_stem=case.claimed_stem,
        span_class=case.span_class,
        place_err_s=case.place_err_s,
        gt_audible_onset_s=gt,
        pred_set_start_s=pred,
        n_candidates=len(cands),
        argmax_set_start_s=None if argmax is None else float(argmax[0]),
        argmax_votes=None if argmax is None else int(argmax[2]),
        argmax_err_s=None if argmax is None else abs(float(argmax[0]) - gt),
        nearest_rank=nearest_rank,
        nearest_set_start_s=None if nearest is None else float(nearest[0]),
        nearest_votes=None if nearest is None else int(nearest[2]),
        nearest_err_s=None if nearest is None else abs(float(nearest[0]) - gt),
        pred_err_s=abs(pred - gt),
        verdict=verdict,
        candidates_json=json.dumps(cand_payload),
    )


def run(
    *,
    topk: int = 6,
    gap_s: float = 6.0,
    near_s: float = _NEAR_S,
    cases_path: Path | None = None,
    contrast_path: Path | None = None,
    out_tsv: Path | None = None,
) -> list[ModeAuditRow]:
    cases_path = cases_path or (_OUT / "cases.json")
    contrast_path = contrast_path or (_OUT / "contrast_table.tsv")
    out_tsv = out_tsv or (_OUT / "mode_audit.tsv")
    cases = _load_decoder_wall_cases(cases_path, contrast_path)
    if not cases:
        raise SystemExit("no decoder_wall cases found in contrast_table.tsv")

    # hash each mix once
    mix_hashes_by_set: dict[str, dict] = {}
    rows: list[ModeAuditRow] = []
    for case in cases:
        if case.set_id not in mix_hashes_by_set:
            set_dir = find_aligning_dir(case.set_id)
            if set_dir is None:
                raise FileNotFoundError(f"no aligning dir for {case.set_id}")
            mix_file = _mix_full_path(set_dir)
            print(f"hashing mix {case.set_id} ({mix_file.name})…", flush=True)
            mix_hashes_by_set[case.set_id] = hashes(
                *constellation(load_mix_mono(mix_file))
            )
        row = audit_case(
            case,
            mix_hashes_by_set[case.set_id],
            topk=topk,
            gap_s=gap_s,
            near_s=near_s,
        )
        rows.append(row)
        print(
            f"  {row.case_id}: {row.verdict}  "
            f"pred_err={row.pred_err_s:.1f}s  "
            f"argmax_err={row.argmax_err_s}  "
            f"nearest_rank={row.nearest_rank}  "
            f"nearest_err={row.nearest_err_s}",
            flush=True,
        )

    _OUT.mkdir(parents=True, exist_ok=True)
    fields = list(ModeAuditRow.__dataclass_fields__.keys())
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"wrote {out_tsv}", flush=True)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    print("aggregate:", counts, flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--gap-s", type=float, default=6.0)
    p.add_argument("--near-s", type=float, default=_NEAR_S)
    p.add_argument("--cases-json", type=Path, default=None)
    p.add_argument("--contrast-tsv", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    run(
        topk=args.topk,
        gap_s=args.gap_s,
        near_s=args.near_s,
        cases_path=args.cases_json,
        contrast_path=args.contrast_tsv,
        out_tsv=args.out,
    )


if __name__ == "__main__":
    main()

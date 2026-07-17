"""Validate the co-training seam's ACCEPT precision on the BB hand-GT sets.

THE gate on the 40k-scale flywheel: before harvesting pseudo-labels from ~1,016
real sets, prove that the seam's ACCEPT band ("≥2 independent probe channels
agree tightly") actually means "the candidate ref is GT-correct" — measured on
the two sets where we have hand ground truth (BB11 `2nvzlh2k`, BB12 `1fsnxchk`).

The seam already computes the confusion (`cotrain_seam.evaluate_banding`,
`BandingScore.precision`); what was missing is the bridge from hand GT to
`GtCandidateCase`s. `build_gt_cases` supplies it:

- **Positive** case per alignable span: the GT-correct ref (recording_id +
  claimed_stem + resolved ref audio). Measures ACCEPT *recall* (yield).
- **Decoy** cases: the SAME span paired with a WRONG ref (another track in the
  set, routed as the span's stem, real audio). Measures ACCEPT *precision* —
  whether cross-channel agreement ever fires on a wrong ref (the poison risk).

Pure builder (unit-tested with a fake resolver); the CLI wires the real manifest
resolver + `real_probe_scorer` and runs it against `~/aligning/<set>/` audio.
No canonical writes; CPU-only probes (fp/chroma/HuBERT) — no GPU, no new data.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, load
from workspaces.pws_aligner.cotrain_seam import (
    DEFAULT_THRESHOLDS,
    Band,
    BandingScore,
    BandThresholds,
    CandidateProposal,
    GtCandidateCase,
    MixSpan,
    RefCandidate,
    evaluate_banding,
    real_probe_scorer,
)

# canonical BB GT set-ids (memory: BB11=2nvzlh2k, BB12=1fsnxchk)
_SET_ALIASES = {"bb11": "2nvzlh2k", "bb12": "1fsnxchk"}
_GT_FIXTURE = {
    "2nvzlh2k": "bb11_ground_truth.yaml",
    "1fsnxchk": "bb12_ground_truth.yaml",
}

# a resolver maps (recording_id, claimed_stem) -> ref audio path (or None if absent)
RefResolver = Callable[[str, str], Path | None]


def gt_relative_offset(track: GroundTruthTrack) -> float:
    """GT offset in the seam's RELATIVE frame (ref_start - set_start).

    The seam's consensus offset is RELATIVE (capture_votes convention); compare
    against this to check whether an ACCEPTed span was placed at the right ref.
    """
    return track.ref_start_s - track.set_start_s


def _claim_axes(stem: str) -> dict[str, str]:
    return {"version": "original", "stem": stem, "variant": "regular"}


def score_by_stem(
    cases: list[GtCandidateCase], proposals: list[CandidateProposal]
) -> dict[str, BandingScore]:
    """Per-axis ACCEPT confusion, keyed by the candidate's claimed stem.

    Aggregate precision hides the axis story — instrumental/regular identity is
    strong, acappella is the weak axis. The go/no-go on the flywheel is usually
    per-axis ("turn it on for the strong axes, hold acappella"), so break the
    confusion out by stem.
    """
    out: dict[str, BandingScore] = {}
    for case, prop in zip(cases, proposals):
        stem = case.candidate.stem or "regular"
        out.setdefault(stem, BandingScore()).observe(
            prop.agreement.band, case.candidate_is_correct
        )
    return out


def build_gt_cases(
    gt: GroundTruthSet,
    ref_resolver: RefResolver,
    *,
    n_decoys: int = 1,
    seed: int = 0,
    stem_filter: str | None = None,
    max_spans: int | None = None,
) -> list[GtCandidateCase]:
    """Turn a hand `GroundTruthSet` into positive + decoy `GtCandidateCase`s.

    A track is a positive span iff it is alignable (`not unalignable`), carries a
    `track_id`, resolves to real ref audio at its `claimed_stem`, and (when given)
    matches `stem_filter`. Each positive gets up to `n_decoys` wrong-ref cases on
    the same span, drawn from other eligible tracks (same stem preferred — a fair
    hard negative), routed as the span's stem so the same probe channels run.
    Deterministic given `seed`.
    """
    rng = random.Random(seed)

    eligible: list[tuple[GroundTruthTrack, Path]] = []
    for t in gt.tracks:
        if t.unalignable or not t.track_id:
            continue
        if stem_filter is not None and t.claimed_stem != stem_filter:
            continue
        path = ref_resolver(t.track_id, t.claimed_stem)
        if path is None:
            continue
        eligible.append((t, path))
    if max_spans is not None:
        eligible = eligible[:max_spans]

    cases: list[GtCandidateCase] = []
    for t, path in eligible:
        span = MixSpan(
            set_id=gt.set_id,
            slot_label=t.slot_label,
            set_start_s=t.set_start_s,
            span_dur_s=t.set_end_s - t.set_start_s,
        )
        cases.append(
            GtCandidateCase(
                candidate=RefCandidate(
                    recording_id=t.track_id or "",
                    source_url="gt://correct",
                    source_path=str(path),
                    display_name=t.label,
                    stem=t.claimed_stem,
                ),
                span=span,
                candidate_is_correct=True,
                claim_axes=_claim_axes(t.claimed_stem),
            )
        )
        cases.extend(
            _decoys_for(t, span, eligible, ref_resolver, n_decoys=n_decoys, rng=rng)
        )
    return cases


def _decoys_for(
    truth: GroundTruthTrack,
    span: MixSpan,
    eligible: list[tuple[GroundTruthTrack, Path]],
    ref_resolver: RefResolver,
    *,
    n_decoys: int,
    rng: random.Random,
) -> list[GtCandidateCase]:
    if n_decoys <= 0:
        return []
    span_stem = truth.claimed_stem
    others = [ot for ot, _ in eligible if ot.track_id != truth.track_id]
    same_stem = [ot for ot in others if ot.claimed_stem == span_stem]
    diff_stem = [ot for ot in others if ot.claimed_stem != span_stem]
    rng.shuffle(same_stem)
    rng.shuffle(diff_stem)

    out: list[GtCandidateCase] = []
    for ot in same_stem + diff_stem:  # prefer same-stem hard negatives
        if len(out) >= n_decoys:
            break
        dpath = ref_resolver(ot.track_id or "", span_stem)
        if dpath is None:
            continue  # this track has no audio routable as the span's stem
        out.append(
            GtCandidateCase(
                candidate=RefCandidate(
                    recording_id=ot.track_id or "",
                    source_url="gt://decoy",
                    source_path=str(dpath),
                    display_name=ot.label,
                    stem=span_stem,
                ),
                span=span,
                candidate_is_correct=False,
                claim_axes=_claim_axes(span_stem),
            )
        )
    return out


# ── CLI: run the real-probe validation on a BB GT set ─────────────────────────


def _manifest_resolver(manifest_by_rid: dict[str, dict]) -> RefResolver:
    from workspaces.pws_aligner.capture_votes import _ref_audio_path

    def resolve(rid: str, stem: str) -> Path | None:
        row = manifest_by_rid.get(rid)
        if row is None:
            return None
        return _ref_audio_path(row, stem)

    return resolve


def _load_gt(set_id: str) -> GroundTruthSet:
    fixture = _GT_FIXTURE.get(set_id)
    if fixture is None:
        raise SystemExit(f"no GT fixture registered for set_id={set_id}")
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "labeling" / "fixtures" / fixture
    res = load(path)
    if not res.is_ok():
        raise SystemExit(f"failed to load GT {path}: {res.error}")
    return res.value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, help="bb11 | bb12 | <set_id>")
    ap.add_argument("--n-decoys", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stem", default=None, help="restrict to one claimed_stem")
    ap.add_argument("--max-spans", type=int, default=None)
    ap.add_argument(
        "--accept-tol-s", type=float, default=DEFAULT_THRESHOLDS.accept_tol_s
    )
    ap.add_argument("--accept-conf", type=float, default=DEFAULT_THRESHOLDS.accept_conf)
    ap.add_argument(
        "--offset-tol-s",
        type=float,
        default=2.0,
        help="ACCEPTed positive counts as correctly PLACED within this",
    )
    args = ap.parse_args(argv)

    from workspaces.pws_aligner.capture_votes import (
        _find_aligning_dir,
        _load_manifest_by_rid,
    )

    set_id = _SET_ALIASES.get(args.set, args.set)
    gt = _load_gt(set_id)
    set_dir = _find_aligning_dir(set_id)
    if set_dir is None:
        raise SystemExit(f"no ~/aligning/{set_id}__* dir found — pull the set first")
    manifest = _load_manifest_by_rid(set_dir, set_id)

    resolver = _manifest_resolver(manifest)
    cases = build_gt_cases(
        gt,
        resolver,
        n_decoys=args.n_decoys,
        seed=args.seed,
        stem_filter=args.stem,
        max_spans=args.max_spans,
    )
    thresholds = BandThresholds(
        accept_tol_s=args.accept_tol_s, accept_conf=args.accept_conf
    )
    scorer = real_probe_scorer(set_dir)

    print(f"=== ACCEPT-precision validation: {args.set} ({set_id}) ===")
    print(f"aligning_dir={set_dir}")
    print(f"cases={len(cases)}  n_decoys={args.n_decoys}  stem={args.stem or 'all'}")
    print("running real probes (fp/chroma/HuBERT) — CPU, may take a while...\n")

    score, proposals = evaluate_banding(cases, scorer, thresholds=thresholds)

    # placement quality: of ACCEPTed POSITIVES, the |consensus - GT| offset error
    # distribution (report the spread, not a single too-strict threshold — the
    # instrumental/acappella axes are known-loose at seconds scale).
    import statistics as _stats

    by_slot = {t.slot_label: t for t in gt.tracks}
    placement_errors: list[float] = []
    accept_wrong_detail: list[str] = []
    for prop, case in zip(proposals, cases):
        band = prop.agreement.band
        if band is Band.ACCEPT and case.candidate_is_correct:
            truth = by_slot.get(case.span.slot_label)
            cons = prop.agreement.consensus_offset_s
            if truth is not None and cons is not None:
                placement_errors.append(abs(cons - gt_relative_offset(truth)))
        if band is Band.ACCEPT and not case.candidate_is_correct:
            accept_wrong_detail.append(
                f"  slot {case.span.slot_label}: ACCEPTed wrong ref "
                f"{case.candidate.recording_id} — {prop.agreement.detail}"
            )

    print(f"n={score.n}  PRECISION={score.precision:.3f}  recall={score.recall:.3f}")
    print(
        f"  accept_correct={score.accept_correct}  accept_wrong={score.accept_wrong}  "
        f"review={score.review}  abstain_correct={score.abstain_correct}  "
        f"abstain_missed={score.abstain_missed}"
    )
    per_stem = score_by_stem(cases, proposals)
    print("  by stem (the go/no-go is per-axis):")
    for stem in sorted(per_stem):
        s = per_stem[stem]
        print(
            f"    {stem:>12}: precision={s.precision:.3f} recall={s.recall:.3f} "
            f"(accept_correct={s.accept_correct} accept_wrong={s.accept_wrong} "
            f"review={s.review} abstain_missed={s.abstain_missed})"
        )
    if placement_errors:
        errs = sorted(placement_errors)
        med = _stats.median(errs)
        p90 = errs[min(len(errs) - 1, int(len(errs) * 0.9))]
        within = sum(1 for e in errs if e <= args.offset_tol_s)
        print(
            f"  placement of ACCEPTed positives (n={len(errs)}): "
            f"median |err|={med:.1f}s  p90={p90:.1f}s  "
            f"{within}/{len(errs)} within {args.offset_tol_s}s"
        )
    if accept_wrong_detail:
        print(
            "\n!!! ACCEPT_WRONG (poison risk) — the flywheel gate cares MOST about these:"
        )
        print("\n".join(accept_wrong_detail))
    else:
        print("\nno ACCEPT_WRONG — 2-channel agreement never fired on a decoy (good).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Co-training seam (DRY-RUN skeleton) for the aligner weak-supervision flywheel.

The "check for better versions online / acquire the missing ref" capability is
the *acquisition arm* of a co-training loop: acquisition proposes a candidate
reference, and **the aligner run against the mix is the validator**. When the
candidate aligns cleanly to the mix span, that agreement is itself training
signal — a labeled (ref ↔ mix-span) pair plus a proposed ``track_audio``
correction. When the probes disagree, the seam abstains (or escalates to human
review) rather than laundering a guess into the canonical DB.

This module wires TWO seams as pure, testable functions:

1. ``build_acquisition_cases`` — suspect slots + identity-gate signals →
   ``AcquisitionCase`` objects (Seam 1). No search is executed here; the case is
   only *opened* with its problem classes. Feeds the acquisition executor.

2. ``propose_from_candidate`` — a candidate ref + a mix span, scored by the
   EXISTING probes (fp-offset + HuBERT/chroma), banded Fellegi–Sunter style into
   ACCEPT / REVIEW / ABSTAIN by *independent-probe agreement* (never one
   channel). On ACCEPT it returns a ``TrainingSignal`` and a **PROPOSED**
   ``track_audio_correction`` ledger entry (Seam 2).

Hard invariant — ZERO autonomous canonical mutation. Every function here is
pure and returns *proposals*. Nothing writes to the canonical DB, to
``track_audio_correction``, or to the acquisition-case JSONL. The ``--dry-run``
CLI only *reports* precision/recall on the BB GT sets and prints the proposals.

Substrate: imports ``core`` (identity, acquisition-case types) + the alignment
contract's ``AlignmentResult`` shape. The real probes are injected as a scorer
callable (``RefMixScorer``) so all banding logic is unit-testable offline with a
FAKE scorer. The real-probe scorer (``real_probe_scorer``) is now wired: it
delegates to the SAME harness-probe machinery ``capture_votes`` runs (context
construction, axis routing, per-probe error absorption, and the load-bearing
ABSOLUTE→RELATIVE offset-frame normalization), and is a safe all-abstain no-op
when the set/ref audio isn't present. Per-probe confidence *calibration* to a
shared [0,1] scale is the remaining open work — banding leans first on offset
*agreement*, only secondarily on confidence. This module does NOT import or
mutate ``ingest/`` / ``analysis/`` — it reads their outputs
(``ingest.identity_gate.VerifyResult``).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence

from core.acquisition_case import (
    AcquisitionCase,
    Actor,
    Attempt,
    AttemptAction,
    AttemptVerdict,
    CaseClaim,
    CaseStatus,
    ProblemClass,
    Resolution,
    TrainingSignal,
    with_problem_classes,
)
from core.identity import normalize_stem, normalize_variant, normalize_version
from core.slot_inventory import LayerRole, derive_layer_role

# The probe output shape (one AlignmentResult per independent probe channel).
from alignment.harness.contract import AlignmentResult

# ── Seam 1: suspect slot → AcquisitionCase ────────────────────────────────────

# identity_gate.Verdict.value → the acquisition-case problem class it implies.
# We key on the *string* value so we never import ingest at module load, keeping
# the substrate independent of ingest internals (read outputs, not code).
_VERDICT_TO_PROBLEM: Mapping[str, ProblemClass] = {
    "WRONG_SONG": ProblemClass.WRONG_SONG,
    "DURATION_MISMATCH": ProblemClass.WRONG_VARIANT,
    "NO_REFERENCE": ProblemClass.MISSING_ASSET,
    "MISSING_FILE": ProblemClass.MISSING_ASSET,
    "FALLBACK_TO_ORIGINAL": ProblemClass.WRONG_VERSION,
    "LIVE_SUSPECT": ProblemClass.WRONG_VERSION,
    "WEAK_SIGNAL": ProblemClass.WRONG_VERSION,
}

# Verdicts that mean "this slot is fine" — never open a case for these.
_CLEAN_VERDICTS = frozenset({"OK", "SKIPPED"})


@dataclass(frozen=True)
class SlotSignal:
    """One slot's claim + the identity-gate verdict observed for its audio.

    A thin, ingest-free view: ``verdict`` is ``identity_gate.Verdict.value`` (a
    plain string) and ``reason`` is any extra tag from the reacquire queue
    (e.g. ``needs_search`` / ``media_link_available``). ``layer_role`` may be
    left None and is then derived from ``slot_label`` + ``claimed_stem``.
    """

    slot_label: str
    recording_id: str
    display_name: str = ""
    claimed_version: str = "original"
    claimed_stem: str = "regular"
    claimed_variant: str = "regular"
    verdict: str = "NO_REFERENCE"
    reason: str = ""
    layer_role: LayerRole | None = None
    scrape_urls: tuple[str, ...] = ()


def _claim_of(sig: SlotSignal) -> CaseClaim:
    return CaseClaim(
        recording_id=sig.recording_id,
        display_name=sig.display_name,
        version=normalize_version(sig.claimed_version),
        stem=normalize_stem(sig.claimed_stem),
        variant=normalize_variant(sig.claimed_variant),
        scrape_urls=tuple(sig.scrape_urls),
    )


def slot_is_suspect(sig: SlotSignal) -> bool:
    """True when this slot's identity-gate verdict warrants a case."""
    return sig.verdict not in _CLEAN_VERDICTS


def build_acquisition_case(set_id: str, sig: SlotSignal) -> AcquisitionCase | None:
    """Seam 1 — produce one opened ``AcquisitionCase`` for a suspect slot.

    Returns ``None`` for a clean slot (OK/SKIPPED). NO search is executed: the
    case is opened OPEN with its problem class(es) and a single PENDING attempt
    recording *what triggered* the case (so the trace is honest about provenance)
    — the acquisition executor fills in the real search attempts later.
    """
    if not slot_is_suspect(sig):
        return None

    claim = _claim_of(sig)
    role: LayerRole = sig.layer_role or derive_layer_role(
        sig.slot_label, claimed_stem=claim.stem
    )
    problem = _VERDICT_TO_PROBLEM.get(sig.verdict, ProblemClass.MISSING_ASSET)

    # A stem/variant claim that the default (regular) inventory can't satisfy is
    # a stem/variant problem regardless of the identity verdict — surface both.
    extra: list[ProblemClass] = []
    if claim.stem in ("acappella", "instrumental"):
        extra.append(ProblemClass.WRONG_STEM)
    if claim.variant == "extended":
        extra.append(ProblemClass.WRONG_VARIANT)

    trigger = Attempt(
        action=AttemptAction.MANUAL,
        actor=Actor.AGENT,
        verdict=AttemptVerdict.PENDING,
        checks={"identity_gate_verdict": sig.verdict, "reason": sig.reason},
        notes="case opened by cotrain_seam suspect detector (dry-run)",
    )
    case = AcquisitionCase(
        set_id=set_id,
        slot_label=sig.slot_label,
        layer_role=role,
        claim=claim,
        status=CaseStatus.OPEN,
        attempts=(trigger,),
    )
    return with_problem_classes(case, problem, *extra)


def build_acquisition_cases(
    set_id: str, signals: Sequence[SlotSignal]
) -> list[AcquisitionCase]:
    """Seam 1 (batch) — cases for every suspect slot (clean slots skipped)."""
    out: list[AcquisitionCase] = []
    for sig in signals:
        case = build_acquisition_case(set_id, sig)
        if case is not None:
            out.append(case)
    return out


# ── Seam 2: candidate ref → align-to-mix → verdict → TrainingSignal ───────────


class Band(str, Enum):
    """Fellegi–Sunter decision band on candidate-ref → mix agreement."""

    ACCEPT = "accept"  # independent probes agree tightly → auto-label
    REVIEW = "review"  # some agreement / one strong channel → human ears
    ABSTAIN = "abstain"  # no agreement → do nothing, no signal


@dataclass(frozen=True)
class BandThresholds:
    """Fellegi–Sunter bands over independent-probe agreement.

    The seam accepts ONLY on multi-channel agreement — the co-training
    confirmation-drift guard. A single confident probe can never reach ACCEPT;
    it can at most reach REVIEW. Thresholds are conservative on purpose: this
    labels *training data*, so precision >> recall.

    - ``min_probes`` independent channels must fire (non-abstain) at all.
    - ``accept`` needs ``min_agreeing`` channels within ``accept_tol_s`` of the
      consensus offset AND mean confidence ≥ ``accept_conf``.
    - ``review`` needs ≥2 firing channels within ``review_tol_s`` OR one channel
      ≥ ``review_conf`` — anything weaker abstains.
    """

    min_probes: int = 2
    min_agreeing: int = 2
    accept_tol_s: float = 1.0
    accept_conf: float = 0.55
    review_tol_s: float = 3.0
    review_conf: float = 0.70


DEFAULT_THRESHOLDS = BandThresholds()


@dataclass(frozen=True)
class MixSpan:
    """The mix-side target: which set, which slot, and the played window."""

    set_id: str
    slot_label: str
    set_start_s: float
    span_dur_s: float


@dataclass(frozen=True)
class RefCandidate:
    """A candidate reference proposed by acquisition for a slot."""

    recording_id: str
    source_url: str  # where it came from (yt/sc/discord/…); provenance only
    source_path: str | None = None  # local audio path if already fetched
    display_name: str = ""
    version: str = "original"
    stem: str = "regular"
    variant: str = "regular"
    track_audio_id: int | None = None  # set once it lands a row (dry-run: None)


# A scorer maps (candidate, span) → one AlignmentResult per INDEPENDENT probe.
# Real impl wires fp_probe.fp_offset + stem_placement.place_joint (HuBERT) +
# refine_ref_offsets.detect_offset (chroma). Injected so banding is testable
# with a FAKE scorer offline.
RefMixScorer = Callable[["RefCandidate", "MixSpan"], Sequence[AlignmentResult]]


@dataclass(frozen=True)
class AgreementReport:
    """The measured cross-probe agreement for a (candidate, span)."""

    band: Band
    fired: tuple[AlignmentResult, ...]
    consensus_offset_s: float | None
    n_agreeing: int
    agree_tol_s: float | None
    mean_confidence: float
    detail: str = ""


def _consensus(offsets: Sequence[float]) -> float:
    """Median offset — robust to a single outlier channel."""
    return float(statistics.median(offsets))


def band_agreement(
    results: Sequence[AlignmentResult],
    thresholds: BandThresholds = DEFAULT_THRESHOLDS,
) -> AgreementReport:
    """Fellegi–Sunter banding over independent probe results.

    Distinct probes are identified by ``source`` — two results from the same
    ``source`` count as ONE channel (no double-counting a single sensor into
    agreement). Agreement is measured as the size of the largest cluster of
    firing channels whose offsets fall within a tolerance of their median.
    """
    fired = tuple(r for r in results if not r.abstain)
    # collapse to one result per source (highest confidence wins the channel)
    by_source: dict[str, AlignmentResult] = {}
    for r in fired:
        key = r.source or "unknown"
        cur = by_source.get(key)
        if cur is None or r.confidence > cur.confidence:
            by_source[key] = r
    channels = tuple(by_source.values())

    if len(channels) < 1:
        return AgreementReport(
            Band.ABSTAIN, fired, None, 0, None, 0.0, "no probe fired"
        )

    offsets = [c.offset_s for c in channels]
    consensus = _consensus(offsets)
    confs = [c.confidence for c in channels]

    def cluster(tol: float) -> tuple[int, list[float]]:
        agreeing = [c for c in channels if abs(c.offset_s - consensus) <= tol]
        return len(agreeing), [c.confidence for c in agreeing]

    n_accept, accept_confs = cluster(thresholds.accept_tol_s)
    n_review, _ = cluster(thresholds.review_tol_s)
    mean_conf = (
        statistics.mean(accept_confs) if accept_confs else statistics.mean(confs)
    )

    # ACCEPT: enough independent channels agree tightly + confident.
    if (
        len(channels) >= thresholds.min_probes
        and n_accept >= thresholds.min_agreeing
        and statistics.mean(accept_confs) >= thresholds.accept_conf
    ):
        return AgreementReport(
            Band.ACCEPT,
            fired,
            consensus,
            n_accept,
            thresholds.accept_tol_s,
            statistics.mean(accept_confs),
            f"{n_accept} channels agree within {thresholds.accept_tol_s}s",
        )

    # REVIEW: two loosely-agreeing channels, or one very strong single channel.
    strong_single = max(confs) >= thresholds.review_conf
    if (len(channels) >= 2 and n_review >= 2) or strong_single:
        return AgreementReport(
            Band.REVIEW,
            fired,
            consensus,
            n_review,
            thresholds.review_tol_s,
            mean_conf,
            "loose agreement or one strong channel → human review",
        )

    return AgreementReport(
        Band.ABSTAIN,
        fired,
        consensus,
        n_review,
        thresholds.review_tol_s,
        mean_conf,
        "insufficient independent agreement",
    )


@dataclass(frozen=True)
class ProposedCorrection:
    """A PROPOSED ``track_audio_correction`` ledger entry — NEVER written here.

    Shape mirrors the correction-ledger axes (``version`` | ``stem`` |
    ``variant``) so a downstream executor can apply it after human sign-off. It
    is a *return value* only; producing one has no side effects.
    """

    set_id: str
    slot_label: str
    recording_id: str
    axis: str  # "version" | "stem" | "variant"
    old_value: str
    new_value: str
    source_url: str
    reason: str
    band: Band = Band.ACCEPT
    dry_run: bool = True  # invariant: always True in this module


@dataclass(frozen=True)
class CandidateProposal:
    """Full Seam-2 output for one (candidate, span): band + proposed artefacts.

    ``training_signal`` and ``correction`` are populated ONLY on ACCEPT. On
    REVIEW/ABSTAIN they are None — the seam does not manufacture labels it
    cannot independently corroborate.
    """

    candidate: RefCandidate
    span: MixSpan
    agreement: AgreementReport
    training_signal: TrainingSignal | None = None
    correction: ProposedCorrection | None = None
    resolution: Resolution | None = None


def _correction_for(
    candidate: RefCandidate, span: MixSpan, claim_axes: Mapping[str, str] | None
) -> ProposedCorrection | None:
    """Propose a correction only when the candidate differs from the claim axis.

    ``claim_axes`` is the slot's claimed ``{version,stem,variant}``; if the
    accepted candidate matches on all axes there's nothing to *correct* (it's
    a fill of a missing ref, logged via the TrainingSignal, not the ledger).
    """
    if claim_axes is None:
        return None
    for axis in ("version", "stem", "variant"):
        old = claim_axes.get(axis, "")
        new = getattr(candidate, axis, "")
        if old and new and old != new:
            return ProposedCorrection(
                set_id=span.set_id,
                slot_label=span.slot_label,
                recording_id=candidate.recording_id,
                axis=axis,
                old_value=old,
                new_value=new,
                source_url=candidate.source_url,
                reason="aligner-validated candidate differs from scrape claim",
            )
    return None


def propose_from_candidate(
    candidate: RefCandidate,
    span: MixSpan,
    scorer: RefMixScorer,
    *,
    thresholds: BandThresholds = DEFAULT_THRESHOLDS,
    claim_axes: Mapping[str, str] | None = None,
) -> CandidateProposal:
    """Seam 2 — score a candidate against the mix span and band the verdict.

    Runs the injected ``scorer`` (the aligner-as-validator), bands agreement,
    and on ACCEPT constructs a ``TrainingSignal`` (the labeled ref↔mix-span pair,
    encoded as a preference pair (winner_ref, span_key)) plus a PROPOSED
    correction. Pure: no writes.
    """
    results = scorer(candidate, span)
    report = band_agreement(results, thresholds)

    if report.band is not Band.ACCEPT:
        return CandidateProposal(candidate=candidate, span=span, agreement=report)

    # ACCEPT — build the (proposed) artefacts.
    span_key = f"{span.set_id}:{span.slot_label}@{span.set_start_s:.1f}s"
    winner = candidate.source_path or candidate.source_url
    signal = TrainingSignal(
        negatives=(),
        preference_pairs=((winner, span_key),),
    )
    correction = _correction_for(candidate, span, claim_axes)
    resolution = Resolution(
        track_audio_id=candidate.track_audio_id,
        ref_source="online_candidate",
        source_path=candidate.source_path or candidate.source_url,
        gt_confirmed=False,
    )
    return CandidateProposal(
        candidate=candidate,
        span=span,
        agreement=report,
        training_signal=signal,
        correction=correction,
        resolution=resolution,
    )


# ── DRY-RUN evaluation on GT (precision / recall of the banding) ──────────────


@dataclass
class BandingScore:
    """Confusion of the ACCEPT decision against GT correctness of the candidate."""

    n: int = 0
    accept_correct: int = 0  # ACCEPT and candidate is the GT-correct ref
    accept_wrong: int = 0  # ACCEPT but candidate is NOT GT-correct (the danger)
    review: int = 0
    abstain_correct: int = 0  # ABSTAIN and candidate was wrong (good abstain)
    abstain_missed: int = 0  # ABSTAIN but candidate was right (missed recall)

    @property
    def precision(self) -> float:
        acc = self.accept_correct + self.accept_wrong
        return self.accept_correct / acc if acc else 1.0

    @property
    def recall(self) -> float:
        # over the truly-correct candidates, how many did we ACCEPT?
        pos = self.accept_correct + self.abstain_missed
        return self.accept_correct / pos if pos else 1.0

    def observe(self, band: Band, candidate_is_correct: bool) -> None:
        self.n += 1
        if band is Band.ACCEPT:
            if candidate_is_correct:
                self.accept_correct += 1
            else:
                self.accept_wrong += 1
        elif band is Band.REVIEW:
            self.review += 1
        else:  # ABSTAIN
            if candidate_is_correct:
                self.abstain_missed += 1
            else:
                self.abstain_correct += 1


@dataclass(frozen=True)
class GtCandidateCase:
    """A candidate + span + whether GT says the candidate is the correct ref.

    The dry-run harness evaluates the banding against this label. In real
    operation ``candidate_is_correct`` is unknown; on the BB GT sets it is the
    calibration ground truth (BB11=2nvzlh2k, BB12=1fsnxchk).
    """

    candidate: RefCandidate
    span: MixSpan
    candidate_is_correct: bool
    claim_axes: Mapping[str, str] | None = None


def evaluate_banding(
    cases: Sequence[GtCandidateCase],
    scorer: RefMixScorer,
    *,
    thresholds: BandThresholds = DEFAULT_THRESHOLDS,
) -> tuple[BandingScore, list[CandidateProposal]]:
    """Run Seam 2 over GT-labeled cases; report banding precision/recall + proposals.

    Precision = of what we ACCEPT, how much is GT-correct (co-training safety).
    Recall = of the GT-correct candidates, how many we ACCEPT (yield).
    """
    score = BandingScore()
    proposals: list[CandidateProposal] = []
    for c in cases:
        prop = propose_from_candidate(
            c.candidate, c.span, scorer, thresholds=thresholds, claim_axes=c.claim_axes
        )
        score.observe(prop.agreement.band, c.candidate_is_correct)
        proposals.append(prop)
    return score, proposals


# ── mix-audio resolution (injectable: aligning-dir vs pi-storage corpus) ──────

# The scorer needs the mix-side audio for a routed stem. Two layouts exist:
#   - GT sets: ``~/aligning/<set>/`` (mix_vocals.flac / mix_instrumental.flac /
#     mix.*), resolved by ``capture_votes._mix_audio_path``.
#   - corpus sets: pi-storage ``/mnt/storage/stems/set/<set_audio_id>/`` for the
#     separated stems + the full mix at ``set_audio.path``.
# A MixResolver decouples the scorer from the layout so the SAME probes harvest
# both (the corpus flywheel needs the pi-storage layout — GT sets have no
# analog on the 1,011 downloaded corpus).
MixResolver = Callable[[str], "Path | None"]


def corpus_mix_resolver(mix_full_path: Path, mix_stem_dir: Path) -> MixResolver:
    """Mix-audio resolver for the pi-storage corpus layout.

    ``mix_stem_dir`` = ``/mnt/storage/stems/set/<set_audio_id>/`` (holds
    ``vocals.flac`` / ``instrumental.flac`` from the RoFormer mix-side pass);
    ``mix_full_path`` = the full mix (``set_audio.path``). Mirrors
    ``_mix_audio_path``'s stem routing, so a corpus scorer agrees channel-for-
    channel with the aligning-dir scorer.
    """
    mix_stem_dir = Path(mix_stem_dir)
    mix_full_path = Path(mix_full_path)

    def resolve(stem: str) -> Path | None:
        if stem == "acappella":
            p = mix_stem_dir / "vocals.flac"
            return p if p.is_file() else None
        if stem == "instrumental":
            p = mix_stem_dir / "instrumental.flac"
            return p if p.is_file() else None
        return mix_full_path if mix_full_path.is_file() else None

    return resolve


# ── real-probe scorer (gated no-op when audio absent) ─────────────────────────


# NOTE on offset frames: the banding compares offsets across channels, so every
# channel must be in ONE frame. The absolute→relative conversion (chroma/hubert/
# continuity are absolute ref-time; fp is already the relative diagonal) is the
# single load-bearing convention, owned by ``capture_votes._ABSOLUTE_FRAME_PROBES``
# / ``_run_probe_safe``. This scorer delegates there rather than keeping a second
# copy — getting it wrong makes the absolute-frame probes agree in the WRONG frame
# and outvote fp (the documented catastrophe).


def real_probe_scorer(
    aligning_dir: Path | None = None,
    ref_audio_root: Path | None = None,
    *,
    mix_resolver: MixResolver | None = None,
    compute_mix_fp: Callable[[object], object] | None = None,
) -> RefMixScorer:
    """Build a scorer that runs the REAL harness probes (fp + HuBERT + chroma).

    Delegates to the SAME probe machinery ``capture_votes`` exercises — context
    construction, axis-routed probe selection, per-probe error absorption, and
    (crucially) the ABSOLUTE→RELATIVE offset-frame normalization — rather than
    reimplementing the catastrophic-if-wrong parts. The three probes are
    INDEPENDENT channels (landmark fingerprint / HuBERT self-similarity / chroma
    matched filter) — exactly the cross-channel agreement the banding requires.

    ``aligning_dir`` is the set's ``~/aligning/<set_id>__*`` folder (holds
    ``mix_vocals.flac`` / ``mix_instrumental.flac`` / ``mix.*``, routed per
    candidate stem). ``ref_audio_root`` is a fallback root for ref audio when a
    ``RefCandidate`` carries no ``source_path`` (``<root>/<recording_id>``).

    Gated + robust: absent mix or ref audio → all routed probes abstain (a safe
    no-op offline), and any probe exception is absorbed into an abstain (never
    crashes the scorer). Per-probe confidence calibration to a shared [0,1]
    scale remains the open work — the banding leans first on offset *agreement*
    (frame-normalized consensus) and only secondarily on confidence.
    """
    from pws_aligner.capture_votes import (
        _STEM_TO_PROBES,
        _mix_audio_path,
        _run_probe_safe,
    )
    from pws_aligner.mix_feature_cache import MixFeatureCache

    # Resolve mix-side audio via the injected resolver, or default to the
    # aligning-dir layout (backward compatible with every existing caller).
    resolve_mix: MixResolver
    if mix_resolver is not None:
        resolve_mix = mix_resolver
    elif aligning_dir is not None:
        _adir = aligning_dir

        def resolve_mix(stem: str) -> Path | None:
            return _mix_audio_path(_adir, stem) or _mix_audio_path(_adir, "regular")

    else:
        raise ValueError("real_probe_scorer needs aligning_dir or mix_resolver")

    # One feature cache for the whole scorer: a span's positive + decoys share the
    # identical mix window, so the full-mix fingerprint / chroma / HuBERT compute
    # ONCE per span instead of once per candidate (the ~1 min/case perf fix — see
    # [[project_accept_precision_gate]]). Probe instances built from the cache carry
    # cache-injected mix/ref extractors; a failed import yields a None instance that
    # _run_probe_safe turns into an abstain.
    feat_cache = MixFeatureCache(compute_mix_fp=compute_mix_fp)
    _probe_cache: dict[str, object] = {}

    def _instances(names: tuple[str, ...]) -> dict[str, object]:
        missing = tuple(n for n in names if n not in _probe_cache)
        if missing:
            _probe_cache.update(feat_cache.build_probes(missing))
        return {n: _probe_cache.get(n) for n in names}

    def _scorer(candidate: RefCandidate, span: MixSpan) -> Sequence[AlignmentResult]:
        from alignment.harness.contract import (
            CandidatePool,
            MixContext,
            RefContext,
        )
        from alignment.records import SlotCandidate

        stem = candidate.stem or "regular"
        probe_names = _STEM_TO_PROBES.get(stem, _STEM_TO_PROBES["regular"])

        mix_audio = resolve_mix(stem)
        ref_path = (
            Path(candidate.source_path)
            if candidate.source_path
            else (ref_audio_root / candidate.recording_id if ref_audio_root else None)
        )
        if (
            mix_audio is None
            or not mix_audio.is_file()
            or ref_path is None
            or not ref_path.is_file()
        ):
            return [AlignmentResult.abstained(source=p) for p in probe_names]

        set_start = span.set_start_s
        mix_ctx = MixContext(
            audio_path=mix_audio,
            set_id=span.set_id,
            span_start_s=set_start,
            span_end_s=set_start + span.span_dur_s,
        )
        ref_ctx = RefContext(
            recording_id=candidate.recording_id, audio_path=ref_path, stem=stem
        )
        pool = CandidatePool(
            candidates=(
                SlotCandidate(recording_id=candidate.recording_id, claimed_stem=stem),
            )
        )

        instances = _instances(probe_names)
        results: list[AlignmentResult] = []
        for name in probe_names:
            entry = _run_probe_safe(
                name,
                instances.get(name),
                mix_ctx,
                ref_ctx,
                pool,
                span_set_start_s=set_start,
            )
            # _run_probe_safe already frame-normalized offset_s to RELATIVE; we
            # rebuild an AlignmentResult from its entry (source = probe name).
            results.append(
                AlignmentResult(
                    recording_id=entry["recording_id"],
                    offset_s=None if entry["abstain"] else entry["offset_s"],
                    confidence=entry["confidence"],
                    abstain=entry["abstain"],
                    source=name,
                )
            )
        return results

    return _scorer


# ── CLI (dry-run only) ────────────────────────────────────────────────────────


def _demo_cases() -> list[GtCandidateCase]:
    """A tiny synthetic GT set so ``--dry-run`` runs with no audio.

    Two agreeing channels on the correct candidate (ACCEPT) and a disagreeing
    pair on a wrong candidate (ABSTAIN). Real runs replace this with BB GT
    cases + the real-probe scorer.
    """
    # placeholder set_id — this is a synthetic demo, NOT a real BB GT run (real
    # runs pass BB11 2nvzlh2k / BB12 1fsnxchk cases in; keeps the hardcoded-set_id
    # guardrail honest).
    span = MixSpan(
        set_id="demo-set", slot_label="027", set_start_s=812.0, span_dur_s=40.0
    )
    good = RefCandidate(recording_id="tlpX", source_url="yt://good", display_name="A")
    bad = RefCandidate(recording_id="tlpY", source_url="yt://bad", display_name="B")
    return [
        GtCandidateCase(good, span, candidate_is_correct=True),
        GtCandidateCase(bad, span, candidate_is_correct=False),
    ]


def _demo_scorer(candidate: RefCandidate, span: MixSpan) -> Sequence[AlignmentResult]:
    """Fake scorer for the CLI demo: good → agreement, bad → disagreement."""
    if candidate.recording_id == "tlpX":
        return [
            AlignmentResult(
                recording_id="tlpX", offset_s=12.0, confidence=0.8, source="fp"
            ),
            AlignmentResult(
                recording_id="tlpX", offset_s=12.4, confidence=0.7, source="hubert"
            ),
        ]
    return [
        AlignmentResult(
            recording_id="tlpY", offset_s=12.0, confidence=0.6, source="fp"
        ),
        AlignmentResult(
            recording_id="tlpY", offset_s=99.0, confidence=0.6, source="hubert"
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report banding precision/recall + proposed ledger entries (no writes)",
    )
    ap.add_argument(
        "--accept-tol-s", type=float, default=DEFAULT_THRESHOLDS.accept_tol_s
    )
    ap.add_argument("--accept-conf", type=float, default=DEFAULT_THRESHOLDS.accept_conf)
    args = ap.parse_args(argv)

    if not args.dry_run:
        ap.error("only --dry-run is supported (zero canonical mutation by design)")

    thresholds = BandThresholds(
        accept_tol_s=args.accept_tol_s, accept_conf=args.accept_conf
    )
    cases = _demo_cases()
    score, proposals = evaluate_banding(cases, _demo_scorer, thresholds=thresholds)

    print("=== cotrain_seam DRY-RUN (demo GT; no canonical writes) ===")
    print(
        f"n={score.n}  precision={score.precision:.3f}  recall={score.recall:.3f}  "
        f"accept_correct={score.accept_correct} accept_wrong={score.accept_wrong} "
        f"review={score.review} abstain_correct={score.abstain_correct} "
        f"abstain_missed={score.abstain_missed}"
    )
    print("\n--- proposed artefacts (ACCEPT only) ---")
    for p in proposals:
        tag = p.agreement.band.value.upper()
        print(f"[{tag}] {p.candidate.recording_id} {p.agreement.detail}")
        if p.training_signal is not None:
            print(f"    TrainingSignal: {p.training_signal.preference_pairs}")
        if p.correction is not None:
            print(f"    PROPOSED correction: {p.correction}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

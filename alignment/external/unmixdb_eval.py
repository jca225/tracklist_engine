"""External benchmark: this repo's alignment probes on UnmixDB (vs André 2024 SOTA).

Drives the driver-agnostic harness probes (``FingerprintProbe``, ``ChromaProbe``)
through UnmixDB (Schwarz & Fourer 2018, v1.1) — 2460 synthetic 3-track mixotic
mixes with perfect ground truth — to get a *calibrated standing* against the
published SOTA (André et al. 2024, multi-pass NMF; DTW baseline).

WHAT THE PROBES ARE ASKED TO DO
-------------------------------
For each mix, the source tracks are KNOWN (this is the closed identification
setting André uses — his NMF dictionary is the source tracks). We build a
``CandidatePool`` of the mix's source tracks and, per GT span, run each probe
over the whole mix vs each candidate ref. We score:

* **identification** — did the probe's highest-confidence candidate for the span
  match the GT source? (argmax over candidates by ``confidence``, abstains excluded)
* **alignment / offset error** — |predicted ref-offset − GT ref-offset| at the mix
  origin (mix_t=0). Reported as median + fraction <1s / <5s. This is the analog
  of André's *warp error* (MAE of the mix→ref warp function, in seconds).

GROUND-TRUTH TIME MAP
---------------------
UnmixDB labels give, per track: a ``start`` event timestamp ``start_t`` (mix time,
often negative — where ref t=0 lands) and a ``speed`` factor. The mix→ref map is
affine::

    ref_t(mix_t) = (mix_t - start_t) * speed

Verified against the refsong ``.excerpt40.txt`` cue points (``cueinstart`` equals
``ref_t(fadein_start)``). Our probes report the ref-time placed at the mix origin,
so the comparable GT scalar is ``GT_offset = ref_t(0) = -start_t * speed``.

DIFFICULTY AXES (André's, encoded in the mix filename)
------------------------------------------------------
``set<NNN>mix3-<timewarp>-<effect>-<NN>``:

* **timewarp**: ``none`` (speed 1) / ``resample`` (pitch+tempo) / ``stretch`` (tempo only)
* **effect**:   ``none`` / ``bass`` (boost) / ``compressor`` / ``distortion``

We break results out by both, since that's the axis André reports against.

SOTA REFERENCE (for the honest standing — see ``unmixdb_findings.md``)
---------------------------------------------------------------------
André 2024 assumes sources known and reports **warp error** (MAE of f in s) +
gain error, NOT identification (identification "is well handled by fingerprinting"
— our fp probe is exactly that class of method). Their warp-error medians (Fig. 4)
run ~1 s (none/none) to ~10 s under compression/distortion, log scale, heavy
tails to ~100 s; the DTW baseline is generally *better* on warp than NMF. So the
offset bar to calibrate against is roughly **~1 s median (clean) worsening to
~10 s under effects.**

Run::

    venvs/audio/bin/python -m alignment.external.unmixdb_eval \\
        --unmixdb-root ~/data/unmixdb-v1.1 --sample 200

Findings summary: ``unmixdb_findings.md`` (written next to this file).
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..harness.chroma_probe import ChromaProbe, chroma_result_to_alignment
from ..harness.contract import (
    AlignmentResult,
    CandidatePool,
    MixContext,
    RefContext,
)
from ..refine_ref_offsets import detect_offset
from ..harness.fingerprint_probe import FingerprintProbe
from ..records import SlotCandidate
from .unmixdb import UnmixMix, discover_root, load_mix

# UnmixDB timewarp factors span ~0.75–1.32; the probes' default stretch grid is
# far too narrow (0.98–1.02). Widen it for this dataset (offline eval only — does
# not touch the BB-tuned defaults). Coarse-ish to keep the whole-mix search cheap.
_STRETCH_GRID = (0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20, 1.30)

# filename: set<NNN>mix3-<timewarp>-<effect>-<NN>
_MIX_RE = re.compile(r"^set\d+mix\d+-(?P<warp>[a-z]+)-(?P<effect>[a-z]+)-\d+$")


class TunableChromaProbe(ChromaProbe):
    """ChromaProbe with a configurable abstain floor.

    The stock probe's ``_MIN_PEAK=0.5`` is BB-tuned; on electronic mixotic content
    the chroma peak sits below it, so the stock probe abstains on everything. A
    lower floor lets us also report chroma's RAW placement quality (with the
    genericness caveat). Math is unchanged — same ``detect_offset``."""

    def __init__(self, *, min_peak: float, **kw) -> None:
        super().__init__(**kw)
        self._min_peak = min_peak

    def run(self, mix, ref, candidates) -> AlignmentResult:
        win_f = self._mix_chroma(mix)
        ref_f = self._ref_chroma(ref)
        ref_start_s, peak, stretch = detect_offset(win_f, ref_f, self._stretches)
        return chroma_result_to_alignment(
            ref_start_s,
            peak,
            stretch,
            recording_id=ref.recording_id,
            min_peak=self._min_peak,
            source=self.name,
        )


@dataclass(frozen=True)
class GtSpan:
    """GT for one placed source in a mix (affine mix->ref map)."""

    recording_id: str  # ref stem name (== SlotCandidate.recording_id)
    start_t: float  # mix time where ref t=0 lands
    speed: float
    set_start_s: float  # audible (fadein) start in the mix
    set_end_s: float

    def ref_offset_at_mix0(self) -> float:
        """GT ref-time at mix origin — the scalar the probes' offset_s targets."""
        return -self.start_t * self.speed


@dataclass(frozen=True)
class MixCase:
    mix_id: str
    warp: str
    effect: str
    mix_audio: Path
    candidates: tuple[SlotCandidate, ...]
    refs: dict[str, Path]  # recording_id -> ref audio path
    gt: tuple[GtSpan, ...]


def _recording_id(filename: str) -> str:
    return Path(filename.strip('"')).stem


def parse_gt_spans(labels_path: Path) -> tuple[GtSpan, ...]:
    """Re-parse a labels file for the raw affine map (loader drops start_t/speed)."""
    by_track: dict[int, dict[str, list[tuple[float, float, str]]]] = {}
    with labels_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            start, end, track, label = (
                float(parts[0]),
                float(parts[1]),
                int(parts[2]),
                parts[3].strip(),
            )
            param = parts[4].strip().strip('"') if len(parts) > 4 else ""
            by_track.setdefault(track, {}).setdefault(label, []).append(
                (start, end, param)
            )

    out: list[GtSpan] = []
    for track in sorted(by_track):
        ev = by_track[track]
        starts = ev.get("start", [])
        if not starts:
            continue
        start_t = starts[0][0]
        rid = _recording_id(starts[0][2])
        speed_rows = ev.get("speed", [])
        speed = float(speed_rows[0][2]) if speed_rows else 1.0

        fadeins = ev.get("fadein", [])
        fadeouts = ev.get("fadeout", [])
        stops = ev.get("stop", [])
        set_start = fadeins[0][0] if fadeins else starts[0][0]
        if fadeouts:
            set_end = fadeouts[0][1]
        elif stops:
            set_end = stops[0][0]
        else:
            set_end = starts[0][0] + 20.0
        if set_end <= set_start:
            continue
        out.append(
            GtSpan(
                recording_id=rid,
                start_t=start_t,
                speed=speed,
                set_start_s=set_start,
                set_end_s=set_end,
            )
        )
    return tuple(out)


def _classify(mix_id: str) -> tuple[str, str]:
    m = _MIX_RE.match(mix_id)
    if not m:
        return ("unknown", "unknown")
    return (m.group("warp"), m.group("effect"))


def load_case(labels_path: Path, *, root: Path) -> MixCase | None:
    try:
        mix: UnmixMix = load_mix(labels_path, root=root)
    except (FileNotFoundError, ValueError):
        return None
    gt = parse_gt_spans(labels_path)
    if not gt:
        return None
    refs = {_recording_id(p.name): p for p in mix.track_audio.values()}
    # keep only GT spans whose ref audio actually resolved
    gt = tuple(g for g in gt if g.recording_id in refs)
    if not gt:
        return None
    candidates = tuple(
        SlotCandidate(recording_id=rid, claimed_stem="regular") for rid in refs
    )
    warp, effect = _classify(mix.mix_id)
    return MixCase(
        mix_id=mix.mix_id,
        warp=warp,
        effect=effect,
        mix_audio=mix.mix_audio,
        candidates=candidates,
        refs=refs,
        gt=gt,
    )


def enumerate_cases(root: Path, *, good_only: bool) -> list[MixCase]:
    base = discover_root(root)
    from .unmixdb import good_mix_ids

    good = good_mix_ids(base) if good_only else None
    label_files = sorted(base.rglob("*.labels.txt"))
    cases: list[MixCase] = []
    for lp in label_files:
        mix_id = lp.name.replace(".labels.txt", "")
        if good is not None and mix_id not in good:
            continue
        c = load_case(lp, root=base)
        if c is not None:
            cases.append(c)
    return cases


def stratified_sample(cases: list[MixCase], n: int, *, seed: int) -> list[MixCase]:
    """Balance the sample across (warp, effect) strata, no silent truncation."""
    if n <= 0 or n >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[MixCase]] = defaultdict(list)
    for c in cases:
        buckets[(c.warp, c.effect)].append(c)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    out: list[MixCase] = []
    i = 0
    # round-robin draw across strata until we hit n or exhaust
    while len(out) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


# A GT source is "detected" if the probe placed IT (non-abstain) within this many
# seconds of its true ref-offset. This is the meaningful identification metric in
# UnmixDB's CLOSED setting: every candidate in the pool is genuinely present in the
# mix, so "pick the 1 right source among 3" is ill-posed — the task (as in André)
# is to recover EACH known source's placement. Detection = did we recover this one.
_DETECT_TOL_S = 5.0


@dataclass(frozen=True)
class SpanScore:
    warp: str
    effect: str
    probe: str
    # detected = GT source placed (non-abstain) within _DETECT_TOL_S of GT offset.
    detected: bool
    offset_err: float | None  # |pred - gt| for the GT source; None if abstained


# chroma needs a SHORT mix window sliding over the whole ref (a whole-mix window is
# longer than the ~45s ref → no valid slide → peak 0). So chroma is run per span
# with a fixed window at the GT set_start — i.e. it's given the COARSE placement
# prior ("rough alignment") that fp derives for free from the diagonal. This
# matches André's two-stage decomposition (rough alignment, then precise offset)
# and the repo's own usage (refine_ref_offsets windows the mix at the placed
# position). The asymmetry is documented in the findings.
_CHROMA_WIN_S = 12.0


def _run_fp_over_candidates(
    fp: FingerprintProbe, mix_ctx: MixContext, case: MixCase
) -> dict[str, AlignmentResult]:
    """fp: whole mix vs whole ref, once per candidate (reused across spans).

    Returns ref-time at MIX ORIGIN (mix_t=0) per candidate."""
    out: dict[str, AlignmentResult] = {}
    for cand in case.candidates:
        ref_ctx = RefContext(
            recording_id=cand.recording_id,
            audio_path=case.refs[cand.recording_id],
            stem="regular",
        )
        out[cand.recording_id] = fp.run(
            mix_ctx, ref_ctx, CandidatePool(case.candidates)
        )
    return out


def _run_chroma_for_span(
    chroma: ChromaProbe, case: MixCase, gt: GtSpan
) -> AlignmentResult:
    """chroma: a GT-set_start window of the mix vs the whole GT-source ref.

    Returns ref-time at the WINDOW START (mix_t = set_start)."""
    win_end = min(gt.set_start_s + _CHROMA_WIN_S, gt.set_end_s)
    mix_ctx = MixContext(
        audio_path=case.mix_audio,
        set_id=case.mix_id,
        span_start_s=gt.set_start_s,
        span_end_s=win_end,
    )
    ref_ctx = RefContext(
        recording_id=gt.recording_id,
        audio_path=case.refs[gt.recording_id],
        stem="regular",
    )
    return chroma.run(mix_ctx, ref_ctx, CandidatePool(case.candidates))


def score_case(
    case: MixCase,
    fp: FingerprintProbe,
    chroma: ChromaProbe,
) -> list[SpanScore]:
    mix_ctx = MixContext(audio_path=case.mix_audio, set_id=case.mix_id)
    fp_res = _run_fp_over_candidates(fp, mix_ctx, case)

    scores: list[SpanScore] = []
    for gt in case.gt:
        gt_off_origin = gt.ref_offset_at_mix0()  # fp: ref-time at mix_t=0
        gt_off_window = (gt.set_start_s - gt.start_t) * gt.speed  # chroma: at set_start
        fp_r = fp_res.get(gt.recording_id)
        ch_r = _run_chroma_for_span(chroma, case, gt)
        scores.append(_span_score_from_result("fp", case, gt_off_origin, fp_r))
        scores.append(_span_score_from_result("chroma", case, gt_off_window, ch_r))
        scores.append(_score_fuse(case, gt_off_origin, gt_off_window, fp_r, ch_r))
    return scores


def _span_score_from_result(
    probe_name: str,
    case: MixCase,
    gt_offset: float,
    res: AlignmentResult | None,
) -> SpanScore:
    if res is None or res.abstain:
        return SpanScore(case.warp, case.effect, probe_name, False, None)
    offset_err = abs(res.offset_s - gt_offset)
    detected = offset_err <= _DETECT_TOL_S
    return SpanScore(case.warp, case.effect, probe_name, detected, offset_err)


def _score_fuse(
    case: MixCase,
    gt_off_origin: float,
    gt_off_window: float,
    fp_r: AlignmentResult | None,
    ch_r: AlignmentResult | None,
) -> SpanScore:
    # fuse = higher-confidence non-abstain probe, scored in ITS OWN time frame
    # (fp offset is at mix origin, chroma at the window start).
    live = [
        (r, gt)
        for r, gt in ((fp_r, gt_off_origin), (ch_r, gt_off_window))
        if r is not None and not r.abstain
    ]
    if not live:
        return SpanScore(case.warp, case.effect, "fuse", False, None)
    winner, gt_off = max(live, key=lambda rg: rg[0].confidence)
    return _span_score_from_result("fuse", case, gt_off, winner)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


@dataclass
class Agg:
    n_spans: int = 0
    n_detected: int = 0
    n_abstain: int = 0  # GT source not placed at all (probe declined)
    offsets: list[float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.offsets is None:
            self.offsets = []

    def add(self, s: SpanScore) -> None:
        self.n_spans += 1
        self.n_detected += int(s.detected)
        if s.offset_err is None:
            self.n_abstain += 1
        else:
            self.offsets.append(s.offset_err)

    @property
    def detect_acc(self) -> float | None:
        # detection over ALL GT spans (an abstain is a miss — the source WAS present)
        return self.n_detected / self.n_spans if self.n_spans else None

    @property
    def median_off(self) -> float | None:
        return statistics.median(self.offsets) if self.offsets else None

    def frac_under(self, t: float) -> float | None:
        if not self.offsets:
            return None
        return sum(o < t for o in self.offsets) / len(self.offsets)


def _fmt_pct(x: float | None) -> str:
    return "  -  " if x is None else f"{100 * x:4.0f}%"


def _fmt_s(x: float | None) -> str:
    return "  -  " if x is None else f"{x:6.2f}s"


def report(scores: list[SpanScore], sample_log: str) -> str:
    probes = ("fp", "chroma", "fuse")
    by_probe_effect: dict[tuple[str, str], Agg] = defaultdict(Agg)
    by_probe_warp: dict[tuple[str, str], Agg] = defaultdict(Agg)
    by_probe_overall: dict[str, Agg] = defaultdict(Agg)
    for s in scores:
        by_probe_effect[(s.probe, s.effect)].add(s)
        by_probe_warp[(s.probe, s.warp)].add(s)
        by_probe_overall[s.probe].add(s)

    effects = sorted({s.effect for s in scores})
    warps = sorted({s.warp for s in scores})

    lines: list[str] = []
    lines.append(sample_log)
    lines.append("")
    header = (
        f"{'probe':7} {'stratum':16} {'n':>4} {'detect':>7} "
        f"{'abst':>5} {'med_off':>8} {'<1s':>6} {'<5s':>6}"
    )

    def row(probe: str, name: str, agg: Agg) -> str:
        return (
            f"{probe:7} {name:16} {agg.n_spans:>4} "
            f"{_fmt_pct(agg.detect_acc):>7} {agg.n_abstain:>5} "
            f"{_fmt_s(agg.median_off):>8} "
            f"{_fmt_pct(agg.frac_under(1.0)):>6} {_fmt_pct(agg.frac_under(5.0)):>6}"
        )

    lines.append("=== by EFFECT (André's difficulty axis) ===")
    lines.append(header)
    for probe in probes:
        for eff in effects:
            lines.append(row(probe, f"effect:{eff}", by_probe_effect[(probe, eff)]))
        lines.append("")

    lines.append("=== by TIMEWARP ===")
    lines.append(header)
    for probe in probes:
        for w in warps:
            lines.append(row(probe, f"warp:{w}", by_probe_warp[(probe, w)]))
        lines.append("")

    lines.append("=== OVERALL ===")
    lines.append(header)
    for probe in probes:
        lines.append(row(probe, "ALL", by_probe_overall[probe]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unmixdb-root", type=Path, required=True)
    p.add_argument(
        "--sample",
        type=int,
        default=200,
        help="stratified subset size (0 = all mixes)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--all-mixes",
        action="store_true",
        help="ignore the v1.1 goodmix filter (use every mix)",
    )
    p.add_argument(
        "--chroma-min-peak",
        type=float,
        default=0.2,
        help="chroma abstain floor for this dataset (stock BB floor 0.5 abstains on "
        "all electronic content; 0.2 surfaces raw placement — genericness caveat)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the report table to this path",
    )
    args = p.parse_args(argv)

    t0 = time.time()
    print("enumerating cases (parsing labels + resolving audio) ...", flush=True)
    cases = enumerate_cases(args.unmixdb_root, good_only=not args.all_mixes)
    print(f"  total loadable cases: {len(cases)}", flush=True)

    sample = stratified_sample(cases, args.sample, seed=args.seed)
    strat_counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in sample:
        strat_counts[(c.warp, c.effect)] += 1
    n_spans = sum(len(c.gt) for c in sample)
    sample_log = (
        f"SAMPLE: {len(sample)}/{len(cases)} mixes, {n_spans} GT spans "
        f"(seed={args.seed}, sample-arg={args.sample})\n"
        "  strata (warp,effect -> #mixes): "
        + ", ".join(f"{w}/{e}={n}" for (w, e), n in sorted(strat_counts.items()))
    )
    print(sample_log, flush=True)

    fp = FingerprintProbe(stretches=_STRETCH_GRID)
    chroma = TunableChromaProbe(min_peak=args.chroma_min_peak, stretches=_STRETCH_GRID)

    all_scores: list[SpanScore] = []
    for i, case in enumerate(sample, 1):
        try:
            all_scores.extend(score_case(case, fp, chroma))
        except Exception as exc:  # noqa: BLE001 - keep the batch alive, log loudly
            print(f"  !! {case.mix_id} failed: {exc!r}", file=sys.stderr, flush=True)
        if i % 10 == 0:
            dt = time.time() - t0
            print(
                f"  … {i}/{len(sample)} mixes ({dt:.0f}s, {dt / i:.1f}s/mix)",
                flush=True,
            )

    text = report(all_scores, sample_log)
    print("\n" + text)
    if args.out is not None:
        args.out.write_text(text + "\n")
        print(f"\nwrote {args.out}")
    print(f"\ntotal wall: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

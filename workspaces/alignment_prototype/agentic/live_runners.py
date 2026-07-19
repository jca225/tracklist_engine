"""Live probe runners — the agentic loop calls the real DSP/ML probes on demand.

Replay (``__main__._replay_runner``) re-emits the observations the offline chain
already serialized into ``out/<set>_predicted_timeline.json``; it validates the
LADDER, not the probes. This module is the other half: runners that recompute
each placement channel from audio, so a live run exercises the actual probe
interfaces the north-star aligner ships.

Shape mirrors ``novelty.surprise_runner`` exactly — a factory closes over
per-set precomputed features and returns ``_run(data: dict) -> Observation``.
Each channel is precomputed once in :class:`LiveContext` and INDEPENDENTLY
GUARDED: a missing cache/file empties that one feature (its runner abstains)
without crashing the context. The three channels mirror the corresponding
``infer.py`` blocks (``--fp-placement``, ``--lyrics-placement``,
``--stem-placement``) so live and offline placements agree by construction:

* **fp** — JOINT whole-set landmark-fp decode over tracklist order
  (``decode_placements``); one placement per span, ``ref_start = set_start +
  offset``. Precomputed (the mix is hashed once).
* **lyrics** — JOINT Whisper-word IDF-diagonal monotonic decode over the
  acappella spans; sets BOTH set_start and ref_start. Precomputed.
* **stem_hubert** — PER-SPAN banded HuBERT matched filter over ``mix_vocals``.
  ``mix_hub`` is computed ONCE and held; ref HuBERT is computed on demand and
  cached by recording_id. set_start only (ref_start repeat-ambiguous).
* **mert_decode** / **cue_prior** — read the anchored priors already serialized
  on the span (``probe_proposals.mert_decode`` / ``cue_anchor_s``); the coarse
  MERT decode is NOT re-run.

Runners abstain (``set_start_s=None``) whenever their channel has nothing for a
span — never fabricate a placement.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.agentic.actions import REGISTRY, Runner
from workspaces.alignment_prototype.agentic.belief import Observation, fiber_gate

log = logging.getLogger(__name__)

# HuBERT feature grid — same SR/HOP as the validated path_decode fiber pipeline,
# so fibers computed here are on the frame rate compute_fibers/same_fiber expect.
_FIBER_FPS = 22050 / 512


# ---------------------------------------------------------------------------
# per-set context: precompute the JOINT decodes + hold the on-demand caches
# ---------------------------------------------------------------------------


@dataclass
class LiveContext:
    """Precomputed per-set placement features + on-demand HuBERT caches.

    Every field is optional: an absent one means that channel could not load
    (missing audio / cache / manifest) and its runner abstains everywhere.
    """

    set_id: str
    # JOINT decodes, keyed by slot_label -> (set_start_s, ref_start_s | None)
    fp_by_slot: dict[str, tuple[float, float | None]] = field(default_factory=dict)
    lyrics_by_slot: dict[str, tuple[float, float]] = field(default_factory=dict)
    # HuBERT: mix vocals features computed once + ref path map + per-rid cache
    mix_hub: np.ndarray | None = None
    rid_vocal_path: dict[str, str] = field(default_factory=dict)
    _ref_hub_cache: dict[str, np.ndarray | None] = field(default_factory=dict)
    _ref_fiber_cache: dict[str, tuple | None] = field(default_factory=dict)
    # down-weight instance-ambiguous ref placements (B3 fiber abstain). OPT-IN:
    # on BB11 it demotes ~4 vocal spans auto->suggest at unchanged set_start
    # cleanliness (the ambiguity is in ref_start/which-instance, which the rung
    # metric doesn't measure) — a ref-instance safety lever with no measured
    # set_start win yet, so default off until a ref-instance metric validates it.
    apply_fiber_gate: bool = False
    # which channels are usable (loaded), for build_live_runners / logging
    loaded: set[str] = field(default_factory=set)

    def ref_hub(self, recording_id: str):
        """On-demand HuBERT of a recording's vocal ref, cached by recording_id."""
        from workspaces.alignment_prototype.stem_placement import hubert_of

        if recording_id in self._ref_hub_cache:
            return self._ref_hub_cache[recording_id]
        path = self.rid_vocal_path.get(recording_id)
        feat = hubert_of(path) if path and Path(path).is_file() else None
        self._ref_hub_cache[recording_id] = feat
        return feat

    def ref_fibers(self, recording_id: str):
        """Self-repeat classes of a recording's vocal ref (labels, label_hz), cached.

        HuBERT + RMS silence-gate (never chroma), on the same grid the validated
        fiber pipeline uses — feeds ``fiber_gate`` so a placement landing in a
        repeat class is flagged instance-ambiguous. None when unavailable."""
        if recording_id in self._ref_fiber_cache:
            return self._ref_fiber_cache[recording_id]
        out = None
        feat = self.ref_hub(recording_id)
        path = self.rid_vocal_path.get(recording_id)
        if feat is not None and path:
            try:
                from workspaces.alignment_prototype.ref_fibers import compute_fibers

                out = compute_fibers(feat, _FIBER_FPS, audio_path=path)
            except Exception as e:  # noqa: BLE001 — no fibers => gate no-ops
                log.warning("live fibers: skipped (%s: %s)", type(e).__name__, e)
        self._ref_fiber_cache[recording_id] = out
        return out

    def gate_instance(self, obs: Observation, recording_id: str) -> Observation:
        """Apply the fiber instance-ambiguity gate to a vocal placement (or no-op)."""
        if not self.apply_fiber_gate or obs.ref_start_s is None:
            return obs
        fib = self.ref_fibers(recording_id)
        if fib is None:
            return obs
        labels, label_hz = fib
        return fiber_gate(obs, labels, label_hz)

    @classmethod
    def from_set(
        cls,
        set_id: str,
        spans: list[dict],
        *,
        apply_fiber_gate: bool = False,
    ) -> LiveContext | None:
        """Build the live context from cached MERT + the pulled aligning folder.

        Returns None only if MERT (the mix timebase everything else keys off)
        is unavailable — every downstream channel degrades to abstain rather
        than aborting the context.
        """
        from workspaces.alignment_prototype.mert_store import load_bb12_mert

        ctx = cls(set_id=set_id, apply_fiber_gate=apply_fiber_gate)

        # --- mix timebase (MERT): the one hard dependency -------------------
        from core.result import Err, Ok

        match load_bb12_mert(set_id):
            case Err(msg):
                log.warning(
                    "live: MERT load failed for %s (%s) — no context", set_id, msg
                )
                return None
            case Ok((_sid, mix, _refs)):
                pass
        mix_end = float(mix.end_s[-1])

        from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir

        set_dir = find_aligning_dir(set_id)
        if set_dir is None:
            log.warning(
                "live: no aligning dir for %s — fp/lyrics/hubert abstain", set_id
            )
            return ctx

        cls._load_fp(ctx, set_dir, spans, mix_end)
        cls._load_lyrics(ctx, set_dir, spans, mix_end)
        cls._load_hubert(ctx, set_dir, spans)
        log.info(
            "live context %s: loaded=%s (fp=%d slots, lyrics=%d slots, mix_hub=%s)",
            set_id,
            sorted(ctx.loaded),
            len(ctx.fp_by_slot),
            len(ctx.lyrics_by_slot),
            ctx.mix_hub is not None,
        )
        return ctx

    # -- channel loaders (each fully self-guarded) --------------------------

    @staticmethod
    def _load_fp(
        ctx: LiveContext, set_dir: Path, spans: list[dict], mix_end: float
    ) -> None:
        """JOINT landmark-fp decode over tracklist order (mirrors infer's fp block)."""
        try:
            from workspaces.alignment_prototype.fp_index import FpKey
            from workspaces.alignment_prototype.fp_index import load as fp_load
            from workspaces.alignment_prototype.landmark_fp import constellation, hashes
            from workspaces.alignment_prototype.mix_fp_hits import (
                decode_placements,
                load_mix_mono,
            )

            mix_file = set_dir / "mix.m4a"
            if not mix_file.is_file():
                log.warning("live fp: mix.m4a missing in %s — abstain", set_dir)
                return
            # ref fps in tracklist (span) order — DO NOT sort; None where absent
            fps = [
                fp_load(FpKey(s["recording_id"], "regular"))
                if s.get("recording_id")
                else None
                for s in spans
            ]
            keep = [i for i, fp in enumerate(fps) if fp is not None]
            if not keep:
                log.warning(
                    "live fp: no ref fingerprints cached for %s — abstain", ctx.set_id
                )
                return
            hm = hashes(*constellation(load_mix_mono(mix_file)))
            placements = decode_placements(
                hm,
                [fps[i] for i in keep],
                mix_dur_s=mix_end,
                topk=6,
                gap_s=6.0,
                with_offset=True,
            )
            for r, i in enumerate(keep):
                pl = placements[r]
                if pl is None:
                    continue
                ss, _se, off = pl
                ctx.fp_by_slot[str(spans[i]["slot_label"])] = (
                    float(ss),
                    float(ss + off),
                )
            if ctx.fp_by_slot:
                ctx.loaded.add("fp")
        except Exception as e:  # noqa: BLE001 — any failure guards to abstain
            log.warning("live fp: skipped (%s: %s)", type(e).__name__, e)

    @staticmethod
    def _load_lyrics(
        ctx: LiveContext, set_dir: Path, spans: list[dict], mix_end: float
    ) -> None:
        """JOINT lyrics decode over acappella spans (mirrors infer's lyrics block)."""
        try:
            from workspaces.alignment_prototype.infer import (
                _manifest_by_tid,
                _vocal_ref_path,
            )
            from workspaces.alignment_prototype.lyrics_align import (
                _bigram_times,
                _norm,
                _slot_order,
                candidate_diagonals,
                monotonic_decode,
                transcribe_words,
            )

            ac = [
                s for s in spans if (s.get("claimed_stem") or "regular") == "acappella"
            ]
            mixv = set_dir / "mix_vocals.flac"
            if not ac:
                return
            if not mixv.is_file():
                log.warning("live lyrics: mix_vocals.flac missing — abstain")
                return
            manifest = json.loads((set_dir / "manifest.json").read_text())
            by_tid = _manifest_by_tid(set_dir, ctx.set_id)
            mix_dur = float(manifest.get("mix_duration_s") or 0) or mix_end
            max_slot = (
                max(
                    (
                        _slot_order(s["slot_label"])[0]
                        for s in spans
                        if s.get("slot_label")
                    ),
                    default=1,
                )
                or 1
            )
            mix_bt = _bigram_times(_norm(transcribe_words(mixv)))
            decode_spans, keys = [], []
            for s in ac:
                t = by_tid.get(s.get("recording_id"))
                vpath = _vocal_ref_path(t)
                if not vpath or not Path(vpath).is_file():
                    continue
                cw = transcribe_words(vpath)
                if not cw:
                    continue
                cands = candidate_diagonals(_norm(cw), mix_bt)
                if not cands:
                    continue
                epos = _slot_order(s["slot_label"])[0] / max_slot * mix_dur
                decode_spans.append((cands, epos))
                keys.append(str(s["slot_label"]))
            if not decode_spans:
                log.warning(
                    "live lyrics: no acappella span produced diagonals — abstain"
                )
                return
            for key, (ss, rs) in zip(keys, monotonic_decode(decode_spans)):
                if ss is not None:
                    ctx.lyrics_by_slot[key] = (float(ss), float(rs))
            if ctx.lyrics_by_slot:
                ctx.loaded.add("lyrics")
        except Exception as e:  # noqa: BLE001
            log.warning("live lyrics: skipped (%s: %s)", type(e).__name__, e)

    @staticmethod
    def _load_hubert(ctx: LiveContext, set_dir: Path, spans: list[dict]) -> None:
        """mix_vocals HuBERT (once) + rid->ref-vocal-path map (mirrors infer's stem block)."""
        try:
            from workspaces.alignment_prototype.infer import (
                _manifest_by_tid,
                _vocal_ref_path,
            )
            from workspaces.alignment_prototype.stem_placement import hubert_of

            ac = [
                s for s in spans if (s.get("claimed_stem") or "regular") == "acappella"
            ]
            mixv = set_dir / "mix_vocals.flac"
            if not ac:
                return
            if not mixv.is_file():
                log.warning("live hubert: mix_vocals.flac missing — abstain")
                return
            by_tid = _manifest_by_tid(set_dir, ctx.set_id)
            for s in ac:
                t = by_tid.get(s.get("recording_id"))
                vpath = _vocal_ref_path(t)
                if vpath and Path(vpath).is_file() and s.get("recording_id"):
                    ctx.rid_vocal_path[str(s["recording_id"])] = vpath
            if not ctx.rid_vocal_path:
                log.warning("live hubert: no ref vocal paths resolved — abstain")
                return
            ctx.mix_hub = hubert_of(mixv)
            if ctx.mix_hub is not None:
                ctx.loaded.add("stem_hubert")
        except Exception as e:  # noqa: BLE001
            log.warning("live hubert: skipped (%s: %s)", type(e).__name__, e)


# ---------------------------------------------------------------------------
# runner factories — one per registry action, closing over the context
# ---------------------------------------------------------------------------


def _abstain(probe: str, detail: str) -> Observation:
    return Observation(
        probe=probe,
        set_start_s=None,
        confidence=0.0,
        precision=REGISTRY[probe].precision,
        cost=REGISTRY[probe].cost,
        detail=detail,
    )


def _prior_ss(data: dict) -> float | None:
    """The anchored coarse prior a probe bands around: MERT decode, else cue."""
    mert = (data.get("probe_proposals") or {}).get("mert_decode")
    if mert is not None:
        return float(mert)
    cue = data.get("cue_anchor_s")
    return float(cue) if cue is not None else None


def cue_prior_runner(ctx: LiveContext) -> Runner:
    """The scraped tracklist cue (0.0 on w-rows is fake — abstain there)."""
    spec = REGISTRY["cue_prior"]

    def _run(data: dict) -> Observation:
        cue = data.get("cue_anchor_s")
        slot = str(data.get("slot_label") or "")
        if cue is None or (float(cue) == 0.0 and "w" in slot):
            return _abstain("cue_prior", "no scraped cue (w-row fake 0.0)")
        return Observation(
            probe="cue_prior",
            set_start_s=float(cue),
            confidence=0.6,
            precision=spec.precision,
            cost=spec.cost,
            detail="scraped cue anchor",
        )

    return _run


def mert_runner(ctx: LiveContext) -> Runner:
    """The anchored coarse MERT placement — read from the serialized proposal.

    This is the prior the sequence decode produced; treated as an anchored
    observation, NOT a live re-decode (the head is trained/decoded in infer).
    """
    spec = REGISTRY["mert_decode"]

    def _run(data: dict) -> Observation:
        mert = (data.get("probe_proposals") or {}).get("mert_decode")
        if mert is None:
            return _abstain("mert_decode", "no mert_decode proposal on span")
        return Observation(
            probe="mert_decode",
            set_start_s=float(mert),
            ref_start_s=None,
            confidence=float(data.get("confidence") or 0.7),
            precision=spec.precision,
            cost=spec.cost,
            detail="anchored MERT coarse decode (from proposal)",
        )

    return _run


def fp_runner(ctx: LiveContext) -> Runner:
    """Landmark-fp JOINT placement for this span (precomputed set-wide decode)."""
    spec = REGISTRY["fp"]

    def _run(data: dict) -> Observation:
        slot = str(data.get("slot_label") or "")
        hit = ctx.fp_by_slot.get(slot)
        if hit is None:
            return _abstain("fp", "no fp diagonal for this span")
        ss, rs = hit
        return Observation(
            probe="fp",
            set_start_s=ss,
            ref_start_s=rs,
            confidence=0.8,
            precision=spec.precision,
            cost=spec.cost,
            detail=f"fp decode_placements set_start={ss:.1f}s off={rs - ss:+.1f}s",
        )

    return _run


def lyrics_runner(ctx: LiveContext) -> Runner:
    """Whisper-lyrics JOINT placement (set_start + ref_start) for acappella spans."""
    spec = REGISTRY["lyrics"]

    def _run(data: dict) -> Observation:
        stem = data.get("claimed_stem") or "regular"
        if stem != "acappella":
            return _abstain("lyrics", "not an acappella span")
        slot = str(data.get("slot_label") or "")
        hit = ctx.lyrics_by_slot.get(slot)
        if hit is None:
            return _abstain("lyrics", "lyrics decode abstained on this span")
        ss, rs = hit
        obs = Observation(
            probe="lyrics",
            set_start_s=ss,
            ref_start_s=rs,
            confidence=0.85,
            precision=spec.precision,
            cost=spec.cost,
            detail=f"lyrics monotonic_decode set_start={ss:.1f}s ref_start={rs:.1f}s",
        )
        return ctx.gate_instance(obs, str(data.get("recording_id") or ""))

    return _run


def stem_hubert_runner(ctx: LiveContext) -> Runner:
    """Banded HuBERT matched filter over mix_vocals — PER-SPAN, on demand.

    ref HuBERT is computed on demand and cached by recording_id; the mix vocals
    features were computed once in the context. Refines set_start only.
    """
    spec = REGISTRY["stem_hubert"]

    def _run(data: dict) -> Observation:
        from workspaces.alignment_prototype.stem_placement import place_joint

        stem = data.get("claimed_stem") or "regular"
        if stem != "acappella":
            return _abstain("stem_hubert", "not an acappella span")
        if ctx.mix_hub is None:
            return _abstain("stem_hubert", "no mix_vocals HuBERT")
        rid = str(data.get("recording_id") or "")
        ref_hub = ctx.ref_hub(rid)
        if ref_hub is None:
            return _abstain("stem_hubert", "no ref vocal HuBERT for this recording")
        prior = _prior_ss(data)
        if prior is None:
            return _abstain("stem_hubert", "no prior set_start to band around")
        try:
            se = data.get("set_end_s")
            span_dur = (
                float(se) - float(data.get("set_start_s") or prior) if se else 45.0
            )
            span_dur = max(15.0, span_dur)
            res = place_joint(ctx.mix_hub, ref_hub, prior, span_dur, band_s=90.0)
        except Exception as e:  # noqa: BLE001
            return _abstain("stem_hubert", f"place_joint failed ({type(e).__name__})")
        if res is None:
            return _abstain("stem_hubert", "no dominant HuBERT diagonal")
        ss, rs, peak = res
        obs = Observation(
            probe="stem_hubert",
            set_start_s=float(ss),
            ref_start_s=float(rs),
            confidence=float(min(1.0, max(0.3, peak))),
            precision=spec.precision,
            cost=spec.cost,
            detail=f"HuBERT place_joint set_start={ss:.1f}s peak={peak:.2f}",
        )
        return ctx.gate_instance(obs, rid)

    return _run


_FACTORIES: dict[str, tuple[str, object]] = {
    "cue_prior": ("cue", cue_prior_runner),  # cue always available (span metadata)
    "mert_decode": ("mert", mert_runner),  # always available (serialized proposal)
    "fp": ("fp", fp_runner),
    "lyrics": ("lyrics", lyrics_runner),
    "stem_hubert": ("stem_hubert", stem_hubert_runner),
}


def build_live_runners(ctx: LiveContext) -> dict[str, Runner]:
    """Runners whose channel loaded (cue/mert always; fp/lyrics/hubert if present)."""
    out: dict[str, Runner] = {}
    for name, (loaded_key, factory) in _FACTORIES.items():
        if loaded_key in ("cue", "mert") or loaded_key in ctx.loaded:
            out[name] = factory(ctx)  # type: ignore[operator]
    return out

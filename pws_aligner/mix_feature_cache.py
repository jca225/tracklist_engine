"""Per-run mix/ref feature cache for the ACCEPT-precision harness.

The harness scores a span's GT-correct ref plus N decoys against the SAME mix
window. Every mix-side feature a probe needs — the full-mix landmark fingerprint,
the whole-mix chroma (continuity), the windowed mix chroma (chroma), the windowed
mix HuBERT (acappella) — depends only on the mix path (and, for windowed ones,
the span), NOT on the candidate. Recomputing them per candidate is the ~1 min/case
perf bug that made the rigorous per-axis run ~70 min and blocked the flywheel gate
([[project_accept_precision_gate]]).

This cache memoizes each feature so a span's candidates share one computation. It
does NOT reimplement any DSP — it wraps the probes' own default extractors and
injects the cached versions back into freshly-built probe instances via their
existing seams (``FingerprintProbe(mix_fp=...)``, ``ChromaProbe(mix_chroma=...)``,
etc.). A run touches one or two mix files, so the cache stays tiny; it is scoped
to one ``real_probe_scorer`` and discarded after.

Keying:
  mix_fp / mix_chroma_whole  → mix path                (span-independent)
  mix_chroma_window / hubert → (mix path, start, end)   (per span, shared by candidates)
  ref_chroma / ref_hubert    → ref path                 (shared across recurring decoys)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from alignment.harness.contract import MixContext, RefContext
from alignment.landmark_fp import LandmarkFingerprint

MixCompute = Callable[[MixContext], object]
RefCompute = Callable[[RefContext], object]


# ── default compute fns (wrap the probes' own extractors — no reimplementation) ──


def _default_mix_fp(mix: MixContext) -> LandmarkFingerprint:
    import librosa

    from alignment.landmark_fp import SR, fingerprint_from_audio

    y, _ = librosa.load(str(mix.audio_path), sr=SR, mono=True)
    return fingerprint_from_audio(y)


def _default_mix_chroma_whole(mix: MixContext):
    from alignment.harness.continuity_probe import (
        _default_chroma_whole,
    )

    return _default_chroma_whole(mix.audio_path)


def _default_mix_chroma_window(mix: MixContext):
    from alignment.harness.chroma_probe import _default_chroma

    return _default_chroma(
        mix.audio_path, start_s=mix.span_start_s, end_s=mix.span_end_s
    )


def _default_mix_hubert_window(mix: MixContext):
    from alignment.harness.hubert_probe import _default_hubert

    return _default_hubert(
        mix.audio_path, start_s=mix.span_start_s, end_s=mix.span_end_s
    )


def _default_ref_chroma(ref: RefContext):
    # ChromaProbe and ContinuityProbe both default their ref feature to the
    # whole-ref chroma — one shared computation serves both channels.
    from alignment.harness.chroma_probe import _default_chroma

    return _default_chroma(ref.audio_path)


def _default_ref_hubert(ref: RefContext):
    from alignment.harness.hubert_probe import _default_hubert

    return _default_hubert(ref.audio_path)


def _mix_key(mix: MixContext) -> str:
    return str(mix.audio_path)


def _mix_window_key(mix: MixContext) -> tuple[str, float | None, float | None]:
    return (str(mix.audio_path), mix.span_start_s, mix.span_end_s)


def _ref_key(ref: RefContext) -> str:
    return str(ref.audio_path)


class MixFeatureCache:
    """Memoize mix/ref features so a span's candidates reuse one computation.

    Compute fns are injectable so the memoization contract is unit-testable with
    counting stubs; they default to the probes' real extractors.
    """

    def __init__(
        self,
        *,
        compute_mix_fp: MixCompute | None = None,
        compute_mix_chroma_whole: MixCompute | None = None,
        compute_mix_chroma_window: MixCompute | None = None,
        compute_mix_hubert_window: MixCompute | None = None,
        compute_ref_chroma: RefCompute | None = None,
        compute_ref_hubert: RefCompute | None = None,
    ) -> None:
        self._compute_mix_fp = compute_mix_fp or _default_mix_fp
        self._compute_mix_chroma_whole = (
            compute_mix_chroma_whole or _default_mix_chroma_whole
        )
        self._compute_mix_chroma_window = (
            compute_mix_chroma_window or _default_mix_chroma_window
        )
        self._compute_mix_hubert_window = (
            compute_mix_hubert_window or _default_mix_hubert_window
        )
        self._compute_ref_chroma = compute_ref_chroma or _default_ref_chroma
        self._compute_ref_hubert = compute_ref_hubert or _default_ref_hubert

        self._fp: dict[str, object] = {}
        self._chroma_whole: dict[str, object] = {}
        self._chroma_window: dict[tuple, object] = {}
        self._hubert_window: dict[tuple, object] = {}
        self._ref_chroma: dict[str, object] = {}
        self._ref_hubert: dict[str, object] = {}

    @staticmethod
    def _memo(store: dict, key, compute, ctx):
        if key not in store:
            store[key] = compute(ctx)
        return store[key]

    # ── mix-side (candidate-independent) ─────────────────────────────────────

    def mix_fp(self, mix: MixContext):
        return self._memo(self._fp, _mix_key(mix), self._compute_mix_fp, mix)

    def mix_chroma_whole(self, mix: MixContext):
        return self._memo(
            self._chroma_whole, _mix_key(mix), self._compute_mix_chroma_whole, mix
        )

    def mix_chroma_window(self, mix: MixContext):
        return self._memo(
            self._chroma_window,
            _mix_window_key(mix),
            self._compute_mix_chroma_window,
            mix,
        )

    def mix_hubert_window(self, mix: MixContext):
        return self._memo(
            self._hubert_window,
            _mix_window_key(mix),
            self._compute_mix_hubert_window,
            mix,
        )

    # ── ref-side (shared across recurring decoys) ────────────────────────────

    def ref_chroma(self, ref: RefContext):
        return self._memo(
            self._ref_chroma, _ref_key(ref), self._compute_ref_chroma, ref
        )

    def ref_hubert(self, ref: RefContext):
        return self._memo(
            self._ref_hubert, _ref_key(ref), self._compute_ref_hubert, ref
        )

    # ── probe construction with cached extractors ────────────────────────────

    def build_probes(self, names: tuple[str, ...]) -> dict[str, object]:
        """Instantiate the named probes wired to this cache's extractors.

        Mirrors ``capture_votes._build_probes`` (lazy import, None on failure so
        ``_run_probe_safe`` abstains rather than crashing) but injects the cached
        mix/ref features so the full-mix work happens once per span, not per
        candidate. Unknown / failed-import probes map to None.
        """
        out: dict[str, object] = {}
        for name in names:
            if name in out:
                continue
            try:
                out[name] = self._build_one(name)
            except Exception as exc:  # pragma: no cover - import-env dependent
                import sys

                print(
                    f"WARNING: probe {name!r} failed to build — will abstain: {exc}",
                    file=sys.stderr,
                )
                out[name] = None
        return out

    def _build_one(self, name: str):
        if name == "fp":
            from alignment.harness.fingerprint_probe import (
                FingerprintProbe,
            )

            return FingerprintProbe(mix_fp=self.mix_fp)
        if name == "chroma":
            from alignment.harness.chroma_probe import ChromaProbe

            return ChromaProbe(
                mix_chroma=self.mix_chroma_window, ref_chroma=self.ref_chroma
            )
        if name == "continuity":
            from alignment.harness.continuity_probe import (
                ContinuityProbe,
            )

            return ContinuityProbe(
                mix_chroma_whole=self.mix_chroma_whole, ref_chroma=self.ref_chroma
            )
        if name == "hubert":
            from alignment.harness.hubert_probe import HubertProbe

            return HubertProbe(
                mix_feat=self.mix_hubert_window, ref_feat=self.ref_hubert
            )
        import sys

        print(f"WARNING: unknown probe name {name!r}, skipping", file=sys.stderr)
        return None

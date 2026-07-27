"""Continuous label model: EM over (categorical recording, continuous offset).

Answer to the Phase-1 kill-gate (see CLAUDE.md): Dawid-Skene over 2s offset
bins was refuted because bin-agreement is the wrong granularity for continuous
offsets with heterogeneous probe precisions (fp ~0.2s vs chroma ~seconds) —
genuinely-right probes land in different bins, DS reads pervasive
disagreement, floors every accuracy, NULL wins.

Here identity stays categorical (recordings ARE categorical — that part of DS
was sound). Offsets are continuous: given an identity-correct vote,
    offset ~ inlier * N(mu_span, sigma_probe^2) + (1-inlier) * U(+-W).
EM learns per-probe (accuracy, sigma_s, inlier) with NO ground truth.
Learned 1/sigma^2 IS the neuro/ inverse-variance fusion, made self-supervised.

Frame invariant: all offsets are RELATIVE (ref_start_s - set_start_s), per
capture_votes.py normalization. Never re-convert here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .votes import Vote

_NULL_PRIOR_WEIGHT = 1.05  # mirror label_model.py
_OUTLIER_HALF_WIDTH_S = 240.0
_LOG_UNIFORM = -math.log(2.0 * _OUTLIER_HALF_WIDTH_S)
_UNIFORM = math.exp(_LOG_UNIFORM)
_SIGMA_FLOOR_S = 0.05
_CLAMP_LO, _CLAMP_HI = 0.02, 0.98
_SIGMA_INIT_S = {"fp": 0.5, "hubert": 3.0, "chroma": 8.0, "continuity": 8.0}
_SIGMA_DEFAULT_INIT_S = 5.0
_ACC_INIT = 0.7
_INLIER_INIT = 0.8
_MU_REFINE_ITERS = 3

# --- v4: supervised σ prior + singleton shrinkage (Gate v3 fix) ---------------
# Gate v3 degeneracy: on singleton matches (a recording matched by ONE probe on a
# span) `_fused_mu` returns that lone vote's own offset as μ, so resid ≡ 0 and σ
# floors uniformly.  v4 (a) accumulates σ ONLY from co-voted matches (≥2 probes on
# the same recording), and (b) shrinks σ toward a supervised per-probe prior with a
# small pseudo-count so σ-uninformative probes reflect the prior ordering rather
# than the floor.  Prior means encode the measured precision ordering (fp tight —
# UnmixDB-anchored — through continuity loose); weight is small so real co-vote
# evidence dominates when present.
_SIGMA_PRIOR_S = {"fp": 0.3, "hubert": 3.0, "chroma": 8.0, "continuity": 10.0}
_SIGMA_PRIOR_DEFAULT_S = 5.0
_SIGMA_PRIOR_WEIGHT = 3.0  # pseudo-count (gamma-weight units) for σ shrinkage


@dataclass(frozen=True)
class ProbeNoise:
    accuracy: float  # P(recording vote correct | fires, truth != NULL)
    sigma_s: float  # offset noise std given identity-correct inlier
    inlier: float  # P(inlier | identity-correct)


@dataclass(frozen=True)
class FusedSpan:
    recording_id: str | None  # None = NULL
    offset_s: float  # fused RELATIVE offset (0.0 when NULL)
    confidence: float  # posterior mass on the MAP recording
    n_votes: int  # non-abstaining votes seen


def _log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _clamp(v: float) -> float:
    return min(_CLAMP_HI, max(_CLAMP_LO, v))


class ContinuousLabelModel:
    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-4,
        *,
        sigma_prior: dict[str, float] | None = None,
        sigma_prior_weight: float = _SIGMA_PRIOR_WEIGHT,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self._noise: dict[str, ProbeNoise] = {}
        self._sigma_prior = dict(_SIGMA_PRIOR_S if sigma_prior is None else sigma_prior)
        self._sigma_prior_weight = sigma_prior_weight

    # -- parameters ---------------------------------------------------------

    def probe_noise(self) -> dict[str, ProbeNoise]:
        return dict(self._noise)

    def _sigma_prior_for(self, probe: str) -> float:
        return self._sigma_prior.get(probe, _SIGMA_PRIOR_DEFAULT_S)

    def _noise_for(self, probe: str) -> ProbeNoise:
        got = self._noise.get(probe)
        if got is not None:
            return got
        return ProbeNoise(
            accuracy=_ACC_INIT,
            sigma_s=_SIGMA_INIT_S.get(probe, _SIGMA_DEFAULT_INIT_S),
            inlier=_INLIER_INIT,
        )

    # -- per-span inference -------------------------------------------------

    def _fused_mu(self, match: list[Vote]) -> tuple[float, list[float]]:
        """Robust precision-weighted mean over identity-matching votes.

        Returns (mu, per-vote inlier responsibilities gamma)."""
        weights = [1.0 / self._noise_for(v.probe).sigma_s ** 2 for v in match]
        mu = sum(w * v.offset_s for w, v in zip(weights, match)) / sum(weights)
        gammas = [1.0] * len(match)
        for _ in range(_MU_REFINE_ITERS):
            gammas = []
            for v in match:
                n = self._noise_for(v.probe)
                dens_in = n.inlier * math.exp(
                    _log_normal_pdf(v.offset_s, mu, n.sigma_s)
                )
                gammas.append(dens_in / (dens_in + (1.0 - n.inlier) * _UNIFORM))
            denom = sum(g * w for g, w in zip(gammas, weights))
            if denom <= 0.0:
                break
            mu = (
                sum(g * w * v.offset_s for g, w, v in zip(gammas, weights, match))
                / denom
            )
        return mu, gammas

    def _span_eval(
        self, span_votes: Sequence[Vote]
    ) -> tuple[dict[str | None, float], dict[str, float], dict[str, list[float]]]:
        """Posterior over recordings (incl. NULL) + fused mu + inlier gammas."""
        fired = [v for v in span_votes if not v.abstained and v.recording_id]
        if not fired:
            return {None: 1.0}, {}, {}
        cands = list(dict.fromkeys(v.recording_id for v in fired))  # type: ignore[arg-type]
        k = len(cands)
        logpost: dict[str | None, float] = {}
        mus: dict[str, float] = {}
        gam: dict[str, list[float]] = {}

        lp_null = math.log(_NULL_PRIOR_WEIGHT)
        for v in fired:
            n = self._noise_for(v.probe)
            # Under NULL every recording is "other", so a wrong vote spreads over
            # all k candidates (-log k).  Under a specific-recording hypothesis a
            # wrong vote spreads over only the (k-1) OTHER recordings (-log(k-1)).
            # Using -log(k) here (not -log(k-1)) is therefore correct and consistent.
            lp_null += math.log(1.0 - n.accuracy) - math.log(k) + _LOG_UNIFORM
        logpost[None] = lp_null

        for rec in cands:
            match = [v for v in fired if v.recording_id == rec]
            mu, gammas = self._fused_mu(match)
            mus[rec], gam[rec] = mu, gammas
            lp = 0.0
            for v in fired:
                n = self._noise_for(v.probe)
                if v.recording_id == rec:
                    dens = (
                        n.inlier * math.exp(_log_normal_pdf(v.offset_s, mu, n.sigma_s))
                        + (1.0 - n.inlier) * _UNIFORM
                    )
                    lp += math.log(n.accuracy) + math.log(max(dens, 1e-300))
                else:
                    lp += (
                        math.log(1.0 - n.accuracy)
                        - math.log(max(k - 1, 1))
                        + _LOG_UNIFORM
                    )
            logpost[rec] = lp

        peak = max(logpost.values())
        raw = {r: math.exp(lp - peak) for r, lp in logpost.items()}
        z = sum(raw.values())
        return {r: p / z for r, p in raw.items()}, mus, gam

    # -- EM -----------------------------------------------------------------

    def fit(self, spans: Sequence[Sequence[Vote]]) -> None:
        probes = {v.probe for span in spans for v in span if not v.abstained}
        self._noise = {p: self._noise_for(p) for p in probes}
        for _ in range(self.max_iter):
            num_correct: dict[str, float] = {p: 0.0 for p in probes}
            denom_fire: dict[str, float] = {p: 0.0 for p in probes}
            sq: dict[str, float] = {p: 0.0 for p in probes}
            swt: dict[str, float] = {p: 0.0 for p in probes}  # co-vote σ weight
            gsum: dict[str, float] = {p: 0.0 for p in probes}
            msum: dict[str, float] = {p: 0.0 for p in probes}
            for span in spans:
                q, mus, gam = self._span_eval(span)
                fired = [v for v in span if not v.abstained and v.recording_id]
                non_null = 1.0 - q.get(None, 0.0)
                for v in fired:
                    # assumes one vote per probe per span — guaranteed at capture
                    # time (capture_votes.py emits exactly one Vote per probe per
                    # span); not asserted here.
                    denom_fire[v.probe] += non_null
                    q_r = q.get(v.recording_id, 0.0)
                    num_correct[v.probe] += q_r
                # Accumulate sigma (sq/gsum/msum) per recording, not per outer vote,
                # so each vote's gamma is fetched by its positional index in match.
                # Using match.index(v) in the per-vote loop is a latent bug: two
                # structurally-identical votes (same probe/offset/confidence) would
                # both resolve to index 0 and share the wrong gamma.
                for rec, gammas in gam.items():
                    match = [m for m in fired if m.recording_id == rec]
                    q_r = q.get(rec, 0.0)
                    # A singleton match (one probe on this recording) has μ ≡ its own
                    # offset ⇒ resid ≡ 0 ⇒ σ-uninformative.  Accumulate σ ONLY from
                    # co-voted matches; inlier (gsum/msum) still uses every match.
                    covote = len(match) >= 2
                    for idx, mv in enumerate(match):
                        g = gammas[idx]
                        gsum[mv.probe] += q_r * g
                        msum[mv.probe] += q_r
                        if covote:
                            resid = mv.offset_s - mus[rec]
                            sq[mv.probe] += q_r * g * resid * resid
                            swt[mv.probe] += q_r * g
            delta = 0.0
            for p in probes:
                old = self._noise[p]
                acc = (
                    _clamp(num_correct[p] / denom_fire[p])
                    if denom_fire[p] > 0
                    else old.accuracy
                )
                # Shrink σ² toward the supervised prior with a small pseudo-count.
                # swt[p]=0 (all-singleton probe) ⇒ σ = prior, not the floor; dense
                # co-vote evidence (swt ≫ κ) ⇒ σ → data.
                kappa = self._sigma_prior_weight
                prior_var = self._sigma_prior_for(p) ** 2
                sigma = max(
                    _SIGMA_FLOOR_S,
                    math.sqrt((sq[p] + kappa * prior_var) / (swt[p] + kappa)),
                )
                inl = _clamp(gsum[p] / msum[p]) if msum[p] > 0 else old.inlier
                delta = max(
                    delta,
                    abs(acc - old.accuracy),
                    abs(sigma - old.sigma_s),
                    abs(inl - old.inlier),
                )
                self._noise[p] = ProbeNoise(accuracy=acc, sigma_s=sigma, inlier=inl)
            if delta < self.tol:
                break

    # -- prediction ----------------------------------------------------------

    def predict(self, span_votes: Sequence[Vote]) -> FusedSpan:
        q, mus, _gam = self._span_eval(span_votes)
        n_votes = sum(1 for v in span_votes if not v.abstained and v.recording_id)
        best = max(q, key=lambda r: q[r])
        if best is None:
            return FusedSpan(None, 0.0, q.get(None, 1.0), n_votes)
        return FusedSpan(best, mus[best], q[best], n_votes)

"""Real `AlignBackend` impls — the seam between the runner's plan and the
existing drivers / training harness.

`EndToEndBackend` (compose grain) dispatches an actor binding to its
`EndToEndDriver` and returns the written PredictedTimeline path for the scorer.
Actor→driver map:

    viterbi             -> ClassicalDriver
    agentic             -> AgenticDriver
    trajectory_decoder  -> HybridMlDriver
    trm                 -> loud skeleton (docs/trm_decoder_bakeoff.md)

Driver execution over real audio is integration (needs a pulled set); the
dispatch + LOSO SetContext wiring is unit-tested with a fake driver."""

from __future__ import annotations

from pathlib import Path

from ...drivers.base import SetContext
from ..stages import Pipeline
from .trm_skeleton import SKELETON_MESSAGE

# actor binding -> driver class name in drivers.DRIVERS (lazy: keep the heavy
# drivers package off import until a real run needs it).
_ACTOR_TO_DRIVER: dict[str, str] = {
    "viterbi": "classical",
    "agentic": "agentic",
    "trajectory_decoder": "ml",
}


def _default_driver_for(actor: str):
    if actor == "trm":
        raise NotImplementedError(SKELETON_MESSAGE)
    if actor not in _ACTOR_TO_DRIVER:
        raise KeyError(f"no end-to-end driver for actor {actor!r}")
    from ...drivers import DRIVERS  # heavy — imported only on a real run

    return DRIVERS[_ACTOR_TO_DRIVER[actor]]


class EndToEndBackend:
    """Runs a whole-set pipeline by delegating to its actor's driver."""

    def __init__(self, driver_by_actor: dict[str, type] | None = None) -> None:
        # test seam: inject fakes; production resolves lazily via DRIVERS
        self._injected = driver_by_actor

    def _driver_for(self, actor: str):
        if self._injected is not None:
            if actor == "trm":
                raise NotImplementedError(SKELETON_MESSAGE)
            if actor not in self._injected:
                raise KeyError(f"no injected driver for actor {actor!r}")
            return self._injected[actor]
        return _default_driver_for(actor)

    def run(self, pipeline: Pipeline, ctx) -> Path:
        actor = pipeline.bindings["actor"]
        driver_cls = self._driver_for(actor)
        # LOSO SetContext with preflight OFF — no filesystem I/O in the wiring;
        # the driver's own align_set does its preflight when it really runs.
        sc = SetContext.for_set(
            ctx.set_id, source_set_id=ctx.source_set_id, preflight=False
        )
        return Path(driver_cls().align_set(sc))


__all__ = ["EndToEndBackend"]

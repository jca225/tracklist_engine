"""Agentic aligner harness — the POMDP design's increments 1–3.

Design: docs/pomdp_agentic_aligner_design.md. The aligner's hidden state (true
span alignment) is never observed; probes are noisy observations with costs;
this package makes the decision process explicit:

- ``belief``  — per-span belief: precision-weighted fusion of probe observations
- ``events``  — append-only event log; belief state = replay of the log
- ``actions`` — action registry with cost tags + the measured dominance table
- ``policy``  — the permission ladder (autonomy graded by belief quality)
- ``loop``    — resolve(): cheapest-informative-first, early-commit, budget

Deterministic heuristic brain only (tier 1 of the design). No RL, no LLM
policy — those are later tiers, gated on scale and measured need.
"""

from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
from workspaces.alignment_prototype.agentic.events import Event, EventLog
from workspaces.alignment_prototype.agentic.policy import Mode

__all__ = ["Observation", "SpanBelief", "Event", "EventLog", "Mode"]

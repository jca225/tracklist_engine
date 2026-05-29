"""Core substrate shared across pipeline stages.

Stage-agnostic primitives that every module (scrape, ingest, analysis,
labeling, alignment) may depend on without creating a cross-stage coupling.
Currently the `Result` monad (`result.py`); the canonical DB schema and
shared data models are the next candidates to land here.
"""

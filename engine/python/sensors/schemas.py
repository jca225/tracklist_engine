"""JSON Schema validation at the Rust↔Python process boundary.

Schemas live in the repo-root ``schemas/`` directory (shared with any future
Rust validators). Call ``validate_payload`` before trusting sensor I/O or LF
emissions so agents get precise, field-level failures — the Python analogue of
``cargo check``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

# python/sensors/schemas.py → repo root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMAS_DIR = _REPO_ROOT / "schemas"


class SchemaValidationError(ValueError):
    """Raised when a sensor payload fails JSON Schema validation."""


@lru_cache(maxsize=32)
def load_schema(name: str) -> dict[str, Any]:
    """Load ``schemas/{name}.json`` (without the ``.json`` suffix in *name*)."""
    path = _SCHEMAS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"schema not found: {path}")
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def validate_payload(schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate *payload* against a named schema; return it unchanged on success."""
    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        messages = "; ".join(
            f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors
        )
        raise SchemaValidationError(f"{schema_name}: {messages}")
    return payload

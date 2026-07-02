"""Concrete-syntax layer: gzipped session XML ↔ lxml tree.

The lxml tree is the codec's lossless CST — write-side ops mutate it in place
and this module serializes it back. Never regenerate a session from scratch;
in-place mutation is how fidelity survives and why Live reopens our output.

Uses `lxml` (Py3.14 venv lacks working stdlib expat). Always re-read the
`.als` from disk — never cache a parse across runs.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from lxml import etree


def load_als_xml(als_path: Path) -> etree._Element:
    raw = gzip.decompress(als_path.read_bytes())
    return etree.fromstring(raw)


def dump_als_bytes(root: etree._Element) -> bytes:
    """Serialize a session tree to the uncompressed XML byte form."""
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8")


def save_als_xml(root: etree._Element, als_path: Path) -> None:
    """Gzip a session tree back to disk (the inverse of `load_als_xml`)."""
    als_path.write_bytes(gzip.compress(dump_als_bytes(root)))

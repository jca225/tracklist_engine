"""Unit tests for the registry version-collision fence (fold-in C)."""

from __future__ import annotations

from core.provenance.registry_lint import (
    find_collisions,
    find_undeclared_kinds,
    scan_source,
)

_TXN_A = """
from core.provenance.registry import transformation

@transformation("parse-session", "3", inputs={}, outputs={}, stage="ingest")
def parse_a(x): return x
"""

_TXN_A_DUP = """
from core.provenance.registry import transformation

@transformation("parse-session", "3", inputs={}, outputs={}, stage="ingest")
def parse_b(x): return x
"""

_TXN_A_NEWVER = """
from core.provenance.registry import transformation

@transformation("parse-session", "4", inputs={}, outputs={}, stage="ingest")
def parse_c(x): return x
"""

_ART = """
from core.provenance.registry import artifact_type

@artifact_type("chroma", "application/npy", version=1)
class Chroma: ...
"""


def test_scans_transformation_and_artifact_type():
    d = scan_source(_TXN_A, "a.py") + scan_source(_ART, "b.py")
    idents = {x.identity for x in d}
    assert "txn:parse-session@3" in idents
    assert "kind:chroma" in idents


def test_same_name_version_in_two_files_is_a_collision():
    decls = scan_source(_TXN_A, "a.py") + scan_source(_TXN_A_DUP, "b.py")
    col = find_collisions(decls)
    assert "txn:parse-session@3" in col
    assert len(col["txn:parse-session@3"]) == 2


def test_new_version_is_append_only_not_a_collision():
    # parse-session@3 and parse-session@4 coexist — versioning is additive.
    decls = scan_source(_TXN_A, "a.py") + scan_source(_TXN_A_NEWVER, "b.py")
    assert find_collisions(decls) == {}


def test_single_declaration_is_not_a_collision():
    assert find_collisions(scan_source(_TXN_A, "a.py")) == {}


def test_duplicate_artifact_kind_collides():
    decls = scan_source(_ART, "a.py") + scan_source(_ART, "b.py")
    assert "kind:chroma" in find_collisions(decls)


def test_positional_and_keyword_args_both_parse():
    kw = """
from core.provenance.registry import transformation
@transformation(name="mert-migrate", version="2", inputs={}, outputs={}, stage="analysis")
def f(x): return x
"""
    d = scan_source(kw, "a.py")
    assert d and d[0].identity == "txn:mert-migrate@2"


def test_syntax_error_file_is_skipped_not_crashed():
    assert scan_source("def (:", "broken.py") == []


# --- fold-in E: generated kind-consistency law --------------------------------

_TXN_UNDECLARED = """
from core.provenance.registry import transformation
@transformation("p", "1", inputs={"a": "audio"}, outputs={"b": "not_a_kind"}, stage="s")
def f(x): return x
"""

_ART_SATISFIES = """
from core.provenance.registry import artifact_type
@artifact_type("not_a_kind", "application/json", version=1)
class T: ...
"""

_TXN_BUILTIN = """
from core.provenance.registry import transformation
@transformation("p", "1", inputs={"a": "audio"}, outputs={"b": "feature_blob"}, stage="s")
def f(x): return x
"""


def test_transformation_captures_referenced_kinds():
    d = scan_source(_TXN_UNDECLARED, "a.py")
    assert set(d[0].refs) == {"audio", "not_a_kind"}


def test_undeclared_kind_is_flagged():
    u = find_undeclared_kinds(scan_source(_TXN_UNDECLARED, "a.py"))
    kinds = {x.kind for x in u}
    assert kinds == {"not_a_kind"}  # 'audio' is a built-in ArtifactKind


def test_declaring_the_kind_satisfies_the_reference():
    decls = scan_source(_TXN_UNDECLARED, "a.py") + scan_source(_ART_SATISFIES, "b.py")
    assert find_undeclared_kinds(decls) == []


def test_builtin_kinds_need_no_declaration():
    # both 'audio' and 'feature_blob' are built-in ArtifactKind values
    assert find_undeclared_kinds(scan_source(_TXN_BUILTIN, "a.py")) == []

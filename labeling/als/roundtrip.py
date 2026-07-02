"""Executable round-trip laws — the codec's verification pillar.

Three laws, each returning failure messages as values (empty list = law holds):

- **write locality** — a write op may only touch its own subtree; everything
  else in the session must serialize byte-identically before/after.
- **parse ∘ print = id** — data written by the write side must read back
  equal. For locators this is syntactic; for tempo it is *denotational*
  (the written and re-parsed curves integrate to the same seconds at every
  probe beat — the writer's sentinel init-event makes syntactic equality the
  wrong law).
- **reparse stability** — load → dump → load extracts the identical AST
  (`print ∘ parse = id` on the projection).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from lxml import etree

from labeling.als.cst import dump_als_bytes, load_als_xml
from labeling.als.read import parse_layer_clips
from labeling.als.semantics import parse_master_tempo, tempo_beat_to_sec
from labeling.als.write import write_locators, write_tempo_envelope

# Subtrees the two write ops are allowed to touch.
_TEMPO_TOUCH = (
    ".//MasterTrack//AutomationEnvelopes//AutomationEnvelope/Automation/Events",
    ".//MasterTrack//Tempo/Manual",
)
_LOCATOR_TOUCH = (".//Locators",)


def masked_bytes(root: etree._Element, mask: tuple[str, ...]) -> bytes:
    """Serialized session with the masked subtrees removed (locality probe)."""
    tree = deepcopy(root)
    for xp in mask:
        for el in tree.xpath(xp):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
    return etree.tostring(tree)


def read_locators(root: etree._Element) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    container = root.find(".//Locators/Locators")
    if container is None:
        container = root.find(".//Locators")
    if container is None:
        return out
    for loc in container.findall("Locator"):
        t = loc.find("Time")
        n = loc.find("Name")
        if t is None or t.get("Value") is None:
            continue
        out.append(
            (float(t.get("Value")), (n.get("Value") or "") if n is not None else "")
        )
    return out


def _probe_beats(pts: tuple[tuple[float, float], ...]) -> list[float]:
    beats = {0.0}
    for b, _ in pts:
        beats.update((b, b + 0.5))
    if pts:
        beats.add(pts[-1][0] + 64.0)
    return sorted(beats)


def check_tempo_write(
    root: etree._Element, breakpoints: list[tuple[float, float]]
) -> list[str]:
    """Write a tempo curve onto (a copy of) the session; assert locality and
    denotational parse-back equality."""
    failures: list[str] = []
    work = deepcopy(root)
    before = masked_bytes(work, _TEMPO_TOUCH)
    write_tempo_envelope(work, breakpoints)
    if masked_bytes(work, _TEMPO_TOUCH) != before:
        failures.append("tempo write touched XML outside its envelope/Manual subtree")

    written = tuple(sorted((max(0.0, b), v) for b, v in breakpoints))
    reparsed = parse_master_tempo(work)
    for beat in _probe_beats(written):
        want = tempo_beat_to_sec(written, beat)
        got = tempo_beat_to_sec(reparsed, beat)
        if abs(want - got) > 1e-4:
            failures.append(
                f"tempo denotation diverges at beat {beat}: wrote→{want:.6f}s reparsed→{got:.6f}s"
            )
    return failures


def check_locator_write(
    root: etree._Element, markers: list[tuple[float, str]]
) -> list[str]:
    """Write locators onto (a copy of) the session; assert locality and exact
    parse-back equality."""
    failures: list[str] = []
    work = deepcopy(root)
    before = masked_bytes(work, _LOCATOR_TOUCH)
    write_locators(work, markers)  # type: ignore[arg-type]  # names coerced to str
    if masked_bytes(work, _LOCATOR_TOUCH) != before:
        failures.append("locator write touched XML outside the Locators block")

    want = sorted((max(0.0, b), str(n)) for b, n in markers)
    got = read_locators(work)
    if len(want) != len(got):
        failures.append(f"locator count: wrote {len(want)}, reparsed {len(got)}")
    else:
        for (wb, wn), (gb, gn) in zip(want, got):
            if abs(wb - gb) > 1e-6 or wn != gn:
                failures.append(
                    f"locator mismatch: wrote ({wb}, {wn!r}), reparsed ({gb}, {gn!r})"
                )
    return failures


def check_reparse_stable(als_path: Path) -> list[str]:
    """load → dump → load must extract the identical AST projection."""
    failures: list[str] = []
    root1 = load_als_xml(als_path)
    root2 = etree.fromstring(dump_als_bytes(root1))

    clips1, clips2 = parse_layer_clips(root1), parse_layer_clips(root2)
    if clips1 != clips2:
        failures.append(
            f"parse_layer_clips unstable across dump/reload ({len(clips1)} vs {len(clips2)} clips)"
        )
    if parse_master_tempo(root1) != parse_master_tempo(root2):
        failures.append("parse_master_tempo unstable across dump/reload")
    return failures

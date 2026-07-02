"""Write side of the `.als` codec: session mutation (seeding primitives).

These mutate a parsed session tree in place — the seeder
(`workspaces/alignment_prototype/seed_als_from_timeline.py`) gzips the result
back to disk. Per the `.als` crash history, never allocate new PointeeIds;
reuse the template's existing automation targets.
"""

from __future__ import annotations

from lxml import etree

# Ableton's "before-start" sentinel time for an automation's initial value.
_ENV_INIT_TIME = "-63072000"


def write_tempo_envelope(
    root: etree._Element, breakpoints: list[tuple[float, float]]
) -> int:
    """Populate the MasterTrack tempo AutomationEnvelope with (beat, bpm) points.

    Reuses the template's *existing* tempo AutomationTarget + AutomationEnvelope
    (matched by PointeeId), so no new PointeeId is allocated — this is deliberately
    clear of the deep-copy id-duplication path that crashes Live (the seeder's
    strip_automation/renumber_pointee_ids machinery touches copied audio tracks,
    never the MasterTrack). `breakpoints` are (beat_time, bpm), any order; beats
    are arrangement musical time. Returns the number of points written; raises if
    the tempo target/envelope is missing from the template.
    """
    tempo = root.find(".//MasterTrack//Tempo")
    if tempo is None:
        raise ValueError("no MasterTrack/Tempo in document")
    at = tempo.find("AutomationTarget")
    if at is None or at.get("Id") is None:
        raise ValueError("Tempo has no AutomationTarget Id")
    target_id = at.get("Id")
    env = None
    for ae in root.findall(
        ".//MasterTrack//AutomationEnvelopes/Envelopes/AutomationEnvelope"
    ):
        pid = ae.find("EnvelopeTarget/PointeeId")
        if pid is not None and pid.get("Value") == target_id:
            env = ae
            break
    if env is None:
        raise ValueError(f"no tempo AutomationEnvelope (PointeeId={target_id})")
    events = env.find("Automation/Events")
    if events is None:
        raise ValueError("tempo envelope has no Automation/Events")
    for child in list(events):
        events.remove(child)
    pts = sorted(breakpoints)
    first_bpm = pts[0][1] if pts else 120.0
    # leading initial-value event, then one FloatEvent per breakpoint
    init = etree.SubElement(events, "FloatEvent")
    init.set("Id", "0")
    init.set("Time", _ENV_INIT_TIME)
    init.set("Value", f"{first_bpm:.6f}")
    for i, (beat, bpm) in enumerate(pts, start=1):
        fe = etree.SubElement(events, "FloatEvent")
        fe.set("Id", str(i))
        fe.set("Time", f"{max(0.0, beat):.6f}")
        fe.set("Value", f"{bpm:.6f}")
    manual = tempo.find("Manual")
    if manual is not None:
        manual.set("Value", f"{first_bpm:.6f}")
    return len(pts)


def write_locators(root: etree._Element, markers: list[tuple[float, float]]) -> int:
    """Replace the arrangement Locators (markers) with `(beat_time, name)` pairs.

    Clones the document's own <Locator> element when present so the schema matches
    Live exactly (versions differ). Times are arrangement beats. Returns count.
    `name` is a float here only by signature convenience — callers pass
    (beat, label_str); we coerce label to str.
    """
    container = root.find(".//Locators/Locators")
    if container is None:
        outer = root.find(".//Locators")
        if outer is None:
            raise ValueError("no Locators block in document")
        container = outer
    existing = container.findall("Locator")
    proto = existing[0] if existing else None
    for loc in existing:
        container.remove(loc)
    from copy import deepcopy

    for i, (beat, name) in enumerate(sorted(markers, key=lambda m: m[0])):
        if proto is not None:
            el = deepcopy(proto)
        else:
            el = etree.SubElement(container, "Locator")
            for tag in ("LomId", "Time", "Name", "Annotation", "IsSongStart"):
                etree.SubElement(el, tag)
            container.remove(el)
        el.set("Id", str(i))

        def _set(tag: str, val: str, _el=el) -> None:
            e = _el.find(tag)
            if e is None:
                e = etree.SubElement(_el, tag)
            e.set("Value", val)

        _set("Time", f"{max(0.0, beat):.6f}")
        _set("Name", str(name))
        _set("IsSongStart", "false")
        lom = el.find("LomId")
        if lom is not None:
            lom.set("Value", "0")
        container.append(el)
    return len(markers)

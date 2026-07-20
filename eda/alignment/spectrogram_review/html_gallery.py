"""Self-contained HTML gallery — plain-English, grouped by what went wrong/right."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from eda.alignment.spectrogram_review.labels import (
    accuracy_plain,
    fmt_clock,
    fmt_delta,
    group_blurb,
    group_title,
    one_liner,
    place_delta_s,
    ref_delta_s,
    stem_plain,
)
from eda.alignment.spectrogram_review.spans import SpanCard

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Where the aligner got it right / wrong</title>
<style>
  :root {
    --bg: #f4f2ee;
    --panel: #ffffff;
    --text: #1a1a1a;
    --muted: #5c5c5c;
    --ok: #1b7a3d;
    --ok-bg: #e6f5eb;
    --bad: #b42318;
    --bad-bg: #fdecea;
    --line: #e0dcd4;
    --warn: #9a6700;
    --warn-bg: #fff6e0;
    --chip: #f0ece4;
    --truth: #1b7a3d;
    --guess: #a21caf;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    background: var(--bg); color: var(--text);
    line-height: 1.45;
  }
  header {
    position: sticky; top: 0; z-index: 10;
    background: rgba(244,242,238,0.96);
    border-bottom: 1px solid var(--line);
    padding: 16px 20px 14px;
    backdrop-filter: blur(8px);
  }
  h1 { margin: 0 0 4px; font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
  .subtitle { margin: 0 0 12px; color: var(--muted); font-size: 14px; max-width: 52rem; }
  .howto {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px; margin-bottom: 12px;
  }
  .howto .step {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 10px 12px; font-size: 13px;
  }
  .howto .step strong { display: block; margin-bottom: 2px; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted);
    font-family: ui-sans-serif, system-ui, sans-serif; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    margin-right: 4px; vertical-align: middle; }
  .swatch.truth { background: var(--truth); }
  .swatch.guess { background: var(--guess); }
  .controls {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
  }
  .controls label { color: var(--muted); display: flex; gap: 6px; align-items: center; }
  select, input[type="search"] {
    background: var(--panel); color: var(--text);
    border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px;
    font: inherit;
  }
  #count { margin-left: auto; color: var(--muted); font-size: 12px; }
  .weak {
    margin-top: 12px; padding: 12px 14px; background: var(--warn-bg);
    border: 1px solid #efd89a; border-radius: 8px;
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
  }
  .weak h2 { margin: 0 0 8px; font-size: 13px; color: var(--warn);
    text-transform: uppercase; letter-spacing: 0.04em; }
  .weak-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 8px;
  }
  .weak-card {
    background: #fff; border: 1px solid #efd89a; border-radius: 6px;
    padding: 8px 10px;
  }
  .weak-card .k { font-weight: 700; color: var(--text); }
  .weak-card .v { color: var(--muted); font-size: 12px; margin-top: 2px; }
  section.group { padding: 8px 20px 4px; max-width: 1400px; margin: 0 auto; }
  section.group.hidden { display: none; }
  section.group .section-head { margin: 28px 0 6px; }
  section.group h2 {
    margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.01em;
  }
  section.group .section-blurb {
    margin: 4px 0 12px; color: var(--muted); font-size: 14px; max-width: 40rem;
  }
  section.group h2 .n {
    color: var(--muted); font-weight: 500; font-size: 14px;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  section.group.failures h2 { color: var(--bad); }
  section.group.successes h2 { color: var(--ok); }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
    gap: 16px; padding-bottom: 12px;
  }
  .card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 1px 0 rgba(0,0,0,0.03);
  }
  .card.success { border-top: 4px solid var(--ok); }
  .card.failure { border-top: 4px solid var(--bad); }
  .card.hidden { display: none; }
  .card img {
    width: 100%; aspect-ratio: 2 / 1; object-fit: contain;
    display: block; background: #111;
  }
  .meta {
    padding: 12px 14px 14px;
    font-family: ui-sans-serif, system-ui, sans-serif; font-size: 13px;
  }
  .meta .title-row {
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
    margin-bottom: 6px;
  }
  .meta h3 {
    margin: 0; font-size: 15px; font-weight: 700;
    font-family: "Iowan Old Style", Palatino, Georgia, serif;
    flex: 1 1 12rem;
  }
  .meta .blurb { color: var(--text); margin: 0 0 10px; font-size: 14px; }
  .compare {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px;
  }
  .compare .box {
    background: var(--chip); border-radius: 6px; padding: 8px 10px;
  }
  .compare .box .lab {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); margin-bottom: 2px;
  }
  .compare .box .val { font-weight: 600; font-size: 13px; }
  .compare .box .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .pill {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600;
  }
  .pill.ok { background: var(--ok-bg); color: var(--ok); }
  .pill.bad { background: var(--bad-bg); color: var(--bad); }
  .pill.axis { background: var(--chip); color: var(--muted); }
  .foot { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Where the aligner got it right / wrong</h1>
  <p class="subtitle">Each picture is a short spectrogram window. Green boxes are the
  <em>truth</em> from human labeling. Magenta boxes are <em>our guess</em>.
  Left = DJ mix timeline. Right = original song.</p>
  <div class="howto">
    <div class="step"><strong>How to read a card</strong>
      Compare green vs magenta. If they overlap, we were close. If they’re far apart, we missed.</div>
    <div class="step"><strong>Colors</strong>
      <span class="swatch truth"></span>Truth (human GT) &nbsp;
      <span class="swatch guess"></span>Our guess</div>
    <div class="step"><strong>Green / red top bar</strong>
      Green = good enough match. Red = this clip mostly failed.</div>
  </div>
  <div class="controls">
    <label>Show
      <select id="f-outcome">
        <option value="all">Everything</option>
        <option value="failure">Only misses</option>
        <option value="success">Only hits</option>
      </select>
    </label>
    <label>Kind of miss / hit
      <select id="f-group">
        <option value="all">All kinds</option>
        __GROUP_OPTS__
      </select>
    </label>
    <label>Track type
      <select id="f-stem">
        <option value="all">All types</option>
        <option value="acappella">Vocals / acapella</option>
        <option value="instrumental">Instrumental</option>
        <option value="regular">Full track</option>
      </select>
    </label>
    <label>Set
      <select id="f-set">
        <option value="all">All sets</option>
        __SET_OPTS__
      </select>
    </label>
    <label>Search <input id="f-q" type="search" placeholder="song name…"/></label>
    <span id="count"></span>
  </div>
  __WEAK__
</header>
__SECTIONS__
<script>
const cards = [...document.querySelectorAll('.card')];
const sections = [...document.querySelectorAll('section.group')];
const countEl = document.getElementById('count');
function apply() {
  const outcome = document.getElementById('f-outcome').value;
  const group = document.getElementById('f-group').value;
  const stem = document.getElementById('f-stem').value;
  const setId = document.getElementById('f-set').value;
  const q = document.getElementById('f-q').value.trim().toLowerCase();
  let n = 0;
  for (const c of cards) {
    const okOutcome = outcome === 'all' || c.dataset.outcome === outcome;
    const okGroup = group === 'all' || c.dataset.group === group;
    const okStem = stem === 'all' || c.dataset.stem === stem;
    const okSet = setId === 'all' || c.dataset.setId === setId;
    const hay = (c.dataset.search || '');
    const okQ = !q || hay.includes(q);
    const show = okOutcome && okGroup && okStem && okSet && okQ;
    c.classList.toggle('hidden', !show);
    if (show) n++;
  }
  for (const s of sections) {
    const visible = [...s.querySelectorAll('.card')].some(c => !c.classList.contains('hidden'));
    s.classList.toggle('hidden', !visible);
  }
  countEl.textContent = 'Showing ' + n + ' of ' + cards.length;
}
for (const id of ['f-outcome','f-group','f-stem','f-set','f-q']) {
  document.getElementById(id).addEventListener('input', apply);
}
apply();
</script>
</body>
</html>
"""


def _weak_html(weak: dict[str, Any] | None) -> str:
    if not weak:
        return ""
    cards_html: list[str] = []
    for s in weak.get("worst_stems") or []:
        cards_html.append(
            '<div class="weak-card">'
            f'<div class="k">Hardest track type: {html.escape(stem_plain(s["key"]))}</div>'
            f'<div class="v">Only {s["success_rate"]:.0%} of these clips work · '
            f"about {s['sec_lost'] / 60:.0f} minutes of music still wrong "
            f"({s['n']} clips)</div></div>"
        )
    for s in (weak.get("worst_causes") or [])[:2]:
        title = group_title(s["key"], success=False)
        cards_html.append(
            '<div class="weak-card">'
            f'<div class="k">Biggest miss type: {html.escape(title)}</div>'
            f'<div class="v">Costs ~{s["sec_lost"] / 60:.0f} minutes · '
            f"{s['n']} clips</div></div>"
        )
    return (
        '<div class="weak">'
        "<h2>Where we’re weakest overall</h2>"
        f'<p style="margin:0 0 8px;color:var(--muted)">'
        f"Across {weak['n']} scored clips: "
        f"{weak['n_ok']} mostly right, {weak['n_fail']} mostly wrong."
        f"</p>"
        f'<div class="weak-grid">{"".join(cards_html)}</div>'
        "</div>"
    )


def _card_html(
    card: SpanCard,
    img_rel: str,
    *,
    has_source: bool,
) -> str:
    outcome = "success" if card.success else "failure"
    pill = "ok" if card.success else "bad"
    pill_txt = "Hit" if card.success else "Miss"
    search = " ".join(
        [
            card.name,
            card.slot,
            card.cause,
            card.group_key,
            card.gt_stem,
            stem_plain(card.gt_stem),
            group_title(card.group_key, success=card.success),
            one_liner(card),
            outcome,
        ]
    ).lower()
    mix_delta = fmt_delta(place_delta_s(card))
    src_delta = fmt_delta(ref_delta_s(card))
    src_note = (
        ""
        if has_source
        else '<div class="foot">Original-song audio missing for this clip — mix only.</div>'
    )
    return (
        f'<article class="card {outcome}" '
        f'data-outcome="{outcome}" data-group="{html.escape(card.group_key)}" '
        f'data-cause="{html.escape(card.cause)}" '
        f'data-stem="{html.escape(card.gt_stem)}" '
        f'data-set-id="{html.escape(card.set_id)}" '
        f'data-search="{html.escape(search)}">'
        f'<img src="{html.escape(img_rel)}" alt="{html.escape(card.name)}" loading="lazy"/>'
        f'<div class="meta">'
        f'<div class="title-row">'
        f'<span class="pill {pill}">{pill_txt}</span>'
        f'<span class="pill axis">{html.escape(stem_plain(card.gt_stem))}</span>'
        f"<h3>{html.escape(card.name)}</h3>"
        f"</div>"
        f'<p class="blurb">{html.escape(one_liner(card))}</p>'
        f'<div class="compare">'
        f'<div class="box"><div class="lab">In the DJ mix</div>'
        f'<div class="val">Our guess is {html.escape(mix_delta)}</div>'
        f'<div class="sub">Truth {fmt_clock(card.gt_set_start_s)} → '
        f"guess {fmt_clock(card.pred_set_start_s)}</div></div>"
        f'<div class="box"><div class="lab">In the original song</div>'
        f'<div class="val">Our guess is {html.escape(src_delta)}</div>'
        f'<div class="sub">Truth {fmt_clock(card.gt_ref_start_s)} → '
        f"guess {fmt_clock(card.pred_ref_start_s)}</div></div>"
        f"</div>"
        f'<div class="foot">{html.escape(accuracy_plain(card))} · '
        f"slot {html.escape(card.slot)} · {html.escape(card.set)}</div>"
        f"{src_note}"
        f"</div></article>"
    )


def write_gallery(
    out_html: Path,
    cards: Sequence[SpanCard],
    *,
    img_names: Sequence[str],
    has_source: Sequence[bool] | None = None,
    weak: dict[str, Any] | None = None,
) -> None:
    if len(cards) != len(img_names):
        raise ValueError("cards and img_names length mismatch")
    if has_source is None:
        has_source = [True] * len(cards)
    if len(has_source) != len(cards):
        raise ValueError("has_source length mismatch")

    groups: dict[str, list[tuple[SpanCard, str, bool]]] = defaultdict(list)
    for card, img, src in zip(cards, img_names, has_source):
        groups[card.group_key].append((card, img, src))

    fail_keys = sorted(
        (k for k in groups if any(not c.success for c, _, _ in groups[k])),
        key=lambda k: sum(c.sec_lost for c, _, _ in groups[k]),
        reverse=True,
    )
    ok_keys = sorted(
        (k for k in groups if all(c.success for c, _, _ in groups[k])),
        key=lambda k: k,
    )
    other = [k for k in groups if k not in fail_keys and k not in ok_keys]
    ordered_keys = fail_keys + other + ok_keys

    sections: list[str] = []
    for key in ordered_keys:
        items = sort_group(groups[key])
        is_ok = all(c.success for c, _, _ in items)
        outcome_cls = "successes" if is_ok else "failures"
        title = group_title(key, success=is_ok)
        blurb = group_blurb(key, success=is_ok)
        prefix = "Hits" if is_ok else "Misses"
        body = "\n".join(_card_html(c, img, has_source=src) for c, img, src in items)
        sections.append(
            f'<section class="group {outcome_cls}" data-group="{html.escape(key)}">'
            f'<div class="section-head">'
            f"<h2>{prefix}: {html.escape(title)} "
            f'<span class="n">· {len(items)} clips</span></h2>'
            f'<p class="section-blurb">{html.escape(blurb)}</p>'
            f"</div>"
            f'<div class="grid">{body}</div></section>'
        )

    group_opts = "\n".join(
        f'<option value="{html.escape(k)}">'
        f"{html.escape(group_title(k, success=(k in ok_keys)))}"
        f"</option>"
        for k in ordered_keys
    )
    sets = sorted({c.set_id for c in cards})
    set_labels = {c.set_id: c.set for c in cards}
    set_opts = "\n".join(
        f'<option value="{html.escape(s)}">{html.escape(set_labels.get(s, s))}</option>'
        for s in sets
    )
    text = (
        _TEMPLATE.replace("__GROUP_OPTS__", group_opts)
        .replace("__SET_OPTS__", set_opts)
        .replace("__WEAK__", _weak_html(weak))
        .replace("__SECTIONS__", "\n".join(sections))
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(text)
    meta_path = out_html.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {"weak_axes": weak, "cards": [c.to_meta() for c in cards]},
            indent=2,
        )
    )


def sort_group(
    items: list[tuple[SpanCard, str, bool]],
) -> list[tuple[SpanCard, str, bool]]:
    return sorted(items, key=lambda t: t[0].sec_lost, reverse=True)

"""Streamlit UI — browse DJ sets, tokens, HTMLs, and download coverage.

Run from the repo root:
    venvs/audio/bin/streamlit run ui/app.py

Pages:
    1. Overview — corpus-wide stats (mirror of eda.ipynb highlights)
    2. Set browser — filterable table of all sets with key metrics
    3. Set detail — token table, platform flags, raw HTML viewer, cue-recovery probe
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (_REPO_ROOT, _REPO_ROOT / "data_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from big_bootie import (
    extract_cue_points_from_html,
    load_big_bootie_rows,
    load_big_bootie_sets,
    load_big_bootie_track_media_links,
    tokenize_rows,
)
from row_tokens import classify_row


DB_PATH = _REPO_ROOT / "data" / "db" / "music_database.db"
HTML_DIR = _REPO_ROOT / "data" / "html"

st.set_page_config(
    page_title="Tracklist Engine",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- DAW-inspired theme (Ableton × Serato × 1001Tracklists) -----------
# All chrome overrides live here. Page code stays unchanged; it just inherits
# the palette + typography. Everything is scoped with st-* selectors so
# Streamlit version upgrades that rename internal classes fail soft (the
# page still renders, just with the default Streamlit look for that widget).
_THEME_CSS = """
<style>
  :root {
    --bg-0: #0f0f0f;       /* app background — darker than Ableton's default */
    --bg-1: #181818;       /* panels, cards, sidebar */
    --bg-2: #222;          /* hover / active surfaces */
    --bg-3: #2a2a2a;       /* inputs, table rows */
    --line: #2e2e2e;       /* hairline borders */
    --line-2: #3a3a3a;     /* stronger borders */
    --text: #ececec;
    --text-dim: #9a9a9a;
    --text-faint: #6a6a6a;
    --accent: #1a8fa0;     /* deep teal — blue-leaning blue-green */
    --accent-2: #1ec8ff;   /* Ableton cyan (secondary) */
    --ok: #3ea372;
    --warn: #d6a64a;
    --err: #e05a4f;
    --mono: ui-monospace, "JetBrains Mono", "SF Mono", "Menlo", monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
  }

  html, body, [class*="css"] {
    background: var(--bg-0) !important;
    color: var(--text);
    font-family: var(--sans);
    font-feature-settings: "ss01", "cv01", "tnum";
  }

  /* App shell ---------------------------------------------------------- */
  [data-testid="stAppViewContainer"] { background: var(--bg-0); }
  [data-testid="stHeader"]           { background: transparent; }
  .block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1600px; }

  /* Sidebar — Ableton browser feel ------------------------------------ */
  [data-testid="stSidebar"] {
    background: var(--bg-1);
    border-right: 1px solid var(--line);
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: .5rem; }
  [data-testid="stSidebar"] [data-testid="stSidebarNav"] { background: transparent; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    font-family: var(--mono);
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
    margin: 1rem 0 .35rem 0 !important;
  }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stRadio label { color: var(--text) !important; }
  [data-testid="stSidebar"] .stCaption,
  [data-testid="stSidebar"] small { color: var(--text-faint) !important; font-family: var(--mono); font-size: 10.5px; }

  /* Headings — hard typographic hierarchy ----------------------------- */
  h1, .stMarkdown h1 {
    font-family: var(--sans);
    font-weight: 600;
    font-size: 1.55rem !important;
    letter-spacing: -0.01em;
    color: var(--text);
    padding-bottom: .5rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1rem !important;
  }
  h2, .stMarkdown h2 {
    font-family: var(--mono);
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-top: 1.5rem !important;
    margin-bottom: .6rem !important;
  }
  h3, .stMarkdown h3 {
    font-family: var(--mono);
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
    margin-top: 1.2rem !important;
    margin-bottom: .4rem !important;
  }

  /* Metric widgets — Ableton device-like "chip" ----------------------- */
  [data-testid="stMetric"] {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: .55rem .75rem;
  }
  [data-testid="stMetric"]:hover { border-color: var(--line-2); }
  [data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-faint) !important;
  }
  [data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.35rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
  }
  [data-testid="stMetricDelta"] { font-family: var(--mono) !important; }

  /* Buttons — tactile hardware feel ----------------------------------- */
  .stButton > button {
    background: var(--bg-2);
    color: var(--text);
    border: 1px solid var(--line-2);
    border-radius: 3px;
    padding: .35rem .9rem;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
    font-weight: 500;
    transition: background 60ms ease, border-color 60ms ease;
  }
  .stButton > button:hover { background: var(--bg-3); border-color: var(--accent); color: var(--accent); }
  .stButton > button:active { background: var(--accent); color: #111; border-color: var(--accent); }
  .stButton > button:focus:not(:active) { box-shadow: 0 0 0 1px var(--accent); }

  /* Primary buttons (st.button(type="primary")) ----------------------- */
  .stButton > button[kind="primary"] {
    background: var(--accent); color: #111; border-color: var(--accent);
  }
  .stButton > button[kind="primary"]:hover { background: #2aaabd; border-color: #2aaabd; color: #111; }

  /* Tabs — VST plugin tabs ------------------------------------------- */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid var(--line);
    background: transparent;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--text-dim);
    padding: .45rem 1rem; border: none; border-bottom: 2px solid transparent;
    font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
  }
  .stTabs [data-baseweb="tab"]:hover { color: var(--text); }
  .stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

  /* Inputs — unified dark-field look --------------------------------- */
  .stTextInput input, .stNumberInput input, .stTextArea textarea,
  .stSelectbox div[data-baseweb="select"] > div,
  .stMultiSelect div[data-baseweb="select"] > div {
    background: var(--bg-3) !important;
    color: var(--text) !important;
    border: 1px solid var(--line-2) !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
  }
  .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
  }
  .stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; }
  label { color: var(--text-dim) !important; font-size: 11px !important; font-family: var(--mono); letter-spacing: .06em; text-transform: uppercase; }

  /* Checkboxes + radios — compact DAW toggles ------------------------- */
  .stCheckbox label, .stRadio label { text-transform: none; letter-spacing: 0; font-family: var(--sans); font-size: 12px; color: var(--text) !important; }
  .stRadio [role="radiogroup"] { gap: .35rem !important; }

  /* Dataframe / table ------------------------------------------------- */
  [data-testid="stDataFrame"] {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: 4px;
  }
  [data-testid="stDataFrame"] [role="grid"] { font-family: var(--mono); font-size: 11.5px; }
  [data-testid="stDataFrame"] [role="columnheader"] {
    background: var(--bg-2) !important;
    color: var(--text-dim) !important;
    font-weight: 500 !important;
    font-size: 10px !important;
    letter-spacing: .08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--line-2) !important;
  }

  /* Captions, help text, code --------------------------------------- */
  .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-faint) !important;
    font-family: var(--mono);
    font-size: 11px;
  }
  code, .stMarkdown code {
    background: var(--bg-2) !important;
    color: var(--accent) !important;
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 1px 5px;
    font-size: 11px;
  }

  /* Alerts — color-coded DAW notifications -------------------------- */
  [data-testid="stAlert"] {
    background: var(--bg-1) !important;
    border: 1px solid var(--line-2) !important;
    border-left-width: 3px !important;
    border-radius: 3px !important;
    color: var(--text) !important;
  }
  div[data-baseweb="notification"][kind="info"]    { border-left-color: var(--accent-2) !important; }
  div[data-baseweb="notification"][kind="success"] { border-left-color: var(--ok) !important; }
  div[data-baseweb="notification"][kind="warning"] { border-left-color: var(--warn) !important; }
  div[data-baseweb="notification"][kind="error"]   { border-left-color: var(--err) !important; }

  /* Dividers --------------------------------------------------------- */
  hr { border-color: var(--line) !important; margin: 1rem 0 !important; }

  /* Expander --------------------------------------------------------- */
  [data-testid="stExpander"] {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: 4px;
  }
  [data-testid="stExpander"] summary { font-family: var(--mono); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-dim); }

  /* Scrollbars — subtle, dark ---------------------------------------- */
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: var(--bg-0); }
  ::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 5px; border: 2px solid var(--bg-0); }
  ::-webkit-scrollbar-thumb:hover { background: var(--line-2); }

  /* Tracklist-engine components (reusable primitives) ----------------- */
  .te-topbar {
    display: flex; align-items: center; gap: 1rem;
    padding: .55rem .9rem;
    background: linear-gradient(180deg, #1c1c1c, #141414);
    border: 1px solid var(--line);
    border-radius: 4px;
    margin-bottom: 1.2rem;
  }
  .te-topbar .te-logo {
    font-family: var(--mono); font-size: 12px; font-weight: 700;
    letter-spacing: .22em; color: var(--accent);
  }
  .te-topbar .te-logo::before {
    content: "▶"; margin-right: .5rem; color: var(--accent);
  }
  .te-topbar .te-page {
    font-family: var(--mono); font-size: 11px; letter-spacing: .1em;
    text-transform: uppercase; color: var(--text-dim);
    padding: 2px 8px; border: 1px solid var(--line-2); border-radius: 2px;
  }
  .te-topbar .te-spacer { flex: 1; }
  .te-topbar .te-dbchip {
    font-family: var(--mono); font-size: 10.5px; color: var(--text-faint);
    padding: 2px 8px; border: 1px solid var(--line); border-radius: 2px;
    background: var(--bg-2);
  }

  .te-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: .25rem 0 .75rem 0; }
  .te-chip {
    display: inline-flex; align-items: baseline; gap: .5ch;
    padding: 3px 9px;
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: 999px;
    font-family: var(--mono); font-size: 11px;
    color: var(--text);
  }
  .te-chip .te-chip-k { color: var(--text-faint); text-transform: uppercase; letter-spacing: .08em; font-size: 9.5px; }
  .te-chip.te-chip-ok   { border-color: rgba(62,163,114,.45); }
  .te-chip.te-chip-warn { border-color: rgba(214,166,74,.45); }
  .te-chip.te-chip-err  { border-color: rgba(224,90,79,.45); }
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)


def _te_topbar(page_name: str, db_path: Path) -> None:
    """Render the DAW-style top bar shown on every page."""
    st.markdown(
        f"""
        <div class="te-topbar">
          <div class="te-logo">TRACKLIST ENGINE</div>
          <div class="te-page">{page_name}</div>
          <div class="te-spacer"></div>
          <div class="te-dbchip">DB · {db_path.name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _te_chips(items: list[tuple[str, object, str]]) -> None:
    """Render a row of labeled DAW-style pill chips.

    items: list of (label, value, tone) where tone ∈ {"", "ok", "warn", "err"}.
    """
    html_parts = ['<div class="te-chips">']
    for label, value, tone in items:
        cls = f"te-chip te-chip-{tone}" if tone else "te-chip"
        html_parts.append(
            f'<span class="{cls}"><span class="te-chip-k">{label}</span>{value}</span>'
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ---------- cached loaders ---------------------------------------------------

@st.cache_resource
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=3600, show_spinner="Loading corpus…")
def _load_bb() -> dict[str, pd.DataFrame]:
    """Corpus-wide BB tables + tokenized rows. Cached 1h — data changes only
    when the scraper/analysis pipelines write new rows; hit `C` in the
    Streamlit toolbar to clear if you just ran a job."""
    conn = _connect()
    sets = load_big_bootie_sets(conn)
    rows = load_big_bootie_rows(conn)
    tml  = load_big_bootie_track_media_links(conn)
    tokens = tokenize_rows(rows)
    return {"sets": sets, "rows": rows, "tml": tml, "tokens": tokens}


# ---- Alignment-review caches ------------------------------------------------
# Each helper is keyed on its args so different sets don't clobber each other.

@st.cache_data(ttl=3600)
def _aligned_sets_summary() -> list[dict]:
    """One row per set that has `set_section_alignment` rows. Returns plain
    dicts (not sqlite3.Row) so streamlit's pickle-based cache can store it."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT a.set_id, s.title, COUNT(*) AS n_sections,
               AVG(a.confidence) AS mean_match,
               MAX(a.aligned_at)  AS last_run
        FROM set_section_alignment a
        JOIN dj_sets s ON s.set_id = a.set_id
        GROUP BY a.set_id, s.title
        ORDER BY last_run DESC
        """,
    ).fetchall()
    return [
        {"set_id": r["set_id"], "title": r["title"], "n_sections": r["n_sections"],
         "mean_match": r["mean_match"], "last_run": r["last_run"]}
        for r in rows
    ]


@st.cache_data(ttl=3600)
def _set_audio_path_for(set_id: str) -> str | None:
    conn = _connect()
    row = conn.execute(
        "SELECT path FROM set_audio WHERE set_id = ? "
        "ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1",
        (set_id,),
    ).fetchone()
    return row[0] if row else None


@st.cache_data(ttl=3600)
def _set_stem_paths_for(set_id: str) -> dict[str, str]:
    """Return {stem_name → on-disk path} for the chosen set's demucs
    splits, including the pre-summed `instrumental` stem when present.
    Empty dict if no stems have been written yet — caller should
    gracefully prompt the user to run analysis.
    """
    conn = _connect()
    rows = conn.execute(
        """
        SELECT ss.stem_name, ss.path
        FROM set_stems ss
        JOIN set_audio  sa ON sa.set_audio_id = ss.set_audio_id
        WHERE sa.set_id = ?
        ORDER BY sa.is_reference DESC, sa.downloaded_at DESC
        """,
        (set_id,),
    ).fetchall()
    out: dict[str, str] = {}
    for r in rows:
        out.setdefault(r["stem_name"], r["path"])
    return out


@st.cache_data(ttl=3600)
def _playable_sets() -> list[dict]:
    """All sets that have downloaded audio, with best-effort stem
    coverage. Lights up the stem-player dropdown; ordered by most
    recently downloaded first so the pilot sets are easy to find."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT s.set_id, s.title, sa.downloaded_at, sa.duration_s,
               (SELECT COUNT(*) FROM set_stems ss
                  JOIN set_audio sa2 ON sa2.set_audio_id = ss.set_audio_id
                  WHERE sa2.set_id = s.set_id) AS n_stems
        FROM dj_sets s
        JOIN set_audio sa ON sa.set_id = s.set_id
        GROUP BY s.set_id, s.title
        ORDER BY sa.downloaded_at DESC
        """,
    ).fetchall()
    return [
        {"set_id": r["set_id"], "title": r["title"] or r["set_id"],
         "downloaded_at": r["downloaded_at"], "duration_s": r["duration_s"],
         "n_stems": int(r["n_stems"] or 0)}
        for r in rows
    ]


@st.cache_data(ttl=3600)
def _timeline_cue_anchors(set_id: str) -> dict[int, dict]:
    """Return {row_index → {cue, is_concurrent, title}} from the timeline.
    Drives the clip-width heuristic: primary (non-w/) rows occupy a full
    section up to the next primary cue; w/ layers default to ~90 s.
    """
    import json as _json
    conn = _connect()
    row = conn.execute(
        "SELECT payload_json FROM set_timeline WHERE set_id = ?", (set_id,)
    ).fetchone()
    if row is None or not row[0]:
        return {}
    try:
        payload = _json.loads(row[0])
    except _json.JSONDecodeError:
        return {}
    out: dict[int, dict] = {}
    for s in payload.get("segments", []):
        cue = s.get("cue_seconds_section")
        if cue is not None:
            out[int(s["row_index"])] = {
                "cue": float(cue),
                "is_concurrent": bool(s.get("is_concurrent")),
                "title": s.get("title") or "",
            }
    return out


@st.cache_data(ttl=60)
def _aligned_rows_for(set_id: str, *, source: str | None = None) -> pd.DataFrame:
    """Per-section alignment rows with:
      - actual aligner-computed positions (`set_start_s` / `set_end_s`)
        from `set_section_alignment`, whether DTW- or CCC-derived
      - ref-side span (`ref_start_s` / `ref_end_s`) for the "which 15 s
        of the ref played" display
      - section-level stem-mask label (tag > measure majority > classifier)

    Earlier revisions applied a tracklist-cue-based override on top of
    the alignment spans (primary rows → next primary cue, concurrent
    rows → cue + 90 s default) because DTW timestamps drifted too
    widely to trust. CCC changed that: timestamps are now accurate to
    the seconds, so the UI honours them directly. A future toggle
    could restore the cue-based view if useful.
    """
    import json as _json
    from collections import Counter
    from audio_pipeline.alignment.stem_mask import BADGE, classify, parse_version_tag

    conn = _connect()
    params: list[object] = [set_id]
    source_clause = ""
    if source is not None:
        source_clause = " AND COALESCE(a.confidence_source, 'legacy') = ?"
        params.append(source)
    # Label lookup: legacy rows have section_idx == dj_set_rows.row_index,
    # but SOTA rows (section_idx 100000+) don't. Use the track_id →
    # tlp_id → element_id → row lookup as a fallback so every row gets a
    # human-readable label.
    rows = conn.execute(
        f"""
        SELECT a.section_idx, a.set_start_s, a.set_end_s,
               a.ref_start_s, a.ref_end_s, a.ref_section_idx,
               a.transposition_semitones, a.bpm_ratio, a.confidence,
               a.ref_track_id, a.cutup_plan_json, a.stem_match_rates_json,
               COALESCE(a.confidence_source, 'legacy') AS confidence_source,
               ta.path AS ref_path,
               COALESCE(
                   a.label,
                   r_direct.text_excerpt,
                   r_via_track.text_excerpt,
                   a.ref_track_id,
                   '(unknown track)'
               ) AS label
        FROM set_section_alignment a
        LEFT JOIN track_audio ta ON ta.track_id = a.ref_track_id
        LEFT JOIN dj_set_rows r_direct
               ON r_direct.set_id = a.set_id AND r_direct.row_index = a.section_idx
        LEFT JOIN dj_set_track_media_links tml
               ON tml.set_id = a.set_id AND tml.track_id = a.ref_track_id
        LEFT JOIN dj_set_rows r_via_track
               ON r_via_track.set_id = a.set_id AND r_via_track.element_id = tml.tlp_id
        WHERE a.set_id = ?{source_clause}
        GROUP BY a.section_idx
        ORDER BY a.set_start_s
        """,
        params,
    ).fetchall()
    df = pd.DataFrame(rows, columns=[
        "section_idx", "set_start_s", "set_end_s",
        "ref_start_s", "ref_end_s", "ref_section_idx",
        "transposition", "bpm_ratio", "match_rate",
        "ref_track_id", "cutup_plan_json", "stem_rates_json",
        "confidence_source",
        "ref_path", "label",
    ])
    # Guard against any legacy rows where start > end — normalise so
    # downstream lane-packing / width calcs can't produce negative widths.
    df[["set_start_s", "set_end_s"]] = df.apply(
        lambda r: pd.Series(sorted([r["set_start_s"], r["set_end_s"]])), axis=1,
    )
    # Keep the old field names around (some downstream code paths still
    # reference them) but point them at the same aligner-computed span.
    df["dtw_start_s"] = df["set_start_s"]
    df["dtw_end_s"] = df["set_end_s"]

    def _as_json_str(v: object) -> str | None:
        """Coerce a DB value (str, None, NaN float) into a JSON string
        or None. Pandas turns SQLite NULLs into `float('nan')`, which
        is truthy but isn't a string — so the naive `if plan_raw:`
        check passes and `json.loads` then fails on the float. This
        helper gates both conditions in one place."""
        if isinstance(v, str) and v:
            return v
        return None

    def _section_label(row: pd.Series) -> str:
        tag = parse_version_tag(row.get("label"))
        if tag is not None:
            emoji, text = BADGE[tag]
            return f"{emoji} {text}"

        plan_raw = _as_json_str(row.get("cutup_plan_json"))
        if plan_raw:
            try:
                plan = _json.loads(plan_raw)
                per_measure = [m for s in plan for m in s.get("stem_mask", []) if m != "none"]
                if per_measure:
                    label = Counter(per_measure).most_common(1)[0][0]
                    emoji, text = BADGE[label]
                    return f"{emoji} {text}"
            except _json.JSONDecodeError:
                pass

        rates_raw = _as_json_str(row.get("stem_rates_json"))
        if rates_raw:
            try:
                rates = _json.loads(rates_raw)
                label = classify(rates)
                emoji, text = BADGE[label]
                return f"{emoji} {text}"
            except _json.JSONDecodeError:
                pass

        # Fallthrough: the row IS aligned, but we have no version-tag in
        # the label and sota.py doesn't populate stem_match_rates_json /
        # cutup_plan_json, so we can't classify which stem was audible.
        # This is a stem-mask UNKNOWN, not an alignment failure.
        return "🔊 stem unknown"

    df["stem_mask"] = df.apply(_section_label, axis=1)
    return df


# ---- Ableton-style layered timeline -----------------------------------------

_MASK_COLOR: dict[str, str] = {
    "🎛 full":         "#3ea372",   # green
    "🎤 acappella":    "#3a7bd5",   # blue
    "🥁 instrumental": "#d6a64a",   # yellow
    "🧩 partial":      "#c97a3a",   # orange
    "🔊 stem unknown": "#555555",   # gray — aligned but stem class unknown
}

def _assign_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """Greedy lane-packing so overlapping set windows don't occlude each other."""
    rows = df.sort_values("set_start_s").reset_index(drop=True).copy()
    lane_ends: list[float] = []
    lanes: list[int] = []
    for _, r in rows.iterrows():
        placed = False
        for li, end in enumerate(lane_ends):
            if r["set_start_s"] >= end - 0.5:
                lane_ends[li] = r["set_end_s"]
                lanes.append(li)
                placed = True
                break
        if not placed:
            lane_ends.append(r["set_end_s"])
            lanes.append(len(lane_ends) - 1)
    rows["lane"] = lanes
    return rows


@st.cache_resource(show_spinner="Generating UI preview audio (first time only)…")
def _ensure_preview_audio(set_audio_path_str: str) -> str | None:
    """Transcode the set mix to 64 kbps mono MP3 once and cache on disk.
    Keeps the iframe payload small enough for snappy reloads."""
    import subprocess
    src = Path(set_audio_path_str)
    if not src.exists():
        return None
    cache_dir = _REPO_ROOT / "data" / "ui_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / (src.stem + "_ui_preview.mp3")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return str(dst)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ac", "1", "-b:a", "64k", "-vn", str(dst)],
            check=True, timeout=180,
        )
        return str(dst) if dst.exists() else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _render_ableton_timeline(df: pd.DataFrame, set_audio_path: str | None) -> None:
    """Ableton-style arrangement view inside a single custom HTML component.

    Features (all client-side, no server rerun required):
      - Horizontal clips colored by stem mask, labeled with the song title.
      - **Zoom**: buttons + keyboard (+/-) + ctrl-scroll wheel.
      - **Scroll**: native horizontal scrollbar + drag-pan.
      - **Click to seek**: clicking anywhere jumps the audio + playhead.
      - **Playhead**: animates smoothly via requestAnimationFrame while audio plays.
    """
    import base64
    import json as _json

    if df.empty or not set_audio_path:
        st.info("No aligned sections or no set audio.")
        return
    if not Path(set_audio_path).exists():
        st.warning(f"Set audio not on disk: {set_audio_path}")
        return

    preview = _ensure_preview_audio(set_audio_path)
    if preview is None:
        st.warning("Preview audio transcode failed (ffmpeg missing?) — falling back to Plotly view.")
        _render_layered_timeline(df, set_audio_path)
        return

    rows = _assign_lanes(df)
    n_lanes = int(rows["lane"].max()) + 1 if not rows.empty else 1
    set_duration = float(rows["set_end_s"].max())

    # Per-measure mask palette for the intra-clip strip (pure labels, no emojis).
    MASK_HEX = {
        "full":         "#3ea372",
        "acappella":    "#3a7bd5",
        "instrumental": "#d6a64a",
        "partial":      "#c97a3a",
        "none":         "#2a2a2a",
    }

    def _per_measure_colors(cutup_raw: object) -> list[str]:
        """Flatten cutup_plan_json segments into a per-set-measure color list,
        ordered by set_measure_start. Empty string if no plan is present."""
        if not isinstance(cutup_raw, str) or not cutup_raw:
            return []
        try:
            plan = _json.loads(cutup_raw)
        except _json.JSONDecodeError:
            return []
        # Build {set_measure_idx: color} then emit contiguous run from min to max.
        colors: dict[int, str] = {}
        for seg in plan:
            masks = seg.get("stem_mask", [])
            if not masks:
                continue
            sm0 = int(seg.get("set_measure_start", 0))
            for i, m in enumerate(masks):
                colors[sm0 + i] = MASK_HEX.get(str(m), "#444")
        if not colors:
            return []
        mn, mx = min(colors), max(colors)
        return [colors.get(i, "#222") for i in range(mn, mx + 1)]

    clips = [
        {
            "start": float(r["set_start_s"]),
            "end":   float(r["set_end_s"]),
            "lane":  int(r["lane"]),
            "label": _clean_clip_label(str(r.get("label") or r.get("ref_track_id") or "")),
            "color": _MASK_COLOR.get(str(r["stem_mask"]), "#888"),
            "mask":  str(r["stem_mask"]),
            "match": float(r["match_rate"]) if pd.notna(r["match_rate"]) else 0.0,
            "row":   int(r["section_idx"]),
            "cells": _per_measure_colors(r.get("cutup_plan_json")),
        }
        for _, r in rows.iterrows()
    ]
    clips_json = _json.dumps(clips)

    audio_b64 = base64.b64encode(Path(preview).read_bytes()).decode("ascii")

    lane_height = 42
    ruler_height = 22
    audio_bar_height = 60
    total_height = audio_bar_height + ruler_height + (lane_height * n_lanes) + 40

    st.subheader("Arrangement")
    st.caption(
        "wheel = scroll · ⌘/ctrl+wheel = zoom · middle-drag = pan · click = seek · "
        "playhead tracks audio live."
    )

    html = """
<style>
  .ab-wrap { font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
             color: #ececec; background: #0f0f0f;
             border: 1px solid #2e2e2e; border-radius: 4px; overflow: hidden; }
  .ab-audio-bar { display: flex; align-items: center; gap: 8px; padding: 6px 10px;
                  background: #181818; border-bottom: 1px solid #2e2e2e; }
  .ab-audio-bar audio { flex: 1; height: 32px; }
  .ab-btn { background: #222; color: #ececec; border: 1px solid #3a3a3a;
            padding: 4px 10px; border-radius: 3px; cursor: pointer;
            font-family: ui-monospace, "JetBrains Mono", "SF Mono", monospace;
            font-size: 11px; letter-spacing: .06em; user-select: none;
            transition: background 60ms ease, border-color 60ms ease, color 60ms ease; }
  .ab-btn:hover { background: #2a2a2a; border-color: #1a8fa0; color: #1a8fa0; }
  .ab-tlscroll { overflow-x: auto; overflow-y: hidden; background: #0f0f0f;
                 position: relative; cursor: default;
                 scroll-behavior: auto;
                 overscroll-behavior: contain;
                 -webkit-user-select: none; user-select: none; }
  .ab-tlscroll.is-panning { cursor: grabbing; }
  .ab-tlinner { position: relative; background: #0f0f0f;
                background-image: linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
                background-size: 80px 100%;
                transform: translateZ(0);    /* promote to own layer so the
                                                 browser doesn't repaint the
                                                 ruler + every clip every frame
                                                 just to move the playhead */ }
  .ab-ruler { position: sticky; top: 0; height: __RH__px; background: #181818;
              border-bottom: 1px solid #2e2e2e; z-index: 3; pointer-events: none; }
  .ab-ruler span { position: absolute; color: #9a9a9a;
                   font-family: ui-monospace, "JetBrains Mono", "SF Mono", monospace;
                   font-size: 10px; letter-spacing: .04em;
                   padding: 5px 6px;
                   border-left: 1px solid #2e2e2e; height: 100%; box-sizing: border-box;
                   white-space: nowrap; }
  .ab-clip { position: absolute; height: __CLH__px; border-radius: 2px;
             padding: 3px 6px; color: #f8f8f8;
             font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
             font-size: 11px; font-weight: 500;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
             border: 1px solid rgba(0,0,0,0.45); cursor: pointer; box-sizing: border-box;
             box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
             transition: filter 80ms ease, box-shadow 80ms ease;
             will-change: left, width;    /* GPU-promoted — smoother zoom */ }
  .ab-clip:hover { filter: brightness(1.18);
                   box-shadow: 0 0 0 1px #1a8fa0, inset 0 1px 0 rgba(255,255,255,0.12);
                   z-index: 4; }
  .ab-clip .ab-meta { opacity: 0.75;
                      font-family: ui-monospace, "JetBrains Mono", "SF Mono", monospace;
                      font-size: 9px; font-weight: 400; margin-top: 1px;
                      letter-spacing: .02em; }
  .ab-clip .ab-cells { position: absolute; left: 0; right: 0; bottom: 0;
                       height: 5px; display: flex; pointer-events: none;
                       border-top: 1px solid rgba(0,0,0,0.55); }
  .ab-clip .ab-cells span { flex: 1; min-width: 1px; }
  .ab-playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: #1a8fa0;
                 pointer-events: none; z-index: 5;
                 box-shadow: 0 0 6px #1a8fa0, 0 0 14px rgba(26,143,160,.4);
                 transform: translateX(0);
                 will-change: transform; }
  #abTimeReadout { font-family: ui-monospace, "JetBrains Mono", "SF Mono", monospace;
                   font-size: 12px; color: #1a8fa0; font-weight: 600;
                   letter-spacing: .04em; min-width: 56px; text-align: right; }
</style>
<div class="ab-wrap">
  <div class="ab-audio-bar">
    <audio id="abAudio" controls preload="auto" src="data:audio/mpeg;base64,__AUDIO__"></audio>
    <button class="ab-btn" id="abZoomOut" title="Zoom out (−)">−</button>
    <button class="ab-btn" id="abZoomIn"  title="Zoom in (+)">+</button>
    <button class="ab-btn" id="abFit"     title="Fit to width (0)">FIT</button>
    <span id="abTimeReadout">0:00</span>
    <span style="font-size: 10px; color: #6a6a6a; margin-left: auto;
                 font-family: ui-monospace, 'JetBrains Mono', 'SF Mono', monospace;
                 letter-spacing: .08em; text-transform: uppercase;">
      wheel · ⌘+wheel · mid-drag · click
    </span>
  </div>
  <div class="ab-tlscroll" id="abScroll">
    <div class="ab-tlinner" id="abInner">
      <div class="ab-ruler" id="abRuler"></div>
      <div class="ab-playhead" id="abPlayhead"></div>
    </div>
  </div>
</div>
<script>
(function() {
  const clips      = __CLIPS__;
  const setDur     = __DUR__;
  const nLanes     = __NLANES__;
  const laneH      = __LH__;
  const clipH      = __CLH__;
  const rulerH     = __RH__;

  const audio      = document.getElementById('abAudio');
  const scrollEl   = document.getElementById('abScroll');
  const innerEl    = document.getElementById('abInner');
  const rulerEl    = document.getElementById('abRuler');
  const playhead   = document.getElementById('abPlayhead');
  const readout    = document.getElementById('abTimeReadout');

  // Initial zoom fits the whole set into the visible scroll area.
  let pxPerSec = Math.max(1.5, (scrollEl.clientWidth - 20) / setDur);

  function fmtTime(t) {
    t = Math.max(0, t || 0);
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  // --- Clips live as persistent DOM nodes; zoom / scroll updates their
  //     left + width in place instead of destroying and recreating the
  //     DOM tree. This is the single biggest smoothness win on dense
  //     sets (150+ clips) — zoom was stuttery because every wheel tick
  //     rebuilt the whole thing from scratch.
  const clipNodes = [];

  function createClips() {
    clips.forEach((c) => {
      const el = document.createElement('div');
      el.className = 'ab-clip';
      el.style.top        = (rulerH + c.lane * laneH + 4) + 'px';
      el.style.background = c.color;
      el.title = c.label + ' · row ' + c.row + ' · match ' + c.match.toFixed(2) +
                 ' · ' + c.mask + ' · ' + fmtTime(c.start) + '–' + fmtTime(c.end);

      const name = document.createElement('div');
      name.textContent = c.label;
      const meta = document.createElement('div');
      meta.className = 'ab-meta';
      meta.textContent = c.mask + ' · m=' + c.match.toFixed(2);
      el.appendChild(name);
      el.appendChild(meta);

      if (c.cells && c.cells.length) {
        const cells = document.createElement('div');
        cells.className = 'ab-cells';
        c.cells.forEach((hex) => {
          const sp = document.createElement('span');
          sp.style.background = hex;
          cells.appendChild(sp);
        });
        el.appendChild(cells);
      }

      // Seek on click — but only when this isn't the tail of a drag
      // gesture. `dragDidPan` is toggled by the pan handler below.
      el.addEventListener('click', (ev) => {
        if (dragDidPan) return;
        ev.stopPropagation();
        audio.currentTime = c.start;
        audio.play();
      });
      innerEl.appendChild(el);
      clipNodes.push(el);
    });
  }

  function applyZoomLayout() {
    // Update the inner width once, then reposition persistent clip
    // DOM nodes in a single batch. The browser repaints once per
    // frame instead of reflowing per-clip.
    const innerW = Math.ceil(setDur * pxPerSec);
    innerEl.style.width  = innerW + 'px';
    innerEl.style.height = (rulerH + nLanes * laneH + 8) + 'px';
    rulerEl.style.width  = innerW + 'px';
    clipNodes.forEach((el, i) => {
      const c = clips[i];
      el.style.left  = (c.start * pxPerSec) + 'px';
      el.style.width = Math.max(2, (c.end - c.start) * pxPerSec) + 'px';
    });
  }

  function rebuildRuler() {
    rulerEl.innerHTML = '';
    // Choose a tick spacing that lands near 80 px and snaps to a
    // "nice" round value (1/5/10/30/60 sec) so labels don't wobble
    // under zoom.
    const nice = [1, 2, 5, 10, 15, 30, 60, 120, 300];
    const target = 80;
    let secPerTick = nice[nice.length - 1];
    for (const n of nice) {
      if (n * pxPerSec >= target) { secPerTick = n; break; }
    }
    const frag = document.createDocumentFragment();
    for (let t = 0; t <= setDur; t += secPerTick) {
      const sp = document.createElement('span');
      sp.style.left = (t * pxPerSec) + 'px';
      sp.textContent = fmtTime(t);
      frag.appendChild(sp);
    }
    rulerEl.appendChild(frag);
  }

  // Playhead update uses transform instead of `left` so the browser
  // can compose it on the GPU — no layout thrash during playback.
  let lastReadoutSec = -1;
  function updatePlayhead(followScroll) {
    const x = (audio.currentTime || 0) * pxPerSec;
    playhead.style.transform = 'translateX(' + x + 'px)';
    const sec = Math.floor(audio.currentTime || 0);
    if (sec !== lastReadoutSec) {
      readout.textContent = fmtTime(audio.currentTime) + ' / ' + fmtTime(setDur);
      lastReadoutSec = sec;
    }
    if (followScroll && !audio.paused) {
      const scL = scrollEl.scrollLeft;
      const scR = scL + scrollEl.clientWidth;
      if (x < scL + 40 || x > scR - 40) {
        // Jump the scroll position rather than smooth-scroll: smooth
        // scroll + rAF playhead update = visible drift during long
        // follows. An instant jump keeps the playhead glued.
        scrollEl.scrollLeft = Math.max(0, x - scrollEl.clientWidth / 3);
      }
    }
  }

  // --- Zoom. Two entry points (wheel + buttons) funnel through
  //     setZoom so the cursor-anchor logic is shared. rAF batches
  //     multiple wheel events into one layout-apply per frame —
  //     without this, a fast scroll gesture runs applyZoomLayout
  //     dozens of times per frame, which is why zoom felt jumpy.
  let pendingFrame = null;
  function scheduleLayout() {
    if (pendingFrame !== null) return;
    pendingFrame = requestAnimationFrame(() => {
      pendingFrame = null;
      applyZoomLayout();
      rebuildRuler();
      updatePlayhead(false);
    });
  }

  function setZoom(newPxPerSec, cursorClientX) {
    newPxPerSec = Math.max(0.5, Math.min(400, newPxPerSec));
    if (Math.abs(newPxPerSec - pxPerSec) < 1e-6) return;
    // Preserve the time under the cursor (or center if unspecified).
    let cursorX;
    if (cursorClientX == null) {
      cursorX = scrollEl.scrollLeft + scrollEl.clientWidth / 2;
    } else {
      cursorX = cursorClientX - innerEl.getBoundingClientRect().left;
    }
    const cursorT = cursorX / pxPerSec;
    pxPerSec = newPxPerSec;
    scheduleLayout();
    // Fix the scroll position so cursorT stays under the cursor.
    const newCursorX = cursorT * pxPerSec;
    scrollEl.scrollLeft += (newCursorX - cursorX);
  }

  // Ctrl/Cmd + wheel = zoom; plain wheel = horizontal scroll (DAW
  // convention). Passive handler flag: we only call preventDefault
  // on the zoom path, so the scroll path stays smooth.
  scrollEl.addEventListener('wheel', (ev) => {
    if (ev.ctrlKey || ev.metaKey) {
      ev.preventDefault();
      const factor = ev.deltaY < 0 ? 1.2 : 1 / 1.2;
      setZoom(pxPerSec * factor, ev.clientX);
      return;
    }
    // Horizontal scroll: map vertical wheel (deltaY) to scrollLeft
    // so a plain trackpad or wheel scrolls time, not the page.
    const dx = ev.deltaX || 0;
    const dy = ev.deltaY || 0;
    // Don't hijack legit horizontal wheel or shift+wheel.
    if (dx !== 0 || ev.shiftKey) return;
    if (dy !== 0) {
      ev.preventDefault();
      scrollEl.scrollLeft += dy;
    }
  }, {passive: false});

  // --- Middle-button drag to pan. Keeps click-to-seek clean (left
  //     click stays seek, middle-click stays pan) and avoids the
  //     drag-release-triggers-click bug.
  let isPanning = false;
  let dragDidPan = false;
  let panStartX = 0, panStartScroll = 0;
  scrollEl.addEventListener('mousedown', (ev) => {
    if (ev.button !== 1) return;           // middle button only
    ev.preventDefault();
    isPanning = true; dragDidPan = false;
    panStartX = ev.clientX;
    panStartScroll = scrollEl.scrollLeft;
    scrollEl.classList.add('is-panning');
  });
  window.addEventListener('mousemove', (ev) => {
    if (!isPanning) return;
    const dx = ev.clientX - panStartX;
    if (Math.abs(dx) > 3) dragDidPan = true;
    scrollEl.scrollLeft = panStartScroll - dx;
  });
  window.addEventListener('mouseup', () => {
    if (!isPanning) return;
    isPanning = false;
    scrollEl.classList.remove('is-panning');
    // Reset dragDidPan on the next tick so a click immediately after
    // a pan release still cancels; but a fresh click a moment later
    // seeks normally.
    setTimeout(() => { dragDidPan = false; }, 100);
  });

  // Left-click on the timeline background (not a clip) → seek.
  scrollEl.addEventListener('click', (ev) => {
    if (ev.target.closest('.ab-clip')) return;
    if (dragDidPan) return;
    const rect = innerEl.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    audio.currentTime = Math.max(0, Math.min(setDur, x / pxPerSec));
    updatePlayhead(false);
  });

  // Keyboard shortcuts when the component has focus (DAW-style).
  scrollEl.tabIndex = 0;
  scrollEl.addEventListener('keydown', (ev) => {
    if (ev.key === '+' || ev.key === '=') { setZoom(pxPerSec * 1.4); ev.preventDefault(); }
    else if (ev.key === '-' || ev.key === '_') { setZoom(pxPerSec / 1.4); ev.preventDefault(); }
    else if (ev.key === '0') { fitZoom(); ev.preventDefault(); }
    else if (ev.key === ' ') {
      ev.preventDefault();
      if (audio.paused) audio.play(); else audio.pause();
    }
  });

  function fitZoom() {
    setZoom((scrollEl.clientWidth - 20) / setDur);
    scrollEl.scrollLeft = 0;
  }

  document.getElementById('abZoomIn').onclick  = () => setZoom(pxPerSec * 1.4);
  document.getElementById('abZoomOut').onclick = () => setZoom(pxPerSec / 1.4);
  document.getElementById('abFit').onclick     = fitZoom;

  // --- Playhead animation. Drive rAF only while audio is playing;
  //     during pause/seek the handlers below kick a single update.
  //     Separate from zoom's layout rAF so playback stays smooth
  //     even during continuous zoom scrubs.
  let rafId = null;
  function tick() {
    updatePlayhead(true);
    rafId = requestAnimationFrame(tick);
  }
  audio.addEventListener('play',  () => { if (rafId === null) tick(); });
  audio.addEventListener('pause', () => {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    updatePlayhead(false);
  });
  audio.addEventListener('seeked',   () => updatePlayhead(false));
  audio.addEventListener('timeupdate', () => updatePlayhead(false));

  // Size-sensitive rebuild. Instead of binding to window resize
  // (which fires for the whole page and may not reflect the
  // component's actual width), use ResizeObserver on the scroll
  // container.
  const ro = new ResizeObserver(() => rebuildRuler());
  ro.observe(scrollEl);

  // Initial render — works before audio metadata arrives so the
  // layout is never blank.
  createClips();
  applyZoomLayout();
  rebuildRuler();
  updatePlayhead(false);

  audio.addEventListener('loadedmetadata', () => {
    // Only rebuild if the audio duration differs from our best-guess
    // set duration — otherwise skip the repaint flash.
    if (Math.abs((audio.duration || 0) - setDur) > 2) {
      applyZoomLayout(); rebuildRuler();
    }
    updatePlayhead(false);
  });
})();
</script>
"""
    html = (html
            .replace("__AUDIO__", audio_b64)
            .replace("__CLIPS__", clips_json)
            .replace("__DUR__",   str(set_duration))
            .replace("__NLANES__", str(n_lanes))
            .replace("__LH__",    str(lane_height))
            .replace("__CLH__",   str(lane_height - 8))
            .replace("__RH__",    str(ruler_height)))

    st.components.v1.html(html, height=total_height, scrolling=False)


def _render_layered_timeline(df: pd.DataFrame, set_audio_path: str | None) -> None:
    """Ableton-style timeline: each aligned section is a horizontal clip on
    a time axis, stacked onto concurrent-group lanes so overlapping tracks
    sit on separate rows. Color = stem-mask label. Song name renders
    inside the clip. Scroll/pinch to zoom, drag to pan.
    """
    if df.empty:
        return

    # Greedy lane packing so overlapping time windows don't occlude.
    rows = df.sort_values("set_start_s").reset_index(drop=True).copy()
    lane_ends: list[float] = []
    lanes: list[int] = []
    for _, r in rows.iterrows():
        placed = False
        for li, end in enumerate(lane_ends):
            if r["set_start_s"] >= end - 0.5:
                lane_ends[li] = r["set_end_s"]
                lanes.append(li)
                placed = True
                break
        if not placed:
            lane_ends.append(r["set_end_s"])
            lanes.append(len(lane_ends) - 1)
    rows["lane"] = lanes
    n_lanes = int(rows["lane"].max()) + 1 if not rows.empty else 1

    set_duration = float(rows["set_end_s"].max()) if not rows.empty else 0.0
    seek_to = int(st.session_state.get("align_seek", 0))

    st.markdown("### 🎚 Layered timeline")
    st.caption(
        "One clip per aligned section; stacked vertically like Ableton's "
        "Arrangement view. **Scroll to zoom time**, **shift-scroll to pan**, "
        "**drag** to move around. Color = stem mask. Hover a clip for details."
    )

    # Zoom controls — Streamlit widgets that drive the Plotly x-range.
    zc1, zc2, zc3, zc4 = st.columns([1, 1, 1, 2])
    default_span = min(300.0, set_duration)   # default 5 min window
    center = float(st.session_state.get("align_view_center", set_duration / 2))
    span = float(st.session_state.get("align_view_span", default_span))

    with zc1:
        if st.button("⇤ Zoom out", use_container_width=True, key="zoom_out"):
            span = min(set_duration, span * 1.5)
            st.session_state["align_view_span"] = span
    with zc2:
        if st.button("Zoom in ⇥", use_container_width=True, key="zoom_in"):
            span = max(10.0, span / 1.5)
            st.session_state["align_view_span"] = span
    with zc3:
        if st.button("Fit all", use_container_width=True, key="zoom_fit"):
            span = set_duration
            center = set_duration / 2
            st.session_state["align_view_span"] = span
            st.session_state["align_view_center"] = center
    with zc4:
        center = st.slider(
            "Scroll (pan)", 0.0, max(set_duration, 1.0), float(center), step=1.0,
            key="align_view_center_slider",
        )
        st.session_state["align_view_center"] = float(center)

    x0 = max(0.0, center - span / 2)
    x1 = min(set_duration, x0 + span)
    if x1 - x0 < span:
        x0 = max(0.0, x1 - span)

    import plotly.graph_objects as go
    fig = go.Figure()

    for _, r in rows.iterrows():
        color = _MASK_COLOR.get(str(r["stem_mask"]), "#888")
        width = max(r["set_end_s"] - r["set_start_s"], 0.1)
        lane_y = f"lane {int(r['lane']) + 1}"
        # Clip name for in-bar label — parse the tracklist "text_excerpt" into
        # something that reads like "artist - title" by trimming the cue-time
        # prefix and any trailing scrobble counts.
        raw_label = str(r.get("label") or r.get("ref_track_id") or "")
        clean_label = _clean_clip_label(raw_label)
        text_on_bar = clean_label if width > 6 else ""   # hide label on very narrow clips
        fig.add_trace(go.Bar(
            base=[r["set_start_s"]],
            x=[width],
            y=[lane_y],
            orientation="h",
            text=[text_on_bar],
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(size=11, color="#f5f5f5", family="Arial, sans-serif"),
            cliponaxis=True,
            marker=dict(color=color, line=dict(color="#0a0a0a", width=0.6)),
            hovertemplate=(
                f"<b>{clean_label}</b><br>"
                f"row {int(r['section_idx'])}  ·  set {r['set_start_s']:.1f}–{r['set_end_s']:.1f}s "
                f"(dur {r['set_end_s']-r['set_start_s']:.1f}s)<br>"
                f"mask: {r['stem_mask']}<br>"
                f"match: {r['match_rate']:.2f}  ·  Δ={int(r['transposition']) if pd.notna(r['transposition']) else '?'} semitones  ·  "
                f"bpm×{r['bpm_ratio']:.2f}<extra></extra>"
            ),
            showlegend=False,
        ))

    if 0 <= seek_to <= set_duration:
        fig.add_vline(x=seek_to, line_dash="dash", line_color="#ff3366", line_width=2)

    fig.update_layout(
        barmode="overlay",
        bargap=0.15,
        height=max(220, 48 * n_lanes + 100),
        margin=dict(l=72, r=24, t=16, b=44),
        xaxis=dict(
            title="mix time (s)",
            range=[x0, x1],
            showgrid=True, gridcolor="#222",
            rangeslider=dict(visible=True, thickness=0.06, bgcolor="#1a1d24"),
        ),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        dragmode="pan",
    )

    st.plotly_chart(
        fig, width="stretch", theme=None,
        config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )

    # Playhead + mix audio
    pc1, pc2 = st.columns([4, 1])
    with pc1:
        new_seek = st.slider(
            "Playhead (s)",
            min_value=0, max_value=int(set_duration) or 1, value=seek_to, step=1,
            key="align_seek_slider",
        )
    with pc2:
        if st.button("Jump", use_container_width=True):
            st.session_state["align_seek"] = int(new_seek)
            st.rerun()

    if set_audio_path and Path(set_audio_path).exists():
        st.audio(set_audio_path, start_time=int(seek_to))


def _clean_clip_label(raw: str) -> str:
    """Strip tracklist cue-time prefixes and trailing scrobble counters so
    the in-bar label reads like 'artist — title'. Keeps it under 60 chars."""
    import re as _re
    s = raw or ""
    # Strip leading 'w/ ', cue-times, data-attributes noise.
    s = _re.sub(r"^\s*(?:w/\s+)?\d{1,2}:\d{2}(?::\d{2})?\s*", "", s)
    # Drop trailing scrobble/reaction tail ("VIRGIN 305 wouterke (46.6k) S…").
    s = _re.split(r"\s+\d+\s+[a-z]\w+\s+\(\d", s, maxsplit=1)[0]
    s = s.strip(" -·")
    return s[:60] + ("…" if len(s) > 60 else "")


@st.cache_data(ttl=3600)
def _corpus_counts() -> pd.DataFrame:
    conn = _connect()
    return pd.read_sql_query(
        """
        SELECT 'dj_sets' AS t, COUNT(*) n FROM dj_sets
        UNION ALL SELECT 'dj_set_rows',              COUNT(*) FROM dj_set_rows
        UNION ALL SELECT 'dj_set_media_links',       COUNT(*) FROM dj_set_media_links
        UNION ALL SELECT 'dj_set_track_media_links', COUNT(*) FROM dj_set_track_media_links
        UNION ALL SELECT 'scrape_failures',          COUNT(*) FROM scrape_failures
        """, conn,
    )


# ---------- missing-audio helpers (shared between pages) ---------------------
# Hoisted so the "Alignment review" page can render an inline "not aligned"
# panel (skipped tracks + URL entry) without duplicating the parse/insert
# flow that lives on the "Missing audio" page.

def _parse_media_url(raw: str) -> tuple[str, str] | None:
    """Returns (platform, player_id) or None if we can't parse.
    Mirrors the rules that `dj_set_track_media_links` was scraped with:
    11-char YouTube IDs from `?v=`/`youtu.be/`, numeric SoundCloud IDs
    from `api.soundcloud.com/tracks/<n>`, else the SC slug."""
    import re as _re
    raw = (raw or "").strip()
    if not raw:
        return None
    m = _re.search(r"[?&]v=([A-Za-z0-9_-]{11})", raw) or \
        _re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", raw)
    if m:
        return "youtube", m.group(1)
    m = _re.search(r"api\.soundcloud\.com/tracks/(\d+)", raw)
    if m:
        return "soundcloud", m.group(1)
    if "soundcloud.com/" in raw:
        return "soundcloud", raw.rstrip("/").split("/")[-1]
    return None


def _render_not_aligned_panel(set_id: str, aligned_df: pd.DataFrame) -> None:
    """Show the tracklist rows the SOTA aligner dropped for this set,
    with the reason and an inline form to paste a YouTube/SoundCloud
    URL. 'no_url' = scraper never captured a downloadable link (user
    must find one manually — e.g. Barenaked Ladies - One Week only
    had a Spotify row). 'not_downloaded' = link exists but the audio
    asset hasn't been fetched yet (just run the downloader)."""
    from audio_pipeline.adapters import db as db_adapter
    from audio_pipeline.adapters.downloader import DownloadConfig, download_one
    from audio_pipeline.models import (
        MediaSource, soundcloud_api_url, youtube_url,
    )

    tracks_dir = Path.home() / "Desktop" / "tracklist_audio_drive" / "tracks"

    bb = _load_bb()
    tokens = bb["tokens"]
    set_tokens = tokens[
        (tokens["set_id"] == set_id)
        & (tokens["row_kind"] == "track")
        & tokens["track_key"].notna()
    ]

    conn = _connect()
    have_audio = {
        r["track_id"] for r in conn.execute(
            "SELECT DISTINCT ta.track_id FROM track_audio ta "
            "JOIN dj_set_track_media_links tml ON tml.track_id = ta.track_id "
            "WHERE tml.set_id = ?",
            (set_id,),
        ).fetchall()
    }
    have_url = {
        r["track_id"] for r in conn.execute(
            "SELECT DISTINCT track_id FROM dj_set_track_media_links "
            "WHERE set_id = ? AND platform IN ('youtube','soundcloud') "
            "AND track_id IS NOT NULL AND track_id != ''",
            (set_id,),
        ).fetchall()
    }

    rows: list[dict] = []
    seen: set[str] = set()
    for tr in set_tokens.itertuples(index=False):
        tid = str(getattr(tr, "track_key", "") or "")
        if not tid or tid in seen or tid in have_audio:
            continue
        seen.add(tid)
        label = str(
            getattr(tr, "full_name", None) or getattr(tr, "title", None) or tid
        )
        rows.append({
            "row": int(getattr(tr, "row_index", -1)),
            "track_id": tid,
            "label": label,
            "reason": "no_url" if tid not in have_url else "not_downloaded",
        })

    st.markdown("### Tracks not aligned — no audio")
    if not rows:
        st.caption("Every tracklist row has downloaded audio. ✓")
        return

    miss_df = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    n_no_url = int((miss_df["reason"] == "no_url").sum())
    n_not_dl = int((miss_df["reason"] == "not_downloaded").sum())
    st.caption(
        f"{len(miss_df)} tracks were skipped. "
        f"**{n_no_url}** have no YouTube/SoundCloud URL (paste one below "
        f"to recover), **{n_not_dl}** have a URL but no audio file yet "
        "(run the downloader)."
    )
    st.dataframe(
        miss_df[["row", "label", "reason", "track_id"]],
        width="stretch", hide_index=True,
        column_config={
            "row":      st.column_config.NumberColumn("row", format="%d"),
            "label":    st.column_config.TextColumn("song"),
            "reason":   st.column_config.TextColumn("reason"),
            "track_id": st.column_config.TextColumn("track_id"),
        },
    )

    st.markdown("**Add a URL for one of these**")
    pick_i = st.selectbox(
        "Track", list(range(len(miss_df))),
        format_func=lambda i: (
            f"row {int(miss_df.at[i, 'row'])} · "
            f"{str(miss_df.at[i, 'label'])[:70]} "
            f"[{miss_df.at[i, 'reason']}]"
        ),
        key=f"align_missing_pick_{set_id}",
    )
    picked_tid = str(miss_df.at[pick_i, "track_id"])
    picked_label = str(miss_df.at[pick_i, "label"])
    url = st.text_input(
        "Paste YouTube or SoundCloud URL",
        placeholder="https://www.youtube.com/watch?v=... or https://soundcloud.com/...",
        key=f"align_missing_url_{set_id}",
    )
    go_download = st.checkbox(
        "Download immediately after registering the link",
        value=True, key=f"align_missing_go_{set_id}",
    )

    if st.button(
        "Register URL", type="primary", disabled=not url,
        key=f"align_missing_submit_{set_id}",
    ):
        parsed = _parse_media_url(url)
        if parsed is None:
            st.error("Couldn't detect platform. Expected a YouTube or SoundCloud URL.")
            return
        platform, player_id = parsed
        r = db_adapter.insert_track_media_link(
            DB_PATH, set_id=set_id, track_id=picked_tid,
            platform=platform, player_id=player_id, url=url, tlp_id=None,
        )
        if not r.is_ok():
            st.error(f"DB insert failed: {r.error.kind} — {r.error.detail}")
            return
        st.success(f"Registered {platform} · {player_id} for '{picked_label}'.")
        if go_download:
            out_dir = tracks_dir / set_id
            out_dir.mkdir(parents=True, exist_ok=True)
            can_url = (
                youtube_url(player_id) if platform == "youtube"
                else soundcloud_api_url(player_id)
            )
            src = MediaSource(platform=platform, player_id=player_id, url=can_url)
            with st.spinner(f"Downloading via yt-dlp → {out_dir}..."):
                dl = download_one(picked_tid, src, DownloadConfig(out_dir=out_dir))
            if not dl.is_ok():
                st.error(f"Download failed: {dl.error.kind} — {dl.error.detail}")
                return
            ins = db_adapter.insert_audio(DB_PATH, dl.value)
            if not ins.is_ok():
                st.error(f"insert_audio failed: {ins.error}")
                return
            st.success(
                f"Downloaded → {dl.value.path}. Re-run "
                "`audio_pipeline.alignment.sota` to pick up the new ref."
            )
        st.cache_data.clear()
        st.rerun()


# ---------- sidebar / page routing ------------------------------------------

# Pages are grouped like a DAW browser: catalog views up top, playback /
# analysis next, annotation at the bottom.
_PAGE_GROUPS: list[tuple[str, list[str]]] = [
    ("Catalog",  ["Overview", "Set browser", "Set detail"]),
    ("Audio",    ["Missing audio", "Stem player"]),
    ("Analysis", ["Alignment review", "Annotate GT"]),
]
PAGES = [p for _, ps in _PAGE_GROUPS for p in ps]

# Sidebar header — mimics Ableton's browser title strip.
st.sidebar.markdown(
    """
    <div style="padding: .3rem .1rem .9rem .1rem; border-bottom: 1px solid var(--line); margin-bottom: .8rem;">
      <div style="font-family: var(--mono); font-size: 13px; font-weight: 700;
                  letter-spacing: .22em; color: var(--accent);">▶ TRACKLIST</div>
      <div style="font-family: var(--mono); font-size: 9.5px; letter-spacing: .18em;
                  color: var(--text-faint); text-transform: uppercase;">Engine · Big Bootie pilot</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Persist the active page across reruns so that deep-links + jump-buttons work.
if "active_page" not in st.session_state:
    st.session_state["active_page"] = PAGES[0]

for group_name, group_pages in _PAGE_GROUPS:
    st.sidebar.markdown(
        f'<div style="font-family: var(--mono); font-size: 10px; letter-spacing: .18em; '
        f'text-transform: uppercase; color: var(--text-faint); margin: .6rem 0 .25rem 0;">'
        f'{group_name}</div>',
        unsafe_allow_html=True,
    )
    for p in group_pages:
        is_active = st.session_state["active_page"] == p
        # Streamlit buttons inside the sidebar get our pill-button styling;
        # the ::before glyph cues an Ableton-style selection arrow when active.
        label = ("● " if is_active else "○ ") + p
        if st.sidebar.button(label, key=f"nav_{p}", width="stretch",
                              type=("primary" if is_active else "secondary")):
            st.session_state["active_page"] = p
            st.rerun()

page = st.session_state["active_page"]

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.caption(f"DB · {DB_PATH.name}")
st.sidebar.caption("Scope · Big Bootie only")


# ---------- OVERVIEW ---------------------------------------------------------

if page == "Overview":
    _te_topbar("Overview", DB_PATH)
    st.title("Corpus overview")

    counts = _corpus_counts()
    cols = st.columns(len(counts))
    for col, (_, r) in zip(cols, counts.iterrows()):
        col.metric(r["t"], f'{int(r["n"]):,}')

    bb = _load_bb()
    tokens = bb["tokens"]
    tracks = tokens[tokens["row_kind"] == "track"]

    st.subheader("Big Bootie pilot — token breakdown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("sets", f"{len(bb['sets']):,}")
    c2.metric("distinct canonical tracks", f"{tracks['track_key'].dropna().nunique():,}")
    c3.metric("ided %", f'{tracks["is_ided"].fillna(False).mean()*100:.1f}')
    c4.metric("concurrent %", f'{tracks["is_concurrent"].fillna(False).mean()*100:.1f}')

    st.subheader("Layer depth per volume")
    per_set_cues   = tracks.dropna(subset=["cue_seconds_section"]).groupby("set_id")["cue_seconds_section"].nunique().rename("distinct_cues")
    per_set_tracks = tracks.groupby("set_id").size().rename("track_rows")
    df_set = (
        per_set_tracks.to_frame()
                      .join(per_set_cues, how="left")
                      .join(bb["sets"].set_index("set_id")[["volume","date_played"]])
                      .sort_values("volume")
    )
    df_set["layer_ratio"] = df_set["track_rows"] / df_set["distinct_cues"]
    st.bar_chart(df_set.dropna(subset=["volume"]).set_index("volume")["layer_ratio"])

    cueless = df_set[df_set["distinct_cues"].isna()]
    if len(cueless):
        st.warning(f"{len(cueless)} of {len(df_set)} sets have no scraped cue timestamps "
                   "— see the Set detail page to inspect their raw HTML.")
        st.dataframe(cueless[["volume", "date_played", "track_rows"]])


# ---------- SET BROWSER ------------------------------------------------------

elif page == "Set browser":
    _te_topbar("Set browser", DB_PATH)
    st.title("Set browser")
    bb = _load_bb()
    tokens = bb["tokens"]
    tracks = tokens[tokens["row_kind"] == "track"]

    # Per-set aggregates
    per_track = tracks.groupby("set_id").agg(
        track_rows=("row_index", "size"),
        pct_ided=("is_ided", lambda s: s.fillna(False).mean() * 100),
        pct_concurrent=("is_concurrent", lambda s: s.fillna(False).mean() * 100),
        pct_remixish=("is_remixish", lambda s: s.fillna(False).mean() * 100),
        distinct_cues=("cue_seconds_section", "nunique"),
    )

    # Per-set downloadability (YT or SC on the track_media_links table)
    tml_per_track = bb["tml"].dropna(subset=["track_id"]).groupby("track_id")["platform"].apply(set)
    tracks = tracks.copy()
    tracks["downloadable"] = tracks["track_key"].map(tml_per_track).apply(
        lambda p: isinstance(p, set) and bool(p & {"youtube", "soundcloud"}))
    down = tracks.groupby("set_id")["downloadable"].agg(["size", "sum"])
    down["pct_downloadable"] = (100 * down["sum"] / down["size"]).round(1)

    browser = (
        bb["sets"].set_index("set_id")[["volume", "date_played", "title", "play_time"]]
                  .join(per_track)
                  .join(down["pct_downloadable"])
                  .reset_index()
                  .sort_values("volume")
    )

    # Corpus-level summary chips (Serato library-bar feel) -----------------
    _te_chips([
        ("sets", f"{len(browser):,}", ""),
        ("tracks", f"{int(browser['track_rows'].fillna(0).sum()):,}", ""),
        ("avg IDed", f"{browser['pct_ided'].mean():.0f}%", "ok"),
        ("cueless sets", f"{int(browser['distinct_cues'].isna().sum())}",
            "warn" if browser['distinct_cues'].isna().any() else ""),
    ])

    # Filter bar -----------------------------------------------------------
    fc1, fc2, fc3, fc4 = st.columns([2.2, 1.2, 1.2, 1.4])
    vol_min = int(browser["volume"].dropna().min() or 0)
    vol_max = int(browser["volume"].dropna().max() or 0)
    with fc1:
        vol_range = st.slider("Volume", vol_min, vol_max, (vol_min, vol_max),
                               label_visibility="visible")
    with fc2:
        only_cueless = st.checkbox("Cueless only", value=False,
                                    help="Sets with no scraped cue timestamps")
    with fc3:
        only_undl = st.checkbox("< 50% downloadable", value=False)
    with fc4:
        search = st.text_input("Title search", placeholder="filter by title…",
                                label_visibility="visible")

    view = browser.copy()
    view = view[(view["volume"].fillna(-1).between(*vol_range)) | view["volume"].isna()]
    if only_cueless:
        view = view[view["distinct_cues"].isna()]
    if only_undl:
        view = view[view["pct_downloadable"].fillna(0) < 50]
    if search:
        view = view[view["title"].fillna("").str.contains(search, case=False, na=False)]

    # Render — use st.dataframe for density + sortability, styled via CSS.
    # Column config tightens the display and adds progress bars for the
    # pct_* columns (Ableton meter feel).
    st.dataframe(
        view,
        width="stretch",
        height=620,
        hide_index=True,
        column_config={
            "set_id":          st.column_config.TextColumn("set_id", width="small"),
            "volume":          st.column_config.NumberColumn("vol", format="%d", width="small"),
            "date_played":     st.column_config.TextColumn("date", width="small"),
            "title":           st.column_config.TextColumn("title", width="large"),
            "play_time":       st.column_config.TextColumn("dur", width="small"),
            "track_rows":      st.column_config.NumberColumn("rows", format="%d", width="small"),
            "distinct_cues":   st.column_config.NumberColumn("cues", format="%d", width="small"),
            "pct_ided":        st.column_config.ProgressColumn(
                "IDed", format="%.0f%%", min_value=0, max_value=100, width="small"),
            "pct_concurrent":  st.column_config.ProgressColumn(
                "concurrent", format="%.0f%%", min_value=0, max_value=100, width="small"),
            "pct_remixish":    st.column_config.ProgressColumn(
                "remix", format="%.0f%%", min_value=0, max_value=100, width="small"),
            "pct_downloadable":st.column_config.ProgressColumn(
                "DL %", format="%.0f%%", min_value=0, max_value=100, width="small"),
        },
    )
    st.caption(f"{len(view):,} of {len(browser):,} sets · sortable columns · copy a set_id → Set detail")


# ---------- MISSING AUDIO ----------------------------------------------------

elif page == "Missing audio":
    st.title("Missing audio")
    st.caption(
        "Tracks we can't auto-download because no YouTube/SoundCloud link was "
        "scraped. Paste a URL to add it, and the file will be fetched to the "
        "audio drive. Optionally also downloads it immediately."
    )

    from audio_pipeline.adapters import db as db_adapter
    from audio_pipeline.adapters.downloader import DownloadConfig, download_one
    from audio_pipeline.models import (
        MediaSource, soundcloud_api_url, youtube_url,
    )

    MOCK_DRIVE = Path.home() / "Desktop" / "tracklist_audio_drive"
    TRACKS_DIR = MOCK_DRIVE / "tracks"

    # Scope picker
    bb = _load_bb()
    sets_df = bb["sets"].sort_values("volume")
    set_labels = ["— all Big Bootie sets —"] + [
        f'Vol {int(v) if pd.notna(v) else "?"} — {sid}'
        for sid, v in zip(sets_df["set_id"], sets_df["volume"])
    ]
    pick = st.sidebar.selectbox("Scope", set_labels)
    scope_set_id = None if pick.startswith("— all") else pick.split("—")[-1].strip()

    conn = _connect()

    # A track_id is missing if either:
    #   (a) it has no YT/SC row in dj_set_track_media_links, OR
    #   (b) it has a YT/SC row but no track_audio record downloaded yet.
    base_sql = """
        SELECT l.set_id, l.tlp_id, l.track_id,
               MAX(CASE WHEN l.platform = 'youtube'   THEN 1 ELSE 0 END) AS has_yt,
               MAX(CASE WHEN l.platform = 'soundcloud' THEN 1 ELSE 0 END) AS has_sc,
               MAX(CASE WHEN ta.track_audio_id IS NOT NULL THEN 1 ELSE 0 END) AS has_audio
        FROM dj_set_track_media_links l
        LEFT JOIN track_audio ta ON ta.track_id = l.track_id
        WHERE l.track_id IS NOT NULL AND l.track_id != ''
    """
    params: list[object] = []
    if scope_set_id:
        base_sql += " AND l.set_id = ?"
        params.append(scope_set_id)
    base_sql += " GROUP BY l.set_id, l.tlp_id, l.track_id"

    rows = conn.execute(base_sql, params).fetchall()
    cols = ["set_id", "tlp_id", "track_id", "has_yt", "has_sc", "has_audio"]
    miss = pd.DataFrame(rows, columns=cols)
    miss = miss[(miss["has_yt"] == 0) & (miss["has_sc"] == 0) & (miss["has_audio"] == 0)].copy()

    # Join titles / artists from tokenized rows
    tokens = bb["tokens"]
    tcols = ["set_id", "row_dom_id", "title", "artists", "track_key"]
    tok_small = tokens[tokens["row_kind"] == "track"][tcols].rename(
        columns={"row_dom_id": "tlp_id", "track_key": "track_id"}
    )
    miss = miss.merge(tok_small, on=["set_id", "tlp_id", "track_id"], how="left")

    show_ided = st.checkbox("Only IDed rows (has a title)", value=True)
    if show_ided:
        miss = miss[miss["title"].notna() & (miss["title"] != "")]

    st.metric("Missing tracks in scope", len(miss))
    if miss.empty:
        st.success("Nothing missing in this scope — every track has either a "
                   "scraped YT/SC link or a downloaded audio file.")
        st.stop()

    st.dataframe(
        miss[["set_id", "track_id", "title", "artists"]].reset_index(drop=True),
        width="stretch", height=340,
    )

    st.markdown("### Add a URL for a missing track")
    pick_idx = st.selectbox(
        "Track", miss.index,
        format_func=lambda i: (
            f"{miss.at[i, 'track_id']}  ·  "
            f"{(miss.at[i, 'artists'] or '').replace('|', ', ')} — "
            f"{miss.at[i, 'title'] or '(no title)'}  ·  set {miss.at[i, 'set_id']}"
        ),
    )
    picked = miss.loc[pick_idx]

    url = st.text_input(
        "Paste YouTube or SoundCloud URL",
        placeholder="https://www.youtube.com/watch?v=... or https://soundcloud.com/...",
    )
    go_download = st.checkbox("Download immediately after registering the link", value=True)

    if st.button("Register URL", type="primary", disabled=not url):
        parsed = _parse_media_url(url)
        if parsed is None:
            st.error("Couldn't detect platform. Expected a YouTube or SoundCloud URL.")
        else:
            platform, player_id = parsed
            r = db_adapter.insert_track_media_link(
                DB_PATH,
                set_id=str(picked["set_id"]),
                track_id=str(picked["track_id"]),
                platform=platform,
                player_id=player_id,
                url=url,
                tlp_id=(str(picked["tlp_id"]) if pd.notna(picked["tlp_id"]) else None),
            )
            if not r.is_ok():
                st.error(f"DB insert failed: {r.error.kind} — {r.error.detail}")
            else:
                st.success(f"Registered {platform} · {player_id}.")
                if go_download:
                    out_dir = TRACKS_DIR / str(picked["set_id"])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    can_url = youtube_url(player_id) if platform == "youtube" else soundcloud_api_url(player_id)
                    src = MediaSource(platform=platform, player_id=player_id, url=can_url)
                    with st.spinner(f"Downloading via yt-dlp → {out_dir}..."):
                        dl = download_one(str(picked["track_id"]), src, DownloadConfig(out_dir=out_dir))
                    if not dl.is_ok():
                        st.error(f"Download failed: {dl.error.kind} — {dl.error.detail}")
                    else:
                        asset = dl.value
                        ins = db_adapter.insert_audio(DB_PATH, asset)
                        if not ins.is_ok():
                            st.error(f"DB insert_audio failed: {ins.error}")
                        else:
                            st.success(f"Downloaded → {asset.path}")
                st.cache_data.clear()   # refresh the missing list on next rerun
                st.rerun()


# ---------- ALIGNMENT REVIEW -------------------------------------------------

elif page == "Alignment review":
    st.title("Alignment review")
    st.caption(
        "Listen to the set window and the aligned reference track at the "
        "timestamps Stage-1 DTW produced, and judge by ear whether it lines up."
    )

    aligned_sets = _aligned_sets_summary()

    if not aligned_sets:
        st.info(
            "No alignments yet. Run `python -m audio_pipeline.align_main --set-id <id>` "
            "once downloads finish. The page will populate automatically."
        )
    else:
        labels = [
            f"{r['title']}  ·  {r['n_sections']} sections  ·  match≈{(r['mean_match'] or 0):.2f}"
            for r in aligned_sets
        ]
        picked_idx = st.sidebar.selectbox(
            "Aligned set", range(len(aligned_sets)),
            format_func=lambda i: labels[i],
        )
        picked_row = aligned_sets[picked_idx]
        set_id      = picked_row["set_id"]
        title       = picked_row["title"]
        n_sections  = picked_row["n_sections"]
        mean_match  = picked_row["mean_match"]

        st.subheader(title)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("set_id", set_id)
        mc2.metric("aligned sections", n_sections)
        mc3.metric("mean match rate", f"{mean_match:.3f}" if mean_match else "—")

        set_audio_path = _set_audio_path_for(set_id)

        # SOTA-only display. Reads rows written by the canonical
        # orchestrator in audio_pipeline/alignment/sota.py, tagged
        # `confidence_source='sota_v2'`. See docs/SOTA.md for the stack.
        df = _aligned_rows_for(set_id, source="sota_v2")
        st.caption(
            f"SOTA v2 (MERT + cue-detr bracket + mutual exclusion) — **{len(df)}** rows · "
            "`confidence_source='sota_v2'`"
            if len(df) > 0 else
            f"No SOTA rows for this set yet. Run "
            "`venvs/audio/bin/python -m audio_pipeline.alignment.sota --set-id <set_id>` "
            "to populate."
        )

        # --- Ableton-style layered timeline (interactive HTML+JS component) ---
        _render_ableton_timeline(df, set_audio_path)

        st.markdown("### Aligned sections")

        # Stem-mask distribution summary
        mask_counts = df["stem_mask"].value_counts()
        if not mask_counts.empty:
            cols = st.columns(len(mask_counts))
            for col, (k, v) in zip(cols, mask_counts.items()):
                col.metric(k, int(v))

        st.dataframe(
            df[["section_idx", "set_start_s", "set_end_s", "transposition",
                "bpm_ratio", "match_rate", "stem_mask", "ref_track_id", "label"]],
            width="stretch", height=340,
            column_config={
                "set_start_s": st.column_config.NumberColumn("set start (s)", format="%.1f"),
                "set_end_s":   st.column_config.NumberColumn("set end (s)",   format="%.1f"),
                "bpm_ratio":   st.column_config.NumberColumn("bpm ratio",     format="%.3f"),
                "match_rate":  st.column_config.ProgressColumn("match", min_value=0.0, max_value=1.0),
                "stem_mask":   st.column_config.TextColumn("stem mask"),
            },
        )

        # --- Tracks not aligned (no audio) ---------------------------
        # Tracklist rows that SOTA dropped because there's no downloadable
        # audio. Either the scraper never captured a YT/SC URL ('no_url'
        # — user has to paste one manually) or the URL exists but the
        # file hasn't been fetched yet ('not_downloaded' — just run the
        # downloader). Keep this adjacent to the aligned table so users
        # reconciling the tracklist against the timeline see the gaps.
        _render_not_aligned_panel(set_id, df)

        st.markdown("### ▶ Playback")
        pick = st.selectbox(
            "Section to audit",
            df.index,
            format_func=lambda i: (
                f"row {int(df.at[i, 'section_idx'])}  ·  "
                f"set {df.at[i, 'set_start_s']:.0f}–{df.at[i, 'set_end_s']:.0f}s  ·  "
                f"match {df.at[i, 'match_rate']:.2f}  ·  "
                f"{df.at[i, 'label'][:70] or df.at[i, 'ref_track_id']}"
            ),
        )
        row = df.loc[pick]

        c_meta, c_ref = st.columns([1, 1])
        with c_meta:
            st.markdown("**Mix-side window**")
            st.caption(f"{row['set_start_s']:.1f}s — {row['set_end_s']:.1f}s  "
                       f"(duration {row['set_end_s']-row['set_start_s']:.1f}s)")
            if set_audio_path and Path(set_audio_path).exists():
                st.audio(set_audio_path, start_time=int(row["set_start_s"]))
            else:
                st.warning(f"set_audio not on disk: {set_audio_path}")
        with c_ref:
            st.markdown("**Reference track**")
            bpm_str = f"{row['bpm_ratio']:.3f}" if row["bpm_ratio"] is not None else "?"
            trans_str = (f"{int(row['transposition'])}"
                         if row["transposition"] is not None else "?")
            st.caption(
                f"track_id `{row['ref_track_id']}`  ·  "
                f"Δ={trans_str} semitones  ·  "
                f"bpm ratio {bpm_str}  ·  "
                f"**{row['stem_mask']}**"
            )
            if pd.notna(row["ref_path"]) and Path(row["ref_path"]).exists():
                st.audio(row["ref_path"], start_time=0)
            else:
                st.warning(f"ref audio missing: {row['ref_path']}")

        # Per-stem match-rate bars
        import json as _json
        stem_rates_raw = row["stem_rates_json"]
        if isinstance(stem_rates_raw, str) and stem_rates_raw:
            try:
                rates = _json.loads(stem_rates_raw)
            except _json.JSONDecodeError:
                rates = {}
        else:
            rates = {}
        if rates:
            stem_df = pd.DataFrame(
                [{"stem": k, "match_rate": v} for k, v in rates.items()]
            ).sort_values("match_rate", ascending=False)
            st.markdown("**Per-stem match rates**")
            st.dataframe(
                stem_df,
                width="stretch", height=200, hide_index=True,
                column_config={
                    "stem":       st.column_config.TextColumn("stem"),
                    "match_rate": st.column_config.ProgressColumn(
                        "match", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
            )
            st.caption(
                "Stage-1 ran 5 DTWs: one on the whole audio plus one per "
                "Demucs stem. The winning stem (top of this table) provided "
                "the warping path used for the rest of the row's data."
            )

        st.caption(
            "Listening check: the mix-side clip should sound like the ref-side "
            "clip pitch-shifted by Δ semitones and time-stretched by bpm_ratio. "
            "Match rate ≥0.4 is the paper's 'aligned' threshold."
        )

        # --- Measure-level cutup plan ---
        import json as _json
        raw_plan = row.get("cutup_plan_json")
        if isinstance(raw_plan, str) and raw_plan:
            try:
                plan = _json.loads(raw_plan)
            except _json.JSONDecodeError:
                plan = []
        else:
            plan = []

        if plan:
            st.markdown("### 🧩 Cutup plan  (stage-5 measure refinement)")

            # Flag ref-jumps: a segment whose ref_start is not ref_end+1 of
            # the previous segment = the DJ jumped non-contiguously in the ref.
            prev_ref_end: int | None = None
            enriched: list[dict] = []
            n_loops = 0
            n_jumps = 0
            for s in plan:
                is_jump = (
                    prev_ref_end is not None
                    and s["ref_measure_start"] != prev_ref_end + 1
                )
                if s["repeat_count"] > 1:
                    n_loops += 1
                if is_jump:
                    n_jumps += 1
                enriched.append({
                    "ref": f"{s['ref_measure_start']}..{s['ref_measure_end']}",
                    "set": f"{s['set_measure_start']}..{s['set_measure_end']}",
                    "rep": s["repeat_count"],
                    "jump": "↯" if is_jump else "",
                })
                prev_ref_end = s["ref_measure_end"]

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("segments",   len(plan))
            sc2.metric("held/looped", n_loops)
            sc3.metric("ref jumps",  n_jumps)

            st.dataframe(
                pd.DataFrame(enriched),
                width="stretch", height=260,
                column_config={
                    "ref":  st.column_config.TextColumn("ref measures"),
                    "set":  st.column_config.TextColumn("set measures"),
                    "rep":  st.column_config.NumberColumn("repeats", format="%d×"),
                    "jump": st.column_config.TextColumn(
                        "cutup?", help="↯ = ref measure is not contiguous with the previous segment"
                    ),
                },
            )
            st.caption(
                "Each row: ref measures X..Y played as set measures A..B. "
                "`repeats` > 1 means the DJ held or looped that ref range across "
                "multiple set measures. `↯` marks discontinuous ref jumps."
            )
        else:
            st.caption(
                "🧩 No cutup plan yet — either this row aligned before measure "
                "refinement was wired in (Viterbi path doesn't populate it), or "
                "its ref track / the set mix hasn't been analyzed by beat_this. "
                "Re-run CCC alignment to populate."
            )


# ---------- SET DETAIL -------------------------------------------------------

elif page == "Set detail":
    _te_topbar("Set detail", DB_PATH)
    bb = _load_bb()
    tokens = bb["tokens"]

    sets = bb["sets"].sort_values("volume")
    labels = [f'Vol {int(v) if pd.notna(v) else "?"} — {sid}' for sid, v in zip(sets["set_id"], sets["volume"])]
    label_to_sid = dict(zip(labels, sets["set_id"]))
    picked = st.sidebar.selectbox("Set", labels, index=len(labels) - 1)
    set_id = label_to_sid[picked]

    s_meta = sets[sets["set_id"] == set_id].iloc[0]

    # Set header — mimics Ableton's arrangement title strip with a big title,
    # a volume tag, and an accent rule underneath.
    vol_str = f"VOL {int(s_meta['volume'])}" if pd.notna(s_meta["volume"]) else "VOL ?"
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:1rem; margin-bottom:.4rem;">
          <span style="font-family: var(--mono); font-size: 11px; font-weight: 600;
                       letter-spacing: .18em; color: var(--accent);
                       padding: 3px 8px; border: 1px solid var(--accent); border-radius: 2px;">
            {vol_str}
          </span>
          <h1 style="margin: 0; padding: 0; border: none; font-size: 1.6rem;">{s_meta["title"]}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _te_chips([
        ("set_id",   set_id, ""),
        ("date",     s_meta["date_played"] or "—", ""),
        ("play time", s_meta.get("play_time") or "—", ""),
        ("tracks",   int(s_meta["total_tracks"] or 0), ""),
    ])

    # --- Set-level playable ------------------------------------------------
    # Reset seek when switching sets.
    if st.session_state.get("_current_set") != set_id:
        st.session_state["_current_set"] = set_id
        st.session_state["seek_to"] = 0

    conn_for_audio = _connect()
    set_audio_row = conn_for_audio.execute(
        "SELECT path, source_url, platform, duration_s FROM set_audio "
        "WHERE set_id = ? ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1",
        (set_id,),
    ).fetchone()

    # Fall back to the first set-level media link (YT preferred) as an embedded iframe.
    set_link_row = None
    if not set_audio_row or not Path(set_audio_row[0]).exists():
        from audio_pipeline.models import normalize_set_media_url
        set_link_row = conn_for_audio.execute(
            "SELECT platform, url FROM dj_set_media_links WHERE set_id = ? "
            "ORDER BY CASE platform WHEN 'youtube' THEN 1 WHEN 'soundcloud' THEN 2 ELSE 3 END LIMIT 1",
            (set_id,),
        ).fetchone()

    st.markdown("### ▶ Set playback")
    seek = int(st.session_state.get("seek_to") or 0)

    if set_audio_row and Path(set_audio_row[0]).exists():
        path, source_url, platform, duration_s = set_audio_row
        st.audio(path, start_time=seek)
        st.caption(f"Local audio: `{platform}` · {Path(path).name} · seek={seek}s")
    elif set_link_row:
        platform, raw_url = set_link_row
        if platform == "youtube":
            import re as _re
            vid_m = _re.search(r"[?&]v=([A-Za-z0-9_-]{11})", raw_url) or \
                    _re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", raw_url)
            if vid_m:
                embed = f"https://www.youtube.com/embed/{vid_m.group(1)}?start={seek}&autoplay=0"
                st.iframe(embed, height=180)
        elif platform == "soundcloud":
            from audio_pipeline.models import normalize_set_media_url
            inner = normalize_set_media_url(raw_url)
            sc_embed = (
                "https://w.soundcloud.com/player/?url=" + inner +
                f"&auto_play=false&show_artwork=true#t={seek}"
            )
            st.iframe(sc_embed, height=180)
        else:
            st.info(f"Embedded player not supported for `{platform}`; raw URL: {raw_url}")
        st.caption(f"Streaming from {platform}. Download locally with "
                   f"`python -m audio_pipeline.main --set-id {set_id} --mode set`.")
    else:
        st.warning("No set-level media link scraped. Can't play.")

    # Quick seek controls
    sc1, sc2, _ = st.columns([2, 1, 4])
    with sc1:
        new_seek = st.number_input("Jump to (seconds)", min_value=0,
                                   value=int(seek), step=10, key="seek_input")
    with sc2:
        if st.button("Jump"):
            st.session_state["seek_to"] = int(new_seek)
            st.rerun()

    set_tokens = tokens[tokens["set_id"] == set_id]
    tracks = set_tokens[set_tokens["row_kind"] == "track"].copy()

    tml_per_track = bb["tml"].dropna(subset=["track_id"]).groupby("track_id")["platform"].apply(set)
    tracks["platforms"] = tracks["track_key"].map(tml_per_track)
    tracks["downloadable"] = tracks["platforms"].apply(
        lambda p: isinstance(p, set) and bool(p & {"youtube", "soundcloud"}))

    # --- Coverage summary ----------------------------------------------------
    st.subheader("Coverage")
    n = len(tracks)
    n_ided   = int(tracks["is_ided"].fillna(False).sum())
    n_conc   = int(tracks["is_concurrent"].fillna(False).sum())
    n_remix  = int(tracks["is_remixish"].fillna(False).sum())
    n_dl     = int(tracks["downloadable"].fillna(False).sum())
    n_missing = int((~tracks["downloadable"].fillna(False)).sum())
    _te_chips([
        ("tracks",       n, ""),
        ("IDed",         f"{n_ided} ({n_ided/max(n,1)*100:.0f}%)", "ok" if n_ided else "warn"),
        ("concurrent",   n_conc, ""),
        ("remix/acap",   n_remix, ""),
        ("downloadable", f"{n_dl} ({n_dl/max(n,1)*100:.0f}%)",
            "ok" if n_dl/max(n,1) >= 0.8 else "warn" if n_dl else "err"),
        ("missing audio", n_missing,
            "err" if n_missing > n*0.2 else "warn" if n_missing else "ok"),
    ])

    # --- Filter bar ----------------------------------------------------------
    st.subheader("Tracklist")
    filt1, filt2, filt3, filt4, filt5 = st.columns([1.2, 1.2, 1.2, 1.0, 1.4])
    with filt1: show_only_undl       = st.checkbox("Not downloadable")
    with filt2: show_only_remix      = st.checkbox("Remix / acappella")
    with filt3: show_only_concurrent = st.checkbox("Concurrent (`w/`)")
    with filt4: show_only_unided     = st.checkbox("Not IDed")
    with filt5: view_mode            = st.radio("View", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")

    view = tracks.sort_values("row_index").copy()
    # Cast to real bools before bitwise ops — object-dtype columns produce
    # integer results under `~`, which then confuses DataFrame.__getitem__.
    for col in ("downloadable", "is_remixish", "is_concurrent", "is_ided"):
        if col in view.columns:
            view[col] = view[col].fillna(False).astype(bool)

    if show_only_undl:       view = view[~view["downloadable"]]
    if show_only_remix:      view = view[view["is_remixish"]]
    if show_only_concurrent: view = view[view["is_concurrent"]]
    if show_only_unided:     view = view[~view["is_ided"]]
    st.caption(f"{len(view)} of {len(tracks)} tracks shown")

    # --- Render --------------------------------------------------------------
    if view_mode == "Table":
        display_cols = [
            "row_index", "track_number_raw", "artwork_url", "title", "artists",
            "cue_seconds_section", "cue_timecode",
            "is_ided", "is_concurrent", "is_remixish",
            "downloadable", "platforms",
        ]
        display_cols = [c for c in display_cols if c in view.columns]
        st.dataframe(
            view[display_cols],
            width="stretch",
            height=500,
            column_config={
                "artwork_url": st.column_config.ImageColumn("art", width="small"),
                "cue_seconds_section": st.column_config.NumberColumn("cue (s)", format="%d"),
            },
        )
    else:
        # Card layout: group by concurrent layer (same cue_seconds_section)
        def _fmt_cue(cue) -> str:
            if cue is None or (isinstance(cue, float) and pd.isna(cue)):
                return "—"
            s = int(cue)
            h, r = divmod(s, 3600)
            m, s = divmod(r, 60)
            return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

        def _platforms_line(p) -> str:
            if not isinstance(p, set):
                return ""
            order = [("youtube", "YT"), ("soundcloud", "SC"), ("spotify", "SP"),
                     ("apple", "AM"), ("beatport", "BP")]
            tags = [label for key, label in order if key in p]
            return " · ".join(tags)

        def _badges(row) -> str:
            b = []
            if row.get("is_concurrent"): b.append("🔗 w/")
            if row.get("is_remixish"):   b.append("🎛 " + (row.get("version_tag") or "Remix"))
            if not bool(row.get("is_ided") or False): b.append("❓ ID?")
            if not bool(row.get("downloadable") or False): b.append("🚫 no YT/SC")
            return "  ".join(b)

        # Determine the currently-active section (largest cue <= seek_to).
        current_seek = int(st.session_state.get("seek_to") or 0)
        section_cues = sorted(
            {float(c) for c in view["cue_seconds_section"].dropna().unique()}
        )
        active_cue = max(
            (c for c in section_cues if c <= current_seek), default=None
        )

        last_cue = object()
        for _, r in view.iterrows():
            cue = r.get("cue_seconds_section")
            cue_key = None if (cue is None or pd.isna(cue)) else float(cue)
            is_active = (cue_key is not None) and (cue_key == active_cue)
            # Divider between layer groups (same cue section → same audio moment)
            if cue_key != last_cue:
                bg = "#2d4a2f" if is_active else "#1f2a44"
                border = "2px solid #5dba63" if is_active else "1px solid transparent"
                dc1, dc2 = st.columns([8, 1])
                with dc1:
                    st.markdown(
                        f"<div style='margin-top:0.6em;padding:0.3em 0.7em;background:{bg};"
                        f"border:{border};border-radius:6px;color:#cfd;font-size:0.88em;'>"
                        f"{'🔊 ' if is_active else '⏱ '}"
                        f"<b>{_fmt_cue(cue_key)}</b> &nbsp;·&nbsp; section anchor"
                        f"{' &nbsp;·&nbsp; <i>now playing</i>' if is_active else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with dc2:
                    # Only offer ▶ for rows with a real cue; NaN sections can't be seeked.
                    if cue_key is not None:
                        if st.button("▶", key=f"play_{cue_key}_{r['row_index']}",
                                     help=f"Jump to {_fmt_cue(cue_key)}"):
                            st.session_state["seek_to"] = int(cue_key)
                            st.rerun()
                last_cue = cue_key

            c_art, c_info = st.columns([1, 9])
            with c_art:
                url = r.get("artwork_url")
                if isinstance(url, str) and url:
                    st.image(url, width=72)
                else:
                    st.markdown(
                        "<div style='width:72px;height:72px;background:#2a2f3a;border-radius:4px;"
                        "display:flex;align-items:center;justify-content:center;color:#666;"
                        "font-size:0.7em;'>no art</div>",
                        unsafe_allow_html=True,
                    )
            with c_info:
                def _as_str(v) -> str:
                    # Coerce to str, handling NaN / None cleanly (avoid float NaN leaks).
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return ""
                    return str(v)

                trno = _as_str(r.get("track_number_raw"))
                title = _as_str(r.get("title")) or "(untitled)"
                artists_raw = r.get("artists")
                if isinstance(artists_raw, str):
                    artists_line = artists_raw.replace("|", ", ")
                elif isinstance(artists_raw, (list, tuple)):
                    artists_line = ", ".join(map(str, artists_raw))
                else:
                    artists_line = ""

                genre = _as_str(r.get("genre"))
                duration = r.get("duration_seconds")
                dur_line = (
                    f"{int(duration)//60}:{int(duration)%60:02d}"
                    if isinstance(duration, (int, float)) and pd.notna(duration) else ""
                )
                plat = _platforms_line(r.get("platforms"))
                badges = _badges(r)

                header = f"**`{trno or '·'}`**  {artists_line} — **{title}**"
                st.markdown(header)
                meta_bits = [b for b in (genre, dur_line, plat) if isinstance(b, str) and b]
                if meta_bits:
                    st.caption("  ·  ".join(meta_bits))
                if badges:
                    st.markdown(
                        f"<span style='color:#ccc;font-size:0.85em;'>{badges}</span>",
                        unsafe_allow_html=True,
                    )
                # Per-track YouTube quick-link (opens on youtube.com — fallback
                # for when we haven't downloaded per-track audio yet).
                platforms = r.get("platforms")
                if isinstance(platforms, set) and "youtube" in platforms:
                    yt_row = conn_for_audio.execute(
                        "SELECT player_id FROM dj_set_track_media_links "
                        "WHERE set_id = ? AND track_id = ? AND platform = 'youtube' LIMIT 1",
                        (set_id, r.get("track_key")),
                    ).fetchone()
                    if yt_row and yt_row[0]:
                        st.markdown(
                            f"[▶ play on YouTube](https://www.youtube.com/watch?v={yt_row[0]})",
                            unsafe_allow_html=True,
                        )
            st.markdown("<hr style='margin:0.3em 0;border:none;border-top:1px solid #222;'/>",
                        unsafe_allow_html=True)

    # --- Raw HTML inspector (collapsed by default) ---------------------------
    with st.expander("Raw HTML inspector (for debugging individual rows)"):
        html_matches = sorted(HTML_DIR.glob(f"{set_id}_*.html"))
        if html_matches:
            path = html_matches[-1]
            st.caption(f'Stored HTML: `{path.relative_to(_REPO_ROOT)}` ({path.stat().st_size:,} bytes)')
            html = path.read_text(encoding="utf-8", errors="ignore")
            cues = extract_cue_points_from_html(html)
            mc1, mc2 = st.columns(2)
            mc1.metric("cue entries in HTML", len(cues))
            mc2.metric("max cue time (s)", max((c["time_seconds"] for c in cues), default=0))
            pick_row = st.number_input(
                "row_index:", min_value=0,
                max_value=int(set_tokens["row_index"].max()), value=1,
            )
            conn = _connect()
            row = conn.execute(
                "SELECT raw_html, text_excerpt, classes FROM dj_set_rows WHERE set_id = ? AND row_index = ?",
                (set_id, int(pick_row)),
            ).fetchone()
            if row:
                raw_html, text_excerpt, classes = row
                st.write(f'**Classes:** `{classes}`')
                st.write(f'**Text excerpt:** {text_excerpt}')
                st.write(f'**Classify:** `{classify_row(raw_html)}`')
                st.code(raw_html, language="html")
            else:
                st.info("No row at that index.")
        else:
            st.warning(f"No stored HTML at `data/html/{set_id}_*.html`.")

# ============================================================================
# Stem player — pick a set and A/B the full mix against its vocal /
# instrumental stems. Useful for sanity-checking acappella alignments
# (is the vocal stem actually the vocals we think it is at time T?)
# and for spotting mashup dense sections where multiple acappellas
# are layered — the vocal stem there sounds like a crowd, not a song.
# ============================================================================
elif page == "Stem player":
    st.title("Stem player")
    st.caption(
        "Listen to a set alongside its demucs-separated stems. "
        "Vocal + instrumental stems are the input signal the "
        "alignment compares against ref-side stems, so listening here "
        "is the quickest way to hear why an acappella row is "
        "mis-localising (e.g. multiple vocals layered → no single "
        "track dominates the vocal stem)."
    )

    sets = _playable_sets()
    if not sets:
        st.info("No downloaded mix audio yet. Run the downloader first.")
    else:
        # Human-friendly dropdown: stem count in the label so the user
        # can tell at a glance which sets are fully analysed.
        def _stem_label(r: dict) -> str:
            n = r["n_stems"]
            return "✓ stems" if n >= 4 else f"{n}/5 stems"
        labels = [
            f"{r['title']}  ·  {r['set_id']}  ·  {_stem_label(r)}"
            for r in sets
        ]
        picked_idx = st.sidebar.selectbox(
            "Set", range(len(sets)), format_func=lambda i: labels[i],
        )
        picked = sets[picked_idx]
        set_id = picked["set_id"]
        title  = picked["title"]

        st.subheader(title)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("set_id", set_id)
        dur = picked["duration_s"]
        mc2.metric(
            "duration",
            f"{int(dur)//60}:{int(dur)%60:02d}" if dur else "—",
        )
        mc3.metric("stems on disk", picked["n_stems"])

        full_path = _set_audio_path_for(set_id)
        stem_paths = _set_stem_paths_for(set_id)

        # Optional: jump to a specific second. Useful for auditing
        # a particular alignment row without scrolling the player.
        seek = st.number_input(
            "Start time (seconds)", min_value=0, max_value=int(dur or 0),
            value=0, step=1,
            help="Seek every player below to this second. Handy for "
                 "checking a specific mashup section.",
        )

        # Players laid out in the order a DJ thinks about them: full
        # first (what the crowd hears), then the two "version"
        # hypotheses the aligner cares about (vocals = acappella,
        # instrumental = drums+bass+other), then the raw demucs stems
        # for deeper debugging.
        st.markdown("### Full mix")
        if full_path and Path(full_path).exists():
            st.audio(full_path, start_time=int(seek))
        else:
            st.warning(f"full mix not on disk: {full_path}")

        st.markdown("### Vocals (mix acappella)")
        vp = stem_paths.get("vocals")
        if vp and Path(vp).exists():
            st.audio(vp, start_time=int(seek))
            st.caption(
                "What the alignment scores acappella rows against. "
                "If you hear *multiple* vocals at once here, that's "
                "a mashup section — and the reason fingerprint + "
                "chroma-DTW on the vocal stem can still miss the "
                "correct track: the signal is a chorus of acappellas."
            )
        else:
            st.info("No vocals stem — run `demucs_adapter.separate` on this set.")

        st.markdown("### Instrumental (drums + bass + other, pre-summed)")
        ip = stem_paths.get("instrumental")
        if ip and Path(ip).exists():
            st.audio(ip, start_time=int(seek))
            st.caption(
                "Pre-summed drums+bass+other — what the aligner uses "
                "for `(Instrumental)` rows. Listen for whether the "
                "expected instrumental is actually the one playing at "
                "a given section or if something else is layered in."
            )
        else:
            st.info("No pre-summed instrumental stem. Use "
                    "`audio_pipeline/adapters/instrumental_backfill.py` "
                    "or re-run analysis.")

        # Individual demucs stems — kept behind an expander because
        # DJs rarely play isolated drums / bass / other, so these are
        # debug-level and would clutter the primary view.
        other_stems = [
            (name, stem_paths[name])
            for name in ("drums", "bass", "other")
            if name in stem_paths
        ]
        if other_stems:
            with st.expander("Individual demucs stems (drums / bass / other)"):
                for name, path in other_stems:
                    st.markdown(f"**{name}**")
                    if Path(path).exists():
                        st.audio(path, start_time=int(seek))
                    else:
                        st.warning(f"missing on disk: {path}")


elif page == "Annotate GT":
    st.title("Annotate ground truth")
    st.caption(
        "Edit or create a fixture yaml alongside the mix audio. Pick an "
        "existing fixture from the sidebar, or start a new one. Edits live "
        "in-browser; nothing touches disk until you click **Save yaml**. "
        "Every save archives the previous version under "
        "`tests/fixtures/.archive/` so you can restore it."
    )

    import re
    import shutil
    from datetime import datetime as _dt

    FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures"
    ARCHIVE_DIR = FIXTURE_DIR / ".archive"
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import yaml
    except ImportError:
        st.error("PyYAML is not installed in this venv. `venvs/audio/bin/pip install pyyaml`")
        st.stop()

    yaml_paths = sorted(FIXTURE_DIR.glob("*.yaml"))
    NEW_OPTION = "+ New fixture…"
    options = [NEW_OPTION] + [p.name for p in yaml_paths]
    picked = st.sidebar.selectbox("Fixture", options, index=(1 if yaml_paths else 0))

    is_new = (picked == NEW_OPTION)
    if is_new:
        new_set_id = st.sidebar.text_input(
            "set_id for new fixture", value="",
            help="Use the canonical set_id from `dj_sets` if this fixture "
                 "covers a scraped set. For a mix that's not yet in the DB, "
                 "use any stable string — it's the primary key for matching "
                 "GT rows back to aligned rows.",
        )
        new_name = st.sidebar.text_input(
            "file name (without .yaml)", value="",
            help="Defaults to `<set_id>_ground_truth` if blank.",
        )
        if not new_set_id.strip():
            st.info("Enter a `set_id` in the sidebar to start a new fixture.")
            st.stop()
        set_id = new_set_id.strip()
        stem = new_name.strip() or f"{set_id}_ground_truth"
        yaml_path = FIXTURE_DIR / f"{stem}.yaml"
        doc = {"set_id": set_id, "source": "ableton_session", "annotated_by": "user", "tracks": []}
        tracks = []
        st.info(f"Starting new fixture at `{yaml_path.name}`. Add rows in the table below.")
    else:
        yaml_path = FIXTURE_DIR / picked
        try:
            doc = yaml.safe_load(yaml_path.read_text()) or {}
        except yaml.YAMLError as e:
            st.error(f"Failed to parse {yaml_path.name}: {e}")
            st.stop()
        set_id = str(doc.get("set_id") or "").strip()
        if not set_id:
            st.error(f"{yaml_path.name} has no `set_id:` field.")
            st.stop()
        tracks = doc.get("tracks") or []
        if not isinstance(tracks, list):
            st.error(f"{yaml_path.name}'s `tracks:` must be a list.")
            st.stop()

    # Audio
    audio_path_str = _set_audio_path_for(set_id)
    st.subheader(f"{yaml_path.name}  ·  set_id {set_id}  ·  {len(tracks)} tracks")
    if audio_path_str and Path(audio_path_str).exists():
        preview = _ensure_preview_audio(audio_path_str)
        st.audio(preview or audio_path_str)
        st.caption(f"Audio: `{audio_path_str}`")
    else:
        st.info(
            f"No downloaded mix audio for set `{set_id}` yet. You can still "
            "annotate — paste media URLs per track and run the downloader."
        )

    # Build editor rows. Tolerant of missing fields so an older fixture
    # upgrades cleanly when opened. `ref_start_s` / `ref_end_s` are
    # surfaced either from the top-level key or from the FIRST entry of
    # `ref_segments` (loop / cut-up yamls).
    def _first_num(*vals: object) -> float | None:
        for v in vals:
            if isinstance(v, (int, float)):
                return float(v)
        return None

    # Build a track_id → {platform: url} map from the DB. URLs are
    # reconstructed from platform+player_id (the scraper stores IDs, not
    # URLs). Matches the canonical builder in audio_pipeline.models.
    # Uses a fresh connection: the cached st.cache_resource sqlite
    # connection is shared across concurrent Streamlit reruns and can
    # raise sqlite3.InterfaceError ("bad parameter or other API misuse")
    # when another rerun is mid-fetch. A per-page-render connection
    # avoids the contention; cost is a ~1 ms reopen.
    from audio_pipeline.models import youtube_url as _yt_url
    from audio_pipeline.models import soundcloud_api_url as _sc_url
    from audio_pipeline.models import spotify_track_url as _sp_url
    db_links_by_tid: dict[str, dict[str, str]] = {}
    try:
        _ephem = sqlite3.connect(str(DB_PATH))
        _ephem.row_factory = sqlite3.Row
        try:
            for _r in _ephem.execute(
                "SELECT track_id, platform, player_id FROM dj_set_track_media_links "
                "WHERE set_id=? AND track_id IS NOT NULL AND track_id != '' "
                "AND player_id IS NOT NULL AND player_id != ''",
                (str(set_id),),
            ).fetchall():
                _tid = _r["track_id"]; _plat = _r["platform"]; _pid = _r["player_id"]
                if _plat == "youtube":
                    _url = _yt_url(_pid)
                elif _plat == "soundcloud":
                    _url = _sc_url(_pid)
                elif _plat == "spotify":
                    _url = _sp_url(_pid)
                else:
                    _url = ""
                if _url:
                    db_links_by_tid.setdefault(_tid, {})[_plat] = _url
        finally:
            _ephem.close()
    except sqlite3.Error:
        # Catch the full Error hierarchy — InterfaceError, DatabaseError,
        # OperationalError etc. If the lookup fails, fall back to empty
        # (editor still works, URLs just start blank).
        pass

    rows_for_editor: list[dict] = []
    for t in tracks:
        if not isinstance(t, dict):
            continue
        segs = t.get("ref_segments") or []
        seg0 = segs[0] if segs and isinstance(segs[0], dict) else {}
        ml = t.get("media_links") or {}
        if not isinstance(ml, dict):
            ml = {}
        # Merge DB-derived URLs in as defaults. YAML-authored media_links
        # take precedence (the user's explicit edits beat scraper-state).
        tid = str(t.get("track_id") or "").strip()
        db_ml = db_links_by_tid.get(tid, {}) if tid else {}
        def _pref(yaml_val: object, db_key: str) -> str:
            v = str(yaml_val or "").strip()
            return v if v else db_ml.get(db_key, "")
        rows_for_editor.append({
            "track":          str(t.get("track") or "").strip(),
            "track_id":       tid,
            "version_tag":    (t.get("version_tag") or "").strip() if isinstance(t.get("version_tag"), str) else "",
            "set_start_s":    _first_num(t.get("set_start_s")),
            "set_end_s":      _first_num(t.get("set_end_s")),
            "ref_start_s":    _first_num(t.get("ref_start_s"), seg0.get("ref_start_s")),
            "ref_end_s":      _first_num(t.get("ref_end_s"),   seg0.get("ref_end_s")),
            "is_loop":        bool(t.get("is_loop", False)),
            "youtube_url":    _pref(ml.get("youtube"),    "youtube"),
            "spotify_url":    _pref(ml.get("spotify"),    "spotify"),
            "soundcloud_url": _pref(ml.get("soundcloud"), "soundcloud"),
            "other_url":      str(ml.get("other") or "").strip(),
        })

    # Consume any pending "add to editor" rows the user queued from the
    # Tracklist reference panel below. Dedupe on track_id so clicking
    # twice doesn't duplicate. Rows are appended once per page render
    # and then the queue is drained so they don't re-appear on reload.
    _pending_key = f"gt_pending_add::{yaml_path.name}"
    pending_rows = st.session_state.get(_pending_key, [])
    if pending_rows:
        existing_tids = {r["track_id"] for r in rows_for_editor if r["track_id"]}
        for pr in pending_rows:
            if pr.get("track_id") and pr["track_id"] in existing_tids:
                continue
            rows_for_editor.append(pr)
            if pr.get("track_id"):
                existing_tids.add(pr["track_id"])
        st.session_state[_pending_key] = []

    columns = [
        "track", "track_id", "version_tag",
        "set_start_s", "set_end_s", "ref_start_s", "ref_end_s", "is_loop",
        "youtube_url", "spotify_url", "soundcloud_url", "other_url",
    ]
    df_edit = pd.DataFrame(rows_for_editor, columns=columns)

    # Auto-reset the editor's session state when the schema changes (e.g. a
    # new column like youtube_url is added but cached state still has the
    # old 5-column layout and won't surface the new cells).
    _editor_key = f"gt_editor::{yaml_path.name}"
    _schema_marker_key = f"gt_editor_schema::{yaml_path.name}"
    _schema_marker = ",".join(df_edit.columns)
    if st.session_state.get(_schema_marker_key) != _schema_marker:
        st.session_state.pop(_editor_key, None)
        st.session_state[_schema_marker_key] = _schema_marker

    rc1, rc2 = st.columns([1, 4])
    if rc1.button("↻ Reload from disk",
                   help="Discards any in-browser edits and re-reads the "
                        "fixture yaml + DB links. Useful if URL / track_id "
                        "columns look stale."):
        st.session_state.pop(_editor_key, None)
        # Also clear per-track loops state so they reseed from disk.
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and k.startswith(f"gt_segments::{yaml_path.name}::"):
                st.session_state.pop(k, None)
        st.rerun()
    rc2.caption(
        "Editor auto-resets only when the column schema changes. If you "
        "added/edited rows, they're preserved across reruns until you save "
        "— reload here to discard."
    )

    edited = st.data_editor(
        df_edit,
        width="stretch", height=520,
        column_config={
            "track":       st.column_config.TextColumn("Track", width="large"),
            "track_id":    st.column_config.TextColumn("track_id", width="small",
                                                       help="DB track_id. Required for scraped tracks; optional for DJ-added tracks the tracklist missed."),
            "version_tag": st.column_config.SelectboxColumn(
                "version_tag", options=["", "instrumental", "acappella", "full"], width="small",
            ),
            "set_start_s": st.column_config.NumberColumn("mix start (s)", step=1.0, format="%.1f"),
            "set_end_s":   st.column_config.NumberColumn("mix end (s)",   step=1.0, format="%.1f"),
            "ref_start_s": st.column_config.NumberColumn("ref start (s)", step=1.0, format="%.1f",
                                                          help="MANDATORY. Seconds into the ref where the DJ dropped in. 0 = played from start."),
            "ref_end_s":   st.column_config.NumberColumn("ref end (s)",   step=1.0, format="%.1f"),
            "is_loop":     st.column_config.CheckboxColumn(
                "loop", width="small",
                help="Check if the DJ looped or cut up this track. Requires at least one ref_segment row in the Loops/cut-ups editor below.",
            ),
            "youtube_url":    st.column_config.TextColumn("YouTube URL"),
            "spotify_url":    st.column_config.TextColumn("Spotify URL"),
            "soundcloud_url": st.column_config.TextColumn("SoundCloud URL"),
            "other_url":      st.column_config.TextColumn("Other URL"),
        },
        num_rows="dynamic",
        key=_editor_key,
    )

    # Fill metrics. A row counts as "filled" if it has a track name AND
    # both mix-side times. Ref-side times and media links are independent.
    edited = edited.copy()
    edited["track"] = edited["track"].fillna("").astype(str)
    has_track   = edited["track"].str.len() > 0
    has_mix     = edited["set_start_s"].notna() & edited["set_end_s"].notna()
    has_ref     = edited["ref_start_s"].notna() & edited["ref_end_s"].notna()
    has_link    = (
        (edited["youtube_url"].fillna("").astype(str).str.len() > 0)
        | (edited["spotify_url"].fillna("").astype(str).str.len() > 0)
        | (edited["soundcloud_url"].fillna("").astype(str).str.len() > 0)
        | (edited["other_url"].fillna("").astype(str).str.len() > 0)
    )
    filled = edited[has_track & has_mix]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("total rows", len(edited))
    c2.metric("tracks with mix span", int(len(filled)))
    c3.metric("… + ref span",        int((has_track & has_mix & has_ref).sum()))
    c4.metric("… + media link",      int((has_track & has_mix & has_link).sum()))

    # ---- Inline helpers for the player + tracklist-reference panels -----
    def _gt_track_variants(track_id: str) -> list[dict]:
        """Return playable audio options for a given track_id: every
        `track_audio` row (labeled by variant_tag) plus each `track_stem`
        (vocals / instrumental / drums / bass / other). Empty if the
        track isn't in the DB or has no downloaded audio."""
        if not track_id:
            return []
        conn = _connect()
        opts: list[dict] = []
        for ta in conn.execute(
            "SELECT track_audio_id, path, variant_tag, is_reference "
            "FROM track_audio WHERE track_id = ? "
            "ORDER BY is_reference DESC, track_audio_id", (track_id,),
        ).fetchall():
            tag = (ta["variant_tag"] or "original").strip()
            opts.append({"label": f"{tag}", "path": ta["path"],
                         "track_audio_id": int(ta["track_audio_id"])})
            for st_row in conn.execute(
                "SELECT stem_name, path FROM track_stems "
                "WHERE track_audio_id = ? ORDER BY stem_name",
                (int(ta["track_audio_id"]),),
            ).fetchall():
                opts.append({
                    "label": f"{tag} / {st_row['stem_name']}",
                    "path":  st_row["path"],
                    "track_audio_id": int(ta["track_audio_id"]),
                })
        return opts

    # --- Selected-track player + loops editor ----------------------------
    st.markdown("### 🔎 Selected track")
    st.caption(
        "Pick any track you've added above to audition it. The mix player "
        "jumps to `set_start_s`; the ref player picks any variant or "
        "demucs stem that the DB has on disk for that `track_id`. "
        "Use the loops table below for tracks the DJ looped or cut up "
        "(e.g. BB11 Good Grief: ref 0:32–1:53 once, then 0:32–1:27 again)."
    )

    selectable = edited[edited["track"].astype(str).str.len() > 0].copy()
    if len(selectable) == 0:
        st.caption("No tracks yet — add one in the table above (use the "
                   "dynamic `+` row) or from the reference panel below.")
    else:
        selectable["_display"] = selectable.apply(
            lambda r: (
                f"{str(r['track'])[:70]}"
                + (f"  ·  mix {float(r['set_start_s']):.0f}-{float(r['set_end_s']):.0f}s"
                   if pd.notna(r['set_start_s']) and pd.notna(r['set_end_s']) else "")
                + (f"  ·  {r['version_tag']}" if r['version_tag'] else "")
            ), axis=1,
        )
        pick_idx = st.selectbox(
            "Track",
            selectable.index.tolist(),
            format_func=lambda i: selectable.at[i, "_display"],
            key=f"gt_player_sel::{yaml_path.name}",
        )
        pick_row = selectable.loc[pick_idx]
        pick_tid = str(pick_row.get("track_id") or "").strip()
        pick_track_key = pick_tid or str(pick_row["track"])

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Mix-side** (jumps to `set_start_s`)")
            if audio_path_str and Path(audio_path_str).exists():
                preview = _ensure_preview_audio(audio_path_str)
                start = int(pick_row["set_start_s"]) if pd.notna(pick_row["set_start_s"]) else 0
                st.audio(preview or audio_path_str, start_time=start)
                mix_end = (f"{float(pick_row['set_end_s']):.0f}s"
                           if pd.notna(pick_row["set_end_s"]) else "—")
                st.caption(f"set_start_s={start}  ·  set_end_s={mix_end}")
            else:
                st.caption("No mix audio for this set yet.")

        with pc2:
            st.markdown("**Ref-side** (pick a variant / stem)")
            variants = _gt_track_variants(pick_tid)
            if not variants:
                # Inline download: if the row has URLs in the editor, let the
                # user download right here instead of bouncing out to the
                # 'Register missing URL' page. Spotify URLs are shown for
                # reference but yt-dlp can only fetch YouTube / SoundCloud.
                row_urls = [
                    ("youtube",    str(pick_row.get("youtube_url") or "").strip()),
                    ("soundcloud", str(pick_row.get("soundcloud_url") or "").strip()),
                    ("other",      str(pick_row.get("other_url") or "").strip()),
                ]
                downloadable = [
                    (plat, url) for plat, url in row_urls
                    if url and _parse_media_url(url) is not None
                ]
                if not pick_tid:
                    st.caption(
                        "No `track_id` on this row — can't attach downloaded "
                        "audio. Copy a `track_id` from the **Tracklist "
                        "reference** expander below, or pick a track_id "
                        "manually."
                    )
                elif not downloadable:
                    st.caption(
                        f"No track_audio in the DB for `{pick_tid}`. Paste a "
                        "YouTube or SoundCloud URL into the editor row "
                        "(columns above), then return here to download."
                    )
                else:
                    st.caption(
                        f"No track_audio yet for `{pick_tid}`. Download "
                        "below — the file is saved under the audio drive "
                        "and registered as `track_audio` for this track_id."
                    )
                    # Lazy import keeps UI boot time snappy when the GT tab
                    # isn't touched.
                    from audio_pipeline.adapters import db as _gt_db
                    from audio_pipeline.adapters.downloader import (
                        DownloadConfig as _GtDlCfg, download_one as _gt_download_one,
                    )
                    from audio_pipeline.models import (
                        MediaSource as _GtMediaSource,
                        youtube_url as _gt_yt_url,
                        soundcloud_api_url as _gt_sc_url,
                    )
                    _GT_TRACKS_DIR = (
                        Path.home() / "Desktop" / "tracklist_audio_drive" / "tracks"
                    )
                    for plat, url in downloadable:
                        parsed = _parse_media_url(url)
                        if parsed is None:
                            continue
                        platform, player_id = parsed
                        btn_key = f"gt_dl::{yaml_path.name}::{pick_tid}::{platform}::{player_id}"
                        if st.button(
                            f"↓ Download from {platform}",
                            key=btn_key,
                            help=f"{url[:80]}",
                        ):
                            # Register the link first so the scraper's view of
                            # this track gains the URL, then fetch audio.
                            ins = _gt_db.insert_track_media_link(
                                DB_PATH,
                                set_id=str(set_id),
                                track_id=pick_tid,
                                platform=platform,
                                player_id=player_id,
                                url=url,
                                tlp_id=None,
                            )
                            if not ins.is_ok():
                                st.error(
                                    f"DB insert_track_media_link failed: "
                                    f"{ins.error.kind} — {ins.error.detail}"
                                )
                            else:
                                out_dir = _GT_TRACKS_DIR / str(set_id)
                                out_dir.mkdir(parents=True, exist_ok=True)
                                can_url = (
                                    _gt_yt_url(player_id)
                                    if platform == "youtube"
                                    else _gt_sc_url(player_id)
                                )
                                src = _GtMediaSource(
                                    platform=platform,
                                    player_id=player_id,
                                    url=can_url,
                                )
                                with st.spinner(
                                    f"Downloading {platform} → {out_dir}…"
                                ):
                                    dl = _gt_download_one(
                                        pick_tid, src, _GtDlCfg(out_dir=out_dir),
                                    )
                                if not dl.is_ok():
                                    st.error(
                                        f"Download failed: "
                                        f"{dl.error.kind} — {dl.error.detail}"
                                    )
                                else:
                                    asset = dl.value
                                    ia = _gt_db.insert_audio(DB_PATH, asset)
                                    if not ia.is_ok():
                                        st.error(
                                            f"DB insert_audio failed: {ia.error}"
                                        )
                                    else:
                                        st.success(
                                            f"Downloaded → `{asset.path}`. "
                                            "Reloading to pick it up as a "
                                            "variant…"
                                        )
                                        st.cache_data.clear()
                                        st.rerun()
            else:
                vlabels = [v["label"] for v in variants]
                default_idx = 0
                for i, v in enumerate(variants):
                    if v["label"] == "original":
                        default_idx = i; break
                vpick = st.selectbox(
                    "variant / stem", vlabels, index=default_idx,
                    key=f"gt_ref_variant::{yaml_path.name}::{pick_idx}",
                )
                vpath = next(v["path"] for v in variants if v["label"] == vpick)
                if Path(vpath).exists():
                    rstart = int(pick_row["ref_start_s"]) if pd.notna(pick_row["ref_start_s"]) else 0
                    st.audio(vpath, start_time=rstart)
                    rend = (f"{float(pick_row['ref_end_s']):.0f}s"
                            if pd.notna(pick_row["ref_end_s"]) else "—")
                    st.caption(f"ref_start_s={rstart}  ·  ref_end_s={rend}")
                else:
                    st.warning(f"File missing on disk: `{vpath}`")

        # Loops / cut-ups editor for the selected track.
        st.markdown("#### Loops / cut-ups")
        seg_key = f"gt_segments::{yaml_path.name}::{pick_track_key}"
        if seg_key not in st.session_state:
            # Seed from the yaml on first render for this track.
            seeded: list[dict] = []
            for t in tracks:
                if not isinstance(t, dict):
                    continue
                this_key = str(t.get("track_id") or t.get("track") or "")
                if this_key == pick_track_key:
                    for s in (t.get("ref_segments") or []):
                        if not isinstance(s, dict):
                            continue
                        if not all(k in s for k in ("mix_start_s", "ref_start_s", "ref_end_s")):
                            continue
                        seeded.append({
                            "mix_start_s": float(s["mix_start_s"]),
                            "ref_start_s": float(s["ref_start_s"]),
                            "ref_end_s":   float(s["ref_end_s"]),
                        })
                    break
            st.session_state[seg_key] = seeded

        seg_df = pd.DataFrame(
            st.session_state[seg_key],
            columns=["mix_start_s", "ref_start_s", "ref_end_s"],
        )
        seg_edited = st.data_editor(
            seg_df, width="stretch", height=220, num_rows="dynamic",
            column_config={
                "mix_start_s": st.column_config.NumberColumn("mix start (s)", step=1.0, format="%.1f"),
                "ref_start_s": st.column_config.NumberColumn("ref start (s)", step=1.0, format="%.1f"),
                "ref_end_s":   st.column_config.NumberColumn("ref end (s)",   step=1.0, format="%.1f"),
            },
            key=f"gt_seg_editor::{yaml_path.name}::{pick_track_key}",
        )
        # Mirror the editor's current state back into session_state so the
        # save handler (below) can see it. Skip rows with any blank field.
        st.session_state[seg_key] = [
            {"mix_start_s": float(r["mix_start_s"]),
             "ref_start_s": float(r["ref_start_s"]),
             "ref_end_s":   float(r["ref_end_s"])}
            for _, r in seg_edited.iterrows()
            if pd.notna(r["mix_start_s"]) and pd.notna(r["ref_start_s"]) and pd.notna(r["ref_end_s"])
        ]
        if st.session_state[seg_key]:
            st.caption(
                f"{len(st.session_state[seg_key])} segment(s) will be written to "
                "`ref_segments:` on save. The main table's flat `ref_start_s`/"
                "`ref_end_s` are ignored for this track when segments exist."
            )

    # --- Tracklist reference (+ "add to editor") ------------------------
    with st.expander(
        f"Tracklist reference — every track the scraper captured for `{set_id}`",
        expanded=False,
    ):
        st.caption(
            "DB view of this set's tracklist. Includes rows the scraper "
            "never found a media URL for — click **+ Add** to copy them "
            "into the GT editor above, then paste a URL into the main "
            "table to enable download."
        )
        conn = _connect()
        # Use json_extract on data_attrs_json so rows missing from
        # dj_set_track_media_links still appear (user can still add them
        # and paste a URL). track_media_links join remains as LEFT JOIN
        # only to pull scraped URLs when available.
        ref_rows = conn.execute(
            """
            SELECT r.row_index,
                   r.text_excerpt,
                   json_extract(r.data_attrs_json, '$."data-trackid"') AS track_id,
                   MAX(ta.path) AS any_path,
                   MAX(tml.platform || '|' || COALESCE(tml.player_id, '')) AS any_tml
            FROM dj_set_rows r
            LEFT JOIN dj_set_track_media_links tml
                   ON tml.set_id = r.set_id
                      AND tml.track_id = json_extract(r.data_attrs_json, '$."data-trackid"')
            LEFT JOIN track_audio ta
                   ON ta.track_id = json_extract(r.data_attrs_json, '$."data-trackid"')
            WHERE r.set_id = ?
              AND json_extract(r.data_attrs_json, '$."data-trackid"') IS NOT NULL
              AND json_extract(r.data_attrs_json, '$."data-trackid"') != ''
            GROUP BY r.row_index, r.text_excerpt, track_id
            ORDER BY r.row_index
            """,
            (set_id,),
        ).fetchall()
        if not ref_rows:
            st.caption("No tracklist rows with a data-trackid found for this set_id in the DB.")
        else:
            # Already-in-editor track_ids (so the + Add button can dedupe).
            in_editor_tids: set[str] = {
                str(t.get("track_id") or "").strip()
                for t in (tracks or []) if isinstance(t, dict) and t.get("track_id")
            }
            # Also fold pending additions so rapid-fire clicks don't duplicate.
            for pr in st.session_state.get(_pending_key, []):
                if pr.get("track_id"):
                    in_editor_tids.add(pr["track_id"])
            st.caption(f"{len(ref_rows)} tracks on tracklist.")
            for r in ref_rows:
                cols = st.columns([1, 4, 2, 1, 1])
                cols[0].caption(f"row {r['row_index']}")
                cols[1].code(
                    (r["text_excerpt"] or "")[:90],
                    language="text",
                )
                tid_str = r["track_id"] or ""
                cols[2].caption(tid_str or "—")
                playable = r["any_path"] and Path(r["any_path"]).exists()
                if playable:
                    if cols[3].button("▶ play", key=f"gt_ref_play::{r['row_index']}::{tid_str}"):
                        st.session_state[f"gt_ref_inline_play::{tid_str}"] = r["any_path"]
                    inline = st.session_state.get(f"gt_ref_inline_play::{tid_str}")
                    if inline:
                        st.audio(inline)
                        cols[3].caption("playing: original")
                else:
                    cols[3].caption("no audio")
                # + Add to editor. Disabled when the track_id is already
                # present in the fixture (or queued). Copies a prefilled
                # row into the editor's pending queue and reruns so the
                # row appears on the next render.
                already = tid_str and tid_str in in_editor_tids
                add_label = "✓ in" if already else "+ Add"
                if cols[4].button(
                    add_label, key=f"gt_ref_add::{r['row_index']}::{tid_str}",
                    disabled=bool(already) or not tid_str,
                    help="Copy this tracklist row into the GT editor as a new line."
                         if not already else "Already in the editor.",
                ):
                    cleaned = _clean_clip_label(r["text_excerpt"] or "")
                    new_row = {
                        "track": cleaned,
                        "track_id": tid_str,
                        "version_tag": "",
                        "set_start_s": None,
                        "set_end_s": None,
                        "ref_start_s": None,
                        "ref_end_s": None,
                        "is_loop": False,
                        "youtube_url": "",
                        "spotify_url": "",
                        "soundcloud_url": "",
                        "other_url": "",
                    }
                    queue = list(st.session_state.get(_pending_key, []))
                    queue.append(new_row)
                    st.session_state[_pending_key] = queue
                    # Clear the editor key so the new row appears. In-progress
                    # edits to other rows are lost; this is the intended
                    # trade-off (documented next to the Reload button above).
                    st.session_state.pop(_editor_key, None)
                    st.rerun()

    # Archive browser
    archived = sorted(ARCHIVE_DIR.glob(f"{yaml_path.stem}_*.yaml"), reverse=True)
    with st.expander(f"Archive ({len(archived)} prior versions)", expanded=False):
        if not archived:
            st.caption(
                "No archive snapshots yet. Saving an existing fixture "
                "automatically copies its prior contents to "
                f"`{ARCHIVE_DIR.relative_to(_REPO_ROOT)}/` first, so you "
                "can always roll back."
            )
        for ap in archived[:25]:
            ar1, ar2, ar3 = st.columns([3, 1, 1])
            ar1.code(ap.name, language="text")
            if ar2.button("Preview", key=f"gt_preview::{ap.name}"):
                st.text_area(
                    f"Contents of {ap.name}",
                    ap.read_text(),
                    height=280,
                    key=f"gt_preview_area::{ap.name}",
                )
            if ar3.button("Restore", key=f"gt_restore::{ap.name}",
                           help="Overwrites the current fixture with this "
                                "archived copy. Also archives the current "
                                "file first so you can undo the restore."):
                if yaml_path.exists():
                    stamp = _dt.now().strftime("%Y%m%d-%H%M%S")
                    shutil.copy2(yaml_path, ARCHIVE_DIR / f"{yaml_path.stem}_{stamp}.yaml")
                shutil.copy2(ap, yaml_path)
                st.success(f"Restored `{ap.name}` → `{yaml_path.name}`. Reloading…")
                st.rerun()

    # --- Save ---------------------------------------------------------------
    sc1, sc2 = st.columns([1, 3])
    save_clicked = sc1.button("Save yaml", type="primary",
                               help="Writes the editor contents to disk. "
                                    "Archives the previous version first.")
    sc2.caption("Edits stay in-browser until you click Save. Unsaved changes "
                "are lost on page reload.")

    if save_clicked:
        # Archive the current on-disk version before overwriting.
        archived_note = "new fixture (nothing to archive)"
        if yaml_path.exists():
            stamp = _dt.now().strftime("%Y%m%d-%H%M%S")
            archive_path = ARCHIVE_DIR / f"{yaml_path.stem}_{stamp}.yaml"
            shutil.copy2(yaml_path, archive_path)
            archived_note = f"previous version archived → `{archive_path.relative_to(_REPO_ROOT)}`"

        # Serialize.
        out_lines: list[str] = []
        title = ""
        try:
            conn = _connect()
            r = conn.execute("SELECT title FROM dj_sets WHERE set_id=?", (set_id,)).fetchone()
            if r:
                title = r["title"]
        except sqlite3.DatabaseError:
            pass
        out_lines.append(f"# Hand-annotated ground-truth for {title or set_id}")
        out_lines.append("#")
        out_lines.append("# See tests/fixtures/bigbootie11_ground_truth.yaml for the schema.")
        out_lines.append(f"set_id: {set_id}")
        out_lines.append(f"source: {doc.get('source', 'ableton_session')}")
        out_lines.append(f"annotated_by: {doc.get('annotated_by', 'user')}")
        out_lines.append("tracks:")
        for _, row in filled.iterrows():
            tag = (row["version_tag"] or "").strip()
            label = (row["track"] or "").replace('"', r'\"')
            out_lines.append(f"  - track: \"{label}\"")
            tid = str(row.get("track_id") or "").strip()
            if tid:
                out_lines.append(f"    track_id: {tid}")
            if tag:
                out_lines.append(f"    version_tag: {tag}")
            out_lines.append(f"    set_start_s: {float(row['set_start_s']):g}")
            out_lines.append(f"    set_end_s:   {float(row['set_end_s']):g}")
            # ref_start_s is MANDATORY. If the user didn't fill it, default
            # to the first segment's ref_start_s when we have one, else 0.
            row_key = str(row.get("track_id") or "").strip() or str(row["track"])
            row_segs = st.session_state.get(
                f"gt_segments::{yaml_path.name}::{row_key}", []
            )
            if pd.notna(row["ref_start_s"]):
                ref_start_val = float(row["ref_start_s"])
            elif row_segs:
                ref_start_val = float(row_segs[0]["ref_start_s"])
            else:
                ref_start_val = 0.0
            out_lines.append(f"    ref_start_s: {ref_start_val:g}")
            if pd.notna(row["ref_end_s"]):
                out_lines.append(f"    ref_end_s:   {float(row['ref_end_s']):g}")
            # is_loop explicitly, when checked. Auto-promote to True if the
            # user added segments but forgot the checkbox — the schema
            # requires is_loop when segments exist.
            is_loop = bool(row.get("is_loop", False)) or bool(row_segs)
            if is_loop:
                out_lines.append(f"    is_loop:     true")
            if row_segs:
                out_lines.append("    ref_segments:")
                for s in row_segs:
                    out_lines.append(f"      - mix_start_s: {float(s['mix_start_s']):g}")
                    out_lines.append(f"        ref_start_s: {float(s['ref_start_s']):g}")
                    out_lines.append(f"        ref_end_s:   {float(s['ref_end_s']):g}")
            links: list[tuple[str, str]] = []
            for col, key in (
                ("youtube_url", "youtube"),
                ("spotify_url", "spotify"),
                ("soundcloud_url", "soundcloud"),
                ("other_url", "other"),
            ):
                url = str(row.get(col) or "").strip()
                if url:
                    links.append((key, url))
            if links:
                out_lines.append("    media_links:")
                for k, v in links:
                    out_lines.append(f"      {k}: {v}")

        # A _scaffold.yaml saves as a sibling _ground_truth.yaml. Anything
        # else (including new fixtures) overwrites in place.
        if re.search(r"_scaffold\.ya?ml$", yaml_path.name):
            out_path = yaml_path.with_name(yaml_path.name.replace("_scaffold", "_ground_truth"))
        else:
            out_path = yaml_path
        out_path.write_text("\n".join(out_lines) + "\n")
        st.success(
            f"Saved {len(filled)} annotated rows → `{out_path.relative_to(_REPO_ROOT)}`. "
            f"{archived_note}."
        )
        st.caption("Run the eval harness to score against all fixtures: "
                   "`venvs/audio/bin/python -m audio_pipeline.alignment.eval`")

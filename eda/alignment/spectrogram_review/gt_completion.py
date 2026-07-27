"""GT completion via human spectrogram verification.

The de-poisoned GT abstains on 34-43% of appearances (`track_id=None`) — sound
but incomplete (see docs/alignment_status.md). Each abstained row still carries
the human-labeled song NAME (from the .als) + timing. This proposes candidate
`recording_id`s for each, from:

  * NAME-match to the recording catalog (the .als name is the human's truth), and
  * the ALIGNER's prediction overlapping that time,

ranks them (name+aligner agreement is strongest), and renders MIX-vs-candidate
spectrograms for human confirmation. **Truth is the human's spectrogram verdict**,
so completing the GT this way never inflates the aligner score: where the aligner
guessed right it is confirmed, where it guessed wrong the name-match supplies the
truth (or it stays abstained).

Pure helpers here; the catalog/timeline/manifest I/O + rendering live in the CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[3]
_PI = "pi-storage"
_PI_DB = "/mnt/storage/data/db/music_database.db"

_QUALIFIER = re.compile(r"\(.*?\)|\[.*?\]")
# words that are stem/version qualifiers, not identity tokens
_STOPWORDS = frozenset(
    {
        "acappella",
        "acapella",
        "acap",
        "instrumental",
        "remix",
        "rework",
        "edit",
        "bootleg",
        "mashup",
        "vip",
        "ft",
        "feat",
        "featuring",
        "vs",
        "x",
        "the",
        "official",
        "studio",
        "vocals",
        "vocal",
        "lead",
    }
)


def norm_name(name: str) -> str:
    """Lowercase, drop (...)/[...] qualifiers and all non-alphanumerics."""
    s = _QUALIFIER.sub(" ", (name or "").lower())
    return re.sub(r"[^a-z0-9]", "", s)


def _tokens(name: str) -> set[str]:
    s = _QUALIFIER.sub(" ", (name or "").lower())
    return {
        t for t in re.findall(r"[a-z0-9]+", s) if t not in _STOPWORDS and len(t) > 1
    }


def name_candidates(name: str, recs: list[dict]) -> list[str]:
    """recording_ids whose catalog name contains all identity tokens of `name`.

    `recs` = [{"rid": str, "full": str}, ...]. Subset match (name tokens ⊆ rec
    tokens) tolerates the catalog's extra artist/feature words; requires >=2
    shared identity tokens so a lone common word can't match everything.
    """
    want = _tokens(name)
    if len(want) < 2:
        return []
    out: list[str] = []
    for r in recs:
        have = _tokens(r.get("full", ""))
        if want <= have:
            out.append(str(r["rid"]))
    return out


def _f(v, d=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def aligner_candidate(row: dict, spans: list[dict]) -> str | None:
    """recording_id of the predicted span overlapping this row in time (nearest
    set_start on ties), or None if the aligner predicted nothing there."""
    a0, a1 = _f(row.get("set_start_s")), _f(row.get("set_end_s"))
    overlapping = [
        s
        for s in spans
        if s.get("recording_id")
        and _f(s.get("set_start_s")) < a1
        and _f(s.get("set_end_s")) > a0
    ]
    if not overlapping:
        return None
    best = min(overlapping, key=lambda s: abs(_f(s.get("set_start_s")) - a0))
    return str(best["recording_id"])


def to_render_row(
    abst: dict,
    candidate_rid: str,
    confidence: str,
    set_label: str,
    set_id: str,
    idx: int,
) -> dict:
    """Synthesize a span_table row that renders MIX-at-position vs the CANDIDATE
    recording's SOURCE in the spectrogram player, with the GT/pred boxes made
    coincident (the candidate IS the proposal). `name` surfaces the confidence
    so the reviewer sees why it was proposed."""
    ss = _f(abst.get("set_start_s"))
    se = _f(abst.get("set_end_s"))
    rs = _f(abst.get("ref_start_s"))
    re_ = _f(abst.get("ref_end_s"), rs + (se - ss))
    stem = str(abst.get("claimed_stem") or "regular")
    span_class = "multiseg" if abst.get("ref_segments") else "linear"
    return {
        "set": set_label,
        "set_id": set_id,
        "slot": str(abst.get("slot_label") or ""),
        "recording_id": candidate_rid,
        "name": f"{abst.get('track', '')}  [{confidence}]",
        "gt_stem": stem,
        "pred_stem": stem,
        "stem_mismatch": 0,
        "span_class": span_class,
        "gt_set_start_s": ss,
        "gt_ref_start_s": rs,
        "gt_ref_end_s": re_,
        "gt_duration_s": round(se - ss, 2),
        "pred_set_start_s": ss,
        "pred_ref_start_s": rs,
        "pred_n_segments": 1,
        "identity_hit": "",
        "set_start_err_s": 0.0,
        "ref_offset_scored": 0,
        "coverage_miss": 0,
        "idx": idx,
    }


def rank_candidates(name_ids: list[str], aligner_id: str | None) -> list[dict]:
    """Ordered candidate binds, strongest first:
    name+aligner agreement > name-match > aligner-only."""
    out: list[dict] = []
    seen: set[str] = set()
    if aligner_id and aligner_id in name_ids:
        out.append({"recording_id": aligner_id, "confidence": "name+aligner"})
        seen.add(aligner_id)
    for r in name_ids:
        if r in seen:
            continue
        out.append(
            {
                "recording_id": r,
                "confidence": "name-unique" if len(name_ids) == 1 else "name-multi",
            }
        )
        seen.add(r)
    if aligner_id and aligner_id not in seen:
        out.append({"recording_id": aligner_id, "confidence": "aligner-only"})
    return out


def apply_verdicts(
    gt_rows: list[dict], verdicts: list[dict]
) -> tuple[list[dict], dict]:
    """Fill abstained GT rows from human CONFIRM verdicts (spectrogram-verified).

    Matches a verdict to its abstained appearance by (slot, normalized song
    name). Only `confirm` verdicts fill a `track_id` (id_source=human_spectrogram);
    rejects/skips and already-bound rows are left untouched. Truth is the human's
    verdict, so this cannot inflate the aligner score."""
    confirmed: dict[tuple[str, str], str] = {}
    n_conf = 0
    for v in verdicts:
        if v.get("verdict") != "confirm":
            continue
        n_conf += 1
        track = str(v.get("name", "")).split("  [")[0]
        confirmed[(str(v.get("slot")), norm_name(track))] = str(v.get("recording_id"))
    out: list[dict] = []
    filled = 0
    for r in gt_rows:
        r = dict(r)
        if not r.get("track_id"):
            key = (str(r.get("slot_label")), norm_name(r.get("track", "")))
            if key in confirmed:
                r["track_id"] = confirmed[key]
                r["id_source"] = "human_spectrogram"
                filled += 1
        out.append(r)
    return out, {"filled": filled, "confirmed_verdicts": n_conf}


# --------------------------------------------------------------------------- I/O + CLI
def _load_pi_recordings() -> list[dict]:
    sql = (
        "SELECT rec.recording_id, "
        "COALESCE(NULLIF(rec.full_name,''), rec.title, w.full_name, w.title, '') "
        "FROM recording rec LEFT JOIN work w ON rec.work_id=w.work_id;"
    )
    out = subprocess.run(
        ["ssh", _PI, f'sqlite3 -separator "~" {_PI_DB} "{sql}"'],
        capture_output=True,
        text=True,
        timeout=90,
    ).stdout
    recs = []
    for line in out.strip().splitlines():
        p = line.split("~", 1)
        if len(p) == 2 and p[0]:
            recs.append({"rid": p[0], "full": p[1]})
    return recs


def build_candidate_rows(
    gt_yaml: Path,
    timeline_path: Path,
    set_label: str,
    set_id: str,
    recs: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """One render row per (abstained appearance × top-K candidate)."""
    gt = [
        r
        for r in yaml.safe_load(gt_yaml.read_text())["tracks"]
        if not r.get("track_id")
        and str(r.get("slot_label")) != "mix"
        and not r.get("unalignable")
    ]
    spans = json.loads(timeline_path.read_text())["spans"]
    rows: list[dict] = []
    idx = 0
    for r in gt:
        ranked = rank_candidates(
            name_candidates(r.get("track", ""), recs), aligner_candidate(r, spans)
        )
        for cand in ranked[:top_k]:
            rows.append(
                to_render_row(
                    r, cand["recording_id"], cand["confidence"], set_label, set_id, idx
                )
            )
            idx += 1
    return rows


# Self-contained verdict layer appended to the rendered index.html. Reads the
# player's own `ITEMS`/`stageIdx` globals (no player_html edits); y=confirm the
# open candidate's recording binding, n=reject, s=skip; Export downloads JSON.
_VERDICT_JS = """
<script>
(function(){
  const KEY='gtverdict';
  let V=JSON.parse(localStorage.getItem(KEY)||'{}');
  const save=()=>localStorage.setItem(KEY,JSON.stringify(V));
  function cur(){ return (typeof ITEMS!=='undefined' && typeof stageIdx!=='undefined'
      && document.body.classList.contains('playing-mode')) ? ITEMS[stageIdx] : null; }
  function setV(v){ const it=cur(); if(!it){return;}
    V[it.idx]={idx:it.idx,slot:it.slot,recording_id:it.recording_id||'',name:it.name,verdict:v};
    save(); paint(); }
  const bar=document.createElement('div');
  bar.style.cssText='position:fixed;bottom:12px;right:12px;z-index:99999;background:#161c2b;'
    +'border:1px solid #33405c;border-radius:10px;padding:9px 13px;font:13px system-ui;color:#dde;'
    +'box-shadow:0 6px 20px rgba(0,0,0,.5);max-width:320px';
  bar.innerHTML='<b>GT completion verdict</b><br>'
    +'<span style="color:#9fb">y</span>=confirm · <span style="color:#f9a">n</span>=reject · '
    +'<span style="color:#bbb">s</span>=skip (open a clip first)<br>'
    +'<div id="gtv-cur" style="margin:4px 0;color:#cdf"></div>'
    +'<div id="gtv-count" style="color:#9ab"></div>'
    +'<button id="gtv-exp" style="margin-top:6px;cursor:pointer">Export verdicts JSON</button>';
  document.body.appendChild(bar);
  function paint(){ const it=cur();
    document.getElementById('gtv-cur').textContent = it
      ? ('slot '+it.slot+' → '+(it.recording_id||'?')+'  ['+(V[it.idx]?V[it.idx].verdict:'—')+']') : '(no clip open)';
    const n=Object.keys(V).length, y=Object.values(V).filter(x=>x.verdict==='confirm').length;
    document.getElementById('gtv-count').textContent = n+' verdicts · '+y+' confirmed';
  }
  document.getElementById('gtv-exp').onclick=()=>{ const b=new Blob([JSON.stringify(Object.values(V),null,2)],
    {type:'application/json'}); const a=document.createElement('a'); a.href=URL.createObjectURL(b);
    a.download='verdicts.json'; a.click(); };
  window.addEventListener('keydown',e=>{ const t=(e.target&&e.target.tagName)||'';
    if(t==='INPUT'||t==='TEXTAREA'||t==='SELECT')return;
    if(e.key==='y'){setV('confirm');} else if(e.key==='n'){setV('reject');} else if(e.key==='s'){setV('skip');}
  });
  setInterval(paint,400); paint();
})();
</script>
"""


def inject_verdict_ui(index_html: Path) -> None:
    """Append the verdict layer to a rendered spectrogram index.html (idempotent)."""
    html = index_html.read_text()
    if "GT completion verdict" in html:
        return
    marker = "</body>"
    html = (
        html.replace(marker, _VERDICT_JS + "\n" + marker, 1)
        if marker in html
        else html + _VERDICT_JS
    )
    index_html.write_text(html)


def main(argv: list[str] | None = None) -> int:
    from eda.alignment.failure_analysis.build_span_table import FIELDS

    p = argparse.ArgumentParser(
        description="Emit a candidate span_table for GT completion review."
    )
    p.add_argument(
        "--inject",
        type=Path,
        default=None,
        help="append the verdict UI to a rendered index.html and exit",
    )
    p.add_argument(
        "--harvest",
        type=Path,
        default=None,
        help="verdicts.json exported from the UI; fills --gt into --out and exits",
    )
    p.add_argument("--set-id")
    p.add_argument("--label", help="BB11 | BB12")
    p.add_argument("--gt", type=Path)
    p.add_argument("--timeline", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    if args.inject:
        inject_verdict_ui(args.inject)
        print(f"injected verdict UI -> {args.inject}")
        return 0

    if args.harvest:
        if not (args.gt and args.out):
            p.error("--harvest needs --gt (abstained GT) and --out (completed GT)")
        doc = yaml.safe_load(args.gt.read_text())
        verdicts = json.loads(args.harvest.read_text())
        doc["tracks"], stats = apply_verdicts(doc["tracks"], verdicts)
        args.out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        print(
            f"harvested: filled {stats['filled']} abstained rows "
            f"({stats['confirmed_verdicts']} confirms) -> {args.out}"
        )
        return 0

    if not (args.set_id and args.label and args.gt and args.out):
        p.error(
            "--set-id, --label, --gt, --out are required (unless --inject/--harvest)"
        )

    tl = args.timeline or (
        _REPO
        / "alignment/out"
        / f"{args.set_id}_predicted_timeline_lt.json"
    )
    recs = _load_pi_recordings()
    rows = build_candidate_rows(args.gt, tl, args.label, args.set_id, recs, args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    n_appear = len({(r["slot"], r["name"].split("  [")[0]) for r in rows})
    print(
        f"wrote {len(rows)} candidate rows ({n_appear} appearances, top-{args.top_k}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

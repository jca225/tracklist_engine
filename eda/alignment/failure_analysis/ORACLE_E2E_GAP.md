# Oracle ↔ e2e acappella trajectory gap — decomposition

Reproduce:

```bash
venvs/audio/bin/python -m eda.alignment.failure_analysis.oracle_e2e_gap
```

Read-only consumer of `score_timeline_vs_gt`'s own matching internals (span→GT
pairing, extent, identity) + `path_decode.trajectory_acc`. Answers the question
queued in `looptrace/NOTES.md` (seventh pass): **is the oracle(43–44%)↔e2e(~16%)
acappella gap a decoder problem, a placement problem, or a scoring/correspondence
artifact?** Previous ad-hoc joins were unreliable; this uses the scorer's logic.

## Punchline

**The gap is placement + windowing, not decode residual, and it is NOT mainly a
scoring artifact.** On the oracle population (GT spans that also have a decodable
oracle baseline), pooled across BB11+BB12:

- oracle = **37.0%**, e2e = **15.8%**, gap = **298 GT-seconds**.
- **91% of the gap is "content never entered the scored decode window"** (the
  extrapolation-accuracy region: 8%). Only **9%** is genuine in-window decode
  error. When the right content *is* in the window, e2e ≈ oracle (BB12 in-window
  38.1% vs oracle 40.4%).

So the decoder's ceiling (~37%) is a *separate, higher* wall from what we ship
(~16%). Getting from 16→37 is dominated by **set_start placement** and
window/pairing — **none of it gated on a third GT set.** The "which chorus"
instance-selection prize still caps the ceiling at ~37% and still needs BB10.

## Gap decomposition (pooled ALL acappella, 192 GT rows / 7500 GT-sec)

Buckets are mutually exclusive, GT-seconds-weighted, one binding cause per GT
row, assigned by the scorer's own pairing. `repair→oracle` = pooled e2e traj if
that bucket were lifted to the decoder's oracle quality (the honest counterfactual
— repairing to 100% is unreachable because the decoder itself is only ~37%).

| bucket | GT-sec | % GT-sec | e2e if repaired→oracle | lever gated on BB10? |
|---|---|---|---|---|
| CORRECT (already ≤2 s) | 1176 | 15.7% | — | — |
| **B4_PLACEMENT** (set_start off → window misses) | **2943** | **39.2%** | **15.7→30.2%** | no |
| B5_DECODE (placed+covered, wrong ref offset) | 2343 | 31.2% | →27.2% | **yes** (instance sel.) |
| B3_EXTENT (pred span too short to cover GT) | 453 | 6.0% | →17.9% | no (windowing) |
| B1_STARVED (GT row starved by 1:1 pairing) | 343 | 4.6% | →17.4% | no (scorer) |
| B1_ABSENT (recording absent from timeline) | 242 | 3.2% | →16.9% | no (ingest) |

**Placement is the single biggest recoverable lever (+14.5 pp to oracle
quality), larger than the decode wall (+11.5 pp) — and placement is shippable
now.** The two correspondence/scoring buckets (STARVED+EXTENT) total ~10.6% of
GT-sec: real, but not the story. The gap is genuine, not laundered by the
scorer.

## Per-set (oracle population)

| set | oracle | e2e | gap | never-in-window share | in-window acc |
|---|---|---|---|---|---|
| BB12 (1fsnxchk) | 40.4% | 24.3% | 103 s | 93% | 38.1% (≈oracle) |
| BB11 (2nvzlh2k) | 34.2% | 8.8% | 195 s | 87% | 18.4% |
| POOLED | 37.0% | 15.8% | 298 s | 91% | 31% |

The two sets fail placement *differently*, and the diag lines pin it:

- **BB12: windows are over-wide, placement is off.** pred-window median **167 s**
  vs GT span median 29 s; overlap median 0.98, only 11% zero-overlap — so the
  window usually *contains* the span, yet only 49% of GT-sec is "seen" in-window
  and set_start error is **>15 s on 57%** of paired seconds. The failure is the
  set_start offset within the pair, plus 78 `online_candidate` rows (3356 s @
  16.7% e2e) that have no oracle baseline and dominate BB12 acappella seconds.
- **BB11: windows are too short/mis-placed.** pair window∩GT median **0.10**,
  29% zero-overlap, pred-window median 14 s vs GT 31 s — the content is simply
  not in the scored window. This is the same BB11 stem-coverage story from
  FINDINGS.md (fewer vocals stems → weaker HuBERT placement).

## Scorer-pairing caveats (quoted, for the STARVED bucket)

`score_timeline_vs_gt` pairs each predicted span to the nearest same-recording GT
row 1:1. Diagnostics:

- **BB12 multi-instance recordings = 8 (18 rows, 11 starved); w-layer rows
  unpaired 5/5.** When one recording appears in several overlapping w-layer GT
  rows, the 1:1 nearest-row pairing starves the others.
- **Many-to-many rescue is small:** only 45 s of 307 s BB12 (15%) / 4 s of 36 s
  BB11 would flip to correct under many-to-many pairing. So a scorer fix helps
  at the margin, not the core.
- **Unpaired GT rows are mostly identity overlay, not absence:** BB12 549 s
  unpaired, 92% claimed by a *wrong-recording* predicted span sitting on top —
  i.e. the timeline placed a different recording there (identity/inventory), not
  a pairing bug.

**Proposed minimal scorer change (NOT applied):** allow a GT recording that spans
multiple overlapping w-layer rows to be paired many-to-many (one predicted span
may satisfy several GT rows of the same recording). Upside is bounded (~15% of
the STARVED bucket ≈ +0.6 pp pooled) — file it, don't prioritize it.

## What this means for the grid

1. **Acappella "trajectory" moves most by fixing set_start placement, now** — the
   full `infer` re-run with `--stem-placement` (HuBERT) on the formerly
   mis-routed acappellas is exactly this lever; re-measure after.
2. **BB11's placement gap is partly stem coverage** (15 reference tracks lack
   vocals stems → weaker HuBERT placement). The BB11 stem backfill closes it.
3. **The decode-residual wall (B5, 31%) is the only bucket gated on BB10** — the
   instance-selection model. Everything else is shippable without new GT.
4. The oracle↔e2e gap is **not** a scoring illusion: correcting for STARVED+EXTENT
   the grid's acappella numbers are honest to ~1 pp.

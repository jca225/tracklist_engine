# Alignment Walls — a research cockpit

A hands-on sandbox to **assess a set and understand *why* the aligner fails**,
built for poking with your ears, not just staring at metrics. Two binding walls:

- **Placement** — *where* in the mix a track sits. It works within a set but
  **does not generalize** across sets (the per-set model memorizes placement).
  Roughly a third of lost seconds.
- **Structure** — *which* internal span, in what order (the "which repeat of the
  chorus" problem). The biggest bucket of loss. The hard truth: **~half of
  repeats are physically indistinguishable takes** — when the reference audio is
  identical at two positions, *no algorithm* can pick the one the DJ used.

Headline numbers live in `docs/alignment_status.md` (the SSOT — don't trust
figures hand-typed anywhere else, including here).

## Run it

From the **repo root**, with the `venvs/audio` kernel:

```bash
venvs/audio/bin/jupyter lab eda/alignment/walls/walls.ipynb
```

The notebook is thin; the logic lives in `wall_lab.py` (unit-tested:
`venvs/audio/bin/python -m pytest eda/alignment/walls/test_wall_lab.py`).

## What's in the notebook

1. **Set X-ray** — per-span predicted-vs-GT scorecard, worst-first, with a
   transparent binding cause (`identity` / `placement` / `structure/decode`).
   *Needs a predicted timeline* — the notebook prints the one-off command to
   generate it if missing.
2. **Structure / distinguishability** — for each repeat, slides the played
   content over the reference and reports the best-vs-runner-up **margin**.
   `ambiguous` (tiny margin) = the ceiling; `recoverable` = a beatable miss.
   This is the empirical read on "how much of the wall is physics vs. an
   engineering gap." Needs only GT + audio.
3. **Listen** — renders the played span + the two competing reference positions
   as clips. If *you* can't tell them apart, neither can the model — that span
   is the ceiling made audible.
4. **Scratch** — data pre-loaded for your own experiments.

## The one honest question this answers

For any misplaced span: **was it a beatable miss, or physically unwinnable?**
That decides whether the wall is worth attacking. If most of the structure wall
is `ambiguous`, the lever isn't a smarter decoder — it's **calibrated
abstention** (flag it for a human) plus attacking only the `recoverable` slice.

## Do NOT re-walk these (already measured and dead)

Reading before experimenting saves days:

- **Decode-geometry tie-breakers** for instance-selection (flat penalty,
  directional, magnitude-graded `--warp-jump`) — all net neutral-to-negative.
  See `workspaces/alignment_prototype/looptrace/NOTES.md`. Audio is identical
  across a true repeat, so no position/geometry heuristic can disambiguate.
- **"Make synthetic emit loops/multiseg"** — already done; the generator emits
  them (`generate_v2 → labels_v2`). The un-run lever is repeat *ambiguity*, not
  *presence* (see `docs/superpowers/specs/2026-07-17-instance-selection-arbiter-design.md`).
- **Chroma on acappellas** — broken on re-pitched vocals; the tool auto-routes
  vocals to HuBERT. Don't "fix" chroma for vocals.
- **Pooled-MERT for localization** — identity only; it cannot localize
  (~900 s off unconstrained).

## Caveats (so you trust the numbers)

- Distinguishability is computed at **native tempo** (no stretch search) — a
  first-order read; steep tempo rides will look *less* distinguishable than they
  are. Force a stretch-aware comparison yourself if a span looks wrong.
- The binding-cause column in the X-ray is a **transparent heuristic**, not the
  canonical failure attribution (`eda/alignment/failure_analysis/` is that).
  Use it to triage, then confirm by ear.

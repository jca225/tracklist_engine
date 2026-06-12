# audience_prior — listener-conditioned prior over a set's tracklist

Information-dynamics surprise is *listener-relative*. The models in
[`../alignment/info_dynamics`](../alignment/info_dynamics) used a **blank-slate**
observer (prior starts uniform, learns only from the mix). A real listener walks
in with a prior over the catalog from their history — and a track's salience is
how *familiar* it already is, not its acoustics. This module builds that prior.

Specifically it answers: **why was an acappella chosen?** The acappella is the
recognition "payload"; its effect is "oh, I know this vocal!" — which a MERT
embedding cannot see (and is exactly why acappella *audio* surprise was at
chance, info_dynamics §v4). The recognition lives in the **listener prior**, and
we have it: the SoundCloud per-user like graph collected in
[`workspaces/taste_prior`](../../workspaces/taste_prior) (`taste_warehouse.db`).

## Run

```bash
venvs/audio/bin/python -m eda.audience_prior.run   # BB12, ~5 s
```

`familiarity(track | audience)` = fraction of the set's non-bot SoundCloud
audience (~4.8k enriched listeners for BB12) who have liked the track, via a
fuzzy artist+title join ([match.py](match.py)) to their 532k-track like
vocabulary. Read-only on `data/taste/taste_warehouse.db`; writes
`data/analysis/audience_prior/` (track CSV + summary).

## BB12 finding — real at the top, noisy in aggregate

- **The prior cleanly recovers the set's recognition anchors.** The most-familiar
  tracks are exactly the famous vocal hooks used as acappellas — *Love On Me*,
  *Congratulations*, *In The Name Of Love*, *Rather Be*, *Latch* — each liked by
  ~12–13 % of the audience, independently of this set. That is a strong,
  usable **payload feature** for the DJ-selection model.
- **But the acappella-vs-instrumental split does *not* hold in aggregate.**
  acappella mean familiarity 0.021 < instrumental 0.029; top-15 is 73 % acappella
  vs a 60 % base rate (reverts by top-25). Two reasons: acappella familiarity is
  **heavy-tailed** (a few mega-hooks, a long obscure tail), and `claimed_stem`
  labels are noisy/duplicated (a known issue — some famous vocals are tagged
  `instrumental`). So "acappella = always-known payload" is too strong; the true
  statement is "**the biggest recognition moments are acappellas**."

**Takeaway.** The listener prior supplies a genuine recognition signal the
embedding can't — strongest precisely where it matters (the hooks) — and is the
right home for the acappella "payload" term in the selection model. It is *not* a
transition detector (listener attention isn't at the seams; comment density vs GT
boundaries was at chance).

## Caveats / clean next version

- **Endogeneity.** These listeners engaged with the BB12 upload, so familiarity is
  partly circular. The causal version builds each listener's prior from history
  **excluding this set** and tests whether the DJ *anticipated* familiarity, ideally
  generalizing across sets (BB11's audience is user-disjoint but track-overlapping).
- **Match precision/recall unaudited.** Token-containment + artist guard; top
  matches eyeball-correct, but recall on obscure tracks is unknown — an obscure
  instrumental absent from the vocabulary correctly scores 0, but so does a
  mismatch. n is small per stem (22 instrumentals).
- **One set, one audience.** BB12 only.

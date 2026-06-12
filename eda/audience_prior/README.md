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
[`personalization`](../../personalization) (`taste_warehouse.db`).

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

## De-circularized: cross-audience transfer (`run_cross.py`)

```bash
venvs/audio/bin/python -m eda.audience_prior.run_cross
```

Scores BB12's tracklist with the **disjoint BB11 audience** (2,936 listeners, **0
shared users** with BB12, but 154k shared liked tracks). This removes the
endogeneity: BB11's listeners never engaged with BB12, so their familiarity with
BB12's tracks comes purely from their own history.

**Result: familiarity transfers almost perfectly — Spearman ρ = 0.956, Pearson r
= 0.946** (n=140 tracks known to ≥1 audience). BB12's top hooks are independently
known by the disjoint audience at comparable rates (Congratulations 12.6% own /
11.4% disjoint; In The Name Of Love 12.4% / 9.2%). So a track's recognizability is
an **intrinsic, transferable property**, not a BB12-engagement artifact — which is
precisely what lets a DJ *anticipate* it and what makes the prior reusable for
personalization across sets/audiences.

## Caveats / clean next version

- **Endogeneity — resolved** by `run_cross.py` above (ρ=0.956 across user-disjoint
  audiences).
- **Selection test still owed.** Transfer + magnitude show the chosen tracks are
  familiar; a full *selection* test (did the DJ pick familiar tracks over
  equally-available unfamiliar ones?) needs a **negative candidate pool** —
  other DJs' tracklists as non-chosen alternatives, or popularity-matched
  controls (aux.db charts/Last.fm) to isolate audience-specific familiarity from
  generic global popularity. That is the next step.
- **Match precision/recall unaudited.** Token-containment + artist guard; top
  matches eyeball-correct, but recall on obscure tracks is unknown — an obscure
  instrumental absent from the vocabulary correctly scores 0, but so does a
  mismatch. n is small per stem (22 instrumentals).
- **One set, one audience.** BB12 only.

# The Tracklist Data Engine — the plan, in plain words

_Written 2026-07-10. Plain-language companion to the alignment objective + north
star. No jargon on purpose._

## What we're actually building

Picture a DJ mix: one hour-long audio file, ~150 songs blended into each other.
We usually also have the **tracklist** — the list of which songs, in order.

We want the computer to **reverse-engineer the mix**: for every song, figure out
where it starts and stops in the hour, how it was sped up / slowed down / pitched,
and which chunk of the song was used — then write all of that into an **Ableton
project file** you could open and see the whole mix rebuilt on a timeline.

- **Goal A (now):** tracklist + mix + the song files → the Ableton file.
- **Goal B (the dream):** just the mix, no tracklist → the computer figures out
  the songs itself, then builds the Ableton file.

## Why we call it a "data engine" and not just "a model"

Nobody has a big dataset of "DJ mix → exact Ableton rebuild." The only real
answer key is what John makes **by hand in Ableton**, one set at a time. That's
slow, so we don't train once and stop. We build a **loop**, like the one Tesla
uses for self-driving:

1. The computer takes its best guess at the whole mix.
2. It flags the spots where it's **least sure**.
3. John fixes only those spots in Ableton.
4. The computer learns from the fixes and guesses again — better this time.

Each turn of the loop improves it without John labeling everything. **The
hand-labels are the fuel; the loop is the engine.** (This is a known, proven
shape — Tesla/Waymo call it a "data engine"; the ML world calls it a "data
flywheel.")

## The three layers, bottom to top

Think of it like a kitchen.

1. **The pantry** — the song files. You can't rebuild a recipe if the
   ingredients are mislabeled. Every song needs the **right version** (original
   vs remix), the **right form** (full song / vocals-only / instrumental), and
   the **right length** (radio edit vs extended). If the pantry lies, everything
   above it is wrong.
2. **The chef** — the model that takes clean ingredients + the mix and places
   each song on the timeline. We have the chef's _skills_ scattered around
   (recognizing a song, finding where it sits) but not yet **one trained chef**.
3. **The kitchen loop** — the engine that turns John's corrections into lessons,
   keeps the chef improving, and **re-orders missing ingredients** when the chef
   gets stuck.

## The plan, phase by phase

### Phase 0 — Clean the pantry + install a smoke alarm _(mostly done today)_

- **Found and fixed a bug** that was quietly **deleting the "full song" copies**
  whenever a re-download hiccuped. It had eaten ~23 of them, including one song
  50 mixes use.
- **Put the eaten songs back** (DJ Kool, Chainsmokers, and 3 smaller ones).
- **Built a smoke alarm:** `make check-corpus` — one command that scans the whole
  pantry and shouts if any song is mislabeled or missing its reference. Right now
  it reports **zero serious problems**.
- **Leftover:** push the bug fix out to the storage machine (`make deploy`) so it
  can't happen there either. _(Waiting for the current download run to finish
  first.)_

### Phase 1 — Fill the pantry for the sets we're aligning

The smoke alarm found the real gaps: **~2,000 songs are missing their
vocals-only track, ~450 missing their instrumental, and "extended" versions
barely exist.** Most of these we don't need to hunt for — we can **make them for
free by splitting the full song** (the separator we already run). So: wire up
"if a slot needs vocals and we have the full song, just make the vocals stem
automatically." Only truly-missing songs get hunted down.

### Phase 2 — Train **one** chef on John's answer key (tracklist → Ableton)

Right now the chef is a pile of separate tricks glued together by hand. Combine
them into **one model** trained on the sets John has labeled (BB10, 11, 12,
Murph, Disco Lines). Useful insight from the research: John's separate tricks
(fingerprint, voice-match, lyric-match) are already **"opinions" the computer can
learn to weigh automatically** instead of us hand-tuning the mix — a standard
technique (Snorkel) does exactly this. Ship the new chef when it beats the
current hand-tuned score.

### Phase 3 — Turn the engine on itself (the flywheel)

Wire the loop so it runs mostly on its own:

- Chef guesses on a new mix.
- The **shaky spots** (where it's unsure or refuses to answer) go automatically
  to **two places**: John's review screen (the port-8800 UI) **and** the
  re-download queue if the real problem is a bad/missing ingredient.
- John fixes a handful. The chef retrains. Repeat.

This is the Tesla loop applied to us. Off-the-shelf helpers make it cheap: one
tool (cleanlab) ranks "most-likely-wrong" rows so John always sees the worst
ones first; the correction ledger we already keep becomes the engine's memory
**and** the training data at the same time.

### Phase 4 — Take off the training wheels (audio → Ableton)

Once the chef is reliable _with_ a tracklist, remove it: instead of being told
the songs, the chef **proposes** them from a big catalog, **says "I'm not sure"**
when it should, and a human confirms the uncertain ones. That's Goal B — the mix
goes in, the Ableton rebuild comes out, no tracklist required.

## The one-line version

**Clean pantry → one trained chef → a loop that fixes its own mistakes → drop the
tracklist.** We finished cleaning the pantry and installed the alarm today;
everything above it is the road to the north star.

# Open-set alignment — the end state

The north-star form of the aligner: **label a DJ set from its audio without a
hand-curated per-set tracklist** — naming what it can against a reference corpus,
flagging what it can't, and keeping a human in the loop only for the genuinely
hard cases. This is *not* "audio alone with zero references" (naming a song always
needs *some* prior recording); it is "no per-set hand tracklist."

## The two problems, kept separate

- **Segmentation** — *when* tracks change, transitions, structure. Doable from
  audio alone, no references (the info-dynamics work: surprise localizes
  boundaries). This part needs no tracklist at all.
- **Identification** — *which* track. Requires comparison to a reference. Remove
  the per-set tracklist and you face **open-set retrieval** against a large corpus
  + an **abstain** decision ("not confidently anything I know").

## Pipeline

```
set audio
  → segmentation (audio-only)                     # boundaries, no names needed
  → identify vs corpus (MERT/chroma fingerprint index, open-set)
        confident  → name it
        low-conf   → flag "UNKNOWN" (timestamps)  # = the abstain/margin gate we built
                       → recognize_segment.py: ACRCloud/AudD on the CLEAN stem
                            hit  → candidate ID for the annotator to confirm
                            miss → still unknown
                       → human confirms / corrects
                            found online      → acquire_variant downloads the reference
                            truly unreleased  → "use set audio as self-reference"
  → confirmed rows feed back as training data + references
```

Every box except the recognition call already exists in some form:
- **abstain/unknown detector** = the margin / Kim-et-al. match-rate gate
  ([[project_abstention_margin]], built in `workspaces/section_hsmm/`).
- **recognition step** = `scripts/recognize_segment.py` (pluggable ACRCloud/AudD;
  see below).
- **acquire the found reference** = `scripts/acquire_variant.py`.
- **self-reference fallback** = already in GT as `mix`/`mix_instrumental`
  "original unavailable" rows (BB12).

## Two flavors of "unknown" (this decides the tool)

1. **Released but not in our tracklist/corpus** — covers, older songs, scrape
   gaps. A recognition API (ACRCloud/AudD) IDs these automatically. This is the
   bigger bucket, and exactly the live-set missing-tracklist case (e.g. Murph).
2. **Genuinely unreleased / live re-performance / one-off edit** — not in *any*
   database. No API helps; only human + **set-audio self-reference**.

## Recognition: ACRCloud (with AudD fallback), not ShazamKit/WASM

- No usable public Shazam API. **ShazamKit cannot be run via WebAssembly** (closed
  Apple system framework — nothing to compile to wasm, and wasm can't call native
  frameworks). Using ShazamKit at all means PyObjC or a Swift CLI subprocess, and
  it's **Mac-only**.
- **ACRCloud** (REST, cross-platform, custom catalogs, cover/live recognition
  modes — transform-tolerant) is the better fit for a Python/partly-cluster
  pipeline. **AudD** is the easy second opinion. Recognizer is **pluggable** so we
  A/B and never lock in; ShazamKit custom-catalog stays a free/offline backup.
- Feed the **FULL mix** segment, NOT the isolated stem — fingerprinting leans on
  the vocals, so the instrumental stem matched *worse* in testing (BB12 @600s:
  full-mix hits, instrumental stem = no match). Recognition only runs on *flagged*
  segments, so per-call volume (and cost) is low.
- **Recognition is reliable for the SONG, NOT the VERSION.** From a
  tempo/pitch/EQ-transformed mix it snaps to the nearest catalogued master, so the
  remix/edit label is a guess (BB12 @150s -> "AJR - Weak (Jaded Remix)", but the GT
  is a filtered edit, not that remix). The recognizer proposes the song; the human
  confirms the version axis.

### The SoundCloud tier (the part commercial recognizers can't do)

ACRCloud/Shazam/AudD index **licensed commercial catalogs** and structurally miss
the **SoundCloud long tail** — DJ edits, bootlegs, white-labels, unreleased IDs —
which is exactly what live/edit-heavy sets play (Murph scored 0/2 in testing).
The fix is a **SoundCloud-sourced custom index**:
- Route A: an **ACRCloud custom bucket** populated with SoundCloud-sourced tracks
  (least infra).
- Route B: **our own fingerprint index** (chromaprint/MERT) over a curated
  SoundCloud corpus — = the corpus-scale index this doc already calls the missing
  build. The repo already downloads from SoundCloud (`acquire_variant`,
  `replace_track_audio`); the real work is curating + refreshing which accounts to
  index.

End state is a **two-tier recognizer**: commercial API for released tracks +
SoundCloud index for the edit/bootleg/ID tail, with truly-unreleased remainder
still falling to human + set-audio self-reference.

`scripts/recognize_segment.py` — prototype. `.env`: `ACRCLOUD_IDENTIFY_HOST`,
`ACRCLOUD_ACCESS_KEY`, `ACRCLOUD_ACCESS_SECRET`, `AUDD_API_TOKEN`.
First test: known BB segments (measure accuracy vs GT), then the Murph gaps.

## How today's work bootstraps this

The tracklist-supervised aligner we are building now **generates the training data
and the identity representation** the open-set retriever runs on. Standard arc:

> supervised bootstrap (tracklist-guided) → learned identity index → open-set
> retrieval (tracklist-optional) + human-in-the-loop unknown handling

The tracklist degrades from a hard dependency to a **bootstrap + validation**
signal. The missing build is the **corpus-scale fingerprint index** (the repo's
empty `track_fingerprints` / `set_fingerprint_hits` tables are the intended home).

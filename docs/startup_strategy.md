# Startup Strategy — One Engine, Multiple Lines

*Drafted 2026-07-10 from founder strategy discussion. Companion to
[superpowers/specs/2026-07-10-dj-agent-design.md](superpowers/specs/2026-07-10-dj-agent-design.md)
(the shelved DJ-agent spec, which this document argues to unshelve).*

## 1. Vision — the Isabella test

Empower people to actuate musical ideas the way vibe-coding let non-programmers
ship software. The canonical user moment:

> Isabella decides she wants a mashup **just before going on a run**. Snap of
> the fingers on her phone. She types one sentence, hits play, runs.

Every product decision is tested against this moment. What it implies:

- **No DAW, no downloads, no files.** Nobody under 30 has MP3s; they have a
  Spotify/Apple Music subscription. The phone-instant vision is a *catalog*
  product — the songs must already be there.
- **Playback is the job.** Isabella needs to *hear* the mashup on her run, not
  export a WAV. This asymmetry is load-bearing for the legal strategy (§3).
- **Latency must disappear, not merely shrink** — via precompute, not faster
  per-song processing (§5).

## 2. Product shape — one engine, two surfaces (the Claude Code analogy)

Same brain, two skins — exactly the claude.ai / Claude Code split:

| Surface | User | Audio source | Output | Legal posture |
|---|---|---|---|---|
| **Instant app** (phone) | Isabella, Nick | Streamed catalog (partner API) | In-app playback only | Streaming-partner license |
| **Workspace** (desktop) | DJ-adjacent friends, prosumers | User-supplied files | Rendered audio + round-trippable `.als` | Tool on user content (DAW posture) |

The two surfaces are not just market segments — they close a flywheel:

- The **workspace** is the legal wedge that ships first (user brings audio,
  same posture as Serato/Ableton — legally boring), and its power users
  generate correction data: every edit a DJ makes to a generated `.als` is a
  labeled preference, i.e. free RLHF for the arrangement model.
- The **instant app** consumes the model the workspace data trains. It is the
  volume product and the reason the company is big rather than a plugin.

## 3. Copyright — a wedge answer, not a solution

The first YC partner question. "We'll figure it out" is a rejection. The
answer is three doors, walked in order:

1. **Door one — open today: user-supplied audio (workspace surface).** A
   blender company doesn't need to own fruit. Ableton/Serato/djay posture.
   Immediately shippable, demo-safe.
2. **Door one-and-a-half — open today: streaming-partner catalog, playback-only
   (instant surface).** Precedent: **Apple Music opened its API to DJ apps in
   2024** (djay, Serato, rekordbox); djay does real-time on-device stem
   separation on streamed tracks. Beatport LINK and TIDAL run similar
   programs. Restriction: playback only, no export/recording of mixes using
   streamed content — which the Isabella use case doesn't need. This is the
   pitch line: *"djay proved the catalog can legally stream into DJ software;
   we're the version where you don't need to know how to DJ."*
3. **Door three — opening: licensed AI remix.** UMG's 2025
   settlement-and-deal with Udio is explicitly about licensed AI
   remix/creation products. A remix-rights regime is forming. Dubset died
   waiting for labels to move (assets → Pex, 2020); the difference now is the
   labels are moving. Whoever holds the best mashup engine when this door
   opens wins it. Sharing/publishing generated mashups lives behind this door
   — roadmap, not launch.

## 4. Strategy — threatening multiple lines (Liddell Hart), with cautions

Liddell Hart: hold dispositions that credibly threaten **alternative
objectives**, so no defender can concentrate against you. Our single line of
operations is the **engine** (alignment-trained arrangement intelligence +
audio infra). It credibly threatens four product lines:

| Line | Who defends it | Why our threat is credible |
|---|---|---|
| Instant consumer mashups | Suno/Udio (bolt-on remix), Spotify AI DJ | They generate songs; we understand *mixing as a craft* (trained on real DJ GT) |
| Prosumer DJ co-pilot / workspace | Algoriddim, Serato, DAWs | `.als` round-trip codec + arrangement model; they have separation, not arrangement |
| Mix attribution / royalty B2B | Pex, BMAT, Audible Magic | The aligner *is* attribution tech (Dubset's lane, better engine) |
| Licensed remix platform (door three) | Nobody yet | Positioned asset when the UMG/Udio regime generalizes |

The point is **not** to build all four. Liddell Hart cuts both ways for a
solo founder: threatening multiple lines with your *dispositions* is strength;
attacking on multiple lines with your *forces* is dispersion and death. Rule:

> **One engine, many threats, one attack at a time.** The shared engine keeps
> every threat credible while 100% of build effort concentrates on the current
> line (workspace demo → instant app).

**Blitzscaling caveat (Hoffman):** blitzscaling is prioritizing speed over
efficiency *after* product-market fit + capital. We are pre-PMF. The current
phase is the opposite discipline — nail the one moment (§7) before scaling
anything. Premature blitzscaling is the classic death; the word doesn't enter
the vocabulary until Isabella's cohort retains.

## 5. The gates, ranked honestly

The founder's instinct — "next SOTA gate is incredibly fast processing at
scale" — is **half right**. Speed matters, but as a systems trick, not a
research frontier:

1. **Does the mashup slap?** (research risk — the real gate.) Arrangement
   quality: key-clash avoidance, vocal placement, loop/transition grammar.
   This is what the alignment GT uniquely teaches and what no infrastructure
   budget buys. Rave.dj failed here: technically aligned, musically dead.
2. **Catalog access** (BD risk). Getting into Apple/Beatport partner
   programs. Meetings and compliance, not code. Algoriddim did it small.
3. **Processing at scale** (money, not mystery). Full research pipeline runs
   ~40–85 s/track on spot GPUs; a mashup needs a slice of it (stems, key,
   BPM, grid). Precomputing the top ~100k songs is hundreds of dollars,
   once. Latency disappears via cache: nothing is separated when Isabella
   taps "create" — arrangement + render only, seconds. Anyone can buy this;
   it is necessary and not a moat.

## 6. Assets already in hand (repo → product map)

| Repo asset | Product role |
|---|---|
| Roformer separation pipeline | Stems (precompute + on-device path exists in the wild) |
| Essentia key/BPM + beat grids | Harmonic/tempo matching |
| Micro-pitch detune estimator | Fine pitch matching |
| Warp priors (`warp_prior.json`) | Realistic stretch/arrangement priors |
| `.als` codec (`labeling/als/`) | Workspace round-trip — the compiler backend |
| Alignment GT (BB11/BB12, Ableton) | The moat: how real DJs actually blend |
| Personalization layer (`personalization/`) | Taste priors for the instant app |
| Phase-cancel toolkit | Stem/clone verification |
| Shelved DJ-agent spec | The missing middle: intent → arrangement decisions |

The demo is closer than the aligner: mashups from clean stems are the *easy*
end of the problem space vs. aligning a warped 60-minute set.

## 7. YC

**Apply — with the mashup compiler, not the aligner.** The aligner is the
moat story; the compiler is the product.

- **The demo that gets in (and the real PMF probe):** a friend picks two
  songs (their own files — door one), types "put the vocals from A over B,"
  and ~60 s later hears a listenable mashup and can open the `.als`. Demo
  script ends on Nick's phone playing the result. Application names the Apple
  Music DJ-API program explicitly so "instant/phone/legal" isn't a hand-wave.
- **Test:** put it in front of five friends; the signal is an unprompted
  *second* idea. No second idea → learned cheaply.
- **Solo-founder handicap is real.** Four jobs at once: model, infra,
  consumer product, eventually label BD. DJ friends are a **user pipeline,
  not a cofounder pipeline** — recruit a builder, strongest where founder is
  weakest (consumer product/frontend).
- **Costs stated plainly:** ~$500k for 7%, full-time, SF, one batch. This is
  a quit-the-day-job decision wearing a "should I apply" costume.

## 8. Sequence

1. Unshelve the DJ-agent spec; scope the minimal intent→mashup→`.als` path.
2. Build the desktop workspace demo (door one, tests gate #1 only).
3. Five-friend test; measure unprompted return.
4. YC application: demo video + door-one/door-three copyright answer +
   Apple-Music-DJ-API citation.
5. Instant-app surface only after gate #1 evidence; partner-program BD in
   parallel with batch.

## 9. Precedent ledger

| Company | Lesson |
|---|---|
| **Dubset** | Right tech (mix attribution), died waiting for labels; squeezed between platforms and label pace. Don't build a business gated on label speed. |
| **Udio/UMG (2025)** | Labels now *sign* AI-remix deals. Door three is opening. |
| **Algoriddim djay** | Small company got Apple Music/TIDAL/Beatport streaming into a DJ app, on-device stems. Door 1.5 is real. |
| **Rave.dj** | Auto-mashups with dead musicality don't retain. Gate #1 is quality, not capability. |
| **Suno/Udio** | Fund-raised through the lawsuit gauntlet; will bolt on remix. Our differentiation is craft-level mixing knowledge, not generation. |

# Handoff — PWS aligner + co-training data plan (2026-07-16)

Context got long; this hands off to a fresh session. Two threads: (1) finish the
in-flight stem gap-fill + start downloading a **grammar-diverse** co-training
corpus, and (2) continue the **PWS aligner + co-train** plan (synthetic-transition
probe). Read alongside memory: [[project_cotraining_acquisition_frame]],
[[project_durable_compute_buffer]], [[project_low_rank_worldview]],
[[project_als_grammar_roundtrip]], [[project_dj_set_alignment_domain]],
[[project_pws_aligner_reframe]].

---

## 1. LIVE STATE — do not break

**Stem gap-fill of the 27 Big Bootie sets is ~98% done, draining on ONE Vast box.**

- **Box b** = instance `45018139` (label `pws-gapfill-b`), `ssh vast` (direct
  `23.91.224.128:42266`). Box a (`44996114`) was destroyed (flaky link); box b now
  runs **unsharded** and owns all remaining.
- **~54 tracks remaining** (of 3293 in-scope; done ≈ 989 this box + prior). ~40–60 min.
- The loop runs in **tmux session `loop`** executing `/workspace/loop_wrap.sh`
  (auto-restart wrapper, gated on sentinel `/workspace/RESUME_OK`). Log:
  `/workspace/vast_loop.log`. Chain = RoFormer p1 (single bs_roformer, `analysis/roformer_chain.yaml`).
- **Check progress:** `ssh vast 'grep -c "TIMING tid=" /workspace/vast_loop.log'`
  and the remaining-count SQL in §6.
- **If the loop dies:** relaunch with tmux (NOT setsid/nohup — the container reaps
  orphans; NOT `tmux kill-server` — that broke it this session):
  `ssh vast "cd /workspace/tracklist_engine && tmux new -d -s loop 'bash /workspace/loop_wrap.sh'"`.
- **Monitors are session-scoped** — the stall/drain monitor from this session is
  gone. Re-arm one in the new session if you want drain alerts.

**When the 27 sets drain → two closing tasks:**
1. **Set-side stems** for the uncovered BB sets (`set_stems` was ~2/27 — verify; it
   keys on `set_audio_id`, not `set_id`). Run set separation on one box.
2. **Destroy box b** (`vast_box.py destroy 45018139`) + hand the user the teardown
   lines (pi `authorized_keys` sed for `vast-pws-gapfill-b-45018139` + drop tailnet node).

**pi-storage** recovered from a ~4h power-loss outage earlier (01:47 EDT unplug).
DB verified (WAL checkpoint clean), 29 GB backup at
`/mnt/storage/backups/music_database_recovered_20260716_0905.db`, persistent
journald now enabled. **User TODO:** physical power / UPS on the Pi.

---

## 2. Committed this session (branch `pws-alignment-reframe`)

- `216d985` **rsync `-qs` fix** in `scripts/vast_loop.py` — `shlex.quote` failed over
  the SOCKS proxy and silently stalled every space-named (manual-ingest) pull.
  **Already fast-forwarded to `origin/main`** so future box clones get it.
- `e76a2b2` **`vast_box.py race --n N`** — parallel dud-avoidance (rent N, keep first
  port-22-healthy, destroy+quarantine losers). Ported from `jspace-hrm/runs_infra`.
  Pure `race_step`+quarantine unit-tested. On `pws-alignment-reframe` (Mac-side tool,
  not cloned to boxes, so FF-to-main optional).
- `8740721` **durability spec fold** — `docs/durable_compute_buffer_spec.md` now maps
  `jspace-hrm/runs_infra` component-by-component as the **port source** for the
  deferred P1 (S3 buffer + reconciler).
- PWS agent build-out (A5, B1–B4) landed earlier in the worktree
  `~/Desktop/tracklist_engine-pws1b` on branch `pws-phase1b-continuous` (187 tests).

---

## 3. The plan — decisions reached this session

**Aligner target = a "Turing-complete-over-DJ-moves" grammar.** The aligner's output
must be able to *represent any move a DJ could make*. Bounded DJ vocabulary =
achievable; the open DAW/production tail (any VST/automation) = **abstain**, don't
chase completeness there.

**The DJ-move grammar (mashups are ONE cell, not the whole thing):**
```
PLACEMENT/TIME          IDENTITY/LAYER            TIMBRE/FX (fuzzy tail → abstain)
offset                  straight play             EQ / filter
constant warp           overlay / MASHUP          gain / sidechain
continuous warp (RIDE)  version (remix/edit/VIP)  echo / delay / reverb throws
loops                   unreleased / ID           gating
jumps / cuts / hot-cue
reverse / spinback
```

**GT is scarce; it's for VALIDATION, not training.** Hand-labeling ≈ 3 weeks per
*mashup* set (BB11/BB12 done); transitions are ALSO hard (continuous tempo = a moving
warp curve, "unexplored"). GT-cost is **regime-dependent**, not flat. User can squeeze
**one** more real set but "don't count on it — see if 2 is enough." A 3rd, if used,
should be chosen for max distance from Big Bootie (a transition set), and the
correction-loop (aligner proposes → human fixes) is the only way it's feasible.

**Three data sources, three roles:**
- **Synthetic** (`.als → audio`, known labels): generate *any* grammar move, clean
  manifold, at scale, free. For **MEASURE (alignment) only — never JUDGE (taste)**.
  Generate from the low-rank knobs + grammar so it lands on-manifold (low-rank is the
  *recipe*, not the enemy). Synthetic = the recreation stack reversed (see north star).
- **Download** — a **large, diverse, unlabeled** pool of real sets + constituent tracks
  = the **co-training substrate** (flywheel fuel; the aligner learns from this at scale,
  which is *why* 2 GT sets suffice). This is the "we need a lot of DJ data" point.
- **2 GT sets** = validation anchor + sim-to-real lie-detector (if aligner aces
  synthetic but flunks BB11/BB12, that gap = the off-manifold tax).

**"Chosen right" = grammar coverage, NOT popularity.** Current 1,016 downloaded sets
are popularity-seeded → EDM/mashup-skewed (median 7.6k views; top = Hardwell/DVLM
festival sets; DJ parsed from title, 456 distinct). Expansion must rebalance toward
the *underrepresented moves*.

**Measure vs Judge** resolves the "does synthetic conflict with low-rank + subjective?"
worry: the aligner measures the *result* of taste, never models taste. Grammar = the
**structural shadow of taste** (skeleton); taste = the flesh (real data + SoundCloud
priors only). Synthetic needs the skeleton, never the flesh.

**Transition/gradual-tempo regime is IN SCOPE.** Representation = permanent commitment
(grammar must express it). Recovery-by-Aug-1 = an *empirical* question the
**synthetic-transition probe** answers (build fake tempo rides with known curves, feed
the aligner, measure recovery). Don't decide in/out blind — the probe is cheap, let it
pick the branch.

**North-north (DEFERRED behind Aug 1, keep in mind):** open-set — hear ANY mashup/set
(constituents not in our library), identify, source online, **recreate from scratch**.
Recreation = `audio → .als`; synthesis = `.als → audio` — same grammar reversed, so the
synthetic pipeline is half the recreation stack. SoundCloud corpus expansion + taste
model (subjective priors) live here.

---

## 4. How to pick the co-training expansion (the immediate next work)

**Two-stage funnel (moves split into scrape-visible vs audio-only):**

- **Tier-1 (metadata proxies — cheap, drives *selection*):** per-set fingerprint over
  the scraped 41k: `density = total_tracks / play_time` (high→overlay/chops,
  low→straight/blend), **`w/`-fraction** (1001TL marks simultaneously-played =
  mashup/overlay), version-tag fraction (edit/rework/VIP/bootleg), stem-tag fraction
  (acappella/instrumental), **ID/unknown fraction** (unreleased-material corner),
  cue-gap stats, `styles`/genre. Stratified-sample to span this space, over-pulling the
  corners the 1,016 are thin on.
- **Tier-2 (audio-only — invisible in scrape):** loops, jumps, continuous tempo ride,
  key-mixing, effects. Can't select for these from metadata; the aligner/probes reveal
  them *after* download → measure real grammar coverage → top up next round. As the
  aligner matures it becomes the move-detector → **flywheel picks its own next data**.

**User's domain-knowledge seeds (blend with the corpus check):**
- **Alesso → mashups** (overlay corner, *different flavor* than Big Bootie — adds variety).
- **RUFUS DU SOL, ODESZA, Galantis(?) → own/original material.** ⚠️ **CAVEAT:** split
  their **DJ sets** (mix released tracks → alignable, good data) from their **LIVE PA
  sets** (perform originals live → no released constituent to align → open-set/unreleased,
  harder/later). Grab the DJ sets, not the live shows. Use tracks-per-minute + tracklist
  presence to tell which.

**Next action:** run the Tier-1 fingerprint query across `dj_sets`/`dj_set_rows` (verify
actual columns first — `w/`, version, ID markers live in the raw row strings /
`set_track_slots.claimed_*`), produce a coverage map (where we're dense vs empty), and
propose a stratified grammar-spanning list. Check the 4 seed artists (do we have sets?
DJ vs live? tracklists?). Then download + stem via the hardened pipeline
(`vast_box.py race`, rsync `-qs`, RESUME_OK wrappers).

---

## 5. PWS next build (spec, don't swing yet)

Once gap-fill drains, spec the **synthetic-transition probe**:
(a) verify the `.als` round-trip survives a **continuous** tempo curve (not just discrete
warp markers); (b) build the synthetic-transition generator (known tempo curve →
rendered audio + labels); (c) probe the aligner and **report recovery accuracy** on
gradual tempo. That number decides Aug-1 scope for the transition regime.

Also open (from [[project_cotraining_acquisition_frame]]): the co-training seam —
suspect-detector → `AcquisitionCase` producer, and candidate-ref → align-to-mix →
`TrainingSignal` + `track_audio_correction` ledger, GT-calibrated, ZERO canonical
mutation. `bb_reacquire_queue.json` = 879 fetch_missing-only (no executor wired yet).

---

## 6. Quick refs

**The 27 BB set ids:** `w1mgcjt,2nvzlh2k,1fsnxchk,1n81jy3k,1yl70ql1,237tdqmk,2vpur281,qj4v0wt,zwf3n2t,2cxndfmk,l6xqnhk,1jwtbspt,261s43wt,z0mhsf1,1mpqt5wk,2svckg31,21khc009,9l2wdv1,x5yyn4k,1kh4dbd1,2ckm8bjk,66wusst,8ktvhkt,hy83dh1,3b0k6zk,qgvujwt,1d9zwh49` (BB11=`2nvzlh2k`, BB12=`1fsnxchk`).

**Remaining-count SQL** (run on pi-storage; `$CSV` = quoted set-id list):
```sql
WITH scope AS (SELECT DISTINCT ta.track_audio_id AS taid FROM track_audio ta
  JOIN set_track_slots s ON (s.recording_id=ta.recording_id OR s.recording_id=ta.track_id)
  WHERE s.set_id IN ($CSV) AND ta.path IS NOT NULL AND ta.path!='')
SELECT COUNT(*) FROM scope WHERE taid NOT IN (SELECT track_audio_id FROM track_stems);
```

**Pitfalls learned this session:**
- Vast box loops: **use tmux** (its daemon survives ssh-close); never `tmux kill-server`;
  setsid/nohup get reaped. Wrapper gates on `/workspace/RESUME_OK`.
- rsync **stem-push** (large FLACs) can broken-pipe on a flaky SOCKS link — box a died on
  this; the durable-buffer P1 is the real fix. A `tailscale down/up` gave box a a *direct*
  pi link (fixed pulls) but is aggressive (can need re-auth).
- `set_stems` keys on `set_audio_id`, not `set_id`. `track_mert_measures` is the live MERT
  table (NOT `track_mert_sections`, which is legacy/empty).
- pi journald was volatile (rpi default `40-rpi-volatile-storage.conf`); now overridden by
  `/etc/systemd/journald.conf.d/99-persistent.conf` (Storage=persistent, 300M cap).

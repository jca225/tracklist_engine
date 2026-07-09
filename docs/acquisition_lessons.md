# Acquisition lessons ledger — everything downloading has taught us

Reconstructed 2026-07-09 from git history (all branches), docs, code comments,
the correction ledger, and live pi verification. This is the evidence base for
the consolidated ("SOTA") acquisition core: every requirement below traces to
a failure the corpus already paid for. Companion docs:
[entropy_reduction_plan.md](entropy_reduction_plan.md) (workstreams),
[alignment_program_plan.md](alignment_program_plan.md) (stem cascade),
`tests/fixtures/ingest/correction_ledger_snapshot_20260709.tsv` (frozen
1,267-row failure dataset), `eda/alignment/failure_analysis/` (aligner-side
findings). The human's revealed acquisition policy (what the Ableton GT
actually used) is summarized at the bottom.

## A. Chronological lessons (condensed — one line per paid-for lesson)

| # | When / where | Lesson |
|---|---|---|
| 1 | 05-04 `0aa3cb0` | Age-gated YT needs `--cookies` (DownloadConfig, shared) |
| 2 | 05-04 `d6c42cb` | yt-dlp needs a JS runtime or you get image-only formats (node autodetect) |
| 3 | 05-05 (5 commits) | spotdl papercuts: isolated venv, `sys.executable`, CLI 4.x, 120 s timeout |
| 4 | 05-06 `d64dc96` | **spotdl in the main chain = 0/174 in a 14 h run** → chain is `youtube → soundcloud`; spotdl = targeted retry only |
| 5 | 05-06 `3dcc258`+ | spotipy OAuth cache trap: new creds ignored until `~/.config/spotdl/.spotipy` deleted |
| 6 | 05-06 `eba1f22` | YT Music `filter='songs'` = Topic/label masters; provenance platform tag |
| 7 | 05-06 `c94dca4` | Destructive replace must verify the NEW file (exists, ≥100 KB) before deleting the old row (`rescue_common.phase2_replace`) |
| 10 | 05-13 `2cdb892` | **THE wrong-version bug**: bare `Artist - Title` query collapses remixes onto the original (~63k-row class) → search `full_name` with remixer qualifier |
| 11 | 05-30 review | The May 6–9 bulk (99.3% of rows) ran pre-fix → "re-source before re-label; never train on raw bulk" |
| 12 | 05-29 `99bc569` | Every manual fix must land in `track_audio_correction` (axis, old→new) — the ledger is now training data and a frozen test fixture |
| 13 | 05-29 `4ea868f` | Chromaprint identity check: advisory only (thresholds uncalibrated); acappella chroma = `WEAK_SIGNAL` honesty |
| 15 | 05-30 `3a39d98` | **Orphan class** (59/67 final-audio orphans): download-then-insert must be atomic → `core.db.insert_audio_or_reap` everywhere |
| 16 | 05-30 `d459583` | One YTM video landed byte-identical in THREE track folders (search collision) → cross-folder hash dedup; reconcile dry-run default |
| 19 | 06-09 (ledger) | **Stem-cand batch misfire**: 22 acappella candidates attached to unrelated recordings → sha256 cross-check guard (`0505217`) |
| 20 | 06-16 (3 commits) | Discord: no-token DOM paste mode wins; signed CDN URLs expire ~24 h → expiry-first ordering |
| 21 | 06-16 `57db131` | Mac "format not available" = missing EJS/node, NOT cookies |
| 22 | 06-17 `4220042` | URL-shape guards: bare SC profiles / YT channel links pull whole discographies |
| 23 | 06-17 doc | **Preview-clip cascade**: 46 s preview installed as full track → rescue then fetched the WRONG VERSION; 106 suspects audited |
| 24 | 06-18 `d0183c1` | **YTM `hits[0]` bug**: named remix silently resolved to the official original → version-token gate with refusal semantics |
| 26 | 06-23 `00c1a54` | **The fix didn't propagate** — Mac re-source scripts still took `hits[0]` → gates must live in ONE shared function |
| 27 | 06-23/24 | `core/acquisition_case.py`: per-slot decision traces + ProblemClass taxonomy; BB12 backfill = 155 cases |
| 28 | 06-24 `e5808c9` | Title-token hole: right-remixer-of-WRONG-song passed the version gate → `_hit_title_ok` majority-token gate |
| 29 | 06-24 `bad2ecb` | 3-way copy-pasted yt-dlp opts; the API path silently lacked `web_safari` → `ingest/ytdlp_profile.py` single home, test-pinned |
| 30 | 06-24 (3 commits) | Failure classifier + remedies (`preflight.py`); retry ONLY `kind='network'` — bot/no-JS/403 are deterministic |
| 33 | 07-07 `f678f3a` | `claimed_stem` row-text drop: markers living only in row text silently made acappellas `regular` |
| 34 | 07-08 `9bc36f1` | **`is_reference` stealing** (Birdy): stem adds are additive by default; promotion is explicit + ledgered |
| 35 | 07-09 `9ff03f8` | **D2**: Dropbox served login HTML as HTTP 200 → 52 junk files cached as audio; 151 error rows retried forever with zero recoveries → payload sniff + terminal `dead` status |
| 36 | 07-09 `fceff70` | Shared guards core seeded (`ingest/guards.py`): duration-vs-listed sanity wired into set-mix; suspects reaped not canonicalized |
| 38 | 07-09 `3b4d229` | D3 auto-accept: metadata accept AND audio confirm (HuBERT 0.55 / chroma 0.35 floors) → auto; disagreement → human review |
| 41 | doctrine | 'Acappella' label ≠ acap audio (verify via instrumental-stem silence); detect-then-correct, never blanket-redownload; check filesystem not just DB; source type dominates separator (official > community > separation floor) |

(Full 41-row table with root causes and scope lives in the 2026-07-09 research
transcript; the numbers above keep the original indices.)

## B. Guard matrix (2026-07-09)

Entrypoints: **M-t** `ingest.main` tracks · **M-s** `ingest.main` set-mix ·
**RES** ytmusic rescue · **RTA** replace_track_audio · **AV** acquire_variant /
ingest_stem_url · **DIS** discord harvest · **SCB** stem-candidate batch.

| Failure class | M-t | M-s | RES | RTA | AV | DIS | SCB |
|---|---|---|---|---|---|---|---|
| Bot/cookies | ✅ | ✅ | ◐ | ✗ | ◐ | ◐ | — |
| JS runtime | ✅ | ✅ | ◐ | ◐ | ◐ | ◐ | — |
| 403 web_safari (Mac) | — | — | — | ✗ | — | ◐ | ✅ |
| Wrong-version gate | — | — | ✅ | — | ◐ | — | ◐ |
| Wrong-song (title) gate | — | — | ✅ | — | ◐ | — | ◐ |
| Preview-clip / duration | ✗ | ✅* | ◐ (pi cap DISABLED) | ✗ | ◐ | ◐ | ◐ |
| HTML-interstitial sniff | — | — | — | — | — | ✅* | ✗ |
| Orphan atomicity | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| Verified two-phase replace | — | — | ✅ | ✗ **delete-before-insert** | ✅ | — | ✅ |
| is_reference stealing | — | — | ◐ | by-design | ✅ | — | ✅ |
| Content-identity QA | ✗ | ✗ | ✗ | ✗ | ✅ advisory | ✗ | ✅* |
| Idempotent re-run | ✅ | ✅ | ◐ | — | ◐* | ✅ | ◐* |
| Ledger + case trace | ✗ | ✗ | ✗ | ✅ | ✅ | ✗ | ✅ |

`*` = fix exists only on an unmerged branch (see Still-open #3).

## C. Still-open (verified live 2026-07-09)

1. **Birdy `is_reference` revert pending on pi** (23733 instrumental still
   reference over 21151) — blocks BB10 re-pull; two-line UPDATE, human-gated.
2. **track_audio 23070 double-encoded path** (mojibake in DB vs UTF-8 on disk)
   — rsync fails; one-line UPDATE pending.
3. **Hardening stranded on unmerged branches**: D2 HTML guard + `dead` status
   (`discord-retry-hardening`); `ingest/guards.py` + set-mix duration gate
   (`synthetic-warp-wiring`); sha256 re-run guard + D3 auto-accept. pi runs none.
4. **No track-level duration/variant gate**; variant axis essentially unset
   corpus-wide; radio-vs-extended acquisition unsolved.
5. 106 wrong-version suspects only partially remediated (~20 ledgered fixes).
6. **pi rescue duration cap disabled** (`max_duration_s=1e9` in
   `ytmusic_adapter.download_one`) — Mac path caps at 1200 s, pi doesn't.
7. `replace_track_audio` deletes the old row BEFORE inserting the new one —
   the exact class `c94dca4` fixed in the rescue.
8. Rvmor gap: sided rows without `data-trackid` never enter ingest at all.
9. Phantom-track mix-extract path designed, not built.
10. Chromaprint thresholds uncalibrated → identity checks all advisory.
11. Discord residue: 34 manual-host links; D1 scroll-completeness unchecked.

## D. Requirements for the consolidated acquisition core

1. One shared invocation profile + boot preflight; refuse-to-start with a
   named remedy. *(bad2ecb, 00c1a54, 9ba8bf1)*
2. Layered hit-selection with refusal semantics: role-aware query →
   version-token gate → title-token gate → duration cap. Wrong install is
   worse than no install. *(2cdb892, d0183c1, e5808c9)*
3. Payload verification before canonicalization (magic bytes + duration vs
   listed); reap suspects. *(9ff03f8, fceff70, preview-clip cascade)*
4. Atomic acquire→register; verified two-phase replace; never
   delete-before-insert. *(3a39d98, c94dca4)*
5. Additive by default; reference promotion explicit + ledgered. *(9bc36f1)*
6. Every mutation writes provenance twice (correction ledger + acquisition
   case). Corrections are the training signal for future gates. *(99bc569)*
7. Content-identity channel independent of metadata, calibrated per axis;
   sha256 cross-checks against wrong-recording attaches. *(4ea868f, 0505217)*
8. Idempotence keyed on content + on-disk reality, not row existence.
   *(identity_gate, d459583)*
9. Terminal `dead` states; retry only transient `network` failures. *(9ff03f8)*
10. One core covering the whole source topology (YT/SC/YTM/spotdl-retry/
    community hosts/mix-extract) under the solvability cascade: **source type
    dominates separator**. *(d64dc96, 4220042, program plan P2)*

## E. The human's revealed policy (from the Ableton GT, n=316 rows)

| rule | evidence |
|---|---|
| Full download first, always; nothing replaces it | 73/73 regular claims use the canonical rip |
| Acappella → search online studio acapellas FIRST | 80% of acappella claims resolved online vs 20% separated |
| Instrumental → separate FIRST, search opportunistically | 55% separated vs 39% online; ~59% of instrumental claims are marker-less remixes with no official instrumental |
| Fetch exactly ~3 candidates | cand1 wins 67%; ranks 2–4 supply 33% of wins; rank 4+ ≈ never |
| Separation is the floor — budget 27% fallback on stem claims | stable across BB11+BB12 |
| HuBERT vocal gate auto-promotes with an override lane | 84% agreement with the human (56/67); 16% overridden to demucs |
| Phase-cancel is DEAD as acquisition | 53 rendered outputs, 0/166 Ableton wins |
| Accept degenerate-axis resolutions | ~10% of acappella and >50% of instrumental claims have no true asset; verify by audio, don't force a label to exist |

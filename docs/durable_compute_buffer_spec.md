# Durable Compute Buffer — decoupling GPU analysis from pi-storage uptime

**Status:** SPEC / deferred (write-up 2026-07-16). Prompted by a pi-storage outage
(~05:47 UTC 2026-07-16, ~4h) that stalled both Vast GPU boxes mid-run and lost ~3h of
compute. Pairs with the scale-up to the 16k-track / 40k-set corpus — a single home Pi as
a hard dependency for all compute is unacceptable at that scale.

## Problem

`scripts/vast_loop.py` hard-depends on pi-storage for **both** ends of every track:

1. **Input** — pulls source audio *from pi* (rsync/ssh) before analyzing.
2. **Output** — pushes stems + DB rows *to pi* (ssh sqlite insert + stem files) after.

So pi-storage is a single point of failure for the entire compute fleet. When it went
down 2026-07-16:
- box b **crashed** on the first failed `ssh_pi` (`next_task`) call;
- box a **hung** blocked on a dead pi connection;
- ~3h of potential compute produced nothing;
- everything the boxes *had* computed was already on pi (fine), but no new work could
  proceed and a naive resume risked writing into a possibly-corrupted DB.

## Goal

Decouple ephemeral GPU compute from the canonical store's availability. Bound loss from
**any single failure** (pi outage *or* box death) to seconds/minutes, never hours. No
silent data loss. Secondary win: an off-Pi copy of stems+audio (pi stops being the only
copy).

## Design — durable object buffer + async reconciler

```
  Vast box(es)                  Durable buffer (S3/R2/B2)              pi-storage
  ───────────                   ─────────────────────────             ──────────
  pull input  ◄──────────────── s3://…/audio/{taid}/…  ◄── (mirror, once, while pi up)
  analyze
  write stems ──────────────►   s3://…/stems/{taid}/…
  write manifest ───────────►   s3://…/manifests/{taid}.json  ──►  reconciler ──► DB + /mnt/stems
                                (manifest = commit marker,           (idempotent
                                 written last, after stems)           DELETE+INSERT,
                                                                      integrity-gated)
```

### 1. Durable output buffer (the core change)
`vast_loop.py` stops pushing directly to pi. Per track it writes to an object store:
- stem FLACs → `s3://<bucket>/stems/{track_audio_id}/{vocals,drums,...}.flac`
- a **result manifest** → `s3://<bucket>/manifests/{track_audio_id}.json` containing:
  the DB rows it would have inserted (track_stems paths, track_mert_measures, beats,
  cues, lufs), per-file **checksums**, box id, git SHA, UTC timestamp.
- Manifest is written **last**, after all stems land — it is the atomic commit marker.
  A half-written track has no manifest and is simply redone.

Object store (S3 / Cloudflare R2 / Backblaze B2) is durable, always-available, cheap
(~$2/mo for ~100 GB), and independent of both pi and the ephemeral box. Vast boxes have
strong bandwidth.

### 2. Input staging
- **Recommended (scale):** mirror the batch's canonical audio into `s3://…/audio/` once
  (bulk copy from pi while pi is up); boxes pull input from S3, not pi. Fully decouples
  input *and* gives audio a second home.
- **Interim (cheap):** keep pulling from pi but with a local prefetch cache of depth N; a
  pi blip is survived for N tracks. Bounded, simpler.

### 3. Async reconciler (buffer → pi) — NEW service
Runs on pi (or pi-worker, or the Mac — anything that can reach pi):
- watches the `manifests/` prefix for unreconciled entries;
- per manifest: verify checksums → download stems to `/mnt/storage/stems/…` →
  idempotent `DELETE+INSERT` of the DB rows (keyed on track_audio_id + stem) → move the
  manifest to `reconciled/` (or mark in a ledger table);
- **integrity-gated**: runs `PRAGMA quick_check` before a batch of inserts; on failure it
  **halts and alerts** rather than writing into a damaged DB (this reuses exactly the
  gate we built during the 2026-07-16 outage — the parked-loops + `RESUME_OK` sentinel
  pattern generalizes to the reconciler);
- when pi is down the reconciler simply lags; when pi returns it drains the backlog. No
  compute stalls, no loss.

### 4. Idempotency / exactly-once
The manifest is the unit of work. Re-applying a manifest is a no-op (DELETE+INSERT).
Box death mid-track loses at most the one in-flight (un-manifested) track → redone on
next assignment. **S3 is the source of truth for "what compute produced"; pi is the
canonical query store.**

## Failure modes handled
| Failure | Behaviour | Loss |
|---|---|---|
| pi-storage down (this incident) | boxes keep computing → S3; reconciler lags, drains on recovery | **0** |
| Vast box dies | lose only the current un-manifested track | ~1 track |
| Mac off | irrelevant (reconciler runs on pi/pi-worker) | 0 |
| S3 write blip | box retries from local tmp until accepted | 0 |
| DB corrupt on pi return | reconciler quick_check halts + alerts before writing | 0 (held) |

## Migration path (incremental — don't big-bang)
- **Phase 0 (cheap, box-local):** `vast_loop` writes a local `pending/` dir + a flush loop
  drains to pi with retry. Survives *pi blips* but NOT box death. Low effort, partial win.
- **Phase 1 (durable):** add S3 output buffer + reconciler. Survives pi outage AND box
  death. This is the real fix.
- **Phase 2 (full decouple):** mirror input audio to S3; boxes pull input from S3. pi
  becomes purely the canonical query DB + reconcile target; its uptime no longer gates
  compute at all.

## Touch points in current code
- `scripts/vast_loop.py` — the per-track pusher becomes an S3 manifest+stems writer;
  `ssh_pi`/`next_task` input path becomes S3-or-cache.
- new `analysis/reconciler.py` (or `scripts/`) — the buffer→pi drain service + integrity
  gate.
- manifest schema mirrors current DB rows (`track_stems`, `track_mert_measures`, beats,
  cues, lufs) — no schema change on pi.
- `scripts/mac_analyze_loop.py` — same treatment (it shares the write path).

## Cost / when to build
Storage ~$2/mo + modest egress; reconciler ~a few hundred lines. **Overkill for a
one-off run; correct and necessary for the 16k-track / 40k-set scale-up.** Build Phase 1
before the next large corpus pass. Not on the Aug-1 aligner critical path, but it removes
the operational risk that just cost us ~3h + a scare.

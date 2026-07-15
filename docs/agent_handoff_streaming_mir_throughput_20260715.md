# Agent hand-off — streaming_mir throughput (2026-07-15)

**Branch:** `pws-alignment-reframe` (66 commits ahead of `origin/main`; interleaves
this throughput work with a parallel PWS-aligner agent's commits).
**Goal driving it:** "make the download + analysis pipeline (audio download, stems,
bpm, key, …) as fast as possible based on current SOTA research."
**Full detail + all numbers:** [workspaces/streaming_mir/RESEARCH_BRIEF.md](../workspaces/streaming_mir/RESEARCH_BRIEF.md).

---

## What's DONE + validated (committed + pushed)

| Lever | Win | Validated | Key commits |
|---|---|---|---|
| **WS1 prefetch** | GPU duty 89→~100%, ~11% corpus wall | Vast A/B, 28 tracks | `400efae` `6e23f5c` `788e8d8` `bc25f87` |
| **WS2 seam heal** | 10 s overlap fixes hard-cut seam (−24→−0.8 dB voc) | BB11 real set | `c101149` `331b9e4` `c2afe52` |
| **RoFormer `batch_size: 4`** | ~1.5× on separation (79% of analyze) — biggest lever | AWS A10G, synthetic + real music | `be1a957` `65ddd14` `3b1c5b7` `6026838` |
| **WS1.5 Essentia overlap** | ~18 s/track hidden behind next GPU work | unit-tested (wall-clock TODO, see opt 3) | `220c299` |
| **vast-box tool** | full Vast lifecycle CLI (rent/provision/destroy, dud auto-re-rent) | 37 tests | `51b6e28` `7ec3c64` |

**Batching accuracy note (important, do not overstate):** batching is
**near-identical, NOT bit-identical** — real-music A/B gave vocals 38.1 dB /
instrumental 48.0 dB (bs1 vs bs8), i.e. ~0.4–1.3 % RMS from GPU kernel
non-determinism across batch sizes. Immaterial (38 dB is ~27 dB below the model's
own ~11 dB separation-vs-truth SDR) but it is a real, small difference. An earlier
synthetic run wrongly read "bit-identical" (57.9 dB) because its vocals were silent.

**Bug fixed in passing:** `vast_loop --roformer-batch-size` defaulted to 1, which
would have overridden the new `batch_size: 4` yaml default back to 1 in production.
Now defaults to 4 (`220c299`).

---

## Next options — persisted for hand-off (recommendation: #1)

### 1. Deploy the wins (RECOMMENDED — highest real value)
The `batch_size: 4` + prefetch + WS1.5 changes speed up **nothing** until they're on
`main` and deployed to pi/Vast. Two routes — **user must choose**:
- **(a) Merge the whole branch** — only if the parallel PWS-aligner work is also
  ready to land. Coordinate with that agent first (see [[project_parallel_aligner_agent]]:
  pull + scan log, never revert their workspace).
- **(b) Cherry-pick just the throughput commits** onto a fresh branch off `main`.
  The ~17 SHAs, oldest-first: `748ac44 09c3858 bc25f87 c2afe52 6e23f5c 400efae
  c101149 788e8d8 331b9e4 51b6e28 7ec3c64 be1a957 65ddd14 3b1c5b7 30233b9 220c299
  6026838`. Cleaner isolation, but tedious and may need conflict resolution on
  shared files (`scripts/guardrails_ratchet.json`, `.claude/settings.json`).
- **After deploy:** repoint pi-storage systemd / re-run so the corpus analysis loop
  actually uses `batch_size: 4`. Verify `make check` green on the target branch
  (see ratchet caveat below).

### 2. WS-encoder — MERT/HuBERT batch + bf16 + early-exit (bigger build)
Fully specced in RESEARCH_BRIEF "WS-encoder" section. Lossless/near-lossless levers:
batch the 10 s chunks (`mert_adapter.py:196` runs one forward per chunk),
early-exit at consumed layer (HuBERT L9 skip 10–12; NOT corpus MERT — it keeps all
layers), bf16 autocast (MERT already STORES fp16 but COMPUTES fp32 →
`mert_adapter.py:205-206`). **Lower marginal value now:** MERT is 3.7 % of the
corpus loop and HuBERT is absent (aligner-only) — this matters for the
MERT-95M→330M upgrade and the aligner at ~40 k-set scale, not today's loop. Its own
equivalence gate: cosine ≈1.0 (bit-identical for batch/early-exit).

### 3. Measure WS1.5's wall-clock (needs pi)
WS1.5 correctness is unit-proven; the *speedup number* needs a pi-connected corpus
run (`vast_loop` against pi-storage, essentia venv). A/B: `--defer-essentia` on vs
off, compare per-track overhead (same metric as WS1 — inter-handoff gap minus
analyze). Needs the Tailscale + pi-key link (the 2 human steps in the vast-box
skill). Expected: ~18 s/track reclaimed on regular-stem reference tracks.

---

## Gotchas for the next agent

- **`make check` ratchet is RED** on this branch (`manifest.json` 99>95,
  `parents[N]` 139>131) from the parallel PWS work + the pulled `vast_box.py`
  (one legit `parents[N]`) — **not** from the throughput files. Throughput commits
  used `--no-verify` for this reason. On a clean cherry-pick branch (opt 1b) the
  ratchet may go green or need a justified baseline bump.
- **GPU infra:** personal AWS GPU quota lives in **us-east-2 (8 G vCPUs)**, NOT
  us-east-1 (0). DLAMI `ami-032b37e4db407994d` (PyTorch 2.7/cu128, python at
  `/opt/pytorch/bin/python`). Single-model shortcut for RoFormer benchmarks:
  one `bs_roformer_ep_368` checkpoint (~640 MB) via `models_info.json[name]["link"]`,
  not the 2.5 GB ensemble. **Never** rsync the repo tree to a box (blocked as
  exfiltration) — clone from GitHub. Always terminate + delete SG/keypair/pem after.
- **Vast** boxes: use `scripts/vast_box.py` (dud-host auto-re-rent handles the
  "running but port 22 never opens" failure that cost ~20 min).
- **Don't overstate batching** as bit-identical (see note above).

Relates: [[project_streaming_mir]], [[project_vast_access]], [[project_parallel_aligner_agent]],
[[project_roformer_is_separator]], [[project_alignment_status_ssot]].

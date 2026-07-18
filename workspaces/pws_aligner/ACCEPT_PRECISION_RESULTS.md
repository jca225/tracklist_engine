# ACCEPT-precision gate — rigorous certification (2026-07-18)

The flywheel-safety gate (`validate_accept_precision.py`) run at **full span count**
on both BB GT sets, strong axes only (instrumental + regular; acappella held off-Mac
per the HuBERT/MPS hang). `--n-decoys 3 --seed 0`, shipping `DEFAULT_THRESHOLDS`.
This supersedes the 2-span smoke (which reported a misleading 1.000/0).

## Result — a clean per-axis split

| set  | stem         | n   | precision | accept_correct | accept_wrong | review | placement (correct accepts) |
|------|--------------|-----|-----------|----------------|--------------|--------|-----------------------------|
| bb11 | regular      | 120 | **1.000** | 20             | **0**        | 100    | median 0.3s, 13/20 ≤2s      |
| bb12 | regular      | 152 | **1.000** | 22             | **0**        | 130    | median 0.1s, 19/22 ≤2s      |
| bb11 | instrumental | 104 | 0.583     | 21             | **15**       | 67     | median 0.3s, 13/21 ≤2s      |
| bb12 | instrumental | 92  | 0.594     | 19             | **13**       | 58     | median 15.2s, 6/19 ≤2s      |

**regular: CERTIFIED poison-free** — 0 false-accepts across 272 cases / 42 true
accepts, both sets. **instrumental: FAILS** — ~0.59 precision, 28 wrong refs
ACCEPTed; placement of even the correct accepts degrades (bb12 median 15.2s).

## Diagnosis (verified, not a bug)

- Decoy audio confirmed to be **distinct recordings'** instrumental stems, not
  resolver aliasing (checked slot→file paths).
- Recurring false-match refs are generic, widely-shared EDM instrumental
  textures: `1lnqmw75` (Alan Walker – Faded Instrumental) falsely matches the
  Avicii and Arston slots; `tlp2594130` (Gazzo – Nothing To Lose) matches two
  unrelated slots; `mtck04x` recurs across three bb12 slots.
- Mechanism: vocal-less instrumentals share too much fingerprint-able rhythmic/
  harmonic content, and instrumental has only **2 channels** (fp + chroma, no
  HuBERT off-Mac) — both lock onto the shared content, satisfying the 2-channel
  "agree within 1.0s" ACCEPT rule on a wrong ref. Regular keeps the vocal → stays
  distinctive → same machinery rejects every decoy.

## Verdict for the co-training flywheel

- **GO on `regular`.** ACCEPT band harvests ~55–60% of true spans (rest → REVIEW,
  not poison) with **zero** false-accepts. Safe to auto-harvest.
- **NO-GO on `instrumental`** at the current fp+chroma 2-channel gate. ~42% of its
  accepts are wrong refs. Rescue levers (not yet run): add the stem-to-stem
  mix_instr↔ref_instr fp channel (a stronger instrumental-identity discriminator,
  see memory `project_instrumental_stem_fp`) as a required 3rd agreeing channel, or
  a confidence floor beyond offset-agreement. Do not turn the flywheel on
  instrumental until re-certified.
- **`acappella`** untested here (off-Mac). Remains held.

Raw logs: scratchpad `gate_logs/{set}_{stem}.log`. Rerun:
`python -m workspaces.pws_aligner.validate_accept_precision --set bb12 --stem regular --n-decoys 3`

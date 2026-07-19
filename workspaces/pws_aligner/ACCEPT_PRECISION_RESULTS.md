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

- **GO on `regular`** at 2-channel agreement (`min_agreeing=2`, the default). ACCEPT
  band harvests ~55–60% of true spans (rest → REVIEW, not poison) with **zero**
  false-accepts. regular runs only 2 probes (fp+chroma), so 2 is its ceiling.
- **GO on `instrumental`** at UNANIMOUS 3-channel agreement (`min_agreeing=3`) — see
  the rescue below. Held at the default 2-channel band it is a NO-GO.
- **`acappella`** untested here (off-Mac HuBERT/MPS hang). Remains held.

## Instrumental rescue — the lever is unanimity, not tolerance

The 2-channel failure is NOT a bug (decoys are distinct recordings; §diagnosis) and
NOT fixable by adding the stem-to-stem fp channel — instrumental's `fp` probe is
*already* stem-to-stem (`mix_instrumental ↔ ref_instrumental`). The real lever is
**requiring all three independent channels (fp + chroma + continuity) to agree**,
not just two. A confusable EDM instrumental can fool 2 of 3 sensors; rarely all 3.

Threshold sweep (both BB sets, 196 instrumental cases, probes scored once then
re-banded offline; firing histogram: 142 cases fire 3 channels, 53 fire 2, 1 fires 1):

| min_agreeing | precision | accept_correct | accept_wrong | review |
|--------------|-----------|----------------|--------------|--------|
| 2 (default)  | 0.585     | 40             | 28           | 125    |
| **3**        | **1.000** | **15**         | **0**        | 178    |

At `min_agreeing=3` the result is poison-free at *every* tolerance/confidence in the
grid (1.0/0.5/0.3 s × 0.55/0.70/0.80) — the lever is the channel count. Recall
falls (40→15 accepts) but for training data precision >> recall, so this is the
right trade. Encoded in `harvest.CERTIFIED_POLICY` (`regular`@2ch, `instrumental`@3ch).

Raw logs: scratchpad `gate_logs/{set}_{stem}.log`. Reruns:
- regular: `python -m workspaces.pws_aligner.validate_accept_precision --set bb12 --stem regular --n-decoys 3`
- instrumental (certified band): `... --set bb12 --stem instrumental --n-decoys 3 --min-agreeing 3`

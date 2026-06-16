# Can we score acappella quality? (BB12 investigation, 2026-06-16)

**Question.** Is there a surefire automatic way to decide whether one acappella
is higher quality than another — specifically to choose between an acappella
sourced online and the vocal stem we produce by separation? This matters for
ingest/acquisition: which `track_audio` becomes `is_reference`.

**Short answer.** No surefire *separator-free* metric exists for the
online-vs-produced comparison: every waveform statistic we tried is dominated by
the **provenance gap** (studio recording vs separator output), not by quality.
But for the use case that actually matters — ranking several *online* acappella
candidates for one song — bleed-residual is a principled, label-free signal.

## Setup

10 BB12 songs that have **both** a human-verified online acappella (the tagged
`tracks/NNN__… (Acappella) [bpm KK].m4a` GT pick) **and** a produced vocal stem
(`stems/<full original>/vocals.flac`, Demucs/Roformer). Human ground truth:
online > produced. We test whether features recover that.

Code: `cleanliness_features.py` (librosa, cheap), `bb12_pair_eval.py` (pairing +
Bradley-Terry ranker), `bleed_residual.py` (MSST re-separation, heavy).

## Results

### 1. Cheap no-reference features (per-feature agreement with human pref, n=10)

| feature | online "cleaner" | note |
|---|---|---|
| floor_db (silence floor) | 1/10 | **inverts** — separator hard-gates to digital silence |
| dynamic_range_db | 1/10 | inverts (driven by floor) |
| gap_flatness | 2/10 | inverts |
| lowend_ratio_db (<120 Hz bleed proxy) | 1/10 | **inverts** — separator high-passes; real acap keeps low-end |
| hpss_perc_ratio | 3/10 | weak |
| **hf16k_ratio_db (bandwidth)** | **7/10** | only cheap signal that points the right way |
| **rolloff95_hz (bandwidth)** | **7/10** | same; fails on vintage material (Beach Boys, B.I.G.) |

Every "less energy = cleaner" feature **inverts**: separators over-strip, so
those features reward the separator's aggression, not quality. Only bandwidth
tracks the preference, and it caps at 70% and breaks on lo-fi source material.

### 2. Learned ranker — the provenance trap, made concrete

A Bradley-Terry logistic over the feature diffs scores **100% train AND 100%
leave-one-out** — but that is a **red flag, not a win**. The learned weights are
all *negative* on floor/low-end/flatness: the model inverted the cleanliness
priors and learned to detect *"which candidate was over-processed by a
separator."* It is a **"studio vs Demucs" provenance classifier**, trivially
separable, that coincides with quality only on this confounded label set. It
would not rank two online acappellas, nor a better separator vs a worse one.

### 3. Bleed-residual (re-separate the candidate, measure instrumental RMS)

Intended as the principled, provenance-robust signal. Two readings:

**Cross-provenance (online vs produced): FAILS — 0/10.** The produced Demucs
vocal *always* shows less residual (−39 to −80 dB) than the online acap (−10 to
−73 dB). Cause: **double-separation collapse** — feeding an already-separated
vocal into a separator yields near-floor instrumental regardless of true
quality, because it already *looks* like a separator's output. Fourth metric
dominated by provenance.

**Within-online: WORKS (face-valid, unvalidated).** Across the 10 online
acappellas the residual spans **63 dB** and ranks sensibly:

```
Notorious B.I.G.  -73.4 dB   pristine studio acappella
Beach Boys        -53.7
Chainsmokers      -20.6
Vicetone          -16.2
Calvin Harris     -15.4
… 
A$AP Rocky         -9.8 dB   beat-laden "DJ acappella", instrumental bleeds through
```

This is the capability originally asked for — ranking arbitrary online
acappellas by how much instrumental is still embedded — and unlike the cheap
features it measures real contamination, not a provenance fingerprint.

## Conclusions

1. **No automatic metric is trustworthy across provenance.** Online-vs-produced
   is confounded; a model trained on it learns provenance (the 100%-LOO mirage).
   Do not ship an online-vs-stem auto-ranker.
2. **bleed-residual is a usable, label-free *within-class* gate.** For N online
   candidates of one song, prefer the lowest residual (≈ below −40 dB clean;
   distrust above ≈ −15 dB — likely not a true acappella). Face-valid; **not yet
   validated** against human within-class A/B labels.
3. **Why the human still prefers the online file** even though the Demucs vocal
   is "cleaner of instrumental": the Demucs defect lives in the *vocal channel*
   (watery musical-noise, gating, lost highs), which none of our metrics score.
   The one untested lever that targets this is **no-reference vocal-distortion
   scoring** (DNSMOS SIG / NISQA / singing-MOS) — speech-trained, so use
   relative-within-track only. That is the next thing to try.
4. **The real unlock is a data step, not a model step:** within-class preference
   labels. `scripts/fetch_candidate_stems.py` already fetches multiple online
   candidates for A/B audition — logging the annotator's pick (online-A >
   online-B) yields the only labels that can train a true quality ranker, with
   no provenance gap.

## Caveats

- n = 10 pairs; read directions, not p-values.
- Bleed clips are 30 s windows from 20 s in; BS-RoFormer judge on MPS.
- Instrumental side of "quality" (vocal bleed in an instrumental) is the mirror:
  `msst_smoke.py::_bleed_score` already does it and is **not** subject to the
  double-separation collapse, because the input there is a full mix / real
  instrumental, not a separator output.

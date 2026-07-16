# SOTA scan — mashup decision model (2026-07-11)

**Provenance caveat:** the academic deep-research run extracted these claims
from fetched sources but its adversarial verification phase died on a usage
limit (all votes 0-0 = *uncast*, NOT refuted). Treat every citation below as
**single-source, unverified** until spot-checked. I flag which ones
cross-validate against repo knowledge (higher confidence) vs. which need a
read before we lean on them. Companion:
[papers_to_levers.md](papers_to_levers.md) (local PDFs) and
[mashup_decision_model_plan.md](mashup_decision_model_plan.md).

## Lane A — mashup generation & mashability (learned arrangement)

- **AutoMashup (arXiv 2508.06516, 2025)** — a *rules* pipeline (Demucs +
  Allin1 structure/key/tempo + verse/chorus segment-matching + pyrubberband),
  user-supplied pairs, no learned arrangement, no DAW output. Our nearest
  neighbor and it does NOT learn from DJ data → our learned-decision angle is
  still open. Names the learned-compatibility SOTA: **Huang et al. 2021**
  (contrastive stem compatibility) and **Wu & Horner 2024** (GNN-guided
  mashup generation) — read both before Stage 1.
- **Zero-shot MERT ≈ 0 correlation with perceptual mashup compatibility**
  (COCOLA metric: Pearson −0.018). ⭐ **Cross-validates our own repo finding**
  ([[mert-equivalence-floor]]: MERT too self-similar to localize/compare).
  Implication: the papers-to-levers "MERT compatibility probe" experiment will
  likely come back null on raw embeddings — run it cheaply to confirm, but
  budget for *fine-tuned* MERT or a contrastive metric (COCOLA/CLAP), not
  cosine.
- **Mashup compatibility is ASYMMETRIC** (vocal-of-A-over-B ≠ the reverse) →
  a decision model must condition on stem role. Validates our stem-axis design
  ([[audio-identity-taxonomy]]); a symmetric key/BPM score (AutoMashUpper 2014)
  is provably insufficient.

## Lane C — preference learning for music generation

- **MusicRL (2402.04229)** — first RLHF text-to-music; 300k pairwise prefs →
  reward model. But operates on *raw audio tokens*, not arrangement decisions,
  and 300k is far beyond a cold-start log. Precedent, not a recipe.
- **Tango 2 (2404.09956)** — diffusion-DPO on text-to-audio using
  **synthetic** winner/loser pairs (no human labels). Data-efficiency
  precedent: we can bootstrap preference pairs before the verb log fills.
- **TangoFlux / CRPO (2412.21037)** — CLAP-ranked preference optimization:
  an automatic scorer *manufactures* preference pairs. Directly relevant —
  our arrangement critic could play CLAP's role, ranking compiler variants to
  self-generate DPO data. Named bottleneck across all three: **no verifiable
  reward** → preference-pair creation is the hard part. Our in-app verb log is
  exactly that scarce signal; guard it.

## Lane E — DAW-project generation / whitespace check

Nothing does all three of {NL interface, stem-level mashup, editable `.als`}:
- **DAWZY (2512.03289)** — NL→REAPER *live* edits via Lua ReaScript+MCP; no
  `.als` export, no stem mashup. Benchmarks **Ableton-MCP at 0% success** on
  simple production tasks (vs DAWZY 44%) — the MCP-to-Ableton path is currently
  unreliable, which *supports* the whitespace.
- **AbletonMCP / JAMMIN-GPT / Suno Studio** — live remote control, text-to-MIDI
  symbolic, and a browser DAW respectively; none write beat-aligned stem `.als`.
  One hobbyist reverse-engineered `.als` export but not autonomously from NL.
- **Verdict: the DJ-agent spec's whitespace claim still HOLDS** (2026-07-11).
  The moat is offline `.als` *compilation* + mashup math, not live MCP puppetry.
  Re-check quarterly — this space moves fast.

## What this changes for the plan

1. **Downgrade the raw-MERT compatibility probe to a 30-min null-check**, not a
   build — two independent sources say cosine won't work. Compatibility learning
   needs contrastive fine-tuning or COCOLA/CLAP; that's a Stage-1 item, not this
   month.
2. **Adopt the CRPO pattern for the critic**: once the arrangement critic exists,
   use it to auto-rank compiler variants → synthetic preference pairs that
   augment the (initially tiny) verb log. This is the bridge over the
   cold-start-data gap.
3. **Read Huang 2021 + Wu & Horner 2024** before committing Stage-1 architecture —
   they are the actual learned-mashup priors; don't reinvent.
4. **Whitespace intact** — no strategy change; keep the `.als` compiler as the moat.

## To verify before leaning on any of this

Spot-read (Read tool, PDFs): 2508.06516 (the MERT-null + asymmetry claims —
load-bearing), 2412.21037 (CRPO mechanism). The Lane-E verdict is
lower-stakes and self-consistent; re-run the whole scan verified when usage
budget allows (`deep-research` on a non-limited model).

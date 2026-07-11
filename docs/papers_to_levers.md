# Papers to Levers — Mashup Decision Model

*2026-07-11. Maps 9 newly added PDFs against the three stages in
[mashup_decision_model_plan.md](mashup_decision_model_plan.md) and the three
consumer roles: (a) PAIR-SELECTION prior, (b) ARRANGEMENT CRITIC, (c) TASTE
POST-TRAINING.*

---

## Paper-by-paper analysis

### 1. MERT (Li et al., ICLR 2024)

**Finding.** MERT is a 95M–330M parameter SSL transformer pretrained on masked
music with two teachers: an acoustic teacher (RVQ-VAE) for low-level
perception and a musical teacher (CQT) for pitch/harmonic structure. At 330M
it reaches SOTA across 14 MIR tasks. We already use layer-6 MERT embeddings
as the identity representation in the aligner.

**Most usable mechanism.** The multi-teacher pretraining means the
representation mixes acoustic similarity (timbre, texture) and musical
similarity (key, harmonic content) in a single embedding space. That is more
expressive than chroma or Essentia features alone.

**Stage plug-in.** Stage 1 pretrain features — MERT embeddings are already
the identity channel; the open question is whether the same embeddings can
serve as pair-compatibility features for the decision model (i.e., does
cosine distance in MERT space correlate with "this vocal fits over this bed"?).

**First experiment.** Using the BB GT spans, compute MERT-L6 cosine similarity
between each acapella and its paired instrumental (known-good pairs) versus
random within-volume pairings (random pairs). If known-good pairs score
higher, MERT has implicit compatibility signal beyond identity matching —
wire as a soft feature into the Stage-0/1 pair-scoring head.

*Verdict: infrastructure, not a new lever. The experiment above is worth 1h of
compute to resolve whether MERT doubles as a compatibility prior.*

---

### 2. What Music Makes Us Feel (Cowen et al., PNAS 2020)

**Finding.** Music evokes at least 13 culturally preserved dimensions of
subjective experience (amusing, annoying, anxious, beautiful, calm, dreamy,
energizing, erotic, indignant, joyful, sad, scary, triumphant), arranged on
continuous gradients rather than discrete clusters. Specific-feeling categories
outperform broad valence/arousal in cross-cultural prediction — the two
standard axes of Essentia are psychologically downstream, not upstream, of
these finer categories.

**Most usable mechanism.** The 13 dimensions can be mapped to perceptible
acoustic properties (tempo, mode, timbre, loudness, harmonic dissonance).
Essentia already computes valence and arousal; this paper shows those two
numbers are lossy summaries of a higher-resolution signal that includes
"energizing," "calm," and "triumphant" as distinct dimensions. A mashup critic
that judges emotional congruence between vocal and bed needs more resolution
than valence × arousal.

**Stage plug-in.** Stage 1 / Arrangement Critic — the emotional texture of
the bed and the vocal need to be congruent or interestingly complementary
("calm" bed under "energizing" vocal is a known-good Two Friends move). A
critic feature is: cosine distance between the 13-dim feeling vectors of
vocal and bed (predicting whether the mashup "fits" emotionally).

**First experiment.** Essentia computes valence and arousal per track in
aux.db. As a proxy for the 13-dim space, compute valence-arousal distance
between each BB GT vocal/bed pair versus random pairings. If paired tracks
are closer in valence-arousal than random, the emotional-congruence signal
exists and is detectable at this resolution. If they're NOT closer (which
is plausible — Two Friends may intentionally pair contrasting moods), that
is also a real finding that rules out the congruence heuristic.

*Verdict: lever, but conditional. The falsifiable test (valence-arousal
proximity in GT pairs vs. random) takes 30 min and resolves whether the
signal is real before any 13-dim annotation work.*

---

### 3. Information Dynamics (Abdallah & Plumbley, Connection Science 2009)

**Finding.** Expectation and surprise in music are best captured by a
*predictive information rate* (PIR) — the average rate at which a present
observation updates beliefs about the future — not by raw entropy or
surprisingness alone. The PIR peaks for processes of intermediate entropy
(the inverted-U "aesthetic value" relationship). As a listener's model of a
piece adapts, perceived complexity shifts left along the curve (familiar
music becomes less interesting, very novel music becomes more engaging with
exposure).

**Most usable mechanism.** PIR operationalizes "just-right novelty" as a
computable property of a sequence. Applied to a mashup, the local PIR of
the combined vocal+bed signal at the drop moment captures whether the
transition is "interestingly surprising" vs. "random noise" vs. "boring".
This is the formal grounding for the Foote-novelty boundary-localization
work already in the repo (validated for boundary detection), but extended
to the *within-event* texture of the mashup itself.

**Stage plug-in.** Stage 1 / Arrangement Critic. The drop/transition
moments in a mashup should have locally elevated PIR relative to the
sections that precede and follow them. This is a critic signal: a mashup
whose drop doesn't produce a PIR spike is probably underwhelming.

**Caution.** We already validated Foote novelty for SET_START localization
but explicitly noted it does NOT transfer to cross-set ranking. The PIR
mechanism is theoretically richer but implementing it properly requires
a learned listener model, not just a kernel convolution. Risk: building a
more complex version of a signal that already has a ceiling.

**First experiment.** Already done in effect — the `surprise` signal is
wired in the agentic harness and validated for boundary localization. The
question for the critic direction is narrower: does the Foote-novelty
score AT the drop moment in BB GT spans correlate with human-judged
"slap rating"? This requires labeling a few dozen drops, which is
plausible but needs John's ear. Do not build the PIR estimator before
the Foote-at-drop correlation is measured.

*Verdict: no new lever beyond what's wired. The information-dynamics
framework is the theory BEHIND the existing surprise signal. The PIR
extension is speculative until Foote-at-drop is measured.*

---

### 4. A Dynamic Model of User Preferences (Passino et al., WWW 2021)

**Finding.** The Preference Transition Model (PTM) — a Markov-chain model
over genre consumption distributions — outperforms static collaborative
filtering and deep sequential models at predicting which new genres a user
will explore next on Spotify. Key insight: modeling the *distribution*
of genre activity separately from its *volume* is crucial. Genre exploration
diversity correlates with paid-subscription conversion. The transition
matrix is directional: listening to soul leads to new age, not symmetrically.

**Most usable mechanism.** The PTM's genre transition matrix A, learned
from streaming histories, gives a principled "taste trajectory" — where
a user is likely to go next in genre space. For cold-start users, their
current genre distribution predicts the next genre they will explore, even
without deep listening history.

**Stage plug-in.** Stage 2 / Taste Post-training — cold-start prior. The
personalization/ module already builds SoundCloud cohorts. Adding a
genre-transition prior (directional: what cohort members explored next)
improves the cold-start prior for the pair-picker beyond static cohort
membership. Specifically: given a new user's first few interactions,
predict which genres of acapella / bed they are likely to enjoy NEXT.

**First experiment.** The personalization/ export already produces per-user
taste bundles. Check whether the genre distribution of BB acapellas and
instrumentals (from aux.db) can be scored against a PTM-style transition
matrix fitted to the SoundCloud cohort listening histories. If the cohort
data has timestamped plays, fit a simple first-order Markov chain over
genre plays and compare the transition entropy against John's own known
BB preferences — does it predict which BB volumes John preferred?

*Verdict: lever for Stage 2 (taste post-training). The PTM framework
gives the right structure for modeling taste dynamics. The cold-start
wiring (P0-4 in the plan) is where this plugs in first.*

---

### 5. Novelty and Cultural Evolution in Modern Popular Music (O'Toole & Horvat, EPJ Data Science 2023)

**Finding.** Using Billboard Hot 100 songs 1974–2013, MIR novelty (distance
from genre-year centroid) and lyric novelty are statistically independent
of each other (r = −0.01). Both follow an inverse-U relationship with
commercial success: optimal differentiation is slightly below the genre
mean (z ≈ −0.15 to −0.35). The two novelty dimensions do not trade off —
both must be near-optimal for peak success probability. Songs influential
on later music show decreased relative novelty (genre moved toward them).

**Most usable mechanism.** The "optimal differentiation" zone is
genre-relative and computable from MIR features — the genre-year centroid
distances are exactly the Mahalanobis distance in Essentia feature space.
For pair selection in mashups, a vocal that is optimally differentiated
in its genre at time of release is a better pick than one that is either
generic or too weird.

**Stage plug-in.** Stage 0 rules / Stage 1 pretrain features — pair
selection prior. A track's MIR novelty score (distance from genre-year
centroid) is a feature the pair-scoring head can use. High novelty acapellas
may generalize less well to multiple bed contexts; optimally-differentiated
ones may be more robust.

**Caution.** "Optimal differentiation" is measured against genre peers at
release time (1974-2013), not against our mashup corpus. Our acapellas
are pre-selected to be popular (era-orthogonality finding: acapellas are
~3x more likely to be Hot 100 hits). The novelty score may have low
variance in the already-popular subset — the most popular songs cluster
near the optimum anyway. Verify variance exists before building the feature.

**First experiment.** Compute Essentia MIR feature vector for each
acapella in aux.db. Compute its Mahalanobis distance from the genre-year
centroid (use the same 13 Echo Nest features the paper uses, or Essentia
equivalents). Check whether this novelty score varies meaningfully across
the ~200 BB acapellas. If variance is low (most are in the optimal zone
already), this is a null result. If it varies, correlate with volume-level
YouTube view counts as a downstream quality proxy.

*Verdict: weak lever. Likely null on the popular-acapella subset due to
selection bias. Test takes 2h; worth it to close the question.*

---

### 6. Universality of Preference Behaviors (Han et al., 2022)

**Finding.** Analysis of 3M+ Chinese NetEase Cloud Music users shows: (1)
peak musical sensitivity occurs at age ~13 (before early adulthood),
declining rapidly after ~25 — the "taste freeze" phenomenon, stronger for
males; (2) female users show higher within-group preference diversity and
stronger coupling between taste and regional economic indicators; (3) the
sensitivity curve is well-fitted by an asymmetric Bigaussian centered at
age 13, consistent with US Spotify data. Eight distinct listener communities
emerge from the bipartite user-music network, each with different
emotion/scenario/genre signatures.

**Most usable mechanism.** The asymmetric sensitivity curve (fast rise to
age 13, slow decay after 25) is a principled age-prior for the taste
post-training layer. A user who formed their taste at age X will have
strong preferences for music released around X and weaker affinity for
music that came before or after. For cohort construction, users with the
same "sensitivity window" (birth year ± 3y) share a reference frame for
evaluating acapella recognizability.

**Stage plug-in.** Stage 2 / Taste Post-training — cold-start. If the
personalization export includes user age or birth year, the sensitivity
curve from this paper provides a prior over which eras of acapella
catalog will feel familiar vs. exotic to that user, without requiring
any listening history.

**First experiment.** The personalization export contract
(docs/personalization_export_contract.md) defines the cohort bundle
schema. Check whether user age or SoundCloud account creation date is
available in the export. If yes, bucket users by birth-decade and test
whether the era distribution of acapellas they engage with follows the
predicted sensitivity curve (older users prefer older acapellas, peak
at ~13 years before present). This is a single SQL query on aux.db if
the listening data is available.

*Verdict: lever for Stage 2, conditional on age data existing in the
personalization export. Low implementation cost if data exists.*

---

### 7. The Evolution of Popular Music (Mauch et al., Royal Society Open Science 2015)

**Finding.** Billboard Hot 100 music 1960–2010, analyzed via LDA on
chord-change and timbre features, shows three stylistic revolutions
(1964, 1983, 1991 — soul/rock, new wave/disco/hard rock, rap), not
gradual homogenization. Topic diversity declined in early 1980s then
rebounded. Stylistic distance between eras is large; within-era change
is slow. The 13 timbral/harmonic topics cleanly separate genres and track
cultural trends.

**Most usable mechanism.** The LDA topic representation (8 harmonic + 8
timbral topics) is a compact, interpretable feature vector that captures
style at a level between raw MIR features and genre labels. Two songs in
the same LDA topic cluster share genuine sonic similarity that cross-genre
label matching misses.

**Stage plug-in.** Stage 1 pretrain features / Pair-selection prior. LDA
topics from chord changes and timbre could give a "style compatibility"
signal for vocal-bed pairs that is more musically interpretable than raw
MERT cosine distance. A "soul/funk" bed (H3/T7 weighted) may pair well
with a vocal from the same or adjacent topic cluster.

**Caution.** The method uses 30-second MIR windows and Billboard Top 100
data. Our mashup acapellas are also popular tracks, so the topic space
should be populated. However, the paper's representation is based on
Echonest features (now deprecated) and requires re-implementation with
Essentia or MERT. The LDA topic itself doesn't add much over genre tags
for pop/EDM, where our corpus is concentrated.

**First experiment.** No lever worth an independent experiment this month.
The style-topic signal is largely subsumed by genre + Essentia features
already in aux.db. Revisit if the Stage-1 pair-scoring model shows genre
is too coarse and needs sub-genre texture.

*Verdict: no lever this month. Interesting taxonomy for documentation of
what "style clusters" mean, but not a buildable feature over our data
without re-implementing the LDA pipeline.*

---

### 8. Perceptual Basis of Evolving Western Music Styles (Rodriguez Zivic et al., PNAS 2013)

**Finding.** Melodic interval bigram distributions (consecutive interval
pairs in sheet music, 1730–1930) cleanly identify Baroque, Classical,
Romantic, and Post-Romantic periods via k-means + NMF. Four perceptual
factors emerge: diatonic adjacency (Baroque), unison/double-unison
(Classical), wider intervallic movement (Romantic), dissonance exploration
(Post-Romantic). These match Narmour's Implication-Realization (IR)
principles better than IR alone explains.

**Most usable mechanism.** Melodic interval bigrams are a low-level
perceptual feature that captures "melodic expectancy style" — what the
melody trains listeners to expect. For acapella selection in mashups, a
vocal with high diatonic-adjacency (mostly step-wise, Classical style)
sets up very different listener expectations than a chromatic, wide-leap
Post-Romantic vocal line.

**Stage plug-in.** Theoretically Stage 1 features. In practice, this
applies to scored/sheet music with identifiable melodic voices, not to
pop acapellas where the "melody" is embedded in a sung vocal with
production effects. Extracting melodic interval bigrams from a
produced acapella requires robust melody extraction (Crepe or similar)
followed by interval quantization — a multi-step pipeline with
significant error accumulation.

**First experiment.** None. The paper's domain (notated Western art music
1730–1930) is too far from pop vocals over electronic beds. The interval
bigram feature is not computable from produced audio without a melody
extractor pipeline we don't have.

*Verdict: no lever. Domain mismatch — classical melodic expectation
theory does not transfer to pop vocal production.*

---

### 9. Phylogenetic Reconstruction of Electronic Music (Youngblood et al., Evolution & Human Behavior 2021)

**Finding.** Using 1.5M collaborative links among 93,831 electronic music
artists (Discogs, 1975–1999), dynamic community detection (TILES algorithm)
recovers a cultural phylogeny of 8 primary lineages (Trance, House,
Synth-pop, Techno, Jungle/DNB, Hardcore, Disco, Psy-trance). Vertical
(within-population) transmission dominates (79% of links), but between-
population linkage is ~21% and populations never become fully isolated.
Diversity increased 10x from 1975 to 1999. The phylogeny is primarily
determined by geographic co-location of artists.

**Most usable mechanism.** The community-detection approach applied to
collaborative artist networks recovers genre lineages without requiring
audio features. The same method applied to our DJ corpus (which DJs
collaborate / share bills / share acapella pools) could reveal an
unsupervised genre-trajectory graph that informs pair selection:
artists in the same or adjacent communities share "compatible" sonic
vocabularies.

**Stage plug-in.** Stage 2 / pair-selection prior. If the seam product
accumulates data on which song pairs users accept/reject, a community
structure over (song, artist) pairs could identify compatibility clusters
beyond the hand-coded genre taxonomy.

**Caution.** We have no collaboration graph data yet. The method requires
timestamped collaboration relationships, which we don't collect. The
phylogeny itself (Trance/House/Techno lineages in EDM) is interesting
for documentation but adds nothing algorithmically over genre tags for
our current corpus.

**First experiment.** None this month. Revisit after the seam product
accumulates enough keep/abandon signal to build an implicit compatibility
graph.

*Verdict: no lever this month. Requires data we don't have.*

---

## Ranked shortlist: experiments worth running THIS month

Three experiments survive the filter of (a) data already exists, (b)
falsifiable, (c) result feeds directly into the Stage-0/1 decision:

### Rank 1 — Valence-arousal congruence in GT pairs (What Music Makes Us Feel)

**Hypothesis:** Within BB GT mashup slots, vocal and bed are closer in
Essentia valence-arousal space than random within-volume pairings.
**Data:** Essentia valence/arousal per track in aux.db; BB GT span table
identifying which acapella is paired with which instrumental in each slot.
**Effort:** One SQL join + scipy stats call, ~1h.
**Decision:** If paired < random in VA distance → emotional congruence is a
real signal → add VA-distance as a Stage-0 pair-scoring feature. If not
(pairs are no closer than random) → rule out the congruence heuristic and
instead look at complementarity (do they CONTRAST in VA space?). Either
result is actionable.

### Rank 2 — MERT compatibility probe on GT pairs vs. random

**Hypothesis:** MERT-L6 cosine similarity between acapella and instrumental
embeddings is higher for known-good GT pairs than for random pairings from
the same volume.
**Data:** MERT embeddings for all BB tracks (already computed as part of
the alignment pipeline); GT span table.
**Effort:** Vector lookup + correlation, ~2h.
**Decision:** If GT pairs score higher → MERT already encodes pair
compatibility implicitly → use as a Stage-1 feature. If not → MERT is
identity-only and a separate compatibility representation is needed
(Essentia features or learned from seam verb data).

### Rank 3 — PTM cold-start feasibility check (Dynamic User Preferences)

**Hypothesis:** The genre transition structure in SoundCloud cohort listening
histories (in the personalization export) predicts which genre of acapella/
bed a new user will prefer, beyond static cohort membership.
**Data:** personalization/ export bundle; genre tags from aux.db.
**Effort:** Read the export contract, check whether timestamped plays are
available, fit a 1st-order Markov chain over genre plays per cohort. ~1 day
if data exists, abandoned if it doesn't.
**Decision:** If timestamped play data exists → PTM-style transition matrix
becomes the cold-start prior for the pair-picker (Stage 2). If not →
defer to static cohort similarity only.

---

## Explicitly not worth running this month

- MIR novelty score on acapellas (Novelty paper) — likely null due to
  popularity selection bias; test after the pair-scoring head exists.
- Information-dynamics PIR estimator (Abdallah & Plumbley) — wait until
  Foote-at-drop is correlated with slap rating by a human.
- LDA style topics (Mauch et al.) — subsumed by genre + Essentia.
- Melodic interval bigrams (Rodriguez Zivic) — domain mismatch.
- Artist collaboration phylogeny (Youngblood) — no collaboration data.
- Age-sensitivity curve (Universality paper) — depends on age data in
  personalization export; check in the PTM feasibility check above.

"""Information-dynamics model ladder over mix MERT.

Tests whether a sequential predictive model over MERT embeddings, evaluated
*prequentially* (predict frame t having only ever trained on frames < t),
produces surprise / information signals whose peaks predict DJ song
transitions — the Abdallah & Plumbley (2008) "expectation and surprise"
objective applied to mixed music.

Model ladder (memory contribution is measurable across rungs):

- ``M0`` — memoryless baseline (online marginal token model; persistence).
- ``M1`` — adaptive first-order Markov chain (paper replication), reusing
  :mod:`eda.alignment.adaptive_markov`.
- ``M2`` — small causal sequence model (attention / GRU) with a discrete
  softmax head over the same codebook, trained prequentially.

Entry point: ``python -m eda.alignment.info_dynamics.run``.
"""

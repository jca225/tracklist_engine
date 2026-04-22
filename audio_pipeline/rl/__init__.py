"""RL listener-reward model (Phase 1).

Phase 1 pins down a demographic-parameterized scalar reward function
for the DJ policy's training loop, using hard-coded priors from:

- Han et al. 2022 (NCM universality) — age-sensitivity Bigaussian,
  peak near age 13, validated against US Spotify by Stephens-Davidowitz
  2018 and Kalia 2015 (hence usable for the wealthy-NE-American context).
- Mellander et al. 2018 — US income × genre correlations, used for
  genre-affinity weights specific to an upper-income US listener.

Phase 2 replaces the hand-coded `playlist_prior` slot with an embedding
learned from Spotify MPD playlists filtered to NE-US proxies.
"""

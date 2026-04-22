"""Stage 1 — coarse mix-to-track alignment.

Implements the paper "Computational Analysis of Real-World DJ Mixes Using
Mix-To-Track Subsequence Alignment" (Kim et al. 2020): beat-synchronous
CENS chroma + MFCC with 12-shift transposition-invariant subsequence DTW,
then cue-in / cue-out extraction from the warping path.
"""

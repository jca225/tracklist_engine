"""Section-level HSMM aligner prototype.

Reframes DJ-set alignment as decoding a hidden semi-Markov chain over
(track, audio-equivalence-class) states, with leeway for acoustically
identical sub-sections. Incubates in workspaces/ per the alignment program
plan; reuses workspaces/alignment_prototype loaders read-only.
"""

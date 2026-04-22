"""Stage-1b: identity detection over the full mix via chromaprint.

Not a similarity scorer — a presence detector. For every short window
of the mix, chromaprint reports which reference tracks' fingerprints
hash-match. Because fingerprints index spectral-peak constellations
(not feature averages), random chroma coincidences don't produce hits;
actual plays do.

This is what distinguishes "is Good Grief playing right now" from
"does chroma at this instant look like Good Grief's chroma" — the
failure mode that sent every alignment span past the cue window in
the 00:26 run.
"""

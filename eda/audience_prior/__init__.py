"""Audience-conditioned prior over a DJ set's tracklist (read-only on taste_warehouse).

Information-dynamics surprise is *listener-relative*: a track's salience to an
audience is how familiar it already is to them, not its acoustics. This module
builds ``familiarity(track | audience)`` for a set's tracklist from the
SoundCloud per-listener like graph collected in ``workspaces/taste_prior``
(``data/taste/taste_warehouse.db``), by fuzzy-joining the tracklist's
artist+title to the audience's liked-track vocabulary.

It is the *recognition* dimension a MERT embedding cannot see — and the reason
acappella **audio** surprise was at chance (see eda/alignment/info_dynamics §v4):
the acappella's effect is recognition, which lives in the listener prior.

Read-only consumer; never writes to the taste warehouse. New eda module, kept
separate from the live ``workspaces/taste_prior`` collection code.
"""

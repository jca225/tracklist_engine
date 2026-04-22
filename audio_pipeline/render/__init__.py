"""Playback-score renderer.

Given a `measure_alignment` table — per mix-measure, which ref measure of
which track plays at what pitch-shift / tempo-ratio / stem-mask — this
package reconstructs the mix audio by time-stretching, pitch-shifting,
and mixing the ref stems.

The reconstructed audio is the self-consistency ground truth: if the
alignment score is correct, rendering + comparing against the actual
mix on MFCC distance produces a small number. This replaces hand-
annotated yaml as the primary evaluation signal.
"""

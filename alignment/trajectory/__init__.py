"""Learned segment-trajectory decode — training scaffold.

The 2026-07-07 end-to-end eval confirmed the bottleneck: richer candidates
moved placement but NOT trajectory accuracy (multiseg+loop ~26-27% on BB12,
acappella ~10%). The non-learned Viterbi (`path_decode.decode_path`) is the
ceiling; this package is the training substrate for its learned replacement:

- `targets`   — GT span -> per-frame ref-position labels (+NULL, +abstain),
                rasterized with the scorer's own interpolation so training
                optimizes exactly what `trajectory_acc` measures.
- `features`  — pooled, stem-routed match features (chroma/HuBERT) + mel,
                built on the existing `.feat_cache`.
- `data`      — torch Dataset over N GT sets (BB12 + BB11 today) + collate.
- `recon_loss`— the label-free reconstruction objective (host channel only,
                per docs/reconstruction_supervision_plan.md).
- `model`     — small conv decoder over cross-similarity channels.
- `train`     — CLI: train on one set, eval held-out on the other.
"""

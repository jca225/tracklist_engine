"""The TRM actor — a typed skeleton slot, not yet an implementation.

The framework registers `actor:trm` so the ablation matrix carries a TRM row
today; the recursive binder itself is built by a follow-up effort scoped in
`docs/trm_decoder_bakeoff.md`. Until then any backend asked to *run* `trm` raises
`NotImplementedError` pointing at that spec — a loud skeleton, never a silent
passthrough that would flatter an empty row.

`TRM_KNOBS` is the ablation surface the bake-off will expose (wired here as
documentation so config authors know the axes ahead of time)."""

from __future__ import annotations

BAKEOFF_DOC = "alignment/docs/trm_decoder_bakeoff.md"

SKELETON_MESSAGE = (
    "actor:trm is a skeleton slot — the recursive binder is not implemented yet. "
    f"Build it per {BAKEOFF_DOC} (replace trajectory/model.py, add train.py "
    "--model trm). Until then it cannot run."
)

# The ablation axes the TRM decoder will expose (bake-off §2). Config-visible now
# so studies can be authored against a stable knob vocabulary.
TRM_KNOBS: dict[str, object] = {
    "T": 3,  # inner recursion depth (full backprop through all T)
    "N_sup": 6,  # deep-supervision improvement steps
    "offset_bin_res": 0.5,  # answer encoded in clip-start OFFSET coords (bake-off §2.3)
    "mixer": "mlp",  # {mlp, attn} — TRM-MLP beat attention on fixed-grid tasks
    "y_mode": "once",  # {once, every} — y-once is TRM-faithful
}

__all__ = ["BAKEOFF_DOC", "SKELETON_MESSAGE", "TRM_KNOBS"]

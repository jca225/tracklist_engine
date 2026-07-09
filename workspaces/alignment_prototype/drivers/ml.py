"""Driver C1 — hybrid ML: learned TrajectoryDecoder owns the ref-segment stage.

The honest scope (locked decision): identity + placement are BORROWED from the
classical base timeline; the learned piece is the trajectory decode — replacing
`joint_ref_decode`'s hand Viterbi with the trained `TrajectoryDecoder` +
`viterbi_segments`. A fully-learned placement selector (C2) is deferred until a
third GT set exists (n=2 overfits), so this driver does NOT re-place or re-id.

Per span it rebuilds the exact `(2, Tm, Tr)` cross-similarity grid the decoder
trained on (`trajectory.data.__getitem__`: pooled stem-routed match feature @ ref,
pooled mel @ ref), forwards the checkpoint, offset-state-Viterbi-decodes the
logits, and writes the segments back in `joint_ref_decode`'s schema. Feature
routing follows the GT stem (acappella->HuBERT, else chroma) to match the decoder's
training distribution. Spans whose audio the manifest can't serve keep the
classical ref_segments (reported, never silently zero-filled).

Checkpoint: `.cache/trajectory/decoder_<source_set_id>.pt` — the decoder trained
on the OTHER GT set (honest cross-set transfer). Reconstructed from its saved
`args` so stretch-slope channels match.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import (
    OUT_DIR,
    SetContext,
    finalize,
    gt_stem_by_slot,
    load_timeline_dict,
    norm_slot,
)

# OUT_DIR is <prototype>/out; the trajectory checkpoints live beside it.
_CKPT_DIR = OUT_DIR.parent / ".cache" / "trajectory"


class HybridMlDriver:
    name = "ml"

    def __init__(
        self,
        base_timeline: Path,
        *,
        checkpoint: Path | None = None,
        lam: float = 16.0,
        back_ratio: float = 3.0,
        device: str = "auto",
        gate_margin: float | None = None,
    ) -> None:
        self.base = Path(base_timeline)
        self.checkpoint = checkpoint
        self.lam = lam
        self.back_ratio = back_ratio
        self.device = device
        # Per-span confidence gate: keep classical segments unless the learned
        # decode's margin clears this bar. None = replace unconditionally (the
        # ungated driver). The clean board showed ml wins where classical decode
        # is weak (BB11) but regresses where it is already strong (BB12); gating
        # on ml confidence lets the learned segments through only where the model
        # is sure, so it can add without subtracting.
        self.gate_margin = gate_margin

    def align_set(self, ctx: SetContext) -> Path:
        import torch

        from workspaces.alignment_prototype.recon_probe import find_aligning_dir
        from workspaces.alignment_prototype.trajectory.decode import viterbi_segments
        from workspaces.alignment_prototype.trajectory.features import (
            BIN_S,
            FEAT_KIND,
            FeatureBank,
            find_mix_stems,
            resolve_span_audio,
        )
        from workspaces.alignment_prototype.trajectory.model import TrajectoryDecoder

        device = _pick_device(self.device)
        model = self._load_model(ctx, device)

        timeline = load_timeline_dict(self.base)
        gt_stems = gt_stem_by_slot(ctx.gt_yaml)

        aligning = find_aligning_dir(ctx.set_id)
        if aligning is None:
            raise FileNotFoundError(f"no ~/aligning folder for {ctx.set_id}")
        manifest = json.loads((aligning / "manifest.json").read_text())
        # by track_id AND recording_id — timeline spans carry canonical
        # recording_id, the manifest keys on scrape track_id (mirrors
        # joint_ref_decode's bridge).
        by_tid = {str(t["track_id"]): t for t in manifest["tracks"]}
        for t in manifest["tracks"]:
            if t.get("recording_id"):
                by_tid.setdefault(str(t["recording_id"]), t)
        mix_stems = find_mix_stems(aligning)
        mix_full = Path(manifest.get("mix_local_path") or (aligning / "mix.m4a"))
        bank = FeatureBank(BIN_S)

        out_spans, n_decoded, n_gated, skips = [], 0, 0, []
        for s in timeline["spans"]:
            stem = gt_stems.get(norm_slot(s["slot_label"])) or (
                s.get("claimed_stem") or "regular"
            )
            row = {
                "track_id": str(s["recording_id"]),
                "claimed_stem": stem,
                "set_start_s": float(s["set_start_s"]),
                "set_end_s": float(s["set_end_s"]),
            }
            audio = resolve_span_audio(aligning, by_tid, mix_stems, mix_full, row)
            if isinstance(audio, str):  # skip reason — keep classical segments
                skips.append((s["slot_label"], audio))
                out_spans.append(s)
                continue

            grid = _build_sim(bank, audio, row, BIN_S)
            if grid is None:
                skips.append((s["slot_label"], "span too short after pooling"))
                out_spans.append(s)
                continue
            sim, tr = grid
            with torch.no_grad():
                sim_t = torch.from_numpy(sim)[None].to(device)
                fk = torch.tensor([FEAT_KIND[audio.match_feature]], device=device)
                rv = torch.ones(1, tr, dtype=torch.bool, device=device)
                logits = model(sim_t, fk, rv)[0]  # (Tm, Tr+1)
            segs, score = viterbi_segments(
                logits,
                BIN_S,
                lam=self.lam,
                back_ratio=self.back_ratio,
                return_score=True,
            )
            if not segs:
                skips.append((s["slot_label"], "decoder returned no segments"))
                out_spans.append(s)
                continue
            # confidence gate: below the bar, the model isn't sure enough to
            # override a (possibly already-good) classical decode — keep classical.
            if self.gate_margin is not None and score < self.gate_margin:
                skips.append(
                    (
                        s["slot_label"],
                        f"gated: ml conf {score:.3f} < {self.gate_margin}",
                    )
                )
                out_spans.append({**s, "ml_decode_score": round(score, 4)})
                n_gated += 1
                continue

            s0 = float(s["set_start_s"])
            ref_segments = [
                {
                    "mix_start_s": round(s0 + ms, 2),
                    "ref_start_s": round(rs, 2),
                    "ref_end_s": round(re, 2),
                }
                for (ms, rs, re) in segs
            ]
            out_spans.append(
                {
                    **s,
                    "claimed_stem": stem,
                    "ref_segments": ref_segments,
                    "ref_start_s": ref_segments[0]["ref_start_s"],
                    "ref_end_s": ref_segments[-1]["ref_end_s"],
                    "ref_decoder": "trajectory",
                    "ml_decode_score": round(score, 4),
                }
            )
            n_decoded += 1

        gate_note = (
            f", {n_gated} gated (conf < {self.gate_margin})"
            if self.gate_margin is not None
            else ""
        )
        print(
            f"  ml: {n_decoded} spans decoded by TrajectoryDecoder, "
            f"{len(skips)} kept classical segments{gate_note}"
        )
        for slot, reason in skips[:8]:
            print(f"      keep-classical {slot}: {reason}")
        payload = {
            **timeline,
            "spans": out_spans,
            "ref_decode": "trajectory decoder (learned) + classical placement",
            "driver": f"hybrid-ml{f' gate>={self.gate_margin}' if self.gate_margin is not None else ''}",
        }
        return finalize(payload, ctx.timeline_path(self.name))

    def _load_model(self, ctx: SetContext, device):
        import torch

        from workspaces.alignment_prototype.trajectory.model import TrajectoryDecoder

        ckpt = self.checkpoint or (_CKPT_DIR / f"decoder_{ctx.source_set_id}.pt")
        if not Path(ckpt).is_file():
            raise FileNotFoundError(
                f"no trajectory checkpoint at {ckpt}. Train the cross-set decoder:\n"
                f"  venvs/audio/bin/python -m workspaces.alignment_prototype.trajectory.train "
                f"--split set --train-set {ctx.source_set_id} --eval-set {ctx.set_id}"
            )
        state = torch.load(ckpt, map_location=device)
        saved = state.get("args", {}) or {}
        slopes = tuple(
            float(v) for v in str(saved.get("slopes", "")).split(",") if v.strip()
        )
        model = TrajectoryDecoder(stretch_slopes=slopes).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        print(f"  ml: loaded {Path(ckpt).name} (slopes={slopes or 'none'})")
        return model


def _build_sim(bank, audio, row: dict, bin_s: float):
    """Rebuild the (2, Tm, Tr) grid exactly as trajectory.data.__getitem__ —
    same pooling, same span slice, same channel order — so the checkpoint sees
    its training distribution. Returns (sim, Tr) or None if the span is too short."""
    mix_match = bank.match(audio.mix_path, audio.match_feature)
    ref_match = bank.match(audio.ref_path, audio.match_feature)
    mix_mel = bank.mel(audio.mix_path)
    ref_mel = bank.mel(audio.ref_path)

    s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
    b0 = max(0, int(round(s0 / bin_s)))
    n_bins = max(1, int(round((s1 - s0) / bin_s)))
    tm = min(n_bins, mix_match.shape[0] - b0, mix_mel.shape[0] - b0)
    tr = min(ref_match.shape[0], ref_mel.shape[0])
    if tm < 2 or tr < 2:
        return None
    mm = mix_match[b0 : b0 + tm]
    rm = ref_match[:tr]
    me = mix_mel[b0 : b0 + tm]
    re = ref_mel[:tr]
    sim = np.stack([mm @ rm.T, me @ re.T]).astype(np.float32)
    return sim, tr


def _pick_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

"""Torch Dataset over N ground-truth sets (BB12 + BB11 today).

Each example is one GT play-span:

- `sim`        (2, Tm, Tr) — cross-similarity channels between the span's mix
               bins and the whole ref: [0] the stem-routed match feature
               (chroma or HuBERT), [1] mel. Dimension-free across feature
               types, so acappella (HuBERT, D=768) and regular (chroma, D=12)
               spans batch together.
- `mix_mel`    (Tm, 64) / `ref_mel` (Tr, 64) — reconstruction-loss inputs.
- `target_idx` (Tm,) — GT ref bin per mix bin (-1 = ignore).
- `target_null`(Tm,) — frames where the fader has the clip inaudible.
- `recon_ok`   scalar — host/regular span (the only class where per-window
               reconstruction is a valid teacher, per the plan).
- `abstain`    scalar — `unalignable` GT row (positive abstain label).

Rows the manifest can't serve are skipped LOUDLY (`Dataset.skipped`), never
silently zero-filled — that hid the slot-039 miss once already.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from workspaces.alignment_prototype.recon_probe import find_aligning_dir, find_mix_stems

from .features import BIN_S, FEAT_KIND, FeatureBank, SpanAudio, resolve_span_audio
from .targets import KIND_IGNORE, KIND_NULL, KIND_POSITION, raster_targets

_REPO = Path(__file__).resolve().parent.parent.parent.parent


@dataclass(frozen=True)
class SpanSpec:
    """Lazy pointer to one trainable span (no audio work at build time)."""

    set_id: str
    row: dict
    audio: SpanAudio


@dataclass(frozen=True)
class SkippedSpan:
    set_id: str
    slot_label: str
    reason: str


def _load_gt_rows(yaml_path: Path) -> tuple[str, list[dict]]:
    import yaml

    d = yaml.safe_load(open(yaml_path))
    return str(d["set_id"]), list(d["tracks"])


class TrajectorySpanDataset(Dataset):
    """GT spans from one or more sets, feature-lazy, ready for a DataLoader.

    `sets` = [(set_id, gt_yaml_path)]; each set needs its pulled
    `~/aligning/<set_id>__*/` folder (mix, stems, manifest.json).
    """

    def __init__(
        self,
        sets: Sequence[tuple[str, Path | str]],
        bin_s: float = BIN_S,
        min_gain: float = 0.1,
        aligning_dirs: dict[str, Path] | None = None,
    ) -> None:
        self.bin_s = bin_s
        self.min_gain = min_gain
        self.bank = FeatureBank(bin_s)
        self.specs: list[SpanSpec] = []
        self.skipped: list[SkippedSpan] = []
        aligning_dirs = aligning_dirs or {}

        for set_id, yaml_path in sets:
            # Synthetic sets pass an explicit folder (no ~/aligning entry); real
            # sets resolve by set-id prefix as before.
            aligning = aligning_dirs.get(set_id) or find_aligning_dir(set_id)
            if aligning is None:
                raise FileNotFoundError(f"no ~/aligning folder for {set_id}")
            manifest = json.load(open(aligning / "manifest.json"))
            by_tid = {str(t["track_id"]): t for t in manifest["tracks"]}
            mix_stems = find_mix_stems(aligning)
            mix_full = Path(manifest.get("mix_local_path") or (aligning / "mix.m4a"))
            gt_set_id, rows = _load_gt_rows(Path(yaml_path))
            if gt_set_id != set_id:
                raise ValueError(f"GT yaml is for {gt_set_id}, expected {set_id}")
            for row in rows:
                slot = str(row.get("slot_label") or row.get("track") or "?")
                if row.get("skip_training"):
                    self.skipped.append(SkippedSpan(set_id, slot, "skip_training"))
                    continue
                audio = resolve_span_audio(aligning, by_tid, mix_stems, mix_full, row)
                if isinstance(audio, str):
                    self.skipped.append(SkippedSpan(set_id, slot, audio))
                    continue
                self.specs.append(SpanSpec(set_id=set_id, row=row, audio=audio))

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> dict:
        spec = self.specs[idx]
        row, audio = spec.row, spec.audio
        bin_s = self.bin_s

        mix_match = self.bank.match(audio.mix_path, audio.match_feature)
        ref_match = self.bank.match(audio.ref_path, audio.match_feature)
        mix_mel = self.bank.mel(audio.mix_path)
        ref_mel = self.bank.mel(audio.ref_path)

        s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
        b0 = max(0, int(round(s0 / bin_s)))
        n_bins = max(1, int(round((s1 - s0) / bin_s)))
        tm = min(n_bins, mix_match.shape[0] - b0, mix_mel.shape[0] - b0)
        tr = min(ref_match.shape[0], ref_mel.shape[0])
        if tm < 2 or tr < 2:
            raise ValueError(
                f"span too short after pooling: {spec.set_id}/{row.get('slot_label')}"
                f" tm={tm} tr={tr}"
            )

        mm = mix_match[b0 : b0 + tm]
        rm = ref_match[:tr]
        me = mix_mel[b0 : b0 + tm]
        re = ref_mel[:tr]
        sim = np.stack([mm @ rm.T, me @ re.T]).astype(np.float32)

        times = (b0 + np.arange(tm) + 0.5) * bin_s
        tgt = raster_targets(row, times, ref_dur_s=tr * bin_s, min_gain=self.min_gain)
        target_idx = np.full(tm, -1, dtype=np.int64)
        pos = tgt.kind == KIND_POSITION
        target_idx[pos] = np.clip(
            np.floor(tgt.ref_pos_s[pos] / bin_s).astype(np.int64), 0, tr - 1
        )

        return {
            "sim": torch.from_numpy(sim),
            "mix_mel": torch.from_numpy(np.ascontiguousarray(me)),
            "ref_mel": torch.from_numpy(np.ascontiguousarray(re)),
            "target_idx": torch.from_numpy(target_idx),
            "target_null": torch.from_numpy(tgt.kind == KIND_NULL),
            "gain": torch.from_numpy(tgt.gain.astype(np.float32)),
            "feat_kind": torch.tensor(FEAT_KIND[audio.match_feature]),
            "recon_ok": torch.tensor(
                str(row.get("claimed_stem") or "regular") == "regular"
            ),
            "abstain": torch.tensor(tgt.abstain),
            "meta": {
                "set_id": spec.set_id,
                "slot_label": str(row.get("slot_label") or ""),
                "claimed_stem": str(row.get("claimed_stem") or "regular"),
                "span_class": tgt.span_class,
                "index": idx,
            },
        }

    def report_skipped(self) -> str:
        lines = [f"  SKIP {s.set_id}/{s.slot_label}: {s.reason}" for s in self.skipped]
        return "\n".join(lines) if lines else "  (none)"


def collate_spans(batch: list[dict]) -> dict:
    """Zero-pad Tm/Tr to the batch max; padding is masked via *_valid."""
    b = len(batch)
    tm = max(x["sim"].shape[1] for x in batch)
    tr = max(x["sim"].shape[2] for x in batch)
    sim = torch.zeros(b, 2, tm, tr)
    mix_mel = torch.zeros(b, tm, batch[0]["mix_mel"].shape[1])
    ref_mel = torch.zeros(b, tr, batch[0]["ref_mel"].shape[1])
    target_idx = torch.full((b, tm), -1, dtype=torch.int64)
    target_null = torch.zeros(b, tm, dtype=torch.bool)
    gain = torch.zeros(b, tm)
    mix_valid = torch.zeros(b, tm, dtype=torch.bool)
    ref_valid = torch.zeros(b, tr, dtype=torch.bool)
    for i, x in enumerate(batch):
        ti, ri = x["sim"].shape[1], x["sim"].shape[2]
        sim[i, :, :ti, :ri] = x["sim"]
        mix_mel[i, :ti] = x["mix_mel"]
        ref_mel[i, :ri] = x["ref_mel"]
        target_idx[i, :ti] = x["target_idx"]
        target_null[i, :ti] = x["target_null"]
        gain[i, :ti] = x["gain"]
        mix_valid[i, :ti] = True
        ref_valid[i, :ri] = True
    return {
        "sim": sim,
        "mix_mel": mix_mel,
        "ref_mel": ref_mel,
        "target_idx": target_idx,
        "target_null": target_null,
        "gain": gain,
        "mix_valid": mix_valid,
        "ref_valid": ref_valid,
        "feat_kind": torch.stack([x["feat_kind"] for x in batch]),
        "recon_ok": torch.stack([x["recon_ok"] for x in batch]),
        "abstain": torch.stack([x["abstain"] for x in batch]),
        "meta": [x["meta"] for x in batch],
    }


GT_FIXTURES = {
    "1fsnxchk": _REPO / "labeling/fixtures/bb12_ground_truth.yaml",
    "2nvzlh2k": _REPO / "labeling/fixtures/bb11_ground_truth.yaml",
}

"""Research engine for the two alignment walls — placement & structure.

Companion to `walls.ipynb`. Keeps the *logic* here (unit-tested) and leaves the
notebook thin. The reason-to-exist is a single question you can answer with your
ears: **when the aligner puts a span in the wrong place, was it a beatable miss
or a physically-unwinnable one?**

Two families:

- **Set X-ray** (`set_xray`): per-span predicted-vs-GT scorecard (needs a
  predicted timeline on disk — run `ensure_timeline`). Columns: identity right?,
  placement error, trajectory score, and a transparent binding-cause guess.
- **Structure / distinguishability** (`structure_report`, `span_distinguishability`):
  needs ONLY GT + reference audio. For each repeated span it slides the played
  content over the reference and measures the best-vs-runner-up peak *margin* —
  small margin = two ref positions look identical = the ~50% ceiling (no
  algorithm can win); large margin = a recoverable miss. `render_span_snippets`
  writes the played clip + the two competing reference positions so you can
  listen and judge for yourself.

Everything reuses the production probes/scorer (chroma, HuBERT, score_spans) —
no reinvented DSP. Audio conventions: mono, SR=22050, HOP=512.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 22050
HOP = 512

SET_ALIASES = {"bb11": "2nvzlh2k", "bb12": "1fsnxchk"}
_GT_FIXTURE = {
    "2nvzlh2k": "bb11_ground_truth.yaml",
    "1fsnxchk": "bb12_ground_truth.yaml",
}

_REPO = Path(__file__).resolve().parents[3]


# ── pure engine (unit-tested) ─────────────────────────────────────────────────


def frames_to_s(frame: float) -> float:
    """Feature-frame index → seconds (SR/HOP grid)."""
    return float(frame) * HOP / SR


def similarity_curve(win: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-position similarity of a (D,M) window sliding over a (D,N) reference.

    Score[k] = mean per-frame cosine of `win` against `ref[:, k:k+M]`. Peaks
    mark reference positions whose content matches the window; two near-equal
    peaks mean the window's true position is physically ambiguous.
    """
    d, m = win.shape
    _, n = ref.shape
    if n < m:
        return np.zeros(0, dtype=np.float32)
    wn = win / (np.linalg.norm(win, axis=0, keepdims=True) + 1e-8)
    rn = ref / (np.linalg.norm(ref, axis=0, keepdims=True) + 1e-8)
    out = np.empty(n - m + 1, dtype=np.float32)
    for k in range(n - m + 1):
        out[k] = float(np.mean(np.sum(wn * rn[:, k : k + m], axis=0)))
    return out


def top_two_peaks(scores: np.ndarray, min_sep: int) -> tuple[int, float, int, float]:
    """Best peak + the best OTHER peak at least `min_sep` frames away.

    Returns (best_idx, best_score, runnerup_idx, runnerup_score). The separation
    stops a single broad bump from counting as its own runner-up — the runner-up
    must be a genuinely distinct reference position.
    """
    if scores.size == 0:
        return 0, 0.0, 0, 0.0
    best = int(np.argmax(scores))
    masked = scores.astype(np.float64).copy()
    lo, hi = max(0, best - min_sep), min(len(scores), best + min_sep + 1)
    masked[lo:hi] = -np.inf
    if not np.isfinite(masked).any():
        return best, float(scores[best]), best, float(scores[best])
    second = int(np.argmax(masked))
    return best, float(scores[best]), second, float(scores[second])


def classify_distinguishability(
    best: float, runner_up: float, margin_thresh: float = 0.1
) -> str:
    """'recoverable' if the true position clearly wins, else 'ambiguous' (ceiling).

    Margin at or below threshold → ambiguous (conservative: we do not claim a
    repeat is winnable unless the best position beats the runner-up decisively).
    The 1e-9 slack keeps float noise (e.g. 0.8-0.7 != 0.1 exactly) from flipping
    an at-threshold margin to 'recoverable'.
    """
    return "recoverable" if (best - runner_up) > margin_thresh + 1e-9 else "ambiguous"


# ── audio / feature wrappers (thin; smoke-tested in the notebook) ─────────────


def load_mono(
    path: str | Path, start_s: float | None = None, dur_s: float | None = None
) -> np.ndarray:
    import warnings

    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(
            str(path), sr=SR, mono=True, offset=start_s or 0.0, duration=dur_s
        )
    return y.astype(np.float32)


def chroma_feat(y: np.ndarray) -> np.ndarray:
    from workspaces.alignment_prototype.refine_ref_offsets import chroma

    return chroma(y)


def hubert_feat(y: np.ndarray, layer: int = 9) -> np.ndarray:
    """(768, T) HuBERT embeddings — for vocal/acappella distinguishability."""
    from workspaces.section_hsmm.similarity_probe import _hubert

    return _hubert(y, layer)


def feature_for_stem(stem: str, feature: str = "auto") -> str:
    """Route to the nuisance-invariant feature: vocals→HuBERT, else chroma.

    Chroma is broken on re-pitched acappellas (a large fraction are pitched to
    the bed key); HuBERT is key-invariant. `feature='auto'` picks correctly;
    pass 'chroma'/'hubert' to force.
    """
    if feature != "auto":
        return feature
    return "hubert" if stem == "acappella" else "chroma"


# ── set loading ───────────────────────────────────────────────────────────────


def resolve_set_id(set_ref: str) -> str:
    return SET_ALIASES.get(set_ref, set_ref)


def aligning_dir(set_ref: str) -> Path | None:
    from workspaces.pws_aligner.capture_votes import _find_aligning_dir

    return _find_aligning_dir(resolve_set_id(set_ref))


def load_gt(set_ref: str):
    from labeling.ground_truth.schema import load

    set_id = resolve_set_id(set_ref)
    path = _REPO / "labeling" / "fixtures" / _GT_FIXTURE[set_id]
    res = load(path)
    if not res.is_ok():
        raise RuntimeError(f"GT load failed for {set_id}: {res.error}")
    return res.value


def load_manifest(set_ref: str) -> dict[str, dict]:
    from workspaces.pws_aligner.capture_votes import _load_manifest_by_rid

    set_id = resolve_set_id(set_ref)
    d = aligning_dir(set_id)
    if d is None:
        raise RuntimeError(f"no ~/aligning/{set_id}__* dir — pull the set first")
    return _load_manifest_by_rid(d, set_id)


def timeline_path(set_ref: str) -> Path:
    set_id = resolve_set_id(set_ref)
    return (
        _REPO
        / "workspaces"
        / "alignment_prototype"
        / "out"
        / f"{set_id}_predicted_timeline.json"
    )


def have_timeline(set_ref: str) -> bool:
    return timeline_path(set_ref).is_file()


def ensure_timeline_cmd(set_ref: str) -> str:
    """The exact command to generate a missing predicted timeline (run in a shell)."""
    set_id = resolve_set_id(set_ref)
    return (
        "venvs/audio/bin/python -m workspaces.alignment_prototype.infer "
        f"--set-id {set_id} --band-s 45 && "
        "venvs/audio/bin/python -m workspaces.alignment_prototype.joint_ref_decode "
        f"--set-id {set_id}"
    )


# ── audio routing per stem ────────────────────────────────────────────────────


def mix_stem_path(set_ref: str, stem: str) -> Path | None:
    from workspaces.pws_aligner.capture_votes import _mix_audio_path

    d = aligning_dir(set_ref)
    return _mix_audio_path(d, stem) if d else None


def ref_audio_path(set_ref: str, recording_id: str, stem: str) -> Path | None:
    from workspaces.pws_aligner.capture_votes import _ref_audio_path

    row = load_manifest(set_ref).get(recording_id)
    return _ref_audio_path(row, stem) if row else None


# ── Set X-ray (predicted vs GT) ───────────────────────────────────────────────


def _binding_cause(id_correct, place_err_s, strict) -> str:
    """Transparent one-cause-per-span guess (inspect, don't trust blindly)."""
    if id_correct is False:
        return "identity"
    if place_err_s is not None and place_err_s > 10.0:
        return "placement"
    if strict is not None and strict < 0.5:
        return "structure/decode"
    return "ok"


def set_xray(set_ref: str, *, fibers: bool = True):
    """Per-span predicted-vs-GT scorecard as a DataFrame, worst-first.

    Requires a predicted timeline on disk (see `have_timeline` / `ensure_timeline_cmd`).
    Columns: slot, stem, id_correct, place_err_s, ref_err_s, strict, fiber, cause.
    """
    import pandas as pd

    if not have_timeline(set_ref):
        raise RuntimeError(
            f"no predicted timeline for {resolve_set_id(set_ref)} — run:\n  "
            + ensure_timeline_cmd(set_ref)
        )
    from workspaces.alignment_prototype.score_timeline_vs_gt import score_spans

    set_id = resolve_set_id(set_ref)
    scores = score_spans(set_id, timeline_path(set_ref), fibers=fibers)
    rows = []
    for s in scores:
        idc = getattr(s, "id_correct", None)
        pe = getattr(s, "place_err_s", None)
        st = getattr(s, "strict", None)
        rows.append(
            {
                "slot": getattr(s, "slot_label", getattr(s, "slot", "?")),
                "stem": getattr(s, "claimed_stem", getattr(s, "stem", "?")),
                "id_correct": idc,
                "place_err_s": pe,
                "ref_err_s": getattr(s, "ref_err_s", None),
                "strict": st,
                "fiber": getattr(s, "fiber", None),
                "cause": _binding_cause(idc, pe, st),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["strict", "place_err_s"], ascending=[True, False], na_position="first"
        ).reset_index(drop=True)
    return df


# ── Structure / distinguishability (GT + ref audio only) ─────────────────────


@dataclass(frozen=True)
class Distinguishability:
    slot: str
    recording_id: str
    stem: str
    feature: str
    best_ref_s: float
    best_score: float
    runnerup_ref_s: float
    runnerup_score: float
    margin: float
    gt_ref_s: float
    best_is_gt: bool  # did the top peak land near the GT ref position?
    verdict: str  # 'recoverable' | 'ambiguous'


def span_distinguishability(
    set_ref: str,
    track,
    *,
    feature: str = "auto",
    min_sep_s: float = 2.0,
    margin_thresh: float = 0.1,
    gt_tol_s: float = 2.0,
) -> Distinguishability | None:
    """Is this span's reference position physically winnable? (chroma or hubert).

    Slides the PLAYED content (mix stem over the span) across the full reference,
    finds the best + runner-up peaks, and returns the margin + verdict. `track`
    is a GroundTruthTrack. Returns None if audio is missing. NOTE: computed at
    native tempo (no stretch search) — a first-order read; steep tempo rides will
    look less distinguishable than they are.
    """
    stem = track.claimed_stem
    mp = mix_stem_path(set_ref, stem)
    rp = ref_audio_path(set_ref, track.track_id or "", stem)
    if mp is None or not mp.is_file() or rp is None or not rp.is_file():
        return None
    dur = max(4.0, track.set_end_s - track.set_start_s)
    mix_span = load_mono(mp, start_s=track.set_start_s, dur_s=dur)
    ref = load_mono(rp)
    feature = feature_for_stem(stem, feature)  # 'auto' → hubert for vocals, else chroma
    feat = hubert_feat if feature == "hubert" else chroma_feat
    win, ref_f = feat(mix_span), feat(ref)
    if win.shape[1] < 4 or ref_f.shape[1] <= win.shape[1]:
        return None
    curve = similarity_curve(win, ref_f)
    min_sep = int(round(min_sep_s * SR / HOP))
    bi, bv, si, sv = top_two_peaks(curve, min_sep)
    best_s, ru_s = frames_to_s(bi), frames_to_s(si)
    return Distinguishability(
        slot=track.slot_label,
        recording_id=track.track_id or "",
        stem=stem,
        feature=feature,
        best_ref_s=best_s,
        best_score=bv,
        runnerup_ref_s=ru_s,
        runnerup_score=sv,
        margin=bv - sv,
        gt_ref_s=track.ref_start_s,
        best_is_gt=abs(best_s - track.ref_start_s) <= gt_tol_s,
        verdict=classify_distinguishability(bv, sv, margin_thresh),
    )


def _is_repeatish(track) -> bool:
    """Spans where 'which repeat?' can bite: loops, multiseg, or long plays."""
    if getattr(track, "is_loop", False):
        return True
    if len(getattr(track, "ref_segments", ()) or ()) > 1:
        return True
    return (track.set_end_s - track.set_start_s) >= 20.0


def structure_report(
    set_ref: str,
    *,
    feature: str = "auto",
    only_repeatish: bool = True,
    limit: int | None = None,
    **kw,
):
    """Distinguishability over a set's spans → DataFrame, most-ambiguous first.

    This is the empirical read on the ~50% ceiling: `verdict=='ambiguous'` spans
    are ones no algorithm can win from audio alone. Set `only_repeatish=False`
    to include every alignable span. `limit` caps how many spans are scored
    (HuBERT on acappellas is slow — use a small limit for a quick first pass).
    """
    import pandas as pd

    gt = load_gt(set_ref)
    rows = []
    for t in gt.tracks:
        if limit is not None and len(rows) >= limit:
            break
        if t.unalignable or not t.track_id:
            continue
        if only_repeatish and not _is_repeatish(t):
            continue
        d = span_distinguishability(set_ref, t, feature=feature, **kw)
        if d is None:
            continue
        rows.append(
            {
                "slot": d.slot,
                "stem": d.stem,
                "margin": round(d.margin, 3),
                "verdict": d.verdict,
                "best_ref_s": round(d.best_ref_s, 1),
                "runnerup_ref_s": round(d.runnerup_ref_s, 1),
                "gt_ref_s": round(d.gt_ref_s, 1),
                "best_is_gt": d.best_is_gt,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("margin").reset_index(drop=True)
    return df


def render_span_snippets(
    set_ref: str, track, out_dir: str | Path, *, feature: str = "auto", **kw
) -> dict[str, Path] | None:
    """Write 3 clips so you can JUDGE by ear: played span, best ref, runner-up ref.

    If the played clip is indistinguishable from BOTH ref positions, that span is
    the ceiling made audible. Returns {name: path} or None if audio missing.
    """
    from workspaces.alignment_prototype.review.render_review_snippets import (
        ffmpeg_snippet,
    )

    d = span_distinguishability(set_ref, track, feature=feature, **kw)
    if d is None:
        return None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dur = max(4.0, track.set_end_s - track.set_start_s)
    mp = mix_stem_path(set_ref, track.claimed_stem)
    rp = ref_audio_path(set_ref, track.track_id or "", track.claimed_stem)
    tag = f"{d.slot}_{d.feature}"
    jobs = {
        f"{tag}__PLAYED": (mp, track.set_start_s),
        f"{tag}__refBEST_{d.best_ref_s:.0f}s": (rp, d.best_ref_s),
        f"{tag}__refRUNNERUP_{d.runnerup_ref_s:.0f}s": (rp, d.runnerup_ref_s),
    }
    written: dict[str, Path] = {}
    for name, (src, start) in jobs.items():
        dst = out / f"{name}.mp3"
        if src and ffmpeg_snippet(Path(src), float(start), min(dur, 15.0), dst):
            written[name] = dst
    return written

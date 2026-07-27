#!/usr/bin/env python3
"""External validation of this repo's HuBERT fibers against human structure GT.

Question this answers: do our self-repeat classes (``ref_fibers.compute_fibers``)
actually recover human-annotated *section repeats* — independent of our tiny Big
Bootie ground truth?

**What a fiber is (recap).** ``compute_fibers`` labels each downsampled frame with
a self-repeat class id: frames sharing an id are the SAME repeated content (a
chorus played 3x = one fiber, 3 instances); -1 = silence / non-repeated. Feats
MUST be HuBERT L9 on the SR/HOP=22050/512 grid (fps=SR/HOP≈43.07) — mirrored from
``path_decode.fibers_for`` / ``continuity_refine._compute_hubert``.

**The claim under test.** A human section annotation (SALAMI uppercase letters)
IS a repeat labeling: every segment tagged ``A`` is the same section recurring.
So "did our fibers cluster the same-section frames together" == the standard
segment-clustering scores ``mir_eval.segment.pairwise`` (pairwise P/R/F) and
``mir_eval.segment.nce`` (V-measure S_over/S_under).

**Dataset.** SALAMI via mirdata. Audio is NOT distributed with the annotations,
but the SALAMI-Internet-Archive subset (``id_index_internetarchive.csv``, 477
tracks) ships direct archive.org URLs. Those 2012-era derivative filenames have
rotted (~5% resolve by literal URL), but the *items* still exist — so we resolve
each track through the archive.org metadata API (item-id + track stem -> current
.mp3) rather than trusting the stale URL. NOTE: SALAMI-IA is live jam-band /
improv audio — looser repeat structure than studio pop, so treat the absolute
number as a floor for what HuBERT fibers do on the cleaner material our aligner
actually sees. See findings for the honest caveat.

**Timebase / interval conversion (the bug-prone part, done explicitly).**
Fibers come back as per-frame labels at ``label_hz`` (~8 Hz, the compute_fibers
downsample grid). GT is (intervals Nx2 seconds, letter labels). mir_eval's
pairwise/nce compare two *segmentations that tile the same span*. To make the two
strictly comparable with no fps/offset drift we:
  1. pick a common dense grid at ``GRID_HZ`` (default 10 Hz) over
     ``[0, min(track_dur, gt_end, fiber_end))``;
  2. sample BOTH labelings at grid-cell centers (fiber via
     ``round(t*label_hz)``, GT via interval containment);
  3. emit unit-width intervals ``[k/GRID_HZ, (k+1)/GRID_HZ]`` with those labels —
     identical boundaries for est and ref, so mir_eval sees a pure clustering
     comparison. Fiber silence (-1) becomes its own singleton label per frame
     (each -1 frame a distinct label) so "ungrouped" is never rewarded as a
     cluster. GT has no silence label; its letters tile fully.

**Sanity baselines** (so the number is interpretable), scored the same way:
  - ``all_one``   : every frame one cluster (max recall, min precision floor);
  - ``each_own``  : every frame its own cluster (max precision, min recall floor).
The GT-vs-GT pairwise-F is 1.0 by construction (ceiling).

**Phase 2 — MULTIMODAL / complementarity.** Same tracks/scoring, a second arm:
``compute_fibers`` on the repo's chroma (``refine_ref_offsets.chroma``, harmony
axis) instead of HuBERT. Then HuBERT vs chroma vs their UNION ("a repeat pair is
caught if EITHER modality merges it"). The union is not a partition, so its P/R/F
come from frame-PAIR set algebra (``_same_matrix`` / ``_pr_f`` — identical
definition to mir_eval.pairwise, works on ``hub_mask | chr_mask``). Hypothesis:
HuBERT under-merges (timbre/language-only) and chroma catches the melodic/harmonic
repeats it misses → union recall >> either alone. See findings for the verdict
(complementary in DIRECTION, but chroma collapses to one fiber → union precision
unusable). The lyric arm (DALI) is SKIPPED — see findings for why.

Usage:
  venvs/audio/bin/python -m alignment.external.fiber_validation \\
      --n 30 [--layer 9] [--grid-hz 10] [--seed 0] [--max-download 120]

Writes per-track + aggregate JSON to external/out/fiber_validation.json.
NEW file; imports ref_fibers / continuity_refine / refine_ref_offsets read-only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# read-only reuse of the real interpreter + real hubert/chroma loaders
from alignment.continuity_refine import _compute_hubert  # noqa: E402
from alignment.refine_ref_offsets import HOP, SR, chroma  # noqa: E402
from alignment.ref_fibers import compute_fibers  # noqa: E402

FPS = SR / HOP  # ≈ 43.066 — the HuBERT feature grid compute_fibers expects
_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "out"
_AUDIO_CACHE = _HERE / ".salami_audio"
_SALAMI_HOME = _HERE / ".salami_data"
_FEAT_CACHE = _HERE / ".chroma_cache"
_UA = {"User-Agent": "Mozilla/5.0 (fiber-validation research)"}


def _compute_chroma(path: str) -> np.ndarray:
    """(12, T) chroma-CQT on the SAME SR/HOP grid as HuBERT (FPS≈43.07), via the
    repo's `refine_ref_offsets.chroma`. Cached to disk by path. Drop-in feature
    for `compute_fibers` — same (D, T) contract, different modality (harmony)."""
    import hashlib

    key = hashlib.md5(str(path).encode()).hexdigest()[:16]
    cf = _FEAT_CACHE / f"{key}_chroma.npy"
    if cf.is_file():
        return np.load(cf)
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    c = chroma(y)  # (12, T), L2-normed per frame, SR/HOP grid
    _FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cf, c)
    return c


def _compute_harmony_labels(path: str) -> tuple[np.ndarray, float]:
    """(labels, label_hz) from `harmony_fibers.harmony_fibers` — the harmony
    "key fiber" arm. Cached to disk by path. Same output contract as
    `compute_fibers` (per-frame int repeat-class ids at label_hz), so it flows
    through the SAME scoring as the HuBERT/chroma arms. Only invoked with
    --harmony; the existing arms are untouched."""
    import hashlib

    from alignment.harmony_fibers import harmony_fibers

    key = hashlib.md5(str(path).encode()).hexdigest()[:16]
    cf = _FEAT_CACHE / f"{key}_harm.npz"
    if cf.is_file():
        z = np.load(cf)
        return z["labels"], float(z["hz"])
    labels, hz = harmony_fibers(str(path))
    _FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cf, labels=labels, hz=np.float32(hz))
    return labels, float(hz)


# --------------------------------------------------------------------------- #
# archive.org audio resolution                                                #
# --------------------------------------------------------------------------- #
def _parse_ia_url(url: str) -> tuple[str, str] | None:
    """(item_id, track_stem) from a SALAMI-IA download URL, or None.

    e.g. .../download/brdnhand2005-07-27.shnf/brdnhand2005-07-27d1t03_vbr.mp3
         -> ("brdnhand2005-07-27.shnf", "brdnhand2005-07-27d1t03")
    The item id is the FIRST path segment after /download/; the stem is the
    leaf filename minus the _vbr / extension so it matches the item's current
    derivative (the 2012 URLs used _vbr.mp3; items now carry plain .mp3)."""
    marker = "/download/"
    i = url.find(marker)
    if i < 0:
        return None
    rest = url[i + len(marker) :]
    parts = rest.split("/")
    if len(parts) < 2:
        return None
    item = parts[0]
    leaf = parts[-1]
    for ext in (".mp3", ".ogg", ".flac", ".shn"):
        if leaf.lower().endswith(ext):
            leaf = leaf[: -len(ext)]
            break
    if leaf.endswith("_vbr"):
        leaf = leaf[: -len("_vbr")]
    return item, leaf


def _resolve_ia_audio(url: str, timeout: float = 30.0) -> str | None:
    """Current archive.org audio URL for a (rotted) SALAMI-IA URL, via the item
    metadata API. Matches the track stem; prefers .mp3 > .ogg. None if the item
    is gone or no audio matches."""
    parsed = _parse_ia_url(url)
    if parsed is None:
        return None
    item, stem = parsed
    meta_url = f"https://archive.org/metadata/{item}"
    try:
        req = urllib.request.Request(meta_url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            meta = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    files = meta.get("files") or []
    server = meta.get("server")
    d = meta.get("dir")
    if not files or not server or d is None:
        return None
    audio = [
        f["name"]
        for f in files
        if isinstance(f.get("name"), str)
        and f["name"].lower().endswith((".mp3", ".ogg", ".flac"))
    ]

    # exact-stem match first, then any that contains the stem
    def rank(name: str) -> tuple:
        base = name.rsplit(".", 1)[0]
        ext = name.rsplit(".", 1)[-1].lower()
        ext_pref = {"mp3": 0, "ogg": 1, "flac": 2}.get(ext, 3)
        exact = 0 if base == stem else (1 if stem and stem in base else 2)
        return (exact, ext_pref, len(name))

    cands = [f for f in audio if not f.lower().endswith(".flac")] or audio
    if not cands:
        return None
    cands.sort(key=rank)
    best = cands[0]
    # only accept if it plausibly matches the intended track (stem overlap),
    # else the item has many tracks and we'd grab the wrong one.
    if stem and stem not in best.rsplit(".", 1)[0] and best.rsplit(".", 1)[0] != stem:
        # fall back only when the item has a SINGLE audio track (unambiguous)
        if len(cands) != 1:
            return None
    return f"https://{server}{d}/{best}"


def _download(url: str, dest: Path, timeout: float = 120.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers=_UA)
        with (
            urllib.request.urlopen(req, timeout=timeout) as resp,
            open(tmp, "wb") as fh,
        ):
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
        if tmp.stat().st_size < 10_000:  # too small to be real audio
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        tmp.unlink(missing_ok=True)
        return False


# --------------------------------------------------------------------------- #
# labeling -> common dense grid -> mir_eval intervals                         #
# --------------------------------------------------------------------------- #
def _fiber_labels_on_grid(
    labels: np.ndarray, label_hz: float, centers: np.ndarray
) -> list:
    """Fiber class per grid-cell-center time. Silence/-1 frames each become a
    UNIQUE label (so 'ungrouped' isn't rewarded as one big cluster)."""
    out: list = []
    neg = 0
    for c in centers:
        b = int(round(c * label_hz))
        b = min(max(b, 0), labels.size - 1)
        v = int(labels[b])
        if v < 0:
            out.append(f"_sil{neg}")  # singleton
            neg += 1
        else:
            out.append(f"f{v}")
    return out


def _gt_labels_on_grid(
    intervals: np.ndarray, gt_labels: list, centers: np.ndarray
) -> list:
    """Human section letter per grid-cell-center via interval containment."""
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    out: list = []
    for c in centers:
        idx = np.searchsorted(starts, c, side="right") - 1
        if 0 <= idx < len(gt_labels) and starts[idx] <= c < ends[idx]:
            out.append(str(gt_labels[idx]))
        elif idx >= 0 and c >= ends[-1] and idx == len(gt_labels) - 1:
            out.append(str(gt_labels[-1]))
        else:
            out.append(f"_gap{len(out)}")  # outside any labeled span -> singleton
    return out


def _to_intervals(centers: np.ndarray, grid_hz: float) -> np.ndarray:
    """Unit-width intervals for a shared dense grid (identical est/ref boundaries)."""
    w = 1.0 / grid_hz
    return np.stack([centers - w / 2.0, centers + w / 2.0], axis=1)


@dataclass
class TrackResult:
    salami_id: str
    title: str
    artist: str
    duration: float
    n_gt_sections: int
    n_gt_unique: int
    n_fiber_ids: int
    fiber_coverage: float  # frac of frames in SOME fiber (>=0)
    pairwise_f: float
    pairwise_p: float
    pairwise_r: float
    v_measure: float
    s_over: float
    s_under: float
    base_all_one_f: float
    base_each_own_f: float
    audio_url: str = ""
    # --- multimodal arms (pair-based P/R/F vs same GT mask) ---
    n_chroma_ids: int = 0
    chroma_coverage: float = 0.0
    chr_pairwise_p: float = 0.0
    chr_pairwise_r: float = 0.0
    chr_pairwise_f: float = 0.0
    hub_pair_p: float = 0.0  # HuBERT pair-based (for apples-to-apples w/ chroma+union)
    hub_pair_r: float = 0.0
    hub_pair_f: float = 0.0
    uni_pairwise_p: float = 0.0  # HuBERT ∪ chroma
    uni_pairwise_r: float = 0.0
    uni_pairwise_f: float = 0.0
    chr_only_true_pairs: float = 0.0  # GT-true pairs chroma catches that HuBERT missed
    gt_true_pairs: float = 0.0
    # --- harmony-fiber arm (harmony_fibers.py; only populated with --harmony) ---
    n_harm_ids: int = 0
    harm_coverage: float = 0.0
    harm_pairwise_p: float = 0.0
    harm_pairwise_r: float = 0.0
    harm_pairwise_f: float = 0.0
    harm_mireval_p: float = (
        0.0  # mir_eval.pairwise (partition scoring, like HuBERT phase1)
    )
    harm_mireval_r: float = 0.0
    harm_mireval_f: float = 0.0
    huni_pairwise_p: float = 0.0  # HuBERT ∪ harmony (pair-based)
    huni_pairwise_r: float = 0.0
    huni_pairwise_f: float = 0.0
    harm_only_true_pairs: float = (
        0.0  # GT-true pairs harmony catches that HuBERT missed
    )


def _score_track(
    labels: np.ndarray,
    label_hz: float,
    gt_intervals: np.ndarray,
    gt_labels: list,
    duration: float,
    grid_hz: float,
) -> dict:
    import mir_eval

    fiber_end = labels.size / label_hz
    gt_end = float(gt_intervals[-1, 1])
    span = min(duration, gt_end, fiber_end)
    if span < 5.0:
        raise ValueError(f"span too short: {span:.1f}s")
    n = int(span * grid_hz)
    centers = (np.arange(n) + 0.5) / grid_hz
    ref_lab = _gt_labels_on_grid(gt_intervals, gt_labels, centers)
    est_lab = _fiber_labels_on_grid(labels, label_hz, centers)
    ivs = _to_intervals(centers, grid_hz)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, r, f = mir_eval.segment.pairwise(ivs, ref_lab, ivs, est_lab)
        s_over, s_under, v = mir_eval.segment.nce(ivs, ref_lab, ivs, est_lab)
        # baselines on the same ref/grid
        all_one = ["A"] * n
        each_own = [f"c{i}" for i in range(n)]
        _, _, f_all = mir_eval.segment.pairwise(ivs, ref_lab, ivs, all_one)
        _, _, f_own = mir_eval.segment.pairwise(ivs, ref_lab, ivs, each_own)

    cover = float((labels >= 0).mean())
    return dict(
        pairwise_f=float(f),
        pairwise_p=float(p),
        pairwise_r=float(r),
        v_measure=float(v),
        s_over=float(s_over),
        s_under=float(s_under),
        base_all_one_f=float(f_all),
        base_each_own_f=float(f_own),
        fiber_coverage=cover,
    )


# --------------------------------------------------------------------------- #
# multimodal / complementarity — frame-PAIR bookkeeping                       #
# --------------------------------------------------------------------------- #
# The union of two fiber labelings ("a pair is caught if EITHER modality merges
# it") is NOT a partition, so mir_eval.segment.pairwise (which needs a labeling)
# can't score it directly. We compute pairwise P/R/F straight from the frame-pair
# masks instead — identical definition to mir_eval.pairwise, just set-algebra so
# UNION works. A "positive pair" for a labeling = two grid frames sharing a
# NON-singleton label (fiber id >=0; GT letters always count; each_own singletons
# never pair). Counts are over the C(n,2) unordered frame pairs.
def _same_matrix(int_labels: np.ndarray) -> np.ndarray:
    """Boolean (n,n) upper-triangular 'same cluster' mask for integer labels,
    where negative labels (silence/singleton) never match anything (not even
    each other)."""
    n = int_labels.size
    eq = int_labels[:, None] == int_labels[None, :]
    valid = int_labels[:, None] >= 0
    m = eq & valid & (int_labels[None, :] >= 0)
    return np.triu(m, k=1)


def _pr_f(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float]:
    tp = float((pred_mask & gt_mask).sum())
    pred_pos = float(pred_mask.sum())
    gt_pos = float(gt_mask.sum())
    p = tp / pred_pos if pred_pos > 0 else (1.0 if gt_pos == 0 else 0.0)
    r = tp / gt_pos if gt_pos > 0 else 1.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def _fiber_int_on_grid(
    labels: np.ndarray, label_hz: float, centers: np.ndarray
) -> np.ndarray:
    """Fiber id per grid center as an int array (-1 = silence/ungrouped,
    treated as non-matching by _same_matrix)."""
    out = np.empty(centers.size, dtype=int)
    for i, c in enumerate(centers):
        b = int(round(c * label_hz))
        b = min(max(b, 0), labels.size - 1)
        out[i] = int(labels[b])
    return out


def _gt_int_on_grid(
    intervals: np.ndarray, gt_labels: list, centers: np.ndarray
) -> np.ndarray:
    """Human section letter per grid center mapped to a stable int id; frames
    outside any labeled span get a UNIQUE negative id (never pair)."""
    letter_to_id: dict[str, int] = {}
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    out = np.empty(centers.size, dtype=int)
    neg = -1
    for i, c in enumerate(centers):
        idx = np.searchsorted(starts, c, side="right") - 1
        lab = None
        if 0 <= idx < len(gt_labels) and starts[idx] <= c < ends[idx]:
            lab = str(gt_labels[idx])
        elif idx >= 0 and c >= ends[-1] and idx == len(gt_labels) - 1:
            lab = str(gt_labels[-1])
        if lab is None:
            out[i] = neg
            neg -= 1
        else:
            out[i] = letter_to_id.setdefault(lab, len(letter_to_id))
    return out


def _score_multimodal(
    hub_labels: np.ndarray,
    hub_hz: float,
    chr_labels: np.ndarray,
    chr_hz: float,
    gt_intervals: np.ndarray,
    gt_labels: list,
    duration: float,
    grid_hz: float,
) -> dict:
    """HuBERT vs chroma vs (HuBERT ∪ chroma) pairwise P/R/F on a shared grid,
    all measured against the SAME GT same-section pair mask."""
    hub_end = hub_labels.size / hub_hz
    chr_end = chr_labels.size / chr_hz
    gt_end = float(gt_intervals[-1, 1])
    span = min(duration, gt_end, hub_end, chr_end)
    if span < 5.0:
        raise ValueError(f"span too short: {span:.1f}s")
    n = int(span * grid_hz)
    centers = (np.arange(n) + 0.5) / grid_hz

    gt_i = _gt_int_on_grid(gt_intervals, gt_labels, centers)
    hub_i = _fiber_int_on_grid(hub_labels, hub_hz, centers)
    chr_i = _fiber_int_on_grid(chr_labels, chr_hz, centers)

    gt_m = _same_matrix(gt_i)
    hub_m = _same_matrix(hub_i)
    chr_m = _same_matrix(chr_i)
    uni_m = hub_m | chr_m

    hp, hr, hf = _pr_f(hub_m, gt_m)
    cp, cr, cf = _pr_f(chr_m, gt_m)
    up, ur, uf = _pr_f(uni_m, gt_m)

    return dict(
        hub_p=hp,
        hub_r=hr,
        hub_f=hf,
        chr_p=cp,
        chr_r=cr,
        chr_f=cf,
        uni_p=up,
        uni_r=ur,
        uni_f=uf,
        hub_cover=float((hub_i >= 0).mean()),
        chr_cover=float((chr_i >= 0).mean()),
        # pairs chroma catches that HuBERT missed (of all GT-true pairs)
        chr_only_true=float(((chr_m & ~hub_m) & gt_m).sum()),
        gt_true_pairs=float(gt_m.sum()),
    )


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def _load_salami():
    import mirdata

    _SALAMI_HOME.mkdir(parents=True, exist_ok=True)
    d = mirdata.initialize("salami", data_home=str(_SALAMI_HOME))
    ann_dir = _SALAMI_HOME / "salami-data-public-hierarchy-corrections"
    if not ann_dir.exists():
        print("[salami] downloading annotations (~5MB)...", flush=True)
        d.download(["annotations"])
    return d


def _ia_rows() -> list[dict]:
    import csv

    p = (
        _SALAMI_HOME
        / "salami-data-public-hierarchy-corrections"
        / "metadata"
        / "id_index_internetarchive.csv"
    )
    rows = list(csv.DictReader(open(p)))
    return rows


def _gt_for(dataset, salami_id: str):
    """(intervals, labels) for the uppercase (large-scale) annotation of the
    first available annotator, or None if the track has no uppercase sections."""
    try:
        t = dataset.track(salami_id)
    except Exception:
        return None
    for attr in (
        "sections_annotator_1_uppercase",
        "sections_annotator_2_uppercase",
    ):
        try:
            s = getattr(t, attr, None)
        except Exception:
            s = None
        if s is not None and s.intervals is not None and len(s.intervals) >= 2:
            labs = [str(x) for x in s.labels]
            return s.intervals, labs, t
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="target # tracks to score")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--grid-hz", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-download",
        type=int,
        default=150,
        help="cap on tracks attempted (download+hubert budget)",
    )
    ap.add_argument(
        "--harmony",
        action="store_true",
        help="add the harmony_fibers 'key fiber' arm + HuBERT∪harmony union",
    )
    args = ap.parse_args()

    d = _load_salami()
    rows = _ia_rows()
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    _AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    results: list[TrackResult] = []
    attempted = 0
    fetch_fail = 0
    no_gt = 0
    hubert_fail = 0

    for row in rows:
        if len(results) >= args.n or attempted >= args.max_download:
            break
        sid = row["SONG_ID"]
        gt = _gt_for(d, sid)
        if gt is None:
            no_gt += 1
            continue
        gt_intervals, gt_labels, track = gt
        # need >=1 repeated section for the comparison to be meaningful
        uniq = set(gt_labels)
        if len(gt_labels) - len(uniq) < 1:
            no_gt += 1  # every section distinct: no repeat structure to recover
            continue

        attempted += 1
        dest = _AUDIO_CACHE / f"{sid}.mp3"
        if not dest.is_file():
            resolved = _resolve_ia_audio(row["URL"])
            if resolved is None:
                print(f"[{sid}] resolve FAIL ({row.get('TITLE', '')[:30]})", flush=True)
                fetch_fail += 1
                continue
            if not _download(resolved, dest):
                print(f"[{sid}] download FAIL", flush=True)
                fetch_fail += 1
                continue

        try:
            t0 = time.time()
            gt_iv = np.asarray(gt_intervals, dtype=float)
            dur = float(track.duration or gt_intervals[-1][1])
            # --- HuBERT arm (identical to phase 1) ---
            feat = _compute_hubert(str(dest), args.layer)  # (768, T) on SR/HOP grid
            if feat.shape[1] < int(20 * FPS):
                raise ValueError(f"hubert too short: T={feat.shape[1]}")
            labels, label_hz = compute_fibers(feat, FPS, audio_path=str(dest))
            sc = _score_track(labels, label_hz, gt_iv, gt_labels, dur, args.grid_hz)
            # --- chroma arm: SAME compute_fibers, chroma feature, SAME grid ---
            cfeat = _compute_chroma(str(dest))  # (12, T) on SR/HOP grid
            clabels, clabel_hz = compute_fibers(cfeat, FPS, audio_path=str(dest))
            csc = _score_track(clabels, clabel_hz, gt_iv, gt_labels, dur, args.grid_hz)
            # --- complementarity: HuBERT vs chroma vs union (pair-based) ---
            mm = _score_multimodal(
                labels,
                label_hz,
                clabels,
                clabel_hz,
                gt_iv,
                gt_labels,
                dur,
                args.grid_hz,
            )
            # --- harmony arm (harmony_fibers.py) + HuBERT∪harmony union --------
            harm = None
            hmm = None
            if args.harmony:
                hlabels, hlabel_hz = _compute_harmony_labels(str(dest))
                harm = _score_track(
                    hlabels, hlabel_hz, gt_iv, gt_labels, dur, args.grid_hz
                )
                hmm = _score_multimodal(
                    labels,
                    label_hz,
                    hlabels,
                    hlabel_hz,
                    gt_iv,
                    gt_labels,
                    dur,
                    args.grid_hz,
                )
        except Exception as e:  # noqa: BLE001
            print(
                f"[{sid}] hubert/chroma/score FAIL: {type(e).__name__}: {e}", flush=True
            )
            hubert_fail += 1
            continue

        n_fibers = int(len({v for v in labels.tolist() if v >= 0}))
        n_cfibers = int(len({v for v in clabels.tolist() if v >= 0}))
        tr = TrackResult(
            salami_id=sid,
            title=row.get("TITLE", ""),
            artist=row.get("ARTIST", ""),
            duration=float(track.duration or 0.0),
            n_gt_sections=len(gt_labels),
            n_gt_unique=len(uniq),
            n_fiber_ids=n_fibers,
            fiber_coverage=sc["fiber_coverage"],
            pairwise_f=sc["pairwise_f"],
            pairwise_p=sc["pairwise_p"],
            pairwise_r=sc["pairwise_r"],
            v_measure=sc["v_measure"],
            s_over=sc["s_over"],
            s_under=sc["s_under"],
            base_all_one_f=sc["base_all_one_f"],
            base_each_own_f=sc["base_each_own_f"],
            audio_url=dest.name,
            n_chroma_ids=n_cfibers,
            chroma_coverage=csc["fiber_coverage"],
            chr_pairwise_p=csc["pairwise_p"],
            chr_pairwise_r=csc["pairwise_r"],
            chr_pairwise_f=csc["pairwise_f"],
            hub_pair_p=mm["hub_p"],
            hub_pair_r=mm["hub_r"],
            hub_pair_f=mm["hub_f"],
            uni_pairwise_p=mm["uni_p"],
            uni_pairwise_r=mm["uni_r"],
            uni_pairwise_f=mm["uni_f"],
            chr_only_true_pairs=mm["chr_only_true"],
            gt_true_pairs=mm["gt_true_pairs"],
        )
        if args.harmony and harm is not None and hmm is not None:
            n_hfibers = int(len({v for v in hlabels.tolist() if v >= 0}))
            tr.n_harm_ids = n_hfibers
            tr.harm_coverage = harm["fiber_coverage"]
            tr.harm_mireval_p = harm["pairwise_p"]
            tr.harm_mireval_r = harm["pairwise_r"]
            tr.harm_mireval_f = harm["pairwise_f"]
            tr.harm_pairwise_p = hmm["chr_p"]  # pair-based (apples-to-apples w/ union)
            tr.harm_pairwise_r = hmm["chr_r"]
            tr.harm_pairwise_f = hmm["chr_f"]
            tr.huni_pairwise_p = hmm["uni_p"]
            tr.huni_pairwise_r = hmm["uni_r"]
            tr.huni_pairwise_f = hmm["uni_f"]
            tr.harm_only_true_pairs = hmm["chr_only_true"]
        results.append(tr)
        harm_str = ""
        if args.harmony:
            harm_str = (
                f"HARM pF={tr.harm_pairwise_f:.3f}(P{tr.harm_pairwise_p:.2f}/R{tr.harm_pairwise_r:.2f}) "
                f"H∪HARM pF={tr.huni_pairwise_f:.3f}(P{tr.huni_pairwise_p:.2f}/R{tr.huni_pairwise_r:.2f}) "
                f"m{tr.n_harm_ids} "
            )
        print(
            f"[{sid}] {row.get('TITLE', '')[:22]:22s} "
            f"HUB pF={tr.pairwise_f:.3f}(P{tr.pairwise_p:.2f}/R{tr.pairwise_r:.2f}) "
            f"CHR pF={tr.chr_pairwise_f:.3f}(P{tr.chr_pairwise_p:.2f}/R{tr.chr_pairwise_r:.2f}) "
            f"UNI pF={tr.uni_pairwise_f:.3f}(P{tr.uni_pairwise_p:.2f}/R{tr.uni_pairwise_r:.2f}) "
            f"{harm_str}"
            f"h{n_fibers}/c{n_cfibers} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    if not results:
        print("NO TRACKS SCORED — audio fetch failed for all attempts.", flush=True)
        print(
            f"attempted={attempted} fetch_fail={fetch_fail} "
            f"no_gt_skips={no_gt} hubert_fail={hubert_fail}",
            flush=True,
        )
        return 1

    def agg(key: str) -> dict:
        v = np.array([getattr(r, key) for r in results], dtype=float)
        return {
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "std": float(v.std()),
        }

    summary = {
        "dataset": "SALAMI-IA (uppercase annotator-1 sections)",
        "n_scored": len(results),
        "n_attempted": attempted,
        "fetch_fail": fetch_fail,
        "no_gt_or_no_repeat_skips": no_gt,
        "hubert_fail": hubert_fail,
        "layer": args.layer,
        "grid_hz": args.grid_hz,
        "seed": args.seed,
        "pairwise_f": agg("pairwise_f"),
        "pairwise_p": agg("pairwise_p"),
        "pairwise_r": agg("pairwise_r"),
        "v_measure": agg("v_measure"),
        "s_over": agg("s_over"),
        "s_under": agg("s_under"),
        "fiber_coverage": agg("fiber_coverage"),
        "baseline_all_one_f": agg("base_all_one_f"),
        "baseline_each_own_f": agg("base_each_own_f"),
        # --- chroma arm (mir_eval pairwise, same as HuBERT phase-1 numbers) ---
        "chroma_pairwise_f": agg("chr_pairwise_f"),
        "chroma_pairwise_p": agg("chr_pairwise_p"),
        "chroma_pairwise_r": agg("chr_pairwise_r"),
        "chroma_coverage": agg("chroma_coverage"),
        # --- complementarity (pair-based; hub_pair_* is apples-to-apples w/ union) ---
        "hub_pair_p": agg("hub_pair_p"),
        "hub_pair_r": agg("hub_pair_r"),
        "hub_pair_f": agg("hub_pair_f"),
        "union_pairwise_p": agg("uni_pairwise_p"),
        "union_pairwise_r": agg("uni_pairwise_r"),
        "union_pairwise_f": agg("uni_pairwise_f"),
        "tracks": [r.__dict__ for r in results],
    }
    # micro-averaged (pooled over all frame-pairs) union-recall gain — the single
    # number the multimodal hypothesis lives or dies on. Aggregating raw pair
    # counts avoids per-track small-denominator noise.
    tot_gt = sum(r.gt_true_pairs for r in results)
    tot_chr_only = sum(r.chr_only_true_pairs for r in results)
    summary["micro_gt_true_pairs"] = tot_gt
    summary["micro_chroma_only_recovered_frac"] = (
        (tot_chr_only / tot_gt) if tot_gt > 0 else 0.0
    )
    zero_hub = [r for r in results if r.n_fiber_ids == 0]
    summary["hub_zero_fiber_tracks"] = [r.salami_id for r in zero_hub]
    summary["hub_zero_but_chroma_found"] = [
        {
            "id": r.salami_id,
            "n_chroma_ids": r.n_chroma_ids,
            "chr_r": round(r.chr_pairwise_r, 3),
        }
        for r in zero_hub
    ]

    if args.harmony:
        # mir_eval-partition arm (apples-to-apples with the HuBERT/chroma phase-1
        # F/P/R numbers) + pair-based union arm.
        summary["harmony_mireval_f"] = agg("harm_mireval_f")
        summary["harmony_mireval_p"] = agg("harm_mireval_p")
        summary["harmony_mireval_r"] = agg("harm_mireval_r")
        summary["harmony_coverage"] = agg("harm_coverage")
        summary["harmony_pair_p"] = agg("harm_pairwise_p")
        summary["harmony_pair_r"] = agg("harm_pairwise_r")
        summary["harmony_pair_f"] = agg("harm_pairwise_f")
        summary["hub_union_harmony_pair_p"] = agg("huni_pairwise_p")
        summary["hub_union_harmony_pair_r"] = agg("huni_pairwise_r")
        summary["hub_union_harmony_pair_f"] = agg("huni_pairwise_f")
        tot_harm_only = sum(r.harm_only_true_pairs for r in results)
        summary["micro_harmony_only_recovered_frac"] = (
            (tot_harm_only / tot_gt) if tot_gt > 0 else 0.0
        )
        summary["hub_zero_but_harmony_found"] = [
            {
                "id": r.salami_id,
                "n_harm_ids": r.n_harm_ids,
                "harm_mireval_f": round(r.harm_mireval_f, 3),
                "harm_pair_r": round(r.harm_pairwise_r, 3),
            }
            for r in zero_hub
        ]

    _OUT.mkdir(parents=True, exist_ok=True)
    out_path = _OUT / "fiber_validation.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print(f"SALAMI-IA multimodal fiber validation — n={len(results)} scored")
    print("=" * 72)
    print("  arm     pairwise-F        precision       recall     coverage")
    print(
        f"  HuBERT  {summary['pairwise_f']['mean']:.3f} (med {summary['pairwise_f']['median']:.3f})  "
        f"{summary['pairwise_p']['mean']:.3f}         {summary['pairwise_r']['mean']:.3f}      "
        f"{summary['fiber_coverage']['mean']:.2f}"
    )
    print(
        f"  chroma  {summary['chroma_pairwise_f']['mean']:.3f} (med {summary['chroma_pairwise_f']['median']:.3f})  "
        f"{summary['chroma_pairwise_p']['mean']:.3f}         {summary['chroma_pairwise_r']['mean']:.3f}      "
        f"{summary['chroma_coverage']['mean']:.2f}"
    )
    print(
        f"  baselines  all_one_F={summary['baseline_all_one_f']['mean']:.3f}  "
        f"each_own_F={summary['baseline_each_own_f']['mean']:.3f}"
    )
    print("\n  COMPLEMENTARITY (pair-based, apples-to-apples):")
    print(
        f"    HuBERT    P={summary['hub_pair_p']['mean']:.3f}  R={summary['hub_pair_r']['mean']:.3f}  F={summary['hub_pair_f']['mean']:.3f}"
    )
    print(
        f"    chroma    P={summary['chroma_pairwise_p']['mean']:.3f}  R={summary['chroma_pairwise_r']['mean']:.3f}  F={summary['chroma_pairwise_f']['mean']:.3f}"
    )
    print(
        f"    HUB∪CHR   P={summary['union_pairwise_p']['mean']:.3f}  R={summary['union_pairwise_r']['mean']:.3f}  F={summary['union_pairwise_f']['mean']:.3f}"
    )
    print(
        f"    union recall gain over HuBERT: "
        f"{summary['union_pairwise_r']['mean'] - summary['hub_pair_r']['mean']:+.3f} "
        f"(micro chroma-only-recovered = {summary['micro_chroma_only_recovered_frac']:.3f} of all GT-true pairs)"
    )
    print(
        f"    HuBERT-zero-fiber tracks: {len(summary['hub_zero_fiber_tracks'])}; "
        f"of those, chroma found >=1 fiber on "
        f"{sum(1 for x in summary['hub_zero_but_chroma_found'] if x['n_chroma_ids'] > 0)}"
    )
    if args.harmony:
        print("\n  HARMONY FIBER ARM (harmony_fibers.py):")
        print(
            "    mir_eval partition (apples-to-apples w/ HuBERT F 0.10 / chroma-blob F 0.51):"
        )
        print(
            f"      harmony   F={summary['harmony_mireval_f']['mean']:.3f} "
            f"(med {summary['harmony_mireval_f']['median']:.3f})  "
            f"P={summary['harmony_mireval_p']['mean']:.3f}  R={summary['harmony_mireval_r']['mean']:.3f}  "
            f"cov={summary['harmony_coverage']['mean']:.2f}"
        )
        print("    pair-based (apples-to-apples w/ union):")
        print(
            f"      HuBERT    P={summary['hub_pair_p']['mean']:.3f}  R={summary['hub_pair_r']['mean']:.3f}  F={summary['hub_pair_f']['mean']:.3f}"
        )
        print(
            f"      harmony   P={summary['harmony_pair_p']['mean']:.3f}  R={summary['harmony_pair_r']['mean']:.3f}  F={summary['harmony_pair_f']['mean']:.3f}"
        )
        print(
            f"      HUB∪HARM  P={summary['hub_union_harmony_pair_p']['mean']:.3f}  R={summary['hub_union_harmony_pair_r']['mean']:.3f}  F={summary['hub_union_harmony_pair_f']['mean']:.3f}"
        )
        print(
            f"      union recall gain over HuBERT: "
            f"{summary['hub_union_harmony_pair_r']['mean'] - summary['hub_pair_r']['mean']:+.3f} "
            f"(micro harmony-only-recovered = {summary['micro_harmony_only_recovered_frac']:.3f} of GT-true pairs)"
        )
        hz = summary["hub_zero_but_harmony_found"]
        print(
            f"      HuBERT-zero-fiber tracks: {len(hz)}; harmony found >=1 fiber on "
            f"{sum(1 for x in hz if x['n_harm_ids'] > 0)}: "
            + ", ".join(
                f"{x['id']}(m{x['n_harm_ids']},F{x['harm_mireval_f']})" for x in hz
            )
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

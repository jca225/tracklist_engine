#!/usr/bin/env python3
"""Build an interactive spectrogram player for alignment hits/misses.

Outputs a folder with:
  index.html     — pick a clip (↑/↓ + Enter), fullscreen spectrogram + playhead
  images/*.png   — mix / source OD-labeled squares
  audio/*.mp3    — matching windows you can hear

Usage:
    venvs/audio/bin/python -m eda.alignment.spectrogram_review.render \\
        --set-id 1fsnxchk --limit 24
"""

from __future__ import annotations

import argparse
import html
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eda.alignment.spectrogram_review.ableton_label import (
    format_ableton_line,
    resolve_ableton,
)
from eda.alignment.spectrogram_review.audible import (
    mix_truth_span,
    ref_truth_span,
    ref_view_window,
)
from eda.alignment.spectrogram_review.audio_clips import cut_clip
from eda.alignment.spectrogram_review.labels import (
    fmt_clock,
    group_blurb,
    group_title,
    one_liner,
    stem_plain,
)
from eda.alignment.spectrogram_review.player_html import write_player
from eda.alignment.spectrogram_review.source_audio import (
    load_manifest_tracks,
    resolve_source_audio,
)
from eda.alignment.spectrogram_review.spans import (
    DEFAULT_SPAN_TABLE,
    SpanCard,
    iter_cards,
    load_span_rows,
    select_cards,
)
from eda.alignment.spectrogram_review.stats import weak_axes_blurb
from eda.alignment.spectrogram_review.spectrogram import (
    DEFAULT_SIZE,
    FAIL_EDGE,
    GT_COLOR,
    PRED_COLOR,
    SUCCESS_EDGE,
    TimeBox,
    draw_od_boxes,
    mel_rgb,
    to_square,
)
from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir

OUT_ROOT = Path(__file__).resolve().parent / "out"
SR = 22_050


def _mix_path(set_id: str) -> Path:
    d = find_aligning_dir(set_id)
    if d is None:
        raise FileNotFoundError(f"no ~/aligning folder for {set_id}")
    for name in ("mix.m4a", "mix.flac", "mix.wav", "mix_instrumental.flac"):
        cand = d / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no mix audio in {d}")


def _load_window(path: Path, start_s: float, end_s: float) -> np.ndarray:
    import librosa

    dur = max(0.05, end_s - start_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(
            str(path),
            sr=SR,
            mono=True,
            offset=max(0.0, start_s),
            duration=dur,
        )
    return y


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)[:60]


def _od_square(
    audio: Path,
    win_start: float,
    win_end: float,
    actual: tuple[float, float],
    pred: tuple[float, float],
    *,
    size: int,
    edge: tuple[int, int, int],
    truth_label: str = "Truth",
    guess_label: str = "Our guess",
) -> Image.Image:
    y = _load_window(audio, win_start, win_end)
    canvas = to_square(mel_rgb(y, sr=SR), size=size)
    boxes = [
        TimeBox(
            actual[0], actual[1], class_name=truth_label, color=GT_COLOR, band="top"
        ),
        TimeBox(
            pred[0], pred[1], class_name=guess_label, color=PRED_COLOR, band="bottom"
        ),
    ]
    return draw_od_boxes(
        canvas,
        boxes,
        win_start_s=win_start,
        win_end_s=win_end,
        edge_color=edge,
    )


def _gt_display_name(card: SpanCard, ab: dict) -> str:
    """Prefer Ableton/GT clip name — spine names lie when stem was rematched."""
    gt_name = str(ab.get("ableton_name") or "").strip()
    return gt_name or card.name


def _spine_mismatch_note(card: SpanCard, gt_name: str) -> str:
    spine = (card.name or "").strip()
    if not spine or not gt_name:
        return ""
    if spine.lower() == gt_name.lower():
        return ""
    # Same song, different version/stem wording
    if spine[:24].lower() in gt_name.lower() or gt_name[:24].lower() in spine.lower():
        if (card.gt_stem or "") not in spine.lower():
            return f"Tracklist spine said “{spine}”; Truth is the {stem_plain(card.gt_stem)} GT."
        return ""
    return f"Tracklist spine said “{spine}”; Truth follows Ableton GT."


def _placeholder(size: int, msg: str) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(30, 30, 30))
    d = ImageDraw.Draw(img)
    d.text((16, size // 2), msg, fill=(180, 180, 180))
    return img


def _weak_html(weak: dict) -> str:
    if not weak:
        return ""
    bits = []
    for s in weak.get("worst_stems") or []:
        bits.append(
            f"<div><strong>{html.escape(stem_plain(s['key']))}</strong> — "
            f"only {s['success_rate']:.0%} of clips work "
            f"(~{s['sec_lost'] / 60:.0f} min still wrong)</div>"
        )
    return (
        '<div class="weak"><strong>Where we’re weakest</strong>'
        + "".join(bits)
        + "</div>"
    )


def _list_html(cards: list[SpanCard]) -> str:
    groups: dict[str, list[tuple[int, SpanCard]]] = defaultdict(list)
    for i, c in enumerate(cards):
        groups[c.group_key].append((i, c))
    fail_keys = sorted(
        (k for k, rows in groups.items() if any(not c.success for _, c in rows)),
        key=lambda k: sum(c.sec_lost for _, c in groups[k]),
        reverse=True,
    )
    ok_keys = sorted(k for k in groups if k not in fail_keys)
    parts: list[str] = []
    for key in fail_keys + ok_keys:
        rows = groups[key]
        is_ok = all(c.success for _, c in rows)
        cls = "ok" if is_ok else "fail"
        title = group_title(key, success=is_ok)
        blurb = group_blurb(key, success=is_ok)
        prefix = "Hits" if is_ok else "Misses"
        btns = []
        for i, c in sorted(rows, key=lambda t: t[1].sec_lost, reverse=True):
            badge = "hit" if c.success else "miss"
            badge_txt = "Hit" if c.success else "Miss"
            ab = resolve_ableton(
                c.set_id,
                recording_id=c.recording_id,
                gt_set_start_s=c.gt_set_start_s,
                fallback_slot=c.slot,
                fallback_name=c.name,
                gt_stem=c.gt_stem,
                pred_set_start_s=c.pred_set_start_s,
                pred_set_end_s=c.pred_set_end_s,
                pred_stem=c.pred_stem,
            )
            ab_line = format_ableton_line(ab)
            gt_name = _gt_display_name(c, ab)
            note = _spine_mismatch_note(c, gt_name)
            search = (
                f"{gt_name} {c.name} {c.slot} {c.cause} {stem_plain(c.gt_stem)} "
                f"{ab_line}"
            ).lower()
            meta = f"{one_liner(c)} · Truth stem: {stem_plain(c.gt_stem)}"
            if note:
                meta = f"{meta} · {note}"
            btns.append(
                f'<button type="button" class="clip-btn" data-idx="{i}" '
                f'data-outcome="{"success" if c.success else "failure"}" '
                f'data-stem="{html.escape(c.gt_stem)}" '
                f'data-search="{html.escape(search)}">'
                f'<span class="badge {badge}">{badge_txt}</span>'
                f'<div><div class="name">{html.escape(gt_name)}</div>'
                f'<div class="meta">{html.escape(meta)}</div>'
                f'<div class="ableton">{html.escape(ab_line)}</div></div>'
                f'<span class="hint">Enter ↵</span></button>'
            )
        parts.append(
            f'<section class="group {cls}">'
            f"<h2>{prefix}: {html.escape(title)} "
            f'<span style="color:var(--muted);font-weight:500">({len(rows)})</span></h2>'
            f'<p class="blurb">{html.escape(blurb)}</p>'
            f'<div class="list">{"".join(btns)}</div></section>'
        )
    return "\n".join(parts)


def render_player(
    cards: list[SpanCard],
    out_dir: Path,
    *,
    size: int = DEFAULT_SIZE,
) -> list[dict]:
    mix_cache: dict[str, Path] = {}
    track_cache: dict[str, dict[str, dict]] = {}
    img_dir = out_dir / "images"
    audio_dir = out_dir / "audio"
    img_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []

    for i, card in enumerate(cards):
        if card.set_id not in mix_cache:
            mix_cache[card.set_id] = _mix_path(card.set_id)
        if card.set_id not in track_cache:
            track_cache[card.set_id] = load_manifest_tracks(card.set_id)

        edge = SUCCESS_EDGE if card.success else FAIL_EDGE
        mix_path = mix_cache[card.set_id]
        stem = _safe(f"{i:04d}_{card.set}_{card.slot}")

        ab = resolve_ableton(
            card.set_id,
            recording_id=card.recording_id,
            gt_set_start_s=card.gt_set_start_s,
            fallback_slot=card.slot,
            fallback_name=card.name,
            gt_stem=card.gt_stem,
            pred_set_start_s=card.pred_set_start_s,
            pred_set_end_s=card.pred_set_end_s,
            pred_stem=card.pred_stem,
        )
        truth_stem = ab.get("ableton_stem") or card.gt_stem

        # Truth = audible region only (fader / stem silence is not alignable).
        mix_truth = mix_truth_span(card)
        truth_lbl = f"Truth · {stem_plain(truth_stem)}"
        guess_lbl = f"Guess · {stem_plain(card.pred_stem)}"

        mix_img = img_dir / f"{stem}_mix.png"
        mix_audio = audio_dir / f"{stem}_mix.mp3"
        _od_square(
            mix_path,
            card.win_start_s,
            card.win_end_s,
            mix_truth,
            (card.pred_set_start_s, card.pred_set_end_s),
            size=size,
            edge=edge,
            truth_label=truth_lbl,
            guess_label=guess_lbl,
        ).save(mix_img)
        cut_clip(mix_path, card.win_start_s, card.win_end_s, mix_audio)

        src = resolve_source_audio(
            card.set_id,
            ab.get("ableton_recording_id") or card.recording_id,
            gt_stem=ab.get("ableton_stem") or card.gt_stem,
            tracks=track_cache[card.set_id],
            slot=ab.get("ableton_slot") or card.slot,
            name=ab.get("ableton_name") or card.name,
        )
        src_img_rel = None
        src_audio_rel = None
        ref_win_lo, ref_win_hi = card.ref_win_start_s, card.ref_win_end_s
        if src is not None:
            src_truth = ref_truth_span(card, source_path=src)
            ref_win_lo, ref_win_hi = ref_view_window(card, src_truth)
            src_img = img_dir / f"{stem}_src.png"
            src_audio = audio_dir / f"{stem}_src.mp3"
            _od_square(
                src,
                ref_win_lo,
                ref_win_hi,
                src_truth,
                (card.pred_ref_start_s, card.pred_ref_end_s),
                size=size,
                edge=edge,
                truth_label=truth_lbl,
                guess_label=guess_lbl,
            ).save(src_img)
            if cut_clip(src, ref_win_lo, ref_win_hi, src_audio):
                src_img_rel = f"images/{src_img.name}"
                src_audio_rel = f"audio/{src_audio.name}"
        else:
            _placeholder(size, "Original song missing").save(
                img_dir / f"{stem}_src.png"
            )

        gt_name = _gt_display_name(card, ab)
        note = _spine_mismatch_note(card, gt_name)
        blurb = one_liner(card)
        if note:
            blurb = f"{blurb} {note}"
        items.append(
            {
                "idx": i,
                "name": gt_name,
                "spine_name": card.name,
                "slot": card.slot,
                "set": card.set,
                "set_id": card.set_id,
                "success": card.success,
                "outcome": "success" if card.success else "failure",
                "stem": card.gt_stem,
                "pred_stem": card.pred_stem,
                "group": card.group_key,
                "group_title": group_title(card.group_key, success=card.success),
                "blurb": blurb,
                "ableton_line": format_ableton_line(ab),
                "ableton_track_num": ab.get("ableton_track_num"),
                "ableton_slot": ab.get("ableton_slot"),
                "ableton_name": ab.get("ableton_name"),
                "mix_time": fmt_clock(
                    float(ab.get("mix_start_s") or card.gt_set_start_s)
                ),
                "mix_img": f"images/{mix_img.name}",
                "mix_audio": f"audio/{mix_audio.name}",
                "src_img": src_img_rel,
                "src_audio": src_audio_rel,
                "mix_dur_s": max(0.05, card.win_end_s - card.win_start_s),
                "ref_dur_s": max(0.05, ref_win_hi - ref_win_lo),
            }
        )
        print(
            f"[{i + 1}/{len(cards)}] {card.set} slot={card.slot} "
            f"{'OK' if card.success else 'FAIL'} [{card.group_key}] "
            f"src={'yes' if src_audio_rel else 'no'}",
            flush=True,
        )
    return items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--span-table", type=Path, default=DEFAULT_SPAN_TABLE)
    p.add_argument("--set-id", default=None)
    p.add_argument("--outcome", choices=("all", "failure", "success"), default="all")
    p.add_argument("--cause", default=None)
    p.add_argument(
        "--stem", choices=("regular", "acappella", "instrumental"), default=None
    )
    p.add_argument("--limit", type=int, default=24)
    p.add_argument(
        "--worst-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--pad-s", type=float, default=5.0)
    p.add_argument("--max-window-s", type=float, default=90.0)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument(
        "--serve",
        action="store_true",
        help="after render, start local HTTP server + open browser (fixes silent audio)",
    )
    args = p.parse_args(argv)

    if not args.span_table.is_file():
        print(
            f"missing span table: {args.span_table}\n"
            "run: venvs/audio/bin/python -m eda.alignment.failure_analysis.build_span_table",
            file=sys.stderr,
        )
        return 1

    rows = load_span_rows(args.span_table)
    all_cards = list(
        iter_cards(
            rows,
            set_id=args.set_id,
            outcome=args.outcome,
            cause=args.cause,
            stem=args.stem,
            pad_s=args.pad_s,
            max_window_s=args.max_window_s,
        )
    )
    weak = weak_axes_blurb(all_cards)
    cards = select_cards(
        all_cards,
        limit=args.limit,
        outcome=args.outcome,
        worst_first=args.worst_first,
    )
    if not cards:
        print("no spans matched filters", file=sys.stderr)
        return 2

    tag = args.set_id or "all"
    out_dir = args.out_dir or (OUT_ROOT / f"{tag}_{args.outcome}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"corpus={len(all_cards)} rendering={len(cards)} → {out_dir}",
        flush=True,
    )
    items = render_player(cards, out_dir, size=args.size)
    write_player(
        out_dir / "index.html",
        items,
        list_html=_list_html(cards),
        weak_html=_weak_html(weak),
    )
    print(f"wrote {out_dir / 'index.html'}")
    print(
        "Audio needs HTTP (file:// often silent). Serve with:\n"
        f"  venvs/audio/bin/python -m eda.alignment.spectrogram_review.serve "
        f"--dir {out_dir}"
    )
    if args.serve:
        from eda.alignment.spectrogram_review.serve import main as serve_main

        return serve_main(["--dir", str(out_dir)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

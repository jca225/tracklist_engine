from __future__ import annotations

import tempfile
from pathlib import Path

from downloader.analysis import analyze_downloaded_audio
from downloader.constants import SUPPORTED_PLATFORMS
from downloader.providers import download_spotify, download_with_ytdl
from downloader.sources import canonicalize_url, get_music_url_rows, infer_provider
from downloader.storage import (
    build_source_audio_metadata,
    build_track_object_id,
    get_track_paths,
    transcode_to_pipeline_wav,
    write_source_audio_metadata,
)


def download_from_music_db(
    *,
    db_path: Path,
    output_root: Path,
    spotify_downloader_dir: Path,
    cue_detr_dir: Path,
    cue_detr_checkpoint: str = "disco-eth/cue-detr",
    cue_detr_sensitivity: float = 0.9,
    cue_detr_radius: int = 16,
    run_demucs: bool = True,
    demucs_model: str = "htdemucs",
    demucs_device: str = "mps",
    demucs_segment: int = 7,
    demucs_overlap: float = 0.25,
    timeout: float = 10.0,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    rows = get_music_url_rows(db_path)
    seen_urls: set[str] = set()

    stats = {
        "loaded_rows": len(rows),
        "processed": 0,
        "downloaded": 0,
        "failed": 0,
        "skipped_duplicate": 0,
        "skipped_unknown": 0,
        "analyzed": 0,
        "analysis_failed": 0,
        "stems_generated": 0,
        "stems_skipped": 0,
        "stems_failed": 0,
    }

    for row in rows:
        raw_url = row["url"]
        canonical_url = canonicalize_url(raw_url, timeout=timeout)
        provider = infer_provider(canonical_url)

        if provider not in SUPPORTED_PLATFORMS:
            stats["skipped_unknown"] += 1
            continue

        if canonical_url in seen_urls:
            stats["skipped_duplicate"] += 1
            continue
        seen_urls.add(canonical_url)

        track_object_id = build_track_object_id(row, canonical_url)
        track_paths = get_track_paths(output_root, track_object_id)

        if limit is not None and stats["processed"] >= limit:
            break
        stats["processed"] += 1

        if dry_run:
            print(f"[DRY RUN] {provider}: {canonical_url} -> {track_paths['source_audio']}")
            continue

        downloaded_file: Path | None = None
        with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            if provider == "spotify":
                ok, err, downloaded_file = download_spotify(
                    canonical_url,
                    tmpdir,
                    spotify_downloader_dir,
                )
            else:
                ok, err, downloaded_file = download_with_ytdl(
                    canonical_url,
                    tmpdir,
                )

            if ok and downloaded_file is not None:
                try:
                    transcode_to_pipeline_wav(downloaded_file, track_paths["source_audio"])
                    source_meta = build_source_audio_metadata(track_paths["source_audio"])
                    source_meta.update(
                        {
                            "track_object_id": track_object_id,
                            "canonical_url": canonical_url,
                            "provider": provider,
                            "set_id": row.get("set_id"),
                            "track_id": row.get("track_id"),
                        }
                    )
                    write_source_audio_metadata(track_paths["source_meta"], source_meta)
                    downloaded_file = track_paths["source_audio"]
                except Exception as exc:
                    ok = False
                    err = str(exc)
                    downloaded_file = None

        if ok:
            stats["downloaded"] += 1
            print(f"[OK] {provider}: {canonical_url} -> {track_paths['source_audio']}")
            if downloaded_file is None:
                stats["analysis_failed"] += 1
                print("         Could not identify downloaded file for analysis.")
                continue

            analysis_ok, analysis_err, analysis_summary = analyze_downloaded_audio(
                downloaded_file,
                cue_detr_dir=cue_detr_dir,
                cue_detr_checkpoint=cue_detr_checkpoint,
                cue_detr_sensitivity=cue_detr_sensitivity,
                cue_detr_radius=cue_detr_radius,
                run_demucs=run_demucs,
                demucs_model=demucs_model,
                demucs_device=demucs_device,
                demucs_segment=demucs_segment,
                demucs_overlap=demucs_overlap,
            )
            if analysis_summary.get("stems_generated"):
                stats["stems_generated"] += 1
            elif analysis_summary.get("stems_skipped"):
                stats["stems_skipped"] += 1
            elif analysis_summary.get("stems_error"):
                stats["stems_failed"] += 1

            if analysis_ok:
                stats["analyzed"] += 1
                print(f"         Analysis saved: {downloaded_file.parent / 'analysis.json'}")
                if analysis_summary.get("stems_generated"):
                    print("         Demucs stems created (acapella + instrumental).")
                elif analysis_summary.get("stems_skipped"):
                    print("         Demucs stems already exist; skipped.")
                elif analysis_summary.get("stems_error"):
                    print(f"         Demucs stem extraction failed: {analysis_summary['stems_error']}")
            else:
                stats["analysis_failed"] += 1
                if analysis_err:
                    print(f"         Analysis failed: {analysis_err}")
        else:
            stats["failed"] += 1
            print(f"[FAILED] {provider}: {canonical_url}")
            if err:
                print(f"         {err}")

    return stats
